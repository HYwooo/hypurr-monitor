"""
Train LSTM+CatBoost with Walk-Forward
Train: 2024-01 to 2025-08
Valid: 2025-09, Test: 2025-10
"""

import sys
from pathlib import Path
import json
import random
import numpy as np
import pandas as pd
import talib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from datetime import datetime

SEQ_LEN = 48
HIDDEN_DIM = 128
LSTM_OUTPUT_DIM = 64
LSTM_EPOCHS = 5
LSTM_BATCH_SIZE = 2048
LSTM_LR = 0.01
NEUTRAL_SCALE = 1.3
LOOKFORWARD_BARS = 6
FOCAL_GAMMA = 2.0
CLASS_WEIGHTS = [2.5, 1.0, 2.5]
PROBABILITY_THRESHOLD = 0.65
DOWN_LABEL = 0
NEUTRAL_LABEL = 1
UP_LABEL = 2
DEVICE = torch.device("cpu")


class LSTMEncoder(nn.Module):
    def __init__(self, n_features, hidden_dim=128, lstm_output_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, lstm_output_dim))

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])


class LSTMClassifier(nn.Module):
    def __init__(self, n_features, hidden_dim=128, lstm_output_dim=64, n_classes=3):
        super().__init__()
        self.encoder = LSTMEncoder(n_features, hidden_dim, lstm_output_dim)
        self.classifier = nn.Linear(lstm_output_dim, n_classes)

    def forward(self, x):
        return self.encoder(x), self.classifier(self.encoder(x))


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha if alpha is not None else [1.0, 1.0, 1.0])

    def forward(self, logits, targets):
        ce_loss = nn.CrossEntropyLoss(reduction="none", weight=self.alpha.to(logits.device))
        loss = ce_loss(logits, targets)
        probs = torch.softmax(logits, dim=1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - p_t) ** self.gamma
        return (focal_weight * loss).mean()


def generate_features(df):
    close, high, low, volume = df["close"].values, df["high"].values, df["low"].values, df["volume"].values
    f = pd.DataFrame(index=df.index)
    for p in [1, 3, 6, 12, 24]:
        f[f"return_{p}"] = pd.Series(close).pct_change(p)
    f["volatility_6"] = pd.Series(close).pct_change().rolling(6).std()
    f["volatility_12"] = pd.Series(close).pct_change().rolling(12).std()
    f["volatility_24"] = pd.Series(close).pct_change().rolling(24).std()
    f["rsi"] = talib.RSI(close, 14)
    k, d = talib.STOCH(high, low, close, 9, 3, 3)
    f["kdj_k"] = k
    f["kdj_d"] = d
    m, mh, mg = talib.MACD(close, 12, 26, 9)
    f["macd_dif"] = m
    f["macd_dea"] = mh
    f["macd_hist"] = mg
    e12 = talib.EMA(close, 12)
    e26 = talib.EMA(close, 26)
    atr26 = talib.ATR(high, low, close, 26)
    mv = ((e12 - e26) / (atr26 + 1e-10)) * 100
    f["macd_v"] = mv
    f["macd_v_signal"] = talib.EMA(mv, 9)
    f["macd_v_hist"] = mv - f["macd_v_signal"]
    u, _, l = talib.BBANDS(close, 20)
    f["bb_position"] = (close - l) / (u - l + 1e-10)
    f["volume_ratio"] = volume / (pd.Series(volume).rolling(20).mean() + 1e-10)
    f["high_low_ratio"] = (high - low) / close
    f["close_position"] = (close - low) / (high - low + 1e-10)
    atr = talib.ATR(high, low, close, 14)
    f["atr"] = atr
    f["natr"] = (atr / close) * 100
    f["momentum"] = pd.Series(close).pct_change(10)
    for p in [7, 25, 50]:
        f[f"ema_ratio_{p}"] = close / talib.EMA(close, p) - 1
    for p in [7, 25, 50]:
        f[f"sma_ratio_{p}"] = close / talib.SMA(close, p) - 1
    return f


def gen_labels(closes, highs, lows, ns=1.3, lb=6):
    atr = talib.ATR(highs, lows, closes, 14)
    fc = closes[lb:]
    cc = closes[:-lb]
    fa = atr[:-lb]
    fr = (fc - cc) / cc
    labels = np.full(len(fr), NEUTRAL_LABEL)
    labels[fr > (fa / cc) * ns] = UP_LABEL
    labels[fr < -(fa / cc) * ns] = DOWN_LABEL
    return labels


def compute_metrics(y_true, y_pred, y_proba):
    m = {}
    m["accuracy"] = float(np.mean(y_pred == y_true))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    n0, n1, n2 = cm.sum(axis=1)
    p0, p1, p2 = cm[:, 0].sum(), cm[:, 1].sum(), cm[:, 2].sum()
    m["P_aL_pL"] = float(cm[0][0] / p0) if p0 else 0
    m["P_aN_pL"] = float(cm[1][0] / p0) if p0 else 0
    m["P_aS_pL"] = float(cm[2][0] / p0) if p0 else 0
    m["P_aL_pN"] = float(cm[0][1] / p1) if p1 else 0
    m["P_aN_pN"] = float(cm[1][1] / p1) if p1 else 0
    m["P_aS_pN"] = float(cm[2][1] / p1) if p1 else 0
    m["P_aL_pS"] = float(cm[0][2] / p2) if p2 else 0
    m["P_aN_pS"] = float(cm[1][2] / p2) if p2 else 0
    m["P_aS_pS"] = float(cm[2][2] / p2) if p2 else 0
    m["P_pL_aL"] = float(cm[0][0] / n0) if n0 else 0
    m["P_pN_aL"] = float(cm[1][0] / n0) if n0 else 0
    m["P_pS_aL"] = float(cm[2][0] / n0) if n0 else 0
    m["P_pL_aN"] = float(cm[0][1] / n1) if n1 else 0
    m["P_pN_aN"] = float(cm[1][1] / n1) if n1 else 0
    m["P_pS_aN"] = float(cm[2][1] / n1) if n1 else 0
    m["P_pL_aS"] = float(cm[0][2] / n2) if n2 else 0
    m["P_pN_aS"] = float(cm[1][2] / n2) if n2 else 0
    m["P_pS_aS"] = float(cm[2][2] / n2) if n2 else 0
    m["precision_down"] = m["P_aL_pL"]
    m["precision_neutral"] = m["P_aN_pN"]
    m["precision_up"] = m["P_aS_pS"]
    m["recall_down"] = m["P_pL_aL"]
    m["recall_neutral"] = m["P_pN_aN"]
    m["recall_up"] = m["P_pS_aS"]
    for pk, rk, fk in [
        ("precision_down", "recall_down", "f1_down"),
        ("precision_neutral", "recall_neutral", "f1_neutral"),
        ("precision_up", "recall_up", "f1_up"),
    ]:
        p, r = m[pk], m[rk]
        m[fk] = float(2 * p * r / (p + r)) if (p + r) > 0 else 0
    m["support_down"], m["support_neutral"], m["support_up"] = int(n0), int(n1), int(n2)
    m["macro_f1"] = float(np.mean([m["f1_down"], m["f1_neutral"], m["f1_up"]]))
    m["long_recall"] = m["recall_up"]
    m["short_recall"] = m["recall_down"]
    m["long_precision"] = m["precision_up"]
    m["short_precision"] = m["precision_down"]
    lp = m["precision_up"]
    sp = m["precision_down"]
    dq = (lp + sp) / 2
    gm = np.sqrt(lp * sp)
    boost = 1.0 + 0.5 * dq
    m["composite_score"] = (
        m["macro_f1"] * gm * boost - 8.0 * (m["P_aS_pL"] + m["P_aL_pS"]) - 3.0 * (m["P_aL_pN"] + m["P_aS_pN"])
    )
    return m, cm


def extract_features(model, dl):
    model.eval()
    fl, ll = [], []
    with torch.no_grad():
        for X, y in dl:
            f, _ = model(X.to(DEVICE))
            fl.append(f.cpu().numpy())
            ll.append(y.numpy())
    return np.vstack(fl), np.hstack(ll)


def train_and_evaluate(train_months, valid_month, test_month, fold_idx):
    """Train one fold and evaluate on both valid and test"""
    print(f"\n{'=' * 60}")
    print(f"Fold {fold_idx}: Train {train_months[0]}-{train_months[-1]}, Valid {valid_month}, Test {test_month}")
    print(f"{'=' * 60}")

    closes = df["close"].values[: len(features)]
    highs = df["high"].values[: len(features)]
    lows = df["low"].values[: len(features)]
    labels = gen_labels(closes, highs, lows)
    dates = pd.to_datetime(df["open_time"].values[: len(features)], unit="ms")
    months = dates.year * 100 + dates.month

    scaler = StandardScaler()
    fs = scaler.fit_transform(features.values)

    train_mask = np.isin(months, train_months)
    valid_mask = months == valid_month
    test_mask = months == test_month
    train_idx = np.where(train_mask)[0]
    valid_idx = np.where(valid_mask)[0]
    test_idx = np.where(test_mask)[0]

    print(f"Train: {len(train_idx)} samples, Valid: {len(valid_idx)} samples, Test: {len(test_idx)} samples")

    # LSTM
    train_si = train_idx[train_idx >= SEQ_LEN]
    trX = np.array([fs[i - SEQ_LEN : i] for i in train_si])
    trY = np.array([labels[i] for i in train_si])
    valid_si = valid_idx[valid_idx >= SEQ_LEN]
    vaX = np.array([fs[i - SEQ_LEN : i] for i in valid_si])
    vaY = np.array([labels[i] for i in valid_si])

    trDS = TensorDataset(torch.FloatTensor(trX), torch.LongTensor(trY))
    vaDS = TensorDataset(torch.FloatTensor(vaX), torch.LongTensor(vaY))
    trDL = DataLoader(trDS, batch_size=LSTM_BATCH_SIZE, shuffle=True)
    vaDL = DataLoader(vaDS, batch_size=LSTM_BATCH_SIZE)

    model = LSTMClassifier(trX.shape[2]).to(DEVICE)
    crit = FocalLoss(gamma=FOCAL_GAMMA, alpha=CLASS_WEIGHTS)
    opt = torch.optim.Adam(model.parameters(), lr=LSTM_LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)

    best_loss = float("inf")
    best_state = None
    patience = 0
    for epoch in range(LSTM_EPOCHS):
        model.train()
        tl = 0
        for X, y in trDL:
            X, y = X.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            _, logits = model(X)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            tl += loss.item()
        tl /= len(trDL)
        model.eval()
        vl = 0
        with torch.no_grad():
            for X, y in vaDL:
                X, y = X.to(DEVICE), y.to(DEVICE)
                _, logits = model(X)
                vl += crit(logits, y).item()
        vl /= len(vaDL)
        sched.step(vl)
        if vl < best_loss:
            best_loss = vl
            best_state = model.state_dict().copy()
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break
        print(f"  Epoch {epoch + 1}: train={tl:.4f}, valid={vl:.4f}")

    if best_state:
        model.load_state_dict(best_state)

    trF, trL = extract_features(model, trDL)
    vaF, vaL = extract_features(model, vaDL)

    from catboost import CatBoostClassifier

    cb = CatBoostClassifier(
        iterations=300,
        learning_rate=0.08,
        depth=6,
        l2_leaf_reg=3.0,
        loss_function="MultiClass",
        class_weights=CLASS_WEIGHTS,
        early_stopping_rounds=30,
        verbose=False,
    )
    cb.fit(trF, trL, eval_set=(vaF, vaL), verbose=False)

    # Save models
    md = Path(f"models/lstm_catboost/fold_{fold_idx}")
    md.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "n_features": trX.shape[2],
            "hidden_dim": HIDDEN_DIM,
            "lstm_output_dim": LSTM_OUTPUT_DIM,
            "seq_len": SEQ_LEN,
            "config": {"neutral_scale": NEUTRAL_SCALE, "lookforward_bars": LOOKFORWARD_BARS},
        },
        md / "lstm_model.pt",
    )
    cb.save_model(str(md / "catboost_model.cbm"))
    import joblib

    joblib.dump(scaler, md / "scaler.joblib")

    # Evaluate on both valid and test
    results = {}
    for phase, idx in [("Valid", valid_idx), ("Test", test_idx)]:
        te_idx = idx[idx >= SEQ_LEN]
        teX = np.array([fs[i - SEQ_LEN : i] for i in te_idx])
        teY = np.array([labels[i] for i in te_idx])
        teDS = TensorDataset(torch.FloatTensor(teX), torch.LongTensor(teY))
        teDL = DataLoader(teDS, batch_size=2048)
        teF, _ = extract_features(model, teDL)

        yp = cb.predict_proba(teF)
        ypr = cb.predict(teF).flatten()
        mp = np.maximum(yp[:, 0], np.maximum(yp[:, 1], yp[:, 2]))
        nm = mp < PROBABILITY_THRESHOLD
        ypr[nm] = NEUTRAL_LABEL

        m, cm = compute_metrics(teY, ypr, yp)
        pd_dist = pd.Series(ypr).value_counts().sort_index()

        results[phase] = {
            "macro_f1": m["macro_f1"],
            "composite_score": m["composite_score"],
            "long_recall": m["long_recall"],
            "short_recall": m["short_recall"],
            "long_precision": m["long_precision"],
            "short_precision": m["short_precision"],
            "precision_down": m["precision_down"],
            "precision_neutral": m["precision_neutral"],
            "precision_up": m["precision_up"],
            "recall_down": m["recall_down"],
            "recall_neutral": m["recall_neutral"],
            "recall_up": m["recall_up"],
            "f1_down": m["f1_down"],
            "f1_neutral": m["f1_neutral"],
            "f1_up": m["f1_up"],
            "P_aL_pL": m["P_aL_pL"],
            "P_aN_pL": m["P_aN_pL"],
            "P_aS_pL": m["P_aS_pL"],
            "P_aL_pN": m["P_aL_pN"],
            "P_aN_pN": m["P_aN_pN"],
            "P_aS_pN": m["P_aS_pN"],
            "P_aL_pS": m["P_aL_pS"],
            "P_aN_pS": m["P_aN_pS"],
            "P_aS_pS": m["P_aS_pS"],
            "P_pL_aL": m["P_pL_aL"],
            "P_pN_aL": m["P_pN_aL"],
            "P_pS_aL": m["P_pS_aL"],
            "P_pL_aN": m["P_pL_aN"],
            "P_pN_aN": m["P_pN_aN"],
            "P_pS_aN": m["P_pS_aN"],
            "P_pL_aS": m["P_pL_aS"],
            "P_pN_aS": m["P_pN_aS"],
            "P_pS_aS": m["P_pS_aS"],
            "support_down": m["support_down"],
            "support_neutral": m["support_neutral"],
            "support_up": m["support_up"],
            "pred_distribution": {
                "DOWN": int(pd_dist.get(0, 0)),
                "NEUTRAL": int(pd_dist.get(1, 0)),
                "UP": int(pd_dist.get(2, 0)),
            },
            "confusion_matrix": cm.tolist(),
            "test_size": len(teY),
        }
        print(
            f"  {phase}: Macro F1={m['macro_f1']:.4f}, Composite={m['composite_score']:.4f}, Neutral%={pd_dist.get(1, 0) / len(ypr) * 100:.1f}%"
        )

    return results


# Main
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

print("Loading data...")
df = pd.read_parquet("data/futures/um/klines/BTCUSDT_15m_mark.parquet")
df = df.sort_values("open_time").reset_index(drop=True)
print(f"Data: {len(df)} rows")

print("Generating features...")
features = generate_features(df).dropna()
min_len = min(len(df), len(features))
df = df.iloc[:min_len].reset_index(drop=True)
features = features.iloc[:min_len].reset_index(drop=True)

# Walk-Forward folds
# Fold 0: Train 2024-01 to 2025-08, Valid 2025-09, Test 2025-10
train_months = []
current = datetime(2024, 1, 1)
end = datetime(2025, 8, 1)
while current <= end:
    train_months.append(current.year * 100 + current.month)
    if current.month == 12:
        current = datetime(current.year + 1, 1, 1)
    else:
        current = datetime(current.year, current.month + 1, 1)

all_results = {}
fold_0_results = train_and_evaluate(train_months, 202509, 202510, 0)
all_results["Fold 0"] = fold_0_results

# Fold 1: Train 2024-01 to 2025-09, Valid 2025-10, Test 2025-11
train_months_f1 = train_months + [202509]
fold_1_results = train_and_evaluate(train_months_f1, 202510, 202511, 1)
all_results["Fold 1"] = fold_1_results

# Generate report
ts = datetime.now().strftime("%Y%m%dT%H%M%z")

md_content = f"""# LSTM+CatBoost Walk-Forward 训练报告

**Generated**: {ts}
**Train**: 2024-01 ~ 2025-08 (20 months)
**Valid**: 2025-09 | **Test**: 2025-10, 2025-11
**probability_threshold**: {PROBABILITY_THRESHOLD}

---

## 1. 模型配置

| Parameter | Value |
|-----------|-------|
| neutral_scale (K) | {NEUTRAL_SCALE} |
| lookforward_bars | {LOOKFORWARD_BARS} |
| seq_len | {SEQ_LEN} |
| hidden_dim | {HIDDEN_DIM} |
| lstm_output_dim | {LSTM_OUTPUT_DIM} |
| probability_threshold | {PROBABILITY_THRESHOLD} |

---

## 2. Walk-Forward 划分

| Fold | Train | Valid | Test |
|------|-------|-------|------|
| Fold 0 | 2024-01 ~ 2025-08 | 2025-09 | 2025-10 |
| Fold 1 | 2024-01 ~ 2025-09 | 2025-10 | 2025-11 |

---

## 3. 核心指标汇总

### Fold 0 (Valid: 2025-09)

| Metric | Value |
|--------|-------|
| **Macro F1** | {fold_0_results["Valid"]["macro_f1"]:.4f} |
| **Composite Score** | {fold_0_results["Valid"]["composite_score"]:.4f} |
| **Neutral Ratio** | {fold_0_results["Valid"]["pred_distribution"]["NEUTRAL"] / fold_0_results["Valid"]["test_size"] * 100:.1f}% |
| **Long Recall** | {fold_0_results["Valid"]["long_recall"]:.4f} |
| **Short Recall** | {fold_0_results["Valid"]["short_recall"]:.4f} |
| **Long Precision** | {fold_0_results["Valid"]["long_precision"]:.4f} |
| **Short Precision** | {fold_0_results["Valid"]["short_precision"]:.4f} |

### Fold 0 (Test: 2025-10)

| Metric | Value |
|--------|-------|
| **Macro F1** | {fold_0_results["Test"]["macro_f1"]:.4f} |
| **Composite Score** | {fold_0_results["Test"]["composite_score"]:.4f} |
| **Neutral Ratio** | {fold_0_results["Test"]["pred_distribution"]["NEUTRAL"] / fold_0_results["Test"]["test_size"] * 100:.1f}% |
| **Long Recall** | {fold_0_results["Test"]["long_recall"]:.4f} |
| **Short Recall** | {fold_0_results["Test"]["short_recall"]:.4f} |
| **Long Precision** | {fold_0_results["Test"]["long_precision"]:.4f} |
| **Short Precision** | {fold_0_results["Test"]["short_precision"]:.4f} |

### Fold 1 (Valid: 2025-10)

| Metric | Value |
|--------|-------|
| **Macro F1** | {fold_1_results["Valid"]["macro_f1"]:.4f} |
| **Composite Score** | {fold_1_results["Valid"]["composite_score"]:.4f} |
| **Neutral Ratio** | {fold_1_results["Valid"]["pred_distribution"]["NEUTRAL"] / fold_1_results["Valid"]["test_size"] * 100:.1f}% |
| **Long Recall** | {fold_1_results["Valid"]["long_recall"]:.4f} |
| **Short Recall** | {fold_1_results["Valid"]["short_recall"]:.4f} |
| **Long Precision** | {fold_1_results["Valid"]["long_precision"]:.4f} |
| **Short Precision** | {fold_1_results["Valid"]["short_precision"]:.4f} |

### Fold 1 (Test: 2025-11)

| Metric | Value |
|--------|-------|
| **Macro F1** | {fold_1_results["Test"]["macro_f1"]:.4f} |
| **Composite Score** | {fold_1_results["Test"]["composite_score"]:.4f} |
| **Neutral Ratio** | {fold_1_results["Test"]["pred_distribution"]["NEUTRAL"] / fold_1_results["Test"]["test_size"] * 100:.1f}% |
| **Long Recall** | {fold_1_results["Test"]["long_recall"]:.4f} |
| **Short Recall** | {fold_1_results["Test"]["short_recall"]:.4f} |
| **Long Precision** | {fold_1_results["Test"]["long_precision"]:.4f} |
| **Short Precision** | {fold_1_results["Test"]["short_precision"]:.4f} |

---

## 4. Composite Score 公式 (NEW)

```
Composite = Macro_F1 × √(Long_P × Short_P) × (1 + 0.5 × Dir_Quality)
          - 8.0 × (P_aS_pL + P_aL_pS)
          - 3.0 × (P_aL_pN + P_aS_pN)
```

---

## 5. 18-Probability Evaluation System

### Fold 0 - Valid (2025-09)

**P(actual|predicted)**
| Predicted | Actual Long | Actual Neutral | Actual Short |
|------------|-------------|----------------|--------------|
| Long | {fold_0_results["Valid"]["P_aL_pL"] * 100:.1f}% | {fold_0_results["Valid"]["P_aN_pL"] * 100:.1f}% | {fold_0_results["Valid"]["P_aS_pL"] * 100:.1f}% |
| Neutral | {fold_0_results["Valid"]["P_aL_pN"] * 100:.1f}% | {fold_0_results["Valid"]["P_aN_pN"] * 100:.1f}% | {fold_0_results["Valid"]["P_aS_pN"] * 100:.1f}% |
| Short | {fold_0_results["Valid"]["P_aL_pS"] * 100:.1f}% | {fold_0_results["Valid"]["P_aN_pS"] * 100:.1f}% | {fold_0_results["Valid"]["P_aS_pS"] * 100:.1f}% |

**P(predicted|actual)**
| Actual | Pred Long | Pred Neutral | Pred Short |
|--------|-----------|--------------|------------|
| Long | {fold_0_results["Valid"]["P_pL_aL"] * 100:.1f}% | {fold_0_results["Valid"]["P_pN_aL"] * 100:.1f}% | {fold_0_results["Valid"]["P_pS_aL"] * 100:.1f}% |
| Neutral | {fold_0_results["Valid"]["P_pL_aN"] * 100:.1f}% | {fold_0_results["Valid"]["P_pN_aN"] * 100:.1f}% | {fold_0_results["Valid"]["P_pS_aN"] * 100:.1f}% |
| Short | {fold_0_results["Valid"]["P_pL_aS"] * 100:.1f}% | {fold_0_results["Valid"]["P_pN_aS"] * 100:.1f}% | {fold_0_results["Valid"]["P_pS_aS"] * 100:.1f}% |

### Fold 0 - Test (2025-10)

**P(actual|predicted)**
| Predicted | Actual Long | Actual Neutral | Actual Short |
|------------|-------------|----------------|--------------|
| Long | {fold_0_results["Test"]["P_aL_pL"] * 100:.1f}% | {fold_0_results["Test"]["P_aN_pL"] * 100:.1f}% | {fold_0_results["Test"]["P_aS_pL"] * 100:.1f}% |
| Neutral | {fold_0_results["Test"]["P_aL_pN"] * 100:.1f}% | {fold_0_results["Test"]["P_aN_pN"] * 100:.1f}% | {fold_0_results["Test"]["P_aS_pN"] * 100:.1f}% |
| Short | {fold_0_results["Test"]["P_aL_pS"] * 100:.1f}% | {fold_0_results["Test"]["P_aN_pS"] * 100:.1f}% | {fold_0_results["Test"]["P_aS_pS"] * 100:.1f}% |

**P(predicted|actual)**
| Actual | Pred Long | Pred Neutral | Pred Short |
|--------|-----------|--------------|------------|
| Long | {fold_0_results["Test"]["P_pL_aL"] * 100:.1f}% | {fold_0_results["Test"]["P_pN_aL"] * 100:.1f}% | {fold_0_results["Test"]["P_pS_aL"] * 100:.1f}% |
| Neutral | {fold_0_results["Test"]["P_pL_aN"] * 100:.1f}% | {fold_0_results["Test"]["P_pN_aN"] * 100:.1f}% | {fold_0_results["Test"]["P_pS_aN"] * 100:.1f}% |
| Short | {fold_0_results["Test"]["P_pL_aS"] * 100:.1f}% | {fold_0_results["Test"]["P_pN_aS"] * 100:.1f}% | {fold_0_results["Test"]["P_pS_aS"] * 100:.1f}% |

### Fold 1 - Valid (2025-10)

**P(actual|predicted)**
| Predicted | Actual Long | Actual Neutral | Actual Short |
|------------|-------------|----------------|--------------|
| Long | {fold_1_results["Valid"]["P_aL_pL"] * 100:.1f}% | {fold_1_results["Valid"]["P_aN_pL"] * 100:.1f}% | {fold_1_results["Valid"]["P_aS_pL"] * 100:.1f}% |
| Neutral | {fold_1_results["Valid"]["P_aL_pN"] * 100:.1f}% | {fold_1_results["Valid"]["P_aN_pN"] * 100:.1f}% | {fold_1_results["Valid"]["P_aS_pN"] * 100:.1f}% |
| Short | {fold_1_results["Valid"]["P_aL_pS"] * 100:.1f}% | {fold_1_results["Valid"]["P_aN_pS"] * 100:.1f}% | {fold_1_results["Valid"]["P_aS_pS"] * 100:.1f}% |

**P(predicted|actual)**
| Actual | Pred Long | Pred Neutral | Pred Short |
|--------|-----------|--------------|------------|
| Long | {fold_1_results["Valid"]["P_pL_aL"] * 100:.1f}% | {fold_1_results["Valid"]["P_pN_aL"] * 100:.1f}% | {fold_1_results["Valid"]["P_pS_aL"] * 100:.1f}% |
| Neutral | {fold_1_results["Valid"]["P_pL_aN"] * 100:.1f}% | {fold_1_results["Valid"]["P_pN_aN"] * 100:.1f}% | {fold_1_results["Valid"]["P_pS_aN"] * 100:.1f}% |
| Short | {fold_1_results["Valid"]["P_pL_aS"] * 100:.1f}% | {fold_1_results["Valid"]["P_pN_aS"] * 100:.1f}% | {fold_1_results["Valid"]["P_pS_aS"] * 100:.1f}% |

### Fold 1 - Test (2025-11)

**P(actual|predicted)**
| Predicted | Actual Long | Actual Neutral | Actual Short |
|------------|-------------|----------------|--------------|
| Long | {fold_1_results["Test"]["P_aL_pL"] * 100:.1f}% | {fold_1_results["Test"]["P_aN_pL"] * 100:.1f}% | {fold_1_results["Test"]["P_aS_pL"] * 100:.1f}% |
| Neutral | {fold_1_results["Test"]["P_aL_pN"] * 100:.1f}% | {fold_1_results["Test"]["P_aN_pN"] * 100:.1f}% | {fold_1_results["Test"]["P_aS_pN"] * 100:.1f}% |
| Short | {fold_1_results["Test"]["P_aL_pS"] * 100:.1f}% | {fold_1_results["Test"]["P_aN_pS"] * 100:.1f}% | {fold_1_results["Test"]["P_aS_pS"] * 100:.1f}% |

**P(predicted|actual)**
| Actual | Pred Long | Pred Neutral | Pred Short |
|--------|-----------|--------------|------------|
| Long | {fold_1_results["Test"]["P_pL_aL"] * 100:.1f}% | {fold_1_results["Test"]["P_pN_aL"] * 100:.1f}% | {fold_1_results["Test"]["P_pS_aL"] * 100:.1f}% |
| Neutral | {fold_1_results["Test"]["P_pL_aN"] * 100:.1f}% | {fold_1_results["Test"]["P_pN_aN"] * 100:.1f}% | {fold_1_results["Test"]["P_pS_aN"] * 100:.1f}% |
| Short | {fold_1_results["Test"]["P_pL_aS"] * 100:.1f}% | {fold_1_results["Test"]["P_pN_aS"] * 100:.1f}% | {fold_1_results["Test"]["P_pS_aS"] * 100:.1f}% |

---

## 6. 混淆矩阵

### Fold 0 - Test (2025-10)
```
         Predicted
         DOWN    NEUTRAL  UP
Actual DOWN   {fold_0_results["Test"]["confusion_matrix"][0][0]}      {fold_0_results["Test"]["confusion_matrix"][0][1]}       {fold_0_results["Test"]["confusion_matrix"][0][2]}
       NEUTRAL {fold_0_results["Test"]["confusion_matrix"][1][0]}      {fold_0_results["Test"]["confusion_matrix"][1][1]}      {fold_0_results["Test"]["confusion_matrix"][1][2]}
       UP     {fold_0_results["Test"]["confusion_matrix"][2][0]}       {fold_0_results["Test"]["confusion_matrix"][2][1]}       {fold_0_results["Test"]["confusion_matrix"][2][2]}
```

### Fold 1 - Test (2025-11)
```
         Predicted
         DOWN    NEUTRAL  UP
Actual DOWN   {fold_1_results["Test"]["confusion_matrix"][0][0]}      {fold_1_results["Test"]["confusion_matrix"][0][1]}       {fold_1_results["Test"]["confusion_matrix"][0][2]}
       NEUTRAL {fold_1_results["Test"]["confusion_matrix"][1][0]}      {fold_1_results["Test"]["confusion_matrix"][1][1]}      {fold_1_results["Test"]["confusion_matrix"][1][2]}
       UP     {fold_1_results["Test"]["confusion_matrix"][2][0]}       {fold_1_results["Test"]["confusion_matrix"][2][1]}       {fold_1_results["Test"]["confusion_matrix"][2][2]}
```

---

## 7. 模型存储

```
models/lstm_catboost/
├── fold_0/
│   ├── lstm_model.pt
│   ├── catboost_model.cbm
│   └── scaler.joblib
└── fold_1/
    ├── lstm_model.pt
    ├── catboost_model.cbm
    └── scaler.joblib
```
"""

md_path = f"models/walkforward_lstm_results_{ts}.md"
with open(md_path, "w", encoding="utf-8") as f:
    f.write(md_content)
print(f"\nMarkdown report saved to {md_path}")

# Save JSON
json_path = f"models/walkforward_lstm_results_{ts}.json"
with open(json_path, "w") as f:
    json.dump(
        {
            "config": {
                "neutral_scale": NEUTRAL_SCALE,
                "lookforward_bars": LOOKFORWARD_BARS,
                "seq_len": SEQ_LEN,
                "probability_threshold": PROBABILITY_THRESHOLD,
                "train_months": train_months,
            },
            "results": all_results,
        },
        f,
        indent=2,
        default=float,
    )
print(f"JSON saved to {json_path}")
