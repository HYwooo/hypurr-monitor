"""
Multi-Fold Walk-Forward Training: 3/6/12/24 months training windows
Fixed Valid: 2025-09, Test: 2025-10
Each fold has independent dynamic threshold optimization
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
        return ((1 - p_t) ** self.gamma * loss).mean()


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
    m = {"accuracy": float(np.mean(y_pred == y_true))}
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

    # New TCS Formula (v2 - optimized for Macro F1)
    # Components:
    #   - Geometric mean of direction Precision (balance Long/Short)
    #   - Geometric mean of direction Recall (balance Long/Short)
    #   - Neutral F1 (balance all 3 classes)
    #   - Direction F1 average (overall quality)
    # Penalties:
    #   - Direction flip (predict Long but actual Short, or vice versa)
    #   - False break (predict direction but actual Neutral)
    long_p = m["precision_up"]
    short_p = m["precision_down"]
    long_r = m["recall_up"]
    short_r = m["recall_down"]
    f1_l = m["f1_up"]
    f1_s = m["f1_down"]
    f1_n = m["f1_neutral"]

    geo_precision = np.sqrt(long_p * short_p) if (long_p * short_p) > 0 else 0
    geo_recall = np.sqrt(long_r * short_r) if (long_r * short_r) > 0 else 0
    direction_f1_avg = (f1_l + f1_s) / 2

    flip = m["P_aS_pL"] + m["P_aL_pS"]  # Direction flip penalty
    false_break = m["P_aL_pN"] + m["P_aS_pN"]  # False break penalty

    # New TCS: balanced for Macro F1
    m["tcs"] = (
        0.35 * geo_precision
        + 0.35 * geo_recall
        + 0.15 * f1_n
        + 0.15 * direction_f1_avg
        - 1.0 * flip
        - 0.2 * false_break
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


def compute_metrics_for_thresh(y_proba, y_true, thresh):
    """Compute metrics for a given threshold"""
    max_prob = np.maximum(y_proba[:, 0], np.maximum(y_proba[:, 1], y_proba[:, 2]))
    y_pred_raw = (np.argmax(y_proba, axis=1)).astype(int)
    y_pred = y_pred_raw.copy()
    neutral_mask = max_prob < thresh
    y_pred[neutral_mask] = NEUTRAL_LABEL
    neutral_ratio = np.mean(y_pred == NEUTRAL_LABEL)
    return neutral_ratio, y_pred


def find_optimal_threshold(y_proba, target_neutral_ratio):
    """Find threshold using two-stage search with boundary extension:
    1. Coarse search: 0.55, 0.60, 0.65, 0.70, 0.75
    2. If optimal hits boundary, extend search outward until Neutral% closest to target
       (search range: 0.00 to 1.00, step 0.05)
    """
    # Stage 1: Coarse search
    coarse_thresholds = [0.55, 0.60, 0.65, 0.70, 0.75]
    coarse_results = []

    for thresh in coarse_thresholds:
        neutral_ratio, _ = compute_metrics_for_thresh(y_proba, None, thresh)
        coarse_results.append(
            {"threshold": thresh, "neutral_ratio": neutral_ratio, "distance": abs(neutral_ratio - target_neutral_ratio)}
        )

    # Find best in coarse search
    best = min(coarse_results, key=lambda x: x["distance"])
    best_thresh = best["threshold"]
    best_dist = best["distance"]
    best_neutral = best["neutral_ratio"]

    # Stage 2: If best hits boundary, extend search
    if best_thresh == 0.55:
        # Extend downward: 0.50, 0.45, 0.40, ...
        extended_thresholds = [t / 100 for t in range(50, -5, -5)]  # 0.50, 0.45, 0.40, ..., 0.00
    elif best_thresh == 0.75:
        # Extend upward: 0.80, 0.85, 0.90, ...
        extended_thresholds = [t / 100 for t in range(80, 105, 5)]  # 0.80, 0.85, 0.90, ..., 1.00
    else:
        # No boundary hit, return best
        return best_thresh

    # Search extended range and find the one with Neutral% closest to target
    for thresh in extended_thresholds:
        neutral_ratio, _ = compute_metrics_for_thresh(y_proba, None, thresh)
        dist = abs(neutral_ratio - target_neutral_ratio)
        if dist < best_dist:
            best_dist = dist
            best_thresh = thresh
            best_neutral = neutral_ratio
        # Continue searching even if distance increases - we want closest to target

    return best_thresh


def generate_train_months(n_months):
    """Generate n months immediately before 202509"""
    months = []
    # 202509 = September 2025
    year, month = 2025, 9
    for _ in range(n_months):
        months.append(year * 100 + month)
        if month == 1:
            year, month = year - 1, 12
        else:
            month -= 1
    return sorted(months)


# Fixed Valid and Test
VALID_MONTH = 202509
TEST_MONTH = 202510

# Fold configurations: different training window lengths
FOLD_CONFIGS = [
    {"name": "3-month", "n_months": 3},
    {"name": "6-month", "n_months": 6},
    {"name": "12-month", "n_months": 12},
    {"name": "24-month", "n_months": 24},
]

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

closes = df["close"].values[: len(features)]
highs = df["high"].values[: len(features)]
lows = df["low"].values[: len(features)]
labels = gen_labels(closes, highs, lows)
dates = pd.to_datetime(df["open_time"].values[: len(features)], unit="ms")
months = dates.year * 100 + dates.month

scaler = StandardScaler()
fs = scaler.fit_transform(features.values)

# Pre-compute indices
valid_mask = months == VALID_MONTH
test_mask = months == TEST_MONTH
valid_idx = np.where(valid_mask)[0]
test_idx = np.where(test_mask)[0]

# Store all fold results
all_fold_results = []

for fold_cfg in FOLD_CONFIGS:
    fold_name = fold_cfg["name"]
    n_months = fold_cfg["n_months"]
    train_months = generate_train_months(n_months)

    print(f"\n{'=' * 60}")
    print(f"FOLD: {fold_name} (Train: {train_months[0]} ~ {train_months[-1]})")
    print(f"{'=' * 60}")

    train_mask = np.isin(months, train_months)
    train_idx = np.where(train_mask)[0]

    # Calculate training Neutral ratio (for target)
    train_labels = labels[train_idx]
    train_neutral_ratio = np.mean(train_labels == NEUTRAL_LABEL)
    print(f"Training Neutral ratio: {train_neutral_ratio * 100:.1f}%")
    print(f"Target range: {(train_neutral_ratio - 0.025) * 100:.1f}% - {(train_neutral_ratio + 0.025) * 100:.1f}%")

    # LSTM training
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

    # Prepare test data
    te_idx = test_idx[test_idx >= SEQ_LEN]
    teX = np.array([fs[i - SEQ_LEN : i] for i in te_idx])
    teY = np.array([labels[i] for i in te_idx])
    teDS = TensorDataset(torch.FloatTensor(teX), torch.LongTensor(teY))
    teDL = DataLoader(teDS, batch_size=2048)
    teF, _ = extract_features(model, teDL)

    yp = cb.predict_proba(teF)
    ypr_raw = (np.argmax(yp, axis=1)).astype(int)

    # Dynamic threshold
    optimal_thresh = find_optimal_threshold(yp, train_neutral_ratio)
    print(f"\nOptimal threshold: {optimal_thresh:.2f}")

    # Calculate validation metrics for report (using same threshold)
    va_proba = cb.predict_proba(vaF)
    va_pred_raw = (np.argmax(va_proba, axis=1)).astype(int)
    va_max_prob = np.maximum(va_proba[:, 0], np.maximum(va_proba[:, 1], va_proba[:, 2]))
    va_pred = va_pred_raw.copy()
    va_neutral_mask = va_max_prob < optimal_thresh
    va_pred[va_neutral_mask] = NEUTRAL_LABEL
    m_valid, cm_valid = compute_metrics(vaL, va_pred, va_proba)
    va_neutral_ratio = np.mean(va_pred == NEUTRAL_LABEL)

    # Apply optimal threshold to test
    max_prob = np.maximum(yp[:, 0], np.maximum(yp[:, 1], yp[:, 2]))
    ypr = ypr_raw.copy()
    neutral_mask = max_prob < optimal_thresh
    ypr[neutral_mask] = NEUTRAL_LABEL

    # Calculate test metrics
    m, cm = compute_metrics(teY, ypr, yp)
    pd_dist = pd.Series(ypr).value_counts().sort_index()

    print(f"\n{'=' * 60}")
    print(f"RESULTS ({fold_name} - Dynamic threshold={optimal_thresh:.2f})")
    print(f"{'=' * 60}")
    print(f"Macro F1:         {m['macro_f1']:.4f}")
    print(f"TCS:              {m['tcs']:.4f}")
    print(f"Neutral Ratio:    {pd_dist.get(1, 0) / len(ypr) * 100:.1f}% (target: {train_neutral_ratio * 100:.1f}%)")
    print(f"Long Recall:      {m['long_recall']:.4f}")
    print(f"Short Recall:     {m['short_recall']:.4f}")
    print(f"Long Precision:   {m['long_precision']:.4f}")
    print(f"Short Precision:  {m['short_precision']:.4f}")
    print(f"P(aS|pL):        {m['P_aS_pL']:.4f}")
    print(f"P(aL|pS):        {m['P_aL_pS']:.4f}")

    # Threshold comparison (extended search: 0.00 to 1.00, step 0.05)
    threshold_comp = []
    for thresh in [t / 100 for t in range(0, 105, 5)]:
        yp_temp = ypr_raw.copy()
        nm_temp = max_prob < thresh
        yp_temp[nm_temp] = NEUTRAL_LABEL
        m_temp, _ = compute_metrics(teY, yp_temp, yp)
        threshold_comp.append(
            {
                "threshold": float(thresh),
                "neutral_ratio": float(np.mean(yp_temp == NEUTRAL_LABEL)),
                "macro_f1": float(m_temp["macro_f1"]),
                "tcs": float(m_temp["tcs"]),
            }
        )

    print(f"\n{'=' * 60}")
    print("THRESHOLD COMPARISON")
    print(f"{'=' * 60}")
    print(f"{'Threshold':<12} {'Neutral%':<12} {'Macro F1':<12} {'TCS':<12}")
    print("-" * 50)
    for tc in threshold_comp:
        print(f"{tc['threshold']:<12.2f} {tc['neutral_ratio'] * 100:<12.1f} {tc['macro_f1']:<12.4f} {tc['tcs']:<12.4f}")

    # Store fold results
    fold_result = {
        "fold_name": fold_name,
        "n_months": n_months,
        "train_months": train_months,
        "config": {
            "neutral_scale": NEUTRAL_SCALE,
            "lookforward_bars": LOOKFORWARD_BARS,
            "seq_len": SEQ_LEN,
            "train_neutral_ratio": float(train_neutral_ratio),
            "optimal_threshold": float(optimal_thresh),
        },
        "valid_results": {
            "macro_f1": float(m_valid["macro_f1"]),
            "tcs": float(m_valid["tcs"]),
            "neutral_ratio": float(va_neutral_ratio),
            "long_recall": float(m_valid["long_recall"]),
            "short_recall": float(m_valid["short_recall"]),
            "long_precision": float(m_valid["long_precision"]),
            "short_precision": float(m_valid["short_precision"]),
            "confusion_matrix": cm_valid.tolist(),
            # 18 probabilities
            "P_aL_pL": float(m_valid["P_aL_pL"]),
            "P_aN_pL": float(m_valid["P_aN_pL"]),
            "P_aS_pL": float(m_valid["P_aS_pL"]),
            "P_aL_pN": float(m_valid["P_aL_pN"]),
            "P_aN_pN": float(m_valid["P_aN_pN"]),
            "P_aS_pN": float(m_valid["P_aS_pN"]),
            "P_aL_pS": float(m_valid["P_aL_pS"]),
            "P_aN_pS": float(m_valid["P_aN_pS"]),
            "P_aS_pS": float(m_valid["P_aS_pS"]),
            "P_pL_aL": float(m_valid["P_pL_aL"]),
            "P_pN_aL": float(m_valid["P_pN_aL"]),
            "P_pS_aL": float(m_valid["P_pS_aL"]),
            "P_pL_aN": float(m_valid["P_pL_aN"]),
            "P_pN_aN": float(m_valid["P_pN_aN"]),
            "P_pS_aN": float(m_valid["P_pS_aN"]),
            "P_pL_aS": float(m_valid["P_pL_aS"]),
            "P_pN_aS": float(m_valid["P_pN_aS"]),
            "P_pS_aS": float(m_valid["P_pS_aS"]),
        },
        "test_results": {
            "macro_f1": float(m["macro_f1"]),
            "tcs": float(m["tcs"]),
            "neutral_ratio": float(pd_dist.get(1, 0) / len(ypr)),
            "long_recall": float(m["long_recall"]),
            "short_recall": float(m["short_recall"]),
            "long_precision": float(m["long_precision"]),
            "short_precision": float(m["short_precision"]),
            "confusion_matrix": cm.tolist(),
            # 18 probabilities
            "P_aL_pL": float(m["P_aL_pL"]),
            "P_aN_pL": float(m["P_aN_pL"]),
            "P_aS_pL": float(m["P_aS_pL"]),
            "P_aL_pN": float(m["P_aL_pN"]),
            "P_aN_pN": float(m["P_aN_pN"]),
            "P_aS_pN": float(m["P_aS_pN"]),
            "P_aL_pS": float(m["P_aL_pS"]),
            "P_aN_pS": float(m["P_aN_pS"]),
            "P_aS_pS": float(m["P_aS_pS"]),
            "P_pL_aL": float(m["P_pL_aL"]),
            "P_pN_aL": float(m["P_pN_aL"]),
            "P_pS_aL": float(m["P_pS_aL"]),
            "P_pL_aN": float(m["P_pL_aN"]),
            "P_pN_aN": float(m["P_pN_aN"]),
            "P_pS_aN": float(m["P_pS_aN"]),
            "P_pL_aS": float(m["P_pL_aS"]),
            "P_pN_aS": float(m["P_pN_aS"]),
            "P_pS_aS": float(m["P_pS_aS"]),
            "pred_distribution": {
                "DOWN": int(pd_dist.get(0, 0)),
                "NEUTRAL": int(pd_dist.get(1, 0)),
                "UP": int(pd_dist.get(2, 0)),
            },
        },
        "threshold_comparison": threshold_comp,
    }
    all_fold_results.append(fold_result)


# Generate comprehensive multi-fold report
ts = datetime.now().strftime("%Y%m%dT%H%M%z")
json_path = f"models/walkforward_lstm_results_{ts}.json"
with open(json_path, "w") as f:
    json.dump({"folds": all_fold_results, "timestamp": ts}, f, indent=2)
print(f"\nJSON saved to {json_path}")


# Generate multi-fold markdown report
def generate_multifold_markdown_report(ts, all_fold_results):
    """Generate comprehensive multi-fold Chinese markdown report"""

    def p(v):
        return f"{v * 100:.1f}%"

    def f4(v):
        return f"{v:.4f}"

    def fmt18(val):
        return f"{val * 100:.1f}%"

    lines = []
    lines.append("# LSTM+CatBoost Walk-Forward 多 Fold 训练报告")
    lines.append("")
    lines.append(f"**Generated**: {ts}")
    lines.append(f"**Valid**: 202509 | **Test**: 202510")
    lines.append(f"**Folds**: {', '.join([f['fold_name'] for f in all_fold_results])}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. TCS 公式 (v2 - Optimized for Macro F1)")
    lines.append("")
    lines.append("```")
    lines.append(
        "TCS = 0.35×√(Long_P×Short_P) + 0.35×√(Long_R×Short_R) + 0.15×F1_N + 0.15×(F1_L+F1_S)/2 - 1.0×Flip - 0.2×FalseBreak"
    )
    lines.append("```")
    lines.append("")
    lines.append("| 组成部分 | 说明 |")
    lines.append("|---------|------|")
    lines.append("| 0.35×√(Long_P×Short_P) | 几何平均 - Long/Short Precision 平衡 |")
    lines.append("| 0.35×√(Long_R×Short_R) | 几何平均 - Long/Short Recall 平衡 |")
    lines.append("| 0.15×F1_N | Neutral F1 贡献 |")
    lines.append("| 0.15×(F1_L+F1_S)/2 | 方向 F1 平均 |")
    lines.append("| -1.0×Flip | 方向翻转惩罚 (P_aS_pL + P_aL_pS) |")
    lines.append("| -0.2×FalseBreak | False Break 惩罚 (P_aL_pN + P_aS_pN) |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. 模型配置 (All Folds)")
    lines.append("")
    lines.append("| 参数 | 值 |")
    lines.append("|------|-----|")
    lines.append(f"| neutral_scale (K) | {NEUTRAL_SCALE} |")
    lines.append(f"| lookforward_bars | {LOOKFORWARD_BARS} |")
    lines.append(f"| seq_len | {SEQ_LEN} |")
    lines.append(f"| hidden_dim | {HIDDEN_DIM} |")
    lines.append(f"| lstm_output_dim | {LSTM_OUTPUT_DIM} |")
    lines.append(f"| focal_gamma | {FOCAL_GAMMA} |")
    lines.append(f"| class_weights | {CLASS_WEIGHTS} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. Fold Summary")
    lines.append("")
    lines.append("| Fold | Train Period | Months | Threshold | Train Neutral% |")
    lines.append("|------|-------------|--------|-----------|---------------|")
    for fold in all_fold_results:
        tm = fold["train_months"]
        lines.append(
            f"| {fold['fold_name']} | {tm[0]}~{tm[-1]} | {fold['n_months']} | {fold['config']['optimal_threshold']:.2f} | {p(fold['config']['train_neutral_ratio'])} |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 4. Test Metrics by Fold")
    lines.append("")
    lines.append("### Macro F1 & TCS")
    lines.append("")
    lines.append("| Fold | Valid Macro F1 | Valid TCS | Test Macro F1 | Test TCS | Test Neutral% |")
    lines.append("|------|----------------|-----------|---------------|----------|---------------|")
    for fold in all_fold_results:
        vr = fold["valid_results"]
        tr = fold["test_results"]
        lines.append(
            f"| {fold['fold_name']} | {f4(vr['macro_f1'])} | {f4(vr['tcs'])} | {f4(tr['macro_f1'])} | {f4(tr['tcs'])} | {p(tr['neutral_ratio'])} |"
        )
    lines.append("")
    lines.append("### Direction Metrics (Test)")
    lines.append("")
    lines.append("| Fold | Long Recall | Short Recall | Long Precision | Short Precision | P(aS\\|pL) | P(aL\\|pS) |")
    lines.append("|------|------------|-------------|----------------|----------------|----------|-----------|")
    for fold in all_fold_results:
        tr = fold["test_results"]
        lines.append(
            f"| {fold['fold_name']} | {f4(tr['long_recall'])} | {f4(tr['short_recall'])} | {f4(tr['long_precision'])} | {f4(tr['short_precision'])} | {f4(tr['P_aS_pL'])} | {f4(tr['P_aL_pS'])} |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 5. 18-Probability Evaluation System")
    lines.append("")
    for fold in all_fold_results:
        fold_name = fold["fold_name"]
        lines.append(f"### {fold_name}")
        lines.append("")
        lines.append("**P(actual|predicted)**")
        lines.append("| Predicted | Actual Long | Actual Neutral | Actual Short |")
        lines.append("|------------|-------------|----------------|--------------|")
        r = fold["test_results"]
        lines.append(f"| Long | {fmt18(r['P_aL_pL'])} | {fmt18(r['P_aN_pL'])} | {fmt18(r['P_aS_pL'])} |")
        lines.append(f"| Neutral | {fmt18(r['P_aL_pN'])} | {fmt18(r['P_aN_pN'])} | {fmt18(r['P_aS_pN'])} |")
        lines.append(f"| Short | {fmt18(r['P_aL_pS'])} | {fmt18(r['P_aN_pS'])} | {fmt18(r['P_aS_pS'])} |")
        lines.append("")
        lines.append("**P(predicted|actual)**")
        lines.append("| Actual | Pred Long | Pred Neutral | Pred Short |")
        lines.append("|--------|-----------|--------------|------------|")
        lines.append(f"| Long | {fmt18(r['P_pL_aL'])} | {fmt18(r['P_pN_aL'])} | {fmt18(r['P_pS_aL'])} |")
        lines.append(f"| Neutral | {fmt18(r['P_pL_aN'])} | {fmt18(r['P_pN_aN'])} | {fmt18(r['P_pS_aN'])} |")
        lines.append(f"| Short | {fmt18(r['P_pL_aS'])} | {fmt18(r['P_pN_aS'])} | {fmt18(r['P_pS_aS'])} |")
        lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 6. Threshold Comparison by Fold")
    lines.append("")
    lines.append("注: 搜索范围 0.00~1.00，step=0.05；最优阈值已标星(*)")

    for fold in all_fold_results:
        fold_name = fold["fold_name"]
        tc_list = fold["threshold_comparison"]
        opt_thresh = fold["config"]["optimal_threshold"]
        lines.append(f"### {fold_name}")
        lines.append("")
        lines.append("| Threshold | Neutral% | Macro F1 | TCS |")
        lines.append("|-----------|---------|----------|-----|")
        for tc in tc_list:
            marker = " **← optimal**" if abs(tc["threshold"] - opt_thresh) < 0.001 else ""
            lines.append(
                f"| {tc['threshold']:.2f}{marker} | {p(tc['neutral_ratio'])} | {f4(tc['macro_f1'])} | {f4(tc['tcs'])} |"
            )
        lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 7. Best Folds Analysis")
    lines.append("")
    lines.append("### Highest Test Macro F1")
    best_f1_fold = max(all_fold_results, key=lambda x: x["test_results"]["macro_f1"])
    lines.append(
        f"- **{best_f1_fold['fold_name']}**: Macro F1 = {f4(best_f1_fold['test_results']['macro_f1'])}, TCS = {f4(best_f1_fold['test_results']['tcs'])}"
    )
    lines.append("")
    lines.append("### Highest Test TCS")
    best_tcs_fold = max(all_fold_results, key=lambda x: x["test_results"]["tcs"])
    lines.append(
        f"- **{best_tcs_fold['fold_name']}**: TCS = {f4(best_tcs_fold['test_results']['tcs'])}, Macro F1 = {f4(best_tcs_fold['test_results']['macro_f1'])}"
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 8. JSON Results")
    lines.append("")
    lines.append(f"- Timestamp: {ts}")
    lines.append(f"- Path: models/walkforward_lstm_results_{ts}.json")
    return "\n".join(lines)


# Generate and save the multi-fold report
md_report = generate_multifold_markdown_report(ts, all_fold_results)
md_path = f"models/walkforward_lstm_results_{ts}.md"
with open(md_path, "w", encoding="utf-8") as f:
    f.write(md_report)
print(f"Markdown report saved to {md_path}")
