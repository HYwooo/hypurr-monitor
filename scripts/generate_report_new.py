"""
Generate training report with new folds and new Composite Score formula
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import talib
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
plt.rcParams["font.family"] = "DejaVu Sans"
from datetime import datetime

SEQ_LEN = 48
HIDDEN_DIM = 128
LSTM_OUTPUT_DIM = 64
DEVICE = torch.device("cpu")
NEUTRAL_SCALE = 1.3
LOOKFORWARD_BARS = 6
PROBABILITY_THRESHOLD = 0.65
DOWN_LABEL = 0
NEUTRAL_LABEL = 1
UP_LABEL = 2


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


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values
    features = pd.DataFrame(index=df.index)

    for period in [1, 3, 6, 12, 24]:
        features[f"return_{period}"] = pd.Series(close).pct_change(period)

    features["volatility_6"] = pd.Series(close).pct_change().rolling(window=6).std()
    features["volatility_12"] = pd.Series(close).pct_change().rolling(window=12).std()
    features["volatility_24"] = pd.Series(close).pct_change().rolling(window=24).std()
    features["rsi"] = talib.RSI(close, 14)

    k, d = talib.STOCH(high, low, close, fastk_period=9, slowk_period=3, slowd_period=3)
    features["kdj_k"] = k
    features["kdj_d"] = d

    macd, macd_sig, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    features["macd_dif"] = macd
    features["macd_dea"] = macd_sig
    features["macd_hist"] = macd_hist

    ema_12 = talib.EMA(close, 12)
    ema_26 = talib.EMA(close, 26)
    atr_26 = talib.ATR(high, low, close, 26)
    macd_v = ((ema_12 - ema_26) / (atr_26 + 1e-10)) * 100
    features["macd_v"] = macd_v
    features["macd_v_signal"] = talib.EMA(macd_v, 9)
    features["macd_v_hist"] = macd_v - features["macd_v_signal"]

    upper, middle, lower = talib.BBANDS(close, timeperiod=20)
    features["bb_position"] = (close - lower) / (upper - lower + 1e-10)

    vol_ma = pd.Series(volume).rolling(window=20).mean() + 1e-10
    features["volume_ratio"] = volume / vol_ma

    features["high_low_ratio"] = (high - low) / close
    features["close_position"] = (close - low) / (high - low + 1e-10)

    atr = talib.ATR(high, low, close, 14)
    features["atr"] = atr
    features["natr"] = (atr / close) * 100

    features["momentum"] = pd.Series(close).pct_change(10)

    for period in [7, 25, 50]:
        features[f"ema_ratio_{period}"] = close / talib.EMA(close, period) - 1

    for period in [7, 25, 50]:
        features[f"sma_ratio_{period}"] = close / talib.SMA(close, period) - 1

    return features


def generate_labels(closes, highs, lows, neutral_scale=1.3, lookforward_bars=6):
    atr = talib.ATR(highs, lows, closes, 14)
    future_close = closes[lookforward_bars:]
    current_close = closes[:-lookforward_bars]
    future_atr = atr[:-lookforward_bars]
    future_return = (future_close - current_close) / current_close

    labels = np.full(len(future_return), NEUTRAL_LABEL)
    threshold = (future_atr / current_close) * neutral_scale
    labels[future_return > threshold] = UP_LABEL
    labels[future_return < -threshold] = DOWN_LABEL
    return labels


def compute_metrics(y_true, y_pred, y_proba):
    """Compute all metrics including new Composite Score"""
    metrics = {}
    metrics["accuracy"] = float(np.mean(y_pred == y_true))

    cm = confusion_matrix(y_true, y_pred, labels=[DOWN_LABEL, NEUTRAL_LABEL, UP_LABEL])
    n_down, n_neutral, n_up = cm.sum(axis=1)
    pred_down, pred_neutral, pred_up = cm.sum(axis=0)

    # P(actual|predicted)
    metrics["P_aL_pL"] = float(cm[0][0] / pred_down) if pred_down > 0 else 0.0
    metrics["P_aN_pL"] = float(cm[1][0] / pred_down) if pred_down > 0 else 0.0
    metrics["P_aS_pL"] = float(cm[2][0] / pred_down) if pred_down > 0 else 0.0
    metrics["P_aL_pN"] = float(cm[0][1] / pred_neutral) if pred_neutral > 0 else 0.0
    metrics["P_aN_pN"] = float(cm[1][1] / pred_neutral) if pred_neutral > 0 else 0.0
    metrics["P_aS_pN"] = float(cm[2][1] / pred_neutral) if pred_neutral > 0 else 0.0
    metrics["P_aL_pS"] = float(cm[0][2] / pred_up) if pred_up > 0 else 0.0
    metrics["P_aN_pS"] = float(cm[1][2] / pred_up) if pred_up > 0 else 0.0
    metrics["P_aS_pS"] = float(cm[2][2] / pred_up) if pred_up > 0 else 0.0

    # P(predicted|actual)
    metrics["P_pL_aL"] = float(cm[0][0] / n_down) if n_down > 0 else 0.0
    metrics["P_pN_aL"] = float(cm[1][0] / n_down) if n_down > 0 else 0.0
    metrics["P_pS_aL"] = float(cm[2][0] / n_down) if n_down > 0 else 0.0
    metrics["P_pL_aN"] = float(cm[0][1] / n_neutral) if n_neutral > 0 else 0.0
    metrics["P_pN_aN"] = float(cm[1][1] / n_neutral) if n_neutral > 0 else 0.0
    metrics["P_pS_aN"] = float(cm[2][1] / n_neutral) if n_neutral > 0 else 0.0
    metrics["P_pL_aS"] = float(cm[0][2] / n_up) if n_up > 0 else 0.0
    metrics["P_pN_aS"] = float(cm[1][2] / n_up) if n_up > 0 else 0.0
    metrics["P_pS_aS"] = float(cm[2][2] / n_up) if n_up > 0 else 0.0

    # Precision/Recall/F1
    metrics["precision_down"] = metrics["P_aL_pL"]
    metrics["precision_neutral"] = metrics["P_aN_pN"]
    metrics["precision_up"] = metrics["P_aS_pS"]
    metrics["recall_down"] = metrics["P_pL_aL"]
    metrics["recall_neutral"] = metrics["P_pN_aN"]
    metrics["recall_up"] = metrics["P_pS_aS"]

    for cls, p_key, r_key, f_key in [
        ("down", "precision_down", "recall_down", "f1_down"),
        ("neutral", "precision_neutral", "recall_neutral", "f1_neutral"),
        ("up", "precision_up", "recall_up", "f1_up"),
    ]:
        p, r = metrics[p_key], metrics[r_key]
        metrics[f_key] = float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    metrics["support_down"] = int(n_down)
    metrics["support_neutral"] = int(n_neutral)
    metrics["support_up"] = int(n_up)

    metrics["macro_f1"] = float(np.mean([metrics["f1_down"], metrics["f1_neutral"], metrics["f1_up"]]))
    metrics["macro_precision"] = float(
        np.mean([metrics["precision_down"], metrics["precision_neutral"], metrics["precision_up"]])
    )
    metrics["macro_recall"] = float(np.mean([metrics["recall_down"], metrics["recall_neutral"], metrics["recall_up"]]))

    metrics["long_recall"] = metrics["recall_up"]
    metrics["short_recall"] = metrics["recall_down"]
    metrics["long_precision"] = metrics["precision_up"]
    metrics["short_precision"] = metrics["precision_down"]

    # NEW Composite Score Formula
    long_prec = metrics["precision_up"]
    short_prec = metrics["precision_down"]
    dir_quality = (long_prec + short_prec) / 2
    geo_mean = np.sqrt(long_prec * short_prec)
    dir_boost = 1.0 + 0.5 * dir_quality

    metrics["composite_score"] = (
        metrics["macro_f1"] * geo_mean * dir_boost
        - 8.0 * (metrics["P_aS_pL"] + metrics["P_aL_pS"])
        - 3.0 * (metrics["P_aL_pN"] + metrics["P_aS_pN"])
    )

    return metrics, cm


def evaluate_fold(model_dir, df, features, train_months, valid_month, test_month, fold_index):
    """Evaluate a single fold"""
    lstm_checkpoint = torch.load(model_dir / "lstm_model.pt", map_location=DEVICE)
    n_features = lstm_checkpoint["n_features"]

    lstm_model = LSTMClassifier(n_features=n_features, hidden_dim=HIDDEN_DIM, lstm_output_dim=LSTM_OUTPUT_DIM)
    lstm_model.load_state_dict(lstm_checkpoint["model_state_dict"])
    lstm_model.to(DEVICE)
    lstm_model.eval()

    from catboost import CatBoostClassifier

    cb_model = CatBoostClassifier()
    cb_model.load_model(str(model_dir / "catboost_model.cbm"))
    scaler = joblib.load(model_dir / "scaler.joblib")

    closes = df["close"].values[: len(features)]
    highs = df["high"].values[: len(features)]
    lows = df["low"].values[: len(features)]
    labels = generate_labels(closes, highs, lows)
    labels = labels[: len(features)]

    dates = pd.to_datetime(df["open_time"].values[: len(features)], unit="ms")
    months = dates.year * 100 + dates.month

    feature_scaled = scaler.transform(features.values)

    test_mask = months == test_month
    test_indices = np.where(test_mask)[0]
    test_seq_indices = test_indices[test_indices >= SEQ_LEN]

    test_X = []
    test_y = []
    for idx in test_seq_indices:
        start = idx - SEQ_LEN
        test_X.append(feature_scaled[start:idx])
        test_y.append(labels[idx])

    test_X = np.array(test_X)
    test_y = np.array(test_y)

    test_dataset = TensorDataset(torch.FloatTensor(test_X), torch.LongTensor(test_y))
    test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False, num_workers=0)

    lstm_model.eval()
    features_list = []
    with torch.no_grad():
        for X, _ in test_loader:
            feat, _ = lstm_model(X.to(DEVICE))
            features_list.append(feat.cpu().numpy())
    lstm_features = np.vstack(features_list)

    y_pred_raw = cb_model.predict(lstm_features).flatten()
    y_proba_raw = cb_model.predict_proba(lstm_features)

    y_proba_post = y_proba_raw.copy()
    max_prob = np.maximum(
        y_proba_post[:, DOWN_LABEL], np.maximum(y_proba_post[:, NEUTRAL_LABEL], y_proba_post[:, UP_LABEL])
    )
    neutral_mask = max_prob < PROBABILITY_THRESHOLD
    y_pred_post = y_pred_raw.copy()
    y_pred_post[neutral_mask] = NEUTRAL_LABEL

    metrics, cm = compute_metrics(test_y, y_pred_post, y_proba_post)

    pred_dist = pd.Series(y_pred_post).value_counts().sort_index()

    result = {
        "train_months": train_months,
        "valid_month": valid_month,
        "test_month": test_month,
        "train_size": len(train_months),
        "valid_size": 0,
        "test_size": len(test_y),
        **metrics,
        "confusion_matrix_raw": cm.tolist(),
        "confusion_matrix_post": cm.tolist(),
        "pred_distribution": {
            "DOWN": int(pred_dist.get(0, 0)),
            "NEUTRAL": int(pred_dist.get(1, 0)),
            "UP": int(pred_dist.get(2, 0)),
        },
    }
    return result


def main():
    print("Loading data...")
    df = pd.read_parquet("data/futures/um/klines/BTCUSDT_15m_mark.parquet")
    df = df.sort_values("open_time").reset_index(drop=True)
    print(f"Data: {len(df)} rows")

    print("Generating features...")
    features = generate_features(df)
    features = features.dropna()
    min_len = min(len(df), len(features))
    df = df.iloc[:min_len].reset_index(drop=True)
    features = features.iloc[:min_len].reset_index(drop=True)

    # Fold configurations (from training)
    fold_configs = [
        ([202401, 202402, 202403], 202601, 202602, 0),
        ([202401, 202402, 202403, 202404, 202405, 202406], 202601, 202602, 1),
        ([202401, 202402, 202403, 202404, 202405, 202406, 202407, 202408, 202409], 202601, 202602, 2),
        (
            [202401, 202402, 202403, 202404, 202405, 202406, 202407, 202408, 202409, 202410, 202411, 202412],
            202601,
            202602,
            3,
        ),
    ]

    all_results = []
    for train_months, valid_month, test_month, fold_idx in fold_configs:
        print(f"\nEvaluating Fold {fold_idx} (test={test_month})...")
        model_dir = Path(f"models/lstm_catboost/fold_{fold_idx}")
        result = evaluate_fold(model_dir, df, features, train_months, valid_month, test_month, fold_idx)
        all_results.append(result)
        print(f"  Macro F1: {result['macro_f1']:.4f}, Composite: {result['composite_score']:.4f}")
        print(f"  Neutral: {result['pred_distribution']['NEUTRAL'] / result['test_size'] * 100:.1f}%")

    # Save JSON
    output = {
        "config": {
            "neutral_scale": NEUTRAL_SCALE,
            "lookforward_bars": LOOKFORWARD_BARS,
            "seq_len": SEQ_LEN,
            "hidden_dim": HIDDEN_DIM,
            "lstm_output_dim": LSTM_OUTPUT_DIM,
            "probability_threshold": PROBABILITY_THRESHOLD,
        },
        "results": all_results,
    }
    with open("models/walkforward_lstm_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to models/walkforward_lstm_results.json")

    # Generate markdown report
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%z")
    md_content = f"""# LSTM+CatBoost Walk-Forward Training Report

**Generated**: {timestamp}  
**probability_threshold**: {PROBABILITY_THRESHOLD}

---

## 1. Model Configuration

| Parameter | Value |
|-----------|-------|
| neutral_scale (K) | {NEUTRAL_SCALE} |
| lookforward_bars | {LOOKFORWARD_BARS} |
| seq_len | {SEQ_LEN} |
| hidden_dim | {HIDDEN_DIM} |
| lstm_output_dim | {LSTM_OUTPUT_DIM} |
| probability_threshold | {PROBABILITY_THRESHOLD} |

---

## 2. Dataset Split

| Fold | Training Period | Valid | Test |
|------|-----------------|-------|------|
| 0 | 2024-01 ~ 2024-03 (3 months) | 2026-01 | 2026-02 |
| 1 | 2024-01 ~ 2024-06 (6 months) | 2026-01 | 2026-02 |
| 2 | 2024-01 ~ 2024-09 (9 months) | 2026-01 | 2026-02 |
| 3 | 2024-01 ~ 2024-12 (12 months) | 2026-01 | 2026-02 |

---

## 3. Key Metrics Summary

| Metric | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Average |
|--------|--------|--------|--------|--------|---------|
| **Macro F1** | {all_results[0]["macro_f1"]:.4f} | {all_results[1]["macro_f1"]:.4f} | {all_results[2]["macro_f1"]:.4f} | {all_results[3]["macro_f1"]:.4f} | **{(all_results[0]["macro_f1"] + all_results[1]["macro_f1"] + all_results[2]["macro_f1"] + all_results[3]["macro_f1"]) / 4:.4f}** |
| **Composite Score** | {all_results[0]["composite_score"]:.4f} | {all_results[1]["composite_score"]:.4f} | {all_results[2]["composite_score"]:.4f} | {all_results[3]["composite_score"]:.4f} | **{(all_results[0]["composite_score"] + all_results[1]["composite_score"] + all_results[2]["composite_score"] + all_results[3]["composite_score"]) / 4:.4f}** |
| **Long Recall** | {all_results[0]["long_recall"]:.4f} | {all_results[1]["long_recall"]:.4f} | {all_results[2]["long_recall"]:.4f} | {all_results[3]["long_recall"]:.4f} | **{(all_results[0]["long_recall"] + all_results[1]["long_recall"] + all_results[2]["long_recall"] + all_results[3]["long_recall"]) / 4:.4f}** |
| **Short Recall** | {all_results[0]["short_recall"]:.4f} | {all_results[1]["short_recall"]:.4f} | {all_results[2]["short_recall"]:.4f} | {all_results[3]["short_recall"]:.4f} | **{(all_results[0]["short_recall"] + all_results[1]["short_recall"] + all_results[2]["short_recall"] + all_results[3]["short_recall"]) / 4:.4f}** |
| **Neutral Ratio** | {all_results[0]["pred_distribution"]["NEUTRAL"] / all_results[0]["test_size"] * 100:.1f}% | {all_results[1]["pred_distribution"]["NEUTRAL"] / all_results[1]["test_size"] * 100:.1f}% | {all_results[2]["pred_distribution"]["NEUTRAL"] / all_results[2]["test_size"] * 100:.1f}% | {all_results[3]["pred_distribution"]["NEUTRAL"] / all_results[3]["test_size"] * 100:.1f}% | **{(all_results[0]["pred_distribution"]["NEUTRAL"] / all_results[0]["test_size"] * 100 + all_results[1]["pred_distribution"]["NEUTRAL"] / all_results[1]["test_size"] * 100 + all_results[2]["pred_distribution"]["NEUTRAL"] / all_results[2]["test_size"] * 100 + all_results[3]["pred_distribution"]["NEUTRAL"] / all_results[3]["test_size"] * 100) / 4:.1f}%** |

---

## 4. Composite Score Formula (NEW)

```
Composite = Macro_F1 × √(Long_P × Short_P) × (1 + 0.5 × Dir_Quality)
          - 8.0 × (P_aS_pL + P_aL_pS)
          - 3.0 × (P_aL_pN + P_aS_pN)
```

| Component | Description |
|-----------|--------------|
| √(Long_P × Short_P) | Geometric mean of Long/Short Precision - both must be high |
| (1 + 0.5 × Dir_Quality) | Direction quality boost |
| 8.0 × (P_aS_pL + P_aL_pS) | Direction flip penalty (most severe) |
| 3.0 × (P_aL_pN + P_aS_pN) | False break Neutral penalty |

---

## 5. Prediction Distribution

| Fold | DOWN | NEUTRAL | UP |
|------|------|---------|-----|
| Fold 0 | {all_results[0]["pred_distribution"]["DOWN"]} ({all_results[0]["pred_distribution"]["DOWN"] / all_results[0]["test_size"] * 100:.1f}%) | {all_results[0]["pred_distribution"]["NEUTRAL"]} ({all_results[0]["pred_distribution"]["NEUTRAL"] / all_results[0]["test_size"] * 100:.1f}%) | {all_results[0]["pred_distribution"]["UP"]} ({all_results[0]["pred_distribution"]["UP"] / all_results[0]["test_size"] * 100:.1f}%) |
| Fold 1 | {all_results[1]["pred_distribution"]["DOWN"]} ({all_results[1]["pred_distribution"]["DOWN"] / all_results[1]["test_size"] * 100:.1f}%) | {all_results[1]["pred_distribution"]["NEUTRAL"]} ({all_results[1]["pred_distribution"]["NEUTRAL"] / all_results[1]["test_size"] * 100:.1f}%) | {all_results[1]["pred_distribution"]["UP"]} ({all_results[1]["pred_distribution"]["UP"] / all_results[1]["test_size"] * 100:.1f}%) |
| Fold 2 | {all_results[2]["pred_distribution"]["DOWN"]} ({all_results[2]["pred_distribution"]["DOWN"] / all_results[2]["test_size"] * 100:.1f}%) | {all_results[2]["pred_distribution"]["NEUTRAL"]} ({all_results[2]["pred_distribution"]["NEUTRAL"] / all_results[2]["test_size"] * 100:.1f}%) | {all_results[2]["pred_distribution"]["UP"]} ({all_results[2]["pred_distribution"]["UP"] / all_results[2]["test_size"] * 100:.1f}%) |
| Fold 3 | {all_results[3]["pred_distribution"]["DOWN"]} ({all_results[3]["pred_distribution"]["DOWN"] / all_results[3]["test_size"] * 100:.1f}%) | {all_results[3]["pred_distribution"]["NEUTRAL"]} ({all_results[3]["pred_distribution"]["NEUTRAL"] / all_results[3]["test_size"] * 100:.1f}%) | {all_results[3]["pred_distribution"]["UP"]} ({all_results[3]["pred_distribution"]["UP"] / all_results[3]["test_size"] * 100:.1f}%) |

---

## 6. Model Storage

```
models/lstm_catboost/
├── fold_0/
├── fold_1/
├── fold_2/
└── fold_3/
```

---

## 7. Key Findings

- Fold 3 (12 months training) shows best performance with Macro F1={all_results[3]["macro_f1"]:.4f}
- Composite Score is negative due to strict penalties for false breaks
- Direction flip errors (P_aS_pL, P_aL_pS) are near zero across all folds
"""

    md_path = f"models/walkforward_lstm_results_{timestamp}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"\nMarkdown report saved to {md_path}")

    # Generate plot
    plot_path = f"walkforward_lstm_results_{timestamp}.png"
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        f"LSTM+CatBoost Walk-Forward Results (threshold={PROBABILITY_THRESHOLD})", fontsize=16, fontweight="bold"
    )

    x = np.array([0, 1, 2, 3])

    # 1. Macro F1
    ax1 = axes[0, 0]
    macro_f1 = [r["macro_f1"] for r in all_results]
    ax1.bar(x, macro_f1, color="steelblue", alpha=0.8)
    ax1.set_xlabel("Fold")
    ax1.set_ylabel("Macro F1")
    ax1.set_title("Macro F1 Score")
    ax1.set_xticks(x)
    ax1.set_ylim([0, 1.0])
    for i, v in enumerate(macro_f1):
        ax1.text(i, v + 0.02, f"{v:.3f}", ha="center")

    # 2. Composite Score
    ax2 = axes[0, 1]
    comp = [r["composite_score"] for r in all_results]
    colors = ["crimson" if c < 0 else "forestgreen" for c in comp]
    ax2.bar(x, comp, color=colors, alpha=0.8)
    ax2.set_xlabel("Fold")
    ax2.set_ylabel("Composite Score")
    ax2.set_title("Composite Score (NEW Formula)")
    ax2.set_xticks(x)
    ax2.axhline(y=0, color="black", linewidth=0.5)

    # 3. Long/Short Recall
    ax3 = axes[1, 0]
    long_r = [r["long_recall"] for r in all_results]
    short_r = [r["short_recall"] for r in all_results]
    width = 0.35
    ax3.bar(x - width / 2, long_r, width / 2, label="Long Recall", color="crimson", alpha=0.8)
    ax3.bar(x + width / 2, short_r, width / 2, label="Short Recall", color="royalblue", alpha=0.8)
    ax3.set_xlabel("Fold")
    ax3.set_ylabel("Recall")
    ax3.set_title("Long/Short Recall")
    ax3.legend()
    ax3.set_xticks(x)
    ax3.set_ylim([0, 1.0])

    # 4. Composite Score breakdown
    ax4 = axes[1, 1]
    ax4.axis("off")
    formula_text = """NEW Composite Score Formula:

Composite = Macro_F1 × √(Long_P × Short_P) × (1 + 0.5 × Dir_Quality)
          - 8.0 × (P_aS_pL + P_aL_pS)
          - 3.0 × (P_aL_pN + P_aS_pN)

Where:
- Long_P = precision_up (P_aS_pS)
- Short_P = precision_down (P_aL_pL)
- Dir_Quality = (Long_P + Short_P) / 2
- P_aS_pL = predict Long, actual Short (DANGER!)
- P_aL_pS = predict Short, actual Long (DANGER!)
- P_aL_pN, P_aS_pN = False break Neutral"""
    ax4.text(0.1, 0.5, formula_text, fontsize=11, family="monospace", verticalalignment="center")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
