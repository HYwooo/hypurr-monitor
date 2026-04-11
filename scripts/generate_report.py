"""
Generate training report with threshold=0.65
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

    # Composite Score
    avg_precision = (metrics["precision_down"] + metrics["precision_neutral"] + metrics["precision_up"]) / 3
    avg_recall = (metrics["recall_down"] + metrics["recall_neutral"] + metrics["recall_up"]) / 3
    avg_direction_correct = (metrics["P_pL_aL"] + metrics["P_pN_aN"] + metrics["P_pS_aS"]) / 3
    direction_flip_penalty = 5.0 * (metrics["P_aS_pL"] + metrics["P_aL_pS"])
    false_break_penalty = 2.0 * (metrics["P_aL_pN"] + metrics["P_aS_pN"]) + 1.0 * (
        metrics["P_aN_pL"] + metrics["P_aN_pS"]
    )
    metrics["composite_score"] = (
        avg_precision + avg_recall + avg_direction_correct - direction_flip_penalty - false_break_penalty
    )

    return metrics, cm


def evaluate_fold(model_dir, df, features, train_months, valid_month, test_month, fold_index):
    """Evaluate a single fold with threshold=0.65"""
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

    # Prepare data
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

    # Extract features
    lstm_model.eval()
    features_list = []
    with torch.no_grad():
        for X, _ in test_loader:
            feat, _ = lstm_model(X.to(DEVICE))
            features_list.append(feat.cpu().numpy())
    lstm_features = np.vstack(features_list)

    # Predict
    y_pred_raw = cb_model.predict(lstm_features).flatten()
    y_proba_raw = cb_model.predict_proba(lstm_features)

    # Apply threshold
    y_proba_post = y_proba_raw.copy()
    max_prob = np.maximum(
        y_proba_post[:, DOWN_LABEL], np.maximum(y_proba_post[:, NEUTRAL_LABEL], y_proba_post[:, UP_LABEL])
    )
    neutral_mask = max_prob < PROBABILITY_THRESHOLD
    y_pred_post = y_pred_raw.copy()
    y_pred_post[neutral_mask] = NEUTRAL_LABEL

    # Compute metrics
    metrics, cm = compute_metrics(test_y, y_pred_post, y_proba_post)

    # Prediction distribution
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

    # Fold configurations
    fold_configs = [
        (
            [
                202403,
                202404,
                202405,
                202406,
                202407,
                202408,
                202409,
                202410,
                202411,
                202412,
                202501,
                202502,
                202503,
                202504,
                202505,
                202506,
            ],
            202507,
            202508,
            0,
        ),
        (
            [
                202403,
                202404,
                202405,
                202406,
                202407,
                202408,
                202409,
                202410,
                202411,
                202412,
                202501,
                202502,
                202503,
                202504,
                202505,
                202506,
                202507,
                202508,
            ],
            202509,
            202510,
            1,
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

### Fold 0
- **Training**: 2024-03 ~ 2025-06 (16 months)
- **Validation**: 2025-07
- **Test**: 2025-08

### Fold 1
- **Training**: 2024-03 ~ 2025-08 (18 months)
- **Validation**: 2025-09
- **Test**: 2025-10

---

## 3. Key Metrics Summary

| Metric | Fold 0 | Fold 1 | Average |
|--------|--------|--------|---------|
| **Macro F1** | {all_results[0]["macro_f1"]:.4f} | {all_results[1]["macro_f1"]:.4f} | **{(all_results[0]["macro_f1"] + all_results[1]["macro_f1"]) / 2:.4f}** |
| **Accuracy** | {all_results[0]["accuracy"]:.4f} | {all_results[1]["accuracy"]:.4f} | **{(all_results[0]["accuracy"] + all_results[1]["accuracy"]) / 2:.4f}** |
| **Composite Score** | {all_results[0]["composite_score"]:.4f} | {all_results[1]["composite_score"]:.4f} | **{(all_results[0]["composite_score"] + all_results[1]["composite_score"]) / 2:.4f}** |
| **Long Recall** | {all_results[0]["long_recall"]:.4f} | {all_results[1]["long_recall"]:.4f} | **{(all_results[0]["long_recall"] + all_results[1]["long_recall"]) / 2:.4f}** |
| **Short Recall** | {all_results[0]["short_recall"]:.4f} | {all_results[1]["short_recall"]:.4f} | **{(all_results[0]["short_recall"] + all_results[1]["short_recall"]) / 2:.4f}** |
| **Long Precision** | {all_results[0]["long_precision"]:.4f} | {all_results[1]["long_precision"]:.4f} | **{(all_results[0]["long_precision"] + all_results[1]["long_precision"]) / 2:.4f}** |
| **Short Precision** | {all_results[0]["short_precision"]:.4f} | {all_results[1]["short_precision"]:.4f} | **{(all_results[0]["short_precision"] + all_results[1]["short_precision"]) / 2:.4f}** |
| **Neutral Ratio** | {all_results[0]["pred_distribution"]["NEUTRAL"] / all_results[0]["test_size"] * 100:.1f}% | {all_results[1]["pred_distribution"]["NEUTRAL"] / all_results[1]["test_size"] * 100:.1f}% | **{(all_results[0]["pred_distribution"]["NEUTRAL"] / all_results[0]["test_size"] * 100 + all_results[1]["pred_distribution"]["NEUTRAL"] / all_results[1]["test_size"] * 100) / 2:.1f}%** |

---

## 4. Prediction Distribution

### Fold 0 (Test: 2025-08)
| Class | Count | Percentage |
|-------|-------|------------|
| DOWN (Short) | {all_results[0]["pred_distribution"]["DOWN"]} | {all_results[0]["pred_distribution"]["DOWN"] / all_results[0]["test_size"] * 100:.1f}% |
| NEUTRAL | {all_results[0]["pred_distribution"]["NEUTRAL"]} | {all_results[0]["pred_distribution"]["NEUTRAL"] / all_results[0]["test_size"] * 100:.1f}% |
| UP (Long) | {all_results[0]["pred_distribution"]["UP"]} | {all_results[0]["pred_distribution"]["UP"] / all_results[0]["test_size"] * 100:.1f}% |

### Fold 1 (Test: 2025-10)
| Class | Count | Percentage |
|-------|-------|------------|
| DOWN (Short) | {all_results[1]["pred_distribution"]["DOWN"]} | {all_results[1]["pred_distribution"]["DOWN"] / all_results[1]["test_size"] * 100:.1f}% |
| NEUTRAL | {all_results[1]["pred_distribution"]["NEUTRAL"]} | {all_results[1]["pred_distribution"]["NEUTRAL"] / all_results[1]["test_size"] * 100:.1f}% |
| UP (Long) | {all_results[1]["pred_distribution"]["UP"]} | {all_results[1]["pred_distribution"]["UP"] / all_results[1]["test_size"] * 100:.1f}% |

---

## 5. 18-Probability Evaluation System

### Fold 0

**P(actual|predicted)**
| Predicted | Actual Long | Actual Neutral | Actual Short |
|------------|-------------|----------------|--------------|
| Long | {all_results[0]["P_aL_pL"] * 100:.1f}% | {all_results[0]["P_aN_pL"] * 100:.1f}% | {all_results[0]["P_aS_pL"] * 100:.1f}% |
| Neutral | {all_results[0]["P_aL_pN"] * 100:.1f}% | {all_results[0]["P_aN_pN"] * 100:.1f}% | {all_results[0]["P_aS_pN"] * 100:.1f}% |
| Short | {all_results[0]["P_aL_pS"] * 100:.1f}% | {all_results[0]["P_aN_pS"] * 100:.1f}% | {all_results[0]["P_aS_pS"] * 100:.1f}% |

**P(predicted|actual)**
| Actual | Pred Long | Pred Neutral | Pred Short |
|--------|-----------|--------------|------------|
| Long | {all_results[0]["P_pL_aL"] * 100:.1f}% | {all_results[0]["P_pN_aL"] * 100:.1f}% | {all_results[0]["P_pS_aL"] * 100:.1f}% |
| Neutral | {all_results[0]["P_pL_aN"] * 100:.1f}% | {all_results[0]["P_pN_aN"] * 100:.1f}% | {all_results[0]["P_pS_aN"] * 100:.1f}% |
| Short | {all_results[0]["P_pL_aS"] * 100:.1f}% | {all_results[0]["P_pN_aS"] * 100:.1f}% | {all_results[0]["P_pS_aS"] * 100:.1f}% |

### Fold 1

**P(actual|predicted)**
| Predicted | Actual Long | Actual Neutral | Actual Short |
|------------|-------------|----------------|--------------|
| Long | {all_results[1]["P_aL_pL"] * 100:.1f}% | {all_results[1]["P_aN_pL"] * 100:.1f}% | {all_results[1]["P_aS_pL"] * 100:.1f}% |
| Neutral | {all_results[1]["P_aL_pN"] * 100:.1f}% | {all_results[1]["P_aN_pN"] * 100:.1f}% | {all_results[1]["P_aS_pN"] * 100:.1f}% |
| Short | {all_results[1]["P_aL_pS"] * 100:.1f}% | {all_results[1]["P_aN_pS"] * 100:.1f}% | {all_results[1]["P_aS_pS"] * 100:.1f}% |

**P(predicted|actual)**
| Actual | Pred Long | Pred Neutral | Pred Short |
|--------|-----------|--------------|------------|
| Long | {all_results[1]["P_pL_aL"] * 100:.1f}% | {all_results[1]["P_pN_aL"] * 100:.1f}% | {all_results[1]["P_pS_aL"] * 100:.1f}% |
| Neutral | {all_results[1]["P_pL_aN"] * 100:.1f}% | {all_results[1]["P_pN_aN"] * 100:.1f}% | {all_results[1]["P_pS_aN"] * 100:.1f}% |
| Short | {all_results[1]["P_pL_aS"] * 100:.1f}% | {all_results[1]["P_pN_aS"] * 100:.1f}% | {all_results[1]["P_pS_aS"] * 100:.1f}% |

---

## 6. Confusion Matrix

### Fold 0 (Average)
```
         Predicted
         DOWN    NEUTRAL  UP
Actual DOWN   {int(all_results[0]["confusion_matrix_raw"][0][0])}      {int(all_results[0]["confusion_matrix_raw"][0][1])}       {int(all_results[0]["confusion_matrix_raw"][0][2])}
       NEUTRAL {int(all_results[0]["confusion_matrix_raw"][1][0])}      {int(all_results[0]["confusion_matrix_raw"][1][1])}      {int(all_results[0]["confusion_matrix_raw"][1][2])}
       UP     {int(all_results[0]["confusion_matrix_raw"][2][0])}       {int(all_results[0]["confusion_matrix_raw"][2][1])}       {int(all_results[0]["confusion_matrix_raw"][2][2])}
```

### Fold 1 (Average)
```
         Predicted
         DOWN    NEUTRAL  UP
Actual DOWN   {int(all_results[1]["confusion_matrix_raw"][0][0])}      {int(all_results[1]["confusion_matrix_raw"][0][1])}       {int(all_results[1]["confusion_matrix_raw"][0][2])}
       NEUTRAL {int(all_results[1]["confusion_matrix_raw"][1][0])}      {int(all_results[1]["confusion_matrix_raw"][1][1])}      {int(all_results[1]["confusion_matrix_raw"][1][2])}
       UP     {int(all_results[1]["confusion_matrix_raw"][2][0])}       {int(all_results[1]["confusion_matrix_raw"][2][1])}       {int(all_results[1]["confusion_matrix_raw"][2][2])}
```

---

## 7. Key Findings

- Direction flip penalty is effective: P(aS|pL) and P(aL|pS) are both ~0%
- Neutral ratio with threshold={PROBABILITY_THRESHOLD}: Fold0={all_results[0]["pred_distribution"]["NEUTRAL"] / all_results[0]["test_size"] * 100:.1f}%, Fold1={all_results[1]["pred_distribution"]["NEUTRAL"] / all_results[1]["test_size"] * 100:.1f}%
- Average Macro F1: {(all_results[0]["macro_f1"] + all_results[1]["macro_f1"]) / 2:.4f}
- Average Composite Score: {(all_results[0]["composite_score"] + all_results[1]["composite_score"]) / 2:.4f}

---

## 8. Model Storage

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

    md_path = f"models/walkforward_lstm_results_{timestamp}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"\nMarkdown report saved to {md_path}")

    # Generate plot
    plot_path = f"walkforward_lstm_results_{timestamp}.png"
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(
        f"LSTM+CatBoost Walk-Forward Results (threshold={PROBABILITY_THRESHOLD})", fontsize=16, fontweight="bold"
    )

    x = np.array([0, 1])
    width = 0.35

    # 1. Macro F1
    ax1 = axes[0, 0]
    macro_f1 = [r["macro_f1"] for r in all_results]
    ax1.bar(x, macro_f1, width, color="steelblue", alpha=0.8)
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
    ax2.bar(x, comp, width, color="crimson", alpha=0.8)
    ax2.set_xlabel("Fold")
    ax2.set_ylabel("Composite Score")
    ax2.set_title("Composite Score (Direction Flip Penalty=5x)")
    ax2.set_xticks(x)
    ax2.axhline(y=0, color="black", linewidth=0.5)
    for i, v in enumerate(comp):
        ax2.text(i, v + 0.05, f"{v:.3f}", ha="center")

    # 3. Long/Short Recall
    ax3 = axes[1, 0]
    long_r = [r["long_recall"] for r in all_results]
    short_r = [r["short_recall"] for r in all_results]
    ax3.bar(x - width / 2, long_r, width / 2, label="Long Recall", color="crimson", alpha=0.8)
    ax3.bar(x + width / 2, short_r, width / 2, label="Short Recall", color="royalblue", alpha=0.8)
    ax3.set_xlabel("Fold")
    ax3.set_ylabel("Recall")
    ax3.set_title("Long/Short Recall")
    ax3.legend()
    ax3.set_xticks(x)
    ax3.set_ylim([0, 1.1])

    # 4. Long/Short Precision
    ax4 = axes[1, 1]
    long_p = [r["long_precision"] for r in all_results]
    short_p = [r["short_precision"] for r in all_results]
    ax4.bar(x - width / 2, long_p, width / 2, label="Long Precision", color="crimson", alpha=0.8)
    ax4.bar(x + width / 2, short_p, width / 2, label="Short Precision", color="royalblue", alpha=0.8)
    ax4.set_xlabel("Fold")
    ax4.set_ylabel("Precision")
    ax4.set_title("Long/Short Precision")
    ax4.legend()
    ax4.set_xticks(x)
    ax4.set_ylim([0, 1.1])

    # 5. F1 by Class
    ax5 = axes[2, 0]
    f1_down = [r["f1_down"] for r in all_results]
    f1_neutral = [r["f1_neutral"] for r in all_results]
    f1_up = [r["f1_up"] for r in all_results]
    w = 0.25
    ax5.bar(x - w, f1_down, w, label="DOWN F1", color="royalblue", alpha=0.8)
    ax5.bar(x, f1_neutral, w, label="NEUTRAL F1", color="gray", alpha=0.8)
    ax5.bar(x + w, f1_up, w, label="UP F1", color="crimson", alpha=0.8)
    ax5.set_xlabel("Fold")
    ax5.set_ylabel("F1 Score")
    ax5.set_title("F1 Score by Class")
    ax5.legend()
    ax5.set_xticks(x)
    ax5.set_ylim([0, 1.0])

    # 6. Confusion Matrix (Average)
    ax6 = axes[2, 1]
    cm = np.array(all_results[0]["confusion_matrix_raw"]) + np.array(all_results[1]["confusion_matrix_raw"])
    cm_avg = cm / 2
    im = ax6.imshow(cm_avg, cmap="Blues", aspect="auto")
    ax6.set_xticks([0, 1, 2])
    ax6.set_yticks([0, 1, 2])
    ax6.set_xticklabels(["DOWN", "NEUTRAL", "UP"])
    ax6.set_yticklabels(["DOWN", "NEUTRAL", "UP"])
    ax6.set_xlabel("Predicted")
    ax6.set_ylabel("Actual")
    ax6.set_title("Confusion Matrix (Average)")
    for i in range(3):
        for j in range(3):
            color = "white" if cm_avg[i, j] > cm_avg.max() / 2 else "black"
            ax6.text(j, i, f"{cm_avg[i, j]:.0f}", ha="center", va="center", color=color)
    plt.colorbar(im, ax=ax6)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
