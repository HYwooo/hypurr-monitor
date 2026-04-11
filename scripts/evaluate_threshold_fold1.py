"""
Evaluate Fold 1 with different probability_threshold values

Usage:
    uv run python scripts/evaluate_threshold_fold1.py
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import talib
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))

# Constants (must match training)
NEUTRAL_SCALE = 1.3
LOOKFORWARD_BARS = 6
SEQ_LEN = 48
HIDDEN_DIM = 128
LSTM_OUTPUT_DIM = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Labels
DOWN_LABEL = 0
NEUTRAL_LABEL = 1
UP_LABEL = 2


class LSTMEncoder(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int = HIDDEN_DIM, lstm_output_dim: int = LSTM_OUTPUT_DIM):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, lstm_output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        last_hidden = hidden[-1]
        return self.fc(last_hidden)


class LSTMClassifier(nn.Module):
    def __init__(
        self, n_features: int, hidden_dim: int = HIDDEN_DIM, lstm_output_dim: int = LSTM_OUTPUT_DIM, n_classes: int = 3
    ):
        super().__init__()
        self.encoder = LSTMEncoder(n_features, hidden_dim, lstm_output_dim)
        self.classifier = nn.Linear(lstm_output_dim, n_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        logits = self.classifier(features)
        return features, logits


def load_data(symbol: str = "BTCUSDT", interval: str = "15m") -> pd.DataFrame:
    mark_parquet = Path("data/futures/um/klines/BTCUSDT_15m_mark.parquet")
    if mark_parquet.exists():
        df = pd.read_parquet(mark_parquet)
        df = df.sort_values("open_time").reset_index(drop=True)
        return df
    raise FileNotFoundError(f"Mark price data not found at {mark_parquet}")


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
        ema = talib.EMA(close, period)
        features[f"ema_ratio_{period}"] = close / ema - 1

    for period in [7, 25, 50]:
        sma = talib.SMA(close, period)
        features[f"sma_ratio_{period}"] = close / sma - 1

    return features


def generate_labels(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    neutral_scale: float = NEUTRAL_SCALE,
    lookforward_bars: int = LOOKFORWARD_BARS,
) -> np.ndarray:
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


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, Any]:
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

    return metrics


def evaluate_with_threshold(
    model_dir: Path, df: pd.DataFrame, features: pd.DataFrame, threshold: float
) -> dict[str, Any]:
    """Evaluate Fold 1 test set with given threshold"""

    # Load models
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

    # Prepare Fold 1 test data (202510)
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values

    labels = generate_labels(closes, highs, lows)
    labels = labels[: len(features)]

    open_times = df["open_time"].values[: len(features)]
    dates = pd.to_datetime(open_times, unit="ms")
    months = dates.year * 100 + dates.month

    feature_values = features.values
    feature_scaled = scaler.transform(feature_values)

    # Test month for Fold 1 is 202510
    test_mask = months == 202510
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

    # Extract features and predict
    lstm_model.eval()
    features_list = []
    with torch.no_grad():
        for X, _ in test_loader:
            X = X.to(DEVICE)
            feat, _ = lstm_model(X)
            features_list.append(feat.cpu().numpy())

    lstm_features = np.vstack(features_list)

    y_pred_raw = cb_model.predict(lstm_features).flatten()
    y_proba_raw = cb_model.predict_proba(lstm_features)

    # Apply threshold
    y_proba_post = y_proba_raw.copy()
    max_prob = np.maximum(
        y_proba_post[:, DOWN_LABEL], np.maximum(y_proba_post[:, NEUTRAL_LABEL], y_proba_post[:, UP_LABEL])
    )
    neutral_mask = max_prob < threshold
    y_pred_post = y_pred_raw.copy()
    y_pred_post[neutral_mask] = NEUTRAL_LABEL

    metrics = compute_metrics(test_y, y_pred_post, y_proba_post)

    # Compute prediction distribution
    pred_dist = pd.Series(y_pred_post).value_counts().sort_index()
    total = len(y_pred_post)

    return {
        "threshold": threshold,
        "test_size": len(test_y),
        "pred_distribution": {
            "DOWN": int(pred_dist.get(0, 0)),
            "NEUTRAL": int(pred_dist.get(1, 0)),
            "UP": int(pred_dist.get(2, 0)),
            "DOWN_pct": pred_dist.get(0, 0) / total * 100,
            "NEUTRAL_pct": pred_dist.get(1, 0) / total * 100,
            "UP_pct": pred_dist.get(2, 0) / total * 100,
        },
        **metrics,
    }


def main():
    print("=" * 80)
    print("Evaluate Fold 1 with Different probability_threshold Values")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    df = load_data()
    print(f"Data: {len(df)} rows")
    print(
        f"Time range: {datetime.fromtimestamp(df['open_time'].min() / 1000)} to {datetime.fromtimestamp(df['open_time'].max() / 1000)}"
    )

    print("\nGenerating features...")
    features = generate_features(df)
    features = features.dropna()
    print(f"Features: {features.shape}")

    min_len = min(len(df), len(features))
    df = df.iloc[:min_len].reset_index(drop=True)
    features = features.iloc[:min_len].reset_index(drop=True)

    model_dir = Path("models/lstm_catboost/fold_1")

    thresholds = [0.16, 0.18, 0.22]

    print("\n" + "=" * 80)
    print("Evaluating Fold 1 (test_month=202510) with different thresholds")
    print("=" * 80)

    all_results = []

    for threshold in thresholds:
        print(f"\n{'=' * 60}")
        print(f"Threshold = {threshold}")
        print("=" * 60)

        result = evaluate_with_threshold(model_dir, df, features, threshold)
        all_results.append(result)

        print(f"\nTest size: {result['test_size']}")
        print(f"\nPrediction Distribution:")
        print(f"  DOWN:    {result['pred_distribution']['DOWN']:4d} ({result['pred_distribution']['DOWN_pct']:5.1f}%)")
        print(
            f"  NEUTRAL: {result['pred_distribution']['NEUTRAL']:4d} ({result['pred_distribution']['NEUTRAL_pct']:5.1f}%)"
        )
        print(f"  UP:      {result['pred_distribution']['UP']:4d} ({result['pred_distribution']['UP_pct']:5.1f}%)")

        print(f"\nMetrics:")
        print(f"  Macro F1:         {result['macro_f1']:.4f}")
        print(f"  Accuracy:         {result['accuracy']:.4f}")
        print(f"  Composite Score: {result['composite_score']:.4f}")
        print(f"  Long Recall:     {result['long_recall']:.4f}")
        print(f"  Short Recall:    {result['short_recall']:.4f}")
        print(f"  Long Precision:  {result['long_precision']:.4f}")
        print(f"  Short Precision: {result['short_precision']:.4f}")

        print(f"\n18-Probability Metrics:")
        print(f"  P(aS|pL) = {result['P_aS_pL']:.4f}  (predict Long, actual Short - DANGER!)")
        print(f"  P(aL|pS) = {result['P_aL_pS']:.4f}  (predict Short, actual Long - DANGER!)")
        print(f"  P(aN|pL) = {result['P_aN_pL']:.4f}")
        print(f"  P(aN|pS) = {result['P_aN_pS']:.4f}")

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Threshold':<12} {'Macro F1':<12} {'Accuracy':<12} {'Composite':<12} {'Neutral%':<12}")
    print("-" * 60)
    for r in all_results:
        print(
            f"{r['threshold']:<12.2f} {r['macro_f1']:<12.4f} {r['accuracy']:<12.4f} {r['composite_score']:<12.4f} {r['pred_distribution']['NEUTRAL_pct']:<12.1f}"
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
