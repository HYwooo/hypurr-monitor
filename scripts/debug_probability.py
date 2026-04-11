"""
Debug: Check probability distribution for Fold 1 test set
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import talib
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

SEQ_LEN = 48
HIDDEN_DIM = 128
LSTM_OUTPUT_DIM = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DOWN_LABEL = 0
NEUTRAL_LABEL = 1
UP_LABEL = 2


class LSTMEncoder(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int = HIDDEN_DIM, lstm_output_dim: int = LSTM_OUTPUT_DIM):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, lstm_output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])


class LSTMClassifier(nn.Module):
    def __init__(
        self, n_features: int, hidden_dim: int = HIDDEN_DIM, lstm_output_dim: int = LSTM_OUTPUT_DIM, n_classes: int = 3
    ):
        super().__init__()
        self.encoder = LSTMEncoder(n_features, hidden_dim, lstm_output_dim)
        self.classifier = nn.Linear(lstm_output_dim, n_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x), self.classifier(self.encoder(x))


def load_data():
    df = pd.read_parquet("data/futures/um/klines/BTCUSDT_15m_mark.parquet")
    return df.sort_values("open_time").reset_index(drop=True)


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


def main():
    print("Loading data...")
    df = load_data()
    features = generate_features(df).dropna()
    min_len = min(len(df), len(features))
    df = df.iloc[:min_len].reset_index(drop=True)
    features = features.iloc[:min_len].reset_index(drop=True)

    # Load models
    model_dir = Path("models/lstm_catboost/fold_1")
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

    # Prepare test data (202510)
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    labels = generate_labels(closes, highs, lows)[: len(features)]

    dates = pd.to_datetime(df["open_time"].values[: len(features)], unit="ms")
    months = dates.year * 100 + dates.month

    feature_scaled = scaler.transform(features.values)

    test_mask = months == 202510
    test_indices = np.where(test_mask)[0]
    test_seq_indices = test_indices[test_indices >= SEQ_LEN]

    test_X = np.array([feature_scaled[i - SEQ_LEN : i] for i in test_seq_indices])
    test_y = np.array([labels[i] for i in test_seq_indices])

    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(test_X), torch.LongTensor(test_y)), batch_size=2048, shuffle=False
    )

    # Get probabilities
    lstm_model.eval()
    features_list = []
    with torch.no_grad():
        for X, _ in test_loader:
            feat, _ = lstm_model(X.to(DEVICE))
            features_list.append(feat.cpu().numpy())
    lstm_features = np.vstack(features_list)

    y_proba = cb_model.predict_proba(lstm_features)
    max_prob = np.maximum(y_proba[:, 0], np.maximum(y_proba[:, 1], y_proba[:, 2]))

    print(f"\nProbability Distribution (max of 3 classes):")
    print(f"  Min:    {max_prob.min():.4f}")
    print(f"  25%:    {np.percentile(max_prob, 25):.4f}")
    print(f"  50%:    {np.percentile(max_prob, 50):.4f}")
    print(f"  75%:    {np.percentile(max_prob, 75):.4f}")
    print(f"  Max:    {max_prob.max():.4f}")
    print(f"  Mean:   {max_prob.mean():.4f}")

    print(f"\nSamples below thresholds:")
    for t in [0.12, 0.16, 0.18, 0.20, 0.22, 0.25, 0.30]:
        count = np.sum(max_prob < t)
        print(f"  < {t}: {count} ({count / len(max_prob) * 100:.1f}%)")

    print(f"\nClass probability distributions:")
    print(
        f"  P(DOWN):    min={y_proba[:, 0].min():.3f}, max={y_proba[:, 0].max():.3f}, mean={y_proba[:, 0].mean():.3f}"
    )
    print(
        f"  P(NEUTRAL): min={y_proba[:, 1].min():.3f}, max={y_proba[:, 1].max():.3f}, mean={y_proba[:, 1].mean():.3f}"
    )
    print(
        f"  P(UP):      min={y_proba[:, 2].min():.3f}, max={y_proba[:, 2].max():.3f}, mean={y_proba[:, 2].mean():.3f}"
    )


if __name__ == "__main__":
    main()
