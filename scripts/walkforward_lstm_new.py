"""
Walk-Forward LSTM+CatBoost Training Script (New Folds)
- Train: 2024-01 to 2025-12 (expanding window)
- Valid: 2026-01
- Test: 2026-02
- New Composite Score formula
"""

import sys
import json
import random
from pathlib import Path
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import talib

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))

# ============ Configuration ============
NEUTRAL_SCALE = 1.3
LOOKFORWARD_BARS = 6
SEQ_LEN = 48
HIDDEN_DIM = 128
LSTM_OUTPUT_DIM = 64
FOCAL_GAMMA = 2.0
CLASS_WEIGHTS = [2.5, 1.0, 2.5]
PROBABILITY_THRESHOLD = 0.65

# LSTM pre-training
LSTM_EPOCHS = 5
LSTM_BATCH_SIZE = 2048
LSTM_LR = 0.01

# Labels
DOWN_LABEL = 0
NEUTRAL_LABEL = 1
UP_LABEL = 2

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ============ Data Loading ============
def load_data(symbol: str = "BTCUSDT", interval: str = "15m") -> pd.DataFrame:
    """Load K-line data"""
    mark_parquet = Path("data/futures/um/klines/BTCUSDT_15m_mark.parquet")
    if mark_parquet.exists():
        print(f"Loading mark price data from {mark_parquet}")
        df = pd.read_parquet(mark_parquet)
        df = df.sort_values("open_time").reset_index(drop=True)
        return df
    raise FileNotFoundError(f"Mark price data not found at {mark_parquet}")


# ============ Feature Engineering ============
def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate time-series features"""
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values

    features = pd.DataFrame(index=df.index)

    # Returns
    for period in [1, 3, 6, 12, 24]:
        features[f"return_{period}"] = pd.Series(close).pct_change(period)

    # Volatility
    features["volatility_6"] = pd.Series(close).pct_change().rolling(window=6).std()
    features["volatility_12"] = pd.Series(close).pct_change().rolling(window=12).std()
    features["volatility_24"] = pd.Series(close).pct_change().rolling(window=24).std()

    # RSI
    features["rsi"] = talib.RSI(close, 14)

    # KDJ
    k, d = talib.STOCH(high, low, close, fastk_period=9, slowk_period=3, slowd_period=3)
    features["kdj_k"] = k
    features["kdj_d"] = d

    # MACD
    macd, macd_sig, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    features["macd_dif"] = macd
    features["macd_dea"] = macd_sig
    features["macd_hist"] = macd_hist

    # MACD-V
    ema_12 = talib.EMA(close, 12)
    ema_26 = talib.EMA(close, 26)
    atr_26 = talib.ATR(high, low, close, 26)
    macd_v = ((ema_12 - ema_26) / (atr_26 + 1e-10)) * 100
    features["macd_v"] = macd_v
    features["macd_v_signal"] = talib.EMA(macd_v, 9)
    features["macd_v_hist"] = macd_v - features["macd_v_signal"]

    # Bollinger Bands
    upper, middle, lower = talib.BBANDS(close, timeperiod=20)
    bb_position = (close - lower) / (upper - lower + 1e-10)
    features["bb_position"] = bb_position

    # Volume ratio
    vol_ma = pd.Series(volume).rolling(window=20).mean() + 1e-10
    features["volume_ratio"] = volume / vol_ma

    # High-Low ratio
    features["high_low_ratio"] = (high - low) / close

    # Close position
    features["close_position"] = (close - low) / (high - low + 1e-10)

    # ATR
    atr = talib.ATR(high, low, close, 14)
    features["atr"] = atr
    features["natr"] = (atr / close) * 100

    # Momentum
    features["momentum"] = pd.Series(close).pct_change(10)

    # EMA ratio
    for period in [7, 25, 50]:
        ema = talib.EMA(close, period)
        features[f"ema_ratio_{period}"] = close / ema - 1

    # SMA ratio
    for period in [7, 25, 50]:
        sma = talib.SMA(close, period)
        features[f"sma_ratio_{period}"] = close / sma - 1

    return features


# ============ Sequence Windowing ============
class SequenceDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, seq_len: int = SEQ_LEN):
        self.features = features
        self.labels = labels
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.features) - self.seq_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        X = self.features[idx : idx + self.seq_len]
        y = self.labels[idx + self.seq_len]
        return torch.FloatTensor(X), torch.LongTensor([y])[0]


# ============ LSTM Encoder ============
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


# ============ FocalLoss ============
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: list[float] | None = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha if alpha is not None else [1.0, 1.0, 1.0])

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.CrossEntropyLoss(reduction="none", weight=self.alpha.to(logits.device))
        loss = ce_loss(logits, targets)
        probs = torch.softmax(logits, dim=1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - p_t) ** self.gamma
        loss = focal_weight * loss
        return loss.mean()


# ============ Label Generation ============
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


# ============ LSTM Training ============
def train_lstm(
    train_loader: DataLoader,
    valid_loader: DataLoader,
    n_features: int,
    device: torch.device,
    epochs: int = LSTM_EPOCHS,
    lr: float = LSTM_LR,
) -> LSTMClassifier:
    model = LSTMClassifier(n_features=n_features).to(device)
    criterion = FocalLoss(gamma=FOCAL_GAMMA, alpha=CLASS_WEIGHTS)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_loss = float("inf")
    best_state: dict | None = None
    patience_counter = 0
    max_patience = 2

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            _, logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for X, y in valid_loader:
                X, y = X.to(device), y.to(device)
                _, logits = model(X)
                loss = criterion(logits, y)
                valid_loss += loss.item()
        valid_loss /= len(valid_loader)
        scheduler.step(valid_loss)

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"  LSTM Early stopping at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def extract_lstm_features(
    model: LSTMClassifier, data_loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    features_list = []
    labels_list = []
    with torch.no_grad():
        for X, y in data_loader:
            X = X.to(device)
            features, _ = model(X)
            features_list.append(features.cpu().numpy())
            labels_list.append(y.numpy())
    return np.vstack(features_list), np.hstack(labels_list)


# ============ Walk-Forward Training ============
def walkforward_train(
    df: pd.DataFrame,
    features: pd.DataFrame,
    train_months: list[int],
    valid_month: int,
    test_month: int,
    fold_index: int = 0,
) -> dict[str, Any]:
    from catboost import CatBoostClassifier

    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    labels = generate_labels(closes, highs, lows)
    labels = labels[: len(features)]

    open_times = df["open_time"].values[: len(features)]
    dates = pd.to_datetime(open_times, unit="ms")
    months = dates.year * 100 + dates.month

    feature_values = features.values
    scaler = StandardScaler()
    feature_scaled = scaler.fit_transform(feature_values)

    train_mask = np.isin(months, train_months)
    valid_mask = months == valid_month
    test_mask = months == test_month

    train_indices = np.where(train_mask)[0]
    valid_indices = np.where(valid_mask)[0]
    test_indices = np.where(test_mask)[0]

    if len(train_indices) < SEQ_LEN or len(valid_indices) < SEQ_LEN or len(test_indices) < SEQ_LEN:
        raise ValueError(
            f"Insufficient data: train={len(train_indices)}, valid={len(valid_indices)}, test={len(test_indices)}"
        )

    print(f"  Train: {len(train_indices)} samples ({len(train_months)} months)")
    print(f"  Valid: {len(valid_indices)} samples ({valid_month})")
    print(f"  Test:  {len(test_indices)} samples ({test_month})")

    # Phase 1: LSTM Pre-training
    print("  Phase 1: LSTM Pre-training...")

    train_seq_indices = train_indices[train_indices >= SEQ_LEN]
    train_X = []
    train_y = []
    for idx in train_seq_indices:
        start = idx - SEQ_LEN
        train_X.append(feature_scaled[start:idx])
        train_y.append(labels[idx])
    train_X = np.array(train_X)
    train_y = np.array(train_y)

    valid_seq_indices = valid_indices[valid_indices >= SEQ_LEN]
    valid_X = []
    valid_y = []
    for idx in valid_seq_indices:
        start = idx - SEQ_LEN
        valid_X.append(feature_scaled[start:idx])
        valid_y.append(labels[idx])
    valid_X = np.array(valid_X)
    valid_y = np.array(valid_y)

    print(f"    LSTM train sequences: {len(train_X)}, valid sequences: {len(valid_X)}")

    train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(train_X), torch.LongTensor(train_y))
    valid_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(valid_X), torch.LongTensor(valid_y))
    train_loader = DataLoader(train_dataset, batch_size=LSTM_BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=LSTM_BATCH_SIZE, shuffle=False, num_workers=0)

    n_features = train_X.shape[2]
    lstm_model = train_lstm(train_loader, valid_loader, n_features, DEVICE)

    # Phase 2: CatBoost Training
    print("  Phase 2: CatBoost Training...")

    train_features, train_labels_extracted = extract_lstm_features(lstm_model, train_loader, DEVICE)
    valid_features, valid_labels_extracted = extract_lstm_features(lstm_model, valid_loader, DEVICE)

    print(f"    Train features: {train_features.shape}, Valid features: {valid_features.shape}")

    for name, lbls in [("Train", train_labels_extracted), ("Valid", valid_labels_extracted)]:
        counts = pd.Series(lbls).value_counts().sort_index()
        print(f"    {name} labels: DOWN={counts.get(0, 0)}, NEUTRAL={counts.get(1, 0)}, UP={counts.get(2, 0)}")

    cb_model = CatBoostClassifier(
        iterations=300,
        learning_rate=0.08,
        depth=6,
        l2_leaf_reg=3.0,
        loss_function="MultiClass",
        class_weights=CLASS_WEIGHTS,
        early_stopping_rounds=30,
        verbose=False,
    )
    cb_model.fit(
        train_features, train_labels_extracted, eval_set=(valid_features, valid_labels_extracted), verbose=False
    )

    # Save models
    model_dir = Path(f"models/lstm_catboost/fold_{fold_index}")
    model_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": lstm_model.state_dict(),
            "n_features": n_features,
            "hidden_dim": HIDDEN_DIM,
            "lstm_output_dim": LSTM_OUTPUT_DIM,
            "seq_len": SEQ_LEN,
            "config": {
                "neutral_scale": NEUTRAL_SCALE,
                "lookforward_bars": LOOKFORWARD_BARS,
                "focal_gamma": FOCAL_GAMMA,
                "class_weights": CLASS_WEIGHTS,
            },
        },
        model_dir / "lstm_model.pt",
    )
    cb_model.save_model(str(model_dir / "catboost_model.cbm"))
    import joblib

    joblib.dump(scaler, model_dir / "scaler.joblib")
    print(f"  Models saved to {model_dir}")

    # Phase 3: Test Evaluation
    print("  Phase 3: Test Evaluation...")

    test_seq_indices = test_indices[test_indices >= SEQ_LEN]
    test_X = []
    test_y = []
    for idx in test_seq_indices:
        start = idx - SEQ_LEN
        test_X.append(feature_scaled[start:idx])
        test_y.append(labels[idx])

    test_X = np.array(test_X)
    test_y = np.array(test_y)

    if len(test_X) == 0:
        raise ValueError("No test sequences")

    test_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(test_X), torch.LongTensor(test_y))
    test_loader = DataLoader(test_dataset, batch_size=LSTM_BATCH_SIZE, shuffle=False, num_workers=0)

    test_features, _ = extract_lstm_features(lstm_model, test_loader, DEVICE)

    y_pred_raw = cb_model.predict(test_features).flatten()
    y_proba_raw = cb_model.predict_proba(test_features)

    # Post-processing
    y_proba_post = y_proba_raw.copy()
    max_prob = np.maximum(
        y_proba_post[:, DOWN_LABEL], np.maximum(y_proba_post[:, NEUTRAL_LABEL], y_proba_post[:, UP_LABEL])
    )
    neutral_mask = max_prob < PROBABILITY_THRESHOLD
    y_pred_post = y_pred_raw.copy()
    y_pred_post[neutral_mask] = NEUTRAL_LABEL

    metrics_raw = compute_metrics(test_y, y_pred_raw, y_proba_raw)
    metrics_post = compute_metrics(test_y, y_pred_post, y_proba_post)

    cm_raw = confusion_matrix(test_y, y_pred_raw, labels=[DOWN_LABEL, NEUTRAL_LABEL, UP_LABEL])
    cm_post = confusion_matrix(test_y, y_pred_post, labels=[DOWN_LABEL, NEUTRAL_LABEL, UP_LABEL])

    result = {
        "train_months": train_months,
        "valid_month": valid_month,
        "test_month": test_month,
        "train_size": len(train_features),
        "valid_size": len(valid_features),
        "test_size": len(test_features),
        **metrics_raw,
        **{f"{k}_post": v for k, v in metrics_post.items()},
        "confusion_matrix_raw": cm_raw.tolist(),
        "confusion_matrix_post": cm_post.tolist(),
    }

    return result


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, Any]:
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

    # ========== NEW Composite Score Formula ==========
    # Composite = Macro_F1 × √(Long_P × Short_P) × (1 + 0.5 × Dir_Quality)
    #           - 8.0 × (P_aS_pL + P_aL_pS)
    #           - 3.0 × (P_aL_pN + P_aS_pN)

    long_prec = metrics["precision_up"]  # P_aS_pS
    short_prec = metrics["precision_down"]  # P_aL_pL
    dir_quality = (long_prec + short_prec) / 2

    geo_mean = np.sqrt(long_prec * short_prec)
    dir_boost = 1.0 + 0.5 * dir_quality

    metrics["composite_score"] = (
        metrics["macro_f1"] * geo_mean * dir_boost
        - 8.0 * (metrics["P_aS_pL"] + metrics["P_aL_pS"])
        - 3.0 * (metrics["P_aL_pN"] + metrics["P_aS_pN"])
    )

    return metrics


# ============ Main ============
def main():
    print("=" * 80)
    print("Walk-Forward LSTM+CatBoost Training (New Folds)")
    print("=" * 80)
    print(f"\nConfig:")
    print(f"  neutral_scale: {NEUTRAL_SCALE}")
    print(f"  lookforward_bars: {LOOKFORWARD_BARS}")
    print(f"  seq_len: {SEQ_LEN}")
    print(f"  probability_threshold: {PROBABILITY_THRESHOLD}")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    print("\nLoading data...")
    df = load_data()
    print(f"Data: {len(df)} rows")
    print(
        f"Time range: {datetime.fromtimestamp(df['open_time'].min() / 1000)} to {datetime.fromtimestamp(df['open_time'].max() / 1000)}"
    )

    print("\nGenerating features...")
    features = generate_features(df)
    features = features.dropna()
    print(f"Features: {features.shape[1]} features")

    min_len = min(len(df), len(features))
    df = df.iloc[:min_len].reset_index(drop=True)
    features = features.iloc[:min_len].reset_index(drop=True)

    # Define fold configurations
    # Train: 2024-01 to 2025-12 (expanding window)
    # Valid: 2026-01
    # Test: 2026-02
    all_months = []
    current = datetime(2024, 1, 1)
    end = datetime(2025, 12, 1)
    while current <= end:
        all_months.append(current.year * 100 + current.month)
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)

    # Split into 4 folds (Q1, H1, 3Q, Full)
    fold_configs = [
        # Fold 0: Train 2024-Q1 (3 months), Valid 2601, Test 2602
        ([202401, 202402, 202403], 202601, 202602, 0),
        # Fold 1: Train 2024-H1 (6 months), Valid 2601, Test 2602
        ([202401, 202402, 202403, 202404, 202405, 202406], 202601, 202602, 1),
        # Fold 2: Train 2024-9 months, Valid 2601, Test 2602
        ([202401, 202402, 202403, 202404, 202405, 202406, 202407, 202408, 202409], 202601, 202602, 2),
        # Fold 3: Train 2024-Full (12 months), Valid 2601, Test 2602
        (
            [202401, 202402, 202403, 202404, 202405, 202406, 202407, 202408, 202409, 202410, 202411, 202412],
            202601,
            202602,
            3,
        ),
    ]

    all_results = []

    for train_months, valid_month, test_month, fold_idx in fold_configs:
        print(f"\n{'=' * 60}")
        print(
            f"Fold {fold_idx}: Train {train_months[0]}-{train_months[-1]} ({len(train_months)} months), Valid {valid_month}, Test {test_month}"
        )
        print("=" * 60)
        try:
            result = walkforward_train(df, features, train_months, valid_month, test_month, fold_idx)
            all_results.append(result)
            print(f"\n  Macro F1: {result['macro_f1']:.4f}")
            print(f"  Composite Score: {result['composite_score']:.4f}")
        except Exception as e:
            print(f"Fold {fold_idx} failed: {e}")

    if not all_results:
        print("No valid results")
        return

    # Print summary
    print("\n" + "=" * 100)
    print("RESULTS SUMMARY")
    print("=" * 100)
    print(f"\n{'Fold':<6} {'Train Months':<15} {'Macro F1':<12} {'Composite':<12} {'LongRec':<10} {'ShortRec':<10}")
    print("-" * 70)
    for i, r in enumerate(all_results):
        print(
            f"{i:<6} {r['train_months'][0]}-{r['train_months'][-1]:<15} {r['macro_f1']:<12.4f} {r['composite_score']:<12.4f} {r['long_recall']:<10.4f} {r['short_recall']:<10.4f}"
        )

    # Average
    avg_macro = np.mean([r["macro_f1"] for r in all_results])
    avg_comp = np.mean([r["composite_score"] for r in all_results])
    avg_long_r = np.mean([r["long_recall"] for r in all_results])
    avg_short_r = np.mean([r["short_recall"] for r in all_results])
    print("-" * 70)
    print(f"{'AVG':<6} {'':<15} {avg_macro:<12.4f} {avg_comp:<12.4f} {avg_long_r:<10.4f} {avg_short_r:<10.4f}")

    # Save results
    output_path = Path("models/walkforward_lstm_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def convert_to_native(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(i) for i in obj]
        return obj

    with open(output_path, "w") as f:
        json.dump(
            {
                "config": {
                    "neutral_scale": NEUTRAL_SCALE,
                    "lookforward_bars": LOOKFORWARD_BARS,
                    "seq_len": SEQ_LEN,
                    "hidden_dim": HIDDEN_DIM,
                    "lstm_output_dim": LSTM_OUTPUT_DIM,
                    "focal_gamma": FOCAL_GAMMA,
                    "class_weights": CLASS_WEIGHTS,
                    "probability_threshold": PROBABILITY_THRESHOLD,
                },
                "results": convert_to_native(all_results),
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to {output_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
