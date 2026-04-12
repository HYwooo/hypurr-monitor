"""
LSTM+CatBoost Training
Train: 202502~202601, Valid: 202602, Test: 202603
Saves model with timestamp and full 18-probability evaluation
"""

import sys
import json
import random
import pickle
import numpy as np
import pandas as pd
import talib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from datetime import datetime
from catboost import CatBoostClassifier
from pathlib import Path

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
DOWN_LABEL, NEUTRAL_LABEL, UP_LABEL = 0, 1, 2
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
        encoded = self.encoder(x)
        return encoded, self.classifier(encoded)


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
    f["kdj_k"], f["kdj_d"] = k, d
    m, mh, mg = talib.MACD(close, 12, 26, 9)
    f["macd_dif"], f["macd_dea"], f["macd_hist"] = m, mh, mg
    e12, e26 = talib.EMA(close, 12), talib.EMA(close, 26)
    atr26 = talib.ATR(high, low, close, 26)
    mv = ((e12 - e26) / (atr26 + 1e-10)) * 100
    f["macd_v"], f["macd_v_signal"] = mv, talib.EMA(mv, 9)
    f["macd_v_hist"] = mv - f["macd_v_signal"]
    u, _, l = talib.BBANDS(close, 20)
    f["bb_position"] = (close - l) / (u - l + 1e-10)
    f["volume_ratio"] = volume / (pd.Series(volume).rolling(20).mean() + 1e-10)
    f["high_low_ratio"] = (high - low) / close
    f["close_position"] = (close - low) / (high - low + 1e-10)
    atr = talib.ATR(high, low, close, 14)
    f["atr"], f["natr"] = atr, (atr / close) * 100
    f["momentum"] = pd.Series(close).pct_change(10)
    for p in [7, 25, 50]:
        f[f"ema_ratio_{p}"] = close / talib.EMA(close, p) - 1
        f[f"sma_ratio_{p}"] = close / talib.SMA(close, p) - 1
    return f


def gen_labels(closes, highs, lows, ns=1.3, lb=6):
    atr = talib.ATR(highs, lows, closes, 14)
    fc, cc, fa = closes[lb:], closes[:-lb], atr[:-lb]
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

    # P(actual|predicted) - 9 values
    m["P_aL_pL"] = float(cm[0][0] / p0) if p0 else 0
    m["P_aN_pL"] = float(cm[1][0] / p0) if p0 else 0
    m["P_aS_pL"] = float(cm[2][0] / p0) if p0 else 0
    m["P_aL_pN"] = float(cm[0][1] / p1) if p1 else 0
    m["P_aN_pN"] = float(cm[1][1] / p1) if p1 else 0
    m["P_aS_pN"] = float(cm[2][1] / p1) if p1 else 0
    m["P_aL_pS"] = float(cm[0][2] / p2) if p2 else 0
    m["P_aN_pS"] = float(cm[1][2] / p2) if p2 else 0
    m["P_aS_pS"] = float(cm[2][2] / p2) if p2 else 0

    # P(predicted|actual) - 9 values
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
    m["long_recall"], m["short_recall"] = m["recall_up"], m["recall_down"]
    m["long_precision"], m["short_precision"] = m["precision_up"], m["precision_down"]
    long_p, short_p = m["precision_up"], m["precision_down"]
    long_r, short_r = m["recall_up"], m["recall_down"]
    f1_l, f1_s, f1_n = m["f1_up"], m["f1_down"], m["f1_neutral"]
    geo_precision = np.sqrt(long_p * short_p) if (long_p * short_p) > 0 else 0
    geo_recall = np.sqrt(long_r * short_r) if (long_r * short_r) > 0 else 0
    flip = m["P_aS_pL"] + m["P_aL_pS"]
    false_break = m["P_aL_pN"] + m["P_aS_pN"]
    m["tcs"] = (
        0.35 * geo_precision
        + 0.35 * geo_recall
        + 0.15 * f1_n
        + 0.15 * (f1_l + f1_s) / 2
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
    max_prob = np.maximum(y_proba[:, 0], np.maximum(y_proba[:, 1], y_proba[:, 2]))
    y_pred_raw = np.argmax(y_proba, axis=1).astype(int)
    y_pred = y_pred_raw.copy()
    neutral_mask = max_prob < thresh
    y_pred[neutral_mask] = NEUTRAL_LABEL
    return np.mean(y_pred == NEUTRAL_LABEL), y_pred


def find_optimal_threshold(y_proba, target_neutral_ratio):
    coarse_thresholds = [0.55, 0.60, 0.65, 0.70, 0.75]
    coarse_results = []
    for thresh in coarse_thresholds:
        neutral_ratio, _ = compute_metrics_for_thresh(y_proba, None, thresh)
        coarse_results.append(
            {"threshold": thresh, "neutral_ratio": neutral_ratio, "distance": abs(neutral_ratio - target_neutral_ratio)}
        )
    best = min(coarse_results, key=lambda x: x["distance"])
    best_thresh, best_dist = best["threshold"], best["distance"]
    if best_thresh == 0.55:
        extended = [t / 100 for t in range(50, -5, -5)]
    elif best_thresh == 0.75:
        extended = [t / 100 for t in range(80, 105, 5)]
    else:
        return best_thresh
    for thresh in extended:
        neutral_ratio, _ = compute_metrics_for_thresh(y_proba, None, thresh)
        dist = abs(neutral_ratio - target_neutral_ratio)
        if dist < best_dist:
            best_dist, best_thresh = dist, thresh
    return best_thresh


# ========== CUSTOM CONFIG ==========
TRAIN_START = 202502
TRAIN_END = 202601
VALID_MONTH = 202602
TEST_MONTH = 202603

train_months = []
year, month = 2025, 2
while (year * 100 + month) <= 202601:
    train_months.append(year * 100 + month)
    if month == 12:
        year, month = year + 1, 1
    else:
        month += 1

ts = datetime.now().strftime("%Y%m%dT%H%M%S")
MODEL_DIR = Path(f"models/train_{TRAIN_START}_{TRAIN_END}_{ts}")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print(f"Training months: {train_months[0]} ~ {train_months[-1]} ({len(train_months)} months)")
print(f"Model dir: {MODEL_DIR}")

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

valid_mask = months == VALID_MONTH
test_mask = months == TEST_MONTH
valid_idx = np.where(valid_mask)[0]
test_idx = np.where(test_mask)[0]

train_mask = np.isin(months, train_months)
train_idx = np.where(train_mask)[0]

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

best_loss, best_state, patience = float("inf"), None, 0
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
        best_loss, best_state, patience = vl, model.state_dict().copy(), 0
    else:
        patience += 1
        if patience >= 2:
            break
    print(f"  Epoch {epoch + 1}: train={tl:.4f}, valid={vl:.4f}")

if best_state:
    model.load_state_dict(best_state)

# Save LSTM model
torch.save(best_state, MODEL_DIR / "lstm_model.pt")
print(f"LSTM model saved to {MODEL_DIR / 'lstm_model.pt'}")

trF, trL = extract_features(model, trDL)
vaF, vaL = extract_features(model, vaDL)

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
cb.fit(trF, trL, eval_set=(vaF, vaY), verbose=False)

# Save CatBoost model
cb.save_model(str(MODEL_DIR / "catboost_model.cbm"))
print(f"CatBoost model saved to {MODEL_DIR / 'catboost_model.cbm'}")

# Save scaler
with open(MODEL_DIR / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print(f"Scaler saved to {MODEL_DIR / 'scaler.pkl'}")

te_idx = test_idx[(test_idx >= SEQ_LEN) & (test_idx < len(labels))]
teX = np.array([fs[i - SEQ_LEN : i] for i in te_idx])
teY = np.array([labels[i] for i in te_idx])
teDS = TensorDataset(torch.FloatTensor(teX), torch.LongTensor(teY))
teDL = DataLoader(teDS, batch_size=2048)
teF, _ = extract_features(model, teDL)

yp = cb.predict_proba(teF)
ypr_raw = np.argmax(yp, axis=1).astype(int)

optimal_thresh = find_optimal_threshold(yp, train_neutral_ratio)
print(f"\nOptimal threshold: {optimal_thresh:.2f}")

# Validation metrics
va_proba = cb.predict_proba(vaF)
va_pred_raw = np.argmax(va_proba, axis=1).astype(int)
va_max_prob = np.maximum(va_proba[:, 0], np.maximum(va_proba[:, 1], va_proba[:, 2]))
va_pred = va_pred_raw.copy()
va_neutral_mask = va_max_prob < optimal_thresh
va_pred[va_neutral_mask] = NEUTRAL_LABEL
m_valid, cm_valid = compute_metrics(vaY, va_pred, va_proba)
va_neutral_ratio = np.mean(va_pred == NEUTRAL_LABEL)

# Test metrics
max_prob = np.maximum(yp[:, 0], np.maximum(yp[:, 1], yp[:, 2]))
ypr = ypr_raw.copy()
neutral_mask = max_prob < optimal_thresh
ypr[neutral_mask] = NEUTRAL_LABEL
m, cm = compute_metrics(teY, ypr, yp)
pd_dist = pd.Series(ypr).value_counts().sort_index()

print(f"\n{'=' * 60}")
print(f"RESULTS (Train: {train_months[0]}~{train_months[-1]}, threshold={optimal_thresh:.2f})")
print(f"{'=' * 60}")
print(f"Macro F1:         {m['macro_f1']:.4f}")
print(f"TCS:              {m['tcs']:.4f}")
print(f"Neutral Ratio:    {pd_dist.get(1, 0) / len(ypr) * 100:.1f}% (target: {train_neutral_ratio * 100:.1f}%)")
print(f"Long Recall:      {m['long_recall']:.4f}")
print(f"Short Recall:     {m['short_recall']:.4f}")
print(f"Long Precision:   {m['long_precision']:.4f}")
print(f"Short Precision:  {m['short_precision']:.4f}")

# Threshold comparison
print(f"\n{'=' * 60}")
print("THRESHOLD COMPARISON")
print(f"{'=' * 60}")
print(f"{'Threshold':<12} {'Neutral%':<12} {'Macro F1':<12} {'TCS':<12}")
print("-" * 50)
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
    print(
        f"{thresh:<12.2f} {np.mean(yp_temp == NEUTRAL_LABEL) * 100:<12.1f} {m_temp['macro_f1']:<12.4f} {m_temp['tcs']:<12.4f}"
    )

# All 18 probabilities for test
test_18_probs = {
    # P(actual|predicted)
    "P_aL_pL": float(m["P_aL_pL"]),
    "P_aN_pL": float(m["P_aN_pL"]),
    "P_aS_pL": float(m["P_aS_pL"]),
    "P_aL_pN": float(m["P_aL_pN"]),
    "P_aN_pN": float(m["P_aN_pN"]),
    "P_aS_pN": float(m["P_aS_pN"]),
    "P_aL_pS": float(m["P_aL_pS"]),
    "P_aN_pS": float(m["P_aN_pS"]),
    "P_aS_pS": float(m["P_aS_pS"]),
    # P(predicted|actual)
    "P_pL_aL": float(m["P_pL_aL"]),
    "P_pN_aL": float(m["P_pN_aL"]),
    "P_pS_aL": float(m["P_pS_aL"]),
    "P_pL_aN": float(m["P_pL_aN"]),
    "P_pN_aN": float(m["P_pN_aN"]),
    "P_pS_aN": float(m["P_pS_aN"]),
    "P_pL_aS": float(m["P_pL_aS"]),
    "P_pN_aS": float(m["P_pN_aS"]),
    "P_pS_aS": float(m["P_pS_aS"]),
}

# All 18 probabilities for valid
valid_18_probs = {
    # P(actual|predicted)
    "P_aL_pL": float(m_valid["P_aL_pL"]),
    "P_aN_pL": float(m_valid["P_aN_pL"]),
    "P_aS_pL": float(m_valid["P_aS_pL"]),
    "P_aL_pN": float(m_valid["P_aL_pN"]),
    "P_aN_pN": float(m_valid["P_aN_pN"]),
    "P_aS_pN": float(m_valid["P_aS_pN"]),
    "P_aL_pS": float(m_valid["P_aL_pS"]),
    "P_aN_pS": float(m_valid["P_aN_pS"]),
    "P_aS_pS": float(m_valid["P_aS_pS"]),
    # P(predicted|actual)
    "P_pL_aL": float(m_valid["P_pL_aL"]),
    "P_pN_aL": float(m_valid["P_pN_aL"]),
    "P_pS_aL": float(m_valid["P_pS_aL"]),
    "P_pL_aN": float(m_valid["P_pL_aN"]),
    "P_pN_aN": float(m_valid["P_pN_aN"]),
    "P_pS_aN": float(m_valid["P_pS_aN"]),
    "P_pL_aS": float(m_valid["P_pL_aS"]),
    "P_pN_aS": float(m_valid["P_pN_aS"]),
    "P_pS_aS": float(m_valid["P_pS_aS"]),
}

result = {
    "config": {
        "train_start": TRAIN_START,
        "train_end": TRAIN_END,
        "valid_month": VALID_MONTH,
        "test_month": TEST_MONTH,
        "train_months": train_months,
        "n_train_months": len(train_months),
        "neutral_scale": NEUTRAL_SCALE,
        "lookforward_bars": LOOKFORWARD_BARS,
        "seq_len": SEQ_LEN,
        "hidden_dim": HIDDEN_DIM,
        "lstm_output_dim": LSTM_OUTPUT_DIM,
        "lstm_epochs": LSTM_EPOCHS,
        "lstm_batch_size": LSTM_BATCH_SIZE,
        "focal_gamma": FOCAL_GAMMA,
        "class_weights": CLASS_WEIGHTS,
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
        "accuracy": float(m_valid["accuracy"]),
        "f1_down": float(m_valid["f1_down"]),
        "f1_neutral": float(m_valid["f1_neutral"]),
        "f1_up": float(m_valid["f1_up"]),
        "support_down": int(m_valid["support_down"]),
        "support_neutral": int(m_valid["support_neutral"]),
        "support_up": int(m_valid["support_up"]),
        "confusion_matrix": cm_valid.tolist(),
        "18_probabilities": valid_18_probs,
    },
    "test_results": {
        "macro_f1": float(m["macro_f1"]),
        "tcs": float(m["tcs"]),
        "neutral_ratio": float(pd_dist.get(1, 0) / len(ypr)),
        "long_recall": float(m["long_recall"]),
        "short_recall": float(m["short_recall"]),
        "long_precision": float(m["long_precision"]),
        "short_precision": float(m["short_precision"]),
        "accuracy": float(m["accuracy"]),
        "f1_down": float(m["f1_down"]),
        "f1_neutral": float(m["f1_neutral"]),
        "f1_up": float(m["f1_up"]),
        "support_down": int(m["support_down"]),
        "support_neutral": int(m["support_neutral"]),
        "support_up": int(m["support_up"]),
        "confusion_matrix": cm.tolist(),
        "pred_distribution": {
            "DOWN": int(pd_dist.get(0, 0)),
            "NEUTRAL": int(pd_dist.get(1, 0)),
            "UP": int(pd_dist.get(2, 0)),
        },
        "18_probabilities": test_18_probs,
    },
    "threshold_comparison": threshold_comp,
    "model_files": {
        "lstm": str(MODEL_DIR / "lstm_model.pt"),
        "catboost": str(MODEL_DIR / "catboost_model.cbm"),
        "scaler": str(MODEL_DIR / "scaler.pkl"),
    },
    "timestamp": ts,
}

json_path = MODEL_DIR / f"train_results_{ts}.json"
with open(json_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nJSON saved to {json_path}")


# Markdown report
def p(v):
    return f"{v * 100:.1f}%"


def f4(v):
    return f"{v:.4f}"


def fmt18(v):
    return f"{v * 100:.1f}%"


md_lines = [
    f"# LSTM+CatBoost 训练报告",
    "",
    f"**Generated**: {ts}",
    f"**训练集**: {TRAIN_START}~{TRAIN_END} ({len(train_months)} 个月)",
    f"**验证集**: {VALID_MONTH} | **测试集**: {TEST_MONTH}",
    f"**模型目录**: {MODEL_DIR}",
    "",
    "---",
    "",
    "## 模型文件",
    "",
    f"- LSTM: `{MODEL_DIR / 'lstm_model.pt'}`",
    f"- CatBoost: `{MODEL_DIR / 'catboost_model.cbm'}`",
    f"- Scaler: `{MODEL_DIR / 'scaler.pkl'}`",
    "",
    "---",
    "",
    "## TCS 公式 (v2)",
    "",
    "`TCS = 0.35×√(Long_P×Short_P) + 0.35×√(Long_R×Short_R) + 0.15×F1_N + 0.15×(F1_L+F1_S)/2 - 1.0×Flip - 0.2×FalseBreak`",
    "",
    "---",
    "",
    "## 模型配置",
    "",
    "| 参数 | 值 |",
    "|------|-----|",
    f"| neutral_scale | {NEUTRAL_SCALE} |",
    f"| lookforward_bars | {LOOKFORWARD_BARS} |",
    f"| seq_len | {SEQ_LEN} |",
    f"| hidden_dim | {HIDDEN_DIM} |",
    f"| lstm_output_dim | {LSTM_OUTPUT_DIM} |",
    f"| focal_gamma | {FOCAL_GAMMA} |",
    f"| class_weights | {CLASS_WEIGHTS} |",
    f"| optimal_threshold | {optimal_thresh:.2f} |",
    "",
    "---",
    "",
    f"## 验证集结果 ({VALID_MONTH})",
    "",
    "| Metric | Value |",
    "|--------|-------|",
    f"| Macro F1 | {f4(m_valid['macro_f1'])} |",
    f"| TCS | {f4(m_valid['tcs'])} |",
    f"| Neutral Ratio | {p(va_neutral_ratio)} |",
    f"| Long Recall | {f4(m_valid['long_recall'])} |",
    f"| Short Recall | {f4(m_valid['short_recall'])} |",
    f"| Long Precision | {f4(m_valid['long_precision'])} |",
    f"| Short Precision | {f4(m_valid['short_precision'])} |",
    "",
    "---",
    "",
    f"## 测试集结果 ({TEST_MONTH})",
    "",
    "| Metric | Value |",
    "|--------|-------|",
    f"| **Macro F1** | **{f4(m['macro_f1'])}** |",
    f"| **TCS** | **{f4(m['tcs'])}** |",
    f"| Neutral Ratio | {p(pd_dist.get(1, 0) / len(ypr))} (target: {p(train_neutral_ratio)}) |",
    f"| Long Recall | {f4(m['long_recall'])} |",
    f"| Short Recall | {f4(m['short_recall'])} |",
    f"| Long Precision | {f4(m['long_precision'])} |",
    f"| Short Precision | {f4(m['short_precision'])} |",
    "",
    "### Confusion Matrix",
    "",
    "| Pred\\Actual | Long | Neutral | Short |",
    "|-------------|------|---------|-------|",
    f"| Long | {cm[0][0]} | {cm[1][0]} | {cm[2][0]} |",
    f"| Neutral | {cm[0][1]} | {cm[1][1]} | {cm[2][1]} |",
    f"| Short | {cm[0][2]} | {cm[1][2]} | {cm[2][2]} |",
    "",
    "### Prediction Distribution",
    "",
    "| Class | Count | Percentage |",
    "|-------|-------|------------|",
    f"| Long | {pd_dist.get(2, 0)} | {p(pd_dist.get(2, 0) / len(ypr))} |",
    f"| Neutral | {pd_dist.get(1, 0)} | {p(pd_dist.get(1, 0) / len(ypr))} |",
    f"| Short | {pd_dist.get(0, 0)} | {p(pd_dist.get(0, 0) / len(ypr))} |",
    "",
    "---",
    "",
    "## 18-Probability 详细结果 (测试集)",
    "",
    "### P(actual | predicted)",
    "",
    "| Predicted | Actual Long | Actual Neutral | Actual Short |",
    "|------------|-------------|----------------|--------------|",
    f"| Long | {fmt18(test_18_probs['P_aL_pL'])} | {fmt18(test_18_probs['P_aN_pL'])} | {fmt18(test_18_probs['P_aS_pL'])} |",
    f"| Neutral | {fmt18(test_18_probs['P_aL_pN'])} | {fmt18(test_18_probs['P_aN_pN'])} | {fmt18(test_18_probs['P_aS_pN'])} |",
    f"| Short | {fmt18(test_18_probs['P_aL_pS'])} | {fmt18(test_18_probs['P_aN_pS'])} | {fmt18(test_18_probs['P_aS_pS'])} |",
    "",
    "### P(predicted | actual)",
    "",
    "| Actual | Pred Long | Pred Neutral | Pred Short |",
    "|--------|-----------|--------------|------------|",
    f"| Long | {fmt18(test_18_probs['P_pL_aL'])} | {fmt18(test_18_probs['P_pN_aL'])} | {fmt18(test_18_probs['P_pS_aL'])} |",
    f"| Neutral | {fmt18(test_18_probs['P_pL_aN'])} | {fmt18(test_18_probs['P_pN_aN'])} | {fmt18(test_18_probs['P_pS_aN'])} |",
    f"| Short | {fmt18(test_18_probs['P_pL_aS'])} | {fmt18(test_18_probs['P_pN_aS'])} | {fmt18(test_18_probs['P_pS_aS'])} |",
    "",
    "---",
    "",
    "## 18-Probability 详细结果 (验证集)",
    "",
    "### P(actual | predicted)",
    "",
    "| Predicted | Actual Long | Actual Neutral | Actual Short |",
    "|------------|-------------|----------------|--------------|",
    f"| Long | {fmt18(valid_18_probs['P_aL_pL'])} | {fmt18(valid_18_probs['P_aN_pL'])} | {fmt18(valid_18_probs['P_aS_pL'])} |",
    f"| Neutral | {fmt18(valid_18_probs['P_aL_pN'])} | {fmt18(valid_18_probs['P_aN_pN'])} | {fmt18(valid_18_probs['P_aS_pN'])} |",
    f"| Short | {fmt18(valid_18_probs['P_aL_pS'])} | {fmt18(valid_18_probs['P_aN_pS'])} | {fmt18(valid_18_probs['P_aS_pS'])} |",
    "",
    "### P(predicted | actual)",
    "",
    "| Actual | Pred Long | Pred Neutral | Pred Short |",
    "|--------|-----------|--------------|------------|",
    f"| Long | {fmt18(valid_18_probs['P_pL_aL'])} | {fmt18(valid_18_probs['P_pN_aL'])} | {fmt18(valid_18_probs['P_pS_aL'])} |",
    f"| Neutral | {fmt18(valid_18_probs['P_pL_aN'])} | {fmt18(valid_18_probs['P_pN_aN'])} | {fmt18(valid_18_probs['P_pS_aN'])} |",
    f"| Short | {fmt18(valid_18_probs['P_pL_aS'])} | {fmt18(valid_18_probs['P_pN_aS'])} | {fmt18(valid_18_probs['P_pS_aS'])} |",
    "",
    "---",
    "",
    "## 阈值对比",
    "",
    "| Threshold | Neutral% | Macro F1 | TCS |",
    "|-----------|---------|----------|-----|",
]

for tc in threshold_comp:
    marker = " **← optimal**" if abs(tc["threshold"] - optimal_thresh) < 0.001 else ""
    md_lines.append(
        f"| {tc['threshold']:.2f}{marker} | {p(tc['neutral_ratio'])} | {f4(tc['macro_f1'])} | {f4(tc['tcs'])} |"
    )

md_path = MODEL_DIR / f"train_report_{ts}.md"
with open(md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))
print(f"Markdown report saved to {md_path}")
print(f"\nAll outputs in: {MODEL_DIR}")
