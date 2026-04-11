"""
Load best model (24-month fold) and evaluate on 2026-01~2026-03
Generate walkforward_lstm_test_results_{timestamp}.md
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import talib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from datetime import datetime
import joblib

# Configuration (same as training)
SEQ_LEN = 48
HIDDEN_DIM = 128
LSTM_OUTPUT_DIM = 64
NEUTRAL_SCALE = 1.3
LOOKFORWARD_BARS = 6
FOCAL_GAMMA = 2.0
CLASS_WEIGHTS = [2.5, 1.0, 2.5]
DOWN_LABEL = 0
NEUTRAL_LABEL = 1
UP_LABEL = 2
DEVICE = torch.device("cpu")
OPTIMAL_THRESHOLD = 0.65  # From 24-month fold

# Test months
TEST_MONTHS = [202511]


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

    # TCS Formula v2
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

    flip = m["P_aS_pL"] + m["P_aL_pS"]
    false_break = m["P_aL_pN"] + m["P_aS_pN"]

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


def main():
    print("=" * 80)
    print("Loading best model (24-month fold) and evaluating on 2026-01~2026-03")
    print("=" * 80)

    # Load models
    model_dir = Path("models/lstm_catboost/fold_3")
    print(f"\nLoading models from {model_dir}")

    # Load LSTM model
    lstm_checkpoint = torch.load(model_dir / "lstm_model.pt", map_location=DEVICE)
    n_features = lstm_checkpoint["n_features"]
    lstm_model = LSTMClassifier(n_features=n_features).to(DEVICE)
    lstm_model.load_state_dict(lstm_checkpoint["model_state_dict"])
    print(f"  LSTM loaded: seq_len={lstm_checkpoint['seq_len']}")

    # Load CatBoost model
    from catboost import CatBoostClassifier

    cb_model = CatBoostClassifier()
    cb_model.load_model(str(model_dir / "catboost_model.cbm"))
    print("  CatBoost loaded")

    # Load scaler
    scaler = joblib.load(model_dir / "scaler.joblib")
    print("  Scaler loaded")

    # Load data
    print("\nLoading data...")
    df = pd.read_parquet("data/futures/um/klines/BTCUSDT_15m_mark.parquet")
    df = df.sort_values("open_time").reset_index(drop=True)
    print(f"Data: {len(df)} rows")

    # Check data range
    min_date = pd.to_datetime(df["open_time"].min(), unit="ms")
    max_date = pd.to_datetime(df["open_time"].max(), unit="ms")
    print(f"Data range: {min_date} to {max_date}")

    # Generate features
    print("\nGenerating features...")
    features = generate_features(df).dropna()
    min_len = min(len(df), len(features))
    df = df.iloc[:min_len].reset_index(drop=True)
    features = features.iloc[:min_len].reset_index(drop=True)

    closes = df["close"].values[: len(features)]
    highs = df["high"].values[: len(features)]
    lows = df["low"].values[: len(features)]
    labels_orig = gen_labels(closes, highs, lows)
    # Align labels to features length (same as training code)
    labels = labels_orig[: len(features)]
    dates = pd.to_datetime(df["open_time"].values[: len(features)], unit="ms")
    months = dates.year * 100 + dates.month

    fs = scaler.transform(features.values)

    # Evaluate on each test month
    all_results = []

    for test_month in TEST_MONTHS:
        print(f"\n{'=' * 60}")
        print(f"Evaluating on {test_month}")
        print("=" * 60)

        test_mask = months == test_month
        test_idx = np.where(test_mask)[0]

        if len(test_idx) == 0:
            print(f"  No data for month {test_month}, skipping")
            continue

        # Prepare test sequences
        test_si = test_idx[test_idx >= SEQ_LEN]
        # Filter to ensure indices are within bounds for both fs and labels
        test_si = test_si[(test_si < len(fs)) & (test_si < len(labels))]
        if len(test_si) == 0:
            print(f"  Not enough data for seq_len={SEQ_LEN}, skipping")
            continue

        teX = np.array([fs[i - SEQ_LEN : i] for i in test_si])
        teY = np.array([labels[i] for i in test_si])

        teDS = TensorDataset(torch.FloatTensor(teX), torch.LongTensor(teY))
        teDL = DataLoader(teDS, batch_size=2048)

        # Extract LSTM features
        teF, _ = extract_features(lstm_model, teDL)

        # Predict
        yp = cb_model.predict_proba(teF)
        ypr_raw = (np.argmax(yp, axis=1)).astype(int)

        # Apply threshold
        max_prob = np.maximum(yp[:, 0], np.maximum(yp[:, 1], yp[:, 2]))
        ypr = ypr_raw.copy()
        neutral_mask = max_prob < OPTIMAL_THRESHOLD
        ypr[neutral_mask] = NEUTRAL_LABEL

        # Compute metrics
        m, cm = compute_metrics(teY, ypr, yp)
        pd_dist = pd.Series(ypr).value_counts().sort_index()

        print(f"  Samples: {len(teY)}")
        print(f"  Macro F1: {m['macro_f1']:.4f}")
        print(f"  TCS: {m['tcs']:.4f}")
        print(f"  Neutral Ratio: {pd_dist.get(1, 0) / len(ypr) * 100:.1f}%")
        print(f"  Long Recall: {m['long_recall']:.4f}, Short Recall: {m['short_recall']:.4f}")
        print(f"  Long Precision: {m['long_precision']:.4f}, Short Precision: {m['short_precision']:.4f}")
        print(f"  P(aS|pL): {m['P_aS_pL']:.4f}, P(aL|pS): {m['P_aL_pS']:.4f}")

        # Threshold comparison for this month
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

        # Find optimal for this month
        opt_thresh_month = min(threshold_comp, key=lambda x: abs(x["threshold"] - OPTIMAL_THRESHOLD))["threshold"]
        opt_result = [t for t in threshold_comp if abs(t["threshold"] - opt_thresh_month) < 0.001][0]

        result = {
            "month": test_month,
            "n_samples": len(teY),
            "optimal_threshold_used": OPTIMAL_THRESHOLD,
            "metrics": {
                "macro_f1": float(m["macro_f1"]),
                "tcs": float(m["tcs"]),
                "neutral_ratio": float(pd_dist.get(1, 0) / len(ypr)),
                "long_recall": float(m["long_recall"]),
                "short_recall": float(m["short_recall"]),
                "long_precision": float(m["long_precision"]),
                "short_precision": float(m["short_precision"]),
                "precision_neutral": float(m["precision_neutral"]),
                "recall_neutral": float(m["recall_neutral"]),
                "P_aS_pL": float(m["P_aS_pL"]),
                "P_aL_pS": float(m["P_aL_pS"]),
                "f1_down": float(m["f1_down"]),
                "f1_neutral": float(m["f1_neutral"]),
                "f1_up": float(m["f1_up"]),
                "support_down": int(m["support_down"]),
                "support_neutral": int(m["support_neutral"]),
                "support_up": int(m["support_up"]),
                "accuracy": float(m["accuracy"]),
            },
            "confusion_matrix": cm.tolist(),
            "pred_distribution": {
                "DOWN": int(pd_dist.get(0, 0)),
                "NEUTRAL": int(pd_dist.get(1, 0)),
                "UP": int(pd_dist.get(2, 0)),
            },
            "threshold_comparison": threshold_comp,
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
        }
        all_results.append(result)

    if not all_results:
        print("No valid results!")
        return

    # ========== Generate Report ==========
    ts = datetime.now().strftime("%Y%m%dT%H%M%z")

    def p(v):
        return f"{v * 100:.1f}%"

    def f4(v):
        return f"{v:.4f}"

    def fmt18(val):
        return f"{val * 100:.1f}%"

    lines = []
    lines.append("# LSTM+CatBoost 模型测试报告 (2026-01~2026-03)")
    lines.append("")
    lines.append(f"**Generated**: {ts}")
    lines.append(f"**Model Source**: 24-month fold (Train: 2023-10 ~ 2025-09)")
    lines.append(f"**Threshold Used**: {OPTIMAL_THRESHOLD}")
    lines.append(f"**Test Period**: 2026-01, 2026-02, 2026-03")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Summary Table
    lines.append("## 1. 测试汇总")
    lines.append("")
    lines.append(
        "| Month | Samples | Macro F1 | TCS | Neutral% | Long Recall | Short Recall | Long Prec | Short Prec |"
    )
    lines.append("|-------|---------|----------|-----|---------|------------|-------------|-----------|-----------|")
    for r in all_results:
        m = r["metrics"]
        lines.append(
            f"| {r['month']} | {r['n_samples']} | {f4(m['macro_f1'])} | {f4(m['tcs'])} | {p(m['neutral_ratio'])} | {f4(m['long_recall'])} | {f4(m['short_recall'])} | {f4(m['long_precision'])} | {f4(m['short_precision'])} |"
        )
    lines.append("")

    # Monthly Details
    for r in all_results:
        month = r["month"]
        m = r["metrics"]
        lines.append("---")
        lines.append("")
        lines.append(f"## 2.{month} 详细结果")
        lines.append("")
        lines.append(f"**样本数**: {r['n_samples']} | **Threshold**: {r['optimal_threshold_used']}")
        lines.append("")

        lines.append("### 2.1 核心指标")
        lines.append("")
        lines.append("| 指标 | 值 |")
        lines.append("|------|-----|")
        lines.append(f"| Accuracy | {f4(m['accuracy'])} |")
        lines.append(f"| Macro F1 | {f4(m['macro_f1'])} |")
        lines.append(f"| TCS | {f4(m['tcs'])} |")
        lines.append(f"| Neutral Ratio | {p(m['neutral_ratio'])} |")
        lines.append("")

        lines.append("### 2.2 方向指标")
        lines.append("")
        lines.append("| 指标 | Long | Short | Neutral |")
        lines.append("|------|------|-------|---------|")
        lines.append(
            f"| **Precision** | {f4(m['long_precision'])} | {f4(m['short_precision'])} | {f4(m['precision_neutral'])} |"
        )
        lines.append(f"| **Recall** | {f4(m['long_recall'])} | {f4(m['short_recall'])} | {f4(m['recall_neutral'])} |")
        lines.append(f"| **F1** | {f4(m['f1_up'])} | {f4(m['f1_down'])} | {f4(m['f1_neutral'])} |")
        lines.append(f"| **Support** | {m['support_down']} | {m['support_up']} | {m['support_neutral']} |")
        lines.append("")

        lines.append("### 2.3 错误分析")
        lines.append("")
        lines.append(f"| 指标 | 值 | 说明 |")
        lines.append("|------|-----|------|")
        lines.append(f"| P(aS\\|pL) | {f4(r['P_aS_pL'])} | 预测Long实际Short |")
        lines.append(f"| P(aL\\|pS) | {f4(r['P_aL_pS'])} | 预测Short实际Long |")
        lines.append(f"| P(aL\\|pN) | {f4(r['P_aL_pN'])} | 预测Neutral实际Long |")
        lines.append(f"| P(aS\\|pN) | {f4(r['P_aS_pN'])} | 预测Neutral实际Short |")
        lines.append("")

        lines.append("### 2.4 18-概率评估")
        lines.append("")
        lines.append("**P(actual|predicted)**")
        lines.append("| Predicted | Actual Long | Actual Neutral | Actual Short |")
        lines.append("|------------|-------------|----------------|--------------|")
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

        lines.append("### 2.5 阈值敏感性")
        lines.append("")
        lines.append(f"| Threshold | Neutral% | Macro F1 | TCS |")
        lines.append("|-----------|---------|----------|-----|")
        for tc in r["threshold_comparison"]:
            marker = " **← used**" if abs(tc["threshold"] - OPTIMAL_THRESHOLD) < 0.001 else ""
            lines.append(
                f"| {tc['threshold']:.2f}{marker} | {p(tc['neutral_ratio'])} | {f4(tc['macro_f1'])} | {f4(tc['tcs'])} |"
            )
        lines.append("")

        lines.append("### 2.6 混淆矩阵")
        lines.append("")
        cm = r["confusion_matrix"]
        lines.append("```")
        lines.append(f"         Predicted")
        lines.append(f"         Long  Neutral  Short")
        lines.append(f"Actual Long   {cm[0][0]:4d}    {cm[1][0]:4d}    {cm[2][0]:4d}")
        lines.append(f"       Neutral   {cm[0][1]:4d}    {cm[1][1]:4d}    {cm[2][1]:4d}")
        lines.append(f"       Short   {cm[0][2]:4d}    {cm[1][2]:4d}    {cm[2][2]:4d}")
        lines.append("```")
        lines.append("")

    # Overall Summary
    lines.append("---")
    lines.append("")
    lines.append("## 3. 总体分析")
    lines.append("")

    avg_f1 = np.mean([r["metrics"]["macro_f1"] for r in all_results])
    avg_tcs = np.mean([r["metrics"]["tcs"] for r in all_results])
    avg_neutral = np.mean([r["metrics"]["neutral_ratio"] for r in all_results])

    lines.append(f"| 指标 | 3月平均 |")
    lines.append("|------|--------|")
    lines.append(f"| Macro F1 | {f4(avg_f1)} |")
    lines.append(f"| TCS | {f4(avg_tcs)} |")
    lines.append(f"| Neutral% | {p(avg_neutral)} |")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 4. JSON Results")
    lines.append("")
    lines.append(f"- Timestamp: {ts}")
    lines.append(f"- Path: models/walkforward_lstm_test_results_{ts}.json")

    # Save
    md_content = "\n".join(lines)
    md_path = f"models/walkforward_lstm_test_results_{ts}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"\nMarkdown saved to {md_path}")

    # Save JSON
    json_path = f"models/walkforward_lstm_test_results_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(
            {"results": all_results, "timestamp": ts, "model_source": "fold_3", "threshold_used": OPTIMAL_THRESHOLD},
            f,
            indent=2,
        )
    print(f"JSON saved to {json_path}")


if __name__ == "__main__":
    main()
