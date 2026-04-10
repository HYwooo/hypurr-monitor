"""
LSTM+CatBoost 推理脚本

加载已保存的模型，对新数据进行预测

Usage:
    uv run python scripts/inference_lstm.py --model models/lstm_catboost/fold_1 --data data/futures/um/klines/BTCUSDT_15m_mark.parquet
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import talib
import joblib
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import Kline
from ml.labels.three_class import ThreeClassLabeler


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMEncoder(nn.Module):
    """LSTM 编码器"""

    def __init__(self, n_features: int, hidden_dim: int = 128, lstm_output_dim: int = 64):
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
        output = self.fc(last_hidden)
        return output


class LSTMClassifier(nn.Module):
    """LSTM 分类器"""

    def __init__(self, n_features: int, hidden_dim: int = 128, lstm_output_dim: int = 64, n_classes: int = 3):
        super().__init__()

        self.encoder = LSTMEncoder(n_features, hidden_dim, lstm_output_dim)
        self.classifier = nn.Linear(lstm_output_dim, n_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        logits = self.classifier(features)
        return features, logits


def load_model(model_dir: Path) -> tuple[LSTMClassifier, Any, dict]:
    """加载 LSTM 模型、CatBoost 模型和 Scaler"""
    # 加载 LSTM
    lstm_checkpoint = torch.load(model_dir / "lstm_model.pt", map_location=DEVICE)
    n_features = lstm_checkpoint["n_features"]
    hidden_dim = lstm_checkpoint["hidden_dim"]
    lstm_output_dim = lstm_checkpoint["lstm_output_dim"]

    lstm_model = LSTMClassifier(n_features=n_features, hidden_dim=hidden_dim, lstm_output_dim=lstm_output_dim)
    lstm_model.load_state_dict(lstm_checkpoint["model_state_dict"])
    lstm_model.to(DEVICE)
    lstm_model.eval()

    # 加载 CatBoost
    from catboost import CatBoostClassifier

    cb_model = CatBoostClassifier()
    cb_model.load_model(str(model_dir / "catboost_model.cbm"))

    # 加载 Scaler
    scaler = joblib.load(model_dir / "scaler.joblib")

    config = lstm_checkpoint["config"]

    return lstm_model, cb_model, scaler, config


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """生成特征"""
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values

    features = pd.DataFrame(index=df.index)

    # 收益率
    for period in [1, 3, 6, 12, 24]:
        features[f"return_{period}"] = pd.Series(close).pct_change(period)

    # 波动率
    features["volatility_6"] = pd.Series(close).pct_change().rolling(window=6).std()
    features["volatility_12"] = pd.Series(close).pct_change().rolling(window=12).std()
    features["volatility_24"] = pd.Series(close).pct_change().rolling(window=24).std()

    # RSI
    features["rsi"] = talib.RSI(close, 14)

    # KDJ
    k, d = talib.STOCH(high, low, close, fastk_period=9, slowk_period=3, slowd_period=3)
    features["kdj_k"] = k
    features["kdj_d"] = d

    # MACD (标准)
    macd, macd_sig, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    features["macd_dif"] = macd
    features["macd_dea"] = macd_sig
    features["macd_hist"] = macd_hist

    # MACD-V = [(12-EMA - 26-EMA) / ATR(26)] * 100
    ema_12 = talib.EMA(close, 12)
    ema_26 = talib.EMA(close, 26)
    atr_26 = talib.ATR(high, low, close, 26)
    macd_v = ((ema_12 - ema_26) / (atr_26 + 1e-10)) * 100
    features["macd_v"] = macd_v

    # Signal line = 9-period EMA of MACD-V
    macd_v_signal = talib.EMA(macd_v, 9)
    features["macd_v_signal"] = macd_v_signal

    # Histogram = MACD-V - Signal Line
    features["macd_v_hist"] = macd_v - macd_v_signal

    # 布林带位置
    upper, middle, lower = talib.BBANDS(close, timeperiod=20)
    bb_position = (close - lower) / (upper - lower + 1e-10)
    features["bb_position"] = bb_position

    # 成交量比率
    vol_ma = pd.Series(volume).rolling(window=20).mean() + 1e-10
    features["volume_ratio"] = volume / vol_ma

    # 高低价比率
    features["high_low_ratio"] = (high - low) / close

    # 收盘位置
    features["close_position"] = (close - low) / (high - low + 1e-10)

    # ATR
    atr = talib.ATR(high, low, close, 14)
    features["atr"] = atr
    features["natr"] = (atr / close) * 100

    # 动量
    features["momentum"] = pd.Series(close).pct_change(10)

    # EMA 比率
    for period in [7, 25, 50]:
        ema = talib.EMA(close, period)
        features[f"ema_ratio_{period}"] = close / ema - 1

    # SMA 比率
    for period in [7, 25, 50]:
        sma = talib.SMA(close, period)
        features[f"sma_ratio_{period}"] = close / sma - 1

    return features


def predict(
    lstm_model: LSTMClassifier,
    cb_model: Any,
    scaler: Any,
    features: np.ndarray,
    seq_len: int = 96,
    probability_threshold: float = 0.12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    批量预测

    Args:
        lstm_model: LSTM 模型
        cb_model: CatBoost 模型
        scaler: StandardScaler
        features: 特征数组 (n_samples, n_features)
        seq_len: 序列长度
        probability_threshold: 概率阈值

    Returns:
        (predictions, probabilities)
    """
    # 标准化
    features_scaled = scaler.transform(features)

    # 构建序列
    n_samples = len(features_scaled) - seq_len
    if n_samples <= 0:
        raise ValueError(f"特征数量不足: {len(features_scaled)}, 需要至少 {seq_len}")

    sequences = []
    for i in range(seq_len, len(features_scaled)):
        sequences.append(features_scaled[i - seq_len : i])

    X = torch.FloatTensor(np.array(sequences)).to(DEVICE)

    # LSTM 特征提取
    with torch.no_grad():
        lstm_features, _ = lstm_model(X)
        lstm_features = lstm_features.cpu().numpy()

    # CatBoost 预测
    y_proba = cb_model.predict_proba(lstm_features)
    y_pred = cb_model.predict(lstm_features).flatten()

    # 后处理
    max_prob = np.maximum(y_proba[:, 0], np.maximum(y_proba[:, 1], y_proba[:, 2]))
    neutral_mask = max_prob < probability_threshold
    y_pred[neutral_mask] = 1  # Neutral

    return y_pred, y_proba


def predict_single(
    lstm_model: LSTMClassifier,
    cb_model: Any,
    scaler: Any,
    recent_klines: list[Kline],
    seq_len: int = 96,
    probability_threshold: float = 0.12,
) -> dict[str, Any]:
    """
    单次预测（用于实时推理）

    Args:
        lstm_model: LSTM 模型
        cb_model: CatBoost 模型
        scaler: StandardScaler
        recent_klines: 最近 N 根 K 线
        seq_len: 序列长度

    Returns:
        预测结果字典
    """
    if len(recent_klines) < seq_len:
        raise ValueError(f"K线数量不足: {len(recent_klines)}, 需要至少 {seq_len}")

    # 转换为 DataFrame
    df = pd.DataFrame(
        {
            "open_time": [k.open_time for k in recent_klines],
            "open": [k.open for k in recent_klines],
            "high": [k.high for k in recent_klines],
            "low": [k.low for k in recent_klines],
            "close": [k.close for k in recent_klines],
            "volume": [k.volume for k in recent_klines],
        }
    )

    # 生成特征
    features = generate_features(df)
    features = features.dropna()

    if len(features) < seq_len:
        raise ValueError(f"特征不足: {len(features)}, 需要至少 {seq_len}")

    feature_values = features.values[-seq_len:]

    # 预测
    y_pred, y_proba = predict(
        lstm_model,
        cb_model,
        scaler,
        feature_values.reshape(1, -1),
        seq_len=seq_len,
        probability_threshold=probability_threshold,
    )

    label_names = ["DOWN", "NEUTRAL", "UP"]

    return {
        "signal": label_names[int(y_pred[0])],
        "probability": {
            "down": float(y_proba[0, 0]),
            "neutral": float(y_proba[0, 1]),
            "up": float(y_proba[0, 2]),
        },
        "confidence": float(np.max(y_proba[0])),
        "filtered": bool(y_proba[0, 0] < probability_threshold and y_proba[0, 2] < probability_threshold),
    }


def main():
    parser = argparse.ArgumentParser(description="LSTM+CatBoost 推理")
    parser.add_argument("--model", type=str, default="models/lstm_catboost/fold_1", help="模型目录")
    parser.add_argument("--data", type=str, default="data/futures/um/klines/BTCUSDT_15m_mark.parquet", help="数据文件")
    parser.add_argument("--seq-len", type=int, default=48, help="序列长度")
    parser.add_argument("--threshold", type=float, default=0.12, help="概率阈值")
    args = parser.parse_args()

    model_dir = Path(args.model)
    data_path = Path(args.data)

    print("=" * 60)
    print("LSTM+CatBoost 推理")
    print("=" * 60)
    print(f"模型: {model_dir}")
    print(f"数据: {data_path}")
    print()

    # 加载模型
    print("加载模型...")
    lstm_model, cb_model, scaler, config = load_model(model_dir)
    print(f"  LSTM: {config}")
    print(f"  CatBoost: 加载成功")

    # 加载数据
    print("\n加载数据...")
    df = pd.read_parquet(data_path)
    df = df.sort_values("open_time").reset_index(drop=True)
    print(f"  数据量: {len(df)} rows")
    print(
        f"  时间范围: {datetime.fromtimestamp(df['open_time'].min() / 1000)} to {datetime.fromtimestamp(df['open_time'].max() / 1000)}"
    )

    # 生成特征
    print("\n生成特征...")
    features = generate_features(df)
    features = features.dropna()
    print(f"  特征数量: {len(features)}")

    # 预测
    print("\n预测...")
    y_pred, y_proba = predict(
        lstm_model,
        cb_model,
        scaler,
        features.values,
        seq_len=args.seq_len,
        probability_threshold=args.threshold,
    )

    # 统计
    label_counts = pd.Series(y_pred).value_counts().sort_index()
    total = len(y_pred)
    print(f"\n预测结果:")
    for label, name in [(0, "DOWN"), (1, "NEUTRAL"), (2, "UP")]:
        count = label_counts.get(label, 0)
        print(f"  {name}: {count} ({count / total * 100:.1f}%)")

    # 计算平均置信度
    max_prob = np.maximum(y_proba[:, 0], np.maximum(y_proba[:, 1], y_proba[:, 2]))
    print(f"\n平均置信度: {max_prob.mean():.3f}")
    print(f"最小置信度: {max_prob.min():.3f}")
    print(f"最大置信度: {max_prob.max():.3f}")

    # 保存结果
    output_path = model_dir / "predictions.parquet"
    result_df = pd.DataFrame(
        {
            "open_time": df["open_time"].values[args.seq_len :],
            "close": df["close"].values[args.seq_len :],
            "prediction": y_pred,
            "prob_down": y_proba[:, 0],
            "prob_neutral": y_proba[:, 1],
            "prob_up": y_proba[:, 2],
        }
    )
    result_df.to_parquet(output_path, index=False)
    print(f"\n预测结果已保存: {output_path}")


if __name__ == "__main__":
    main()
