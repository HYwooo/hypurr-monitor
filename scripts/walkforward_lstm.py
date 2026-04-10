"""
Walk-Forward LSTM+CatBoost 三分类训练脚本

按照指南实现：
- LSTM 编码器 → 64维特征向量
- CatBoost 三分类 (Long/Neutral/Short)
- FocalLoss (gamma=2.0)
- Walk-Forward Expanding Window 验证
- Macro F1 为主指标
- 概率阈值后处理

Architecture:
  输入：BTC 15m K线历史数据
      ↓
序列窗口化 → LSTM 编码器 → 特征向量 (64维)
      ↓
CatBoost 分类器 → Long / Neutral / Short
      ↓
后处理过滤 → 信号输出

Usage:
    uv run python scripts/walkforward_lstm.py
"""

import sys
import json
import random
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # 非交互式后端
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
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import Kline
from ml.labels.three_class import ThreeClassLabeler

# ============ 配置 ============
NEUTRAL_SCALE = 0.55
LOOKFORWARD_BARS = 6  # 未来6根K线 (1.5h)
SEQ_LEN = 48  # 窗口长度 (12h)
HIDDEN_DIM = 128
LSTM_OUTPUT_DIM = 64
FOCAL_GAMMA = 2.0
CLASS_WEIGHTS = [2.5, 1.0, 2.5]  # [DOWN, NEUTRAL, UP]
PROBABILITY_THRESHOLD = 0.12

# LSTM 预训练参数
LSTM_EPOCHS = 20
LSTM_BATCH_SIZE = 512
LSTM_LR = 0.002

# Walk-Forward 配置（根据实际数据范围调整）
# 数据范围：2024-01-01 到 2025-04-01 (15m K线)
# Fold 0: 训练2024Q1-Q2 (1-6月), 验证2024Q3 (7-9月), 测试2024Q4 (10-12月)
# Fold 1: 训练2024Q1-Q3 (1-9月), 验证2024Q4 (10-12月), 测试2025Q1 (1-3月)

# 标签常量
DOWN_LABEL = 0
NEUTRAL_LABEL = 1
UP_LABEL = 2

# 设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")


# ============ 数据加载 ============


def load_data(symbol: str = "BTCUSDT", interval: str = "15m") -> pd.DataFrame:
    """加载 K 线数据"""
    mark_parquet = Path("data/futures/um/klines/BTCUSDT_15m_mark.parquet")

    if mark_parquet.exists():
        print(f"加载 mark price 数据 from {mark_parquet}")
        df = pd.read_parquet(mark_parquet)
        df = df.sort_values("open_time").reset_index(drop=True)
        return df

    raise FileNotFoundError(f"Mark price data not found at {mark_parquet}")


# ============ 特征工程 ============


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    生成时序特征

    时序特征（组成滑动窗口，用于LSTM输入）：
    - 收益率：1/3/6/12/24 周期
    - 成交量比率：当期成交量 / N期均值
    - 波动率：N期收益率标准差
    - RSI(14)
    - KDJ 的 K值/D值
    - MACD 的 dif / dea / histogram
    - 布林带位置：(close - lower) / (upper - lower)

    Returns:
        特征 DataFrame
    """
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values

    features = pd.DataFrame(index=df.index)

    # 收益率特征
    for period in [1, 3, 6, 12, 24]:
        features[f"return_{period}"] = pd.Series(close).pct_change(period)

    # 波动率 (N期收益率标准差)
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


# ============ 序列窗口化 ============


class SequenceDataset(Dataset):
    """序列数据集"""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        seq_len: int = SEQ_LEN,
    ):
        """
        Args:
            features: 特征数组 (n_samples, n_features)
            labels: 标签数组 (n_samples,)
            seq_len: 序列长度
        """
        self.features = features
        self.labels = labels
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.features) - self.seq_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        X = self.features[idx : idx + self.seq_len]
        y = self.labels[idx + self.seq_len]
        return torch.FloatTensor(X), torch.LongTensor([y])[0]


# ============ LSTM 编码器 ============


class LSTMEncoder(nn.Module):
    """
    LSTM 编码器

    架构：
    - LSTM(n_features, hidden_dim, n_layers=2, batch_first=True, dropout=0.2)
    - 输出：hidden_state from last time step → (batch, hidden_dim)
    - Linear(hidden_dim, 128) + ReLU + Dropout(0.3)
    - Linear(128, 64)  ← 这个64维向量就是输出特征
    """

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
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            (batch, lstm_output_dim)
        """
        # LSTM 输出
        _, (hidden, _) = self.lstm(x)
        # 取最后一层的 hidden state
        last_hidden = hidden[-1]  # (batch, hidden_dim)
        # 全连接层
        output = self.fc(last_hidden)  # (batch, 64)
        return output


class LSTMClassifier(nn.Module):
    """
    LSTM 分类器（用于预训练）

    LSTMEncoder + 分类头
    """

    def __init__(
        self, n_features: int, hidden_dim: int = HIDDEN_DIM, lstm_output_dim: int = LSTM_OUTPUT_DIM, n_classes: int = 3
    ):
        super().__init__()

        self.encoder = LSTMEncoder(n_features, hidden_dim, lstm_output_dim)
        self.classifier = nn.Linear(lstm_output_dim, n_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            features: (batch, lstm_output_dim) - 编码特征
            logits: (batch, n_classes) - 分类 logits
        """
        features = self.encoder(x)
        logits = self.classifier(features)
        return features, logits


# ============ FocalLoss ============


class FocalLoss(nn.Module):
    """
    Focal Loss 实现

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    其中：
    - p_t 是正确类别的预测概率
    - gamma 控制难分样本的聚焦程度
    - alpha 是类别权重
    """

    def __init__(self, gamma: float = 2.0, alpha: list[float] | None = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha if alpha is not None else [1.0, 1.0, 1.0])

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, n_classes)
            targets: (batch,)
        Returns:
            loss: scalar
        """
        ce_loss = nn.CrossEntropyLoss(reduction="none", weight=self.alpha.to(logits.device))
        loss = ce_loss(logits, targets)

        # 获取正确类别的概率
        probs = torch.softmax(logits, dim=1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # 加权
        loss = focal_weight * loss

        return loss.mean()


# ============ 标签生成 ============


def generate_labels(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    neutral_scale: float = NEUTRAL_SCALE,
    lookforward_bars: int = LOOKFORWARD_BARS,
) -> np.ndarray:
    """
    生成三分类标签

    - UP (2):   未来6根K线累计收益率 > +0.4%
    - DOWN (0): 未来6根K线累计收益率 < -0.4%
    - NEUTRAL (1): 其余
    """
    atr = talib.ATR(highs, lows, closes, 14)

    future_close = closes[lookforward_bars:]
    current_close = closes[:-lookforward_bars]
    future_atr = atr[:-lookforward_bars]

    future_return = (future_close - current_close) / current_close

    labels = np.full(len(future_return), NEUTRAL_LABEL)

    # 动态阈值：ATR * neutral_scale
    threshold = (future_atr / current_close) * neutral_scale

    labels[future_return > threshold] = UP_LABEL
    labels[future_return < -threshold] = DOWN_LABEL

    return labels


# ============ Walk-Forward 训练 ============


def train_lstm(
    train_loader: DataLoader,
    valid_loader: DataLoader,
    n_features: int,
    device: torch.device,
    epochs: int = LSTM_EPOCHS,
    lr: float = LSTM_LR,
) -> LSTMClassifier:
    """训练 LSTM 分类器"""

    model = LSTMClassifier(n_features=n_features).to(device)
    criterion = FocalLoss(gamma=FOCAL_GAMMA, alpha=CLASS_WEIGHTS)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_loss = float("inf")
    best_state: dict | None = None
    patience_counter = 0
    max_patience = 5

    for epoch in range(epochs):
        # 训练
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

        # 验证
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

        # 早停
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"  LSTM Early stopping at epoch {epoch + 1}, best valid loss: {best_loss:.4f}")
                break

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}: train_loss={train_loss:.4f}, valid_loss={valid_loss:.4f}")

    # 恢复最佳模型
    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def extract_lstm_features(
    model: LSTMClassifier, data_loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """用训练好的 LSTM 提取特征"""
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


def walkforward_train(
    df: pd.DataFrame,
    features: pd.DataFrame,
    train_months: list[int],
    valid_month: int,
    test_month: int,
) -> dict[str, Any]:
    """
    Walk-Forward 训练一个 Fold

    Args:
        df: 原始 DataFrame
        features: 特征 DataFrame
        train_months: 训练集月份列表 (YYYYMM 格式)
        valid_month: 验证集月份 (YYYYMM 格式)
        test_month: 测试集月份 (YYYYMM 格式)

    Returns:
        评估结果
    """
    from catboost import CatBoostClassifier

    # 准备数据
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values

    # 生成标签
    labels = generate_labels(closes, highs, lows)
    labels = labels[: len(features)]  # 对齐

    # 时间戳
    open_times = df["open_time"].values[: len(features)]

    # 月份 (YYYYMM)
    dates = pd.to_datetime(open_times, unit="ms")
    months = dates.year * 100 + dates.month

    # 特征数组
    feature_values = features.values

    # 标准化（LSTM 输入需要）
    scaler = StandardScaler()
    feature_scaled = scaler.fit_transform(feature_values)

    # 划分数据集
    train_mask = np.isin(months, train_months)
    valid_mask = months == valid_month
    test_mask = months == test_month

    # 确保有足够数据
    train_indices = np.where(train_mask)[0]
    valid_indices = np.where(valid_mask)[0]
    test_indices = np.where(test_mask)[0]

    if len(train_indices) < SEQ_LEN or len(valid_indices) < SEQ_LEN or len(test_indices) < SEQ_LEN:
        raise ValueError(f"数据不足: train={len(train_indices)}, valid={len(valid_indices)}, test={len(test_indices)}")

    print(f"  训练集: {len(train_indices)} 样本 ({train_months})")
    print(f"  验证集: {len(valid_indices)} 样本 ({valid_month})")
    print(f"  测试集: {len(test_indices)} 样本 ({test_month})")

    # ============ 阶段1: LSTM 预训练 ============
    print("  阶段1: LSTM 预训练...")

    # 创建训练数据集（只用于 LSTM 预训练）
    train_seq_indices = train_indices[train_indices >= SEQ_LEN]

    # 构建序列数据集
    train_X = []
    train_y = []
    for idx in train_seq_indices:
        start = idx - SEQ_LEN
        train_X.append(feature_scaled[start:idx])
        train_y.append(labels[idx])

    train_X = np.array(train_X)
    train_y = np.array(train_y)

    # 验证集序列
    valid_seq_indices = valid_indices[valid_indices >= SEQ_LEN]
    valid_X = []
    valid_y = []
    for idx in valid_seq_indices:
        start = idx - SEQ_LEN
        valid_X.append(feature_scaled[start:idx])
        valid_y.append(labels[idx])

    valid_X = np.array(valid_X)
    valid_y = np.array(valid_y)

    print(f"    LSTM 训练序列: {len(train_X)}, 验证序列: {len(valid_X)}")

    # 创建 DataLoader（数据已经是 (n_sequences, seq_len, n_features) 形状）
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_X),
        torch.LongTensor(train_y),
    )
    valid_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(valid_X),
        torch.LongTensor(valid_y),
    )

    train_loader = DataLoader(train_dataset, batch_size=LSTM_BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=LSTM_BATCH_SIZE, shuffle=False, num_workers=0)

    # 训练 LSTM
    n_features = train_X.shape[2]
    lstm_model = train_lstm(train_loader, valid_loader, n_features, DEVICE)

    # ============ 阶段2: CatBoost 训练 ============
    print("  阶段2: CatBoost 训练...")

    # 用 LSTM 提取训练集特征
    train_features, train_labels_extracted = extract_lstm_features(lstm_model, train_loader, DEVICE)

    # 验证集特征
    valid_features, valid_labels_extracted = extract_lstm_features(lstm_model, valid_loader, DEVICE)

    print(f"    训练特征: {train_features.shape}, 验证特征: {valid_features.shape}")

    # 标签分布
    for name, lbls in [("Train", train_labels_extracted), ("Valid", valid_labels_extracted)]:
        counts = pd.Series(lbls).value_counts().sort_index()
        print(f"    {name} 标签分布: DOWN={counts.get(0, 0)}, NEUTRAL={counts.get(1, 0)}, UP={counts.get(2, 0)}")

    # CatBoost 训练
    cb_model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        loss_function="MultiClass",
        class_weights=CLASS_WEIGHTS,
        early_stopping_rounds=100,
        verbose=100,
    )

    cb_model.fit(
        train_features,
        train_labels_extracted,
        eval_set=(valid_features, valid_labels_extracted),
        verbose=False,
    )

    # ============ 阶段3: 测试集评估 ============
    print("  阶段3: 测试集评估...")

    # 构建测试序列
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
        raise ValueError("测试集序列不足")

    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(test_X),
        torch.LongTensor(test_y),
    )
    test_loader = DataLoader(test_dataset, batch_size=LSTM_BATCH_SIZE, shuffle=False, num_workers=0)

    # 提取测试特征
    test_features, _ = extract_lstm_features(lstm_model, test_loader, DEVICE)

    # 预测
    y_pred_raw = cb_model.predict(test_features).flatten()
    y_proba_raw = cb_model.predict_proba(test_features)

    # 后处理
    y_proba_post = y_proba_raw.copy()
    max_prob = np.maximum(
        y_proba_post[:, DOWN_LABEL], np.maximum(y_proba_post[:, NEUTRAL_LABEL], y_proba_post[:, UP_LABEL])
    )
    neutral_mask = max_prob < PROBABILITY_THRESHOLD
    y_pred_post = y_pred_raw.copy()
    y_pred_post[neutral_mask] = NEUTRAL_LABEL

    # ============ 计算指标 ============
    metrics_raw = compute_metrics(test_y, y_pred_raw, y_proba_raw)
    metrics_post = compute_metrics(test_y, y_pred_post, y_proba_post)

    # 混淆矩阵
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


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, Any]:
    """计算所有评估指标"""
    metrics = {}

    # 基础准确率
    metrics["accuracy"] = float(np.mean(y_pred == y_true))

    # Macro F1
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[DOWN_LABEL, NEUTRAL_LABEL, UP_LABEL], average=None
    )

    metrics["precision_down"] = float(precision[0])
    metrics["precision_neutral"] = float(precision[1])
    metrics["precision_up"] = float(precision[2])
    metrics["recall_down"] = float(recall[0])
    metrics["recall_neutral"] = float(recall[1])
    metrics["recall_up"] = float(recall[2])
    metrics["f1_down"] = float(f1[0])
    metrics["f1_neutral"] = float(f1[1])
    metrics["f1_up"] = float(f1[2])
    metrics["support_down"] = int(support[0])
    metrics["support_neutral"] = int(support[1])
    metrics["support_up"] = int(support[2])

    metrics["macro_f1"] = float(np.mean(f1))
    metrics["macro_precision"] = float(np.mean(precision))
    metrics["macro_recall"] = float(np.mean(recall))

    # Long/Short 召回率（重点关注）
    metrics["long_recall"] = metrics["recall_up"]
    metrics["short_recall"] = metrics["recall_down"]
    metrics["long_precision"] = metrics["precision_up"]
    metrics["short_precision"] = metrics["precision_down"]

    return metrics


# ============ 可视化 ============


def plot_results(results: list[dict[str, Any]], output_prefix: str = "walkforward_lstm") -> None:
    """绘制结果图表"""
    if not results:
        print("没有结果可绘图")
        return

    windows = range(len(results))

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle("LSTM+CatBoost Walk-Forward Results", fontsize=16, fontweight="bold")

    # 1. Macro F1 (主指标)
    ax1 = axes[0, 0]
    macro_f1 = [r["macro_f1"] for r in results]
    macro_f1_post = [r["macro_f1_post"] for r in results]
    ax1.bar(windows, macro_f1, width=0.4, label="Raw", color="steelblue", alpha=0.8)
    ax1.bar([w + 0.4 for w in windows], macro_f1_post, width=0.4, label="Post", color="forestgreen", alpha=0.8)
    ax1.set_xlabel("Fold")
    ax1.set_ylabel("Macro F1")
    ax1.set_title("Macro F1 Score (Primary Metric)")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.set_ylim([0, 1.0])
    ax1.set_xticks(list(windows))
    for i, v in enumerate(macro_f1_post):
        ax1.text(i + 0.4, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)

    # 2. Accuracy
    ax2 = axes[0, 1]
    acc = [r["accuracy"] for r in results]
    acc_post = [r["accuracy_post"] for r in results]
    ax2.bar(windows, acc, width=0.4, label="Raw", color="steelblue", alpha=0.8)
    ax2.bar([w + 0.4 for w in windows], acc_post, width=0.4, label="Post", color="forestgreen", alpha=0.8)
    ax2.set_xlabel("Fold")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_ylim([0, 1.0])
    ax2.set_xticks(list(windows))

    # 3. Long/Short Recall
    ax3 = axes[1, 0]
    long_r = [r["long_recall"] for r in results]
    short_r = [r["short_recall"] for r in results]
    x = np.array(list(windows))
    width = 0.35
    ax3.bar(x - width / 2, long_r, width, label="Long Recall", color="crimson", alpha=0.8)
    ax3.bar(x + width / 2, short_r, width, label="Short Recall", color="royalblue", alpha=0.8)
    ax3.set_xlabel("Fold")
    ax3.set_ylabel("Recall")
    ax3.set_title("Long/Short Recall")
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis="y")
    ax3.set_ylim([0, 1.1])
    ax3.set_xticks(list(windows))
    for i, (lr, sr) in enumerate(zip(long_r, short_r)):
        ax3.text(i - width / 2, lr + 0.02, f"{lr:.2f}", ha="center", fontsize=8)
        ax3.text(i + width / 2, sr + 0.02, f"{sr:.2f}", ha="center", fontsize=8)

    # 4. Long/Short Precision
    ax4 = axes[1, 1]
    long_p = [r["long_precision"] for r in results]
    short_p = [r["short_precision"] for r in results]
    ax4.bar(x - width / 2, long_p, width, label="Long Precision", color="crimson", alpha=0.8)
    ax4.bar(x + width / 2, short_p, width, label="Short Precision", color="royalblue", alpha=0.8)
    ax4.set_xlabel("Fold")
    ax4.set_ylabel("Precision")
    ax4.set_title("Long/Short Precision")
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis="y")
    ax4.set_ylim([0, 1.1])
    ax4.set_xticks(list(windows))
    for i, (lp, sp) in enumerate(zip(long_p, short_p)):
        ax4.text(i - width / 2, lp + 0.02, f"{lp:.2f}", ha="center", fontsize=8)
        ax4.text(i + width / 2, sp + 0.02, f"{sp:.2f}", ha="center", fontsize=8)

    # 5. F1 by Class
    ax5 = axes[2, 0]
    f1_down = [r["f1_down"] for r in results]
    f1_neutral = [r["f1_neutral"] for r in results]
    f1_up = [r["f1_up"] for r in results]
    width = 0.25
    ax5.bar(x - width, f1_down, width, label="DOWN F1", color="royalblue", alpha=0.8)
    ax5.bar(x, f1_neutral, width, label="NEUTRAL F1", color="gray", alpha=0.8)
    ax5.bar(x + width, f1_up, width, label="UP F1", color="crimson", alpha=0.8)
    ax5.set_xlabel("Fold")
    ax5.set_ylabel("F1 Score")
    ax5.set_title("F1 Score by Class")
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis="y")
    ax5.set_ylim([0, 1.0])
    ax5.set_xticks(list(windows))

    # 6. Confusion Matrix (Average)
    ax6 = axes[2, 1]
    cm_raw = np.array(results[0]["confusion_matrix_raw"])
    for r in results[1:]:
        cm_raw += np.array(r["confusion_matrix_raw"])
    cm_avg = cm_raw / len(results)

    # 绘制热力图
    im = ax6.imshow(cm_avg, cmap="Blues", aspect="auto")
    ax6.set_xticks([0, 1, 2])
    ax6.set_yticks([0, 1, 2])
    ax6.set_xticklabels(["DOWN", "NEUTRAL", "UP"])
    ax6.set_yticklabels(["DOWN", "NEUTRAL", "UP"])
    ax6.set_xlabel("Predicted")
    ax6.set_ylabel("Actual")
    ax6.set_title("Confusion Matrix (Average)")

    # 添加数值标签
    for i in range(3):
        for j in range(3):
            text = ax6.text(
                j,
                i,
                f"{cm_avg[i, j]:.0f}",
                ha="center",
                va="center",
                color="white" if cm_avg[i, j] > cm_avg.max() / 2 else "black",
                fontsize=12,
            )

    plt.colorbar(im, ax=ax6)

    plt.tight_layout()
    plot_path = f"{output_prefix}_results.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\n图表已保存到 {plot_path}")


# ============ 主流程 ============


def main() -> None:
    print("=" * 80)
    print("Walk-Forward LSTM+CatBoost 三分类训练")
    print("=" * 80)
    print(f"\n配置:")
    print(f"  neutral_scale: {NEUTRAL_SCALE}")
    print(f"  lookforward_bars: {LOOKFORWARD_BARS}")
    print(f"  seq_len: {SEQ_LEN}")
    print(f"  hidden_dim: {HIDDEN_DIM}")
    print(f"  lstm_output_dim: {LSTM_OUTPUT_DIM}")
    print(f"  focal_gamma: {FOCAL_GAMMA}")
    print(f"  class_weights: {CLASS_WEIGHTS}")
    print(f"  probability_threshold: {PROBABILITY_THRESHOLD}")

    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # 加载数据
    print("\n加载数据...")
    df = load_data("BTCUSDT", "15m")
    print(f"数据量: {len(df)} 条 K线")
    print(
        f"时间范围: {datetime.fromtimestamp(df['open_time'].min() / 1000)} to {datetime.fromtimestamp(df['open_time'].max() / 1000)}"
    )

    # 生成特征
    print("\n生成特征...")
    features = generate_features(df)
    features = features.dropna()
    print(f"特征数量: {features.shape[1]}")
    print(f"特征列: {list(features.columns[:10])}...")

    # 对齐数据
    min_len = min(len(df), len(features))
    df = df.iloc[:min_len].reset_index(drop=True)
    features = features.iloc[:min_len].reset_index(drop=True)

    # Walk-Forward 训练
    print("\n开始 Walk-Forward 训练...")

    all_results = []

    # Fold 0: 训练2024Q1-Q2 (1-6月), 验证2024Q3 (7-9月), 测试2024Q4 (10-12月)
    print("\n" + "-" * 40)
    print("Fold 0: 训练2024Q1-Q2, 验证2024Q3, 测试2024Q4")
    print("-" * 40)
    try:
        result = walkforward_train(
            df,
            features,
            train_months=[202401, 202402, 202403, 202404, 202405, 202406],
            valid_month=202407,
            test_month=202410,
        )
        all_results.append(result)
    except Exception as e:
        print(f"Fold 0 失败: {e}")

    # Fold 1: 训练2024Q1-Q3 (1-9月), 验证2024Q4 (10-12月), 测试2025Q1 (1-3月)
    print("\n" + "-" * 40)
    print("Fold 1: 训练2024Q1-Q3, 验证2024Q4, 测试2025Q1")
    print("-" * 40)
    try:
        result = walkforward_train(
            df,
            features,
            train_months=[202401, 202402, 202403, 202404, 202405, 202406, 202407, 202408, 202409],
            valid_month=202410,
            test_month=202501,
        )
        all_results.append(result)
    except Exception as e:
        print(f"Fold 1 失败: {e}")

    if not all_results:
        print("错误: 没有有效的训练结果")
        return

    # 打印汇总
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    print(f"\n{'Fold':<6} {'Period':<20} {'Acc':<8} {'MacroF1':<8} {'LongR':<8} {'ShortR':<8}")
    print("-" * 60)
    for i, r in enumerate(all_results):
        period = f"{r['test_month']}"
        print(
            f"{i:<6} {period:<20} "
            f"{r['accuracy_post']:<8.3f} "
            f"{r['macro_f1_post']:<8.3f} "
            f"{r['long_recall']:<8.3f} "
            f"{r['short_recall']:<8.3f}"
        )

    # 平均
    avg_macro_f1 = np.mean([r["macro_f1_post"] for r in all_results])
    avg_long_recall = np.mean([r["long_recall"] for r in all_results])
    avg_short_recall = np.mean([r["short_recall"] for r in all_results])

    print("-" * 60)
    print(f"{'AVG':<6} {'':<20} {'':8} {avg_macro_f1:<8.3f} {avg_long_recall:<8.3f} {avg_short_recall:<8.3f}")

    # 保存结果
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

    print(f"\n结果已保存到 {output_path}")

    # 绘制结果图表
    plot_results(all_results)

    print("\nDone!")


if __name__ == "__main__":
    main()
