"""
Walk-Forward ML Training Script - FocalLoss 版本

按照指南实现：
- neutral_scale=0.55
- class_weights=[2.5, 1.0, 2.5] (L, N, S)
- FocalLoss (gamma=2.0)
- Walk-Forward 月度滚动验证
- Macro F1 为主指标
- 概率阈值后处理

Usage:
    uv run python scripts/walkforward_focal.py
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import talib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.storage import DataStorage
from ml.features.single_pair import SinglePairFeatureGenerator
from ml.labels.three_class import ThreeClassLabeler

# ============ 配置 ============
NEUTRAL_SCALE = 0.55
LOOKFORWARD_BARS = 1

# FocalLoss 参数
FOCAL_GAMMA = 2.0
CLASS_WEIGHTS = [2.5, 1.0, 2.5]  # [DOWN, NEUTRAL, UP]

# 后处理阈值
PROBABILITY_THRESHOLD = 0.15  # 如果 max(p_l, p_s) < 0.15，强制 Neutral

# Walk-Forward 窗口配置
TRAIN_WINDOW_DAYS = 90  # 训练窗口 90 天
VALID_WINDOW_DAYS = 30  # 验证窗口 30 天
TEST_WINDOW_DAYS = 30  # 测试窗口 30 天

# 标签常量
DOWN_LABEL = 0
NEUTRAL_LABEL = 1
UP_LABEL = 2


# ============ FocalLoss 实现 ============


def focal_loss_objective(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Focal Loss objective for CatBoost custom loss function

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    其中 p_t 是正确类别的预测概率

    Returns:
        gradient, hessian
    """
    num_classes = 3
    y_pred = y_pred.reshape(-1, num_classes)

    # softmax 概率
    exp_pred = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True))
    probs = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)

    # 正确类别的概率
    p_t = probs[np.arange(len(y_true)), y_true.astype(int)]

    # alpha_t
    alpha = np.array(CLASS_WEIGHTS)[y_true.astype(int)]

    # Focal weight: (1 - p_t)^gamma
    focal_weight = np.power(1.0 - p_t, FOCAL_GAMMA)

    # gradient: alpha * focal_weight * (p_t - 1) for correct class
    # 对于多类，需要计算每个类的梯度
    grad = np.zeros_like(probs)
    for c in range(num_classes):
        mask_c = y_true == c
        if mask_c.sum() > 0:
            p_c = probs[mask_c, c]
            alpha_c = CLASS_WEIGHTS[c]

            # dFL/dp_c = alpha * [gamma * (1-p_t)^(gamma-1) * (-1) * p_c * log(p_t) + (1-p_t)^gamma * (-1)/p_c]
            # 简化为：alpha * focal_weight * (p_c - 1_{c=y}) / p_c
            grad[mask_c, c] = alpha_c * focal_weight[mask_c] * (p_c - 1.0) / (p_c + 1e-8)

    # hessian: 近似为常数（CatBoost 需要的二阶导）
    hess = np.abs(grad) * 0.01 + 0.1

    return grad.flatten(), hess.flatten()


def focal_loss_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Focal Loss 评估指标
    """
    num_classes = 3
    y_pred = y_pred.reshape(-1, num_classes)

    exp_pred = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True))
    probs = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)

    p_t = probs[np.arange(len(y_true)), y_true.astype(int)]
    alpha = np.array(CLASS_WEIGHTS)[y_true.astype(int)]

    focal_loss = -alpha * np.power(1.0 - p_t, FOCAL_GAMMA) * np.log(p_t + 1e-8)
    return float(np.mean(focal_loss))


# ============ 数据准备 ============


def load_data(symbol: str = "BTCUSDT", interval: str = "15m") -> tuple[pd.DataFrame, list]:
    """加载 K 线数据，返回 DataFrame 和 Kline 列表"""
    # 优先使用 mark price 数据（更长的时间序列）
    mark_parquet = Path("data/futures/um/klines/BTCUSDT_15m_mark.parquet")

    if mark_parquet.exists():
        print(f"加载 mark price 数据 from {mark_parquet}")
        df = pd.read_parquet(mark_parquet)
        df = df.sort_values("open_time").reset_index(drop=True)

        # 转换为 Kline 对象列表
        from models import Kline

        klines = [
            Kline(
                symbol=symbol,
                interval=interval,
                open_time=int(row.open_time),
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(row.volume),
                close_time=int(row.open_time) + 15 * 60 * 1000,  # 15m later
                is_closed=True,
            )
            for _, row in df.iterrows()
        ]
        return df, klines

    # 回退到普通 kline 数据
    storage = DataStorage()
    klines = storage.load_klines(symbol, interval)

    if not klines:
        raise ValueError(f"No klines found for {symbol} {interval}")

    df = pd.DataFrame(
        [
            {
                "open_time": k.open_time,
                "open": k.open,
                "high": k.high,
                "low": k.low,
                "close": k.close,
                "volume": k.volume,
            }
            for k in klines
        ]
    )
    df = df.sort_values("open_time").reset_index(drop=True)
    return df, klines


def prepare_features_and_labels(
    df: pd.DataFrame,
    klines: list,
    neutral_scale: float = NEUTRAL_SCALE,
    lookforward_bars: int = LOOKFORWARD_BARS,
) -> dict[str, Any]:
    """生成特征和标签"""
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values

    # 生成特征
    single_gen = SinglePairFeatureGenerator()
    features = single_gen.generate(klines)
    features = features.reset_index(drop=True)  # 确保是简单的 0,1,2,... 索引

    # 计算 ATR 和标签
    atr = talib.ATR(highs, lows, closes, 14)
    labeler = ThreeClassLabeler(neutral_scale=neutral_scale, lookforward_bars=lookforward_bars)
    labels = labeler.label(closes, atr)
    labels = labels.reset_index(drop=True)  # 确保是简单的 0,1,2,... 索引

    # 对齐：取较短的长度
    min_len = min(len(features), len(labels))
    features = features.iloc[:min_len].reset_index(drop=True)
    labels = labels.iloc[:min_len].reset_index(drop=True)
    atr = atr[:min_len]
    closes = closes[:min_len]

    # 收益（与 labels 对齐）
    future_close = closes[lookforward_bars:]
    current_close = closes[:-lookforward_bars]
    returns = (future_close - current_close) / current_close

    return {
        "features": features,
        "labels": labels,
        "returns": returns,
        "atr": atr[: len(returns)],
        "closes": closes[: len(returns)],
        "open_times": df["open_time"].values[: len(returns)],
    }


# ============ Walk-Forward 训练 ============


def walkforward_train(
    data: dict[str, Any],
    train_days: int = TRAIN_WINDOW_DAYS,
    valid_days: int = VALID_WINDOW_DAYS,
    test_days: int = TEST_WINDOW_DAYS,
) -> list[dict[str, Any]]:
    """
    Walk-Forward 训练，按时间窗口滚动

    每个月：
    - 用过去 train_days 训练
    - 用 valid_days 验证（调超参）
    - 用 test_days 测试
    """
    from catboost import CatBoostClassifier

    features = data["features"]
    labels = data["labels"]
    returns = data["returns"]
    atr = data["atr"]
    open_times = data["open_times"]

    results = []

    # 估算每多少条数据 = 1 天（15m K线 = 96条/天）
    bars_per_day = 96

    train_bars = train_days * bars_per_day
    valid_bars = valid_days * bars_per_day
    test_bars = test_days * bars_per_day

    step_bars = test_bars  # 每月滚动一步

    start_idx = 0

    while start_idx + train_bars + valid_bars + test_bars < len(features):
        # 窗口划分
        train_end = start_idx + train_bars
        valid_end = train_end + valid_bars
        test_end = valid_end + test_bars

        # 分割数据
        X_train = features.iloc[start_idx:train_end]
        X_valid = features.iloc[train_end:valid_end]
        X_test = features.iloc[valid_end:test_end]

        y_train = labels.iloc[start_idx:train_end].values
        y_valid = labels.iloc[train_end:valid_end].values
        y_test = labels.iloc[valid_end:test_end].values

        ret_train = returns[start_idx:train_end]
        ret_valid = returns[train_end:valid_end]
        ret_test = returns[valid_end:test_end]

        atr_train = atr[start_idx:train_end]
        atr_test = atr[valid_end:test_end]

        # 检查类别
        if len(np.unique(y_train)) < 3:
            print(f"Window {start_idx}: Missing classes in training, skipping")
            start_idx += step_bars
            continue

        # ============ 训练 CatBoost ============
        model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3.0,
            task_type="GPU",
            devices="0",
            loss_function="MultiClass",
            class_weights=CLASS_WEIGHTS,
            verbose=0,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=(X_valid, y_valid),
            early_stopping_rounds=100,
            verbose=False,
        )

        # ============ 预测（无后处理） ============
        y_pred_test_raw = model.predict(X_test).flatten()
        y_proba_test_raw = model.predict_proba(X_test)

        # ============ 预测（有后处理） ============
        y_proba_test_post = y_proba_test_raw.copy()
        max_prob = np.maximum(y_proba_test_post[:, DOWN_LABEL], y_proba_test_post[:, UP_LABEL])
        neutral_mask = max_prob < PROBABILITY_THRESHOLD
        y_pred_test_post = y_pred_test_raw.copy()
        y_pred_test_post[neutral_mask] = NEUTRAL_LABEL

        # ============ 计算指标 ============
        metrics_raw = compute_all_metrics(y_test, y_pred_test_raw, y_proba_test_raw, ret_test, atr_test)
        metrics_post = compute_all_metrics(y_test, y_pred_test_post, y_proba_test_post, ret_test, atr_test)

        # 混淆矩阵
        cm_raw = confusion_matrix(y_test, y_pred_test_raw, labels=[0, 1, 2])
        cm_post = confusion_matrix(y_test, y_pred_test_post, labels=[0, 1, 2])

        # 时间范围
        test_start_time = datetime.fromtimestamp(open_times[valid_end] / 1000)
        test_end_time = datetime.fromtimestamp(open_times[test_end - 1] / 1000)

        result = {
            "window_start_idx": start_idx,
            "train_size": train_end - start_idx,
            "valid_size": valid_end - train_end,
            "test_size": test_end - valid_end,
            "test_period": f"{test_start_time.strftime('%Y-%m-%d')} to {test_end_time.strftime('%Y-%m-%d')}",
            **metrics_raw,
            **{f"{k}_post": v for k, v in metrics_post.items()},
            "confusion_matrix_raw": cm_raw.tolist(),
            "confusion_matrix_post": cm_post.tolist(),
        }
        results.append(result)

        # 打印摘要
        print(
            f"[{result['test_period']}] "
            f"Raw: Acc={metrics_raw['accuracy']:.3f} MacroF1={metrics_raw['macro_f1']:.3f} | "
            f"Post: Acc={metrics_post['accuracy']:.3f} MacroF1={metrics_post['macro_f1']:.3f}"
        )

        start_idx += step_bars

    return results


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    returns: np.ndarray,
    atr: np.ndarray,
) -> dict[str, Any]:
    """计算所有评估指标"""
    metrics = {}

    # 基础准确率
    metrics["accuracy"] = float(np.mean(y_pred == y_true))

    # Macro F1 (主指标)
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

    # Macro F1
    metrics["macro_f1"] = float(np.mean(f1))
    metrics["macro_precision"] = float(np.mean(precision))
    metrics["macro_recall"] = float(np.mean(recall))

    # 条件概率 P(actual=c | predict=c)
    for c, name in [(DOWN_LABEL, "down"), (NEUTRAL_LABEL, "neutral"), (UP_LABEL, "up")]:
        mask = y_pred == c
        if mask.sum() > 0:
            metrics[f"p_actual_{name}_given_predict_{name}"] = float(np.mean(y_true[mask] == c))
        else:
            metrics[f"p_actual_{name}_given_predict_{name}"] = 0.0

    # 加权 Loss
    errors = y_pred != y_true
    if errors.sum() > 0:
        error_returns = np.abs(returns[errors])
        error_atr = atr[errors] + 1e-10
        weighted_loss = np.mean(error_returns / error_atr)
    else:
        weighted_loss = 0.0

    correct_mask = ~errors
    if correct_mask.sum() > 0:
        correct_returns = np.abs(returns[correct_mask])
        correct_atr = atr[correct_mask] + 1e-10
        weighted_loss -= np.mean(correct_returns / correct_atr)

    metrics["weighted_loss"] = float(weighted_loss)

    # Long/Short 召回率（重点关注）
    metrics["long_recall"] = metrics["recall_up"]
    metrics["short_recall"] = metrics["recall_down"]
    metrics["long_precision"] = metrics["precision_up"]
    metrics["short_precision"] = metrics["precision_down"]

    return metrics


# ============ 可视化 ============


def plot_results(results: list[dict[str, Any]], output_prefix: str = "walkforward_focal") -> None:
    """绘制结果图表"""
    windows = range(len(results))

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))

    # 1. Macro F1 (主指标)
    ax1 = axes[0, 0]
    ax1.plot(windows, [r["macro_f1"] for r in results], "b-o", label="Macro F1", linewidth=2, markersize=8)
    ax1.plot(windows, [r["macro_f1_post"] for r in results], "g-s", label="Macro F1 (Post)", linewidth=2, markersize=8)
    ax1.set_xlabel("Fold")
    ax1.set_ylabel("Macro F1")
    ax1.set_title("Macro F1 Score (Primary Metric)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.0])

    # 2. Accuracy
    ax2 = axes[0, 1]
    ax2.plot(windows, [r["accuracy"] for r in results], "b-o", label="Accuracy", linewidth=2, markersize=8)
    ax2.plot(windows, [r["accuracy_post"] for r in results], "g-s", label="Accuracy (Post)", linewidth=2, markersize=8)
    ax2.set_xlabel("Fold")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.0])

    # 3. Long/Short 召回率
    ax3 = axes[1, 0]
    ax3.plot(windows, [r["long_recall"] for r in results], "r-o", label="Long Recall", linewidth=2, markersize=8)
    ax3.plot(windows, [r["short_recall"] for r in results], "b-s", label="Short Recall", linewidth=2, markersize=8)
    ax3.plot(windows, [r["recall_neutral"] for r in results], "g-^", label="Neutral Recall", linewidth=2, markersize=8)
    ax3.set_xlabel("Fold")
    ax3.set_ylabel("Recall")
    ax3.set_title("Recall by Class")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1.0])

    # 4. Long/Short 精确率
    ax4 = axes[1, 1]
    ax4.plot(windows, [r["long_precision"] for r in results], "r-o", label="Long Precision", linewidth=2, markersize=8)
    ax4.plot(
        windows, [r["short_precision"] for r in results], "b-s", label="Short Precision", linewidth=2, markersize=8
    )
    ax4.plot(
        windows, [r["precision_neutral"] for r in results], "g-^", label="Neutral Precision", linewidth=2, markersize=8
    )
    ax4.set_xlabel("Fold")
    ax4.set_ylabel("Precision")
    ax4.set_title("Precision by Class")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1.0])

    # 5. 条件概率 P(actual=c | predict=c)
    ax5 = axes[2, 0]
    ax5.plot(
        windows,
        [r["p_actual_down_given_predict_down"] for r in results],
        "b-o",
        label="P(actual=L|predict=L)",
        linewidth=2,
        markersize=8,
    )
    ax5.plot(
        windows,
        [r["p_actual_neutral_given_predict_neutral"] for r in results],
        "g-s",
        label="P(actual=N|predict=N)",
        linewidth=2,
        markersize=8,
    )
    ax5.plot(
        windows,
        [r["p_actual_up_given_predict_up"] for r in results],
        "r-^",
        label="P(actual=S|predict=S)",
        linewidth=2,
        markersize=8,
    )
    ax5.set_xlabel("Fold")
    ax5.set_ylabel("Probability")
    ax5.set_title("P(actual | predict)")
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([0, 1.0])

    # 6. Weighted Loss
    ax6 = axes[2, 1]
    ax6.plot(windows, [r["weighted_loss"] for r in results], "b-o", linewidth=2, markersize=8)
    ax6.set_xlabel("Fold")
    ax6.set_ylabel("Weighted Loss")
    ax6.set_title("Weighted Loss (Lower is Better)")
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = f"{output_prefix}_results.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {plot_path}")


def print_summary(results: list[dict[str, Any]]) -> None:
    """打印汇总结果"""
    print("\n" + "=" * 80)
    print("WALK-FORWARD RESULTS SUMMARY")
    print("=" * 80)

    # 计算平均值
    avg_macro_f1 = np.mean([r["macro_f1"] for r in results])
    avg_macro_f1_post = np.mean([r["macro_f1_post"] for r in results])
    avg_accuracy = np.mean([r["accuracy"] for r in results])
    avg_accuracy_post = np.mean([r["accuracy_post"] for r in results])
    avg_long_recall = np.mean([r["long_recall"] for r in results])
    avg_short_recall = np.mean([r["short_recall"] for r in results])
    avg_long_precision = np.mean([r["long_precision"] for r in results])
    avg_short_precision = np.mean([r["short_precision"] for r in results])

    print(f"\n主指标 (Macro F1):")
    print(f"  Raw:     {avg_macro_f1:.4f}")
    print(f"  Post:    {avg_macro_f1_post:.4f}")

    print(f"\n准确率 (Accuracy):")
    print(f"  Raw:     {avg_accuracy:.4f}")
    print(f"  Post:    {avg_accuracy_post:.4f}")

    print(f"\nLong/Short 召回率:")
    print(f"  Long Recall:     {avg_long_recall:.4f}")
    print(f"  Short Recall:    {avg_short_recall:.4f}")

    print(f"\nLong/Short 精确率:")
    print(f"  Long Precision:  {avg_long_precision:.4f}")
    print(f"  Short Precision: {avg_short_precision:.4f}")

    # 逐窗口打印
    print("\n" + "-" * 80)
    print(f"{'Period':<20} {'Acc':<8} {'MacroF1':<8} {'LongR':<8} {'ShortR':<8} {'LongP':<8} {'ShortP':<8}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['test_period']:<20} "
            f"{r['accuracy_post']:<8.3f} "
            f"{r['macro_f1_post']:<8.3f} "
            f"{r['long_recall']:<8.3f} "
            f"{r['short_recall']:<8.3f} "
            f"{r['long_precision']:<8.3f} "
            f"{r['short_precision']:<8.3f}"
        )


def main() -> None:
    print("=" * 80)
    print("Walk-Forward ML Training with FocalLoss")
    print("=" * 80)
    print(f"\n配置:")
    print(f"  neutral_scale: {NEUTRAL_SCALE}")
    print(f"  lookforward_bars: {LOOKFORWARD_BARS}")
    print(f"  focal_gamma: {FOCAL_GAMMA}")
    print(f"  class_weights: {CLASS_WEIGHTS}")
    print(f"  probability_threshold: {PROBABILITY_THRESHOLD}")
    print(f"  train_window: {TRAIN_WINDOW_DAYS} days")
    print(f"  valid_window: {VALID_WINDOW_DAYS} days")
    print(f"  test_window: {TEST_WINDOW_DAYS} days")

    # 加载数据
    print("\n加载数据...")
    df, klines = load_data("BTCUSDT", "15m")
    print(f"数据量: {len(df)} 条 K线")
    print(
        f"时间范围: {datetime.fromtimestamp(df['open_time'].min() / 1000)} to {datetime.fromtimestamp(df['open_time'].max() / 1000)}"
    )

    # 生成特征和标签
    print("\n生成特征和标签...")
    data = prepare_features_and_labels(df, klines)
    print(f"样本数: {len(data['features'])}")

    # 标签分布
    label_counts = pd.Series(data["labels"]).value_counts().sort_index()
    total = len(data["labels"])
    print(f"\n标签分布:")
    print(f"  DOWN (L):    {label_counts.get(0, 0):>6} ({label_counts.get(0, 0) / total * 100:.1f}%)")
    print(f"  NEUTRAL (N): {label_counts.get(1, 0):>6} ({label_counts.get(1, 0) / total * 100:.1f}%)")
    print(f"  UP (S):      {label_counts.get(2, 0):>6} ({label_counts.get(2, 0) / total * 100:.1f}%)")

    # Walk-Forward 训练
    print("\n开始 Walk-Forward 训练...")
    results = walkforward_train(data)

    if not results:
        print("错误: 没有有效的训练窗口")
        return

    # 打印汇总
    print_summary(results)

    # 绘制结果
    plot_results(results)

    # 保存结果
    import json

    output_path = Path("models/walkforward_focal_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 转换 numpy 类型为 Python 原生类型
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
                    "focal_gamma": FOCAL_GAMMA,
                    "class_weights": CLASS_WEIGHTS,
                    "probability_threshold": PROBABILITY_THRESHOLD,
                    "train_window_days": TRAIN_WINDOW_DAYS,
                    "valid_window_days": VALID_WINDOW_DAYS,
                    "test_window_days": TEST_WINDOW_DAYS,
                },
                "results": convert_to_native(results),
            },
            f,
            indent=2,
        )

    print(f"\n结果已保存到 {output_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
