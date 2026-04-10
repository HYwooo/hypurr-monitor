"""
Walk-Forward ML Training Script

固定scale=0.4，实现完整的Walk-Forward训练
- CatBoost + LSTM
- 滚动窗口
- 条件概率 + 加权Loss

Usage:
    uv run python scripts/walkforward_train.py --days 365 --interval 15m
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import talib

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.storage import DataStorage
from data.binance_client import BinanceClient
from ml.features.single_pair import SinglePairFeatureGenerator
from ml.labels.three_class import ThreeClassLabeler
from ml.model.catboost_backend import CatBoostBackend
from ml.model.lstm_residual import LSTMResidualModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

NEUTRAL_SCALE = 0.4
LOOKFORWARD_BARS = 1


async def fetch_long_data(symbol: str, interval: str, days: int) -> list:
    """获取长期数据"""
    import time

    end_time_ms = int(time.time() * 1000)
    start_time_ms = end_time_ms - days * 24 * 3600 * 1000

    client = BinanceClient()
    klines = await client.fetch_klines_with_pagination(
        symbol,
        interval,
        start_time=start_time_ms,
        end_time=end_time_ms,
    )
    await client.close()
    return klines


def prepare_data(klines: list) -> dict[str, Any]:
    """准备数据"""
    closes = np.array([k.close for k in klines])
    highs = np.array([k.high for k in klines])
    lows = np.array([k.low for k in klines])

    single_gen = SinglePairFeatureGenerator()
    features = single_gen.generate(klines)
    atr = talib.ATR(highs, lows, closes, 14)

    labeler = ThreeClassLabeler(neutral_scale=NEUTRAL_SCALE, lookforward_bars=LOOKFORWARD_BARS)
    labels = labeler.label(closes, atr)

    # 计算收益
    future_close = closes[LOOKFORWARD_BARS:]
    current_close = closes[:-LOOKFORWARD_BARS]
    actual_returns = (future_close - current_close) / current_close

    # 对齐
    feat_start = features.index.min()
    labels_end = labels.index.max()
    align_start = max(feat_start, labels.index.min())
    align_end = min(features.index.max(), labels_end)

    feat_start_idx = align_start - feat_start
    feat_end_idx = align_end - feat_start + 1

    labels_aligned = labels.loc[align_start:align_end].reset_index(drop=True)
    actual_returns_aligned = actual_returns[feat_start_idx:feat_end_idx]
    atr_aligned = atr[feat_start_idx:feat_end_idx]
    features_aligned = features.loc[align_start:align_end].reset_index(drop=True)

    return {
        "features": features_aligned,
        "labels": labels_aligned,
        "returns": actual_returns_aligned,
        "atr": atr_aligned,
        "closes": np.array([k.close for k in klines[align_start : align_end + 1]]),
    }


def walkforward_train_validate(
    data: dict[str, Any],
    train_window: int = 500,
    step: int = 100,
) -> list[dict[str, Any]]:
    """
    Walk-Forward训练

    滚动窗口：
    - train_window: 训练窗口大小
    - step: 滚动步长
    - 每次：训练 -> 验证 -> 测试
    """
    features = data["features"]
    labels = data["labels"]
    returns = data["returns"]
    atr = data["atr"]

    results = []
    start_idx = 0

    while start_idx + train_window + step < len(features):
        # 窗口划分
        train_end = start_idx + train_window
        valid_end = train_end + step
        test_end = min(valid_end + step, len(features))

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
        atr_valid = atr[train_end:valid_end]
        atr_test = atr[valid_end:test_end]

        # 检查类别
        if len(np.unique(y_train)) < 2:
            logger.warning(f"Window {start_idx}: Only one class in training, skipping")
            start_idx += step
            continue

        # 计算样本权重（基于你的Loss规则）
        # 正确预测：weight = 1 - |return|/ATR（变化越大，奖励越大）
        # 错误预测：weight = 1 + |return|/ATR（变化越大，惩罚越大）
        # 但训练时我们不知道预测结果，所以用历史统计
        # 简化：用|return|/ATR作为惩罚因子（总是增加loss）
        train_weights = np.ones(len(y_train))
        return_ratio = np.abs(ret_train) / (atr_train + 1e-10)
        return_ratio = np.nan_to_num(return_ratio, nan=0.0)  # 处理NaN
        train_weights = 1.0 + return_ratio * 0.1  # 总是轻微惩罚return大的样本
        train_weights = np.clip(train_weights, 0.01, 10.0)  # 确保非负  # 总是轻微惩罚return大的样本

        train_weights = 1.0 + return_ratio * 0.1

        # 训练 CatBoost
        catboost = CatBoostBackend(
            iterations=500,
            learning_rate=0.05,
            depth=4,
            l2_leaf_reg=3.0,
            task_type="CPU",
            class_weights=[1.0, 0.5, 1.0],
            verbose=0,
        )
        catboost.train(X_train, y_train, sample_weight=train_weights)

        # 预测
        y_pred_train = catboost.predict(X_train)
        y_pred_valid = catboost.predict(X_valid)
        y_pred_test = catboost.predict(X_test)

        # 训练 LSTM（如果可用）
        lstm = None
        try:
            probabilities_train = catboost.predict_proba(X_train)
            lstm = LSTMResidualModel(seq_len=20, epochs=20, verbose=0)
            lstm.train(probabilities_train, y_train, X_train)
        except Exception as e:
            logger.debug(f"LSTM training skipped: {e}")

        # 计算指标
        metrics = calculate_conditional_metrics(y_test, y_pred_test)
        metrics["weighted_loss"] = calculate_weighted_loss(y_pred_test, y_test, ret_test, atr_test)
        metrics["test_accuracy"] = np.mean(y_pred_test == y_test)

        # Walk-Forward验证：训练集和验证集表现
        metrics["train_accuracy"] = np.mean(y_pred_train == y_train)
        metrics["valid_accuracy"] = np.mean(y_pred_valid == y_valid)

        # 条件概率：P(predict=c | actual=c)
        for c in [0, 1, 2]:
            mask = y_test == c
            if mask.sum() > 0:
                metrics[f"p_predict_{c}_given_actual_{c}"] = (y_pred_test[mask] == c).mean()
            else:
                metrics[f"p_predict_{c}_given_actual_{c}"] = 0.0

        # 条件概率：P(actual=c | predict=c)
        for c in [0, 1, 2]:
            mask = y_pred_test == c
            if mask.sum() > 0:
                metrics[f"p_actual_{c}_given_predict_{c}"] = (y_test[mask] == c).mean()
            else:
                metrics[f"p_actual_{c}_given_predict_{c}"] = 0.0

        result = {
            "window_start": start_idx,
            "train_size": train_end - start_idx,
            "valid_size": valid_end - train_end,
            "test_size": test_end - valid_end,
            **metrics,
        }
        results.append(result)

        logger.info(
            f"Window [{start_idx}-{test_end}]: "
            f"train_acc={metrics['train_accuracy']:.3f}, "
            f"valid_acc={metrics['valid_accuracy']:.3f}, "
            f"test_acc={metrics['test_accuracy']:.3f}"
        )

        start_idx += step

    return results


def calculate_conditional_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """计算条件概率"""
    metrics = {}

    # P(predict | actual)
    for c in [0, 1, 2]:
        mask = y_true == c
        if mask.sum() > 0:
            metrics[f"cond_prob_predict_given_actual_{c}"] = (y_pred[mask] == c).mean()
        else:
            metrics[f"cond_prob_predict_given_actual_{c}"] = 0.0

    # P(actual | predict)
    for c in [0, 1, 2]:
        mask = y_pred == c
        if mask.sum() > 0:
            metrics[f"cond_prob_actual_given_predict_{c}"] = (y_true[mask] == c).mean()
        else:
            metrics[f"cond_prob_actual_given_predict_{c}"] = 0.0

    return metrics


def calculate_weighted_loss(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    returns: np.ndarray,
    atr: np.ndarray,
) -> float:
    """
    计算加权Loss
    - 正确预测：Loss = -|return|/ATR
    - 错误预测：Loss = +|return|/ATR
    """
    errors = y_pred != y_true
    if errors.sum() == 0:
        return 0.0

    error_returns = np.abs(returns[errors])
    error_atr = atr[errors] + 1e-10
    weighted_losses = error_returns / error_atr

    # 正确预测部分（简化处理，用整体样本）
    correct_returns = np.abs(returns[~errors])
    correct_atr = atr[~errors] + 1e-10
    correct_losses = -correct_returns / correct_atr

    all_losses = np.concatenate([weighted_losses, correct_losses])
    return float(np.mean(all_losses))


async def main(days: int = 365, interval: str = "15m") -> None:
    """主函数"""
    symbol = "BTCUSDT"

logger.info(f"Fetching {days} days of {symbol} data at {interval}...")
    
    # 强制获取足够数据
    import time
    end_time_ms = int(time.time() * 1000)
    start_time_ms = end_time_ms - days * 24 * 3600 * 1000
    
    client = BinanceClient()
    klines = await client.fetch_klines_with_pagination(
        symbol, 
        interval, 
        start_time=start_time_ms,
        end_time=end_time_ms,
    )
    await client.close()
    logger.info(f"Fetched {len(klines)} klines")

    logger.info("Preparing data...")
    data = prepare_data(klines)

    logger.info(f"Data prepared: {len(data['features'])} samples")
    logger.info(f"Label distribution: {pd.Series(data['labels']).value_counts().sort_index().to_dict()}")

    logger.info(f"Running Walk-Forward with scale={NEUTRAL_SCALE}...")
    results = walkforward_train_validate(
        data,
        train_window=800,
        step=100,
    )

    if not results:
        logger.error("No valid windows")
        return

    # 汇总结果
    logger.info("\n" + "=" * 80)
    logger.info("WALK-FORWARD RESULTS")
    logger.info("=" * 80)

    avg_test_acc = np.mean([r["test_accuracy"] for r in results])
    avg_weighted_loss = np.mean([r["weighted_loss"] for r in results])
    avg_p_actual_given_predict_0 = np.mean([r.get("p_actual_0_given_predict_0", 0) for r in results])
    avg_p_actual_given_predict_1 = np.mean([r.get("p_actual_1_given_predict_1", 0) for r in results])
    avg_p_actual_given_predict_2 = np.mean([r.get("p_actual_2_given_predict_2", 0) for r in results])

    logger.info(f"Average Test Accuracy: {avg_test_acc:.4f}")
    logger.info(f"Average Weighted Loss: {avg_weighted_loss:.4f}")
    logger.info(f"Average P(actual=L | predict=L): {avg_p_actual_given_predict_0:.4f}")
    logger.info(f"Average P(actual=N | predict=N): {avg_p_actual_given_predict_1:.4f}")
    logger.info(f"Average P(actual=S | predict=S): {avg_p_actual_given_predict_2:.4f}")

    # 逐窗口打印
    logger.info("\n" + "-" * 80)
    logger.info(f"{'Window':<10} {'Train':<8} {'TestAcc':<10} {'WL':<10} {'P(L|L)':<10} {'P(N|N)':<10} {'P(S|S)':<10}")
    logger.info("-" * 80)

    for r in results:
        logger.info(
            f"{r['window_start']:<10} "
            f"{r['train_size']:<8} "
            f"{r['test_accuracy']:<10.4f} "
            f"{r['weighted_loss']:<10.4f} "
            f"{r.get('p_actual_0_given_predict_0', 0):<10.4f} "
            f"{r.get('p_actual_1_given_predict_1', 0):<10.4f} "
            f"{r.get('p_actual_2_given_predict_2', 0):<10.4f}"
        )

    # 保存结果
    import json

    output_path = Path(f"models/walkforward_results_{symbol}_{interval}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(
            {
                "config": {
                    "neutral_scale": NEUTRAL_SCALE,
                    "lookforward_bars": LOOKFORWARD_BARS,
                    "train_window": 800,
                    "step": 100,
                    "days": days,
                    "interval": interval,
                },
                "summary": {
                    "avg_test_accuracy": avg_test_acc,
                    "avg_weighted_loss": avg_weighted_loss,
                    "avg_p_actual_given_predict_L": avg_p_actual_given_predict_0,
                    "avg_p_actual_given_predict_N": avg_p_actual_given_predict_1,
                    "avg_p_actual_given_predict_S": avg_p_actual_given_predict_2,
                },
                "windows": results,
            },
            f,
            indent=2,
        )

    logger.info(f"\nResults saved to {output_path}")

    # 绘制结果
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        windows = [r["window_start"] for r in results]

        ax1 = axes[0, 0]
        ax1.plot(windows, [r["test_accuracy"] for r in results], "r-o", label="Test", markersize=6)
        ax1.plot(windows, [r["valid_accuracy"] for r in results], "g-s", label="Valid", markersize=6)
        ax1.plot(windows, [r["train_accuracy"] for r in results], "b-^", label="Train", markersize=6)
        ax1.set_xlabel("Window Start")
        ax1.set_ylabel("Accuracy")
        ax1.set_title("Walk-Forward Accuracy")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = axes[0, 1]
        ax2.plot(windows, [r["weighted_loss"] for r in results], "r-o", markersize=6)
        ax2.set_xlabel("Window Start")
        ax2.set_ylabel("Weighted Loss")
        ax2.set_title("Walk-Forward Weighted Loss")
        ax2.grid(True, alpha=0.3)

        ax3 = axes[1, 0]
        ax3.plot(windows, [r.get("p_actual_0_given_predict_0", 0) for r in results], "b-o", label="L", markersize=6)
        ax3.plot(windows, [r.get("p_actual_1_given_predict_1", 0) for r in results], "g-s", label="N", markersize=6)
        ax3.plot(windows, [r.get("p_actual_2_given_predict_2", 0) for r in results], "r-^", label="S", markersize=6)
        ax3.set_xlabel("Window Start")
        ax3.set_ylabel("P(actual | predict)")
        ax3.set_title("P(actual=L/N/S | predict=L/N/S)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        ax4 = axes[1, 1]
        ax4.plot(
            windows, [r.get("cond_prob_predict_given_actual_0", 0) for r in results], "b-o", label="L", markersize=6
        )
        ax4.plot(
            windows, [r.get("cond_prob_predict_given_actual_1", 0) for r in results], "g-s", label="N", markersize=6
        )
        ax4.plot(
            windows, [r.get("cond_prob_predict_given_actual_2", 0) for r in results], "r-^", label="S", markersize=6
        )
        ax4.set_xlabel("Window Start")
        ax4.set_ylabel("P(predict | actual)")
        ax4.set_title("P(predict=L/N/S | actual=L/N/S)")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = output_path.with_suffix(".png")
        plt.savefig(plot_path, dpi=150)
        logger.info(f"Plot saved to {plot_path}")

    except ImportError:
        logger.warning("matplotlib not available")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=365, help="Days of data to fetch")
    parser.add_argument("--interval", type=str, default="15m", help="Kline interval")
    args = parser.parse_args()

    asyncio.run(main(days=args.days, interval=args.interval))
