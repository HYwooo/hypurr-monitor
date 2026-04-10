"""
Scale Experiment Script

多scale训练实验，找出最优neutral_scale
目标：
1. P(predict=L/N/S at t | actual=L/N/S at t+1) - 预测准确率
2. P(predict=L/N/S at t | actual=L/N/S at t) - 实际准确率
3. 加权错误Loss = |price_change| / ATR（降低尾部风险）

Usage:
    uv run python scripts/scale_experiment.py
"""

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import talib

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.storage import DataStorage
from ml.features.single_pair import SinglePairFeatureGenerator
from ml.labels.three_class import ThreeClassLabeler
from ml.model.catboost_backend import CatBoostBackend

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

SCALES = [round(0.1 + i * 0.02, 2) for i in range(11)]  # [0.1, 0.12, 0.14, ..., 0.3]
LOOKFORWARD_BARS = 1


def load_and_prepare_data(symbol: str = "BTCUSDT", interval: str = "15m") -> dict[str, Any]:
    """加载并准备数据"""
    storage = DataStorage()
    klines = storage.load_klines(symbol, interval)

    if len(klines) < 100:
        raise ValueError(f"Insufficient data: {len(klines)} klines")

    closes = np.array([k.close for k in klines])
    highs = np.array([k.high for k in klines])
    lows = np.array([k.low for k in klines])

    single_gen = SinglePairFeatureGenerator()
    features = single_gen.generate(klines)
    atr = talib.ATR(highs, lows, closes, 14)

    return {
        "closes": closes,
        "highs": highs,
        "lows": lows,
        "features": features,
        "atr": atr,
    }


def prepare_aligned_data(
    data: dict[str, Any],
    neutral_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """对齐特征、标签、收益、ATR"""
    closes = data["closes"]
    atr_full = data["atr"]
    features_full = data["features"]

    labeler = ThreeClassLabeler(neutral_scale=neutral_scale, lookforward_bars=LOOKFORWARD_BARS)
    labels = labeler.label(closes, atr_full)

    # 计算未来收益
    future_close = closes[LOOKFORWARD_BARS:]
    current_close = closes[:-LOOKFORWARD_BARS]
    actual_returns = (future_close - current_close) / current_close

    # 对齐 - 使用原始索引
    feat_start_idx = features_full.index.min()
    labels_end_idx = labels.index.max()
    align_start = max(feat_start_idx, labels.index.min())
    align_end = min(features_full.index.max(), labels_end_idx)

    if align_start > align_end:
        raise ValueError("No overlapping data")

    feat_start = align_start - features_full.index.min()
    feat_end = align_end - features_full.index.min() + 1

    labels_aligned = labels.loc[align_start:align_end].reset_index(drop=True)
    actual_returns_aligned = actual_returns[feat_start:feat_end]
    atr_aligned = atr_full[feat_start:feat_end]
    features_aligned = features_full.loc[align_start:align_end].reset_index(drop=True)

    return labels_aligned.values, actual_returns_aligned, atr_aligned, features_aligned


def train_and_evaluate(
    features: pd.DataFrame,
    labels: np.ndarray,
    actual_returns: np.ndarray,
    atr: np.ndarray,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
) -> dict[str, Any]:
    """训练并评估"""
    n = len(features)
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))

    X_train = features.iloc[:train_end]
    X_valid = features.iloc[train_end:valid_end]
    X_test = features.iloc[valid_end:]

    y_train = labels[:train_end]
    y_valid = labels[train_end:valid_end]
    y_test = labels[valid_end:]

    n_test = len(y_test)
    actual_returns_test = actual_returns[valid_end - n_test : valid_end]
    atr_test = atr[valid_end - n_test : valid_end]

    if len(np.unique(y_train)) < 2:
        return {"error": "Only one class in training data"}

    model = CatBoostBackend(
        iterations=500,
        learning_rate=0.05,
        depth=4,
        l2_leaf_reg=3.0,
        task_type="CPU",
        class_weights=[1.0, 0.5, 1.0],
        verbose=0,
    )

    model.train(X_train, y_train)

    y_pred_test = model.predict(X_test)

    return {
        "y_test": y_test,
        "y_pred_test": y_pred_test,
        "actual_returns_test": actual_returns_test,
        "atr_test": atr_test,
        "test_acc": np.mean(y_pred_test == y_test),
    }


def calculate_metrics(result: dict[str, Any]) -> dict[str, float]:
    """计算条件概率和加权loss"""
    y_test = result["y_test"]
    y_pred_test = result["y_pred_test"]
    actual_returns = result["actual_returns_test"]
    atr = result["atr_test"]

    n_classes = 3
    metrics = {}

    # P(predict=c | actual=c) at t+1
    for c in range(n_classes):
        mask = y_test == c
        if mask.sum() > 0:
            metrics[f"p_predict_given_actual_{c}"] = (y_pred_test[mask] == c).mean()
        else:
            metrics[f"p_predict_given_actual_{c}"] = 0.0

    # P(actual=c | predict=c) at t
    for c in range(n_classes):
        mask = y_pred_test == c
        if mask.sum() > 0:
            metrics[f"p_actual_given_predict_{c}"] = (y_test[mask] == c).mean()
        else:
            metrics[f"p_actual_given_predict_{c}"] = 0.0

    # 加权错误Loss
    errors = y_pred_test != y_test
    if errors.sum() > 0:
        error_returns = np.abs(actual_returns[errors])
        error_atr = atr[errors]
        metrics["weighted_error_loss"] = np.mean(error_returns / (error_atr + 1e-10))
        metrics["mean_abs_return_on_error"] = np.mean(error_returns)
    else:
        metrics["weighted_error_loss"] = 0.0
        metrics["mean_abs_return_on_error"] = 0.0

    metrics["overall_accuracy"] = (y_pred_test == y_test).mean()

    return metrics


def run_experiment(scale: float, data: dict[str, Any]) -> dict[str, Any]:
    """运行单个实验"""
    try:
        labels, actual_returns, atr, features = prepare_aligned_data(data, scale)

        if len(features) < 100:
            return {"error": f"Insufficient aligned data: {len(features)}"}

        result = train_and_evaluate(features, labels, actual_returns, atr)

        if "error" in result:
            return result

        metrics = calculate_metrics(result)

        return {
            "neutral_scale": scale,
            "test_acc": result["test_acc"],
            **metrics,
        }

    except Exception as e:
        logger.error(f"Scale {scale} failed: {e}")
        return {"error": str(e), "neutral_scale": scale}


def main() -> None:
    """主函数"""
    logger.info("Loading data...")
    try:
        data = load_and_prepare_data("BTCUSDT", "15m")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    logger.info(f"Running experiments with scales: {SCALES}")

    results = []
    for scale in SCALES:
        logger.info(f"Testing scale={scale}...")
        result = run_experiment(scale, data)
        results.append(result)

        if "error" not in result:
            logger.info(
                f"  scale={scale}: acc={result['test_acc']:.4f}, "
                f"weighted_loss={result.get('weighted_error_loss', 0):.4f}"
            )
        else:
            logger.warning(f"  scale={scale}: ERROR - {result['error']}")

    # 打印结果
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT RESULTS")
    logger.info("=" * 80)
    logger.info(f"{'Scale':<8} {'TestAcc':<10} {'WL':<10} {'P(L|L)':<10} {'P(N|N)':<10} {'P(S|S)':<10}")
    logger.info("-" * 80)

    valid_results = [r for r in results if "error" not in r]

    for r in valid_results:
        logger.info(
            f"{r['neutral_scale']:<8.1f} "
            f"{r['test_acc']:<10.4f} "
            f"{r.get('weighted_error_loss', 0):<10.4f} "
            f"{r.get('p_predict_given_actual_0', 0):<10.4f} "
            f"{r.get('p_predict_given_actual_1', 0):<10.4f} "
            f"{r.get('p_predict_given_actual_2', 0):<10.4f}"
        )

    if valid_results:
        best_by_acc = max(valid_results, key=lambda x: x["test_acc"])
        best_by_loss = min(valid_results, key=lambda x: x.get("weighted_error_loss", 999))

        logger.info("\n" + "=" * 80)
        logger.info(f"Best by accuracy: scale={best_by_acc['neutral_scale']}, acc={best_by_acc['test_acc']:.4f}")
        logger.info(
            f"Best by weighted loss: scale={best_by_loss['neutral_scale']}, loss={best_by_loss.get('weighted_error_loss', 0):.4f}"
        )

        import json

        output_path = Path("models/scale_experiment_results.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(
                {
                    "scales_tested": SCALES,
                    "results": results,
                    "best_by_accuracy": best_by_acc,
                    "best_by_weighted_loss": best_by_loss,
                },
                f,
                indent=2,
            )

        logger.info(f"\nResults saved to {output_path}")

        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            scales = [r["neutral_scale"] for r in valid_results]

            ax1 = axes[0, 0]
            ax1.plot(scales, [r["test_acc"] for r in valid_results], "r-^", markersize=8)
            ax1.set_xlabel("Neutral Scale")
            ax1.set_ylabel("Accuracy")
            ax1.set_title("Test Accuracy vs Neutral Scale")
            ax1.grid(True, alpha=0.3)

            ax2 = axes[0, 1]
            ax2.plot(scales, [r.get("weighted_error_loss", 0) for r in valid_results], "r-o", markersize=8)
            ax2.set_xlabel("Neutral Scale")
            ax2.set_ylabel("Weighted Error Loss")
            ax2.set_title("Weighted Error Loss vs Neutral Scale")
            ax2.grid(True, alpha=0.3)

            ax3 = axes[1, 0]
            ax3.plot(
                scales, [r.get("p_predict_given_actual_0", 0) for r in valid_results], "b-o", label="L", markersize=8
            )
            ax3.plot(
                scales, [r.get("p_predict_given_actual_1", 0) for r in valid_results], "g-s", label="N", markersize=8
            )
            ax3.plot(
                scales, [r.get("p_predict_given_actual_2", 0) for r in valid_results], "r-^", label="S", markersize=8
            )
            ax3.set_xlabel("Neutral Scale")
            ax3.set_ylabel("P(predict | actual)")
            ax3.set_title("P(predict=L/N/S | actual=L/N/S at t+1)")
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            ax4 = axes[1, 1]
            ax4.plot(
                scales, [r.get("p_actual_given_predict_0", 0) for r in valid_results], "b-o", label="L", markersize=8
            )
            ax4.plot(
                scales, [r.get("p_actual_given_predict_1", 0) for r in valid_results], "g-s", label="N", markersize=8
            )
            ax4.plot(
                scales, [r.get("p_actual_given_predict_2", 0) for r in valid_results], "r-^", label="S", markersize=8
            )
            ax4.set_xlabel("Neutral Scale")
            ax4.set_ylabel("P(actual | predict)")
            ax4.set_title("P(actual=L/N/S | predict=L/N/S at t)")
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            plot_path = Path("models/scale_experiment.png")
            plt.savefig(plot_path, dpi=150)
            logger.info(f"Plot saved to {plot_path}")

        except ImportError:
            logger.warning("matplotlib not available")


if __name__ == "__main__":
    main()
