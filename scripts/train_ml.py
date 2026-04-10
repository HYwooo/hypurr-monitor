"""
ML 模型训练脚本

Usage:
    uv run python scripts/train_ml.py                 # 训练所有配对
    uv run python scripts/train_ml.py --pair BTC-ETH  # 训练单个配对
    uv run python scripts/train_ml.py --fetch-data    # 先获取数据再训练
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.binance_fetcher import BinanceFetcher
from data.pair_registry import PairRegistry, TradingPair
from data.storage import DataStorage
from ml.features.single_pair import SinglePairFeatureGenerator
from ml.labels.three_class import ThreeClassLabeler
from ml.model.catboost_backend import CatBoostBackend
from ml.model.lstm_residual import LSTMResidualModel
from ml_common import MLConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


async def fetch_data(pair: TradingPair, days: int = 730, interval: str = "15m") -> dict[str, list]:
    """获取配对数据"""
    logger.info(f"Fetching data for {pair.pair_name}...")
    fetcher = BinanceFetcher()
    try:
        return await fetcher.fetch_pair_klines(
            pair=pair,
            interval=interval,
            days=days,
            force_update=False,
        )
    finally:
        await fetcher.close()


def load_local_data(pair: TradingPair, interval: str = "15m") -> dict[str, list]:
    """从本地加载配对数据"""
    storage = DataStorage()
    result = {}
    for symbol in [pair.symbol_a, pair.symbol_b]:
        klines = storage.load_klines(symbol, interval)
        if klines:
            result[symbol] = klines
            logger.info(f"Loaded {len(klines)} klines for {symbol}")
    return result


def prepare_features(
    klines_a: list,
    klines_b: list | None = None,
    single_gen: SinglePairFeatureGenerator | None = None,
) -> tuple:
    """准备特征"""
    if single_gen is None:
        single_gen = SinglePairFeatureGenerator()

    features_a = single_gen.generate(klines_a)

    if klines_b is None:
        return features_a

    features_b = single_gen.generate(klines_b)

    min_len = min(len(features_a), len(features_b))
    features_a = features_a.iloc[-min_len:].reset_index(drop=True)
    features_b = features_b.iloc[-min_len:].reset_index(drop=True)

    return features_a, features_b


def create_labels(
    close_prices: list,
    labeler: ThreeClassLabeler | None = None,
) -> tuple:
    """创建标签"""
    if labeler is None:
        labeler = ThreeClassLabeler()

    import numpy as np

    closes = np.array([k.close for k in close_prices])

    import talib

    atr_values = talib.ATR(
        np.array([k.high for k in close_prices]),
        np.array([k.low for k in close_prices]),
        closes,
        14,
    )

    return labeler.label(closes, atr_values)


def train_catboost(x_train, y_train) -> CatBoostBackend:
    """训练 CatBoost 模型"""
    logger.info("Training CatBoost model...")

    model = CatBoostBackend(
        iterations=2000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        task_type="GPU",
        devices="0",
        class_weights=[1.0, 0.5, 1.0],
        verbose=100,
    )

    model.train(x_train, y_train)

    return model


def train_lstm(
    model: CatBoostBackend,
    x_train,
    y_train,
    x_valid,
    y_valid,
) -> LSTMResidualModel | None:
    """训练 LSTM 残差模型"""
    try:
        logger.info("Training LSTM residual model...")

        import numpy as np

        train_preds = model.predict(x_train)
        train_proba = model.predict_proba(x_train)

        valid_preds = model.predict(x_valid)
        valid_proba = model.predict_proba(x_valid)

        actuals_train = y_train.values
        actuals_valid = y_valid.values

        lstm = LSTMResidualModel(
            seq_len=10,
            hidden_size=64,
            epochs=50,
            batch_size=32,
        )

        lstm.train(
            predictions=np.concatenate([train_preds, valid_preds]),
            actuals=np.concatenate([actuals_train, actuals_valid]),
            probabilities=np.concatenate([train_proba, valid_proba]),
        )

        return lstm
    except Exception:
        logger.warning("LSTM training failed")
        return None


def evaluate_model(
    model: CatBoostBackend,
    x_test,
    y_test,
) -> dict:
    """评估模型"""
    y_pred = model.predict(x_test)

    from sklearn.metrics import classification_report, precision_recall_fscore_support

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test.values, y_pred, average=None, labels=[0, 1, 2]
    )

    report = classification_report(y_test.values, y_pred, labels=[0, 1, 2])

    logger.info(f"\nClassification Report:\n{report}")

    return {
        "precision_down": precision[0],
        "precision_neutral": precision[1],
        "precision_up": precision[2],
        "recall_down": recall[0],
        "recall_neutral": recall[1],
        "recall_up": recall[2],
        "f1_down": f1[0],
        "f1_neutral": f1[1],
        "f1_up": f1[2],
        "support_down": int(support[0]),
        "support_neutral": int(support[1]),
        "support_up": int(support[2]),
    }


def save_models(
    catboost: CatBoostBackend,
    lstm: LSTMResidualModel | None,
    pair_name: str,
    model_dir: Path,
) -> None:
    """保存模型"""
    pair_dir = model_dir / pair_name.replace("-", "_").replace(":", "_")
    pair_dir.mkdir(parents=True, exist_ok=True)

    catboost.save(pair_dir / "catboost_model.cbm")
    logger.info(f"CatBoost model saved to {pair_dir}")

    if lstm is not None:
        lstm.save(pair_dir / "lstm_residual.keras")
        logger.info(f"LSTM model saved to {pair_dir}")


async def train_pair(
    pair: TradingPair,
    config: MLConfig,
    model_dir: Path,
    fetch: bool = False,
    interval: str = "15m",
    days: int = 730,
) -> dict:
    """训练单个配对"""
    logger.info("=" * 60)
    logger.info("Training models for %s", pair.pair_name)
    logger.info("=" * 60)

    start_time = time.time()

    if fetch:
        klines_data = await fetch_data(pair, days=days, interval=interval)
    else:
        klines_data = load_local_data(pair, interval=interval)

    if pair.symbol_a not in klines_data or pair.symbol_b not in klines_data:
        logger.error("Missing data for %s", pair.pair_name)
        return {"pair": pair.pair_name, "status": "failed", "reason": "missing_data"}

    klines_a = klines_data[pair.symbol_a]
    klines_b = klines_data[pair.symbol_b]

    logger.info("Loaded %d klines for %s", len(klines_a), pair.symbol_a)
    logger.info("Loaded %d klines for %s", len(klines_b), pair.symbol_b)

    single_gen = SinglePairFeatureGenerator()
    features_a, features_b = prepare_features(klines_a, klines_b, single_gen)

    logger.info("Generated %d features", features_a.shape[1])

    labeler = ThreeClassLabeler(
        neutral_scale=config.neutral_scale,
        lookforward_bars=config.lookforward_bars,
    )

    import numpy as np

    closes_a = np.array([k.close for k in klines_a])
    closes_b = np.array([k.close for k in klines_b])

    min_len = min(len(closes_a), len(closes_b))
    closes_a = closes_a[-min_len:]
    closes_b = closes_b[-min_len:]

    import talib

    atr_a = talib.ATR(
        np.array([k.high for k in klines_a[-min_len:]]),
        np.array([k.low for k in klines_a[-min_len:]]),
        closes_a,
        14,
    )

    labels_a = labeler.label(closes_a, atr_a)

    feat_start_idx = features_a.index.min()
    labels_end_idx = labels_a.index.max()
    align_start = max(feat_start_idx, labels_a.index.min())
    align_end = min(features_a.index.max(), labels_end_idx)

    if align_start > align_end:
        logger.error(
            "No overlapping data: features [%d-%d], labels [%d-%d]",
            feat_start_idx,
            features_a.index.max(),
            labels_a.index.min(),
            labels_end_idx,
        )
        return {"pair": pair.pair_name, "status": "failed", "reason": "no_alignment"}

    combined_features = features_a.loc[align_start:align_end].reset_index(drop=True)
    labels_a = labels_a.loc[align_start:align_end].reset_index(drop=True)

    min_samples = 50
    if len(combined_features) < min_samples:
        logger.error("Not enough samples: %d", len(combined_features))
        return {"pair": pair.pair_name, "status": "failed", "reason": "insufficient_samples"}

    n = len(combined_features)
    train_end = int(n * config.train_ratio)
    valid_end = int(n * (config.train_ratio + config.valid_ratio))

    x_train = combined_features.iloc[:train_end]
    x_valid = combined_features.iloc[train_end:valid_end]
    x_test = combined_features.iloc[valid_end:]

    y_train = labels_a.iloc[:train_end]
    y_valid = labels_a.iloc[train_end:valid_end]
    y_test = labels_a.iloc[valid_end:]

    logger.info("Train: %d, Valid: %d, Test: %d", len(x_train), len(x_valid), len(x_test))

    catboost = train_catboost(x_train, y_train)

    lstm = None
    if config.use_lstm_residual:
        lstm = train_lstm(catboost, x_train, y_train, x_valid, y_valid)

    metrics = evaluate_model(catboost, x_test, y_test)

    save_models(catboost, lstm, pair.pair_name, model_dir)

    elapsed = time.time() - start_time
    logger.info(f"Training completed in {elapsed:.1f}s")

    return {
        "pair": pair.pair_name,
        "status": "success",
        "elapsed": elapsed,
        "metrics": metrics,
    }


async def async_main(args: argparse.Namespace) -> None:
    """异步主函数"""
    config = MLConfig(
        neutral_scale=0.5,
        lookforward_bars=1,
        train_ratio=0.7,
        valid_ratio=0.15,
        use_lstm_residual=True,
        primary_model="catboost",
    )

    model_dir = Path("models/ml")
    model_dir.mkdir(parents=True, exist_ok=True)

    if args.pair:
        pair = PairRegistry.get_pair(args.pair)
        if pair is None:
            logger.error(f"Unknown pair: {args.pair}")
            sys.exit(1)
        pairs = [pair]
    else:
        pairs = PairRegistry.get_pairs_by_exchange("binance")
        logger.info(f"Training {len(pairs)} pairs: {[p.pair_name for p in pairs]}")

    results = []
    for pair in pairs:
        try:
            result = await train_pair(
                pair=pair,
                config=config,
                model_dir=model_dir,
                fetch=args.fetch_data,
                interval=args.interval,
                days=args.days,
            )
            results.append(result)
        except Exception:
            logger.exception("Training failed for %s", pair.pair_name)
            results.append({"pair": pair.pair_name, "status": "failed", "reason": "error"})

    logger.info("=" * 60)
    logger.info("Training Summary")
    logger.info("=" * 60)
    for r in results:
        status = r.get("status", "unknown")
        elapsed = r.get("elapsed", 0)
        logger.info("%s: %s (%.1fs)", r["pair"], status, elapsed)


def main() -> None:
    parser = argparse.ArgumentParser(description="ML Model Training")
    parser.add_argument("--pair", type=str, help="Specific pair to train (e.g., BTC-ETH)")
    parser.add_argument("--fetch-data", action="store_true", help="Fetch data before training")
    parser.add_argument("--interval", type=str, default="15m", help="Kline interval (default: 15m)")
    parser.add_argument("--days", type=int, default=730, help="History days (default: 730)")

    args = parser.parse_args()

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
