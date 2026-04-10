"""
ML 信号策略选择器

封装 ML 模型加载和预测，提供统一的信号接口
"""

import logging
from pathlib import Path
from typing import Any

from ml.features.single_pair import SinglePairFeatureGenerator
from ml.model.catboost_backend import CatBoostBackend
from ml.model.lstm_residual import LSTMResidualModel
from ml_common import MLConfig, MLSignal, SignalLabel

logger = logging.getLogger(__name__)


class MLSignalStrategy:
    """
    ML 信号策略

    封装 ML 模型加载和预测：
    - 加载预训练的 CatBoost 主模型
    - 加载 LSTM 残差修正模型（可选）
    - 生成特征并进行预测
    """

    def __init__(self, config: MLConfig | None = None):
        self.config = config or MLConfig()
        self.catboost: CatBoostBackend | None = None
        self.lstm: LSTMResidualModel | None = None
        self.feature_gen = SinglePairFeatureGenerator()
        self._is_loaded = False

    def load_models(self, model_path: str | Path) -> bool:
        """
        加载预训练模型

        Args:
            model_path: 模型目录路径

        Returns:
            是否加载成功
        """
        model_path = Path(model_path)

        catboost_path = model_path / "catboost_model.cbm"
        if catboost_path.exists():
            try:
                self.catboost = CatBoostBackend()
                self.catboost.load(catboost_path)
                logger.info(f"CatBoost model loaded from {catboost_path}")
            except Exception:
                logger.exception("Failed to load CatBoost model")
                self.catboost = None

        lstm_path = model_path / "lstm_residual.keras"
        if lstm_path.exists() and self.config.use_lstm_residual:
            try:
                self.lstm = LSTMResidualModel()
                self.lstm.load(lstm_path)
                logger.info(f"LSTM model loaded from {lstm_path}")
            except Exception as e:
                logger.warning(f"Failed to load LSTM model: {e}")
                self.lstm = None

        self._is_loaded = self.catboost is not None
        return self._is_loaded

    def is_ready(self) -> bool:
        """检查模型是否已加载"""
        return self._is_loaded

    def predict(self, klines: list[Any]) -> MLSignal | None:
        """
        对 K 线数据进行预测

        Args:
            klines: K 线列表

        Returns:
            MLSignal 或 None（如果预测失败）
        """
        if not self.is_ready():
            logger.warning("ML models not loaded. Call load_models() first.")
            return None

        if len(klines) < 50:
            logger.warning(f"Insufficient klines for prediction: {len(klines)}")
            return None

        try:
            features = self.feature_gen.generate(klines)

            if features.empty or len(features) == 0:
                logger.warning("Feature generation returned empty DataFrame")
                return None

            x = features.iloc[-1:]

            y_pred = self.catboost.predict(x)[0]
            y_proba = self.catboost.predict_proba(x)[0]

            if self.lstm is not None and self.lstm.is_trained:
                y_proba = self.lstm.correct(y_pred, y_proba)

            label = int(y_pred)
            signal_type = SignalLabel.to_signal(label)

            symbol = klines[-1].symbol if hasattr(klines[-1], "symbol") else "UNKNOWN"
            price = float(klines[-1].close) if hasattr(klines[-1], "close") else 0.0
            timestamp = int(klines[-1].open_time) if hasattr(klines[-1], "open_time") else 0
            atr_val = float(x["atr"].values[0]) if "atr" in x.columns else 0.0

            confidence = float(max(y_proba))
            probability = (float(y_proba[0]), float(y_proba[1]), float(y_proba[2]))

            return MLSignal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                probability=probability,
                timestamp=timestamp,
                price=price,
                atr=atr_val,
                label=label,
            )

        except Exception:
            logger.exception("Prediction failed")
            return None

    def predict_batch(self, klines: list[Any]) -> list[MLSignal]:
        """
        批量预测（用于回测）

        Args:
            klines: K 线列表

        Returns:
            MLSignal 列表
        """
        if not self.is_ready():
            logger.warning("ML models not loaded. Call load_models() first.")
            return []

        if len(klines) < 50:
            logger.warning(f"Insufficient klines for batch prediction: {len(klines)}")
            return []

        try:
            features = self.feature_gen.generate(klines)

            if features.empty:
                return []

            x = features

            y_pred = self.catboost.predict(x)
            y_proba = self.catboost.predict_proba(x)

            if self.lstm is not None and self.lstm.is_trained:
                y_proba = self.lstm.correct_batch(y_pred, y_proba)

            signals = []
            for i in range(len(y_pred)):
                label = int(y_pred[i])
                signal_type = SignalLabel.to_signal(label)
                confidence = float(max(y_proba[i]))
                probability = (float(y_proba[i][0]), float(y_proba[i][1]), float(y_proba[i][2]))

                price = float(klines[i + 50].close) if i + 50 < len(klines) else 0.0
                timestamp = int(klines[i + 50].open_time) if i + 50 < len(klines) else 0
                atr_val = float(x.iloc[i]["atr"]) if "atr" in x.columns else 0.0

                signals.append(
                    MLSignal(
                        symbol=klines[i].symbol if hasattr(klines[i], "symbol") else "UNKNOWN",
                        signal_type=signal_type,
                        confidence=confidence,
                        probability=probability,
                        timestamp=timestamp,
                        price=price,
                        atr=atr_val,
                        label=label,
                    )
                )

            return signals

        except Exception:
            logger.exception("Batch prediction failed")
            return []

    @property
    def model_info(self) -> dict[str, Any]:
        """获取模型信息"""
        info: dict[str, Any] = {
            "loaded": self._is_loaded,
            "catboost_loaded": self.catboost is not None,
            "lstm_loaded": self.lstm is not None,
            "config": {
                "neutral_scale": self.config.neutral_scale,
                "lookforward_bars": self.config.lookforward_bars,
                "use_lstm_residual": self.config.use_lstm_residual,
            },
        }

        if self.catboost is not None:
            info["catboost"] = {
                "model_name": self.catboost.model_name,
                "is_trained": self.catboost.is_trained,
            }

        if self.lstm is not None:
            info["lstm"] = {
                "is_trained": self.lstm.is_trained,
            }

        return info
