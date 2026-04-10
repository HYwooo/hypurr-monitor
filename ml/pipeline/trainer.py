"""
ML 训练流水线

整合特征生成、标注、模型训练
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.pair_registry import TradingPair
from ml.features.multi_pair import MultiPairFeatureGenerator
from ml.features.single_pair import SinglePairFeatureGenerator
from ml.labels.three_class import ThreeClassLabeler
from ml.model.catboost_backend import CatBoostBackend
from ml.model.lstm_residual import LSTMResidualModel
from ml_common import MLConfig, ProbabilityMetrics

logger = logging.getLogger(__name__)


class MLTrainer:
    """
    ML 训练流水线

    完整流程：
    1. 加载 K 线数据
    2. 生成特征
    3. 生成标签
    4. 划分数据集
    5. 训练 CatBoost 主模型
    6. 训练 LSTM 残差模型（可选）
    7. 评估模型
    """

    def __init__(
        self,
        config: MLConfig | None = None,
        pair: TradingPair | None = None,
    ):
        self.config = config or MLConfig()
        self.pair = pair

        self.single_gen = SinglePairFeatureGenerator()
        self.multi_gen = MultiPairFeatureGenerator()
        self.labeler = ThreeClassLabeler(
            neutral_scale=self.config.neutral_scale,
            lookforward_bars=self.config.lookforward_bars,
        )

        self.catboost: CatBoostBackend | None = None
        self.lstm: LSTMResidualModel | None = None

        self.feature_df: pd.DataFrame | None = None
        self.label_series: pd.Series | None = None
        self.X_train: pd.DataFrame | None = None
        self.X_valid: pd.DataFrame | None = None
        self.X_test: pd.DataFrame | None = None
        self.y_train: pd.Series | None = None
        self.y_valid: pd.Series | None = None
        self.y_test: pd.Series | None = None

    def prepare_data(
        self,
        klines_a: list[Any],
        klines_b: list[Any] | None = None,
    ) -> pd.DataFrame:
        """
        准备训练数据

        Args:
            klines_a: 标的 A 的 K 线
            klines_b: 标的 B 的 K 线（配对交易时需要）

        Returns:
            特征 DataFrame
        """
        if klines_b is None:
            features_a = self.single_gen.generate(klines_a)
            return features_a

        features_a = self.single_gen.generate(klines_a)
        features_b = self.single_gen.generate(klines_b)
        features_pair = self.multi_gen.generate(klines_a, klines_b)

        min_len = min(len(features_a), len(features_pair))

        features_a = features_a.iloc[-min_len:].reset_index(drop=True)
        features_b = features_b.iloc[-min_len:].reset_index(drop=True)
        features_pair = features_pair.iloc[-min_len:].reset_index(drop=True)

        combined = pd.concat(
            [
                features_a.add_suffix("_a"),
                features_b.add_suffix("_b"),
                features_pair,
            ],
            axis=1,
        )

        self.feature_df = combined.dropna()

        return self.feature_df

    def create_labels(
        self,
        close_prices: np.ndarray,
        atr: np.ndarray,
    ) -> pd.Series:
        """
        创建标签

        Args:
            close_prices: 收盘价
            atr: ATR

        Returns:
            标签 Series
        """
        self.label_series = self.labeler.label(close_prices, atr)
        return self.label_series

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        划分训练/验证/测试集

        Returns:
            X_train, X_valid, X_test, y_train, y_valid, y_test
        """
        min_len = min(len(X), len(y))
        X = X.iloc[-min_len:].reset_index(drop=True)
        y = y.iloc[-min_len:].reset_index(drop=True)

        n = len(X)
        train_end = int(n * self.config.train_ratio)
        valid_end = int(n * (self.config.train_ratio + self.config.valid_ratio))

        self.X_train = X.iloc[:train_end]
        self.X_valid = X.iloc[train_end:valid_end]
        self.X_test = X.iloc[valid_end:]

        self.y_train = y.iloc[:train_end]
        self.y_valid = y.iloc[train_end:valid_end]
        self.y_test = y.iloc[valid_end:]

        return self.X_train, self.X_valid, self.X_test, self.y_train, self.y_valid, self.y_test

    def train_catboost(
        self,
        class_weights: list[float] | None = None,
    ) -> CatBoostBackend:
        """
        训练 CatBoost 主模型

        Args:
            class_weights: 类别权重

        Returns:
            训练好的 CatBoostBackend
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not prepared. Call split_data() first.")

        weights = class_weights or [1.0, 0.5, 1.0]

        self.catboost = CatBoostBackend(
            iterations=2000,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3.0,
            task_type="GPU",
            devices="0",
            class_weights=weights,
            verbose=100,
        )

        self.catboost.train(self.X_train, self.y_train)

        return self.catboost

    def train_lstm(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        probabilities: np.ndarray | None = None,
    ) -> LSTMResidualModel:
        """
        训练 LSTM 残差模型

        Args:
            predictions: CatBoost 预测值
            actuals: 实际值
            probabilities: 预测概率

        Returns:
            训练好的 LSTMResidualModel
        """
        self.lstm = LSTMResidualModel(
            seq_len=10,
            hidden_size=64,
            epochs=50,
            batch_size=32,
        )

        self.lstm.train(predictions, actuals, probabilities)

        return self.lstm

    def evaluate(self) -> ProbabilityMetrics:
        """
        评估模型

        Returns:
            概率指标
        """
        if self.X_test is None or self.y_test is None or self.catboost is None:
            raise ValueError("Model not trained. Call train_catboost() first.")

        y_pred = self.catboost.predict(self.X_test)
        y_proba = self.catboost.predict_proba(self.X_test)

        return self._compute_metrics(self.y_test.values, y_pred, y_proba)

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
    ) -> ProbabilityMetrics:
        """计算概率指标"""
        from sklearn.metrics import precision_recall_fscore_support

        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0, 1, 2])

        return ProbabilityMetrics(
            precision_down=precision[0],
            precision_neutral=precision[1],
            precision_up=precision[2],
            recall_down=recall[0],
            recall_neutral=recall[1],
            recall_up=recall[2],
            f1_down=f1[0],
            f1_neutral=f1[1],
            f1_up=f1[2],
            support_down=int(support[0]),
            support_neutral=int(support[1]),
            support_up=int(support[2]),
        )

    def save_models(self, path: str | Path) -> None:
        """
        保存模型

        Args:
            path: 保存目录
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self.catboost:
            self.catboost.save(path / "catboost_model.cbm")
            logger.info(f"CatBoost model saved to {path / 'catboost_model.cbm'}")

        if self.lstm:
            self.lstm.save(path / "lstm_residual.keras")
            logger.info(f"LSTM model saved to {path / 'lstm_residual.keras'}")

    def load_models(self, path: str | Path) -> None:
        """
        加载模型

        Args:
            path: 模型目录
        """
        path = Path(path)

        catboost_path = path / "catboost_model.cbm"
        if catboost_path.exists():
            self.catboost = CatBoostBackend()
            self.catboost.load(catboost_path)
            logger.info(f"CatBoost model loaded from {catboost_path}")

        lstm_path = path / "lstm_residual.keras"
        if lstm_path.exists():
            self.lstm = LSTMResidualModel()
            self.lstm.load(lstm_path)
            logger.info(f"LSTM model loaded from {lstm_path}")

    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        预测

        Args:
            X: 特征 DataFrame

        Returns:
            (预测类别, 预测概率)
        """
        if self.catboost is None:
            raise ValueError("Model not trained. Call train_catboost() first.")

        y_pred = self.catboost.predict(X)
        y_proba = self.catboost.predict_proba(X)

        if self.lstm and self.lstm.is_trained:
            y_proba = self.lstm.correct(y_pred, y_proba)

        return y_pred, y_proba
