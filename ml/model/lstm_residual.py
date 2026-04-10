"""
LSTM 残差学习模块

学习 CatBoost 预测的残差，修正预测偏差
"""

from pathlib import Path
from typing import Any

import numpy as np


class LSTMResidualModel:
    """
    LSTM 残差学习模型

    学习 CatBoost 预测值与真实值之间的残差
    用于修正主模型的预测偏差
    """

    def __init__(
        self,
        seq_len: int = 10,
        hidden_size: int = 64,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
    ):
        """
        Args:
            seq_len: 序列长度（时间步数）
            hidden_size: LSTM 隐藏层大小
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
        """
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.model: Any = None
        self._is_trained = False
        self._scaler_mean: np.ndarray | None = None
        self._scaler_std: np.ndarray | None = None

    def _build_model(self) -> Any:
        """构建 LSTM 模型"""
        try:
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.models import Sequential
        except ImportError:
            raise ImportError("tensorflow is required. Install with: uv add tensorflow")

        model = Sequential(
            [
                LSTM(self.hidden_size, return_sequences=True, input_shape=(self.seq_len, 3)),
                Dropout(0.2),
                LSTM(self.hidden_size // 2, return_sequences=False),
                Dropout(0.2),
                Dense(32, activation="relu"),
                Dense(1),
            ]
        )

        model.compile(optimizer="adam", loss="mse")
        return model

    def _prepare_sequences(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        probabilities: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        准备训练序列

        构建滑动窗口序列：
        - pred_diff: 预测与实际的差异
        - prob_diff: 概率变化
        - volatility: 波动率

        Args:
            predictions: CatBoost 预测值
            actuals: 实际值
            probabilities: 预测概率

        Returns:
            X: 序列数据 [n_samples, seq_len, 3]
            y: 残差目标值
        """
        pred_diff = predictions - actuals

        if probabilities is not None:
            prob_up = probabilities[:, 2]
            prob_change = np.diff(prob_up, prepend=prob_up[0])
            volatility = np.abs(pred_diff)

            features = np.column_stack([pred_diff, prob_change, volatility])
        else:
            volatility = np.abs(pred_diff)
            features = np.column_stack([pred_diff, volatility, volatility])

        self._scaler_mean = np.mean(features[:100], axis=0) if len(features) > 100 else np.zeros(3)
        self._scaler_std = np.std(features[:100], axis=0) if len(features) > 100 else np.ones(3)
        self._scaler_std = np.where(self._scaler_std < 1e-6, 1.0, self._scaler_std)

        features_scaled = (features - self._scaler_mean) / self._scaler_std

        X = []
        y = []

        for i in range(self.seq_len, len(features_scaled)):
            X.append(features_scaled[i - self.seq_len : i])
            y.append(pred_diff[i])

        return np.array(X), np.array(y)

    def train(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        probabilities: np.ndarray | None = None,
    ) -> None:
        """
        训练 LSTM 残差模型

        Args:
            predictions: CatBoost 预测值
            actuals: 实际值
            probabilities: 预测概率
        """
        X, y = self._prepare_sequences(predictions, actuals, probabilities)

        if len(X) < 100:
            raise ValueError(f"Insufficient data for LSTM training: {len(X)} samples, need at least 100")

        self.model = self._build_model()

        self.model.fit(
            X,
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=1,
        )

        self._is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测残差

        Args:
            X: 输入序列 [batch, seq_len, 3]

        Returns:
            预测的残差值
        """
        if not self._is_trained:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X).flatten()

    def correct(
        self,
        predictions: np.ndarray,
        probabilities: np.ndarray,
    ) -> np.ndarray:
        """
        对 CatBoost 预测进行残差修正

        Args:
            predictions: CatBoost 预测值
            probabilities: 预测概率

        Returns:
            修正后的预测值
        """
        if not self._is_trained:
            return predictions

        prob_up = probabilities[:, 2]
        prob_change = np.diff(prob_up, prepend=prob_up[0])
        volatility = np.abs(np.diff(predictions, prepend=predictions[0]))

        features = np.column_stack([predictions, prob_change, volatility])
        features_scaled = (features - self._scaler_mean) / self._scaler_std

        X = []
        for i in range(self.seq_len, len(features_scaled)):
            X.append(features_scaled[i - self.seq_len : i])

        if len(X) == 0:
            return predictions

        residual_pred = self.predict(np.array(X))

        corrected = predictions[self.seq_len :].copy()
        corrected += residual_pred

        result = predictions.copy()
        result[self.seq_len :] = corrected

        return result

    def save(self, path: str | Path) -> None:
        """保存模型"""
        if not self._is_trained:
            raise ValueError("Model not trained. Call train() first.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))

        meta_path = path.with_suffix(".meta.npz")
        np.savez(
            meta_path,
            seq_len=self.seq_len,
            scaler_mean=self._scaler_mean,
            scaler_std=self._scaler_std,
        )

    def load(self, path: str | Path) -> None:
        """加载模型"""
        try:
            from tensorflow.keras.models import load_model
        except ImportError:
            raise ImportError("tensorflow is required. Install with: uv add tensorflow")

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        self.model = load_model(str(path))
        self._is_trained = True

        meta_path = path.with_suffix(".meta.npz")
        if meta_path.exists():
            meta = np.load(meta_path)
            self.seq_len = int(meta["seq_len"])
            self._scaler_mean = meta["scaler_mean"]
            self._scaler_std = meta["scaler_std"]

    @property
    def model_name(self) -> str:
        """模型名称"""
        return "LSTM-Residual"

    @property
    def is_trained(self) -> bool:
        """是否已训练"""
        return self._is_trained
