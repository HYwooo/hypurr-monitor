"""
LightGBM 模型后端

备选模型
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class LightGBMBackend:
    """
    LightGBM 模型后端

    备选模型，支持 GPU 加速
    """

    def __init__(
        self,
        num_iterations: int = 2000,
        learning_rate: float = 0.03,
        max_depth: int = 6,
        num_leaves: int = 31,
        device: str = "gpu",
        class_weights: list[float] | None = None,
        verbose: int = -1,
    ):
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.device = device
        self.class_weights = class_weights or [1.0, 0.5, 1.0]
        self.verbose = verbose

        self.model: Any = None
        self._is_trained = False

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: pd.Series | None = None,
    ) -> None:
        """
        训练 LightGBM 模型

        Args:
            X: 特征 DataFrame
            y: 标签 Series (0=DOWN, 1=NEUTRAL, 2=UP)
            sample_weight: 样本权重
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm is required. Install with: uv add lightgbm")

        params = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "num_iterations": self.num_iterations,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "num_leaves": self.num_leaves,
            "device": self.device,
            "verbose": self.verbose,
            "class_weight": "balanced",
        }

        if sample_weight is not None:
            train_data = lgb.Dataset(X, label=y, weight=sample_weight)
        else:
            train_data = lgb.Dataset(X, label=y)

        self.model = lgb.train(params, train_data)
        self._is_trained = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测类别

        Args:
            X: 特征 DataFrame

        Returns:
            预测类别数组
        """
        if not self._is_trained:
            raise ValueError("Model not trained. Call train() first.")
        proba = self.model.predict(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测概率

        Args:
            X: 特征 DataFrame

        Returns:
            概率数组 [n_samples, 3]，顺序: [P(DOWN), P(NEUTRAL), P(UP)]
        """
        if not self._is_trained:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)

    def save(self, path: str | Path) -> None:
        """保存模型"""
        if not self._is_trained:
            raise ValueError("Model not trained. Call train() first.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))

    def load(self, path: str | Path) -> None:
        """加载模型"""
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm is required. Install with: uv add lightgbm")

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        self.model = lgb.Booster(model_file=str(path))
        self._is_trained = True

    @property
    def model_name(self) -> str:
        """模型名称"""
        return "LightGBM"

    @property
    def is_trained(self) -> bool:
        """是否已训练"""
        return self._is_trained
