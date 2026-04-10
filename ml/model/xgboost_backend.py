"""
XGBoost 模型后端

备选模型
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class XGBoostBackend:
    """
    XGBoost 模型后端

    备选模型，支持 GPU 加速
    """

    def __init__(
        self,
        n_estimators: int = 2000,
        learning_rate: float = 0.03,
        max_depth: int = 6,
        device: str = "cuda",
        class_weights: list[float] | None = None,
        verbosity: int = 0,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.device = device
        self.class_weights = class_weights or [1.0, 0.5, 1.0]
        self.verbosity = verbosity

        self.model: Any = None
        self._is_trained = False

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: pd.Series | None = None,
    ) -> None:
        """
        训练 XGBoost 模型

        Args:
            X: 特征 DataFrame
            y: 标签 Series (0=DOWN, 1=NEUTRAL, 2=UP)
            sample_weight: 样本权重
        """
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost is required. Install with: uv add xgboost")

        params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "device": self.device,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "verbosity": self.verbosity,
        }

        if sample_weight is not None:
            dtrain = xgb.DMatrix(X, label=y, weight=sample_weight)
        else:
            dtrain = xgb.DMatrix(X, label=y)

        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
        )
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
        import xgboost as xgb

        dtest = xgb.DMatrix(X)
        proba = self.model.predict(dtest)
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
        import xgboost as xgb

        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)

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
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost is required. Install with: uv add xgboost")

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        self.model = xgb.Booster()
        self.model.load_model(str(path))
        self._is_trained = True

    @property
    def model_name(self) -> str:
        """模型名称"""
        return "XGBoost"

    @property
    def is_trained(self) -> bool:
        """是否已训练"""
        return self._is_trained
