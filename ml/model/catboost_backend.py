"""
CatBoost 模型后端

GPU 加速的三分类模型
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class CatBoostBackend:
    """
    CatBoost 模型后端

    支持 GPU 加速的三分类模型
    """

    def __init__(
        self,
        iterations: int = 2000,
        learning_rate: float = 0.03,
        depth: int = 6,
        l2_leaf_reg: float = 3.0,
        task_type: str = "GPU",
        devices: str = "0",
        class_weights: list[float] | None = None,
        verbose: int = 100,
    ):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.task_type = task_type
        self.devices = devices
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
        训练 CatBoost 模型

        Args:
            X: 特征 DataFrame
            y: 标签 Series (0=DOWN, 1=NEUTRAL, 2=UP)
            sample_weight: 样本权重
        """
        try:
            from catboost import CatBoostClassifier
        except ImportError:
            raise ImportError("catboost is required. Install with: uv add catboost")

        self.model = CatBoostClassifier(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            loss_function="MultiClass",
            classes_count=3,
            task_type=self.task_type,
            devices=self.devices,
            class_weights=self.class_weights,
            verbose=self.verbose,
        )

        self.model.fit(
            X,
            y,
            sample_weight=sample_weight,
        )

        self._is_trained = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测类别

        Args:
            X: 特征 DataFrame

        Returns:
            预测类别数组 (0=DOWN, 1=NEUTRAL, 2=UP)
        """
        if not self._is_trained:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X).flatten()

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
        return self.model.predict_proba(X)

    def save(self, path: str | Path) -> None:
        """
        保存模型

        Args:
            path: 保存路径 (.cbm 文件)
        """
        if not self._is_trained:
            raise ValueError("Model not trained. Call train() first.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))

    def load(self, path: str | Path) -> None:
        """
        加载模型

        Args:
            path: 模型路径 (.cbm 文件)
        """
        try:
            from catboost import CatBoostClassifier
        except ImportError:
            raise ImportError("catboost is required. Install with: uv add catboost")

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        self.model = CatBoostClassifier()
        self.model.load_model(str(path))
        self._is_trained = True

    @property
    def model_name(self) -> str:
        """模型名称"""
        return "CatBoost"

    @property
    def is_trained(self) -> bool:
        """是否已训练"""
        return self._is_trained

    def get_feature_importance(self) -> np.ndarray:
        """获取特征重要性"""
        if not self._is_trained:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.get_feature_importance()
