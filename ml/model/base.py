"""
模型后端抽象基类

定义 ML 模型的接口规范
"""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd


class ModelBackend(ABC):
    """
    模型后端抽象基类

    所有模型后端必须实现此接口
    """

    @abstractmethod
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: pd.Series | None = None,
    ) -> None:
        """
        训练模型

        Args:
            X: 特征 DataFrame
            y: 标签 Series
            sample_weight: 样本权重
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测类别

        Args:
            X: 特征 DataFrame

        Returns:
            预测类别数组
        """
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测概率

        Args:
            X: 特征 DataFrame

        Returns:
            概率数组 [n_samples, n_classes]
            顺序: [P(DOWN), P(NEUTRAL), P(UP)]
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """
        保存模型

        Args:
            path: 保存路径
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """
        加载模型

        Args:
            path: 模型路径
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def model_name(self) -> str:
        """模型名称"""
        raise NotImplementedError
