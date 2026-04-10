"""
特征生成器抽象基类

定义特征生成的接口规范
"""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class FeatureGenerator(ABC):
    """
    特征生成器抽象基类

    所有特征生成器必须实现此接口
    """

    @abstractmethod
    def generate(self, **kwargs) -> pd.DataFrame:
        """
        生成特征 DataFrame

        Returns:
            特征 DataFrame，index 为时间戳
        """
        raise NotImplementedError

    @abstractmethod
    def get_feature_names(self) -> list[str]:
        """
        获取特征名称列表

        Returns:
            特征名称列表
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def n_features(self) -> int:
        """特征数量"""
        raise NotImplementedError


class SinglePairFeatureGenerator(FeatureGenerator):
    """
    单标的特征生成器

    为单个交易标的生成技术指标特征
    """

    @abstractmethod
    def generate(self, klines: list[Any]) -> pd.DataFrame:
        """为单标的生成特征"""
        raise NotImplementedError


class MultiPairFeatureGenerator(FeatureGenerator):
    """
    配对特征生成器

    为配对交易生成汇率、价差等相关特征
    """

    @abstractmethod
    def generate(self, klines_a: list[Any], klines_b: list[Any]) -> pd.DataFrame:
        """为配对生成特征"""
        raise NotImplementedError


class CrossMarketFeatureGenerator(FeatureGenerator):
    """
    跨市场特征生成器（预留接口）

    生成市场整体相关的特征
    """

    @abstractmethod
    def generate(self, **kwargs) -> pd.DataFrame:
        """
        生成跨市场特征

        包含：
        - BTC dominance
        - total market cap
        - funding rate correlation
        - open interest ratio

        Raises:
            NotImplementedError: 跨市场特征待实现
        """
        raise NotImplementedError("跨市场特征待实现")
