"""
标注器抽象基类

定义标签生成的接口规范
"""

from abc import ABC, abstractmethod

import pandas as pd


class Labeler(ABC):
    """
    标注器抽象基类

    所有标注器必须实现此接口
    """

    @abstractmethod
    def label(self, **kwargs) -> pd.Series:
        """
        生成标签

        Returns:
            标签 Series，index 与输入数据一致
        """
        raise NotImplementedError

    @abstractmethod
    def get_label_names(self) -> list[str]:
        """
        获取标签名称列表

        Returns:
            标签名称列表
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def n_classes(self) -> int:
        """类别数量"""
        raise NotImplementedError
