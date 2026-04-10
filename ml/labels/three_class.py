"""
三分类标注器

基于 ATR 的涨/跌/平标注

平的定义：|return| <= ATR * neutral_scale
"""

from typing import Any

import numpy as np
import pandas as pd
import talib

from ml_common import LabelConstants, SignalType


class ThreeClassLabeler:
    """
    三分类标注器

    标注规则：
    - UP (2):   return >  atr * neutral_scale
    - DOWN (0): return < -atr * neutral_scale
    - NEUTRAL (1): |return| <= atr * neutral_scale
    """

    def __init__(
        self,
        neutral_scale: float = 0.5,
        lookforward_bars: int = 1,
    ):
        self.neutral_scale = neutral_scale
        self.lookforward_bars = lookforward_bars

    def label(
        self,
        close_prices: np.ndarray,
        atr: np.ndarray,
    ) -> pd.Series:
        """
        生成三分类标签

        Args:
            close_prices: 收盘价数组
            atr: ATR 数组

        Returns:
            标签 Series:
            - 2 (UP)
            - 1 (NEUTRAL)
            - 0 (DOWN)
        """
        if len(close_prices) != len(atr):
            raise ValueError("close_prices and atr must have same length")

        if len(close_prices) <= self.lookforward_bars:
            raise ValueError(f"Insufficient data: {len(close_prices)}, need > {self.lookforward_bars}")

        future_close = close_prices[self.lookforward_bars :]
        current_close = close_prices[: -self.lookforward_bars]
        future_atr = atr[: -self.lookforward_bars]

        future_return = (future_close - current_close) / current_close

        labels = np.full(len(future_return), LabelConstants.NEUTRAL_LABEL)

        threshold = (future_atr / current_close) * self.neutral_scale

        labels[future_return > threshold] = LabelConstants.UP_LABEL
        labels[future_return < -threshold] = LabelConstants.DOWN_LABEL

        return pd.Series(labels, index=range(len(labels)))

    def label_with_atr(
        self,
        klines: list[Any],
        neutral_scale: float | None = None,
    ) -> pd.Series:
        """
        从 K 线数据生成标签（自动计算 ATR）

        Args:
            klines: K 线列表
            neutral_scale: 覆盖默认的 neutral_scale

        Returns:
            标签 Series
        """
        scale = neutral_scale or self.neutral_scale

        closes = np.array([k.close for k in klines])
        highs = np.array([k.high for k in klines])
        lows = np.array([k.low for k in klines])

        atr_raw = talib.ATR(highs, lows, closes, 14)

        return self.label(closes, atr_raw)

    def get_label_names(self) -> list[str]:
        """获取标签名称"""
        return ["DOWN", "NEUTRAL", "UP"]

    @property
    def n_classes(self) -> int:
        """类别数量"""
        return 3

    def signal_to_label(self, signal: SignalType) -> int:
        """SignalType 转标签"""
        return LabelConstants.from_signal(signal)

    def label_to_signal(self, label: int) -> SignalType:
        """标签转 SignalType"""
        return LabelConstants.to_signal(label)
