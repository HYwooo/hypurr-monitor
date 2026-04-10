"""
配对特征生成器

为配对交易生成汇率、价差等相关特征
"""

import math

import numpy as np
import pandas as pd
import talib

from models import Kline


class MultiPairFeatureGenerator:
    """
    配对交易特征生成器

    生成特征包括：
    - 比率: ratio = price_a / price_b
    - 价差: spread = ratio - ratio_sma
    - 比率收益: ratio_return
    - 比率ATR: ratio_atr
    - 协整度: hurst_exponent, half_life
    """

    def __init__(
        self,
        ratio_sma_period: int = 20,
        ratio_atr_period: int = 14,
        lookback_periods: list[int] = [5, 10, 20, 50],
    ):
        self.ratio_sma_period = ratio_sma_period
        self.ratio_atr_period = ratio_atr_period
        self.lookback_periods = lookback_periods

    def generate(self, klines_a: list[Kline], klines_b: list[Kline]) -> pd.DataFrame:
        """
        为配对生成特征

        Args:
            klines_a: 标的 A 的 K 线
            klines_b: 标的 B 的 K 线

        Returns:
            配对特征 DataFrame
        """
        if len(klines_a) < 50 or len(klines_b) < 50:
            raise ValueError(f"Insufficient klines: {len(klines_a)}, {len(klines_b)}")

        df_a = self._klines_to_df(klines_a)
        df_b = self._klines_to_df(klines_b)

        merged = pd.merge(df_a, df_b, on="open_time", suffixes=("_a", "_b"))
        merged = merged.sort_values("open_time").reset_index(drop=True)

        features = pd.DataFrame(index=merged.index)
        features["open_time"] = merged["open_time"]

        close_a = merged["close_a"].values
        close_b = merged["close_b"].values
        high_a = merged["high_a"].values
        high_b = merged["high_b"].values
        low_a = merged["low_a"].values
        low_b = merged["low_b"].values
        volume_a = merged["volume_a"].values
        volume_b = merged["volume_b"].values

        ratio = close_a / close_b
        features["ratio"] = ratio
        features["ratio_return"] = np.diff(ratio, prepend=ratio[0]) / ratio[0]

        ratio_sma = talib.SMA(ratio, self.ratio_sma_period)
        features["ratio_sma"] = ratio_sma
        features["ratio_sma_diff"] = (ratio - ratio_sma) / ratio_sma

        ratio_atr = self._calculate_atr(ratio, self.ratio_atr_period)
        features["ratio_atr"] = ratio_atr
        features["ratio_atr_pct"] = ratio_atr / ratio * 100

        spread = ratio - ratio_sma
        features["spread"] = spread
        features["spread_atr_ratio"] = spread / ratio_atr

        for period in self.lookback_periods:
            features[f"ratio_high_{period}"] = pd.Series(ratio).rolling(window=period).max()
            features[f"ratio_low_{period}"] = pd.Series(ratio).rolling(window=period).min()
            features[f"ratio_range_{period}"] = (
                features[f"ratio_high_{period}"] - features[f"ratio_low_{period}"]
            ) / features[f"ratio_low_{period}"]

        features["hurst_exponent"] = self._calculate_hurst(ratio)
        features["half_life"] = self._calculate_half_life(ratio)

        features["return_a"] = np.diff(close_a, prepend=close_a[0]) / close_a[0]
        features["return_b"] = np.diff(close_b, prepend=close_b[0]) / close_b[0]
        features["return_diff"] = features["return_a"] - features["return_b"]
        features["return_corr"] = (
            pd.Series(features["return_a"]).rolling(window=20).corr(pd.Series(features["return_b"]))
        )

        features["volume_ratio"] = volume_a / (volume_b + 1e-10)
        features["volume_total"] = volume_a + volume_b

        ratio_volatility = pd.Series(ratio).rolling(window=20).std()
        features["volatility_ratio"] = ratio_volatility / (ratio + 1e-10)

        sma_5 = talib.SMA(ratio, 5)
        sma_20 = talib.SMA(ratio, 20)
        features["sma_crossover"] = (sma_5 - sma_20) / (sma_20 + 1e-10)

        features = features.dropna()
        features = features.reset_index(drop=True)

        return features

    def get_feature_names(self) -> list[str]:
        """获取特征名称列表"""
        names = [
            "ratio",
            "ratio_return",
            "ratio_sma",
            "ratio_sma_diff",
            "ratio_atr",
            "ratio_atr_pct",
            "spread",
            "spread_atr_ratio",
            "return_a",
            "return_b",
            "return_diff",
            "return_corr",
            "volume_ratio",
            "volume_total",
            "volatility_ratio",
            "sma_crossover",
            "hurst_exponent",
            "half_life",
        ]
        for period in self.lookback_periods:
            names.extend(
                [
                    f"ratio_high_{period}",
                    f"ratio_low_{period}",
                    f"ratio_range_{period}",
                ]
            )
        return names

    @property
    def n_features(self) -> int:
        """特征数量"""
        base = 18
        lookback = len(self.lookback_periods) * 3
        return base + lookback

    @staticmethod
    def _klines_to_df(klines: list[Kline]) -> pd.DataFrame:
        """Kline 列表转 DataFrame"""
        return pd.DataFrame(
            {
                "open_time": [k.open_time for k in klines],
                "open": [k.open for k in klines],
                "high": [k.high for k in klines],
                "low": [k.low for k in klines],
                "close": [k.close for k in klines],
                "volume": [k.volume for k in klines],
            }
        )

    @staticmethod
    def _calculate_atr(values: np.ndarray, period: int) -> np.ndarray:
        """计算 ATR（简化版，用于比率）"""
        tr = np.abs(np.diff(values, prepend=values[0]))
        atr = np.zeros_like(values)
        atr[period - 1] = np.mean(tr[1 : period + 1])
        for i in range(period, len(tr)):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        return atr

    @staticmethod
    def _calculate_hurst(series: np.ndarray, max_lag: int = 100) -> float:
        """
        计算 Hurst 指数

        含义：
        - H > 0.5: 趋势持续（趋势跟随）
        - H < 0.5: 均值回复
        - H = 0.5: 随机游走
        """
        n = min(len(series), max_lag)
        if n < 10:
            return 0.5

        lags = range(2, n // 2)
        tau = []
        tau_std = []

        for lag in lags:
            pp = np.subtract(series[lag:], series[:-lag])
            tau.append(np.std(pp))

        if len(tau) < 2 or np.any(np.isnan(tau)) or np.any(np.isinf(tau)):
            return 0.5

        poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
        hurst = poly[0] * 2

        if math.isnan(hurst) or math.isinf(hurst):
            return 0.5

        return max(0.0, min(1.0, hurst))

    @staticmethod
    def _calculate_half_life(series: np.ndarray) -> float:
        """
        计算均值回复半衰期

        使用 Ornstein-Uhlenbeck 公式:
        half_life = -log(2) / lambda
        where lambda = sum(x * x_lag) / sum(x^2)
        """
        if len(series) < 10:
            return 50.0

        x = series[1:] - series[:-1]
        x_lag = series[:-1]

        x = x[~np.isnan(x) & ~np.isnan(x_lag)]
        x_lag = x_lag[~np.isnan(x) & ~np.isnan(x_lag)]

        if len(x) < 10:
            return 50.0

        lambda_ = np.sum(x * x_lag) / np.sum(x_lag * x_lag)

        if lambda_ <= 0:
            return 100.0

        half_life = -math.log(2) / lambda_

        if math.isnan(half_life) or math.isinf(half_life) or half_life < 0:
            return 50.0

        return min(max(half_life, 1.0), 1000.0)
