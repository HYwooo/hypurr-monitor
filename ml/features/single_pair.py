"""
单标的特征生成器

为单个交易标的生成技术指标特征
"""

import numpy as np
import pandas as pd
import talib

from models import Kline


class SinglePairFeatureGenerator:
    """
    单标的特征生成器

    生成特征包括：
    - 价格类: close, high, low, open, volume
    - 收益类: returns, log_returns
    - 波动类: atr, natr, volatility
    - 趋势类: sma_ratio, ema_ratio, supertrend_state
    - 动量类: rsi, macd, macd_signal, momentum
    """

    def __init__(
        self,
        sma_periods: list[int] = [7, 14, 25, 50],
        ema_periods: list[int] = [7, 14, 25, 50],
        atr_period: int = 14,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        momentum_period: int = 10,
    ):
        self.sma_periods = sma_periods
        self.ema_periods = ema_periods
        self.atr_period = atr_period
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.momentum_period = momentum_period

    def generate(self, klines: list[Kline]) -> pd.DataFrame:
        """
        为单标的生成特征

        Args:
            klines: K 线列表

        Returns:
            特征 DataFrame
        """
        if len(klines) < 50:
            raise ValueError(f"Insufficient klines: {len(klines)}, need at least 50")

        df = self._klines_to_df(klines)

        features = pd.DataFrame(index=df.index)

        features["close"] = df["close"]
        features["high"] = df["high"]
        features["low"] = df["low"]
        features["open"] = df["open"]
        features["volume"] = df["volume"]

        features["return"] = df["close"].pct_change()
        features["log_return"] = np.log(df["close"] / df["close"].shift(1))

        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        volume = df["volume"].values

        atr = self._calculate_atr(high, low, close, self.atr_period)
        features["atr"] = atr
        features["natr"] = (atr / close) * 100
        features["volatility"] = features["return"].rolling(window=20).std()

        features["rsi"] = talib.RSI(close, self.rsi_period)

        macd, macd_sig, macd_hist = talib.MACD(
            close,
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal,
        )
        features["macd"] = macd
        features["macd_signal"] = macd_sig
        features["macd_hist"] = macd_hist
        features["macd_v"] = macd / (atr + 1e-10)

        features["momentum"] = df["close"] / df["close"].shift(self.momentum_period) - 1

        for period in self.sma_periods:
            sma = talib.SMA(close, period)
            features[f"sma_{period}"] = sma
            features[f"sma_ratio_{period}"] = close / sma - 1

        for period in self.ema_periods:
            ema = talib.EMA(close, period)
            features[f"ema_{period}"] = ema
            features[f"ema_ratio_{period}"] = close / ema - 1

        supertrend = self._calculate_supertrend(high, low, close, 10, 3)
        features["supertrend_state"] = supertrend

        features["volume_ratio"] = volume / (pd.Series(volume).rolling(window=20).mean() + 1e-10)

        features["high_low_ratio"] = (high - low) / close
        features["close_position"] = (close - low) / (high - low)

        features = features.dropna()

        return features

    def get_feature_names(self) -> list[str]:
        """获取特征名称列表"""
        names = [
            "close",
            "high",
            "low",
            "open",
            "volume",
            "return",
            "log_return",
            "atr",
            "natr",
            "volatility",
            "rsi",
            "macd",
            "macd_signal",
            "macd_hist",
            "macd_v",
            "momentum",
            "supertrend_state",
            "volume_ratio",
            "high_low_ratio",
            "close_position",
        ]
        for period in self.sma_periods:
            names.extend([f"sma_{period}", f"sma_ratio_{period}"])
        for period in self.ema_periods:
            names.extend([f"ema_{period}", f"ema_ratio_{period}"])
        return names

    @property
    def n_features(self) -> int:
        """特征数量"""
        base_features = 19
        sma_features = len(self.sma_periods) * 2
        ema_features = len(self.ema_periods) * 2
        return base_features + sma_features + ema_features

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
    def _calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        """计算 ATR"""
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]

        atr = np.zeros_like(close)
        atr[period - 1] = np.mean(tr[1 : period + 1])
        for i in range(period, len(tr)):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

        return atr

    @staticmethod
    def _calculate_supertrend(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int, multiplier: float
    ) -> np.ndarray:
        """计算 SuperTrend 状态"""
        atr = np.zeros_like(close)
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]

        atr[period - 1] = np.mean(tr[1 : period + 1])
        for i in range(period, len(tr)):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

        hl_avg = (high + low) / 2
        upper_band = hl_avg + multiplier * atr
        lower_band = hl_avg - multiplier * atr

        supertrend = np.zeros_like(close)
        direction = np.ones_like(close)

        for i in range(len(close)):
            if i == 0:
                supertrend[i] = lower_band[i]
                direction[i] = 1
            else:
                if close[i] > upper_band[i - 1]:
                    direction[i] = 1
                elif close[i] < lower_band[i - 1]:
                    direction[i] = -1
                else:
                    direction[i] = direction[i - 1]

                supertrend[i] = lower_band[i] if direction[i] == 1 else upper_band[i]

        return direction
