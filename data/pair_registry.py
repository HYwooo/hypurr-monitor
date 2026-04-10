"""
配对交易注册表

定义所有支持的交易配对及其元数据
"""

from dataclasses import dataclass
from typing import ClassVar


@dataclass
class TradingPair:
    """
    交易配对定义

    Attributes:
        symbol_a: 标的 A 的交易符号
        symbol_b: 标的 B 的交易符号
        pair_name: 配对显示名称（如 BTC-ETH）
        exchange: 数据源交易所（binance / hyperliquid）
        min_history_days: 最小历史数据天数要求
        data_limit_days: 数据不足时的容错天数
        special_handling: 特殊处理标志
    """

    symbol_a: str
    symbol_b: str
    pair_name: str
    exchange: str
    min_history_days: int = 365
    data_limit_days: int = 90  # xyz:GOLD 等数据量少的配对
    special_handling: bool = False

    @property
    def is_low_data(self) -> bool:
        """是否数据量少的配对"""
        return self.data_limit_days < self.min_history_days


class PairRegistry:
    """
    配对注册表

    管理所有支持的交易配对
    """

    PAIRS: ClassVar[list[TradingPair]] = [
        TradingPair(
            symbol_a="BTCUSDT",
            symbol_b="ETHUSDT",
            pair_name="BTC-ETH",
            exchange="binance",
            min_history_days=365,
            data_limit_days=365,
            special_handling=False,
        ),
        TradingPair(
            symbol_a="BTCUSDT",
            symbol_b="SOLUSDT",
            pair_name="BTC-SOL",
            exchange="binance",
            min_history_days=365,
            data_limit_days=365,
            special_handling=False,
        ),
        TradingPair(
            symbol_a="ETHUSDT",
            symbol_b="SOLUSDT",
            pair_name="ETH-SOL",
            exchange="binance",
            min_history_days=365,
            data_limit_days=365,
            special_handling=False,
        ),
        TradingPair(
            symbol_a="GLDUSDT",
            symbol_b="SLVUSDT",
            pair_name="xyz:GOLD-xyz:SILVER",
            exchange="hyperliquid",
            min_history_days=90,
            data_limit_days=90,
            special_handling=True,  # 数据量少，需要特殊处理
        ),
    ]

    _pair_map: ClassVar[dict[str, TradingPair]] = {p.pair_name: p for p in PAIRS}

    @classmethod
    def get_pair(cls, pair_name: str) -> TradingPair | None:
        """根据名称获取配对"""
        return cls._pair_map.get(pair_name)

    @classmethod
    def get_all_pairs(cls) -> list[TradingPair]:
        """获取所有配对"""
        return cls.PAIRS.copy()

    @classmethod
    def get_pairs_by_exchange(cls, exchange: str) -> list[TradingPair]:
        """根据交易所获取配对"""
        return [p for p in cls.PAIRS if p.exchange == exchange]

    @classmethod
    def get_low_data_pairs(cls) -> list[TradingPair]:
        """获取数据量少的配对"""
        return [p for p in cls.PAIRS if p.is_low_data]

    @classmethod
    def get_all_symbols(cls) -> list[str]:
        """获取所有唯一标的符号"""
        symbols = set()
        for p in cls.PAIRS:
            symbols.add(p.symbol_a)
            symbols.add(p.symbol_b)
        return sorted(list(symbols))

    @classmethod
    def get_symbol_to_pairs(cls) -> dict[str, list[str]]:
        """获取每个标的参与的配对"""
        result: dict[str, list[str]] = {}
        for p in cls.PAIRS:
            if p.symbol_a not in result:
                result[p.symbol_a] = []
            if p.symbol_b not in result:
                result[p.symbol_b] = []
            result[p.symbol_a].append(p.pair_name)
            result[p.symbol_b].append(p.pair_name)
        return result
