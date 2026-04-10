"""
ML 模块共享类型定义

包含：
- 信号枚举
- 标签定义
- 概率指标
- 配置 dataclass
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class SignalType(Enum):
    """信号类型枚举"""

    UP = 1  # 涨
    DOWN = -1  # 跌
    NEUTRAL = 0  # 平

    @classmethod
    def from_value(cls, value: int) -> "SignalType":
        """从整数值创建枚举"""
        mapping = {1: cls.UP, -1: cls.DOWN, 0: cls.NEUTRAL}
        return mapping.get(value, cls.NEUTRAL)

    @classmethod
    def from_string(cls, s: str) -> "SignalType":
        """从字符串创建枚举"""
        s_lower = s.lower()
        if s_lower in ("up", "long", "buy", "1"):
            return cls.UP
        elif s_lower in ("down", "short", "sell", "-1"):
            return cls.DOWN
        return cls.NEUTRAL


@dataclass
class SignalLabel:
    """信号标签"""

    signal: SignalType
    probability: float  # 置信度 0-1
    timestamp: int  # 毫秒时间戳

    @property
    def is_up(self) -> bool:
        return self.signal == SignalType.UP

    @property
    def is_down(self) -> bool:
        return self.signal == SignalType.DOWN

    @property
    def is_neutral(self) -> bool:
        return self.signal == SignalType.NEUTRAL


class LabelConstants:
    """标签常量"""

    UP_LABEL = 2  # scikit-learn 需要从 0 开始
    NEUTRAL_LABEL = 1
    DOWN_LABEL = 0

    @classmethod
    def to_signal(cls, label: int) -> SignalType:
        """标签转信号类型"""
        mapping = {cls.UP_LABEL: SignalType.UP, cls.NEUTRAL_LABEL: SignalType.NEUTRAL, cls.DOWN_LABEL: SignalType.DOWN}
        return mapping.get(label, SignalType.NEUTRAL)

    @classmethod
    def from_signal(cls, signal: SignalType) -> int:
        """信号类型转标签"""
        mapping = {SignalType.UP: cls.UP_LABEL, SignalType.NEUTRAL: cls.NEUTRAL_LABEL, SignalType.DOWN: cls.DOWN_LABEL}
        return mapping.get(signal, cls.NEUTRAL_LABEL)


@dataclass
class ProbabilityMetrics:
    """
    概率相关指标

    精确率 (Precision): P(真X | 预测X) - 预测X时，真的X的概率
    召回率 (Recall): P(预测X | 真X) - 真X时，预测X的概率
    """

    precision_up: float = 0.0
    precision_down: float = 0.0
    precision_neutral: float = 0.0

    recall_up: float = 0.0
    recall_down: float = 0.0
    recall_neutral: float = 0.0

    f1_up: float = 0.0
    f1_down: float = 0.0
    f1_neutral: float = 0.0

    support_up: int = 0
    support_down: int = 0
    support_neutral: int = 0

    @property
    def accuracy(self) -> float:
        """总体准确率"""
        total = self.support_up + self.support_down + self.support_neutral
        if total == 0:
            return 0.0
        return (self.support_up + self.support_down + self.support_neutral) / total

    def to_dict(self) -> dict:
        """转字典"""
        return {
            "precision_up": self.precision_up,
            "precision_down": self.precision_down,
            "precision_neutral": self.precision_neutral,
            "recall_up": self.recall_up,
            "recall_down": self.recall_down,
            "recall_neutral": self.recall_neutral,
            "f1_up": self.f1_up,
            "f1_down": self.f1_down,
            "f1_neutral": self.f1_neutral,
            "support_up": self.support_up,
            "support_down": self.support_down,
            "support_neutral": self.support_neutral,
        }


@dataclass
class BacktestTrade:
    """回测交易记录"""

    timestamp: int  # 开仓时间戳
    pair_name: str  # 配对名称
    side: SignalType  # 交易方向
    entry_price: float  # 开仓价格
    exit_price: float  # 平仓价格
    position_size: float  # 仓位大小
    pnl: float  # 盈亏
    pnl_pct: float  # 盈亏百分比
    commission: float  # 手续费
    funding_fee: float  # 资金费率
    hold_duration: int  # 持仓时间（秒）
    signal_confidence: float  # 信号置信度


@dataclass
class BacktestResult:
    """回测结果"""

    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_duration: int = 0  # 秒
    total_trades: int = 0
    equity_curve: list[float] = None

    prob_metrics: ProbabilityMetrics = None

    trades: list[BacktestTrade] = None

    def __post_init__(self):
        if self.equity_curve is None:
            self.equity_curve = []
        if self.trades is None:
            self.trades = []


@dataclass
class MLConfig:
    """ML 模块配置"""

    timeframe: str = "15m"
    lookback_days: int = 730  # 2年
    neutral_scale: float = 0.5  # ATR 倍数
    lookforward_bars: int = 1  # 向前看几根K线

    train_ratio: float = 0.7
    valid_ratio: float = 0.15
    test_ratio: float = 0.15

    use_lstm_residual: bool = True
    primary_model: str = "catboost"

    initial_capital: float = 10000.0

    fee_taker: float = 0.0005  # 0.05%
    fee_maker: float = 0.0002  # 0.02%
    funding_annual: float = 0.1095  # 年化 10.95%

    @property
    def funding_daily(self) -> float:
        """日资金费率"""
        return self.funding_annual / 365


class FeatureNames:
    """特征名称常量"""

    # 价格类
    CLOSE = "close"
    HIGH = "high"
    LOW = "low"
    OPEN = "open"
    VOLUME = "volume"

    # 收益类
    RETURN = "return"
    LOG_RETURN = "log_return"

    # 波动类
    ATR = "atr"
    NATR = "natr"
    VOLATILITY = "volatility"

    # 趋势类
    SMA_RATIO = "sma_ratio"
    EMA_RATIO = "ema_ratio"
    SUPERTREND_STATE = "supertrend_state"

    # 动量类
    RSI = "rsi"
    MACD = "macd"
    MACD_SIGNAL = "macd_signal"
    MOMENTUM = "momentum"

    # 配对类
    RATIO = "ratio"
    SPREAD = "spread"
    RATIO_RETURN = "ratio_return"
    RATIO_ATR = "ratio_atr"

    # 跨市场（预留）
    BTC_DOMINANCE = "btc_dominance"
    TOTAL_MARKET_CAP = "total_market_cap"
    FUNDING_RATE = "funding_rate"
    OPEN_INTEREST = "open_interest"


@dataclass
class MLSignal:
    """ML 信号"""

    symbol: str
    signal_type: SignalType
    confidence: float
    probability: tuple[float, float, float]
    timestamp: int
    price: float
    atr: float
    label: int

    @property
    def is_up(self) -> bool:
        """是否上涨信号"""
        return self.signal_type == SignalType.UP

    @property
    def is_down(self) -> bool:
        """是否下跌信号"""
        return self.signal_type == SignalType.DOWN

    @property
    def is_neutral(self) -> bool:
        """是否中性信号"""
        return self.signal_type == SignalType.NEUTRAL

    @property
    def prob_down(self) -> float:
        """下跌概率"""
        return self.probability[0]

    @property
    def prob_neutral(self) -> float:
        """中性概率"""
        return self.probability[1]

    @property
    def prob_up(self) -> float:
        """上涨概率"""
        return self.probability[2]

    def to_dict(self) -> dict[str, Any]:
        """转字典"""
        return {
            "symbol": self.symbol,
            "signal_type": self.signal_type.name,
            "confidence": self.confidence,
            "probability": {
                "down": self.prob_down,
                "neutral": self.prob_neutral,
                "up": self.prob_up,
            },
            "timestamp": self.timestamp,
            "price": self.price,
            "atr": self.atr,
            "label": self.label,
        }
