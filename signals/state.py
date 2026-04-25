"""Lightweight runtime state objects for signal monitors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from models import Kline


class StateNormalizationError(TypeError):
    """Raised when legacy signal state cannot be normalized."""


@dataclass(slots=True)
class AtrChannelState:
    """Runtime state for ATR-channel signal direction."""

    ch: int = 0
    sent: str | None = None

    def get(self, key: str, default: Any = None) -> Any:
        """Provide dict-like read access for compatibility."""
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        """Provide dict-like item access for compatibility."""
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Provide dict-like item write access for compatibility."""
        setattr(self, key, value)

    def to_legacy_dict(self) -> dict[str, Any]:
        """Convert state to the legacy dict shape."""
        return {"ch": self.ch, "sent": self.sent}


@dataclass(slots=True)
class ClusteringSignalState:
    """Runtime state for clustering SuperTrend signal direction."""

    trend: int = 0
    sent: str | None = None

    def get(self, key: str, default: Any = None) -> Any:
        """Provide dict-like read access for compatibility."""
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        """Provide dict-like item access for compatibility."""
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Provide dict-like item write access for compatibility."""
        setattr(self, key, value)

    def to_legacy_dict(self) -> dict[str, Any]:
        """Convert state to the legacy dict shape."""
        return {"trend": self.trend, "sent": self.sent}


def normalize_atr_channel_state(value: Any) -> AtrChannelState:
    """Normalize ATR-channel state to a dataclass while preserving legacy inputs."""
    if isinstance(value, AtrChannelState):
        return value
    if not isinstance(value, dict):
        raise StateNormalizationError
    return AtrChannelState(ch=int(value.get("ch", 0)), sent=value.get("sent"))


def normalize_clustering_signal_state(value: Any) -> ClusteringSignalState:
    """Normalize clustering signal state to a dataclass while preserving legacy inputs."""
    if isinstance(value, ClusteringSignalState):
        return value
    if not isinstance(value, dict):
        raise StateNormalizationError
    return ClusteringSignalState(trend=int(value.get("trend", 0)), sent=value.get("sent"))


@dataclass(slots=True)
class TrailingStopState:
    """Runtime state for a trailing stop monitor."""

    direction: str
    entry_price: float
    entry_time: float
    atr_mult: float
    atr15m_upper: float = 0.0
    atr15m_lower: float = 0.0
    atr15m_state: tuple[float, float, int] = field(default_factory=lambda: (float("nan"), float("nan"), 0))
    active: bool = True
    use_clustering_ts: bool = False
    clustering_ts: float = 0.0
    source: str = "ATR_Ch"

    def get(self, key: str, default: Any = None) -> Any:
        """Provide dict-like read access for compatibility."""
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        """Provide dict-like item access for compatibility."""
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Provide dict-like item write access for compatibility."""
        setattr(self, key, value)

    def to_legacy_dict(self) -> dict[str, Any]:
        """Convert state to the legacy dict shape."""
        return {
            "direction": self.direction,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time,
            "atr_mult": self.atr_mult,
            "atr15m_upper": self.atr15m_upper,
            "atr15m_lower": self.atr15m_lower,
            "atr15m_state": self.atr15m_state,
            "active": self.active,
            "use_clustering_ts": self.use_clustering_ts,
            "clustering_ts": self.clustering_ts,
            "source": self.source,
        }


def normalize_trailing_stop_state(value: Any) -> TrailingStopState:
    """Normalize trailing-stop storage to a dataclass while preserving legacy inputs."""
    if isinstance(value, TrailingStopState):
        return value
    if not isinstance(value, dict):
        raise StateNormalizationError
    return TrailingStopState(
        direction=str(value.get("direction", "")),
        entry_price=float(value.get("entry_price", 0.0)),
        entry_time=float(value.get("entry_time", 0.0)),
        atr_mult=float(value.get("atr_mult", 0.0)),
        atr15m_upper=float(value.get("atr15m_upper", 0.0)),
        atr15m_lower=float(value.get("atr15m_lower", 0.0)),
        atr15m_state=tuple(value.get("atr15m_state", (float("nan"), float("nan"), 0))),
        active=bool(value.get("active", True)),
        use_clustering_ts=bool(value.get("use_clustering_ts", False)),
        clustering_ts=float(value.get("clustering_ts", 0.0)),
        source=str(value.get("source", "ATR_Ch")),
    )


@dataclass(slots=True)
class BreakoutMonitorState:
    """Runtime state for breakout confirmation monitoring."""

    direction: str
    trigger_price: float
    trigger_time: float
    kline_15m_count: int = 0
    klines_15m: list[Kline] = field(default_factory=list)

    def get(self, key: str, default: Any = None) -> Any:
        """Provide dict-like read access for compatibility."""
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        """Provide dict-like item access for compatibility."""
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Provide dict-like item write access for compatibility."""
        setattr(self, key, value)

    def to_legacy_dict(self) -> dict[str, Any]:
        """Convert state to the legacy dict shape."""
        return {
            "direction": self.direction,
            "trigger_price": self.trigger_price,
            "trigger_time": self.trigger_time,
            "kline_15m_count": self.kline_15m_count,
            "klines_15m": self.klines_15m,
        }


def normalize_breakout_monitor_state(value: Any) -> BreakoutMonitorState:
    """Normalize breakout-monitor storage to a dataclass while preserving legacy inputs."""
    if isinstance(value, BreakoutMonitorState):
        return value
    if not isinstance(value, dict):
        raise StateNormalizationError
    return BreakoutMonitorState(
        direction=str(value.get("direction", "")),
        trigger_price=float(value.get("trigger_price", 0.0)),
        trigger_time=float(value.get("trigger_time", 0.0)),
        kline_15m_count=int(value.get("kline_15m_count", 0)),
        klines_15m=list(value.get("klines_15m", [])),
    )


def get_atr_channel_state(state_map: dict[str, Any], symbol: str) -> AtrChannelState:
    """Fetch a normalized ATR-channel state for a symbol."""
    return normalize_atr_channel_state(state_map.get(symbol, {"ch": 0, "sent": None}))


def set_atr_channel_state(state_map: dict[str, Any], symbol: str, ch: int, sent: str | None) -> AtrChannelState:
    """Store ATR-channel state in normalized form."""
    state = AtrChannelState(ch=ch, sent=sent)
    state_map[symbol] = state
    return state


def get_clustering_signal_state(state_map: dict[str, Any], symbol: str) -> ClusteringSignalState:
    """Fetch a normalized clustering signal state for a symbol."""
    return normalize_clustering_signal_state(state_map.get(symbol, {"trend": 0, "sent": None}))


def set_clustering_signal_state(
    state_map: dict[str, Any], symbol: str, trend: int, sent: str | None
) -> ClusteringSignalState:
    """Store clustering signal state in normalized form."""
    state = ClusteringSignalState(trend=trend, sent=sent)
    state_map[symbol] = state
    return state


def get_trailing_stop_state(state_map: dict[str, Any], symbol: str) -> TrailingStopState | None:
    """Fetch a normalized trailing-stop state for a symbol."""
    state = state_map.get(symbol)
    if state is None:
        return None
    normalized = normalize_trailing_stop_state(state)
    state_map[symbol] = normalized
    return normalized


def set_trailing_stop_state(state_map: dict[str, Any], symbol: str, state: TrailingStopState) -> TrailingStopState:
    """Store trailing-stop state in normalized form."""
    state_map[symbol] = state
    return state


def get_breakout_monitor_state(state_map: dict[str, Any], symbol: str) -> BreakoutMonitorState | None:
    """Fetch a normalized breakout-monitor state for a symbol."""
    try:
        state = state_map[symbol]
    except KeyError:
        return None
    if state is None:
        return None
    normalized = normalize_breakout_monitor_state(state)
    state_map[symbol] = normalized
    return normalized


def set_breakout_monitor_state(
    state_map: dict[str, Any], symbol: str, state: BreakoutMonitorState
) -> BreakoutMonitorState:
    """Store breakout-monitor state in normalized form."""
    state_map[symbol] = state
    return state
