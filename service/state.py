"""Lightweight runtime state aggregation for service-layer components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from signals.state import BreakoutMonitorState, TrailingStopState


@dataclass(slots=True)
class RuntimeState:
    """Aggregate mutable runtime references used by NotificationService."""

    mark_prices: dict[str, float] = field(default_factory=dict)
    mark_price_times: dict[str, float] = field(default_factory=dict)
    benchmark: dict[str, dict[str, Any]] = field(default_factory=dict)
    trailing_stop: dict[str, TrailingStopState] = field(default_factory=dict)
    breakout_monitor: dict[str, BreakoutMonitorState] = field(default_factory=dict)
    last_alert_time: dict[str, float] = field(default_factory=dict)
    last_atr_state: dict[str, dict[str, Any]] = field(default_factory=dict)
    last_atr4h_state: dict[str, dict[str, Any]] = field(default_factory=dict)
    last_clustering_state: dict[str, dict[str, Any]] = field(default_factory=dict)
