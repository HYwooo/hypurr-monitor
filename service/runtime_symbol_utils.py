"""Helpers for symbol relationships and runtime state maintenance."""

# ruff: noqa: PLR0913

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any


def parse_pair_components(pair_list: list[str]) -> dict[str, tuple[str, str]]:
    """Parse pair symbols into their left/right components."""
    pair_components: dict[str, tuple[str, str]] = {}
    for pair_symbol in pair_list:
        if "-" not in pair_symbol:
            continue
        left, right = pair_symbol.split("-", 1)
        pair_components[pair_symbol] = (left, right)
    return pair_components


def is_pair_trading_symbol(symbol: str, pair_components: dict[str, tuple[str, str]]) -> bool:
    """Return whether a symbol appears as either side of a configured pair."""
    return any(symbol in components for components in pair_components.values())


def get_pair_components(symbol: str, pair_components: dict[str, tuple[str, str]]) -> tuple[str, str] | None:
    """Return pair components for a configured pair symbol."""
    return pair_components.get(symbol)


def cleanup_symbol_state(
    symbol: str,
    *mappings: MutableMapping[str, Any],
    alert_times: MutableMapping[str, float],
) -> None:
    """Remove stale symbol-scoped runtime state."""
    for mapping in mappings:
        mapping.pop(symbol, None)

    for alert_key in (f"ATR_Ch_{symbol}", f"ATR_4H_{symbol}", f"ClusterST_{symbol}", symbol):
        alert_times.pop(alert_key, None)


def seed_initial_signal_states(
    symbols: list[str],
    benchmark: dict[str, dict[str, Any]],
    mark_prices: dict[str, float],
    clustering_states: dict[str, Any],
    is_pair_symbol_fn: Any,
    use_clustering_for_symbol_fn: Any,
    last_atr_state: dict[str, dict[str, Any]],
    last_atr4h_state: dict[str, dict[str, Any]],
    last_clustering_state: dict[str, dict[str, Any]],
    last_st_state: dict[str, str],
) -> None:
    """Seed runtime signal state from the current benchmark snapshot."""
    for symbol in symbols:
        bm = benchmark.get(symbol, {})
        atr_ch = bm.get("atr1h_ch", 0)
        if atr_ch in (1, -1):
            last_atr_state[symbol] = {"ch": atr_ch, "sent": "LONG" if atr_ch == 1 else "SHORT"}

        atr4h_ch = bm.get("atr4h_ch", 0)
        if atr4h_ch in (1, -1):
            last_atr4h_state[symbol] = {"ch": atr4h_ch, "sent": "LONG" if atr4h_ch == 1 else "SHORT"}

        if is_pair_symbol_fn(symbol) and use_clustering_for_symbol_fn(symbol):
            current_trend = 0
            cluster_state = clustering_states.get(symbol)
            if cluster_state is not None and hasattr(cluster_state, "trend"):
                current_trend = int(cluster_state.trend)
            last_clustering_state[symbol] = {"trend": current_trend, "sent": None}

        current_price = mark_prices.get(symbol, 0)
        if current_price > 0:
            st1 = bm.get("st1", 0)
            st2 = bm.get("st2", 0)
            last_st_state[symbol] = ("1" if current_price > st1 else "0") + ("1" if current_price > st2 else "0")
