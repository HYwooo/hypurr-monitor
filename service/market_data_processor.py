"""Market data message processor extracted from NotificationService websocket loop."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any


class MarketDataProcessor:
    """Handle allMids payload processing and downstream signal callbacks."""

    def __init__(  # noqa: PLR0913
        self,
        symbols_fn: Callable[[], list[str]],
        pair_components_fn: Callable[[], dict[str, tuple[str, str]]],
        mark_prices: dict[str, float],
        mark_price_times: dict[str, float],
        logged_initial_price: set[str],
        record_ws_data_activity_fn: Callable[[float], Awaitable[None]],
        log_symbol_state_fn: Callable[[str], None],
        maybe_refresh_runtime_atr_fn: Callable[[str], Awaitable[None]],
        maybe_refresh_runtime_atr_4h_fn: Callable[[str], Awaitable[None]],
        refresh_trailing_stop_channel_fn: Callable[[str], Awaitable[None]],
        check_trailing_stop_fn: Callable[[str, float], Awaitable[None]],
        use_clustering_for_symbol_fn: Callable[[str], bool],
        check_signals_clustering_fn: Callable[[str], Awaitable[None]],
        is_pair_trading_fn: Callable[[str], bool],
        is_pair_symbol_fn: Callable[[str], bool],
        check_signals_fn: Callable[[str], Awaitable[None]],
        check_signals_4h_fn: Callable[[str], Awaitable[None]],
        check_breakout_fn: Callable[[str], Awaitable[None]],
    ) -> None:
        self._symbols_fn = symbols_fn
        self._pair_components_fn = pair_components_fn
        self._mark_prices = mark_prices
        self._mark_price_times = mark_price_times
        self._logged_initial_price = logged_initial_price
        self._record_ws_data_activity_fn = record_ws_data_activity_fn
        self._log_symbol_state_fn = log_symbol_state_fn
        self._maybe_refresh_runtime_atr_fn = maybe_refresh_runtime_atr_fn
        self._maybe_refresh_runtime_atr_4h_fn = maybe_refresh_runtime_atr_4h_fn
        self._refresh_trailing_stop_channel_fn = refresh_trailing_stop_channel_fn
        self._check_trailing_stop_fn = check_trailing_stop_fn
        self._use_clustering_for_symbol_fn = use_clustering_for_symbol_fn
        self._check_signals_clustering_fn = check_signals_clustering_fn
        self._is_pair_trading_fn = is_pair_trading_fn
        self._is_pair_symbol_fn = is_pair_symbol_fn
        self._check_signals_fn = check_signals_fn
        self._check_signals_4h_fn = check_signals_4h_fn
        self._check_breakout_fn = check_breakout_fn

    async def process_payload(self, data: Mapping[str, Any]) -> bool:
        """Process a websocket JSON payload if it belongs to market data."""
        if data.get("channel") != "allMids":
            return False

        payload = data.get("data", {})
        mids_raw = payload.get("mids", {})
        if not isinstance(mids_raw, Mapping):
            return True

        mids = mids_raw
        if mids:
            await self._record_ws_data_activity_fn(time.time())

        updated_symbols = self._update_symbol_prices(mids)

        for symbol in updated_symbols:
            await self._process_updated_symbol(symbol)

        for pair_symbol, (left, right) in self._pair_components_fn().items():
            await self._process_pair_symbol(pair_symbol, left, right)

        return True

    def _update_symbol_prices(self, mids: Mapping[str, Any]) -> set[str]:
        """Update cached prices from allMids payload and return changed symbols."""
        updated_symbols: set[str] = set()
        now = time.time()

        for symbol in self._symbols_fn():
            if symbol in mids:
                self._mark_prices[symbol] = float(mids[symbol])
                self._mark_price_times[symbol] = now
                updated_symbols.add(symbol)

        for left, right in self._pair_components_fn().values():
            for component in (left, right):
                if component in mids and component not in updated_symbols:
                    self._mark_prices[component] = float(mids[component])
                    self._mark_price_times[component] = now
                    updated_symbols.add(component)

        return updated_symbols

    async def _process_updated_symbol(self, symbol: str) -> None:
        """Run signal and breakout checks for a directly updated symbol."""
        if symbol not in self._logged_initial_price:
            self._logged_initial_price.add(symbol)
            self._log_symbol_state_fn(symbol)

        price = self._mark_prices.get(symbol, 0)
        if price <= 0:
            return

        if not self._is_pair_trading_fn(symbol) and not self._is_pair_symbol_fn(symbol):
            await self._maybe_refresh_runtime_atr_fn(symbol)
            await self._maybe_refresh_runtime_atr_4h_fn(symbol)

        await self._refresh_trailing_stop_channel_fn(symbol)
        await self._check_trailing_stop_fn(symbol, price)
        if self._use_clustering_for_symbol_fn(symbol):
            await self._check_signals_clustering_fn(symbol)
        elif not self._is_pair_trading_fn(symbol) and not self._is_pair_symbol_fn(symbol):
            await self._check_signals_fn(symbol)
            await self._check_signals_4h_fn(symbol)

        await self._check_breakout_fn(symbol)

    async def _process_pair_symbol(self, pair_symbol: str, left: str, right: str) -> None:
        """Compute pair price and run pair-specific signal checks."""
        left_price = self._mark_prices.get(left, 0)
        right_price = self._mark_prices.get(right, 0)
        if left_price <= 0 or right_price <= 0:
            return

        now = time.time()
        pair_price = left_price / right_price
        self._mark_prices[pair_symbol] = pair_price
        self._mark_price_times[pair_symbol] = now

        if pair_symbol not in self._logged_initial_price:
            self._logged_initial_price.add(pair_symbol)
            self._log_symbol_state_fn(pair_symbol)

        if not self._use_clustering_for_symbol_fn(pair_symbol):
            await self._maybe_refresh_runtime_atr_fn(pair_symbol)
            await self._maybe_refresh_runtime_atr_4h_fn(pair_symbol)
            await self._refresh_trailing_stop_channel_fn(pair_symbol)

        await self._check_trailing_stop_fn(pair_symbol, pair_price)
        if self._use_clustering_for_symbol_fn(pair_symbol):
            await self._check_signals_clustering_fn(pair_symbol)
        else:
            await self._check_signals_fn(pair_symbol)
            await self._check_signals_4h_fn(pair_symbol)

        await self._check_breakout_fn(pair_symbol)
