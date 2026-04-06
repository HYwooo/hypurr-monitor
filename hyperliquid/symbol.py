"""
Hyperliquid symbol parser - converts user-friendly symbols to Hyperliquid API format.

Supported formats:
- Perpetuals (main dex): BTC, ETH, SOL -> BTC/USDC:USDC
- HIP-3: xyz:GOLD, hyna:BTC, flx:TSLA -> XYZ-GOLD/USDC:USDC, HYNA-BTC/USDE:USDE, etc.
- Spot: spot:HYPE/USDC -> HYPE/USDC
"""

from dataclasses import dataclass


@dataclass
class HyperliquidSymbol:
    """Parsed Hyperliquid symbol."""

    exchange_symbol: str
    coin_param: str | None
    is_spot: bool


_DEX_QUOTE_MAPPING: dict[str, tuple[str, str]] = {
    "xyz": ("USDC", "USDC"),
    "hyna": ("USDE", "USDE"),
    "flx": ("USDH", "USDH"),
    "vntl": ("USDH", "USDH"),
    "km": ("USDH", "USDH"),
    "cash": ("USDT0", "USDT0"),
    "para": ("USDC", "USDC"),
}


def parse_hyperliquid_symbol(symbol: str) -> HyperliquidSymbol:  # noqa: PLR0911
    """Parse user-friendly Hyperliquid symbol to exchange symbol format."""
    original = symbol.strip()

    if original.lower().startswith("spot:"):
        spot_symbol = original[5:].strip()
        return HyperliquidSymbol(exchange_symbol=spot_symbol, coin_param=None, is_spot=True)

    if ":" in original:
        parts = original.split(":")
        if len(parts) == 2:  # noqa: PLR2004
            prefix, coin = parts
            prefix_lower = prefix.lower()

            if prefix_lower in _DEX_QUOTE_MAPPING:
                quote, settle = _DEX_QUOTE_MAPPING[prefix_lower]
                exchange_symbol = f"{prefix.upper()}-{coin}/{quote}:{settle}"
                return HyperliquidSymbol(exchange_symbol=exchange_symbol, coin_param=original, is_spot=False)

            if "/" in original:
                return HyperliquidSymbol(exchange_symbol=original, coin_param=None, is_spot=False)

            return HyperliquidSymbol(exchange_symbol=f"{original}/USDC:USDC", coin_param=None, is_spot=False)

    if "/" in original:
        if ":" in original:
            return HyperliquidSymbol(exchange_symbol=original, coin_param=None, is_spot=False)
        else:
            return HyperliquidSymbol(exchange_symbol=f"{original}:USDC", coin_param=None, is_spot=False)

    return HyperliquidSymbol(exchange_symbol=f"{original}/USDC:USDC", coin_param=None, is_spot=False)


def get_fetch_params(symbol: HyperliquidSymbol) -> dict[str, str]:
    """Get params dict for fetch_ohlcv."""
    params: dict[str, str] = {}
    if symbol.coin_param:
        params["coin"] = symbol.coin_param
    return params


def get_ws_symbol(symbol: HyperliquidSymbol) -> str:
    """Get symbol for WebSocket subscription."""
    return symbol.exchange_symbol
