"""Tests for Hyperliquid symbol parsing."""

from hyperliquid.symbol import (
    HyperliquidSymbol,
    get_fetch_params,
    get_ws_symbol,
    parse_hyperliquid_symbol,
)


class TestParseHyperliquidSymbol:
    """Test Hyperliquid symbol parsing."""

    def test_simple_symbol_btc(self) -> None:
        """Simple symbol BTC -> BTC/USDC:USDC."""
        result = parse_hyperliquid_symbol("BTC")
        assert result.exchange_symbol == "BTC/USDC:USDC"
        assert result.coin_param is None
        assert result.is_spot is False

    def test_simple_symbol_eth(self) -> None:
        """Simple symbol ETH."""
        result = parse_hyperliquid_symbol("ETH")
        assert result.exchange_symbol == "ETH/USDC:USDC"

    def test_simple_symbol_sol(self) -> None:
        """Simple symbol SOL."""
        result = parse_hyperliquid_symbol("SOL")
        assert result.exchange_symbol == "SOL/USDC:USDC"

    def test_xyz_gold(self) -> None:
        """HIP-3 xyz:GOLD -> XYZ-GOLD/USDC:USDC with coin param."""
        result = parse_hyperliquid_symbol("xyz:GOLD")
        assert result.exchange_symbol == "XYZ-GOLD/USDC:USDC"
        assert result.coin_param == "xyz:GOLD"
        assert result.is_spot is False

    def test_hyna_btc(self) -> None:
        """HIP-3 hyna:BTC -> HYNA-BTC/USDE:USDE."""
        result = parse_hyperliquid_symbol("hyna:BTC")
        assert result.exchange_symbol == "HYNA-BTC/USDE:USDE"
        assert result.coin_param == "hyna:BTC"

    def test_flx_tsls(self) -> None:
        """HIP-3 flx:TSLA -> FLX-TSLA/USDH:USDH."""
        result = parse_hyperliquid_symbol("flx:TSLA")
        assert result.exchange_symbol == "FLX-TSLA/USDH:USDH"
        assert result.coin_param == "flx:TSLA"

    def test_vntl(self) -> None:
        """HIP-3 vntl:SOME -> VNTL-SOME/USDH:USDH."""
        result = parse_hyperliquid_symbol("vntl:SOME")
        assert result.exchange_symbol == "VNTL-SOME/USDH:USDH"

    def test_km(self) -> None:
        """HIP-3 km:X -> KM-X/USDH:USDH."""
        result = parse_hyperliquid_symbol("km:X")
        assert result.exchange_symbol == "KM-X/USDH:USDH"

    def test_cash_usdt(self) -> None:
        """HIP-3 cash:SOME -> CASH-SOME/USDT0:USDT0."""
        result = parse_hyperliquid_symbol("cash:SOME")
        assert result.exchange_symbol == "CASH-SOME/USDT0:USDT0"

    def test_para(self) -> None:
        """HIP-3 para:X -> PARA-X/USDC:USDC."""
        result = parse_hyperliquid_symbol("para:X")
        assert result.exchange_symbol == "PARA-X/USDC:USDC"

    def test_spot_prefix(self) -> None:
        """Spot prefix should return symbol without coin param."""
        result = parse_hyperliquid_symbol("spot:HYPE/USDC")
        assert result.exchange_symbol == "HYPE/USDC"
        assert result.coin_param is None
        assert result.is_spot is True

    def test_spot_prefix_lowercase(self) -> None:
        """Spot prefix case insensitive."""
        result = parse_hyperliquid_symbol("SPOT:HYPEUSDT")
        assert result.is_spot is True

    def test_already_formatted_with_slash(self) -> None:
        """Already formatted symbol with slash."""
        result = parse_hyperliquid_symbol("BTC/USDC:USDC")
        assert result.exchange_symbol == "BTC/USDC:USDC"

    def test_already_formatted_dex(self) -> None:
        """Already formatted with colon in name (e.g. some:coin)."""
        result = parse_hyperliquid_symbol("xyz:GOLD/USDC:USDC")
        assert result.exchange_symbol == "xyz:GOLD/USDC:USDC"

    def test_unknown_prefix_uses_usdc(self) -> None:
        """Unknown prefix falls back to USDC settle."""
        result = parse_hyperliquid_symbol("unknown:SYM")
        assert result.exchange_symbol == "unknown:SYM/USDC:USDC"

    def test_whitespace_trimmed(self) -> None:
        """Whitespace should be trimmed."""
        result = parse_hyperliquid_symbol("  BTC  ")
        assert result.exchange_symbol == "BTC/USDC:USDC"

    def test_mixed_case(self) -> None:
        """Mixed case prefix handled."""
        result = parse_hyperliquid_symbol("XyZ:GOLD")
        assert result.exchange_symbol == "XYZ-GOLD/USDC:USDC"
        assert result.coin_param == "XyZ:GOLD"


class TestGetFetchParams:
    """Test fetch params generation."""

    def test_coin_param(self) -> None:
        """With coin param should return dict with coin key."""
        sym = HyperliquidSymbol(exchange_symbol="XYZ-GOLD/USDC:USDC", coin_param="xyz:GOLD", is_spot=False)
        params = get_fetch_params(sym)
        assert params == {"coin": "xyz:GOLD"}

    def test_no_coin_param(self) -> None:
        """Without coin param should return empty dict."""
        sym = HyperliquidSymbol(exchange_symbol="BTC/USDC:USDC", coin_param=None, is_spot=False)
        params = get_fetch_params(sym)
        assert params == {}

    def test_spot_no_coin(self) -> None:
        """Spot symbol has no coin param."""
        sym = HyperliquidSymbol(exchange_symbol="HYPE/USDC", coin_param=None, is_spot=True)
        params = get_fetch_params(sym)
        assert params == {}


class TestGetWSSymbol:
    """Test WebSocket symbol."""

    def test_returns_exchange_symbol(self) -> None:
        """WS symbol is the exchange symbol."""
        sym = HyperliquidSymbol(exchange_symbol="BTC/USDC:USDC", coin_param=None, is_spot=False)
        assert get_ws_symbol(sym) == "BTC/USDC:USDC"
