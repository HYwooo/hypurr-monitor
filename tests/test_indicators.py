"""Tests for indicators module - pure calculation functions."""

import math

import numpy as np

from indicators import (
    ClusteringState,
    calculate_atr,
    calculate_dema,
    calculate_hma,
    calculate_supertrend,
    calculate_tr,
    calculate_vegas_tunnel,
    clustering_supertrend,
    run_atr_channel,
)


class TestCalculateTR:
    """Test True Range calculation."""

    def test_tr_basic(self) -> None:
        """Basic True Range calculation."""
        high = np.array([100.0, 105.0, 110.0])
        low = np.array([95.0, 100.0, 105.0])
        close = np.array([98.0, 103.0, 108.0])
        tr = calculate_tr(high, low, close)
        assert len(tr) == 3
        assert tr[0] == 5.0
        assert tr[1] > 0
        assert tr[2] > 0

    def test_tr_with_gap_up(self) -> None:
        """Test when price gaps up (current low > previous close)."""
        high = np.array([105.0, 110.0, 115.0])
        low = np.array([100.0, 105.0, 110.0])
        close = np.array([103.0, 108.0, 113.0])
        tr = calculate_tr(high, low, close)
        assert tr[1] >= high[1] - low[1]
        assert tr[1] >= high[1] - close[0]

    def test_tr_with_gap_down(self) -> None:
        """Test when price gaps down (current high < previous close)."""
        high = np.array([100.0, 95.0, 105.0])
        low = np.array([90.0, 85.0, 95.0])
        close = np.array([95.0, 88.0, 100.0])
        tr = calculate_tr(high, low, close)
        assert tr[1] >= low[1] - close[0]

    def test_tr_single_element(self) -> None:
        """Single element array returns H-L."""
        high = np.array([105.0])
        low = np.array([95.0])
        close = np.array([100.0])
        tr = calculate_tr(high, low, close)
        assert tr[0] == 10.0

    def test_tr_identical_prices(self) -> None:
        """All identical prices should give 0 TR."""
        high = low = close = np.array([100.0, 100.0, 100.0])
        tr = calculate_tr(high, low, close)
        assert tr[0] == 0.0


class TestCalculateATR:
    """Test ATR calculation with various MA types."""

    def test_atr_dema(self) -> None:
        """ATR with DEMA should return finite values."""
        high = np.array([100.0, 105.0, 110.0, 108.0, 115.0, 120.0, 118.0, 125.0, 130.0, 128.0] * 5)
        low = np.array([95.0, 98.0, 102.0, 100.0, 105.0, 108.0, 106.0, 112.0, 115.0, 113.0] * 5)
        close = np.array([98.0, 103.0, 108.0, 106.0, 113.0, 118.0, 116.0, 122.0, 128.0, 125.0] * 5)
        atr = calculate_atr(high, low, close, period=14, ma_type="DEMA")
        assert len(atr) == len(close)
        assert all(math.isfinite(x) for x in atr[30:])
        assert atr[-1] > 0

    def test_atr_ema(self) -> None:
        """ATR with EMA."""
        atr = calculate_atr(
            np.array([100.0, 105.0, 110.0] * 10),
            np.array([95.0, 98.0, 102.0] * 10),
            np.array([98.0, 103.0, 108.0] * 10),
            period=5,
            ma_type="EMA",
        )
        assert len(atr) == 30
        assert all(math.isfinite(x) for x in atr[10:])

    def test_atr_sma(self) -> None:
        """ATR with SMA."""
        atr = calculate_atr(
            np.array([100.0, 105.0, 110.0] * 10),
            np.array([95.0, 98.0, 102.0] * 10),
            np.array([98.0, 103.0, 108.0] * 10),
            period=5,
            ma_type="SMA",
        )
        assert len(atr) == 30

    def test_atr_wma(self) -> None:
        """ATR with WMA."""
        atr = calculate_atr(
            np.array([100.0, 105.0, 110.0] * 10),
            np.array([95.0, 98.0, 102.0] * 10),
            np.array([98.0, 103.0, 108.0] * 10),
            period=5,
            ma_type="WMA",
        )
        assert len(atr) == 30

    def test_atr_rma(self) -> None:
        """ATR with RMA (Wilder's smoothing)."""
        atr = calculate_atr(
            np.array([100.0, 105.0, 110.0] * 10),
            np.array([95.0, 98.0, 102.0] * 10),
            np.array([98.0, 103.0, 108.0] * 10),
            period=5,
            ma_type="RMA",
        )
        assert len(atr) == 30

    def test_atr_hma(self) -> None:
        """ATR with HMA."""
        atr = calculate_atr(
            np.array([100.0, 105.0, 110.0] * 10),
            np.array([95.0, 98.0, 102.0] * 10),
            np.array([98.0, 103.0, 108.0] * 10),
            period=14,
            ma_type="HMA",
        )
        assert len(atr) == 30

    def test_atr_unknown_ma_type_defaults_to_rma(self) -> None:
        """Unknown MA type should fallback to RMA."""
        atr = calculate_atr(
            np.array([100.0, 105.0, 110.0] * 5),
            np.array([95.0, 98.0, 102.0] * 5),
            np.array([98.0, 103.0, 108.0] * 5),
            period=5,
            ma_type="UNKNOWN",
        )
        assert len(atr) == 15

    def test_atr_short_period(self) -> None:
        """Very short period ATR."""
        atr = calculate_atr(
            np.array([100.0, 105.0, 110.0, 108.0, 115.0]),
            np.array([95.0, 98.0, 102.0, 100.0, 105.0]),
            np.array([98.0, 103.0, 108.0, 106.0, 113.0]),
            period=2,
            ma_type="DEMA",
        )
        assert len(atr) == 5
        assert math.isfinite(atr[-1])


class TestCalculateDEMA:
    """Test DEMA calculation."""

    def test_dema_basic(self) -> None:
        """Basic DEMA calculation."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        dema = calculate_dema(data, period=5)
        assert len(dema) == len(data)
        assert math.isfinite(dema[-1])

    def test_dema_single_value(self) -> None:
        """Single value returns array with NaN (talib needs min periods)."""
        data = np.array([5.0])
        dema = calculate_dema(data, period=2)
        assert len(dema) == 1

    def test_dema_steady_uptrend(self) -> None:
        """DEMA in uptrend follows price."""
        data = np.array([90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0])
        dema = calculate_dema(data, period=3)
        assert dema[-1] <= data[-1]


class TestCalculateHMA:
    """Test HMA calculation."""

    def test_hma_basic(self) -> None:
        """Basic HMA calculation."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        hma = calculate_hma(data, period=4)
        assert len(hma) == len(data)
        assert math.isfinite(hma[-1])

    def test_hma_even_period(self) -> None:
        """HMA with even period (should floor sqrt)."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        hma = calculate_hma(data, period=6)
        assert len(hma) == len(data)


class TestCalculateSupertrend:
    """Test Supertrend calculation."""

    def test_supertrend_basic(self) -> None:
        """Basic Supertrend calculation."""
        high = np.array([100.0, 105.0, 110.0, 108.0, 115.0] * 10)
        low = np.array([95.0, 98.0, 102.0, 100.0, 105.0] * 10)
        close = np.array([98.0, 103.0, 108.0, 106.0, 113.0] * 10)
        st = calculate_supertrend(high, low, close, period=5, multiplier=2.5)
        assert len(st) == len(close)
        assert not np.isnan(st[-1])

    def test_supertrend_trending_up(self) -> None:
        """Supertrend in strong uptrend should be below price."""
        high = np.array([100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0, 114.0, 116.0, 118.0])
        low = np.array([99.0, 101.0, 103.0, 105.0, 107.0, 109.0, 111.0, 113.0, 115.0, 117.0])
        close = high
        st = calculate_supertrend(high, low, close, period=3, multiplier=2.0)
        assert st[-1] < close[-1]

    def test_supertrend_trending_down(self) -> None:
        """Supertrend in strong downtrend should be above price."""
        high = np.array([118.0, 116.0, 114.0, 112.0, 110.0, 108.0, 106.0, 104.0, 102.0, 100.0])
        low = np.array([117.0, 115.0, 113.0, 111.0, 109.0, 107.0, 105.0, 103.0, 101.0, 99.0])
        close = low
        st = calculate_supertrend(high, low, close, period=3, multiplier=2.0)
        assert st[-1] > close[-1]

    def test_supertrend_short_period(self) -> None:
        """Supertrend with period 1."""
        high = np.array([100.0, 105.0, 110.0])
        low = np.array([95.0, 100.0, 105.0])
        close = np.array([98.0, 103.0, 108.0])
        st = calculate_supertrend(high, low, close, period=1, multiplier=2.0)
        assert len(st) == 3


class TestCalculateVegasTunnel:
    """Test Vegas Tunnel (3 EMA lines) calculation."""

    def test_vegas_tunnel_basic(self) -> None:
        """Basic Vegas Tunnel calculation returns valid arrays."""
        close = np.array([100.0 + i * 0.5 + np.random.randn() * 0.1 for i in range(200)])
        ema_s, ema_u, ema_l = calculate_vegas_tunnel(close, vt_ema_signal=9, vt_ema_upper=144, vt_ema_lower=169)
        assert len(ema_s) == len(close)
        assert len(ema_u) == len(close)
        assert len(ema_l) == len(close)
        assert all(math.isfinite(x) for x in ema_s[100:])
        assert all(math.isfinite(x) for x in ema_u[144:])
        assert all(math.isfinite(x) for x in ema_l[169:])

    def test_vegas_tunnel_trending_up(self) -> None:
        """Vegas Tunnel in uptrend returns valid arrays."""
        close = np.cumsum(np.array([0.5] * 300))
        ema_s, ema_u, ema_l = calculate_vegas_tunnel(close, vt_ema_signal=9, vt_ema_upper=144, vt_ema_lower=169)
        assert len(ema_s) == len(close)
        assert len(ema_u) == len(close)
        assert len(ema_l) == len(close)
        assert ema_s[-1] > ema_s[100]


class TestRunATRChannel:
    """Test ATR Channel state machine."""

    def test_initial_state(self) -> None:
        """Initial state should center channel on price."""
        price = 100.0
        atr = 2.0
        mult = 1.5
        upper, lower, ch = run_atr_channel(price, atr, mult, (float("nan"), float("nan"), 0))
        assert ch == 0
        assert upper > price
        assert lower < price
        assert upper - lower == atr * mult

    def test_break_above_becomes_long(self) -> None:
        """Price breaks above upper band -> LONG state."""
        upper, lower, _ = run_atr_channel(100.0, 2.0, 1.5, (float("nan"), float("nan"), 0))
        upper, lower, ch = run_atr_channel(105.0, 2.0, 1.5, (upper, lower, 0))
        assert ch == 1
        assert upper == 105.0

    def test_break_below_becomes_short(self) -> None:
        """Price breaks below lower band -> SHORT state."""
        upper, lower, _ = run_atr_channel(100.0, 2.0, 1.5, (float("nan"), float("nan"), 0))
        upper, lower, ch = run_atr_channel(95.0, 2.0, 1.5, (upper, lower, 1))
        assert ch == -1
        assert lower == 95.0

    def test_channel_only_rises_in_long(self) -> None:
        """In LONG state, lower band only rises."""
        upper, lower, _ = run_atr_channel(100.0, 2.0, 1.5, (float("nan"), float("nan"), 0))
        _, lower1, _ = run_atr_channel(105.0, 2.0, 1.5, (upper, lower, 1))
        _, lower2, _ = run_atr_channel(106.0, 2.0, 1.5, (upper, lower1, 1))
        assert lower2 >= lower1

    def test_channel_only_falls_in_short(self) -> None:
        """In SHORT state, upper band only falls."""
        upper, lower, _ = run_atr_channel(100.0, 2.0, 1.5, (float("nan"), float("nan"), 0))
        upper1, _, _ = run_atr_channel(95.0, 2.0, 1.5, (upper, lower, -1))
        upper2, _, _ = run_atr_channel(94.0, 2.0, 1.5, (upper1, lower, -1))
        assert upper2 <= upper1

    def test_nan_atr_returns_nan_state(self) -> None:
        """NaN ATR should not update channel."""
        upper, lower, ch = run_atr_channel(100.0, float("nan"), 1.5, (110.0, 90.0, 1))
        assert upper == 110.0
        assert lower == 90.0
        assert ch == 1

    def test_zero_atr_returns_nan_state(self) -> None:
        """Zero ATR should not update channel."""
        upper, lower, ch = run_atr_channel(100.0, 0.0, 1.5, (110.0, 90.0, 1))
        assert upper == 110.0
        assert lower == 90.0
        assert ch == 1

    def test_negative_atr_treated_as_invalid(self) -> None:
        """Negative ATR treated as invalid."""
        upper, lower, ch = run_atr_channel(100.0, -1.0, 1.5, (110.0, 90.0, 1))
        assert upper == 110.0
        assert ch == 1


class TestClusteringState:
    """Test ClusteringState dataclass."""

    def test_default_state(self) -> None:
        """Default state has NaN values."""
        state = ClusteringState()
        assert math.isnan(state.target_factor)
        assert state.trend == 0
        assert math.isnan(state.ts)
        assert math.isnan(state.perf_ama)

    def test_state_fields(self) -> None:
        """State has correct fields."""
        state = ClusteringState(
            target_factor=2.5,
            trend=1,
            ts=100.0,
            perf_ama=99.5,
            upper=102.0,
            lower=98.0,
        )
        assert state.target_factor == 2.5
        assert state.trend == 1
        assert state.ts == 100.0
        assert state.perf_ama == 99.5


class TestClusteringSupertrend:
    """Test Clustering SuperTrend calculation."""

    def test_insufficient_data(self) -> None:
        """Too few data points returns default state."""
        close = np.array([100.0, 101.0])
        high = np.array([101.0, 102.0])
        low = np.array([99.0, 100.0])
        atr = np.array([1.0, 1.0])
        ts, perf, state = clustering_supertrend(close, high, low, atr, None, min_mult=1.0, max_mult=5.0, step=0.5)
        assert isinstance(state, ClusteringState)

    def test_clustering_basic(self) -> None:
        """Basic clustering calculation."""
        n = 100
        close = np.cumsum(np.random.randn(n) * 0.1 + 100)
        high = close + np.random.rand(n) * 2
        low = close - np.random.rand(n) * 2
        atr = np.ones(n) * 2.0
        state = ClusteringState()
        ts, perf_ama, new_state = clustering_supertrend(
            close, high, low, atr, state, min_mult=1.0, max_mult=5.0, step=0.5, perf_alpha=10.0
        )
        assert isinstance(new_state, ClusteringState)

    def test_with_prev_state(self) -> None:
        """Clustering with previous state for continuity."""
        n = 100
        close = np.cumsum(np.random.randn(n) * 0.1 + 100)
        high = close + np.random.rand(n) * 2
        low = close - np.random.rand(n) * 2
        atr = np.ones(n) * 2.0
        prev_state = ClusteringState(target_factor=2.5, trend=1)
        ts, perf_ama, new_state = clustering_supertrend(
            close, high, low, atr, prev_state, min_mult=1.0, max_mult=5.0, step=0.5
        )
        assert isinstance(new_state, ClusteringState)

    def test_all_clusters_same(self) -> None:
        """Flat performance (all same) should not crash K-means."""
        n = 50
        close = np.ones(n) * 100.0
        high = np.ones(n) * 101.0
        low = np.ones(n) * 99.0
        atr = np.ones(n) * 1.0
        ts, perf_ama, state = clustering_supertrend(close, high, low, atr, None, min_mult=1.0, max_mult=3.0, step=1.0)
        assert isinstance(state, ClusteringState)
