"""
Clustering SuperTrend indicator (Pinescript V6 port).

Paper: SuperTrend AI (Clustering) [LuxAlgo]
https://www.tradingview.com/scripts/oxT6MfOv/

Core idea:
1. Batch SuperTrend: compute N ST instances with different factors (min_mult~max_mult, step=0.5)
   each records its trend direction, performance (perf)
2. K-Means clustering (K=3):
   - Initialize 3 centroids with perf Q25/Q50/Q75
   - Iterate: assign points to nearest centroid, recalculate centroid, until convergence
3. Final use:
   - target_factor = mean of all factors in selected cluster
   - perf_idx = cluster_avg_perf / EMA(|price_diff|)
   - ts = normal SuperTrend(target_factor)
   - perf_ama = EMA(ts, perf_idx) -- smoothed trailing stop line

Applicable to: PairTrading pairs (e.g. BTCUSDT/ETHUSDT)
"""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import talib

MIN_CLUSTERS = 3


@dataclass
class ClusteringState:
    """
    Clustering SuperTrend state (for cross-call persistence).
    """

    # Cluster state
    centroids: tuple[float, float, float] = (math.nan, math.nan, math.nan)
    cluster_factors: tuple[list[float], list[float], list[float]] = ([], [], [])
    cluster_perfs: tuple[list[float], list[float], list[float]] = ([], [], [])

    # Current ST state
    target_factor: float = math.nan
    ts: float = math.nan
    ts_prev: float = math.nan
    perf_ama: float = math.nan
    trend: int = 0
    upper: float = math.nan
    lower: float = math.nan
    prev_upper: float = math.nan
    prev_lower: float = math.nan


def _kmeans_clustering(
    perfs: np.ndarray[Any, Any],
    factors: np.ndarray[Any, Any],
    max_iter: int = 1000,
) -> tuple[
    tuple[float, float, float],
    tuple[list[float], list[float], list[float]],
    tuple[list[float], list[float], list[float]],
]:
    """
    K-Means clustering (K=3).

    Initialize: use perf percentiles (Q25/Q50/Q75) as 3 centroids.
    Iterate: assign points to nearest centroid -> recalculate centroid -> until centroids stop changing.

    Args:
        perfs: Performance array (one perf per factor)
        factors: Factor array (one-to-one with perfs)
        max_iter: Maximum iterations

    Returns:
        centroids: (c0, c1, c2) three centroid values
        cluster_factors: ([f0...], [f1...], [f2...]) factors per cluster
        cluster_perfs: ([p0...], [p1...], [p2...]) perfs per cluster
    """
    if len(perfs) < MIN_CLUSTERS:
        return (math.nan, math.nan, math.nan), ([], [], [])  # type: ignore[return-value]

    perfs = np.array(perfs, dtype=float)
    factors = np.array(factors, dtype=float)

    def percentile(arr: np.ndarray[Any, Any], q: float) -> float:
        """Calculate percentile."""
        sorted_arr = np.sort(arr)
        idx = (len(sorted_arr) - 1) * q / 100
        lo = math.floor(idx)
        hi = math.ceil(idx)
        if lo == hi:
            return float(sorted_arr[lo])
        frac = idx - lo
        return float(sorted_arr[lo]) * (1 - frac) + float(sorted_arr[hi]) * frac

    c0 = percentile(perfs, 25)
    c1 = percentile(perfs, 50)
    c2 = percentile(perfs, 75)
    centroids = (c0, c1, c2)

    cluster_factors: tuple[list[float], list[float], list[float]] = ([], [], [])
    cluster_perfs: tuple[list[float], list[float], list[float]] = ([], [], [])

    for _ in range(max_iter):
        cf: tuple[list[float], list[float], list[float]] = ([], [], [])
        cp: tuple[list[float], list[float], list[float]] = ([], [], [])

        for i, p in enumerate(perfs):
            d0 = abs(p - centroids[0])
            d1 = abs(p - centroids[1])
            d2 = abs(p - centroids[2])
            factor_i = float(factors[i])
            if d0 <= d1 and d0 <= d2:
                cf[0].append(factor_i)
                cp[0].append(p)
            elif d1 <= d0 and d1 <= d2:
                cf[1].append(factor_i)
                cp[1].append(p)
            else:
                cf[2].append(factor_i)
                cp[2].append(p)

        nc0 = sum(cf[0]) / len(cf[0]) if cf[0] else centroids[0]
        nc1 = sum(cf[1]) / len(cf[1]) if cf[1] else centroids[1]
        nc2 = sum(cf[2]) / len(cf[2]) if cf[2] else centroids[2]

        if math.isclose(nc0, centroids[0]) and math.isclose(nc1, centroids[1]) and math.isclose(nc2, centroids[2]):
            break

        centroids = (nc0, nc1, nc2)
        cluster_factors = (cf[0], cf[1], cf[2])
        cluster_perfs = (cp[0], cp[1], cp[2])

    return (
        centroids,
        (cluster_factors[0], cluster_factors[1], cluster_factors[2]),
        (cluster_perfs[0], cluster_perfs[1], cluster_perfs[2]),
    )


def clustering_supertrend(  # noqa: PLR0912
    close: np.ndarray[Any, Any],
    high: np.ndarray[Any, Any],
    low: np.ndarray[Any, Any],
    atr: np.ndarray[Any, Any],
    prev_state: ClusteringState | None,
    min_mult: float = 1.0,
    max_mult: float = 5.0,
    step: float = 0.5,
    perf_alpha: float = 10.0,
    from_cluster: str = "Best",
    max_iter: int = 1000,
    max_data: int = 10000,
) -> tuple[float, float, ClusteringState]:
    """
    Clustering SuperTrend main function.

    Performs clustering analysis on the last max_data bars of close/high/low/atr arrays,
    selects optimal cluster, calculates target_factor, then computes TS and AMA.

    Args:
        close: Close price array
        high: High price array
        low: Low price array
        atr: ATR array (pre-calculated)
        prev_state: Previous ClusteringState (for cross-call persistence)
        min_mult: ATR multiplier range lower bound
        max_mult: ATR multiplier range upper bound
        step: ATR multiplier step
        perf_alpha: perf EMA smoothing coefficient
        from_cluster: Which cluster to use, "Best" | "Average" | "Worst"
        max_iter: K-Means maximum iterations
        max_data: Maximum bars participating in clustering

    Returns:
        (ts, perf_ama, new_state)
        ts: trailing stop line
        perf_ama: smoothed AMA
        new_state: Updated ClusteringState (for next call)
    """
    n = len(close)
    if n < MIN_CLUSTERS:
        default_state = ClusteringState()
        if prev_state:
            return prev_state.ts, prev_state.perf_ama, prev_state
        return math.nan, math.nan, default_state

    # === 1. Generate factors array ===
    factors: list[float] = []
    f = min_mult
    while f <= max_mult + 1e-9:
        factors.append(f)
        f += step
    factors_arr: np.ndarray[Any, Any] = np.array(factors, dtype=float)

    # === 2. Batch calculate SuperTrend perf ===
    start_idx = max(0, n - max_data)

    perfs_out = np.zeros(len(factors_arr))
    outputs_out = np.zeros(len(factors_arr))
    uppers_out = np.full(len(factors_arr), math.nan)
    lowers_out = np.full(len(factors_arr), math.nan)
    trends_out = np.zeros(len(factors_arr), dtype=int)

    prev_perf = 0.0

    for i in range(start_idx, n):
        close_i = close[i]
        prev_close = close[i - 1] if i > 0 else close_i
        hl2 = (high[i] + low[i]) / 2

        for k, factor in enumerate(factors_arr):
            atr_val = atr[i] if i < len(atr) else atr[-1]
            if not math.isfinite(atr_val) or atr_val <= 0:
                continue

            up = hl2 + atr_val * factor
            dn = hl2 - atr_val * factor

            if close_i > uppers_out[k]:
                trends_out[k] = 1
            elif close_i < lowers_out[k]:
                trends_out[k] = -1

            prev_up = uppers_out[k - 1] if i > start_idx and math.isfinite(uppers_out[k - 1]) else up
            prev_low = lowers_out[k - 1] if i > start_idx and math.isfinite(lowers_out[k - 1]) else dn

            uppers_out[k] = min(up, prev_up) if math.isfinite(prev_up) else up
            lowers_out[k] = max(dn, prev_low) if math.isfinite(prev_low) else dn

            outputs_out[k] = lowers_out[k] if trends_out[k] == 1 else uppers_out[k]

        diff = 0.0
        if close_i != prev_close:
            diff = 1.0 if close_i > prev_close else -1.0

        alpha = 2.0 / (perf_alpha + 1.0)
        price_diff = close_i - prev_close
        perf = alpha * (price_diff * diff) + (1 - alpha) * prev_perf
        prev_perf = perf

        for k in range(len(factors_arr)):
            perfs_out[k] = perf

    # === 3. K-Means clustering ===
    centroids, cluster_factors, cluster_perfs = _kmeans_clustering(perfs_out, factors_arr, max_iter)

    state = prev_state or ClusteringState()
    state.centroids = centroids
    state.cluster_factors = cluster_factors
    state.cluster_perfs = cluster_perfs

    # === 4. Select target cluster ===
    cluster_idx = {"Best": 2, "Average": 1, "Worst": 0}.get(from_cluster, 2)
    target_factors = cluster_factors[cluster_idx]
    target_perfs = cluster_perfs[cluster_idx]

    if target_factors:
        state.target_factor = sum(target_factors) / len(target_factors)
    else:
        state.target_factor = (min_mult + max_mult) / 2

    if not math.isfinite(state.target_factor):
        state.target_factor = (min_mult + max_mult) / 2

    # === 5. Calculate final ST with target_factor ===
    factor = state.target_factor
    atr_last = float(atr[-1]) if len(atr) > 0 else 0.0
    if not math.isfinite(atr_last) or atr_last <= 0:
        return state.ts, state.perf_ama, state

    hl2 = (high[-1] + low[-1]) / 2
    up = hl2 + atr_last * factor
    dn = hl2 - atr_last * factor

    state.prev_upper = state.upper
    state.prev_lower = state.lower
    state.upper = up if math.isfinite(up) else hl2
    state.lower = dn if math.isfinite(dn) else hl2

    if math.isfinite(state.prev_upper):
        state.upper = min(state.upper, state.prev_upper)
    if math.isfinite(state.prev_lower):
        state.lower = max(state.lower, state.prev_lower)

    if close[-1] > state.upper:
        state.trend = 1
    elif close[-1] < state.lower:
        state.trend = -1

    new_ts = state.lower if state.trend == 1 else state.upper
    state.ts_prev = state.ts
    state.ts = new_ts

    # === 6. perf_idx and perf_ama ===
    cluster_avg_perf = sum(target_perfs) / len(target_perfs) if target_perfs else 0.0
    cluster_avg_perf = max(cluster_avg_perf, 0.0)

    price_diffs = np.abs(np.diff(close[start_idx:]))
    if len(price_diffs) > 0:
        den = (
            talib.EMA(price_diffs, timeperiod=int(perf_alpha))[-1]
            if len(price_diffs) >= perf_alpha
            else np.mean(price_diffs)
        )
        den = float(den) if math.isfinite(den) and den > 0 else 1.0
    else:
        den = 1.0

    perf_idx = cluster_avg_perf / den

    if math.isnan(state.perf_ama):
        state.perf_ama = state.ts
    elif math.isfinite(state.ts):
        state.perf_ama = state.perf_ama + perf_idx * (state.ts - state.perf_ama)

    return state.ts, state.perf_ama, state


def clustering_supertrend_single(
    close: float,
    high: float,
    low: float,
    prev_state: ClusteringState | None,
    target_factor: float,
    atr: float,
) -> tuple[float, float, int, float, float, ClusteringState]:
    """
    Single-step update Clustering SuperTrend (for real-time push scenarios).

    No clustering performed (clustering done on startup), only updates ST trailing stop.

    Args:
        close: Current close price
        high: Current high price
        low: Current low price
        prev_state: Previous state
        target_factor: Cluster-derived target_factor (from clustering_supertrend batch calculation)
        atr: Current ATR value

    Returns:
        (ts, perf_ama, trend, upper, lower, new_state)
    """
    if atr <= 0 or not math.isfinite(atr):
        _state = prev_state if prev_state is not None else ClusteringState()
        return (
            _state.ts,
            _state.perf_ama,
            _state.trend,
            _state.upper,
            _state.lower,
            _state,
        )

    state = prev_state or ClusteringState()
    state.target_factor = target_factor

    hl2 = (high + low) / 2
    up = hl2 + atr * target_factor
    dn = hl2 - atr * target_factor

    state.prev_upper = state.upper
    state.prev_lower = state.lower
    state.upper = up
    state.lower = dn

    if math.isfinite(state.prev_upper):
        state.upper = min(state.upper, state.prev_upper)
    if math.isfinite(state.prev_lower):
        state.lower = max(state.lower, state.prev_lower)

    if close > state.upper:
        state.trend = 1
    elif close < state.lower:
        state.trend = -1

    new_ts = state.lower if state.trend == 1 else state.upper
    state.ts_prev = state.ts
    state.ts = new_ts

    perf_idx = target_factor / (atr * 10) if atr > 0 else 0.0
    if math.isnan(state.perf_ama):
        state.perf_ama = state.ts
    else:
        state.perf_ama = state.perf_ama + perf_idx * (state.ts - state.perf_ama)

    return state.ts, state.perf_ama, state.trend, state.upper, state.lower, state
