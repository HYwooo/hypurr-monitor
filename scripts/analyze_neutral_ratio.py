"""
分析不同 K 值下的 Neutral 占比

目标是找到 Neutral 占比在 60-70% 的 K 值

Usage:
    uv run python scripts/analyze_neutral_ratio.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import talib

sys.path.insert(0, str(Path(__file__).parent.parent))

# ============ 配置 ============
LOOKFORWARD_BARS = 6  # 未来6根K线
K_VALUES = [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]


def load_data() -> pd.DataFrame:
    """加载数据"""
    data_path = Path("data/futures/um/klines/BTCUSDT_15m_mark.parquet")
    df = pd.read_parquet(data_path)
    df = df.sort_values("open_time").reset_index(drop=True)
    return df


def compute_labels(
    closes: np.ndarray, highs: np.ndarray, lows: np.ndarray, neutral_scale: float, lookforward_bars: int
) -> np.ndarray:
    """计算标签"""
    atr = talib.ATR(highs, lows, closes, 14)

    future_close = closes[lookforward_bars:]
    current_close = closes[:-lookforward_bars]
    future_atr = atr[:-lookforward_bars]

    future_return = (future_close - current_close) / current_close
    threshold = (future_atr / current_close) * neutral_scale

    labels = np.full(len(future_return), 1)  # 默认 Neutral
    labels[future_return > threshold] = 2  # Long
    labels[future_return < -threshold] = 0  # Short

    return labels


def main():
    print("=" * 60)
    print("Neutral 占比分析")
    print("=" * 60)

    # 加载数据
    df = load_data()
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values

    print(f"\n数据: {len(closes)} 条 K线")
    print(
        f"时间范围: {pd.to_datetime(df['open_time'].min(), unit='ms')} ~ {pd.to_datetime(df['open_time'].max(), unit='ms')}"
    )
    print(f"\nK 值分析:")
    print("-" * 50)
    print(f"{'K值':<8} {'Long %':<10} {'Neutral %':<12} {'Short %':<10}")
    print("-" * 50)

    results = []
    for k in K_VALUES:
        labels = compute_labels(closes, highs, lows, k, LOOKFORWARD_BARS)

        n_total = len(labels)
        n_long = np.sum(labels == 2)
        n_neutral = np.sum(labels == 1)
        n_short = np.sum(labels == 0)

        long_pct = n_long / n_total * 100
        neutral_pct = n_neutral / n_total * 100
        short_pct = n_short / n_total * 100

        results.append({"k": k, "long_pct": long_pct, "neutral_pct": neutral_pct, "short_pct": short_pct})

        flag = " ← 目标区间" if 60 <= neutral_pct <= 70 else ""
        print(f"{k:<8.2f} {long_pct:<10.1f} {neutral_pct:<12.1f} {short_pct:<10.1f}{flag}")

    # 找最接近 65% 的 K
    target = 65
    best_k = min(results, key=lambda x: abs(x["neutral_pct"] - target))

    print("-" * 50)
    print(f"\n建议 K 值: {best_k['k']:.2f} (Neutral 占比: {best_k['neutral_pct']:.1f}%)")

    # 额外分析：按月 Neutral 占比
    print(f"\n按月 Neutral 占比分布:")
    print("-" * 50)

    labels = compute_labels(closes, highs, lows, best_k["k"], LOOKFORWARD_BARS)
    dates = pd.to_datetime(df["open_time"][: len(labels)], unit="ms")
    months = dates.dt.year * 100 + dates.dt.month

    for month in sorted(set(months)):
        mask = months == month
        month_labels = labels[mask]
        n_total = len(month_labels)
        n_neutral = np.sum(month_labels == 1)
        neutral_pct = n_neutral / n_total * 100
        month_name = f"{month // 100}-{month % 100:02d}"
        print(f"  {month_name}: {neutral_pct:.1f}%")


if __name__ == "__main__":
    main()
