"""
分析不同 neutral_scale 下，Actual Neutral 占总样本的百分比

数据源：Binance Vision markPriceKlines
URL: https://data.binance.vision/?prefix=data/futures/um/monthly/markPriceKlines/BTCUSDT/15m/
"""

import zipfile
import io
from urllib.request import urlopen, Request
from pathlib import Path

import numpy as np
import pandas as pd
import talib
import matplotlib.pyplot as plt

URLS = [
    "https://data.binance.vision/data/futures/um/monthly/markPriceKlines/BTCUSDT/15m/BTCUSDT-15m-2024-01.zip",
    "https://data.binance.vision/data/futures/um/monthly/markPriceKlines/BTCUSDT/15m/BTCUSDT-15m-2024-02.zip",
    "https://data.binance.vision/data/futures/um/monthly/markPriceKlines/BTCUSDT/15m/BTCUSDT-15m-2024-03.zip",
    "https://data.binance.vision/data/futures/um/monthly/markPriceKlines/BTCUSDT/15m/BTCUSDT-15m-2024-04.zip",
    "https://data.binance.vision/data/futures/um/monthly/markPriceKlines/BTCUSDT/15m/BTCUSDT-15m-2024-05.zip",
    "https://data.binance.vision/data/futures/um/monthly/markPriceKlines/BTCUSDT/15m/BTCUSDT-15m-2024-06.zip",
    "https://data.binance.vision/data/futures/um/monthly/markPriceKlines/BTCUSDT/15m/BTCUSDT-15m-2024-07.zip",
    "https://data.binance.vision/data/futures/um/monthly/markPriceKlines/BTCUSDT/15m/BTCUSDT-15m-2024-08.zip",
    "https://data.binance.vision/data/futures/um/monthly/markPriceKlines/BTCUSDT/15m/BTCUSDT-15m-2024-09.zip",
    "https://data.binance.vision/data/futures/um/monthly/markPriceKlines/BTCUSDT/15m/BTCUSDT-15m-2024-10.zip",
    "https://data.binance.vision/data/futures/um/monthly/markPriceKlines/BTCUSDT/15m/BTCUSDT-15m-2024-11.zip",
    "https://data.binance.vision/data/futures/um/monthly/markPriceKlines/BTCUSDT/15m/BTCUSDT-15m-2024-12.zip",
    "https://data.binance.vision/data/futures/um/monthly/markPriceKlines/BTCUSDT/15m/BTCUSDT-15m-2025-01.zip",
    "https://data.binance.vision/data/futures/um/monthly/markPriceKlines/BTCUSDT/15m/BTCUSDT-15m-2025-02.zip",
    "https://data.binance.vision/data/futures/um/monthly/markPriceKlines/BTCUSDT/15m/BTCUSDT-15m-2025-03.zip",
]

DATA_FILE = Path("data/futures/um/klines/BTCUSDT_15m_mark.parquet")

# 标签常量
DOWN_LABEL = 0
NEUTRAL_LABEL = 1
UP_LABEL = 2


def download_mark_price_data() -> pd.DataFrame:
    """从 Binance Vision 下载 mark price 数据"""
    all_data = []

    for url in URLS:
        filename = url.split("/")[-1]
        print(f"Downloading {filename}...")

        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=60) as response:
                data = response.read()

            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for name in z.namelist():
                    if name.endswith(".csv"):
                        with z.open(name) as f:
                            content = f.read().decode()
                            lines = content.strip().split("\n")
                            for line in lines[1:]:  # Skip header
                                parts = line.split(",")
                                if len(parts) >= 6:
                                    all_data.append(
                                        {
                                            "open_time": int(parts[0]),
                                            "open": float(parts[1]),
                                            "high": float(parts[2]),
                                            "low": float(parts[3]),
                                            "close": float(parts[4]),
                                            "volume": float(parts[5]),
                                        }
                                    )
        except Exception as e:
            print(f"Error downloading {filename}: {e}")

    df = pd.DataFrame(all_data)
    df = df.drop_duplicates(subset=["open_time"], keep="last")
    df = df.sort_values("open_time").reset_index(drop=True)
    return df


def load_or_download_data() -> pd.DataFrame:
    """加载本地数据或下载"""
    if DATA_FILE.exists():
        print(f"Loading from {DATA_FILE}...")
        df = pd.read_parquet(DATA_FILE)
        print(f"Loaded {len(df)} rows")
        return df

    print("Downloading mark price data from Binance Vision...")
    df = download_mark_price_data()
    print(f"Saving to {DATA_FILE}...")
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATA_FILE, index=False)
    print(f"Saved {len(df)} rows")
    return df


def compute_labels(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    neutral_scale: float,
    lookforward_bars: int = 1,
) -> np.ndarray:
    """
    计算三分类标签

    - UP (2):   return >  atr * neutral_scale
    - DOWN (0): return < -atr * neutral_scale
    - NEUTRAL (1): |return| <= atr * neutral_scale
    """
    if len(closes) <= lookforward_bars + 14:
        raise ValueError(f"Insufficient data: {len(closes)}, need > {lookforward_bars + 14}")

    atr = talib.ATR(highs, lows, closes, 14)

    future_close = closes[lookforward_bars:]
    current_close = closes[:-lookforward_bars]
    future_atr = atr[:-lookforward_bars]

    future_return = (future_close - current_close) / current_close

    labels = np.full(len(future_return), NEUTRAL_LABEL)

    threshold = (future_atr / current_close) * neutral_scale

    labels[future_return > threshold] = UP_LABEL
    labels[future_return < -threshold] = DOWN_LABEL

    return labels


def analyze_neutral_ratio(df: pd.DataFrame, scales: list[float]) -> dict:
    """
    分析不同 scale 下，Actual Neutral 占总样本的百分比

    Returns:
        dict: {
            scale: {
                "p_actual_n": float,  # Actual Neutral / (L + N + S)
                "n_l": int,
                "n_n": int,
                "n_s": int,
            }
        }
    """
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values

    results = {}

    for scale in scales:
        print(f"Analyzing scale={scale}...")

        # 每个 scale 自己产生的标签
        labels = compute_labels(closes, highs, lows, neutral_scale=scale)

        n_l = int(np.sum(labels == DOWN_LABEL))
        n_n = int(np.sum(labels == NEUTRAL_LABEL))
        n_s = int(np.sum(labels == UP_LABEL))
        total = len(labels)

        results[scale] = {
            "p_actual_n": n_n / total if total > 0 else 0,
            "p_actual_l": n_l / total if total > 0 else 0,
            "p_actual_s": n_s / total if total > 0 else 0,
            "n_l": n_l,
            "n_n": n_n,
            "n_s": n_s,
            "total": total,
        }

    return results


def plot_neutral_ratio(results: dict, scales: list[float], output_file: str = "neutral_ratio_analysis.png") -> None:
    """绘制 neutral 比例分析图"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # 子图1：Actual N/L/S 百分比 vs scale
    ax1 = axes[0]
    p_actual_n = [results[s]["p_actual_n"] * 100 for s in scales]
    p_actual_l = [results[s]["p_actual_l"] * 100 for s in scales]
    p_actual_s = [results[s]["p_actual_s"] * 100 for s in scales]

    ax1.plot(scales, p_actual_l, "r-o", label="Actual L (DOWN)", linewidth=2, markersize=8)
    ax1.plot(scales, p_actual_n, "g-s", label="Actual N (NEUTRAL)", linewidth=2, markersize=8)
    ax1.plot(scales, p_actual_s, "b-^", label="Actual S (UP)", linewidth=2, markersize=8)

    ax1.set_xlabel("Neutral Scale", fontsize=12)
    ax1.set_ylabel("Percentage (%)", fontsize=12)
    ax1.set_title("Actual L/N/S Distribution vs Neutral Scale", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([min(scales) - 0.02, max(scales) + 0.02])
    ax1.set_ylim([0, 105])

    # 子图2：堆叠面积图
    ax2 = axes[1]
    ax2.fill_between(scales, 0, p_actual_l, alpha=0.7, color="red", label="Actual L")
    ax2.fill_between(
        scales,
        p_actual_l,
        np.array(p_actual_l) + np.array(p_actual_n),
        alpha=0.7,
        color="green",
        label="Actual N",
    )
    ax2.fill_between(
        scales,
        np.array(p_actual_l) + np.array(p_actual_n),
        100,
        alpha=0.7,
        color="blue",
        label="Actual S",
    )

    ax2.set_xlabel("Neutral Scale", fontsize=12)
    ax2.set_ylabel("Percentage (%)", fontsize=12)
    ax2.set_title("Actual L/N/S Distribution (Stacked)", fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([min(scales) - 0.02, max(scales) + 0.02])
    ax2.set_ylim([0, 100])

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_file}")

    # 打印数值结果
    print("\n=== Actual Neutral 占比分析 ===")
    print(f"{'Scale':>8} | {'Actual L':>10} | {'Actual N':>10} | {'Actual S':>10} | {'Total':>10} | {'N Ratio':>10}")
    print("-" * 75)
    for s in scales:
        r = results[s]
        print(
            f"{s:>8.2f} | {r['n_l']:>10} | {r['n_n']:>10} | {r['n_s']:>10} | {r['total']:>10} | {r['p_actual_n'] * 100:>9.1f}%"
        )


def main() -> None:
    scales = [0.5, 0.55, 0.6, 0.65, 0.7]

    print("Loading data...")
    df = load_or_download_data()
    print(f"Data range: {df['open_time'].min()} - {df['open_time'].max()}")
    print(f"Total rows: {len(df)}")

    print("\nComputing labels and analyzing...")
    results = analyze_neutral_ratio(df, scales)

    print("\nGenerating plot...")
    plot_neutral_ratio(results, scales, "neutral_ratio_analysis.png")

    print("\nDone!")


if __name__ == "__main__":
    main()
