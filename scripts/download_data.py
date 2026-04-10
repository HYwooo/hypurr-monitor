"""Download historical kline data from Binance Vision"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import zipfile
import io
from urllib.request import urlopen, Request

from data.storage import DataStorage
from models import Kline

URLS = [
    "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/15m/BTCUSDT-15m-2024-01.zip",
    "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/15m/BTCUSDT-15m-2024-02.zip",
    "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/15m/BTCUSDT-15m-2024-03.zip",
    "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/15m/BTCUSDT-15m-2024-04.zip",
    "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/15m/BTCUSDT-15m-2024-05.zip",
    "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/15m/BTCUSDT-15m-2024-06.zip",
    "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/15m/BTCUSDT-15m-2024-07.zip",
    "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/15m/BTCUSDT-15m-2024-08.zip",
    "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/15m/BTCUSDT-15m-2024-09.zip",
    "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/15m/BTCUSDT-15m-2024-10.zip",
    "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/15m/BTCUSDT-15m-2024-11.zip",
    "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/15m/BTCUSDT-15m-2024-12.zip",
    "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/15m/BTCUSDT-15m-2025-01.zip",
    "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/15m/BTCUSDT-15m-2025-02.zip",
    "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/15m/BTCUSDT-15m-2025-03.zip",
]


def download_klines() -> list[Kline]:
    """Download klines from Binance Vision"""
    all_klines = []

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
                            for line in lines[1:]:
                                parts = line.split(",")
                                if len(parts) >= 6:
                                    kline = Kline(
                                        symbol="BTCUSDT",
                                        interval="15m",
                                        open_time=int(parts[0]),
                                        open=float(parts[1]),
                                        high=float(parts[2]),
                                        low=float(parts[3]),
                                        close=float(parts[4]),
                                        volume=float(parts[5]),
                                        close_time=int(parts[6]),
                                        is_closed=True,
                                    )
                                    all_klines.append(kline)
        except Exception as e:
            print(f"Error downloading {filename}: {e}")

    return all_klines


if __name__ == "__main__":
    print("Downloading historical klines from Binance Vision...")
    klines = download_klines()
    print(f"Total klines: {len(klines)}")

    if klines:
        klines.sort(key=lambda x: x.open_time)

        storage = DataStorage()
        storage.save_klines("BTCUSDT", "15m", klines)
        print(f"Saved {len(klines)} klines to data/futures/um/klines/BTCUSDT_15m.parquet")
