"""
Download missing BTCUSDT mark price data from Binance Vision
Missing range: 2020-01 to 2024-02
"""

import sys
import zipfile
import io
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm

DATA_DIR = Path("data/futures/um/klines")
BASE_URL = "https://data.binance.vision/data/futures/um/monthly/markPriceKlines/BTCUSDT/15m"


def get_missing_months() -> list[str]:
    """Get list of missing months (YYYY-MM format)"""
    existing_start = "2024-03"
    existing_end = "2026-03"

    # Generate all months from 2020-01 to 2026-03
    all_months = []
    current = datetime(2020, 1, 1)
    end = datetime(2026, 3, 1)

    while current <= end:
        all_months.append(current.strftime("%Y-%m"))
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)

    # Filter: only 2020-01 to 2024-02 (missing range)
    missing = [m for m in all_months if m < existing_start]
    return missing


def download_single_month(month_str: str) -> tuple[str, Optional[pd.DataFrame]]:
    """Download a single month's data"""
    url = f"{BASE_URL}/BTCUSDT-15m-{month_str}.zip"

    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            return month_str, None

        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            for fname in z.namelist():
                if fname.endswith(".csv"):
                    with z.open(fname) as f:
                        df = pd.read_csv(f)
                        # Standard column names
                        df.columns = [
                            "open_time",
                            "open",
                            "high",
                            "low",
                            "close",
                            "volume",
                            "close_time",
                            "quote_volume",
                            "count",
                            "taker_buy_volume",
                            "taker_buy_quote_volume",
                            "ignore",
                        ]
                        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
                        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
                        return month_str, df
    except Exception as e:
        print(f"Error downloading {month_str}: {e}")

    return month_str, None


def download_missing_data() -> pd.DataFrame:
    """Download all missing monthly data and merge"""
    missing_months = get_missing_months()
    print(f"Missing months to download: {len(missing_months)}")
    print(f"Range: {missing_months[0]} to {missing_months[-1]}")

    all_dfs = []

    # Download in parallel
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(download_single_month, m): m for m in missing_months}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            month_str, df = future.result()
            if df is not None:
                all_dfs.append(df)
                print(f"  Downloaded {month_str}: {len(df)} rows")

    if not all_dfs:
        print("No data downloaded")
        return pd.DataFrame()

    # Merge all downloaded data
    merged = pd.concat(all_dfs, ignore_index=True)
    merged = merged.sort_values("open_time").reset_index(drop=True)

    # Remove duplicates
    merged = merged.drop_duplicates(subset=["open_time"], keep="first")

    print(f"\nDownloaded data: {len(merged)} rows")
    print(f"Range: {merged['open_time'].min()} to {merged['open_time'].max()}")

    return merged


def merge_with_existing(downloaded: pd.DataFrame) -> pd.DataFrame:
    """Merge downloaded data with existing parquet file"""
    existing_path = DATA_DIR / "BTCUSDT_15m_mark.parquet"

    if not existing_path.exists():
        return downloaded

    print("\nMerging with existing data...")
    existing = pd.read_parquet(existing_path)
    print(f"Existing: {len(existing)} rows, {existing['open_time'].min()} to {existing['open_time'].max()}")

    # Ensure consistent types
    for col in ["open_time", "open", "high", "low", "close", "volume"]:
        if col in downloaded.columns:
            downloaded[col] = downloaded[col].astype(float if col != "open_time" else "int64")

    # Merge
    combined = pd.concat([downloaded, existing], ignore_index=True)
    combined = combined.sort_values("open_time").reset_index(drop=True)
    combined = combined.drop_duplicates(subset=["open_time"], keep="first")

    # Ensure open_time is int64 (milliseconds)
    combined["open_time"] = combined["open_time"].astype("int64")

    print(f"Combined: {len(combined)} rows")
    print(f"Range: {combined['open_time'].min()} to {combined['open_time'].max()}")

    return combined


def save_combined(df: pd.DataFrame) -> None:
    """Save combined data to parquet"""
    output_path = DATA_DIR / "BTCUSDT_15m_mark.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(output_path, index=False)
    print(f"\nSaved to {output_path}")


def main():
    print("=" * 60)
    print("Download Missing BTCUSDT Mark Price Data")
    print("=" * 60)

    # Check what's missing
    missing = get_missing_months()
    print(f"\nNeed to download {len(missing)} months: {missing[0]} to {missing[-1]}")

    if not missing:
        print("No missing data to download")
        return

    # Download missing data
    downloaded = download_missing_data()

    if downloaded.empty:
        print("No data was downloaded")
        return

    # Merge with existing
    combined = merge_with_existing(downloaded)

    # Save
    save_combined(combined)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total rows: {len(combined)}")
    print(f"Date range: {combined['open_time'].min()} to {combined['open_time'].max()}")

    # Show available months
    combined["month"] = combined["open_time"].dt.to_period("M")
    months = sorted(combined["month"].unique())
    print(f"Total months: {len(months)}")
    print(f"From {months[0]} to {months[-1]}")


if __name__ == "__main__":
    main()
