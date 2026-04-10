"""
下载 Binance Vision BTCUSDT 15m 标记价格 K线数据

下载范围: 2024-03 到 2026-03

Usage:
    uv run python scripts/download_mark_price_data.py
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import time

import requests
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


BASE_URL = "https://data.binance.vision/data/futures/um/monthly/markPriceKlines/BTCUSDT/15m"
# URL 格式: BTCUSDT-15m-2026-03.zip

OUTPUT_DIR = Path("data/futures/um/klines")


def get_monthly_files(start_year: int, start_month: int, end_year: int, end_month: int) -> list[tuple[int, int]]:
    """生成所有需要下载的月份"""
    files = []
    year, month = start_year, start_month
    while (year < end_year) or (year == end_year and month <= end_month):
        files.append((year, month))
        month += 1
        if month > 12:
            month = 1
            year += 1
    return files


def download_file(year: int, month: int, force: bool = False) -> Optional[Path]:
    """
    下载单个月份的文件

    Returns:
        下载的文件路径，如果失败返回 None
    """
    # 文件名格式: BTCUSDT-15m-YYYY-MM.zip
    filename = f"BTCUSDT-15m-{year}-{month:02d}.zip"
    url = f"{BASE_URL}/{filename}"

    output_path = OUTPUT_DIR / f"BTCUSDT_15m_mark_{year}-{month:02d}.parquet"

    # 如果已存在且不强制下载，跳过
    if output_path.exists() and not force:
        print(f"  已存在，跳过: {filename}")
        return output_path

    print(f"  下载: {filename}")

    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        # 保存 zip 文件
        zip_path = OUTPUT_DIR / filename
        zip_path.parent.mkdir(parents=True, exist_ok=True)

        with open(zip_path, "wb") as f:
            f.write(response.content)

        # 解压并转换为 parquet
        import zipfile
        import io

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            csv_names = z.namelist()
            # 查找 CSV 文件
            csv_name = None
            for name in csv_names:
                if name.endswith(".csv"):
                    csv_name = name
                    break
            if csv_name is None:
                raise ValueError(f"No CSV found in zip: {csv_names}")
            with z.open(csv_name) as csv_file:
                df = pd.read_csv(csv_file)

        # 读取CSV并转换为 parquet
        # CSV 列名: open_time, open, high, low, close, volume, close_time, ...
        # 可能列名带前缀或格式不同，尝试自动检测
        if len(df.columns) >= 7:
            # 取前7列
            df = df.iloc[:, :7]
            df.columns = ["open_time", "open", "high", "low", "close", "volume", "close_time"]
        else:
            raise ValueError(f"Unexpected CSV columns: {df.columns.tolist()}")
        df = df[["open_time", "open", "high", "low", "close", "volume", "close_time"]]

        # 保存 parquet
        df.to_parquet(output_path, index=False)
        print(f"    -> {output_path.name} ({len(df)} rows)")

        # 删除 zip 文件
        zip_path.unlink()

        return output_path

    except Exception as e:
        print(f"    失败: {e}")
        return None


def main():
    # 目标范围: 2024-03 到 2026-03
    start_year, start_month = 2024, 3
    end_year, end_month = 2026, 3

    print("=" * 60)
    print("Binance Vision BTCUSDT 15m Mark Price K线下载")
    print("=" * 60)
    print(f"范围: {start_year}-{start_month:02d} 到 {end_year}-{end_month:02d}")
    print(f"输出目录: {OUTPUT_DIR}")
    print()

    # 获取需要下载的月份列表
    months = get_monthly_files(start_year, start_month, end_year, end_month)
    print(f"需要下载 {len(months)} 个月的文件")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    success_count = 0
    fail_count = 0

    for i, (year, month) in enumerate(months):
        print(f"[{i + 1}/{len(months)}] ", end="")
        result = download_file(year, month)
        if result:
            success_count += 1
        else:
            fail_count += 1

        # 避免请求过快
        if i < len(months) - 1:
            time.sleep(0.5)

    print()
    print("=" * 60)
    print(f"下载完成: 成功 {success_count}, 失败 {fail_count}")
    print("=" * 60)

    # 合并所有 parquet 文件
    print("\n合并数据文件...")
    parquet_files = sorted(OUTPUT_DIR.glob("BTCUSDT_15m_mark_*.parquet"))

    if parquet_files:
        dfs = []
        for pf in parquet_files:
            df = pd.read_parquet(pf)
            dfs.append(df)

        merged = pd.concat(dfs, ignore_index=True)
        merged = merged.drop_duplicates(subset=["open_time"])
        merged = merged.sort_values("open_time").reset_index(drop=True)

        output_file = OUTPUT_DIR / "BTCUSDT_15m_mark.parquet"
        merged.to_parquet(output_file, index=False)
        print(f"合并完成: {output_file} ({len(merged)} rows)")
        print(
            f"时间范围: {datetime.fromtimestamp(merged['open_time'].min() / 1000)} to {datetime.fromtimestamp(merged['open_time'].max() / 1000)}"
        )
    else:
        print("没有找到 parquet 文件")


if __name__ == "__main__":
    main()
