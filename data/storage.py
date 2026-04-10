"""
数据存储模块

使用 Parquet 格式存储 K 线数据
"""

import logging
from pathlib import Path

import pandas as pd

from models import Kline

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
KLINE_DIR = DATA_DIR / "futures" / "um" / "klines"


class DataStorage:
    """
    K 线数据存储管理器

    使用 Parquet 格式存储和读取数据
    """

    def __init__(self, data_dir: Path | str = KLINE_DIR):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _get_kline_path(self, symbol: str, interval: str) -> Path:
        """获取 K 线文件路径"""
        return self.data_dir / f"{symbol}_{interval}.parquet"

    def save_klines(self, klines: list[Kline]) -> None:
        """
        保存 K 线数据到 Parquet

        Args:
            klines: K 线列表
        """
        if not klines:
            return

        symbol = klines[0].symbol
        interval = klines[0].interval
        path = self._get_kline_path(symbol, interval)

        df = self._klines_to_df(klines)

        existing_df = None
        if path.exists():
            existing_df = pd.read_parquet(path)
            df = pd.concat([existing_df, df], ignore_index=True)
            df = df.drop_duplicates(subset=["open_time"], keep="last")
            df = df.sort_values("open_time").reset_index(drop=True)

        df.to_parquet(path, index=False)
        logger.info(f"Saved {len(df)} klines to {path}")

    def load_klines(self, symbol: str, interval: str) -> list[Kline]:
        """
        从 Parquet 加载 K 线数据

        Args:
            symbol: 交易对符号
            interval: K 线周期

        Returns:
            K 线列表
        """
        path = self._get_kline_path(symbol, interval)

        if not path.exists():
            logger.warning(f"Kline file not found: {path}")
            return []

        df = pd.read_parquet(path)
        return self._df_to_klines(df, symbol, interval)

    def get_klines_range(
        self,
        symbol: str,
        interval: str,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[Kline]:
        """
        获取指定时间范围的 K 线

        Args:
            symbol: 交易对符号
            interval: K 线周期
            start_time: 开始时间（毫秒）
            end_time: 结束时间（毫秒）

        Returns:
            K 线列表
        """
        path = self._get_kline_path(symbol, interval)

        if not path.exists():
            return []

        df = pd.read_parquet(path)

        if start_time is not None:
            df = df[df["open_time"] >= start_time]
        if end_time is not None:
            df = df[df["open_time"] < end_time]

        return self._df_to_klines(df, symbol, interval)

    def klines_exist(self, symbol: str, interval: str) -> bool:
        """检查 K 线文件是否存在"""
        return self._get_kline_path(symbol, interval).exists()

    def get_klines_count(self, symbol: str, interval: str) -> int:
        """获取 K 线数量"""
        path = self._get_kline_path(symbol, interval)

        if not path.exists():
            return 0

        df = pd.read_parquet(path)
        return len(df)

    def delete_klines(self, symbol: str, interval: str) -> None:
        """删除 K 线文件"""
        path = self._get_kline_path(symbol, interval)
        if path.exists():
            path.unlink()
            logger.info(f"Deleted {path}")

    @staticmethod
    def _klines_to_df(klines: list[Kline]) -> pd.DataFrame:
        """Kline 列表转 DataFrame"""
        return pd.DataFrame(
            [
                {
                    "open_time": k.open_time,
                    "open": k.open,
                    "high": k.high,
                    "low": k.low,
                    "close": k.close,
                    "volume": k.volume,
                    "close_time": k.close_time,
                    "is_closed": k.is_closed,
                }
                for k in klines
            ]
        )

    @staticmethod
    def _df_to_klines(df: pd.DataFrame, symbol: str, interval: str) -> list[Kline]:
        """DataFrame 转 Kline 列表"""
        return [
            Kline(
                symbol=symbol,
                interval=interval,
                open_time=int(row.open_time),
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(row.volume),
                close_time=int(row.close_time) if pd.notna(row.close_time) else 0,
                is_closed=bool(row.is_closed) if pd.notna(row.is_closed) else True,
            )
            for _, row in df.iterrows()
        ]


class ParquetStorage:
    """
    通用 Parquet 存储

    用于存储任意 DataFrame
    """

    def __init__(self, data_dir: Path | str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def save(self, df: pd.DataFrame, name: str) -> Path:
        """
        保存 DataFrame 到 Parquet

        Args:
            df: DataFrame
            name: 文件名（不含扩展名）

        Returns:
            保存的文件路径
        """
        path = self.data_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        logger.info(f"Saved {len(df)} rows to {path}")
        return path

    def load(self, name: str) -> pd.DataFrame | None:
        """
        加载 Parquet 文件

        Args:
            name: 文件名（不含扩展名）

        Returns:
            DataFrame 或 None
        """
        path = self.data_dir / f"{name}.parquet"
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def exists(self, name: str) -> bool:
        """检查文件是否存在"""
        return (self.data_dir / f"{name}.parquet").exists()
