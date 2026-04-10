"""
Binance 数据获取器

自动获取并存储 K 线数据
"""

import asyncio
import logging
import time
from datetime import UTC, datetime
from typing import Any

from data.binance_client import BinanceClient
from data.pair_registry import PairRegistry, TradingPair
from data.storage import DataStorage

logger = logging.getLogger(__name__)


class BinanceFetcher:
    """
    Binance 数据获取器

    自动获取历史 K 线数据并存储到本地
    """

    def __init__(
        self,
        storage: DataStorage | None = None,
        client: BinanceClient | None = None,
        proxy: str | None = None,
    ):
        self.storage = storage or DataStorage()
        self.client = client or BinanceClient(proxy=proxy)

    async def close(self) -> None:
        """关闭客户端"""
        await self.client.close()

    async def fetch_pair_klines(
        self,
        pair: TradingPair,
        interval: str = "15m",
        days: int = 730,
        force_update: bool = False,
    ) -> dict[str, list[Any]]:
        """
        获取配对的 K 线数据

        Args:
            pair: 交易配对
            interval: K 线周期
            days: 回溯天数
            force_update: 是否强制更新

        Returns:
            {symbol: klines} 字典
        """
        logger.info(f"Fetching {pair.pair_name} ({pair.symbol_a}, {pair.symbol_b}) - {days} days @ {interval}")

        result: dict[str, list[Any]] = {}

        end_time = int(time.time() * 1000)
        start_time = int((time.time() - days * 24 * 3600) * 1000)

        if pair.exchange == "binance":
            for symbol in [pair.symbol_a, pair.symbol_b]:
                klines = await self._fetch_symbol_klines(
                    symbol=symbol,
                    interval=interval,
                    start_time=start_time,
                    end_time=end_time,
                    force_update=force_update,
                )
                result[symbol] = klines
        else:
            logger.warning(f"Unsupported exchange for {pair.pair_name}: {pair.exchange}")

        return result

    async def _fetch_symbol_klines(
        self,
        symbol: str,
        interval: str,
        start_time: int,
        end_time: int,
        force_update: bool,
    ) -> list[Any]:
        """获取单个标的的 K 线"""
        cached_count = self.storage.get_klines_count(symbol, interval)

        if not force_update and cached_count > 0:
            existing = self.storage.load_klines(symbol, interval)
            if existing:
                cached_start = existing[0].open_time
                cached_end = existing[-1].open_time

                if start_time >= cached_start and end_time <= cached_end:
                    logger.info(f"Using cached data for {symbol}@{interval}: {len(existing)} klines")
                    return existing

        logger.info(
            "Fetching %s@%s from %s to %s",
            symbol,
            interval,
            datetime.fromtimestamp(start_time / 1000, tz=UTC),
            datetime.fromtimestamp(end_time / 1000, tz=UTC),
        )

        klines = await self.client.fetch_klines_with_pagination(
            symbol=symbol,
            interval=interval,
            start_time=start_time,
            end_time=end_time,
        )

        if klines:
            self.storage.save_klines(klines)

        return klines

    async def fetch_all_pairs(
        self,
        interval: str = "15m",
        days: int = 730,
        force_update: bool = False,
    ) -> dict[str, dict[str, list[Any]]]:
        """
        获取所有配对的 K 线数据

        Args:
            interval: K 线周期
            days: 回溯天数
            force_update: 是否强制更新

        Returns:
            {pair_name: {symbol: klines}} 嵌套字典
        """
        results: dict[str, dict[str, list[Any]]] = {}

        for pair in PairRegistry.get_all_pairs():
            try:
                pair_klines = await self.fetch_pair_klines(
                    pair=pair,
                    interval=interval,
                    days=days,
                    force_update=force_update,
                )
                results[pair.pair_name] = pair_klines

                await asyncio.sleep(0.3)
            except Exception:
                logger.exception("Error fetching %s", pair.pair_name)

        return results

    def get_local_klines(
        self,
        pair: TradingPair,
        interval: str = "15m",
    ) -> dict[str, list[Any]]:
        """
        获取本地缓存的 K 线数据

        Args:
            pair: 交易配对
            interval: K 线周期

        Returns:
            {symbol: klines} 字典
        """
        result: dict[str, list[Any]] = {}

        for symbol in [pair.symbol_a, pair.symbol_b]:
            klines = self.storage.load_klines(symbol, interval)
            if klines:
                result[symbol] = klines

        return result


async def fetch_and_store(
    pair_names: list[str] | None = None,
    interval: str = "15m",
    days: int = 730,
) -> None:
    """
    便捷函数：获取并存储所有 K 线数据

    Args:
        pair_names: 配对名称列表，None 表示所有配对
        interval: K 线周期
        days: 回溯天数
    """
    fetcher = BinanceFetcher()

    try:
        if pair_names:
            for name in pair_names:
                pair = PairRegistry.get_pair(name)
                if pair:
                    await fetcher.fetch_pair_klines(pair, interval, days)
                else:
                    logger.error(f"Unknown pair: {name}")
        else:
            await fetcher.fetch_all_pairs(interval, days)
    finally:
        await fetcher.close()
