"""
Binance USDT-M Futures API 客户端

用于获取历史 K 线数据
"""

import asyncio
import logging
from typing import Any

import aiohttp

from models import Kline

logger = logging.getLogger(__name__)


class BinanceClient:
    """
    Binance USDT-M Futures API 客户端

    文档: https://developers.binance.com/docs/futures/usdm/market-data
    """

    BASE_URL = "https://fapi.binance.com"

    INTERVAL_MAP = {
        "1m": "1m",
        "3m": "3m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "2h": "2h",
        "4h": "4h",
        "6h": "6h",
        "8h": "8h",
        "12h": "12h",
        "1d": "1d",
        "3d": "3d",
        "1w": "1w",
        "1M": "1M",
    }

    MAX_KLINES_PER_REQUEST = 1500

    def __init__(self, proxy: str | None = None):
        self.proxy = proxy
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """获取或创建 HTTP session"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            connector = aiohttp.TCPConnector(limit=10)
            self._session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        return self._session

    async def close(self) -> None:
        """关闭客户端"""
        if self._session and not self._session.closed:
            await self._session.close()

    async def _get(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        max_retries: int = 3,
    ) -> Any:
        """发送 GET 请求（带重试）"""
        import asyncio

        session = await self._get_session()
        url = f"{self.BASE_URL}{endpoint}"

        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                async with session.get(
                    url, params=params, proxy=self.proxy, timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        raise BinanceAPIError(f"HTTP {resp.status}: {text}")
                    return await resp.json()
            except (TimeoutError, aiohttp.ClientError) as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = 2**attempt
                    logger.warning(
                        f"Request failed (attempt {attempt + 1}/{max_retries}): {e}, retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                    continue
                raise BinanceAPIError(f"Request failed after {max_retries} attempts: {e}") from last_error

        raise BinanceAPIError(f"Request failed: {last_error}") from last_error

    async def fetch_klines(
        self,
        symbol: str,
        interval: str = "15m",
        limit: int = 1500,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[Kline]:
        """
        获取 K 线数据

        Args:
            symbol: 交易对符号，如 "BTCUSDT"
            interval: K 线周期，如 "15m", "1h", "1d"
            limit: 每页数量，最大 1500
            start_time: 开始时间（毫秒）
            end_time: 结束时间（毫秒）

        Returns:
            Kline 列表
        """
        if interval not in self.INTERVAL_MAP:
            raise ValueError(f"Invalid interval: {interval}")

        params: dict[str, Any] = {
            "symbol": symbol,
            "interval": self.INTERVAL_MAP[interval],
            "limit": min(limit, self.MAX_KLINES_PER_REQUEST),
        }

        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time

        data = await self._get("/fapi/v1/klines", params)

        klines = []
        for item in data:
            klines.append(
                Kline(
                    symbol=symbol,
                    interval=interval,
                    open_time=int(item[0]),
                    open=float(item[1]),
                    high=float(item[2]),
                    low=float(item[3]),
                    close=float(item[4]),
                    volume=float(item[5]),
                    close_time=int(item[6]),
                    is_closed=True,
                )
            )

        return klines

    async def fetch_klines_with_pagination(
        self,
        symbol: str,
        interval: str = "15m",
        limit: int | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[Kline]:
        """
        分页获取 K 线数据（自动遍历获取完整历史）

        Args:
            symbol: 交易对符号
            interval: K 线周期
            limit: 总数量限制，None 表示不限制
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            Kline 列表
        """
        all_klines: list[Kline] = []
        current_start = start_time

        while True:
            batch_limit = (
                min(limit - len(all_klines), self.MAX_KLINES_PER_REQUEST) if limit else self.MAX_KLINES_PER_REQUEST
            )

            klines = await self.fetch_klines(
                symbol=symbol,
                interval=interval,
                limit=batch_limit,
                start_time=current_start,
                end_time=end_time,
            )

            if not klines:
                break

            all_klines.extend(klines)

            if limit and len(all_klines) >= limit:
                all_klines = all_klines[:limit]
                break

            current_start = int(klines[-1].close_time) + 1

            if end_time and current_start >= end_time:
                break

            await asyncio.sleep(0.2)

        return all_klines

    async def fetch_funding_rate(
        self,
        symbol: str,
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        获取历史资金费率

        Args:
            symbol: 交易对符号
            start_time: 开始时间
            end_time: 结束时间
            limit: 数量限制

        Returns:
            资金费率列表
        """
        params: dict[str, Any] = {
            "symbol": symbol,
            "limit": limit,
        }

        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time

        return await self._get("/fapi/v1/fundingRate", params)

    async def fetch_open_interest(
        self,
        symbol: str,
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """
        获取持仓量历史

        Args:
            symbol: 交易对符号
            start_time: 开始时间
            end_time: 结束时间
            limit: 数量限制

        Returns:
            持仓量列表
        """
        params: dict[str, Any] = {
            "symbol": symbol,
            "period": "1h",
            "limit": limit,
        }

        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time

        return await self._get("/futures/data/openInterestHist", params)


class BinanceAPIError(Exception):
    """Binance API 错误"""

    def __init__(self, message: str, code: int | None = None):
        super().__init__(message)
        self.code = code
