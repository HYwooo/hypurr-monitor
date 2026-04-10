"""
资金费率计算

CS Finance 规范：年化 10.95%
"""

from dataclasses import dataclass


@dataclass
class FundingRateCalculator:
    """
    资金费率计算器

    CS Finance 规范：
    - 年化资金费率: 10.95%
    - 每 8 小时结算一次
    - 日费率 = 年化 / 365
    """

    funding_annual: float = 0.1095

    @property
    def funding_daily(self) -> float:
        """日资金费率"""
        return self.funding_annual / 365

    @property
    def funding_per_8h(self) -> float:
        """每 8 小时资金费率"""
        return self.funding_daily / 3

    def calculate_daily_funding(self, position_value: float) -> float:
        """
        计算日资金费率

        Args:
            position_value: 仓位价值

        Returns:
            日资金费率
        """
        return position_value * self.funding_daily

    def calculate_funding(
        self,
        position_value: float,
        hours: float,
    ) -> float:
        """
        计算持仓期间的资金费率

        Args:
            position_value: 仓位价值
            hours: 持仓小时数

        Returns:
            资金费率
        """
        periods = hours / 8.0
        return position_value * self.funding_per_8h * periods
