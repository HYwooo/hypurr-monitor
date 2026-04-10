"""
回测手续费计算

CS Finance 规范
"""

from dataclasses import dataclass


@dataclass
class CommissionCalculator:
    """
    手续费计算器

    CS Finance 规范：
    - Taker: 0.05%
    - Maker: 0.02%
    """

    fee_taker: float = 0.0005
    fee_maker: float = 0.0002

    def calculate_trade_fee(
        self,
        position_value: float,
        side: str,
        maker: bool = False,
    ) -> float:
        """
        计算交易手续费

        Args:
            position_value: 仓位价值
            side: 交易方向 "LONG" 或 "SHORT"
            maker: 是否为 Maker

        Returns:
            手续费金额
        """
        fee_rate = self.fee_maker if maker else self.fee_taker
        return position_value * fee_rate

    def calculate_roundtrip_fee(
        self,
        position_value: float,
        maker: bool = False,
    ) -> float:
        """
        计算往返手续费（开仓 + 平仓）

        Args:
            position_value: 仓位价值
            maker: 是否为 Maker

        Returns:
            往返手续费
        """
        return position_value * (self.fee_taker + self.fee_maker) if maker else position_value * (self.fee_taker * 2)
