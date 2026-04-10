"""
风险管理器（预留接口）

待实现功能：
- 仓位大小计算
- 最大回撤限制
- 单日亏损限制
- 风险敞口限制
"""


class RiskManager:
    """
    风险管理器（预留）

    待实现功能：
    - 仓位大小计算（基于波动率、Kelly公式等）
    - 最大回撤限制
    - 单日亏损限制
    - 风险敞口限制
    """

    def check_position_size(
        self,
        signal: int,
        capital: float,
        volatility: float,
    ) -> float:
        """
        检查仓位大小

        Args:
            signal: 信号
            capital: 当前资金
            volatility: 波动率

        Returns:
            建议仓位大小

        Raises:
            NotImplementedError: 风险管理待实现
        """
        raise NotImplementedError("风险管理待实现")

    def check_drawdown(
        self,
        equity_curve: list[float],
        max_drawdown: float = 0.2,
    ) -> bool:
        """
        检查回撤限制

        Args:
            equity_curve: 权益曲线
            max_drawdown: 最大回撤限制

        Returns:
            是否超过限制

        Raises:
            NotImplementedError: 风险管理待实现
        """
        raise NotImplementedError("风险管理待实现")

    def check_daily_loss(
        self,
        daily_pnl: float,
        max_daily_loss: float = 0.05,
    ) -> bool:
        """
        检查单日亏损限制

        Args:
            daily_pnl: 当日盈亏
            max_daily_loss: 最大单日亏损比例

        Returns:
            是否超过限制

        Raises:
            NotImplementedError: 风险管理待实现
        """
        raise NotImplementedError("风险管理待实现")
