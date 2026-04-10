"""
回测引擎

事件驱动回测，支持手续费和资金费率
"""

import logging
from dataclasses import dataclass, field

import numpy as np

from backtest.commission import CommissionCalculator
from backtest.funding import FundingRateCalculator
from ml_common import BacktestResult, BacktestTrade, SignalType

logger = logging.getLogger(__name__)


@dataclass
class BacktestEngine:
    """
    回测引擎

    支持：
    - 手续费计算（Taker 0.05%, Maker 0.02%）
    - 资金费率（年化 10.95%）
    - 概率指标输出
    """

    initial_capital: float = 10000.0
    commission: CommissionCalculator = field(default_factory=CommissionCalculator)
    funding: FundingRateCalculator = field(default_factory=FundingRateCalculator)

    def run(
        self,
        close_prices: np.ndarray,
        signals: np.ndarray,
        labels: np.ndarray,
        probabilities: np.ndarray | None = None,
    ) -> BacktestResult:
        """
        运行回测

        Args:
            close_prices: 收盘价数组
            signals: 预测信号数组 (0=DOWN, 1=NEUTRAL, 2=UP)
            labels: 真实标签数组
            probabilities: 预测概率数组

        Returns:
            回测结果
        """
        n = len(close_prices)
        equity_curve = [self.initial_capital]
        trades: list[BacktestTrade] = []

        position: int | None = None
        entry_price: float = 0.0
        entry_time: int = 0

        total_commission = 0.0
        total_funding = 0.0

        for i in range(1, n):
            current_price = close_prices[i]
            prev_price = close_prices[i - 1]
            current_signal = signals[i]

            if position is None:
                if current_signal == 2:
                    position = 1
                    entry_price = current_price
                    entry_time = i
                    fee = self.commission.calculate_trade_fee(current_price, "LONG", maker=False)
                    total_commission += fee
                elif current_signal == 0:
                    position = -1
                    entry_price = current_price
                    entry_time = i
                    fee = self.commission.calculate_trade_fee(current_price, "SHORT", maker=False)
                    total_commission += fee
            elif position == 1 and current_signal == 0:
                pnl = (current_price - entry_price) / entry_price
                exit_fee = self.commission.calculate_trade_fee(current_price, "LONG", maker=False)

                hold_hours = (i - entry_time) * 0.25
                funding_fee = (
                    self.funding.calculate_funding(entry_price * pnl if pnl > 0 else entry_price, hold_hours)
                    if pnl > 0
                    else 0
                )

                total_commission += exit_fee
                total_funding += funding_fee

                trades.append(
                    BacktestTrade(
                        timestamp=i,
                        pair_name="",
                        side=SignalType.UP if position == 1 else SignalType.DOWN,
                        entry_price=entry_price,
                        exit_price=current_price,
                        position_size=1.0,
                        pnl=pnl * self.initial_capital,
                        pnl_pct=pnl,
                        commission=exit_fee + fee,
                        funding_fee=funding_fee,
                        hold_duration=int(hold_hours * 3600),
                        signal_confidence=probabilities[i, 2] if probabilities is not None else 1.0,
                    )
                )

                position = None
                equity_curve.append(equity_curve[-1] * (1 + pnl - exit_fee / self.initial_capital))

            elif position == -1 and current_signal == 2:
                pnl = (entry_price - current_price) / entry_price
                exit_fee = self.commission.calculate_trade_fee(current_price, "SHORT", maker=False)

                hold_hours = (i - entry_time) * 0.25
                funding_fee = (
                    self.funding.calculate_funding(entry_price * pnl if pnl > 0 else entry_price, hold_hours)
                    if pnl > 0
                    else 0
                )

                total_commission += exit_fee
                total_funding += funding_fee

                trades.append(
                    BacktestTrade(
                        timestamp=i,
                        pair_name="",
                        side=SignalType.DOWN,
                        entry_price=entry_price,
                        exit_price=current_price,
                        position_size=1.0,
                        pnl=pnl * self.initial_capital,
                        pnl_pct=pnl,
                        commission=exit_fee + fee,
                        funding_fee=funding_fee,
                        hold_duration=int(hold_hours * 3600),
                        signal_confidence=probabilities[i, 0] if probabilities is not None else 1.0,
                    )
                )

                position = None
                equity_curve.append(equity_curve[-1] * (1 + pnl - exit_fee / self.initial_capital))

            if position is None:
                equity_curve.append(equity_curve[-1])

        equity_array = np.array(equity_curve)
        total_return = (equity_array[-1] - self.initial_capital) / self.initial_capital

        returns = np.diff(equity_array) / equity_array[:-1]
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(365 * 96) if np.std(returns) > 0 else 0

        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max
        max_drawdown = np.min(drawdown)

        winning_trades = [t for t in trades if t.pnl > 0]
        win_rate = len(winning_trades) / len(trades) if trades else 0

        profit_sum = sum(t.pnl for t in trades if t.pnl > 0)
        loss_sum = abs(sum(t.pnl for t in trades if t.pnl < 0))
        profit_factor = profit_sum / loss_sum if loss_sum > 0 else 0

        avg_duration = sum(t.hold_duration for t in trades) / len(trades) if trades else 0

        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade_duration=int(avg_duration),
            total_trades=len(trades),
            equity_curve=equity_curve,
            trades=trades,
        )
