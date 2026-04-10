"""
回测绩效指标

计算各类绩效指标
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from ml_common import BacktestResult, ProbabilityMetrics


@dataclass
class PerformanceMetrics:
    """
    绩效指标

    包含：
    - 收益率指标
    - 风险指标
    - 交易指标
    - 概率指标
    """

    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_trade_duration: int
    total_trades: int

    precision_up: float = 0.0
    precision_down: float = 0.0
    precision_neutral: float = 0.0

    recall_up: float = 0.0
    recall_down: float = 0.0
    recall_neutral: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """转字典"""
        return {
            "total_return": f"{self.total_return * 100:.2f}%",
            "sharpe_ratio": f"{self.sharpe_ratio:.2f}",
            "max_drawdown": f"{self.max_drawdown * 100:.2f}%",
            "win_rate": f"{self.win_rate * 100:.2f}%",
            "profit_factor": f"{self.profit_factor:.2f}",
            "avg_trade_duration": f"{self.avg_trade_duration / 3600:.1f}h",
            "total_trades": self.total_trades,
            "precision_up": f"{self.precision_up * 100:.2f}%",
            "precision_down": f"{self.precision_down * 100:.2f}%",
            "recall_up": f"{self.recall_up * 100:.2f}%",
            "recall_down": f"{self.recall_down * 100:.2f}%",
        }


class MetricsCalculator:
    """
    绩效指标计算器
    """

    def calculate_from_backtest(
        self,
        result: BacktestResult,
        prob_metrics: ProbabilityMetrics | None = None,
    ) -> PerformanceMetrics:
        """
        从回测结果计算绩效指标

        Args:
            result: 回测结果
            prob_metrics: 概率指标

        Returns:
            绩效指标
        """
        metrics = PerformanceMetrics(
            total_return=result.total_return,
            sharpe_ratio=result.sharpe_ratio,
            max_drawdown=result.max_drawdown,
            win_rate=result.win_rate,
            profit_factor=result.profit_factor,
            avg_trade_duration=result.avg_trade_duration,
            total_trades=result.total_trades,
        )

        if prob_metrics:
            metrics.precision_up = prob_metrics.precision_up
            metrics.precision_down = prob_metrics.precision_down
            metrics.precision_neutral = prob_metrics.precision_neutral
            metrics.recall_up = prob_metrics.recall_up
            metrics.recall_down = prob_metrics.recall_down
            metrics.recall_neutral = prob_metrics.recall_neutral

        return metrics

    def calculate_equity_curve_metrics(
        self,
        equity_curve: list[float],
    ) -> dict[str, Any]:
        """
        计算权益曲线指标

        Args:
            equity_curve: 权益曲线

        Returns:
            指标字典
        """
        equity = np.array(equity_curve)

        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max

        return {
            "peak_equity": float(running_max[-1]),
            "current_equity": float(equity[-1]),
            "max_drawdown": float(np.min(drawdown)),
            "max_drawdown_duration": self._calculate_max_dd_duration(drawdown),
        }

    @staticmethod
    def _calculate_max_dd_duration(drawdown: np.ndarray) -> int:
        """计算最大回撤持续时间"""
        in_dd = drawdown < 0
        if not np.any(in_dd):
            return 0

        max_duration = 0
        current_duration = 0

        for is_dd in in_dd:
            if is_dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0

        return max_duration
