"""
跨市场特征生成器（预留接口）

生成市场整体相关的特征
"""


class CrossMarketFeatureGenerator:
    """
    跨市场特征生成器（预留）

    待实现功能：
    - BTC dominance
    - total market cap
    - funding rate correlation
    - open interest ratio
    """

    def generate(self, **kwargs) -> None:
        """
        生成跨市场特征

        包含：
        - BTC dominance: BTC 市值占比
        - total market cap: 总市值
        - funding rate correlation: 资金费率相关性
        - open interest ratio: 持仓量比率

        Raises:
            NotImplementedError: 跨市场特征待实现
        """
        raise NotImplementedError("跨市场特征待实现")
