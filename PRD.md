# ML 信号系统 PRD

> 多配对交易 CatBoost+LSTM 信号预测系统

---

## 一、项目概述

### 1.1 目标

构建一个**多配对交易**的机器学习信号系统，支持：
- 三分类信号：**涨(UP) / 跌(DOWN) / 平(NEUTRAL)**
- 概率预测：精确率、召回率等指标
- 配对交易汇率和标的价格关系学习
- CatBoost 主模型 + LSTM 残差修正

### 1.2 配对列表

| 配对名称 | 标的 A | 标的 B | 数据源 | 备注 |
|---------|--------|--------|--------|------|
| BTC-ETH | BTCUSDT | ETHUSDT | Binance | 主力配对 |
| BTC-SOL | BTCUSDT | SOLUSDT | Binance | - |
| ETH-SOL | ETHUSDT | SOLUSDT | Binance | - |
| xyz:GOLD-xyz:SILVER | GLDUSDT | SLVUSDT | Hyperliquid | 数据量少，需特殊处理 |

### 1.3 核心参数

| 参数 | 值 |
|------|-----|
| 时间帧 | 15m |
| 历史数据 | 2 年 |
| 分类数 | 3 (UP/DOWN/NEUTRAL) |
| 平的定义 | `\|return\| <= ATR * 0.5` |
| Taker 手续费 | 0.05% |
| Maker 手续费 | 0.02% |
| 资金费率 | 年化 10.95% |

---

## 二、模型架构

```
┌──────────────────────────────────────────────────────────────────┐
│                     Ensemble Signal Output                         │
│                  [UP, DOWN, NEUTRAL] + [P(UP), P(DOWN), P(NEUT)] │
└──────────────────────────────────────────────────────────────────┘
                              ↑
                    ┌─────────┴─────────┐
                    │                   │
              ┌─────┴─────┐       ┌─────┴─────┐
              │  CatBoost │       │    LSTM   │
              │  主模型    │       │  残差修正  │
              └───────────┘       └───────────┘
```

---

## 三、实施阶段

### 阶段一：基础设施

| 任务 | 状态 | 备注 |
|------|------|------|
| ml_common.py - 共享类型定义 | ✅ DONE | SignalType, BacktestResult, MLConfig |
| data/pair_registry.py - 配对注册表 | ✅ DONE | BTC-ETH, BTC-SOL, ETH-SOL, xyz:GOLD-xyz:SILVER |
| data/binance_client.py - Binance API 客户端 | ✅ DONE | fetch_klines, fetch_funding_rate |
| data/storage.py - Parquet 数据存储 | ✅ DONE | DataStorage, ParquetStorage |
| data/binance_fetcher.py - 数据获取器 | ✅ DONE | 自动获取并存储 K 线 |
| config.toml 扩展 | ✅ DONE | [ml] 配置段 |

### 阶段二：特征工程

| 任务 | 状态 | 备注 |
|------|------|------|
| ml/features/base.py - FeatureGenerator 抽象基类 | ✅ DONE | SinglePair/MultiPair/CrossMarket |
| ml/features/single_pair.py - 单标的特征 | ✅ DONE | ATR, RSI, MACD, SMA, EMA, SuperTrend |
| ml/features/multi_pair.py - 配对特征 | ✅ DONE | Ratio, Spread, Hurst, HalfLife |
| ml/features/cross_market.py - 跨市场特征 | ✅ DONE | **留空接口** |

### 阶段三：标注与模型

| 任务 | 状态 | 备注 |
|------|------|------|
| ml/labels/base.py - Labeler 抽象基类 | ✅ DONE | - |
| ml/labels/three_class.py - 三分类标注 | ✅ DONE | ATR * neutral_scale |
| ml/model/base.py - ModelBackend 抽象基类 | ✅ DONE | - |
| ml/model/catboost_backend.py - CatBoost 实现 | ✅ DONE | GPU 加速 |
| ml/model/lightgbm_backend.py - LightGBM 实现 | ✅ DONE | **备选** |
| ml/model/xgboost_backend.py - XGBoost 实现 | ✅ DONE | **备选** |

### 阶段四：LSTM 残差

| 任务 | 状态 | 备注 |
|------|------|------|
| ml/model/lstm_residual.py - LSTM 残差学习 | ✅ DONE | TensorFlow |
| ml/pipeline/trainer.py - 训练流水线 | ✅ DONE | 完整流程 |

### 阶段五：回测引擎

| 任务 | 状态 | 备注 |
|------|------|------|
| backtest/commission.py - 手续费计算 | ✅ DONE | Taker 0.05%, Maker 0.02% |
| backtest/funding.py - 资金费率 | ✅ DONE | 年化 10.95% |
| backtest/engine.py - 回测引擎 | ✅ DONE | - |
| backtest/metrics.py - 绩效指标 | ✅ DONE | - |
| backtest/risk.py - 风险管理 | ✅ DONE | **留空接口** |

### 阶段六：策略与集成

| 任务 | 状态 | 备注 |
|------|------|------|
| strategy/signal_selector.py - 信号选择器 | ✅ DONE | MLSignalStrategy |
| MLSignalStrategy 类 - ML 策略封装 | ✅ DONE | 模型加载+预测 |
| 扩展策略分发 - 支持 "ml" 策略 | ✅ DONE | _get_strategy_for_symbol |
| _ct_check_ml_signals - ML 信号检测入口 | ✅ DONE | 生成特征+预测 |
| NotificationService 初始化加载模型 | ✅ DONE | initialize() |
| 配置扩展 - ml.model_path | ✅ DONE | config.toml |
| scripts/train_ml.py - 训练脚本 | ✅ DONE | 训练入口 |

---

## 六、ML 与 NotificationService 集成

### 6.1 集成架构

```
NotificationService
├── ml_strategy: MLSignalStrategy
│   ├── catboost: CatBoostBackend
│   └── lstm: LSTMResidualModel
│
├── _ct_check_signals_by_strategy()
│   ├── atr_channel
│   ├── clustering_st
│   └── ml → _ct_check_ml_signals()
│
└── _ct_check_ml_signals()
    ├── 生成特征
    ├── 模型预测
    └── 触发 _send_webhook
```

### 6.2 集成点

1. **MLSignalStrategy** (`strategy/signal_selector.py`):
   - 封装 ML 模型加载和预测
   - `load_models(path)` - 加载预训练模型
   - `predict(features)` - 返回 (signal, probability)

2. **策略分发扩展** (`notification_service.py`):
   - `_get_strategy_for_symbol()` 返回 "ml" 时
   - 调用 `_ct_check_ml_signals()`

3. **ML 信号检测** (`notification_service.py`):
   - `_ct_check_ml_signals()`:
     - 从 `kline_cache` 获取 K 线
     - 调用 `SinglePairFeatureGenerator.generate()`
     - 调用 `ml_strategy.predict()`
     - 触发 `_send_webhook()`

4. **模型加载** (`notification_service.py`):
   - `initialize()` 时加载 ML 模型
   - 懒加载：仅当配置 `strategy = "ml"` 时加载

### 6.3 配置扩展

```toml
[ml]
enable = true
model_path = "models/ml/"

[symbols]
single_strategy = "ml"      # ML 策略
pair_strategy = "ml"        # ML 策略
```

---

## 四、待定功能（留空接口）

### 4.1 跨市场特征 (ml/features/cross_market.py)

```python
class CrossMarketFeatureGenerator:
    """跨市场特征生成器（待实现）"""
    
    def generate(self, klines: dict[str, list[Kline]]) -> pd.DataFrame:
        """
        生成跨市场特征：
        - BTC dominance
        - total market cap
        - funding rate correlation
        - open interest ratio
        """
        raise NotImplementedError("跨市场特征待实现")
```

### 4.2 风险管理 (backtest/risk.py)

```python
class RiskManager:
    """风险管理器（待实现）"""
    
    def check_position_size(self, signal: Signal, capital: float) -> float:
        """检查仓位大小"""
        raise NotImplementedError("风险管理待实现")
    
    def check_drawdown(self, equity_curve: list[float]) -> bool:
        """检查回撤限制"""
        raise NotImplementedError("风险管理待实现")
```

---

## 五、验收标准

### 5.1 功能验收

- [ ] 能从 Binance 获取 2 年 15m K 线数据
- [ ] 能为 4 个配对生成特征
- [ ] CatBoost 模型能输出三分类 + 概率
- [ ] 回测引擎正确计算手续费和资金费率
- [ ] 能输出精确率/召回率等概率指标

### 5.2 性能验收

- [ ] CatBoost GPU 训练时间 < 30 分钟（2年数据）
- [ ] 单次预测时间 < 100ms
- [ ] 模型文件大小 < 500MB

---

## 六、技术栈

| 组件 | 技术 |
|------|------|
| 主模型 | CatBoost (GPU) |
| 备选模型 | LightGBM, XGBoost |
| 残差学习 | TensorFlow LSTM |
| 超参优化 | Optuna |
| 数据存储 | Parquet |
| 数据源 | Binance Futures API |

---

## 七、依赖

```toml
[project.optional-dependencies]
ml = [
    "catboost>=1.2",
    "lightgbm>=4.0",
    "xgboost>=2.0",
    "optuna>=3.0",
    "tensorflow>=2.15",
    "scikit-learn>=1.3",
    "pandas>=2.0",
    "pyarrow>=14.0",
]
```

---

## 八、变更日志

| 日期 | 版本 | 变更 |
|------|------|------|
| 2026-04-10 | v0.1 | 初始 PRD |

