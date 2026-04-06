# ccxt-monitor

加密货币实时监控系统，支持多交易所（Binance Futures）、多指标（ATR Channel、Clustering SuperTrend）、WebSocket 实时行情、飞书 WebHook 推送。

## 功能特性

- **多交易所支持**：通过 CCXT 库连接 Binance Futures
- **实时数据**：WebSocket 流式接收实时市场数据
- **技术指标**：ATR Channel、Clustering SuperTrend、Supertrend、Vegas Tunnel
- **告警系统**：飞书 WebHook 通知推送
- **代理支持**：REST 和 WebSocket 均支持 HTTP/SOCKS 代理

## 项目架构

```
ccxt-monitor/
├── config/          # 配置管理 (TOML)
├── indicators/      # 纯计算函数
├── models/          # 数据模型 (Kline, Ticker, PairState)
├── rest_api/        # CCXT REST 客户端
├── websocket/       # CCXT Pro WebSocket 订阅
├── signals/         # 信号检测逻辑
├── notifications/   # 飞书 WebHook 格式化
├── candlesticks/    # 微服务接口
├── service/         # 主通知服务
└── ui/              # Textual TUI（预留）
```

## 环境要求

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) 环境管理工具

## 安装

```bash
# 安装依赖
uv sync
uv sync --group dev  # 开发依赖
```

## 配置示例

创建 `config.toml`:

```toml
[exchange]
exchange_id = "binance"
proxy_enable = false
proxy_url = ""

[webhook]
webhook_url = "https://open.feishu.cn/open-apis/bot/v2/hook/xxx"
webhook_format = "card"

[symbols]
single_list = ["BTCUSDT", "ETHUSDT"]
pair_list = []

[indicators]
atr1h_period = 14
atr1h_ma_type = "EMA"
atr1h_mult = 3.0
atr15m_period = 14
atr15m_ma_type = "EMA"
atr15m_mult = 2.0
atr_channel_mult = 1.5
supertrend_period = 10
supertrend_mult = 3.0
vegas_tunnel_ema_signal = 9
vegas_tunnel_ema_upper = 144
vegas_tunnel_ema_lower = 169

[clustering]
enabled = false
min_mult = 1.0
max_mult = 5.0
step = 0.5
perf_alpha = 0.05
from_cluster = "next"
max_iter = 100

[trailing]
enabled = true
atr_mult = 2.0

[breakout]
enabled = true
lookback = 20
threshold = 0.02

[system]
timezone = "+08:00"
log_retention_days = 7
max_log_lines = 10000
```

## 使用方法

```bash
# 运行监控 (INFO 级别)
uv run python main.py

# 调试模式 (DEBUG 级别)
uv run python main.py --debug

# 添加交易对
uv run python main.py --add-symbol BTC/USDT:USD,ETH/USDT:USD

# 后台运行 (Linux)
uv run python main.py --daemon

# 停止后台进程
uv run python main.py --stop

# 查看状态
uv run python main.py --status
```

## 指标说明

### ATR Channel
基于 ATR 波段的追踪止损机制，动态计算支撑/阻力位。

### Clustering SuperTrend
对 SuperTrend 绩效因子进行 K-Means 聚类，识别市场状态（趋势/震荡）。

### Supertrend
经典 Supertrend 指标，利用 ATR 进行趋势判断。

### Vegas Tunnel
三条 EMA 线（信号线、上轨、下轨），用于趋势确认。

## 开发调试

```bash
# 运行测试
pytest -v

# 类型检查
mypy . --strict --ignore-missing-imports

# 代码规范检查
ruff check .

# 完整检查
ruff check . && mypy . --strict --ignore-missing-imports && pytest -v
```

## 代理配置

```toml
[proxy]
enable = true
url = "http://127.0.0.1:7890"
```

| 协议 | REST (httpProxy) | WebSocket (httpsProxy) |
|------|------------------|------------------------|
| HTTP     | ✅ 支持         | ✅ 支持               |
| HTTPS    | ✅ 支持         | ✅ 支持               |
| SOCKS    | ✅ 支持         | ❌ 不支持             |

## 许可证

MIT
