# hypurr-monitor

Hyperliquid 实时监控系统，支持 ATR Channel、Clustering SuperTrend、WebSocket 实时行情（原生）、飞书 WebHook 推送。

## Features

- **Hyperliquid 原生 API**：REST + WebSocket（wss://api.hyperliquid.xyz/ws）
- **实时数据**：WebSocket allMids 流获取 mark price
- **技术指标**：ATR Channel、Clustering SuperTrend、Supertrend (双周期)、Vegas Tunnel
- **交易模式**：单一交易对监控 + 配对交易监控（如 BTC-ETH 价差）
- **告警系统**：飞书 WebHook 卡片推送
- **代理支持**：HTTP/HTTPS 代理（REST + WebSocket 共用配置）
- **高性能**：orjson、numpy/talib、K 线缓存、滑动窗口限速

## Architecture

```
hypurr-monitor/
├── config/          # 配置管理 (TOML)
├── hyperliquid/     # Hyperliquid 原生 API（REST + WebSocket）
├── indicators/      # 纯计算函数（talib/numpy）
├── models/          # 数据模型（Kline、Ticker）
├── signals/         # 信号检测逻辑（ATR Channel、Clustering SuperTrend）
├── notifications/   # 飞书 WebHook 格式化
├── candlesticks/    # K 线服务接口
├── service/         # NotificationService 业务流程编排
└── ui/              # Textual TUI（预留）
```

## Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) for environment management

## Installation

```bash
uv venv .venv
source .venv/Scripts/activate  # Windows
# uv sync
uv pip install -r requirements.txt
```

## Usage

```bash
# 默认运行（INFO level）
python main.py

# Debug 模式
python main.py --debug

# 带配置文件运行
python main.py --config config.toml

# 添加交易对
python main.py --config config.toml --add-symbol BTC,ETH

# 列出交易对
python main.py --config config.toml --list-symbols

# 后台运行（Linux/macOS）
python main.py --daemon

# 停止后台进程
python main.py --stop

# 查看状态
python main.py --status
```

## Configuration

Create `config.toml`:

```toml
[webhook]
url = "https://open.feishu.cn/open-apis/bot/v2/hook/xxx"
format = "card"

[symbols]
single_list = ["BTC", "ETH", "SOL", "HYPE", "XAU"]
pair_list = ["BTC-ETH"]

[atr_1h]
ma_type = "DEMA"
period = 14
mult = 1.618

[atr_15m]
ma_type = "HMA"
period = 14
mult = 1.3

[supertrend]
period1 = 9
multiplier1 = 2.5
period2 = 14
multiplier2 = 1.7

[vegas]
ema_signal = 9
ema_upper = 144
ema_lower = 169

[clustering_st]
min_mult = 1.0
max_mult = 5.0
step = 0.5
perf_alpha = 10.0
from_cluster = "Best"
max_iter = 1000
history_klines = 500

[clustering]
enabled = false
min_mult = 1.0
max_mult = 5.0
step = 0.5
perf_alpha = 10.0
from_cluster = "Best"
max_iter = 1000

[trailing]
enabled = true
atr_mult = 2.0

[service]
heartbeat_file = "heartbeat"
heartbeat_timeout = 120

[proxy]
enable = false
url = "http://127.0.0.1:7890"

[report]
enable = false
times = ["08:00", "20:00"]

[settings]
timezone = "Z"
max_log_lines = 1000
```

## Indicators

### ATR Channel
基于 1h ATR 的动态支撑/阻力通道，突破上轨做多，突破下轨做空，带 15m ATR 追踪止损。

### Clustering SuperTrend
K-Means 聚类分析 SuperTrend 多参数表现，动态选择最优参数组合，适用于配对交易。

### Supertrend (Dual Period)
双周期 Supertrend（9+2.5 和 14+1.7），用于趋势确认。

### Vegas Tunnel
三线 EMA（signal=9, upper=144, lower=169），用于趋势过滤。

## Development

```bash
# 运行测试
pytest

# 类型检查
mypy . --strict --ignore-missing-imports

# 代码检查
ruff check .

# 一键检查
ruff check . && mypy . --strict --ignore-missing-imports && pytest
```

## Hyperliquid API

- **REST**: `https://api.hyperliquid.xyz/info`
- **WebSocket**: `wss://api.hyperliquid.xyz/ws`
- **限速**: 1200 weight/min，滑动窗口 + burst 控制
- **文档**: https://hyperliquid.gitbook.io/hyperliquid-docs/

## Proxy Configuration

```toml
[proxy]
enable = true
url = "http://127.0.0.1:7890"
```

| Protocol | REST | WebSocket |
|----------|------|-----------|
| HTTP     | ok   | ok        |
| HTTPS    | ok   | ok        |
| SOCKS    | ok   | -         |

## License

MIT
