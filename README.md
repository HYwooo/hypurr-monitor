# ccxt-monitor

Cryptocurrency real-time monitoring system with multi-exchange support (Binance Futures), featuring ATR Channel and Clustering SuperTrend indicators, WebSocket real-time data, and Feishu webhook alerts.

## Features

- **Multi-Exchange Support**: Binance Futures via CCXT library
- **Real-Time Data**: WebSocket streaming for live market data
- **Technical Indicators**: ATR Channel, Clustering SuperTrend, Supertrend, Vegas Tunnel
- **Alert System**: Feishu webhook notifications
- **Proxy Support**: HTTP/SOCKS proxy for REST and WebSocket

## Architecture

```
ccxt-monitor/
├── config/          # Configuration management (TOML)
├── indicators/      # Pure calculation functions
├── models/          # Data models (Kline, Ticker, PairState)
├── rest_api/        # CCXT REST client
├── websocket/       # CCXT Pro WebSocket subscriptions
├── signals/         # Signal detection logic
├── notifications/    # Feishu webhook formatting
├── candlesticks/    # Microservice interface
├── service/         # Main NotificationService
└── ui/              # Textual TUI (reserved)
```

## Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) for environment management

## Installation

```bash
uv sync
uv sync --group dev  # with dev dependencies
```

## Usage

```bash
# Default run (INFO level)
uv run python main.py

# Debug mode
uv run python main.py --debug

# Add symbols
uv run python main.py --add-symbol BTC/USDT:USD,ETH/USDT:USD

# List symbols
uv run python main.py --list-symbols

# Run in background (Linux)
uv run python main.py --daemon

# Stop daemon
uv run python main.py --stop

# Check status
uv run python main.py --status
```

## Configuration

Create `config.toml`:

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

## Usage

```bash
# Run with config
python main.py --config config.toml

# Run with debug logging
python main.py --config config.toml --debug

# Add symbols
python main.py --config config.toml --add-symbol BTCUSDT,ETHUSDT

# List symbols
python main.py --config config.toml --list-symbols
```

## Indicators

### ATR Channel
Trailing stop mechanism using ATR bands for dynamic support/resistance levels.

### Clustering SuperTrend
K-Means clustering on SuperTrend performance factors to identify market regimes.

### Supertrend
Classic Supertrend indicator using ATR for trend detection.

### Vegas Tunnel
Three EMA lines (signal, upper, lower) for trend confirmation.

## Development

```bash
# Run tests
pytest -v

# Type checking
mypy . --strict --ignore-missing-imports

# Linting
ruff check .

# All checks
ruff check . && mypy . --strict --ignore-missing-imports && pytest -v
```

## Proxy Configuration

```toml
[proxy]
enable = true
url = "http://127.0.0.1:7890"
```

| Protocol | REST (httpProxy) | WebSocket (httpsProxy) |
|----------|------------------|------------------------|
| HTTP     | ok               | ok                     |
| HTTPS    | ok               | ok                     |
| SOCKS    | ok               | -                      |

## License

MIT
