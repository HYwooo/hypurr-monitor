# hypurr-monitor

Hyperliquid 实时监控系统，支持 ATR Channel、Clustering SuperTrend、WebSocket 实时行情（原生）、飞书 WebHook 推送。

## 环境要求

- Python >= 3.12（由 uv 自动管理，无需用户提前安装）
- 全程使用 `uv` 管理虚拟环境和依赖，无需手动安装任何包

## 快速启动

### 1. 安装 uv（如没有）

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. 初始化项目

```bash
cd hypurr-monitor

# 创建虚拟环境 + 安装依赖（一行命令）
uv sync
```

### 3. 配置

```bash
cp config.example.toml config.toml
```

编辑 `config.toml`：

| 配置项 | 必填 | 说明 |
|--------|------|------|
| `[webhook] url` | 是 | 飞书 WebHook URL |
| `[symbols] single_list` | 是 | 单标的监控列表，如 `["BTC", "ETH"]` |
| `[symbols] pair_list` | 否 | 配对交易列表，如 `["BTC-ETH", "BTC-xyz:GOLD"]` |
| `[proxy] enable/url` | 否 | 如需代理则启用 |

### 4. 运行

```bash
# 前台运行（DEBUG 模式）
uv run python main.py --config config.toml --debug

# 前台运行（INFO 模式）
uv run python main.py --config config.toml
```

### 5. 后台运行（守护进程）

```bash
# 启动
bash scripts/daemon.sh start

# 查看状态
bash scripts/daemon.sh status

# 停止
bash scripts/daemon.sh stop

# 重启
bash scripts/daemon.sh restart

# 查看日志
bash scripts/daemon.sh log

# 连接测试
bash scripts/daemon.sh test
```

### 常用操作

```bash
# 添加交易对
uv run python main.py --config config.toml --add-symbol BTC,ETH

# 移除交易对
uv run python main.py --config config.toml --remove-symbol SOL

# 列出当前交易对
uv run python main.py --config config.toml --list-symbols
```

## 开发

```bash
# 类型检查
uv run mypy . --strict --ignore-missing-imports

# 代码检查
uv run ruff check .

# 自动修复
uv run ruff check --fix .

# 运行测试
uv run pytest

# 一键检查
uv run ruff check . && uv run mypy . --strict --ignore-missing-imports && uv run pytest
```

## 配置说明

详见 `config.example.toml`，主要配置项：

| Section | 说明 |
|---------|------|
| `webhook` | 飞书 WebHook URL 和消息格式 |
| `symbols` | 监控的交易对（single_list 单标的，pair_list 配对） |
| `atr_1h` | 1小时 ATR Channel 参数 |
| `atr_15m` | 15分钟 ATR Channel 参数 |
| `clustering_st` | 聚类超级趋势参数 |
| `proxy` | 代理设置 |
| `settings` | 时区、日志行数等全局设置 |

## Hyperliquid API

- **REST**: `https://api.hyperliquid.xyz/info`
- **WebSocket**: `wss://api.hyperliquid.xyz/ws`
- **限速**: 1200 weight/min，滑动窗口 + burst 控制
- **文档**: https://hyperliquid.gitbook.io/hyperliquid-docs/

## 架构

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
├── scripts/         # 守护进程脚本
└── ui/              # Textual TUI（预留）
```

## License

MIT
