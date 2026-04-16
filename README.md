# hypurr-monitor

Hyperliquid 实时监控系统。

支持：
- ATR Channel（1H 主信号 + 15m trailing stop + 4H breakout）
- Clustering SuperTrend（可选）
- Hyperliquid 原生 REST / WebSocket
- 飞书 WebHook 推送
- daemon / heartbeat / status 健康检查

---

## 给人和 Agent 的 30 秒入口

如果你是第一次进入这个项目，只看这几条就够了：

1. **只用 `uv` 管理环境与执行命令**
2. **主入口是 `main.py`，后台脚本是 `scripts/daemon.sh`**
3. **配置从 `config.example.toml` 复制成 `config.toml` 开始**
4. **做任何修改前，先看 `AGENTS.md`**
5. **改完默认要跑：**

```bash
uv run ruff check .
uv run mypy . --strict --ignore-missing-imports
uv run pytest
```

---

## 给 Agent 的项目工作约定（重要）

如果你是 AI Agent / 自动化协作者，请优先遵守下面这些约定：

- **先探索 Main Branch 当前实现，再设计，再修改**
- **强制使用中文沟通**（除非用户明确要求英文）
- **强制使用 `uv run` 执行 Python / pytest / mypy / ruff**
- **不要绕开 `AGENTS.md` 里的生产规范**
- **不要一上来大重构**，默认走最小可验证变更
- **这是长期运行的 production 服务**，优先考虑：
  - 网络超时 / retry / backoff
  - WS 重连 / ping-pong / 静默检测
  - 日志轮转 / 磁盘占用
  - 缓存边界 / 状态回收
  - heartbeat / status / daemon 一致性

建议进入项目后的阅读顺序：

1. `AGENTS.md`
2. `PRD.md`（如果存在当前任务 PRD）
3. `config.example.toml`
4. `main.py`
5. `service/notification_service.py`
6. `service/` 下新拆出的协调组件

---

## Agent Checklist

任何 Agent 进入这个项目，建议按这个顺序执行：

### Step 1: 先确认上下文

- 读 `AGENTS.md`
- 读 `README.md`
- 看当前是否有 `PRD.md`
- 看 `config.example.toml`

### Step 2: 改代码前先探索

- 先看 `main.py`
- 再看 `service/notification_service.py`
- 再看本次变更相关模块：
  - 网络相关：`hyperliquid/`、`service/ws_runtime_supervisor.py`
  - 告警相关：`notifications/`、`service/alert_dispatcher.py`
  - 行情处理：`service/market_data_processor.py`
  - 信号协调：`service/signal_coordinator.py`
  - 指标/信号：`signals/`、`indicators/`

### Step 3: 实施时必须遵守

- 只用 `uv run ...`
- 非微小修改先给最小变更方案
- 不要直接扩大 `NotificationService` 职责
- 优先复用现有 coordinator / dispatcher / gateway
- 新增函数必须有 docstring
- 网络相关必须考虑 timeout / retry / backoff / 日志

### Step 4: 改完必须验证

```bash
uv run ruff check .
uv run mypy . --strict --ignore-missing-imports
uv run pytest
```

### Step 5: 汇报时建议说明

- 改了哪些文件
- 为什么这么改
- 跑了哪些验证
- 是否涉及外网集成测试 / skip

---

## 项目当前结构

```text
hypurr-monitor/
├── config/                 # 配置加载、路径解析、网络配置
├── hyperliquid/            # Hyperliquid REST / WS / MarketGateway
├── indicators/             # 纯指标计算
├── models/                 # Kline 等数据模型
├── notifications/          # AlertEvent / webhook / sender / 常量与模板
├── service/                # 运行时协调层
│   ├── notification_service.py
│   ├── alert_dispatcher.py
│   ├── market_data_processor.py
│   ├── signal_coordinator.py
│   └── ws_runtime_supervisor.py
├── signals/                # ATR / clustering / breakout 业务判断
├── tests/                  # 单元 / 集成测试
├── main.py                 # CLI / 前台运行 / daemon 控制
├── config.example.toml     # 配置模板
├── AGENTS.md               # 项目协作与生产规范
└── README.md               # 项目入口文档
```

---

## 核心运行链路

### 运行时主链路

```text
main.py
  -> NotificationService
    -> MarketGateway          # REST / WS 网络接入
    -> WSRuntimeSupervisor    # receive / timeout / ping / reconnect
    -> MarketDataProcessor    # allMids 负载处理
    -> SignalCoordinator      # trailing / signals / breakout 协调
    -> AlertDispatcher        # AlertEvent -> webhook sender
```

### 当前告警类型

- `SYSTEM`
- `ERROR`
- `ATR_Ch`
- `ClusterST`
- `BREAKOUT`
- `REPORT`
- `CONFIG`
- `CONFIG ERROR`

### 当前主要策略语义

- **单标的**：1H ATR Channel 触发主信号，15m ATR trailing stop，额外有 4H breakout 告警
- **配对**：当 `clustering_st.enabled = true` 时走 clustering；否则走 ATR Channel 路径
- **breakout monitor**：现在已接入 runtime 主链路，会在新 ATR signal 后启动，并随着 15m cache 推进

---

## 环境要求

- Python 3.12+
- [uv](https://github.com/astral-sh/uv)

初始化：

```bash
uv venv .venv
uv sync
```

> 不要混用 `pip` / `poetry` / `conda run`。本项目唯一标准执行方式是 `uv run ...`

---

## 快速开始

### 1) 复制配置

```bash
cp config.example.toml config.toml
```

至少修改这些：

- `[webhook].url`
- `[symbols].single_list`
- `[symbols].pair_list`
- `[proxy]`（如果你需要代理）

### 2) 前台运行

```bash
uv run python main.py --config config.toml --debug
```

### 3) 后台运行

```bash
bash scripts/daemon.sh start
bash scripts/daemon.sh status
bash scripts/daemon.sh log
bash scripts/daemon.sh stop
```

### 4) Python CLI 常用命令

```bash
uv run python main.py --config config.toml --status
uv run python main.py --config config.toml --restart
uv run python main.py --config config.toml --list-symbols
uv run python main.py --config config.toml --add-symbol BTC,ETH
uv run python main.py --config config.toml --remove-symbol BTC-ETH
```

---

## 配置说明

请直接参考 `config.example.toml`，下面只列最关键的配置块。

### Webhook

```toml
[webhook]
url = "https://open.feishu.cn/open-apis/bot/v2/hook/your-webhook-here"
format = "card"
```

### 监控标的

```toml
[symbols]
single_list = ["BTC", "ETH", "SOL"]
pair_list = ["BTC-ETH"]
```

### ATR / Clustering

```toml
[atr_1h]
ma_type = "DEMA"
period = 14
mult = 1.618

[atr_4h]
ma_type = "DEMA"
period = 14
mult = 1.618

[atr_15m]
ma_type = "HMA"
period = 14
mult = 1.382

[clustering_st]
enabled = false
min_mult = 1.0
max_mult = 5.0
step = 0.5
perf_alpha = 10.0
from_cluster = "Best"
max_iter = 1000
history_klines = 500
```

### 服务与健康检查

```toml
[service]
heartbeat_file = "heartbeat"
heartbeat_timeout = 120
```

### 网络配置（统一入口）

```toml
[proxy]
enable = false
url = "http://127.0.0.1:7890"

[network.rest]
timeout_seconds = 30
max_retries = 2
base_delay_seconds = 1

[network.ws]
connect_timeout_seconds = 65
receive_timeout_seconds = 65
idle_timeout_seconds = 25
reconnect_base_delay_seconds = 2
reconnect_max_delay_seconds = 30

[network.webhook]
timeout_seconds = 10
max_retries = 2
base_delay_seconds = 1
```

---

## 健康检查 / daemon / 日志

### daemon health 的判断标准

状态不是只看 PID，而是同时看：

1. **进程是否还活着**
2. **heartbeat 是否仍在刷新**

`status/test` 共用同一份 Python 真相来源：

- `main.py --status`
- `scripts/daemon.sh status`
- `scripts/daemon.sh test`

### 当前运行时文件

这些文件默认都相对 `config.toml` 所在目录解析：

- `hypurr-monitor.pid`
- `hypurr-monitor.log`
- `debug.log`
- `error.log`
- `webhook.log`
- `heartbeat`

### 日志策略

- 控制台：INFO / WARNING / ERROR
- `debug.log`：debug 模式写入，已支持 rotation
- `error.log`：ERROR 始终落文件，已支持 rotation
- `webhook.log`：记录 webhook 发送历史

---

## 开发命令

### 必跑命令

```bash
uv run ruff check .
uv run mypy . --strict --ignore-missing-imports
uv run pytest
```

### 常用命令

```bash
uv run pytest -q
uv run pytest tests/test_signals.py -q
uv run pytest tests/test_integration.py -q
uv run ruff check --fix .
```

---

## 测试说明

### 单元测试

默认直接跑：

```bash
uv run pytest
```

### 集成测试

`tests/test_integration.py` 会访问真实 Hyperliquid REST / WS。

现在已做“外部网络抖动容忍”：

- 若真实外网暂时不可用，会 `skip`
- 不会因为瞬时网络抖动把整个测试基线打红

---

## 当前重构方向（给未来协作者）

项目已经从早期的“大型 `NotificationService`”逐步拆出：

- `MarketGateway`
- `AlertDispatcher`
- `MarketDataProcessor`
- `SignalCoordinator`
- `WSRuntimeSupervisor`

下一步原则仍然是：

1. **不碰指标逻辑时，优先拆 orchestration / network / alert 边界**
2. **保留兼容接口，先抽 façade，再迁移内部实现**
3. **任何新增重构都要带测试**

---

## 对 Agent 最重要的文件

如果你是新进来的 Agent，优先读这几个：

1. `AGENTS.md`
2. `README.md`
3. `config.example.toml`
4. `main.py`
5. `service/notification_service.py`
6. `service/` 下 4 个新协调组件
7. `notifications/alert_event.py`
8. `hyperliquid/market_gateway.py`

---

## 参考

- Hyperliquid REST: `https://api.hyperliquid.xyz/info`
- Hyperliquid WebSocket: `wss://api.hyperliquid.xyz/ws`
- 官方文档: https://hyperliquid.gitbook.io/hyperliquid-docs/

---

## License

MIT
