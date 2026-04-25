# PRD - 警报逻辑与推送链路稳定性改造

## 1. 背景

本项目是一个长期运行的 Hyperliquid 实时监控服务，当前已经具备以下能力：

- 1H ATR Channel 信号检测
- 4H ATR Channel 突破提醒
- Clustering SuperTrend 信号检测
- Trailing Stop 跟踪止损提醒
- Breakout 突破确认/失败提醒
- SYSTEM / ERROR / REPORT 等系统级通知
- 飞书 Webhook 推送

当前实现整体可运行，但在“长期运行可靠性、告警语义一致性、去重/冷却正确性、推送送达语义、状态污染风险、可测试性”方面存在明显短板。若不治理，实际运行中可能出现：

- 启动后把历史状态误判为新信号
- pair 价格使用陈旧腿数据导致误报
- trailing stop 后冷却未正确释放，影响再入场提醒
- breakout 判断口径与文档/预期不一致
- webhook 失败时上层仍认为告警已发出
- 系统/错误类通知缺乏统一出口去重，可能刷屏
- 静默吞异常导致监控逻辑失效但外部不可见

因此需要对警报与推送链路做分阶段、可验证的稳定性改造。

---

## 2. 目标

本次改造的核心目标：

1. **消除明显误报和漏报风险**
2. **统一告警状态语义与冷却语义**
3. **提升 webhook 推送可靠性与可观测性**
4. **避免运行态状态被无关逻辑污染**
5. **补足测试安全网，为后续重构铺路**

---

## 3. 非目标

本轮不做以下事项：

- 不重写整套通知架构
- 不一次性把全部 dict 状态改造成 dataclass
- 不引入数据库/消息队列等重基础设施
- 不修改交易策略本身（ATR、ClusterST 算法参数与计算公式不调整）
- 不调整飞书卡片 UI 风格

---

## 4. 当前问题清单

### P0 - 必须优先修复

#### 4.1 初始化后历史 ATR 状态可能被误报为新信号

当前 `initialize()` 会计算 benchmark，但没有同步初始化：

- `last_atr_state`
- `last_atr4h_state`
- `last_clustering_state`

结果：服务重启后，如果价格本来已处于轨道外，首个实时 tick 可能被当成新突破推送。

#### 4.2 pair 价格可能使用陈旧腿数据

当前 pair 价格计算仅检查左右腿价格是否大于 0，没有检查两腿更新时间是否新鲜。

结果：一条腿陈旧、另一条腿更新时，pair 仍会生成“伪新价格”，进而绕过 stale 检查触发信号。

#### 4.3 trailing stop 触发后清理了错误的 cooldown key

当前 trailing stop 触发后会写：

```python
last_alert_time[symbol] = 0
```

但真实冷却 key 是：

- `ATR_Ch_{symbol}`
- `ClusterST_{symbol}`

结果：止损后并未真正释放对应信号冷却，可能导致下一次再入场提醒异常。

#### 4.4 breakout 判断语义与实现不一致

文档与注释描述偏向按 close 判断，但当前实现用 `high` 进行判断，且 SHORT 方向也复用了同一口径，存在高概率语义错误。

#### 4.5 breakout 启动逻辑污染实时价格状态

breakout 启动时对 pair 组件价格写入了 0，且传入的是运行态价格字典，存在污染 `mark_prices` 的风险。

#### 4.6 局部异常被静默吞掉

多处 `suppress(Exception)` 无日志，可能导致：

- 信号检测失效
- trailing stop 失效
- 指标重算失败
- breakout 检查失败

但外部无感知。

### P1 - 建议本阶段纳入

#### 4.7 webhook 发送失败没有可靠送达语义

当前发送失败只记录日志，不向上层反馈。上层仍可能：

- 增加 alert count
- 写入 webhook log
- 认为消息已发出

#### 4.8 出口层没有统一 dedupe 保护

虽然 `AlertEvent` 已生成 `dedupe_key`，但出口层未使用，导致 SYSTEM / ERROR / BREAKOUT 缺乏统一防重复能力。

#### 4.9 单 symbol 异常可能影响整批 payload 处理

实时行情处理对 symbol 级别缺少充分隔离。坏 payload 或局部异常可能影响同批其他 symbol 的后续处理。

---

## 5. 产品原则

### 5.1 边沿触发优先于定时重复触发

“是否状态变化”和“是否允许发送”必须分离。

- 状态变化：负责记录市场状态是否发生变化
- 发送冷却：负责控制是否允许推送

不能把这两个语义混在一起。

### 5.2 运行时价格状态不得被非行情逻辑污染

`mark_prices` / `mark_price_times` 只能由行情更新链路维护，不允许 breakout 或通知逻辑写入伪价格。

### 5.3 失败必须可见

所有可恢复失败至少要写 warning；不可恢复失败要写 error；关键路径不允许静默吞没。

### 5.4 长期运行优先

方案优先考虑：

- 不误报
- 不漏报
- 不刷屏
- 不阻塞主行情处理链路
- 易测试

---

## 6. 范围

### 6.1 本轮范围（按阶段落地）

#### Phase 1：纠正错误语义与状态管理（本次优先执行）

1. 初始化时 seed 1H/4H/Cluster 状态
2. pair 价格计算加入双腿 freshness 校验
3. trailing stop 正确清理来源对应 cooldown key
4. 修正 breakout 判断口径，并统一 LONG/SHORT 语义
5. breakout 不得污染 `mark_prices`
6. 关键 `suppress(Exception)` 路径补上下文日志

#### Phase 2：通知出口可靠性增强

1. `WebhookSender.send_json()` 返回成功/失败语义
2. `AlertDispatcher` 区分“事件生成”和“发送成功”
3. 基于 `dedupe_key` 增加 TTL 去重
4. 视风险引入有界异步发送队列

#### Phase 3：结构治理与可维护性增强

1. `TrailingStopState` / `BreakoutMonitorState` dataclass 化
2. 统一 payload 字段类型
3. webhook.log 切换为更稳健的轮转方案
4. 更细粒度的可观测日志和指标

### 6.2 本轮不在范围内

- 全量架构重写
- 全模块状态对象化
- UI / 文案大改
- 新增第三方消息基础设施

---

## 7. 用户故事

### 7.1 作为长期运行的操作者

我希望服务重启后不要把旧状态当作新突破推送，避免误报。

### 7.2 作为策略使用者

我希望 pair 信号只基于新鲜数据计算，避免一条腿陈旧导致错误提醒。

### 7.3 作为风险控制使用者

我希望 trailing stop 触发后，下一次真正的新入场机会可以正常提醒，不要被错误 cooldown 卡住。

### 7.4 作为运维人员

我希望 webhook 推送失败时能明确感知，而不是系统自认为已经送达。

### 7.5 作为维护者

我希望关键路径失败时能在日志中看到上下文，而不是静默失效。

---

## 8. 功能需求

### FR-1 初始化状态种子

系统初始化完成前，必须根据当前 benchmark 同步设置：

- `last_atr_state`
- `last_atr4h_state`
- `last_clustering_state`

要求：启动后首个 WS tick 不得仅因历史状态存在而产生误报。

### FR-2 Pair freshness 校验

系统在计算 pair 价格前，必须验证左右腿价格时间戳均在 freshness 阈值内。

要求：任一腿 stale 时，不更新 pair 价格，也不触发 pair 级信号判断。

### FR-3 Trailing stop cooldown 清理

系统必须根据 trailing stop 来源清理正确的冷却 key。

要求：

- ATR 来源清理 `ATR_Ch_{symbol}`
- ClusterST 来源清理 `ClusterST_{symbol}`

### FR-4 Breakout 语义一致性

系统必须明确 breakout 使用的价格字段与判断方式，并对 LONG / SHORT 使用对称逻辑。

要求：代码、注释、测试三者一致。

### FR-5 运行态价格不可被 breakout 污染

breakout 模块不得写入 `mark_prices` / `mark_price_times`。

### FR-6 关键路径异常可见

关键 `except/suppress` 路径必须记录含 symbol/context 的 warning/error 日志。

---

## 9. 非功能需求

### NFR-1 可靠性

- 不得因单 symbol 异常导致整批 symbol 处理停止
- 不得因旧状态造成启动即误报

### NFR-2 性能

- 不引入显著额外网络请求
- 不在热路径引入重型同步 IO

### NFR-3 可测试性

本轮所有核心修复都必须补 pytest。

### NFR-4 可观测性

错误日志必须可定位到 symbol、阶段、原因。

---

## 10. 技术设计约束

1. 必须保持 `uv run` 工作流
2. 必须通过：
   - `uv run ruff check .`
   - `uv run mypy . --strict --ignore-missing-imports`
   - `uv run pytest`
3. 不允许一次性大爆改
4. 优先最小变更、可回滚
5. 不扩大 `NotificationService` 责任边界，优先在已有模块边界内修复

---

## 11. 分阶段实施计划

### Phase 1（本次）

#### 目标

优先消除误报/漏报/状态污染。

#### 交付项

1. 初始化状态 seed
2. pair freshness 校验
3. trailing stop 正确清 cooldown
4. breakout 语义修正
5. breakout 状态与实时价格隔离
6. 关键异常补日志
7. 对应测试

### Phase 2（下阶段）

#### 目标

提升通知送达可靠性和统一出口治理。

#### 交付项

1. webhook 返回结果语义
2. dispatcher 统一 dedupe
3. 可选发送队列
4. 对应测试

### Phase 3（后续）

#### 目标

降低维护成本、增强结构清晰度。

#### 交付项

1. dataclass 化核心状态
2. payload 类型统一
3. 日志轮转优化

---

## 12. 验收标准

### AC-1 启动不误报

给定服务启动前价格已在 ATR 上轨外，初始化完成后首个 WS tick 到来时，不产生新的 ATR 告警。

### AC-2 Pair stale 防护生效

给定 pair 一条腿 stale，另一条腿更新，则不生成新的 pair 价格、不触发 pair 信号。

### AC-3 Trailing stop 正确释放冷却

给定 ATR/ClusterST trailing stop 触发，系统清理对应来源的 cooldown key，后续新边沿信号可正常发送。

### AC-4 Breakout 判断一致

给定 LONG/SHORT breakout 场景，系统按统一定义正确输出 CONFIRMED / FALSE(REVERSE) / FALSE(NO_CONTINUATION)。

### AC-5 Breakout 不污染价格状态

启动 breakout monitor 前后，`mark_prices` 不被写入无效 0 值。

### AC-6 异常可观测

给定检测逻辑内部抛错，日志中可看到包含 symbol 与阶段信息的 warning/error。

### AC-7 验证通过

以下命令全部通过：

```bash
uv run ruff check .
uv run mypy . --strict --ignore-missing-imports
uv run pytest
```

---

## 13. 风险与回滚

### 风险

1. 修 breakout 语义后，历史“已习惯”的告警节奏可能变化
2. pair freshness 校验更严格后，pair 告警频率可能暂时下降
3. 初始化 seed 若实现错误，可能造成真正新信号被抑制

### 回滚策略

1. 每个阶段独立提交、独立验证
2. 本次只做 Phase 1 最小改动
3. 若 breakout 语义争议较大，先落测试再调整实现

---

## 14. 首批实施范围（给 @fixer）

本次只执行 **Phase 1**，范围限定如下：

1. 初始化阶段补种子状态
2. pair price 计算增加 freshness 防护
3. trailing stop 按来源正确清冷却 key
4. breakout 逻辑修正为一致且对称的判断口径
5. breakout 不再写运行态价格字典
6. 为上述改动补测试

本次先**不做**：

1. webhook 返回成功/失败语义
2. dispatcher dedupe TTL
3. 发送队列
4. dataclass 大重构

---

## 15. 建议修改文件

预计涉及：

- `PRD.md`
- `service/notification_service.py`
- `service/market_data_processor.py`
- `service/signal_coordinator.py`（如需最小配合）
- `signals/detection.py`
- `signals/breakout.py`
- `tests/` 下相关测试文件

---

## 16. 验证命令

```bash
uv run ruff check .
uv run mypy . --strict --ignore-missing-imports
uv run pytest
```

---

## 17. 最终完成清单（收尾快照）

### 17.1 Phase 1 完成情况

- [x] 初始化阶段补种子状态
- [x] pair price 计算增加 freshness 防护
- [x] trailing stop 按来源正确清冷却 key
- [x] breakout 逻辑修正为一致且对称的判断口径（close-based）
- [x] breakout 不再写运行态价格字典
- [x] 关键异常路径补上下文日志
- [x] 对应测试补齐

### 17.2 Phase 2 完成情况

- [x] `WebhookSender.send_json()` 返回成功/失败语义
- [x] 统一出口层支持区分事件生成、去重与发送结果
- [x] 基于 `dedupe_key` 增加 TTL 去重
- [x] 增加 dispatcher 统计：`attempted` / `sent` / `failed` / `deduped`
- [x] 有界异步发送队列（进程内 `asyncio.Queue` + worker，未启用时保持同步兼容路径）
- [x] 对应测试补齐

### 17.3 Phase 3 完成情况

- [x] `TrailingStopState` / `BreakoutMonitorState` dataclass 化（核心状态完成）
- [x] `RuntimeState` 第一阶段聚合，收敛 NotificationService 核心运行态引用并保留兼容别名
- [x] 主要告警路径 payload 字段类型统一为业务层传原始值、渲染层格式化
- [x] `webhook.log` 优化为更稳健的尾部保留式裁剪
- [x] 更细粒度的发送与去重可观测日志/统计

### 17.4 扩展架构升级（超出原 PRD 非目标，按用户要求追加）

- [x] 发送链路从同步直发扩展为“同步兼容 + 可选队列 worker”双模式
- [x] 为后续更大范围解耦建立统一通知边界与队列统计
- [x] 为全模块状态对象化建立 `RuntimeState` 聚合入口
- [x] 全模块状态对象化第二阶段：信号链路中的 `trailing_stop` / `breakout_monitor` 已转为 dataclass 优先，并保留 legacy 兼容
- [x] NotificationService 低风险职责拆分：pair 关系解析、symbol 状态清理、初始化 seed 已抽离到辅助模块
- [x] NotificationService 生命周期辅助拆分：`connect` / `reconnect` / `stop` 已抽离到独立 lifecycle helper
- [x] legacy dict 兼容层进一步收口：对象优先调用点已扩展到 service 层，兼容主要保留在状态 helper 边界
- [x] NotificationService 生命周期编排继续下沉：`run` / `initialize` 已抽离到 orchestration helper，NotificationService 进一步收敛为组装入口
- [x] legacy 兼容层最终进一步收口：主要兼容逻辑集中于状态 helper 边界，业务逻辑走对象主路径

### 17.5 验收标准对照

- [x] AC-1 启动不误报
- [x] AC-2 Pair stale 防护生效
- [x] AC-3 Trailing stop 正确释放冷却
- [x] AC-4 Breakout 判断一致
- [x] AC-5 Breakout 不污染价格状态
- [x] AC-6 异常可观测
- [x] AC-7 `ruff` / `mypy` / `pytest` 全部通过

### 17.6 本轮明确未做

- [ ] 全量架构重写（无必要，当前已完成核心治理）
- [ ] 引入数据库等新基础设施
- [ ] 飞书卡片 UI / 文案大改

### 17.7 当前结论

本 PRD 中的高优先级问题已完成治理；Phase 1 全部完成，Phase 2 完成并追加了进程内发送队列，Phase 3 已完成低风险高收益项、核心状态 dataclass 化、`RuntimeState` 聚合、第二阶段状态对象化、NotificationService 低风险职责拆分、生命周期 helper 拆分、`run` / `initialize` orchestration 下沉，以及 legacy 兼容层进一步收口。当前剩余未做项仅是非必要的全量重写级工作，不影响本 PRD 的核心验收结论。
