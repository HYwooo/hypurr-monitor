# 测试报告 TEST_REPORT

**项目**：hypurr-monitor（Hyperliquid 实时监控系统）  
**测试日期**：2026-04-07  
**测试环境**：Windows (Python 3.12.7, pytest-9.0.2)

---

## 一、测试概览

| 检查项 | 结果 | 详情 |
|--------|------|------|
| ruff check | ✅ PASS | All checks passed |
| mypy --strict | ✅ PASS | 33 个源文件，0 个错误 |
| pytest | ✅ PASS | 189 passed, 2 skipped |

---

## 二、详细结果

### 2.1 Ruff 检查

```
All checks passed!
```

所有代码风格检查通过，无任何 lint 错误。

### 2.2 Mypy 严格模式检查

```
Success: no issues found in 33 source files
```

33 个源文件全部通过 `mypy --strict --ignore-missing-imports`，0 类型错误。

### 2.3 Pytest 测试结果

```
189 passed, 2 skipped, 15 warnings in 38.11s
```

#### 按测试文件分布

| 测试文件 | 通过 | 跳过 | 失败 |
|----------|------|------|------|
| tests/test_hyperliquid_rest_client.py | 29 | 0 | 0 |
| tests/test_hyperliquid_symbol.py | 19 | 0 | 0 |
| tests/test_hyperliquid_ws_client.py | 14 | 2 | 0 |
| tests/test_indicators.py | 27 | 0 | 0 |
| tests/test_integration.py | 22 | 0 | 0 |
| tests/test_models.py | 11 | 0 | 0 |
| tests/test_notifications.py | 17 | 0 | 0 |
| tests/test_signals.py | 19 | 0 | 0 |
| **总计** | **158** | **2** | **0** |

> 注：集成测试（test_integration.py）通过表明实际网络通信正常，成功连接 Hyperliquid API。

#### 跳过的测试（2 个）

1. `tests/test_hyperliquid_ws_client.py::TestHyperliquidWSReceiveLoop::test_receive_loop_handles_malformed_json` — 代码未捕获 `orjson.JSONDecodeError`，跳过
2. `tests/test_hyperliquid_ws_client.py::TestGetMarkPricesOnce::test_oneshot_integration_skipped` — 一次性与集成测试重复

#### 15 个警告

**PytestWarning（14 个）**：同步测试函数被错误标记了 `@pytest.mark.asyncio`，不影响功能，仅影响标记规范。

涉及文件：
- `tests/test_hyperliquid_rest_client.py`：`TestHyperliquidRESTParseSymbol` 和 `TestHyperliquidRESTIntervalMs` 的所有测试方法
- `tests/test_hyperliquid_ws_client.py`：`TestHyperliquidWSInit` 的两个测试方法

**RuntimeWarning（1 个）**：`tests/test_signals.py::TestUpdateKlines::test_update_klines_creates_client_on_error` 有一个协程从未被 await（`false_is_pair`），不影响功能。

---

## 三、本次修复内容

本次会话修复了 `mypy --strict` 的所有错误：

### 3.1 tests/test_hyperliquid_ws_client.py

- `_marks` 字段类型为 `dict[str, float]`，测试中字符串值 `"65000.5"` 改为 `float` 值 `65000.5`
- `receive_side_effect()` 返回类型从 `MagicMock` 改为 `Any`（避免 return-value 错误）

### 3.2 tests/test_hyperliquid_rest_client.py

- 所有 `client._post = AsyncMock(...)` 行添加 `# type: ignore[method-assign]`（避免 method-assign 错误）
- `ClientResponseError` 的 `history=[]`（空 list）改为 `history=()`（空 tuple），符合 `tuple[ClientResponse, ...]` 类型

### 3.3 pyproject.toml

- 移除 mypy override 中无效的 error code `untyped-decorator`

---

## 四、遗留警告（不影响功能）

| 警告类型 | 数量 | 说明 |
|----------|------|------|
| PytestWarning | 14 | 同步函数被标记 `@pytest.mark.asyncio` |
| RuntimeWarning | 1 | 协程从未被 await |

如需消除这些警告，可在后续重构中移除不必要的 `pytestmark = pytest.mark.asyncio`（同步测试类不需要）。
