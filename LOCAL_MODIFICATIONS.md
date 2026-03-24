# Local Modifications (Fork Customizations)

> 本文档记录所有相对于上游 `HKUDS/nanobot` 的本地修改。
> 合并上游更新时，以此文档为准恢复本地功能。
> 最后更新：2026-03-24，基于 merge commit with origin/main (`1d58c9b`)

---

## 1. 斜杠命令系统 (`nanobot/command/builtin.py` + `nanobot/agent/loop.py`)

上游已引入 CommandRouter 模式 (`nanobot/command/router.py`)，本地命令已迁移到此框架。

### 上游内置命令

| 命令 | 功能 | 实现位置 |
|------|------|----------|
| `/new` | 开始新对话，归档当前会话 | `builtin.py::cmd_new` |
| `/stop` | 停止当前任务 | `builtin.py::cmd_stop` |
| `/restart` | 重启 bot | `builtin.py::cmd_restart` |
| `/status` | 显示运行状态 | `builtin.py::cmd_status` |
| `/help` | 显示可用命令 | `builtin.py::cmd_help` |

### 本地新增命令（注册在 `builtin.py::register_builtin_commands`）

| 命令 | 功能 | 实现位置 |
|------|------|----------|
| `/clear` | 清除对话历史（不归档） | `builtin.py::cmd_clear` |
| `/compact [--switch] [model]` | 压缩对话 / 切换 compact model | `builtin.py::cmd_compact` |
| `/skills` | 列出 workspace 可用技能 | `builtin.py::cmd_skills` |
| `/model [number\|name]` | 显示/切换当前模型 | `builtin.py::cmd_model` |
| `/mcp` | 列出 MCP 服务器及连接状态 | `builtin.py::cmd_mcp` |

### 关键实现细节

- `/model` 支持序号、名称、模糊匹配三种方式
- `/help` 已更新包含所有本地命令
- helper 方法仍在 `loop.py` 中：`_list_skills`, `_list_mcp_servers`, `_format_model_list`, `_switch_model` 等

---

## 2. 运行时模型切换 (`nanobot/agent/loop.py`)

### AgentLoop 新增参数

```python
def __init__(self, ...,
    compact_model: str = "",            # compact/consolidation 用的模型
    provider_factory: Callable | None,  # 按模型名动态创建 provider
    available_models: list[str] | None, # config 中配置的可用模型列表
    data_dir: Path | None,              # 状态持久化目录
)
```

### 新增方法（在 `loop.py`）

| 方法 | 用途 |
|------|------|
| `_provider_key(model)` | 从模型名提取 provider 前缀 |
| `_switch_model(arg)` | 切换主模型，更新 provider/subagent |
| `_switch_compact_model(arg)` | 切换 compact model |
| `_persist_state()` | 将 model/compact_model 写入 `.state.json` |
| `load_persisted_state(data_dir)` | 启动时恢复模型选择 |
| `_format_model_list()` | 格式化模型列表展示 |
| `_format_compact_model_list()` | 格式化 compact model 列表 |
| `_list_skills()` | 列出 workspace skills |
| `_list_mcp_servers()` | 列出 MCP 服务器 |

### 状态持久化

- 文件: `<data_dir>/.state.json`
- 内容: `{"model": "...", "compact_model": "..."}`
- 重启 gateway 后自动恢复

---

## 3. MemorySearchTool (`nanobot/agent/tools/memory_tool.py`)

- 上游不存在此文件
- BM25 搜索工具，可在 memory 目录中搜索相关记忆片段
- 在 `_register_default_tools()` 中注册

---

## 4. Claude OAuth Provider (本地独有)

| 文件 | 说明 |
|------|------|
| `nanobot/providers/claude_oauth_provider.py` | 复用 Claude Code OAuth token |
| `nanobot/providers/claude_oauth_auth.py` | OAuth token 发现和刷新 |

### 关键实现

- `_fix_trailing_assistant()` — 将尾部 assistant 消息转为 user role（OAuth 端点不支持 assistant prefill）
- `_ensure_system_prefix()` — 确保 Claude Code system prefix 存在

### 注册位置

- `nanobot/providers/registry.py` — `ProviderSpec(name="claude_oauth", ...)`
- `nanobot/providers/__init__.py` — lazy-import 中包含 `ClaudeOAuthProvider`
- `nanobot/cli/commands.py` — `_make_provider()` 中 `claude_oauth` 分支

---

## 5. CLI 初始化 (`nanobot/cli/commands.py`)

### `_make_provider(config, model=None)`

本地版接受可选 `model` 参数，支持为不同模型创建不同 provider。
新增 `claude_oauth` 处理分支。

### `gateway()` 和 `agent()` 命令

传递给 `AgentLoop` 的额外参数：

```python
AgentLoop(
    ...,
    compact_model=compact_model,
    provider_factory=lambda m: _make_provider(config, m),
    available_models=config.agents.defaults.models,
    data_dir=data_dir,
)
```

启动时从 `.state.json` 恢复 model/compact_model 选择。

---

## 6. Config Schema (`nanobot/config/schema.py`)

### AgentDefaults 新增字段

```python
models: list[str] = []        # /model 可切换的模型列表
compact_model: str = ""       # compact/consolidation 用的模型
memory_window: int = 50       # 上游已移除 (deprecated)，本地保留为 active 字段
```

### `should_warn_deprecated_memory_window`

上游已移除 `memory_window` 字段，本地保留并让 `should_warn_deprecated_memory_window` 永远返回 `False`。

### SubagentConfig

上游不存在，本地新增：

```python
class SubagentConfig(Base):
    exec_timeout: int = 300
    max_iterations: int = 100
    max_duration: int = 600
```

---

## 7. 进度消息抑制 (`nanobot/agent/loop.py`)

`_bus_progress` 直接 `return`，不向任何 channel 发送进度消息。

```python
async def _bus_progress(content: str, *, tool_hint: bool = False) -> None:
    return  # Suppress all progress messages to channels
```

---

## 8. Provider registry (`nanobot/providers/registry.py`)

- Ollama provider: `strip_model_prefix=True`（上游为 `False`），使 `ollama/model` 正确路由

---

## 9. 辅助脚本 (本地独有文件)

| 文件 | 说明 |
|------|------|
| `scripts/sync_upstream.sh` | 上游同步脚本 |
| `scripts/validate_sync.py` | 同步后验证脚本 |

---

## 合并策略

合并上游时，对于冲突文件：

1. **loop.py** — 接受上游框架（CommandRouter、streaming、并发锁等），保留 helper 方法（_switch_model 等）
2. **command/builtin.py** — 接受上游基础命令，补回本地 /clear, /compact, /skills, /model, /mcp，更新 /help 列表
3. **commands.py** — 接受上游重构，补回 `_make_provider` 的 model 参数、claude_oauth 分支、provider_factory 传递、state 恢复
4. **schema.py** — 接受上游重构，补回 models / compact_model / SubagentConfig / memory_window (active)
5. **providers/registry.py** — 接受上游新 provider，补回 claude_oauth ProviderSpec + ollama strip_prefix
6. **providers/__init__.py** — 接受上游 lazy-import，补回 ClaudeOAuthProvider
7. **memory_tool.py / claude_oauth_*.py** — 本地独有文件，直接保留
8. **tests/test_commands.py** — `_make_provider` mock 需要加 `_model=None` 参数；`_FakeAgentLoop` 需要 `load_persisted_state`
9. **tests/test_config_migration.py** — `memoryWindow` 断言保留 active（非 deprecated）
10. **tests/test_providers_init.py** — `__all__` 断言需包含 `ClaudeOAuthProvider`
