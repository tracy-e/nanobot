# Local Modifications (Fork Customizations)

> 本文档记录所有相对于上游 `HKUDS/nanobot` 的本地修改。
> 合并上游更新时，以此文档为准恢复本地功能。
> 最后更新：2026-03-17，基于 merge commit `33efc0f` + fixes

---

## 1. 斜杠命令系统 (`nanobot/agent/loop.py`)

### 命令列表

| 命令 | 功能 | 实现位置 |
|------|------|----------|
| `/new` | 开始新对话，归档当前会话 | `_process_message()` |
| `/clear` | 清除对话历史（不归档） | `_process_message()` |
| `/compact [--switch] [model]` | 压缩对话 / 切换 compact model | `_process_message()` + `_switch_compact_model()` |
| `/skills` | 列出 workspace 可用技能 | `_list_skills()` |
| `/model [number\|name]` | 显示/切换当前模型 | `_format_model_list()` + `_switch_model()` |
| `/mcp` | 列出 MCP 服务器及连接状态 | `_list_mcp_servers()` |
| `/help` | 显示所有命令帮助（含 /stop, /restart） | `_process_message()` |

### 关键实现细节

- `/model` 支持序号、名称、模糊匹配三种方式
- 进度消息 `_bus_progress` 直接 `return` 抑制所有进度到 channel
- `/stop` 和 `/restart` 由上游 `run()` 方法处理（保留上游实现）

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

### 新增方法

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

### 注册位置

- `nanobot/providers/registry.py` — `ProviderSpec(name="claude_oauth", keywords=("claude-oauth",), is_oauth=True, is_direct=True)`
- `nanobot/cli/commands.py` — `_make_provider()` 中 `claude_oauth` 分支，创建 `ClaudeOAuthProvider`

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
memory_window: int = 50       # 上游标记为 deprecated，本地保留为 active 字段
```

### `should_warn_deprecated_memory_window`

上游用于警告 deprecated `memoryWindow`，本地改为永远返回 `False`（因为 `memory_window` 是 active 字段）。

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

上游方案是通过 metadata 标记推送到 bus，channel 端决定是否显示。

---

## 8. 辅助脚本 (本地独有文件)

| 文件 | 说明 |
|------|------|
| `scripts/sync_upstream.sh` | 上游同步脚本 |
| `scripts/validate_sync.py` | 同步后验证脚本 |

---

## 合并策略

合并上游时，对于冲突文件：

1. **loop.py** — 接受上游框架（MemoryConsolidator、_dispatch 等），重新添加斜杠命令和 /model 切换方法
2. **commands.py** — 接受上游重构，补回 `_make_provider` 的 model 参数、claude_oauth 分支、provider_factory 传递、state 恢复
3. **memory.py** — 接受上游的 MemoryConsolidator（token-based）
4. **schema.py** — 接受上游重构，补回 models / compact_model / SubagentConfig / memory_window (active)
5. **registry.py** — 接受上游，补回 claude_oauth ProviderSpec
6. **base.py / litellm_provider.py** — 接受上游（GenerationSettings / chat_with_retry）
7. **memory_tool.py / claude_oauth_*.py** — 本地独有文件，直接保留
8. **tests/test_commands.py** — `_make_provider` mock 需要加 `_model=None` 参数；`_FakeAgentLoop` 需要 `load_persisted_state` 静态方法
9. **tests/test_config_migration.py** — `memoryWindow` 断言需要改为检查值存在（非 deprecated）
