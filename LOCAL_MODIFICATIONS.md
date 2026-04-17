# Local Modifications (Fork Customizations)

> 本文档记录所有相对于上游 `HKUDS/nanobot` 的本地修改。
> 合并上游更新时，以此文档为准恢复本地功能。
> 最后更新：2026-04-17，基于 upstream `ddf2fe4`

---

## 1. 斜杠命令扩展 (`nanobot/command/builtin.py`)

上游内置命令（/new, /stop, /restart, /status, /help）保持不变。

### 本地新增命令

| 命令 | 功能 | 实现 |
|------|------|------|
| `/clear` | 清除对话历史（不归档） | `cmd_clear` |
| `/compact [--switch] [model]` | 压缩对话 / 切换 compact model | `cmd_compact` |
| `/skills` | 列出 workspace 可用技能 | `cmd_skills` → `loop._list_skills()` |
| `/model [number\|name]` | 显示/切换当前模型 | `cmd_model` → `loop._switch_model()` |
| `/mcp` | 列出 MCP 服务器及连接状态 | `cmd_mcp` → `loop._list_mcp_servers()` |

`/help` 已更新包含所有本地命令。

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
| `_switch_model(arg)` | 切换主模型，更新 provider/runner/subagent/consolidator |
| `_switch_compact_model(arg)` | 切换 compact model |
| `_persist_state()` | 将 model/compact_model 写入 `.state.json` |
| `load_persisted_state(data_dir)` | 启动时恢复模型选择 |
| `_format_model_list()` | 格式化模型列表展示 |
| `_format_compact_model_list()` | 格式化 compact model 列表 |
| `_list_skills()` | 列出 workspace skills |
| `_list_mcp_servers()` | 列出 MCP 服务器 |

### 模型切换时更新的组件

- `self.provider` — 主 provider
- `self.runner.provider` — AgentRunner 的 provider
- `self.subagents.provider` / `.model` — 子代理
- `self.memory_consolidator.provider` / `.model` — 记忆合并器

---

## 3. 进度消息默认关闭 (`nanobot/config/schema.py`)

`ChannelsConfig.send_progress` 默认值从上游的 `True` 改为 `False`。
通过官方配置机制控制，可在 config.json 中按需开启。

---

## 4. MemorySearchTool (`nanobot/agent/tools/memory_tool.py`)

- 上游不存在此文件
- BM25 搜索工具，可在 memory 目录中搜索相关记忆片段
- 在 `_register_default_tools()` 中注册

---

## 5. CLI 初始化 (`nanobot/cli/commands.py`)

### `_make_provider(config, model=None)`

本地版接受可选 `model` 参数，支持为不同模型创建不同 provider。

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

## 6. Config Schema 扩展 (不含 Claude OAuth) (`nanobot/config/schema.py`)

### AgentDefaults 新增字段

```python
models: list[str] = []        # /model 可切换的模型列表
compact_model: str = ""       # compact/consolidation 用的模型
memory_window: int = 50       # 上游已移除，本地保留为 active 字段
```

`should_warn_deprecated_memory_window` 永远返回 `False`。

### SubagentConfig（本地新增）

```python
class SubagentConfig(Base):
    exec_timeout: int = 300
    max_iterations: int = 100
    max_duration: int = 600
```

在 `ToolsConfig` 中通过 `subagent` 字段引用。

---

## 7. 测试调整

- `tests/cli/test_commands.py` — `_make_provider` mock 需要加 `_model=None` 参数；`_FakeAgentLoop` 需要 `load_persisted_state`
- `tests/config/test_config_migration.py` — `memoryWindow` 断言保留 active（非 deprecated）

---

## 合并策略

合并上游时，对于冲突文件：

1. **builtin.py** — 接受上游基础命令，补回本地 /clear, /compact, /skills, /model, /mcp，更新 /help
2. **loop.py** — 接受上游 runner/hook 架构，补回 init 参数 + helper 方法 + model 切换方法 + `_bus_progress` 抑制 + MemorySearchTool 注册
3. **commands.py** — 接受上游重构，补回 `_make_provider` 的 model 参数、provider_factory 传递、state 恢复
4. **schema.py** — 接受上游重构，补回 models / compact_model / SubagentConfig / memory_window (active)
5. **memory_tool.py** — 本地独有文件，直接保留
6. **tests/cli/test_commands.py** — `_make_provider` mock 加 `_model=None`；`_FakeAgentLoop` 加 `load_persisted_state`
7. **tests/config/test_config_migration.py** — memory_window 断言保留 active
