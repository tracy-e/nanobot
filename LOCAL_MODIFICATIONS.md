# Local Modifications (Fork Customizations)

> 本文档记录所有相对于上游 `HKUDS/nanobot` 的本地修改。
> 合并上游更新时，以此文档为准恢复本地功能。
> 最后更新：2026-05-05，基于 upstream `9d6afd86`

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

本地版是 `nanobot/providers/factory.make_provider(config, model)` 的薄封装，
负责把 ValueError 转成 typer.Exit 并保留可选 `model` 参数（上游 factory 也已扩展支持 model）。

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

启动时从 `.state.json` 恢复 model/compact_model 选择，并把恢复的 model
传给 `build_provider_snapshot(config, model)` 来构建初始 provider snapshot。

### `nanobot/providers/factory.py` 扩展

为支持本地 `/model` 切换，给上游 factory 加了可选 `model` 参数：

- `make_provider(config, model: str | None = None)`
- `provider_signature(config, model: str | None = None)`
- `build_provider_snapshot(config, model: str | None = None)`

未传 model 时行为与上游一致（取 `config.agents.defaults.model`）。

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

## 7. Discord 频道扩展 (`nanobot/channels/discord.py`)

### 动态 slash command 注册

上游用静态 `commands` 元组注册 `/new /stop /restart /status /history`。
本地改为通过 `CommandRouter` + `register_builtin_commands` 动态注册全部命令
（含 `/clear /skills /mcp /compact /model /dream*` 等），便于一处维护。

`/help` 仍单独注册（保留上游的 ephemeral 回复体验），循环内通过
`if cmd_name == "/help": continue` 跳过避免重复。

### `DISCORD_ALLOW_BOTS` 环境变量

上游 #3217 改成只过滤自己（`_bot_user_id`）+ 系统消息（`_is_system_message`），
其它 bot 默认放行。本地保留 fork 自定义的 `DISCORD_ALLOW_BOTS` 三档过滤：

- `none`（默认）— 忽略所有其它 bot 消息
- `mentions` — 只接收 @ 提到自己时的 bot 消息
- `all` — 接收所有 bot 消息

合并顺序：先按上游做自循环 + 系统消息排除，再按本地 env 过滤其它 bot。

---

## 8. 测试调整

- `tests/cli/test_commands.py` —
  - `_make_provider` mock 需要 `_model=None` 参数
  - `build_provider_snapshot` mock 也需要 `_model=None` 参数（因为本地扩展了它的签名）
  - `_FakeAgentLoop` 需要 `load_persisted_state`
- `tests/config/test_config_migration.py` — `memoryWindow` 断言保留 active（非 deprecated）

---

## 合并策略

合并上游时，对于冲突文件：

1. **builtin.py** — 接受上游基础命令（含 `/history`），补回本地 /clear, /compact, /skills, /model, /mcp，更新 /help（注意去重 /status）
2. **loop.py** — 接受上游 runner/hook 架构，补回 init 参数 + helper 方法 + model 切换方法 + `_bus_progress` 抑制 + MemorySearchTool 注册
3. **commands.py** — 接受上游重构，让 `_make_provider` 转发到 `factory.make_provider(config, model)`；保留持久化 model 恢复并把 model 传给 `build_provider_snapshot`
4. **schema.py** — 接受上游重构，补回 models / compact_model / SubagentConfig / memory_window (active)
5. **memory_tool.py** — 本地独有文件，直接保留
6. **discord.py** — 保留本地动态 router 注册逻辑（在循环内 `continue` 跳过 `/help`，再单独保留上游 ephemeral `/help`）；保留 `DISCORD_ALLOW_BOTS` 三档过滤，置于上游 `_bot_user_id` 自循环 + `_is_system_message` 检查之后
7. **tests/cli/test_commands.py** — `_make_provider` mock 加 `_model=None`；`build_provider_snapshot` mock 加 `_model=None`；`_FakeAgentLoop` 加 `load_persisted_state`
8. **tests/config/test_config_migration.py** — memory_window 断言保留 active
