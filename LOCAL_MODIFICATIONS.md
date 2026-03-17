# Local Modifications (Fork Customizations)

> 本文档记录所有相对于上游 `HKUDS/nanobot` 的本地修改。
> 合并上游更新时，以此文档为准恢复本地功能。
> 最后更新：2026-03-17，基于 commit `b8205ea`

---

## 1. 斜杠命令系统 (`nanobot/agent/loop.py`)

### 命令列表

| 命令 | 功能 | 实现位置 |
|------|------|----------|
| `/new` | 开始新对话，先归档当前会话到 HISTORY.md | `_process_message()` |
| `/clear` | 清除对话历史（不归档） | `_process_message()` |
| `/compact [--switch] [model]` | 压缩对话 / 切换 compact model | `_process_message()` + `_switch_compact_model()` |
| `/skills` | 列出 workspace 可用技能 | `_list_skills()` |
| `/model [number\|name]` | 显示/切换当前模型 | `_format_model_list()` + `_switch_model()` |
| `/mcp` | 列出 MCP 服务器及连接状态 | `_list_mcp_servers()` |
| `/help` | 显示所有命令帮助 | `_process_message()` |

### 关键实现细节

- `/new` 在清除 session 前先用 lock 保护，同步归档到 memory，失败则不清除
- `/model` 支持序号、名称、模糊匹配三种方式
- 进度消息 `_bus_progress` 直接 `return` 抑制所有进度到 channel

---

## 2. 运行时模型切换 (`nanobot/agent/loop.py`)

### AgentLoop 新增参数

```python
def __init__(self, ...,
    compact_model: str = "",            # compact/consolidation 用的模型
    provider_factory: Callable | None,  # 按模型名动态创建 provider
    available_models: list[str] | None, # config 中配置的可用模型列表
    data_dir: Path | None,              # 状态持久化目录
    temperature: float = 0.1,           # LLM 温度
    max_tokens: int = 4096,             # 最大 token
    memory_window: int = 100,           # 记忆窗口大小（消息数）
)
```

### 新增属性和方法

| 属性/方法 | 用途 |
|-----------|------|
| `self._providers: dict[str, LLMProvider]` | 已创建的 provider 缓存 |
| `self._provider_factory` | 工厂函数，按需创建 provider |
| `self._compact_provider` | compact model 的独立 provider |
| `_provider_key(model)` | 从模型名提取 provider 前缀 |
| `_switch_model(arg)` | 切换主模型，更新 provider/subagent |
| `_switch_compact_model(arg)` | 切换 compact model |
| `_persist_state()` | 将 model/compact_model 写入 `.state.json` |
| `load_persisted_state(data_dir)` | 启动时恢复模型选择 |
| `_format_model_list()` | 格式化模型列表展示 |
| `_format_compact_model_list()` | 格式化 compact model 列表 |

### 状态持久化

- 文件: `<data_dir>/.state.json`
- 内容: `{"model": "...", "compact_model": "..."}`
- 重启 gateway 后自动恢复

---

## 3. Memory Consolidation (`nanobot/agent/loop.py` + `nanobot/agent/memory.py`)

### 本地方案 vs 上游方案

| 特性 | 本地 | 上游 |
|------|------|------|
| 触发方式 | `memory_window` 消息计数 | token 估算 |
| 核心类 | `MemoryStore.consolidate()` | `MemoryConsolidator` |
| compact model | 独立 `_compact_provider` | 用主 provider |
| 锁机制 | `_consolidation_locks` / `_consolidating` set | `_processing_lock` |
| `/new` 归档 | 同步归档，失败不清除 | 后台归档，立即清除 |

### 关键代码

```python
async def _consolidate_memory(self, session, archive_all=False) -> bool:
    compact_provider = self._compact_provider  # 独立于 /model 切换
    compact_model = self.compact_model or self.model
    return await MemoryStore(self.workspace).consolidate(
        session, compact_provider, compact_model,
        archive_all=archive_all, memory_window=self.memory_window,
    )
```

---

## 4. MemorySearchTool (`nanobot/agent/tools/memory_tool.py`)

- 上游不存在此文件
- BM25 搜索工具，可在 memory 目录中搜索相关记忆片段
- 在 `_register_default_tools()` 中注册

---

## 5. CLI 初始化参数 (`nanobot/cli/commands.py`)

### `_make_provider(config, model=None)`

本地版接受可选 `model` 参数，支持为不同模型创建不同 provider。

### `gateway()` 和 `agent()` 命令

传递给 `AgentLoop` 的额外参数：

```python
AgentLoop(
    ...,
    compact_model=config.agents.defaults.compact_model,
    provider_factory=lambda m: _make_provider(config, m),
    available_models=config.agents.defaults.models,
    data_dir=data_dir,
    temperature=config.agents.defaults.temperature,
    max_tokens=config.agents.defaults.max_tokens,
    memory_window=config.agents.defaults.memory_window,
)
```

### `.state.json` 恢复

```python
persisted = AgentLoop.load_persisted_state(data_dir)
if persisted.get("model"):
    model = persisted["model"]
```

---

## 6. Config Schema (`nanobot/config/schema.py`)

### AgentDefaults 新增字段

```python
class AgentDefaults:
    models: list[str] = []        # /model 可切换的模型列表
    compact_model: str = ""       # compact/consolidation 用的模型
    temperature: float = 0.1
    max_tokens: int = 4096
    memory_window: int = 100
```

---

## 7. 进度消息抑制

### 本地方案

`_bus_progress` 直接 `return`，不向任何 channel 发送进度消息。

```python
async def _bus_progress(content: str, *, tool_hint: bool = False) -> None:
    return  # Suppress all progress messages to channels
```

### 上游方案

通过 metadata `_progress: True` 标记推送到 bus，channel 端决定是否显示。

**保留理由**：本地方案更简洁，避免频道中出现冗余的进度消息。

---

## 8. Tool Result 截断限制

```python
_TOOL_RESULT_MAX_CHARS = 500  # 本地: 500，上游: 16_000
```

保留较小值以节省 token 和存储。

---

## 9. Claude OAuth Provider (本地独有文件)

| 文件 | 说明 |
|------|------|
| `nanobot/providers/claude_oauth_provider.py` | 复用 Claude Code OAuth token |
| `nanobot/providers/claude_oauth_auth.py` | OAuth token 发现和刷新 |

---

## 10. 辅助脚本 (本地独有文件)

| 文件 | 说明 |
|------|------|
| `scripts/sync_upstream.sh` | 上游同步脚本 |
| `scripts/validate_sync.py` | 同步后验证脚本 |

---

## 11. 其他本地修改

| 文件 | 修改 | 说明 |
|------|------|------|
| `nanobot/agent/context.py` | subagent context | 注入 AGENTS.md, TOOLS.md, MEMORY.md |
| `nanobot/agent/subagent.py` | 参数传递 | temperature, max_tokens 传递 |
| `nanobot/providers/litellm_provider.py` | 重试 + 代理 | 内置重试、macOS NO_PROXY bypass |
| `nanobot/providers/registry.py` | claude-oauth | 注册 claude-oauth provider |
| `nanobot/cron/service.py` | 热重载 | 支持外部修改 jobs.json 自动重载 |
| `nanobot/heartbeat/service.py` | 过滤 | 过滤 HEARTBEAT_OK 不发通知 |
| `nanobot/channels/discord.py` | bugfix | 修复 media double-send |

---

## 合并策略

合并上游时，对于冲突文件：

1. **loop.py** — 接受上游框架（MemoryConsolidator 等），但重新添加斜杠命令系统和 /model 切换
2. **commands.py** — 接受上游重构，补回 provider_factory / available_models / state 持久化参数
3. **memory.py** — 优先采用上游的 MemoryConsolidator（token-based 更合理），但保留 compact_model 独立 provider
4. **schema.py** — 接受上游重构，补回 models / compact_model 字段
5. **base.py** — 接受上游的 GenerationSettings / chat_with_retry（更好），不需要恢复本地重试逻辑
6. **litellm_provider.py** — 接受上游，重试已移到 base.py。保留 NO_PROXY bypass
7. **memory_tool.py** — 直接保留本地文件（上游不存在）
8. **本地独有文件** — 直接保留
