# Local Modifications (Fork Customizations)

> 本文档记录所有相对于上游 `HKUDS/nanobot` 的本地修改。
> 合并上游更新时，以此文档为准恢复本地功能。
> 最后更新：2026-03-28，基于 upstream v0.1.4.post6 (`17d21c8`)

---

## 1. 斜杠命令扩展 (`nanobot/command/builtin.py`)

上游内置命令（/new, /stop, /restart, /status, /help）保持不变。

### 本地新增命令

| 命令 | 功能 | 实现 |
|------|------|------|
| `/clear` | 清除对话历史（不归档） | `cmd_clear` |
| `/skills` | 列出 workspace 可用技能 | `cmd_skills` → `loop._list_skills()` |
| `/mcp` | 列出 MCP 服务器及连接状态 | `cmd_mcp` → `loop._list_mcp_servers()` |

`/help` 已更新包含所有本地命令。

---

## 2. Helper 方法 (`nanobot/agent/loop.py`)

| 方法 | 用途 |
|------|------|
| `_list_skills()` | 列出 workspace skills（解析 SKILL.md frontmatter） |
| `_list_mcp_servers()` | 列出 MCP 服务器配置和连接状态 |

---

## 3. 进度消息抑制 (`nanobot/agent/loop.py`)

`_bus_progress` 直接 `return`，不向任何 channel 发送进度消息。

```python
async def _bus_progress(content: str, *, tool_hint: bool = False) -> None:
    return  # Suppress all progress messages to channels
```

---

## 4. MemorySearchTool (`nanobot/agent/tools/memory_tool.py`)

- 上游不存在此文件
- BM25 搜索工具，可在 memory 目录中搜索相关记忆片段
- 在 `_register_default_tools()` 中注册

---

## 5. Config Schema 扩展 (`nanobot/config/schema.py`)

### AgentDefaults 新增字段

```python
memory_window: int = 50  # 上游已移除，本地保留为 active 字段
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

## 合并策略

合并上游时，对于冲突文件：

1. **builtin.py** — 接受上游基础命令，补回本地 /clear, /skills, /mcp，更新 /help
2. **loop.py** — 接受上游 runner/hook 架构，补回 `_list_skills`、`_list_mcp_servers`、`_bus_progress` 抑制、MemorySearchTool 注册
3. **schema.py** — 接受上游重构，补回 memory_window (active) + SubagentConfig
4. **memory_tool.py** — 本地独有文件，直接保留
