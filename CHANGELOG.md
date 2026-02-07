# Changelog

## 2026-02-07: Claude Code Provider Integration

Replace LiteLLM-based LLM agent with Claude Code CLI for the primary reasoning engine.

### Architecture Change

```
Before: Feishu → nanobot → LiteLLM(kimi) → nanobot tools → Feishu
After:  Feishu → nanobot → Claude Code CLI (full delegation) → Feishu
```

### New Files

- `nanobot/providers/claude_code_provider.py` — LLMProvider implementation that delegates to `claude -p` CLI
- `nanobot/providers/fallback_provider.py` — Fallback chain provider with circuit breaker cooldown

### New Test Files

- `tests/test_claude_code_provider.py` — 27 tests covering init, model resolution, message extraction, output parsing, sessions, chat (mocked subprocess), context building
- `tests/test_fallback_provider.py` — 12 tests covering routing, circuit breaker cooldown, force-retry, session propagation
- `tests/test_provider_routing.py` — 21 tests covering `_resolve_provider_config` keyword/prefix matching, `_build_single_provider` type dispatch

### Modified Files

- `nanobot/providers/base.py` — Added `set_session_key()` as optional method on `LLMProvider` base class (no-op default)
- `nanobot/agent/loop.py` — Calls `provider.set_session_key()` directly (no more `hasattr` duck typing)
- `nanobot/cli/commands.py` — Extracted `_build_single_provider()`, `_resolve_provider_config()`, `_build_provider_chain()` for provider routing; added type hints and keyword-based provider mapping with zhipu/kimi/moonshot support
- `nanobot/config/schema.py` — Added `fallback_models: list[str]` to `AgentDefaults` with description
- `nanobot/providers/__init__.py` — Exported `ClaudeCodeProvider` and `FallbackProvider`

### Features

#### Claude Code Provider
- Calls `claude -p <msg> --output-format json` with `--dangerously-skip-permissions`
- Multi-turn conversation via `--resume <session_id>`
- Session mapping: nanobot session key → claude session_id, persisted to `~/.nanobot/claude_sessions.json` (configurable path)
- Nanobot context (USER.md, AGENTS.md, SOUL.md, Memory, Skills) passed via `--append-system-prompt`
- Model mapping: opus → claude-opus-4-6, sonnet → claude-sonnet-4-5-20250929, haiku → claude-haiku-4-5-20251001 (overridable)
- Subprocess timeout protection via `asyncio.wait_for` (default 300s, configurable)
- Structured `ClaudeCodeError` exception for CLI failures (timeout, not found, non-zero exit)

#### Memory Context
- Memory re-read on every `chat()` call (not just at init), so the LLM always sees the latest state
- Memory management instructions included in `--append-system-prompt` to guide the LLM on when/where to write memory files
- Daily notes (`memory/YYYY-MM-DD.md`) and long-term memory (`memory/MEMORY.md`) follow nanobot's default behavior — written by the LLM via tools at its discretion

#### Fallback Chain with Circuit Breaker
- Config: `"fallbackModels": ["kimi-k2.5:cloud", "openrouter/anthropic/claude-sonnet-4"]`
- Providers tried in priority order; failed ones enter 30-min cooldown
- After cooldown, primary provider is retried automatically
- `_resolve_provider_config()` maps model names to correct API key/base from config
- Handles both exception-based errors (ClaudeCodeProvider) and response-based errors (LiteLLMProvider `finish_reason="error"`)

### Config Changes

```json
{
  "agents": {
    "defaults": {
      "model": "claude-code/opus",
      "fallbackModels": ["kimi-k2.5:cloud", "openrouter/anthropic/claude-sonnet-4"]
    }
  }
}
```
