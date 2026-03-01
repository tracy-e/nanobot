"""Claude OAuth provider — uses LiteLLM with OAuth token from macOS Keychain."""

import asyncio
import json
import time
from typing import Any

import litellm
from litellm import acompletion

# ---------------------------------------------------------------------------
# Patch LiteLLM for Claude Code OAuth compatibility:
#
# 1. validate_environment patch: lets callers remove headers by setting them
#    to None in extra_headers (e.g. x-api-key=None to drop the API-key header).
#
# 2. Register custom beta headers: LiteLLM filters anthropic-beta values
#    through a whitelist (anthropic_beta_headers_config.json). Our Claude Code
#    headers (claude-code-*, oauth-*) would be silently dropped without this.
# ---------------------------------------------------------------------------
from litellm.llms.anthropic.chat.transformation import (
    AnthropicConfig as _AntConfig,
)
from loguru import logger

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from nanobot.providers.claude_oauth_auth import (
    get_claude_oauth_credentials,
    get_claude_oauth_token,
    trigger_claude_token_refresh,
)

_orig_validate_env = _AntConfig.validate_environment


def _patched_validate_env(self, headers, *args, **kwargs):
    caller_overrides = dict(headers) if headers else {}
    result = _orig_validate_env(self, headers, *args, **kwargs)
    # Re-apply caller headers so they take priority over Anthropic defaults
    result.update(caller_overrides)
    # Remove headers explicitly set to None (delete unwanted defaults)
    for key in [k for k, v in result.items() if v is None]:
        del result[key]
    return result


_AntConfig.validate_environment = _patched_validate_env

# Register Claude Code beta headers in LiteLLM's whitelist so they survive
# the update_headers_with_filtered_beta() filter pass.
try:
    from litellm.anthropic_beta_headers_manager import _load_beta_headers_config

    _beta_cfg = _load_beta_headers_config()
    _anthropic_map = _beta_cfg.get("anthropic", {})
    for _h in (
        "claude-code-20250219",
        "oauth-2025-04-20",
    ):
        _anthropic_map.setdefault(_h, _h)
except Exception:
    pass


class ClaudeOAuthProvider(LLMProvider):
    """LLM provider that uses Claude Code's OAuth token via LiteLLM.

    Reads the OAuth token from macOS Keychain (stored by Claude Code),
    injects the required system prefix and auth headers, and delegates
    the actual API call to LiteLLM.
    """

    REQUIRED_SYSTEM_PREFIX = (
        "You are Claude Code, Anthropic's official CLI for Claude."
    )
    BETA_HEADER = (
        "claude-code-20250219,oauth-2025-04-20,interleaved-thinking-2025-05-14"
    )

    # Refresh 5 minutes before expiry to avoid mid-request failures.
    _REFRESH_MARGIN_MS = 5 * 60 * 1000

    # Tool names that the OAuth API rejects (lowercase names matching Claude
    # Code's internal tools). Map them to CamelCase alternatives.
    _TOOL_NAME_MAP: dict[str, str] = {
        "read_file": "ReadFile",
    }
    _TOOL_NAME_RMAP: dict[str, str] = {v: k for k, v in _TOOL_NAME_MAP.items()}

    def __init__(self, model: str = "claude-sonnet-4-6"):
        super().__init__()
        self.model = model
        self._token: str | None = None
        self._expires_at: int | None = None
        litellm.suppress_debug_info = True
        litellm.drop_params = True

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        token = self._get_token()

        # Strip provider prefix (e.g. "claude-oauth/claude-sonnet-..." → "claude-sonnet-...")
        resolved_model = model or self.model
        if "/" in resolved_model:
            resolved_model = resolved_model.split("/", 1)[1]

        # Ensure the Claude Code system prefix is present
        messages = self._ensure_system_prefix(messages)

        kwargs: dict[str, Any] = {
            "model": f"anthropic/{resolved_model}",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            # Explicitly set api_base to prevent litellm from using a stale
            # global litellm.api_base set by another provider (e.g. Ollama).
            "api_base": "https://api.anthropic.com/v1/messages",
            # Anthropic OAuth requires Authorization: Bearer, not x-api-key.
            # LiteLLM natively detects OAuth tokens and adds required headers
            # (anthropic-dangerous-direct-browser-access, oauth beta).
            # We only need to remove x-api-key (litellm sets it to the OAuth
            # token, but Anthropic expects OAuth via Authorization header only).
            "api_key": "oauth-via-header",
            "extra_headers": {
                "authorization": f"Bearer {token}",
                "anthropic-beta": self.BETA_HEADER,
                "x-api-key": None,
            },
        }

        if tools:
            kwargs["tools"] = self._remap_tools(tools)
            kwargs["tool_choice"] = "auto"

        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
            kwargs["drop_params"] = True

        # Apply prompt caching — Anthropic supports cache_control on
        # system message, tool definitions, and conversation turns.
        messages, tools_out = self._apply_cache_control(
            kwargs["messages"], kwargs.get("tools")
        )
        kwargs["messages"] = messages
        if tools_out is not None:
            kwargs["tools"] = tools_out

        # Transient errors worth retrying (server-side / network issues).
        _retryable = (
            litellm.InternalServerError,
            litellm.ServiceUnavailableError,
            litellm.APIConnectionError,
            litellm.Timeout,
        )
        max_retries = 3

        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                response = await acompletion(**kwargs)
                return self._parse_response(response)
            except litellm.AuthenticationError:
                # 401 → trigger Claude Code to refresh token, then retry once
                logger.info("OAuth token rejected (401), triggering refresh…")
                self._token = None
                self._expires_at = None
                trigger_claude_token_refresh()
                token = self._get_token()
                kwargs["extra_headers"]["authorization"] = f"Bearer {token}"
                response = await acompletion(**kwargs)
                return self._parse_response(response)
            except _retryable as e:
                last_exc = e
                delay = 1.0 * (2 ** attempt)
                logger.warning(
                    "OAuth LLM transient error (attempt {}/{}): {} — retrying in {:.0f}s",
                    attempt + 1, max_retries, e, delay,
                )
                await asyncio.sleep(delay)

        # All retries exhausted
        return LLMResponse(
            content=f"Error calling LLM (after {max_retries} retries): {last_exc}",
            finish_reason="error",
        )

    def get_default_model(self) -> str:
        return f"claude-oauth/{self.model}"

    # ------------------------------------------------------------------
    # Prompt caching
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_cache_control(
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
        """Inject cache_control breakpoints for Anthropic prompt caching.

        Breakpoints:
        1. System message — stable across turns.
        2. Last tool definition — semi-stable.
        3. Penultimate message — caches conversation history.
        """
        new_messages = list(messages)

        # Breakpoint 1: last system message — caches ALL system content
        # (Claude OAuth prepends a short prefix system message; we want
        # the breakpoint on the LAST system msg so the entire system
        # prefix including the large nanobot prompt is cached.)
        last_sys_idx = None
        for i, msg in enumerate(new_messages):
            if msg.get("role") == "system":
                last_sys_idx = i
        if last_sys_idx is not None:
            msg = new_messages[last_sys_idx]
            content = msg["content"]
            if isinstance(content, str):
                new_content = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
            else:
                new_content = list(content)
                new_content[-1] = {**new_content[-1], "cache_control": {"type": "ephemeral"}}
            new_messages[last_sys_idx] = {**msg, "content": new_content}

        # Breakpoint 3: penultimate message — caches conversation history
        if len(new_messages) >= 3:
            idx = len(new_messages) - 2
            penultimate = new_messages[idx]
            content = penultimate.get("content")
            if isinstance(content, str) and content:
                new_content = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
                new_messages[idx] = {**penultimate, "content": new_content}

        # Breakpoint 2: last tool definition
        new_tools = tools
        if tools:
            new_tools = list(tools)
            new_tools[-1] = {**new_tools[-1], "cache_control": {"type": "ephemeral"}}

        return new_messages, new_tools

    # ------------------------------------------------------------------
    # Token management
    # ------------------------------------------------------------------

    def _get_token(self) -> str:
        now_ms = int(time.time() * 1000)

        # Check if cached token is still valid (with margin)
        if self._token and self._expires_at:
            if now_ms < self._expires_at - self._REFRESH_MARGIN_MS:
                return self._token

        # Read full credentials from Keychain
        creds = get_claude_oauth_credentials()
        if creds:
            expires_at = creds.get("expiresAt")
            needs_refresh = (
                expires_at is not None
                and now_ms >= expires_at - self._REFRESH_MARGIN_MS
            )

            if needs_refresh:
                # Trigger Claude Code to refresh, then re-read
                logger.info("Token expiring soon, triggering Claude Code refresh…")
                if trigger_claude_token_refresh():
                    creds = get_claude_oauth_credentials()
                    if creds and creds.get("accessToken"):
                        self._token = creds["accessToken"]
                        self._expires_at = creds.get("expiresAt")
                        return self._token

            # Token not expired (or refresh didn't help, use what we have)
            if creds.get("accessToken"):
                self._token = creds["accessToken"]
                self._expires_at = expires_at
                return self._token

        # Fallback: original simple token read
        self._token = get_claude_oauth_token()
        if not self._token:
            raise RuntimeError(
                "No Claude OAuth token found. "
                "Make sure Claude Code is installed and you have logged in."
            )
        return self._token

    # ------------------------------------------------------------------
    # Message preparation
    # ------------------------------------------------------------------

    def _ensure_system_prefix(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Ensure the Claude Code system prefix is present in messages."""
        # Check if any system message already starts with the prefix
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if content.startswith(self.REQUIRED_SYSTEM_PREFIX):
                    return messages

        # Prepend a system message with the required prefix
        prefix_msg = {"role": "system", "content": self.REQUIRED_SYSTEM_PREFIX}
        return [prefix_msg] + list(messages)

    # ------------------------------------------------------------------
    # Tool name mapping
    # ------------------------------------------------------------------

    def _remap_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Rename tools that clash with Claude Code's internal tool names."""
        remapped = []
        for tool in tools:
            fn = tool.get("function", {})
            name = fn.get("name", "")
            mapped = self._TOOL_NAME_MAP.get(name)
            if mapped:
                tool = {**tool, "function": {**fn, "name": mapped}}
            remapped.append(tool)
        return remapped

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse LiteLLM response into LLMResponse."""
        choice = response.choices[0]
        message = choice.message

        tool_calls = []
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}
                tool_calls.append(ToolCallRequest(
                    id=tc.id,
                    name=self._TOOL_NAME_RMAP.get(tc.function.name, tc.function.name),
                    arguments=args,
                ))

        usage = {}
        if hasattr(response, "usage") and response.usage:
            u = response.usage
            usage = {
                "prompt_tokens": u.prompt_tokens,
                "completion_tokens": u.completion_tokens,
                "total_tokens": u.total_tokens,
            }
            # Anthropic cache metrics (returned when cache_control is used)
            for key in ("cache_creation_input_tokens", "cache_read_input_tokens"):
                val = getattr(u, key, None)
                if val is not None:
                    usage[key] = val

        reasoning_content = getattr(message, "reasoning_content", None)

        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
            reasoning_content=reasoning_content,
        )
