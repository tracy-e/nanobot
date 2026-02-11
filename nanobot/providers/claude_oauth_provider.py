"""Claude OAuth provider — uses LiteLLM with OAuth token from macOS Keychain."""

import json
import time
from typing import Any

import litellm
from litellm import acompletion
from loguru import logger

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from nanobot.providers.claude_oauth_auth import (
    get_claude_oauth_credentials,
    get_claude_oauth_token,
    trigger_claude_token_refresh,
)


# ---------------------------------------------------------------------------
# Patch LiteLLM: Anthropic OAuth requires "Authorization: Bearer" but NOT
# "x-api-key". LiteLLM always adds x-api-key which Anthropic rejects for
# OAuth tokens. This patch lets callers remove headers by setting them to
# None in extra_headers.
# ---------------------------------------------------------------------------
from litellm.llms.anthropic.chat.transformation import (
    AnthropicConfig as _AntConfig,
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

    def __init__(self, model: str = "claude-sonnet-4-5-20250929"):
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
            # Anthropic OAuth requires Authorization: Bearer, not x-api-key.
            # LiteLLM detects OAuth tokens via lowercase "authorization" header
            # and auto-adds unwanted headers. We override/remove them:
            # - x-api-key=None: removed (OAuth doesn't use x-api-key)
            # - anthropic-dangerous-direct-browser-access=None: removed
            #   (browser flag triggers Claude Code restriction)
            # - anthropic-beta: our Claude Code beta flags
            "api_key": "oauth-via-header",
            "extra_headers": {
                "authorization": f"Bearer {token}",
                "anthropic-beta": self.BETA_HEADER,
                "x-api-key": None,
                "anthropic-dangerous-direct-browser-access": None,
            },
        }

        if tools:
            kwargs["tools"] = self._remap_tools(tools)
            kwargs["tool_choice"] = "auto"

        try:
            response = await acompletion(**kwargs)
            return self._parse_response(response)
        except litellm.AuthenticationError:
            # 401 → trigger Claude Code to refresh token, then retry
            logger.info("OAuth token rejected (401), triggering refresh…")
            self._token = None
            self._expires_at = None
            trigger_claude_token_refresh()
            token = self._get_token()
            kwargs["extra_headers"]["authorization"] = f"Bearer {token}"
            response = await acompletion(**kwargs)
            return self._parse_response(response)

    def get_default_model(self) -> str:
        return f"claude-oauth/{self.model}"

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
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        reasoning_content = getattr(message, "reasoning_content", None)

        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
            reasoning_content=reasoning_content,
        )
