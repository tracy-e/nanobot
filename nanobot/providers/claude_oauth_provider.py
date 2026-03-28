"""Claude OAuth provider — reuses AnthropicProvider with OAuth token auth.

Reads the OAuth token from macOS Keychain (stored by Claude Code CLI),
injects the required auth headers, and delegates all message handling
to the parent AnthropicProvider.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from collections.abc import Awaitable, Callable
from loguru import logger

from nanobot.providers.anthropic_provider import AnthropicProvider
from nanobot.providers.base import LLMResponse


class ClaudeOAuthProvider(AnthropicProvider):
    """LLM provider using Claude Code's OAuth token via native Anthropic SDK."""

    REQUIRED_SYSTEM_PREFIX = (
        "You are Claude Code, Anthropic's official CLI for Claude."
    )
    BETA_HEADER = (
        "claude-code-20250219,oauth-2025-04-20,interleaved-thinking-2025-05-14"
    )
    _REFRESH_MARGIN_MS = 5 * 60 * 1000  # refresh 5 min before expiry

    def __init__(
        self,
        default_model: str = "claude-sonnet-4-6",
        api_base: str | None = None,
    ):
        self._token: str | None = None
        self._expires_at: int | None = None

        # Load initial token
        self._load_token()

        # Init parent with a dummy api_key to satisfy AnthropicProvider.__init__
        super().__init__(
            api_key="unused",
            api_base=api_base,
            default_model=default_model,
        )

        # Replace client: use auth_token (Bearer) instead of api_key (x-api-key).
        # The Anthropic SDK natively supports auth_token for OAuth Bearer auth.
        from anthropic import AsyncAnthropic

        client_kw: dict[str, Any] = {"auth_token": self._token or "pending"}
        if api_base:
            client_kw["base_url"] = api_base
        self._client = AsyncAnthropic(**client_kw)

    # ------------------------------------------------------------------
    # Token management
    # ------------------------------------------------------------------

    def _load_token(self) -> None:
        """Read token and expiry from Keychain."""
        from nanobot.providers.claude_oauth_auth import get_claude_oauth_credentials

        creds = get_claude_oauth_credentials()
        if creds:
            self._token = creds.get("accessToken")
            self._expires_at = creds.get("expiresAt")
        else:
            logger.warning("No Claude OAuth credentials found")

    def _refresh_token_if_needed(self) -> None:
        """Refresh token if it's about to expire."""
        if self._expires_at is None or self._token is None:
            self._load_token()
            if self._token and hasattr(self, "_client"):
                self._client.auth_token = self._token
            return

        now_ms = int(time.time() * 1000)
        if now_ms >= self._expires_at - self._REFRESH_MARGIN_MS:
            logger.info("Claude OAuth token expiring soon, refreshing...")
            self._force_refresh_token()

    def _force_refresh_token(self) -> None:
        """Force a token refresh via Claude CLI, then reload."""
        from nanobot.providers.claude_oauth_auth import trigger_claude_token_refresh

        trigger_claude_token_refresh()
        self._load_token()
        # Update the client's auth_token so subsequent requests use the new token
        if self._token:
            self._client.auth_token = self._token

    # ------------------------------------------------------------------
    # Message preprocessing (OAuth-specific)
    # ------------------------------------------------------------------

    @staticmethod
    def _fix_trailing_assistant(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert a trailing assistant message to user role.

        The OAuth endpoint rejects assistant-prefill (conversation ending
        with an assistant message).  Re-label the last message so the
        API accepts it.
        """
        if not messages:
            return messages
        last = messages[-1]
        if last.get("role") == "assistant":
            messages = list(messages)
            messages[-1] = {**last, "role": "user"}
        return messages

    def _ensure_system_prefix(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Ensure the Claude Code system prefix is present."""
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, str) and content.startswith(self.REQUIRED_SYSTEM_PREFIX):
                    return messages
        return [{"role": "system", "content": self.REQUIRED_SYSTEM_PREFIX}] + list(messages)

    # ------------------------------------------------------------------
    # Override _build_kwargs to inject OAuth auth + preprocessing
    # ------------------------------------------------------------------

    def _build_kwargs(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str | None,
        max_tokens: int,
        temperature: float,
        reasoning_effort: str | None,
        tool_choice: str | dict[str, Any] | None,
        supports_caching: bool = True,
    ) -> dict[str, Any]:
        # Strip provider prefix (e.g. "claude-oauth/claude-sonnet-4-6" → "claude-sonnet-4-6")
        if model and "/" in model:
            model = model.split("/", 1)[1]

        # OAuth-specific message preprocessing
        messages = self._ensure_system_prefix(messages)
        messages = self._fix_trailing_assistant(messages)

        # Build kwargs via parent (handles Anthropic message conversion, caching, etc.)
        kwargs = super()._build_kwargs(
            messages, tools, model, max_tokens, temperature,
            reasoning_effort, tool_choice, supports_caching,
        )

        # Refresh token if needed (updates self._client.auth_token)
        self._refresh_token_if_needed()
        kwargs["extra_headers"] = {"anthropic-beta": self.BETA_HEADER}

        return kwargs

    # ------------------------------------------------------------------
    # Override chat/chat_stream to handle 401 → refresh → retry
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        try:
            return await super().chat(
                messages, tools, model, max_tokens, temperature,
                reasoning_effort, tool_choice,
            )
        except Exception as e:
            if self._is_auth_error(e):
                logger.warning("OAuth 401, refreshing token and retrying...")
                self._force_refresh_token()
                return await super().chat(
                    messages, tools, model, max_tokens, temperature,
                    reasoning_effort, tool_choice,
                )
            raise

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        on_content_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        try:
            return await super().chat_stream(
                messages, tools, model, max_tokens, temperature,
                reasoning_effort, tool_choice, on_content_delta,
            )
        except Exception as e:
            if self._is_auth_error(e):
                logger.warning("OAuth 401, refreshing token and retrying...")
                self._force_refresh_token()
                return await super().chat_stream(
                    messages, tools, model, max_tokens, temperature,
                    reasoning_effort, tool_choice, on_content_delta,
                )
            raise

    @staticmethod
    def _is_auth_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return "401" in msg or "unauthorized" in msg or "authentication" in msg
