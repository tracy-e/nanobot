"""LiteLLM provider implementation for multi-provider support."""

import asyncio
import os
from typing import Any

import json_repair
import litellm
from litellm import acompletion
from loguru import logger

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from nanobot.providers.registry import find_by_model, find_gateway

# Transient LiteLLM exceptions worth retrying (server-side / network issues).
_RETRYABLE_EXCEPTIONS = (
    litellm.InternalServerError,      # 500 — includes "Server disconnected"
    litellm.ServiceUnavailableError,   # 503
    litellm.APIConnectionError,        # Network-level failures
    litellm.Timeout,                   # Request timeout
)

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0  # seconds, doubles each attempt


# Standard OpenAI chat-completion message keys; extras (e.g. reasoning_content) are stripped for strict providers.
_ALLOWED_MSG_KEYS = frozenset({"role", "content", "tool_calls", "tool_call_id", "name"})


class LiteLLMProvider(LLMProvider):
    """
    LLM provider using LiteLLM for multi-provider support.

    Supports OpenRouter, Anthropic, OpenAI, Gemini, MiniMax, and many other providers through
    a unified interface.  Provider-specific logic is driven by the registry
    (see providers/registry.py) — no if-elif chains needed here.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        default_model: str = "anthropic/claude-opus-4-5",
        extra_headers: dict[str, str] | None = None,
        provider_name: str | None = None,
    ):
        super().__init__(api_key, api_base)
        self.default_model = default_model
        self.extra_headers = extra_headers or {}

        # Detect gateway / local deployment.
        # provider_name (from config key) is the primary signal;
        # api_key / api_base are fallback for auto-detection.
        self._gateway = find_gateway(provider_name, api_key, api_base)

        # Local providers (Ollama, vLLM): bypass system proxy for localhost.
        # macOS system proxy may route 127.0.0.1 through a proxy, causing
        # Python HTTP libraries to fail while curl works fine.
        if self._gateway and self._gateway.is_local and api_base:
            from urllib.parse import urlparse
            host = urlparse(api_base).hostname or ""
            no_proxy = os.environ.get("NO_PROXY", os.environ.get("no_proxy", ""))
            if host not in no_proxy:
                entries = [e for e in no_proxy.split(",") if e] + [host]
                os.environ["NO_PROXY"] = ",".join(entries)

        # Configure environment variables
        if api_key:
            self._setup_env(api_key, api_base, default_model)

        if api_base:
            litellm.api_base = api_base

        # Disable LiteLLM logging noise
        litellm.suppress_debug_info = True
        # Drop unsupported parameters for providers (e.g., gpt-5 rejects some params)
        litellm.drop_params = True

    def _setup_env(self, api_key: str, api_base: str | None, model: str) -> None:
        """Set environment variables based on detected provider."""
        spec = self._gateway or find_by_model(model)
        if not spec:
            return
        if not spec.env_key:
            # OAuth/provider-only specs (for example: openai_codex)
            return

        # Gateway/local overrides existing env; standard provider doesn't
        if self._gateway:
            os.environ[spec.env_key] = api_key
        else:
            os.environ.setdefault(spec.env_key, api_key)

        # Resolve env_extras placeholders:
        #   {api_key}  → user's API key
        #   {api_base} → user's api_base, falling back to spec.default_api_base
        effective_base = api_base or spec.default_api_base
        for env_name, env_val in spec.env_extras:
            resolved = env_val.replace("{api_key}", api_key)
            resolved = resolved.replace("{api_base}", effective_base)
            os.environ.setdefault(env_name, resolved)

    def _resolve_model(self, model: str) -> str:
        """Resolve model name by applying provider/gateway prefixes."""
        if self._gateway:
            # Gateway mode: apply gateway prefix, skip provider-specific prefixes
            prefix = self._gateway.litellm_prefix
            # Strip user-facing prefix (e.g. "vllm/") before adding litellm
            # prefix (e.g. "hosted_vllm/") to avoid double-prefixing.
            user_prefix = f"{self._gateway.name}/"
            if model.startswith(user_prefix):
                model = model[len(user_prefix):]
            elif self._gateway.strip_model_prefix:
                model = model.split("/")[-1]
            if prefix and not model.startswith(f"{prefix}/"):
                model = f"{prefix}/{model}"
            return model

        # Standard mode: auto-prefix for known providers
        spec = find_by_model(model)
        if spec and spec.litellm_prefix:
            model = self._canonicalize_explicit_prefix(model, spec.name, spec.litellm_prefix)
            if not any(model.startswith(s) for s in spec.skip_prefixes):
                model = f"{spec.litellm_prefix}/{model}"

        return model

    @staticmethod
    def _canonicalize_explicit_prefix(model: str, spec_name: str, canonical_prefix: str) -> str:
        """Normalize explicit provider prefixes like `github-copilot/...`."""
        if "/" not in model:
            return model
        prefix, remainder = model.split("/", 1)
        if prefix.lower().replace("-", "_") != spec_name:
            return model
        return f"{canonical_prefix}/{remainder}"

    def _supports_cache_control(self, model: str) -> bool:
        """Return True when the provider supports cache_control on content blocks."""
        if self._gateway is not None:
            return self._gateway.supports_prompt_caching
        spec = find_by_model(model)
        return spec is not None and spec.supports_prompt_caching

    def _apply_cache_control(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
        """Return copies of messages and tools with cache_control injected.

        Cache breakpoints (up to 4 allowed by Anthropic):
        1. System message — stable across turns for high cache hit rate.
        2. Last tool definition — semi-stable, changes only when tools change.
        3. Penultimate message (second-to-last) — caches the conversation
           history so only the latest user turn is uncached.
        """
        new_messages = list(messages)

        # Breakpoint 1: last system message — caches ALL system content
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
        # (the last message is the new user turn which changes every time)
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

    def _apply_model_overrides(self, model: str, kwargs: dict[str, Any]) -> None:
        """Apply model-specific parameter overrides from the registry."""
        model_lower = model.lower()
        spec = find_by_model(model)
        if spec:
            for pattern, overrides in spec.model_overrides:
                if pattern in model_lower:
                    kwargs.update(overrides)
                    return

    @staticmethod
    def _sanitize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Strip non-standard keys and ensure assistant messages have a content key."""
        sanitized = []
        for msg in messages:
            clean = {k: v for k, v in msg.items() if k in _ALLOWED_MSG_KEYS}
            # Strict providers require "content" even when assistant only has tool_calls
            if clean.get("role") == "assistant" and "content" not in clean:
                clean["content"] = None
            sanitized.append(clean)
        return sanitized

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """
        Send a chat completion request via LiteLLM.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            tools: Optional list of tool definitions in OpenAI format.
            model: Model identifier (e.g., 'anthropic/claude-sonnet-4-5').
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.

        Returns:
            LLMResponse with content and/or tool calls.
        """
        original_model = model or self.default_model
        model = self._resolve_model(original_model)

        if self._supports_cache_control(original_model):
            messages, tools = self._apply_cache_control(messages, tools)

        # Clamp max_tokens to at least 1 — negative or zero values cause
        # LiteLLM to reject the request with "max_tokens must be at least 1".
        max_tokens = max(1, max_tokens)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": self._sanitize_messages(self._sanitize_empty_content(messages)),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # Apply model-specific overrides (e.g. kimi-k2.5 temperature)
        self._apply_model_overrides(model, kwargs)

        # Pass api_key directly — more reliable than env vars alone
        if self.api_key:
            kwargs["api_key"] = self.api_key

        # Pass api_base for custom endpoints
        if self.api_base:
            kwargs["api_base"] = self.api_base

        # Pass extra headers (e.g. APP-Code for AiHubMix)
        if self.extra_headers:
            kwargs["extra_headers"] = self.extra_headers

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = await acompletion(**kwargs)
                return self._parse_response(response)
            except _RETRYABLE_EXCEPTIONS as e:
                last_exc = e
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "LLM transient error (attempt {}/{}): {} — retrying in {:.0f}s",
                    attempt + 1, _MAX_RETRIES, e, delay,
                )
                await asyncio.sleep(delay)
            except Exception as e:
                # Non-retryable error — return immediately
                return LLMResponse(
                    content=f"Error calling LLM: {str(e)}",
                    finish_reason="error",
                )

        # All retries exhausted
        return LLMResponse(
            content=f"Error calling LLM (after {_MAX_RETRIES} retries): {last_exc}",
            finish_reason="error",
        )

    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse LiteLLM response into our standard format."""
        choice = response.choices[0]
        message = choice.message

        tool_calls = []
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                # Parse arguments from JSON string if needed
                args = tc.function.arguments
                if isinstance(args, str):
                    args = json_repair.loads(args)

                tool_calls.append(ToolCallRequest(
                    id=tc.id,
                    name=tc.function.name,
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
            # Anthropic cache metrics
            for key in ("cache_creation_input_tokens", "cache_read_input_tokens"):
                val = getattr(u, key, None)
                if val is not None:
                    usage[key] = val

        reasoning_content = getattr(message, "reasoning_content", None) or None
        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
            reasoning_content=reasoning_content,
        )

    def get_default_model(self) -> str:
        """Get the default model."""
        return self.default_model
