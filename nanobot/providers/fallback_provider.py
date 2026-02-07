"""Fallback chain provider with circuit breaker cooldown."""

import time
from typing import Any

from loguru import logger

from nanobot.providers.base import LLMProvider, LLMResponse

# Default cooldown before retrying a failed provider.
DEFAULT_COOLDOWN_SECONDS = 1800  # 30 minutes


class FallbackProvider(LLMProvider):
    """
    Wraps a list of providers and tries them in priority order.

    Circuit breaker behavior:
    - When a provider fails, it enters cooldown for *cooldown* seconds.
    - During cooldown the provider is skipped (next in chain is tried).
    - After cooldown expires the provider is retried automatically.
    - On success the provider's failure record is cleared immediately.
    - If ALL providers are in cooldown, force-retry from the beginning.
    """

    def __init__(
        self,
        chain: list[tuple[str, LLMProvider]],
        cooldown: int = DEFAULT_COOLDOWN_SECONDS,
    ) -> None:
        """
        Args:
            chain: List of (model_name, provider) tuples in priority order.
                   Must contain at least one entry.
            cooldown: Seconds before retrying a failed provider. Default 30 min.

        Raises:
            ValueError: If *chain* is empty.
        """
        super().__init__(api_key=None, api_base=None)
        if not chain:
            raise ValueError("Fallback chain must have at least one provider")
        self.chain = chain
        self._cooldown = cooldown
        self._failures: dict[str, float] = {}  # name -> timestamp of last failure

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        now = time.time()
        last_error: str | None = None

        # Pass 1: try providers in order, skip those in cooldown
        for name, provider in self.chain:
            fail_time = self._failures.get(name)
            if fail_time and (now - fail_time) < self._cooldown:
                remaining = int(self._cooldown - (now - fail_time))
                logger.debug(f"Provider [{name}] in cooldown ({remaining}s left), skipping")
                continue

            success, response, error = await self._try_provider(
                name, provider, messages, tools, model, max_tokens, temperature,
            )
            if success:
                self._failures.pop(name, None)
                if name != self.chain[0][0]:
                    logger.info(f"Provider [{name}] succeeded (fallback)")
                return response  # type: ignore[return-value]

            self._failures[name] = now
            last_error = error

        # Pass 2: all skipped or failed — force retry from the top
        logger.warning("All providers failed or in cooldown, force-retrying from start")
        for name, provider in self.chain:
            success, response, error = await self._try_provider(
                name, provider, messages, tools, model, max_tokens, temperature,
            )
            if success:
                self._failures.pop(name, None)
                return response  # type: ignore[return-value]
            self._failures[name] = now
            last_error = error

        return LLMResponse(
            content=f"All providers unavailable. Last error: {last_error}",
            tool_calls=[],
        )

    async def _try_provider(
        self,
        name: str,
        provider: LLMProvider,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str | None,
        max_tokens: int,
        temperature: float,
    ) -> tuple[bool, LLMResponse | None, str | None]:
        """
        Try a single provider.

        Returns:
            Tuple of (success, response, error_message).

        Handles both exception-based errors (e.g. ClaudeCodeProvider) and
        response-based errors (LiteLLMProvider ``finish_reason="error"``).
        """
        try:
            response = await provider.chat(messages, tools, model, max_tokens, temperature)

            # LiteLLMProvider returns finish_reason="error" instead of raising
            if response.finish_reason == "error":
                error = response.content or "unknown error"
                logger.warning(f"Provider [{name}] returned error: {error[:200]}")
                return False, None, error

            return True, response, None

        except Exception as e:
            error = str(e)
            logger.warning(f"Provider [{name}] exception: {error[:200]}")
            return False, None, error

    def get_default_model(self) -> str:
        return self.chain[0][0]

    def set_session_key(self, key: str) -> None:
        """Propagate session key to all providers in the chain."""
        for _, provider in self.chain:
            provider.set_session_key(key)
