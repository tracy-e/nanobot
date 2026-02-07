"""Tests for FallbackProvider."""

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.providers.base import LLMProvider, LLMResponse
from nanobot.providers.fallback_provider import FallbackProvider, DEFAULT_COOLDOWN_SECONDS


class DummyProvider(LLMProvider):
    """Test provider that returns a canned response or raises."""

    def __init__(self, name: str, response: LLMResponse | None = None, error: Exception | None = None):
        super().__init__()
        self._name = name
        self._response = response
        self._error = error
        self._call_count = 0
        self._last_session_key: str | None = None

    async def chat(self, messages, tools=None, model=None, max_tokens=4096, temperature=0.7) -> LLMResponse:
        self._call_count += 1
        if self._error:
            raise self._error
        return self._response  # type: ignore[return-value]

    def get_default_model(self) -> str:
        return self._name

    def set_session_key(self, key: str) -> None:
        self._last_session_key = key


def _ok(text: str = "ok") -> LLMResponse:
    return LLMResponse(content=text, tool_calls=[])


def _error_response(text: str = "bad") -> LLMResponse:
    return LLMResponse(content=text, tool_calls=[], finish_reason="error")


# ------------------------------------------------------------------
# Constructor
# ------------------------------------------------------------------


class TestInit:
    def test_empty_chain_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            FallbackProvider(chain=[])

    def test_default_cooldown(self):
        p = DummyProvider("a", response=_ok())
        fp = FallbackProvider(chain=[("a", p)])
        assert fp._cooldown == DEFAULT_COOLDOWN_SECONDS

    def test_custom_cooldown(self):
        p = DummyProvider("a", response=_ok())
        fp = FallbackProvider(chain=[("a", p)], cooldown=60)
        assert fp._cooldown == 60

    def test_get_default_model(self):
        p = DummyProvider("primary", response=_ok())
        fp = FallbackProvider(chain=[("primary", p)])
        assert fp.get_default_model() == "primary"


# ------------------------------------------------------------------
# Normal routing
# ------------------------------------------------------------------


class TestNormalRouting:
    @pytest.mark.asyncio
    async def test_primary_succeeds(self):
        p1 = DummyProvider("a", response=_ok("from a"))
        p2 = DummyProvider("b", response=_ok("from b"))
        fp = FallbackProvider(chain=[("a", p1), ("b", p2)])

        resp = await fp.chat(messages=[])
        assert resp.content == "from a"
        assert p1._call_count == 1
        assert p2._call_count == 0

    @pytest.mark.asyncio
    async def test_fallback_on_exception(self):
        p1 = DummyProvider("a", error=RuntimeError("boom"))
        p2 = DummyProvider("b", response=_ok("from b"))
        fp = FallbackProvider(chain=[("a", p1), ("b", p2)])

        resp = await fp.chat(messages=[])
        assert resp.content == "from b"
        assert p1._call_count == 1
        assert p2._call_count == 1

    @pytest.mark.asyncio
    async def test_fallback_on_error_response(self):
        p1 = DummyProvider("a", response=_error_response("a failed"))
        p2 = DummyProvider("b", response=_ok("from b"))
        fp = FallbackProvider(chain=[("a", p1), ("b", p2)])

        resp = await fp.chat(messages=[])
        assert resp.content == "from b"

    @pytest.mark.asyncio
    async def test_all_fail_returns_error(self):
        p1 = DummyProvider("a", error=RuntimeError("a boom"))
        p2 = DummyProvider("b", error=RuntimeError("b boom"))
        fp = FallbackProvider(chain=[("a", p1), ("b", p2)])

        resp = await fp.chat(messages=[])
        assert "All providers unavailable" in resp.content
        assert "b boom" in resp.content


# ------------------------------------------------------------------
# Circuit breaker / cooldown
# ------------------------------------------------------------------


class TestCircuitBreaker:
    @pytest.mark.asyncio
    async def test_failed_provider_in_cooldown(self):
        p1 = DummyProvider("a", error=RuntimeError("boom"))
        p2 = DummyProvider("b", response=_ok("from b"))
        fp = FallbackProvider(chain=[("a", p1), ("b", p2)], cooldown=3600)

        # First call: p1 fails, p2 succeeds
        await fp.chat(messages=[])
        assert p1._call_count == 1
        assert p2._call_count == 1
        assert "a" in fp._failures

        # Second call: p1 should be skipped (in cooldown)
        await fp.chat(messages=[])
        assert p1._call_count == 1  # NOT retried
        assert p2._call_count == 2

    @pytest.mark.asyncio
    async def test_success_clears_failure(self):
        call_count = 0

        class FlakeyProvider(LLMProvider):
            async def chat(self, messages, tools=None, model=None, max_tokens=4096, temperature=0.7):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise RuntimeError("first fail")
                return _ok("recovered")

            def get_default_model(self):
                return "flakey"

        fp = FallbackProvider(chain=[("flakey", FlakeyProvider())], cooldown=0)

        # First call fails, force-retried in pass 2, succeeds
        resp = await fp.chat(messages=[])
        assert resp.content == "recovered"
        assert "flakey" not in fp._failures  # cleared on success

    @pytest.mark.asyncio
    async def test_force_retry_when_all_in_cooldown(self):
        p1 = DummyProvider("a", error=RuntimeError("a fail"))
        p2 = DummyProvider("b", error=RuntimeError("b fail"))
        fp = FallbackProvider(chain=[("a", p1), ("b", p2)], cooldown=3600)

        # First call: both fail
        await fp.chat(messages=[])
        assert p1._call_count == 2  # pass 1 + pass 2
        assert p2._call_count == 2

        # Now make p1 succeed
        p1._error = None
        p1._response = _ok("a recovered")

        # Second call: both in cooldown → force retry → p1 succeeds
        resp = await fp.chat(messages=[])
        assert resp.content == "a recovered"
        assert p1._call_count == 3


# ------------------------------------------------------------------
# Session propagation
# ------------------------------------------------------------------


class TestSessionPropagation:
    def test_set_session_key_propagates(self):
        p1 = DummyProvider("a", response=_ok())
        p2 = DummyProvider("b", response=_ok())
        fp = FallbackProvider(chain=[("a", p1), ("b", p2)])

        fp.set_session_key("feishu:123")
        assert p1._last_session_key == "feishu:123"
        assert p2._last_session_key == "feishu:123"
