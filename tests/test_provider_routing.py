"""Tests for provider routing in cli/commands.py."""

from unittest.mock import MagicMock, patch
from pathlib import Path

import pytest

from nanobot.config.schema import Config, ProvidersConfig, ProviderConfig


def _make_config(**provider_overrides) -> Config:
    """Build a Config with specific provider settings."""
    defaults = {
        "anthropic": ProviderConfig(api_key="sk-ant", api_base=None),
        "openai": ProviderConfig(api_key="sk-oai", api_base=None),
        "openrouter": ProviderConfig(api_key="sk-or", api_base="https://or.example.com/v1"),
        "deepseek": ProviderConfig(api_key="sk-ds", api_base=None),
        "groq": ProviderConfig(api_key="sk-groq", api_base=None),
        "zhipu": ProviderConfig(api_key="sk-zhipu", api_base="https://zhipu.example.com"),
        "vllm": ProviderConfig(api_key="sk-vllm", api_base="http://localhost:11434/v1"),
        "gemini": ProviderConfig(api_key="sk-gem", api_base=None),
    }
    defaults.update(provider_overrides)
    config = Config()
    config.providers = ProvidersConfig(**defaults)
    return config


class TestResolveProviderConfig:
    """Test _resolve_provider_config model→provider mapping."""

    def _resolve(self, model: str, config: Config | None = None):
        from nanobot.cli.commands import _resolve_provider_config
        return _resolve_provider_config(model, config or _make_config())

    def test_openrouter_prefix(self):
        key, base = self._resolve("openrouter/anthropic/claude-sonnet-4")
        assert key == "sk-or"
        assert base == "https://or.example.com/v1"

    def test_openrouter_default_base(self):
        config = _make_config(openrouter=ProviderConfig(api_key="sk-or", api_base=None))
        key, base = self._resolve("openrouter/some-model", config)
        assert base == "https://openrouter.ai/api/v1"

    def test_bedrock_prefix(self):
        key, base = self._resolve("bedrock/anthropic.claude-v2")
        assert key is None
        assert base is None

    def test_deepseek(self):
        key, _ = self._resolve("deepseek-chat")
        assert key == "sk-ds"

    def test_anthropic_keyword(self):
        key, _ = self._resolve("anthropic/claude-opus-4-5")
        assert key == "sk-ant"

    def test_claude_keyword(self):
        key, _ = self._resolve("claude-sonnet-4-5")
        assert key == "sk-ant"

    def test_gpt_keyword(self):
        key, _ = self._resolve("gpt-4o")
        assert key == "sk-oai"

    def test_openai_keyword(self):
        key, _ = self._resolve("openai/gpt-4")
        assert key == "sk-oai"

    def test_gemini_keyword(self):
        key, _ = self._resolve("gemini-pro")
        assert key == "sk-gem"

    def test_groq_keyword(self):
        key, _ = self._resolve("groq/llama-3")
        assert key == "sk-groq"

    def test_zhipu_keyword(self):
        key, base = self._resolve("glm-4")
        assert key == "sk-zhipu"
        assert base == "https://zhipu.example.com"

    def test_chatglm_keyword(self):
        key, _ = self._resolve("chatglm-turbo")
        assert key == "sk-zhipu"

    def test_kimi_keyword(self):
        key, base = self._resolve("kimi-k2.5:cloud")
        assert key == "sk-vllm"
        assert base == "http://localhost:11434/v1"

    def test_moonshot_keyword(self):
        key, _ = self._resolve("moonshot-v1")
        assert key == "sk-vllm"

    def test_unknown_model_defaults_to_vllm(self):
        key, base = self._resolve("some-random-model")
        assert key == "sk-vllm"
        assert base == "http://localhost:11434/v1"

    def test_case_insensitive(self):
        key, _ = self._resolve("Gemini-Pro")
        assert key == "sk-gem"


class TestBuildSingleProvider:
    """Test _build_single_provider creates the right provider type."""

    def test_claude_code_prefix(self):
        from nanobot.cli.commands import _build_single_provider
        from nanobot.providers.claude_code_provider import ClaudeCodeProvider

        config = _make_config()
        config.agents.defaults.workspace = "/tmp/test-workspace"
        provider = _build_single_provider("claude-code/opus", config)
        assert isinstance(provider, ClaudeCodeProvider)
        assert provider.model == "opus"

    def test_litellm_provider(self):
        from nanobot.cli.commands import _build_single_provider
        from nanobot.providers.litellm_provider import LiteLLMProvider

        config = _make_config()
        provider = _build_single_provider("deepseek-chat", config)
        assert isinstance(provider, LiteLLMProvider)

    def test_no_api_key_raises(self):
        from nanobot.cli.commands import _build_single_provider

        config = _make_config(vllm=ProviderConfig(api_key="", api_base=None))
        with pytest.raises(ValueError, match="No API key"):
            _build_single_provider("some-unknown-model", config)
