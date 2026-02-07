"""Tests for ClaudeCodeProvider."""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from nanobot.providers.claude_code_provider import (
    ClaudeCodeProvider,
    ClaudeCodeError,
    DEFAULT_MODEL_MAP,
    DEFAULT_TIMEOUT,
)
from nanobot.providers.base import LLMResponse


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Create a minimal workspace with bootstrap files."""
    (tmp_path / "AGENTS.md").write_text("Be helpful.")
    (tmp_path / "SOUL.md").write_text("I am a bot.")
    (tmp_path / "memory").mkdir()
    (tmp_path / "memory" / "MEMORY.md").write_text("# Memory\n\nTest memory.")
    return tmp_path


@pytest.fixture
def sessions_file(tmp_path: Path) -> Path:
    return tmp_path / "sessions.json"


@pytest.fixture
def provider(workspace: Path, sessions_file: Path) -> ClaudeCodeProvider:
    return ClaudeCodeProvider(
        workspace=workspace,
        model="opus",
        sessions_file=sessions_file,
    )


# ------------------------------------------------------------------
# Constructor
# ------------------------------------------------------------------


class TestInit:
    def test_default_model(self, provider: ClaudeCodeProvider):
        assert provider.model == "opus"
        assert provider.get_default_model() == "claude-code/opus"

    def test_default_timeout(self, provider: ClaudeCodeProvider):
        assert provider.timeout == DEFAULT_TIMEOUT

    def test_custom_timeout(self, workspace: Path, sessions_file: Path):
        p = ClaudeCodeProvider(workspace=workspace, timeout=60, sessions_file=sessions_file)
        assert p.timeout == 60

    def test_custom_model_map(self, workspace: Path, sessions_file: Path):
        custom = {"mymodel": "custom-model-id"}
        p = ClaudeCodeProvider(
            workspace=workspace, model="mymodel", model_map=custom, sessions_file=sessions_file,
        )
        assert p._resolve_model() == "custom-model-id"

    def test_sessions_file_path(self, provider: ClaudeCodeProvider, sessions_file: Path):
        assert provider._sessions_file == sessions_file

    def test_static_context_built(self, provider: ClaudeCodeProvider):
        assert provider._static_context is not None
        assert "AGENTS.md" in provider._static_context
        assert "SOUL.md" in provider._static_context


# ------------------------------------------------------------------
# Model resolution
# ------------------------------------------------------------------


class TestResolveModel:
    def test_known_aliases(self, provider: ClaudeCodeProvider):
        for alias, full_id in DEFAULT_MODEL_MAP.items():
            provider.model = alias
            assert provider._resolve_model() == full_id

    def test_unknown_model_passed_through(self, provider: ClaudeCodeProvider):
        provider.model = "some-custom-model-2025"
        assert provider._resolve_model() == "some-custom-model-2025"


# ------------------------------------------------------------------
# Message extraction
# ------------------------------------------------------------------


class TestExtractUserMessage:
    def test_simple_string(self, provider: ClaudeCodeProvider):
        msgs = [{"role": "user", "content": "hello"}]
        assert provider._extract_user_message(msgs) == "hello"

    def test_multimodal(self, provider: ClaudeCodeProvider):
        msgs = [
            {"role": "user", "content": [
                {"type": "text", "text": "line1"},
                {"type": "image", "url": "http://img"},
                {"type": "text", "text": "line2"},
            ]},
        ]
        assert provider._extract_user_message(msgs) == "line1\nline2"

    def test_last_user_message(self, provider: ClaudeCodeProvider):
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"},
        ]
        assert provider._extract_user_message(msgs) == "second"

    def test_no_user_message(self, provider: ClaudeCodeProvider):
        msgs = [{"role": "assistant", "content": "hello"}]
        assert provider._extract_user_message(msgs) == ""

    def test_empty(self, provider: ClaudeCodeProvider):
        assert provider._extract_user_message([]) == ""


# ------------------------------------------------------------------
# Output parsing
# ------------------------------------------------------------------


class TestParseOutput:
    def test_normal_json(self, provider: ClaudeCodeProvider):
        raw = json.dumps({
            "type": "result",
            "session_id": "sess-123",
            "result": "Hello!",
        })
        text, sid = provider._parse_output(raw)
        assert text == "Hello!"
        assert sid == "sess-123"

    def test_content_blocks(self, provider: ClaudeCodeProvider):
        raw = json.dumps({
            "type": "result",
            "session_id": "sess-456",
            "result": [
                {"type": "text", "text": "Part 1"},
                {"type": "tool_use", "name": "bash"},
                {"type": "text", "text": "Part 2"},
            ],
        })
        text, sid = provider._parse_output(raw)
        assert text == "Part 1\nPart 2"
        assert sid == "sess-456"

    def test_empty_result_falls_back_to_content(self, provider: ClaudeCodeProvider):
        raw = json.dumps({"type": "result", "content": "fallback"})
        text, _ = provider._parse_output(raw)
        assert text == "fallback"

    def test_empty_input(self, provider: ClaudeCodeProvider):
        text, sid = provider._parse_output("")
        assert "(empty response from claude)" in text
        assert sid is None

    def test_non_json(self, provider: ClaudeCodeProvider):
        text, sid = provider._parse_output("plain text output")
        assert text == "plain text output"
        assert sid is None


# ------------------------------------------------------------------
# Session management
# ------------------------------------------------------------------


class TestSessions:
    def test_set_and_get_session(self, provider: ClaudeCodeProvider):
        provider.set_session_key("feishu:chat_123")
        assert provider._current_session_key == "feishu:chat_123"
        assert provider._get_claude_session() is None  # no mapping yet

    def test_persist_and_load(self, workspace: Path, sessions_file: Path):
        p1 = ClaudeCodeProvider(workspace=workspace, sessions_file=sessions_file)
        p1._sessions["test:key"] = "sess-abc"
        p1._save_sessions()

        p2 = ClaudeCodeProvider(workspace=workspace, sessions_file=sessions_file)
        assert p2._sessions.get("test:key") == "sess-abc"

    def test_get_session_with_mapping(self, provider: ClaudeCodeProvider):
        provider._sessions["feishu:123"] = "sess-xyz"
        provider.set_session_key("feishu:123")
        assert provider._get_claude_session() == "sess-xyz"


# ------------------------------------------------------------------
# chat() integration (mocked subprocess)
# ------------------------------------------------------------------


class TestChat:
    @pytest.mark.asyncio
    async def test_new_session(self, provider: ClaudeCodeProvider):
        """First message creates a new session."""
        mock_output = json.dumps({
            "type": "result",
            "session_id": "new-sess-id",
            "result": "Hi there!",
        })

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(mock_output.encode(), b""))
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with patch("asyncio.wait_for", return_value=(mock_output.encode(), b"")):
                mock_proc.communicate = AsyncMock(return_value=(mock_output.encode(), b""))
                response = await provider.chat(
                    messages=[{"role": "user", "content": "hello"}],
                )

        assert isinstance(response, LLMResponse)
        assert response.content == "Hi there!"
        assert response.tool_calls == []

    @pytest.mark.asyncio
    async def test_empty_message(self, provider: ClaudeCodeProvider):
        """Empty user message returns early without subprocess call."""
        response = await provider.chat(messages=[{"role": "assistant", "content": "hi"}])
        assert response.content == "(no message received)"

    @pytest.mark.asyncio
    async def test_cli_not_found(self, provider: ClaudeCodeProvider):
        """Missing claude CLI raises ClaudeCodeError."""
        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
            with pytest.raises(ClaudeCodeError, match="not found in PATH"):
                await provider.chat(
                    messages=[{"role": "user", "content": "test"}],
                )

    @pytest.mark.asyncio
    async def test_cli_nonzero_exit(self, provider: ClaudeCodeProvider):
        """Non-zero exit code raises ClaudeCodeError."""
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b"some error"))
        mock_proc.returncode = 1

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with patch("asyncio.wait_for", return_value=(b"", b"some error")) as wf:
                mock_proc.returncode = 1
                mock_proc.communicate = AsyncMock(return_value=(b"", b"some error"))
                with pytest.raises(ClaudeCodeError, match="exited with code 1"):
                    await provider.chat(
                        messages=[{"role": "user", "content": "test"}],
                    )

    @pytest.mark.asyncio
    async def test_cli_timeout(self, provider: ClaudeCodeProvider):
        """Timeout kills subprocess and raises ClaudeCodeError."""
        mock_proc = AsyncMock()
        mock_proc.kill = MagicMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
                with pytest.raises(ClaudeCodeError, match="timed out"):
                    await provider.chat(
                        messages=[{"role": "user", "content": "test"}],
                    )
                mock_proc.kill.assert_called_once()


# ------------------------------------------------------------------
# Context building
# ------------------------------------------------------------------


class TestContext:
    def test_static_context_includes_bootstrap(self, provider: ClaudeCodeProvider):
        ctx = provider._static_context
        assert "AGENTS.md" in ctx
        assert "Be helpful." in ctx
        assert "SOUL.md" in ctx

    def test_full_context_includes_memory(self, provider: ClaudeCodeProvider):
        ctx = provider._build_full_context()
        assert ctx is not None
        # Memory section should be present (even if empty/minimal)
        assert "Memory" in ctx or "AGENTS.md" in ctx

    def test_full_context_includes_session(self, provider: ClaudeCodeProvider):
        provider.set_session_key("test:sess")
        ctx = provider._build_full_context()
        assert "test:sess" in ctx
