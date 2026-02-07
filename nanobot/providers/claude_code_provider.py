"""Claude Code CLI provider - delegates LLM reasoning and tool execution to claude CLI."""

import asyncio
import json
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.providers.base import LLMProvider, LLMResponse

# Default model aliases → full Claude model IDs.
# Override via constructor or config if new models are released.
DEFAULT_MODEL_MAP: dict[str, str] = {
    "opus": "claude-opus-4-6",
    "sonnet": "claude-sonnet-4-5-20250929",
    "haiku": "claude-haiku-4-5-20251001",
}

# Default timeout (seconds) for the claude CLI subprocess.
DEFAULT_TIMEOUT = 300  # 5 minutes


class ClaudeCodeError(Exception):
    """Raised when the claude CLI returns an error."""
    pass


class ClaudeCodeProvider(LLMProvider):
    """
    LLM provider that delegates to the Claude Code CLI.

    Instead of calling an LLM API directly, this provider runs:
        claude -p <message> --output-format json [--resume <session_id>]

    Claude Code handles all reasoning and tool execution internally.
    The agent loop's tool cycle is bypassed (tool_calls always empty).
    """

    def __init__(
        self,
        workspace: Path,
        model: str = "opus",
        sessions_file: Path | None = None,
        model_map: dict[str, str] | None = None,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        """
        Args:
            workspace: Path to the nanobot workspace directory.
            model: Short model name (e.g. "opus") or full model ID.
            sessions_file: Path to persist session mappings. Defaults to ~/.nanobot/claude_sessions.json.
            model_map: Optional override for short-name → full-model-ID mapping.
            timeout: Subprocess timeout in seconds. Default 300s (5 min).
        """
        super().__init__(api_key=None, api_base=None)
        self.workspace = Path(workspace).expanduser().resolve()
        self.model = model
        self.timeout = timeout
        self._model_map = model_map or DEFAULT_MODEL_MAP
        self._sessions_file = sessions_file or (Path.home() / ".nanobot" / "claude_sessions.json")
        self._current_session_key: str | None = None
        self._sessions: dict[str, str] = self._load_sessions()
        # Static context (bootstrap files + skills) — built once
        self._static_context: str | None = None
        self._build_static_context()

    # ------------------------------------------------------------------
    # LLMProvider interface
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Send message to Claude Code CLI and return the result."""
        user_message = self._extract_user_message(messages)
        if not user_message:
            return LLMResponse(content="(no message received)", tool_calls=[])

        session_id = self._get_claude_session()
        result, new_session_id = await self._call_claude(user_message, session_id)

        # Persist session mapping
        if new_session_id and self._current_session_key:
            self._sessions[self._current_session_key] = new_session_id
            self._save_sessions()

        return LLMResponse(content=result, tool_calls=[])

    def get_default_model(self) -> str:
        return f"claude-code/{self.model}"

    def set_session_key(self, key: str) -> None:
        """Set the current nanobot session key (e.g. 'feishu:ou_xxx')."""
        self._current_session_key = key

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_user_message(self, messages: list[dict[str, Any]]) -> str:
        """Extract the last user message from the message list."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                # Multi-modal content: extract text parts
                if isinstance(content, list):
                    texts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                    return "\n".join(texts)
        return ""

    def _get_claude_session(self) -> str | None:
        """Get the Claude session ID for the current nanobot session."""
        if self._current_session_key:
            return self._sessions.get(self._current_session_key)
        return None

    async def _call_claude(self, message: str, session_id: str | None) -> tuple[str, str | None]:
        """
        Call claude CLI and return (response_text, session_id).

        Returns:
            Tuple of (response content, claude session_id or None).

        Raises:
            ClaudeCodeError: If the CLI is not found, returns non-zero, or times out.
        """
        cmd = [
            "claude", "-p", message,
            "--output-format", "json",
            "--model", self._resolve_model(),
            "--dangerously-skip-permissions",
        ]

        nanobot_context = self._build_full_context()
        if nanobot_context:
            cmd.extend(["--append-system-prompt", nanobot_context])

        if session_id:
            cmd.extend(["--resume", session_id])

        logger.info(f"Calling claude CLI (session={session_id or 'new'}, model={self.model})")
        logger.debug(f"Command: {' '.join(cmd[:6])}...")

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.workspace),
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.timeout
            )

            if proc.returncode != 0:
                err = stderr.decode("utf-8", errors="replace").strip()
                logger.error(f"claude CLI failed (rc={proc.returncode}): {err}")
                raise ClaudeCodeError(f"claude CLI exited with code {proc.returncode}: {err}")

            raw = stdout.decode("utf-8", errors="replace").strip()
            return self._parse_output(raw)

        except asyncio.TimeoutError:
            proc.kill()
            logger.error(f"claude CLI timed out after {self.timeout}s")
            raise ClaudeCodeError(f"claude CLI timed out after {self.timeout}s")
        except FileNotFoundError:
            logger.error("claude CLI not found in PATH")
            raise ClaudeCodeError(
                "claude CLI not found in PATH. "
                "See https://docs.anthropic.com/en/docs/claude-code for installation."
            )
        except ClaudeCodeError:
            raise
        except Exception as e:
            logger.error(f"claude CLI exception: {e}")
            raise ClaudeCodeError(f"Unexpected error: {e}") from e

    def _resolve_model(self) -> str:
        """Map short model names to full Claude model IDs."""
        return self._model_map.get(self.model, self.model)

    def _parse_output(self, raw: str) -> tuple[str, str | None]:
        """
        Parse claude CLI JSON output.

        Expected format:
        {
            "type": "result",
            "session_id": "...",
            "result": "response text",
            ...
        }
        """
        if not raw:
            return "(empty response from claude)", None

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Non-JSON output — treat as plain text
            logger.warning("claude CLI returned non-JSON output")
            return raw, None

        session_id = data.get("session_id")
        result = data.get("result", "")

        # result can be a string or a list of content blocks
        if isinstance(result, list):
            texts = []
            for block in result:
                if isinstance(block, dict) and block.get("type") == "text":
                    texts.append(block.get("text", ""))
            result = "\n".join(texts)

        if not result:
            result = data.get("content", "(no content)")

        return result, session_id

    # ------------------------------------------------------------------
    # Nanobot context (passed via --append-system-prompt)
    # ------------------------------------------------------------------

    def _build_static_context(self) -> None:
        """
        Build static context (bootstrap files + skills) — only needs to run once.

        Includes: USER.md, AGENTS.md, SOUL.md, Skills summary, memory instructions.
        Excludes: nanobot identity, TOOLS.md (Claude Code has its own).
        """
        parts = []

        # Bootstrap files (selective)
        for filename in ["USER.md", "AGENTS.md", "SOUL.md"]:
            file_path = self.workspace / filename
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8").strip()
                if content:
                    parts.append(f"## {filename}\n\n{content}")

        # Skills summary
        try:
            from nanobot.agent.skills import SkillsLoader
            skills = SkillsLoader(self.workspace)
            summary = skills.build_skills_summary()
            if summary:
                parts.append(f"## Available Skills\n\n{summary}")
        except Exception as e:
            logger.warning(f"Failed to load skills: {e}")

        # Memory management instructions
        memory_dir = str(self.workspace / "memory")
        parts.append(f"""## Memory Management

You have a persistent memory system at {memory_dir}/.

### Long-term Memory
- File: {memory_dir}/MEMORY.md
- Purpose: Store user preferences, important facts, and solutions that should persist across sessions.
- Proactively update this file when you encounter important information.

### Daily Notes
- File: {memory_dir}/YYYY-MM-DD.md (e.g. 2026-02-07.md)
- Purpose: Record daily conversation highlights, actions taken, and TODOs.
- Append noteworthy content at the end of conversations.
- If the file does not exist, create it with `# YYYY-MM-DD` as the heading.

### Guidelines
- User preferences and habits → write to MEMORY.md
- Important actions and results → write to daily notes
- Things the user explicitly asks you to remember → write to MEMORY.md
- Skip trivial or casual conversation.""")

        if parts:
            self._static_context = "\n\n---\n\n".join(parts)
        else:
            self._static_context = None

    def _build_full_context(self) -> str | None:
        """
        Build full context for each call: static context + fresh memory.

        Memory is re-read on every call so Claude always sees the latest.
        """
        parts = []

        if self._static_context:
            parts.append(self._static_context)

        # Fresh memory (re-read every call)
        try:
            from nanobot.agent.memory import MemoryStore
            memory = MemoryStore(self.workspace)
            mem_ctx = memory.get_memory_context()
            if mem_ctx:
                parts.append(f"## Current Memory\n\n{mem_ctx}")
        except Exception as e:
            logger.warning(f"Failed to load memory: {e}")

        # Session info
        if self._current_session_key:
            parts.append(f"## Current Session\nSession: {self._current_session_key}")

        return "\n\n---\n\n".join(parts) if parts else None

    # ------------------------------------------------------------------
    # Session persistence
    # ------------------------------------------------------------------

    def _load_sessions(self) -> dict[str, str]:
        """Load nanobot→claude session mapping from disk."""
        if self._sessions_file.exists():
            try:
                return json.loads(self._sessions_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load claude sessions: {e}")
        return {}

    def _save_sessions(self) -> None:
        """Save nanobot→claude session mapping to disk."""
        try:
            self._sessions_file.parent.mkdir(parents=True, exist_ok=True)
            self._sessions_file.write_text(
                json.dumps(self._sessions, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as e:
            logger.warning(f"Failed to save claude sessions: {e}")
