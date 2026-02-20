"""Agent loop: the core processing engine."""

import asyncio
import json
import re
from pathlib import Path
from typing import Awaitable, Callable

import json_repair
from loguru import logger

from nanobot.agent.context import ContextBuilder
from nanobot.agent.memory import MemoryStore
from nanobot.agent.subagent import SubagentManager
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.memory_tool import MemorySearchTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.session.manager import Session, SessionManager


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 20,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        memory_window: int = 50,
        brave_api_key: str | None = None,
        exec_config: "ExecToolConfig | None" = None,
        subagent_config: "SubagentConfig | None" = None,
        cron_service: "CronService | None" = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
    ):
        from nanobot.config.schema import ExecToolConfig, SubagentConfig
        self.bus = bus
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.memory_window = memory_window
        self.brave_api_key = brave_api_key
        self.exec_config = exec_config or ExecToolConfig()
        self.subagent_config = subagent_config or SubagentConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace

        self.context = ContextBuilder(workspace)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            brave_api_key=brave_api_key,
            exec_config=self.exec_config,
            subagent_config=self.subagent_config,
            restrict_to_workspace=restrict_to_workspace,
        )

        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_handles: list = []
        self._mcp_connected = False
        self._consolidating: set[str] = set()  # Session keys with consolidation in progress
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        # File tools (restrict to workspace if configured)
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        self.tools.register(ReadFileTool(allowed_dir=allowed_dir))
        self.tools.register(WriteFileTool(allowed_dir=allowed_dir))
        self.tools.register(EditFileTool(allowed_dir=allowed_dir))
        self.tools.register(ListDirTool(allowed_dir=allowed_dir))

        # Shell tool
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
        ))

        # Web tools
        self.tools.register(WebSearchTool(api_key=self.brave_api_key))
        self.tools.register(WebFetchTool())

        # Message tool
        message_tool = MessageTool(send_callback=self.bus.publish_outbound)
        self.tools.register(message_tool)

        # Spawn tool (for subagents)
        spawn_tool = SpawnTool(manager=self.subagents)
        self.tools.register(spawn_tool)

        # Cron tool (for scheduling)
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))

        # Memory search tool
        self.tools.register(MemorySearchTool(workspace=self.workspace))

    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        if self._mcp_connected or not self._mcp_servers:
            return
        self._mcp_connected = True
        from nanobot.agent.tools.mcp import connect_mcp_servers
        self._mcp_handles = await connect_mcp_servers(self._mcp_servers, self.tools)

    def _set_tool_context(self, channel: str, chat_id: str, message_id: str | None = None) -> None:
        """Update context for all tools that need routing info."""
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.set_context(channel, chat_id, message_id)

        if spawn_tool := self.tools.get("spawn"):
            if isinstance(spawn_tool, SpawnTool):
                spawn_tool.set_context(channel, chat_id)

        if cron_tool := self.tools.get("cron"):
            if isinstance(cron_tool, CronTool):
                cron_tool.set_context(channel, chat_id)

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """Remove <think>…</think> blocks that some models embed in content."""
        if not text:
            return None
        return re.sub(r"<think>[\s\S]*?</think>", "", text).strip() or None

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        """Format tool calls as concise hint, e.g. 'web_search("query")'."""
        def _fmt(tc):
            val = next(iter(tc.arguments.values()), None) if tc.arguments else None
            if not isinstance(val, str):
                return tc.name
            return f'{tc.name}("{val[:40]}…")' if len(val) > 40 else f'{tc.name}("{val}")'
        return ", ".join(_fmt(tc) for tc in tool_calls)

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> tuple[str | None, list[str]]:
        """
        Run the agent iteration loop.

        Args:
            initial_messages: Starting messages for the LLM conversation.
            on_progress: Optional callback to push intermediate content to the user.

        Returns:
            Tuple of (final_content, list_of_tools_used).
        """
        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []
        text_only_retried = False

        while iteration < self.max_iterations:
            iteration += 1

            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            if response.has_tool_calls:
                if on_progress:
                    clean = self._strip_think(response.content)
                    if clean:
                        await on_progress(clean)
                    await on_progress(self._tool_hint(response.tool_calls))

                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False)
                        }
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                )

                for tool_call in response.tool_calls:
                    tools_used.append(tool_call.name)
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info("Tool call: {}({})", tool_call.name, args_str[:200])
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
            else:
                final_content = self._strip_think(response.content)
                # Some models send an interim text response before tool calls.
                # Give them one retry; don't forward the text to avoid duplicates.
                if not tools_used and not text_only_retried and final_content:
                    text_only_retried = True
                    logger.debug("Interim text response (no tools used yet), retrying: {}", final_content[:80])
                    messages = self.context.add_assistant_message(
                        messages, response.content,
                        reasoning_content=response.reasoning_content,
                    )
                    # Add a user nudge so the conversation ends with a user
                    # message — required by models that reject assistant prefill.
                    messages.append({"role": "user", "content": "Continue. Use your tools if needed."})
                    final_content = None
                    continue
                break

        return final_content, tools_used

    async def run(self) -> None:
        """Run the agent loop, processing messages from the bus."""
        self._running = True
        await self._connect_mcp()
        logger.info("Agent loop started")

        while self._running:
            try:
                msg = await asyncio.wait_for(
                    self.bus.consume_inbound(),
                    timeout=1.0
                )
                try:
                    response = await self._process_message(msg)
                    if response:
                        await self.bus.publish_outbound(response)
                except Exception as e:
                    logger.error("Error processing message: {}", e)
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=f"Sorry, I encountered an error: {str(e)}"
                    ))
            except asyncio.TimeoutError:
                continue

    async def close_mcp(self) -> None:
        """Close MCP connections."""
        for handle in self._mcp_handles:
            try:
                await handle.close()
            except (RuntimeError, BaseExceptionGroup, Exception):
                pass  # MCP SDK cancel scope cleanup is noisy but harmless
        self._mcp_handles = []

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """
        Process a single inbound message.

        Args:
            msg: The inbound message to process.
            session_key: Override session key (used by process_direct).
            on_progress: Optional callback for intermediate output (defaults to bus publish).

        Returns:
            The response message, or None if no response needed.
        """
        # System messages route back via chat_id ("channel:chat_id")
        if msg.channel == "system":
            return await self._process_system_message(msg)

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)

        # Handle slash commands
        cmd = msg.content.strip().lower()
        if cmd == "/new":
            # Capture messages before clearing (avoid race condition with background task)
            messages_to_archive = session.messages.copy()
            session.clear()
            self.sessions.save(session)
            self.sessions.invalidate(session.key)

            async def _consolidate_and_cleanup():
                temp_session = Session(key=session.key)
                temp_session.messages = messages_to_archive
                await self._consolidate_memory(temp_session, archive_all=True)

            asyncio.create_task(_consolidate_and_cleanup())
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="New session started. Memory consolidation in progress.")
        if cmd == "/clear":
            msg_count = len(session.messages)
            session.clear()
            self.sessions.save(session)
            logger.info("Session cleared for {} (cleared {} messages)", key, msg_count)
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="Conversation history cleared. Let's start fresh!")
        if cmd == "/compact":
            if hasattr(self.sessions, "compact_session"):
                success, message = await self.sessions.compact_session(key)
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=message)
            else:
                # Fallback: use memory consolidation
                await self._consolidate_memory(session, archive_all=True)
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                      content="Conversation compacted.")
        if cmd == "/skills":
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content=self._list_skills())
        if cmd == "/models":
            main_model = self.provider.get_default_model()
            compact_model = getattr(self, "compact_model", "") or main_model
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content=f"Current models:\n  Main: {main_model}\n  Compact: {compact_model}")
        if cmd == "/mcp":
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content=self._list_mcp_servers())
        if cmd == "/help":
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="nanobot commands:\n\n"
                                          "- /new — Start a new conversation\n"
                                          "- /clear — Clear conversation history\n"
                                          "- /compact — Compact conversation history\n"
                                          "- /skills — List available skills\n"
                                          "- /models — Show current model config\n"
                                          "- /mcp — List configured MCP servers\n"
                                          "- /help — Show available commands")

        if len(session.messages) > self.memory_window and key not in self._consolidating:
            self._consolidating.add(key)

            async def _consolidate_and_unlock():
                try:
                    await self._consolidate_memory(session)
                finally:
                    self._consolidating.discard(key)

            asyncio.create_task(_consolidate_and_unlock())

        self._set_tool_context(msg.channel, msg.chat_id, msg.metadata.get("message_id"))
        initial_messages = self.context.build_messages(
            history=session.get_history(max_messages=self.memory_window),
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel,
            chat_id=msg.chat_id,
        )

        async def _bus_progress(content: str) -> None:
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=content,
                metadata=msg.metadata or {},
            ))

        final_content, tools_used = await self._run_agent_loop(
            initial_messages, on_progress=on_progress or _bus_progress,
        )

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)

        session.add_message("user", msg.content)
        session.add_message("assistant", final_content,
                            tools_used=tools_used if tools_used else None)
        self.sessions.save(session)

        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            metadata=msg.metadata or {},  # Pass through for channel-specific needs (e.g. Slack thread_ts)
        )

    async def _process_system_message(self, msg: InboundMessage) -> OutboundMessage | None:
        """
        Process a system message (e.g., subagent announce).

        The chat_id field contains "original_channel:original_chat_id" to route
        the response back to the correct destination.
        """
        logger.info("Processing system message from {}", msg.sender_id)

        # Parse origin from chat_id (format: "channel:chat_id")
        if ":" in msg.chat_id:
            parts = msg.chat_id.split(":", 1)
            origin_channel = parts[0]
            origin_chat_id = parts[1]
        else:
            # Fallback
            origin_channel = "cli"
            origin_chat_id = msg.chat_id

        session_key = f"{origin_channel}:{origin_chat_id}"
        session = self.sessions.get_or_create(session_key)
        self._set_tool_context(origin_channel, origin_chat_id, msg.metadata.get("message_id"))
        initial_messages = self.context.build_messages(
            history=session.get_history(max_messages=self.memory_window),
            current_message=msg.content,
            channel=origin_channel,
            chat_id=origin_chat_id,
        )
        final_content, _ = await self._run_agent_loop(initial_messages)

        if final_content is None:
            final_content = "Background task completed."

        session.add_message("user", f"[System: {msg.sender_id}] {msg.content}")
        session.add_message("assistant", final_content)
        self.sessions.save(session)

        return OutboundMessage(
            channel=origin_channel,
            chat_id=origin_chat_id,
            content=final_content
        )

    def _list_skills(self) -> str:
        """List available skills from workspace."""
        skills_dir = self.workspace / "skills"
        if not skills_dir.exists():
            return "No skills directory found."
        skills: list[str] = []
        for skill_path in sorted(skills_dir.iterdir()):
            if not skill_path.is_dir():
                continue
            skill_md = skill_path / "SKILL.md"
            if not skill_md.exists():
                continue
            skill_name = skill_path.name
            description = ""
            try:
                content = skill_md.read_text(encoding="utf-8")
                if content.startswith("---"):
                    end = content.find("---", 3)
                    if end != -1:
                        for line in content[3:end].split("\n"):
                            line = line.strip()
                            if line.startswith("description:"):
                                description = line[len("description:"):].strip().strip("\"'")
                                break
            except Exception:
                pass
            if len(description) > 80:
                description = description[:77] + "..."
            entry = f"  {skill_name}" + (f" - {description}" if description else "")
            skills.append(entry)
        if not skills:
            return "No skills found."
        return "Available skills:\n" + "\n".join(skills)

    def _list_mcp_servers(self) -> str:
        """List configured MCP servers."""
        if not self._mcp_servers:
            return "No MCP servers configured."
        lines: list[str] = []
        for name, cfg in self._mcp_servers.items():
            if hasattr(cfg, "url") and cfg.url:
                lines.append(f"  {name} — {cfg.url}")
            elif hasattr(cfg, "command") and cfg.command:
                args = " ".join(cfg.args) if hasattr(cfg, "args") and cfg.args else ""
                lines.append(f"  {name} — {cfg.command} {args}".rstrip())
            else:
                # dict-style config (from JSON)
                url = cfg.get("url", "") if isinstance(cfg, dict) else ""
                cmd = cfg.get("command", "") if isinstance(cfg, dict) else ""
                if url:
                    lines.append(f"  {name} — {url}")
                elif cmd:
                    args = " ".join(cfg.get("args", []))
                    lines.append(f"  {name} — {cmd} {args}".rstrip())
                else:
                    lines.append(f"  {name}")
        status = " (connected)" if self._mcp_connected else ""
        return f"MCP servers{status}:\n" + "\n".join(lines)

    async def _consolidate_memory(self, session, archive_all: bool = False) -> None:
        """Consolidate old messages into MEMORY.md + HISTORY.md.

        Args:
            archive_all: If True, clear all messages and reset session (for /new command).
                       If False, only write to files without modifying session.
        """
        memory = MemoryStore(self.workspace)

        if archive_all:
            old_messages = session.messages
            keep_count = 0
            logger.info("Memory consolidation (archive_all): {} total messages archived", len(session.messages))
        else:
            keep_count = self.memory_window // 2
            if len(session.messages) <= keep_count:
                logger.debug("Session {}: No consolidation needed (messages={}, keep={})", session.key, len(session.messages), keep_count)
                return

            messages_to_process = len(session.messages) - session.last_consolidated
            if messages_to_process <= 0:
                logger.debug("Session {}: No new messages to consolidate (last_consolidated={}, total={})", session.key, session.last_consolidated, len(session.messages))
                return

            old_messages = session.messages[session.last_consolidated:-keep_count]
            if not old_messages:
                return
            logger.info("Memory consolidation started: {} total, {} new to consolidate, {} keep", len(session.messages), len(old_messages), keep_count)

        lines = []
        for m in old_messages:
            if not m.get("content"):
                continue
            tools = f" [tools: {', '.join(m['tools_used'])}]" if m.get("tools_used") else ""
            lines.append(f"[{m.get('timestamp', '?')[:16]}] {m['role'].upper()}{tools}: {m['content']}")
        conversation = "\n".join(lines)
        current_memory = memory.read_long_term()

        prompt = f"""You are a memory consolidation agent. Process this conversation and return a JSON object with exactly two keys:

1. "history_entry": A paragraph (2-5 sentences) summarizing the key events/decisions/topics. Start with a timestamp like [YYYY-MM-DD HH:MM]. Include enough detail to be useful when found by grep search later.

2. "memory_update": The updated long-term memory content. Add any new facts: user location, preferences, personal info, habits, project context, technical decisions, tools/services used. If nothing new, return the existing content unchanged.

## Current Long-term Memory
{current_memory or "(empty)"}

## Conversation to Process
{conversation}

**IMPORTANT**: Both values MUST be strings, not objects or arrays.

Example:
{{
  "history_entry": "[2026-02-14 22:50] User asked about...",
  "memory_update": "- Host: HARRYBOOK-T14P\n- Name: Nado"
}}

Respond with ONLY valid JSON, no markdown fences."""

        try:
            response = await self.provider.chat(
                messages=[
                    {"role": "system", "content": "You are a memory consolidation agent. Respond only with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                model=self.model,
            )
            text = (response.content or "").strip()
            if not text:
                logger.warning("Memory consolidation: LLM returned empty response, skipping")
                return
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            result = json_repair.loads(text)
            if not isinstance(result, dict):
                logger.warning("Memory consolidation: unexpected response type, skipping. Response: {}", text[:200])
                return

            if entry := result.get("history_entry"):
                # Defensive: ensure entry is a string (LLM may return dict)
                if not isinstance(entry, str):
                    entry = json.dumps(entry, ensure_ascii=False)
                memory.append_history(entry)
            if update := result.get("memory_update"):
                # Defensive: ensure update is a string
                if not isinstance(update, str):
                    update = json.dumps(update, ensure_ascii=False)
                if update != current_memory:
                    memory.write_long_term(update)

            if archive_all:
                session.last_consolidated = 0
            else:
                session.last_consolidated = len(session.messages) - keep_count
            logger.info("Memory consolidation done: {} messages, last_consolidated={}", len(session.messages), session.last_consolidated)
        except Exception as e:
            logger.error("Memory consolidation failed: {}", e)

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        """
        Process a message directly (for CLI or cron usage).

        Args:
            content: The message content.
            session_key: Session identifier (overrides channel:chat_id for session lookup).
            channel: Source channel (for tool context routing).
            chat_id: Source chat ID (for tool context routing).
            on_progress: Optional callback for intermediate output.

        Returns:
            The agent's response.
        """
        msg = InboundMessage(
            channel=channel,
            sender_id="user",
            chat_id=chat_id,
            content=content
        )

        # Defer MCP connection until actually needed (slash commands don't need it)
        if not msg.content.strip().startswith("/"):
            await self._connect_mcp()

        response = await self._process_message(msg, session_key=session_key, on_progress=on_progress)
        return response.content if response else ""
