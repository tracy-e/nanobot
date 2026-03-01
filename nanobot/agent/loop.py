"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import json
import re
from contextlib import AsyncExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

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

if TYPE_CHECKING:
    from nanobot.config.schema import ChannelsConfig, ExecToolConfig
    from nanobot.cron.service import CronService


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
        max_iterations: int = 40,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        memory_window: int = 100,
        brave_api_key: str | None = None,
        exec_config: ExecToolConfig | None = None,
        cron_service: CronService | None = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
        compact_model: str = "",
        channels_config: ChannelsConfig | None = None,
        provider_factory: Callable[[str], LLMProvider] | None = None,
        available_models: list[str] | None = None,
        data_dir: Path | None = None,
    ):
        from nanobot.config.schema import ExecToolConfig
        self.bus = bus
        self.channels_config = channels_config
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.compact_model = compact_model
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.memory_window = memory_window
        self.brave_api_key = brave_api_key
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace

        self._data_dir = data_dir

        # /model switching support
        self._provider_factory = provider_factory
        self._available_models = available_models or []
        self._providers: dict[str, LLMProvider] = {}
        self._providers[self._provider_key(self.model)] = provider

        # compact_model uses its own provider, not affected by /model switching
        self._compact_provider: LLMProvider = provider
        if compact_model and self._provider_key(compact_model) != self._provider_key(self.model):
            if provider_factory:
                try:
                    self._compact_provider = provider_factory(compact_model)
                    self._providers[self._provider_key(compact_model)] = self._compact_provider
                except Exception:
                    pass  # Fall back to main provider

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
            restrict_to_workspace=restrict_to_workspace,
        )

        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._mcp_connecting = False
        self._consolidating: set[str] = set()  # Session keys with consolidation in progress
        self._consolidation_tasks: set[asyncio.Task] = set()  # Strong refs to in-flight tasks
        self._consolidation_locks: dict[str, asyncio.Lock] = {}
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        for cls in (ReadFileTool, WriteFileTool, EditFileTool, ListDirTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
        ))
        self.tools.register(WebSearchTool(api_key=self.brave_api_key))
        self.tools.register(WebFetchTool())
        self.tools.register(MessageTool(send_callback=self.bus.publish_outbound))
        self.tools.register(SpawnTool(manager=self.subagents))
        self.tools.register(MemorySearchTool(workspace=self.workspace))
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))

    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        if self._mcp_connected or self._mcp_connecting or not self._mcp_servers:
            return
        self._mcp_connecting = True
        from nanobot.agent.tools.mcp import connect_mcp_servers
        try:
            self._mcp_stack = AsyncExitStack()
            await self._mcp_stack.__aenter__()
            await connect_mcp_servers(self._mcp_servers, self.tools, self._mcp_stack)
            self._mcp_connected = True
        except Exception as e:
            logger.error("Failed to connect MCP servers (will retry next message): {}", e)
            if self._mcp_stack:
                try:
                    await self._mcp_stack.aclose()
                except Exception:
                    pass
                self._mcp_stack = None
        finally:
            self._mcp_connecting = False

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
        """Remove <think>…</think> blocks that some models embed in content.

        Handles paired blocks, orphaned </think> (when the opening tag was
        split into reasoning_content), and unclosed <think>... at the end.
        """
        if not text:
            return None
        # 1. Paired blocks
        text = re.sub(r"<think>[\s\S]*?</think>", "", text)
        # 2. Unclosed <think>... at end of string
        text = re.sub(r"<think>[\s\S]*$", "", text)
        # 3. Orphaned tags
        text = re.sub(r"</?think>", "", text)
        return text.strip() or None

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
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> tuple[str | None, list[str], list[dict]]:
        """Run the agent iteration loop. Returns (final_content, tools_used, messages)."""
        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []

        while iteration < self.max_iterations:
            iteration += 1

            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            if response.usage:
                logger.info("Token usage: {}", response.usage)

            if response.has_tool_calls:
                if on_progress:
                    clean = self._strip_think(response.content)
                    if clean:
                        await on_progress(clean)
                    await on_progress(self._tool_hint(response.tool_calls), tool_hint=True)

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
                # Save the final assistant response into messages so
                # _save_turn persists it into session history.
                messages = self.context.add_assistant_message(
                    messages, final_content,
                    reasoning_content=response.reasoning_content,
                )
                break

        if final_content is None and iteration >= self.max_iterations:
            logger.warning("Max iterations ({}) reached", self.max_iterations)
            final_content = (
                f"I reached the maximum number of tool call iterations ({self.max_iterations}) "
                "without completing the task. You can try breaking the task into smaller steps."
            )

        return final_content, tools_used, messages

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
                    if response is not None:
                        await self.bus.publish_outbound(response)
                    elif msg.channel == "cli":
                        await self.bus.publish_outbound(OutboundMessage(
                            channel=msg.channel, chat_id=msg.chat_id, content="", metadata=msg.metadata or {},
                        ))
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
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass  # MCP SDK cancel scope cleanup is noisy but harmless
            self._mcp_stack = None

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")

    def _get_consolidation_lock(self, session_key: str) -> asyncio.Lock:
        lock = self._consolidation_locks.get(session_key)
        if lock is None:
            lock = asyncio.Lock()
            self._consolidation_locks[session_key] = lock
        return lock

    def _prune_consolidation_lock(self, session_key: str, lock: asyncio.Lock) -> None:
        """Drop lock entry if no longer in use."""
        if not lock.locked():
            self._consolidation_locks.pop(session_key, None)

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """Process a single inbound message and return the response."""
        # System messages: parse origin from chat_id ("channel:chat_id")
        if msg.channel == "system":
            channel, chat_id = (msg.chat_id.split(":", 1) if ":" in msg.chat_id
                                else ("cli", msg.chat_id))
            logger.info("Processing system message from {}", msg.sender_id)
            key = f"{channel}:{chat_id}"
            session = self.sessions.get_or_create(key)
            self._set_tool_context(channel, chat_id, msg.metadata.get("message_id"))
            history = session.get_history(max_messages=self.memory_window)
            messages = self.context.build_messages(
                history=history,
                current_message=msg.content, channel=channel, chat_id=chat_id,
            )
            final_content, _, all_msgs = await self._run_agent_loop(messages)
            self._save_turn(session, all_msgs, 1 + len(history))
            self.sessions.save(session)
            return OutboundMessage(channel=channel, chat_id=chat_id,
                                  content=final_content or "Background task completed.")

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)

        # Slash commands
        cmd = msg.content.strip().lower()
        if cmd == "/new":
            lock = self._get_consolidation_lock(session.key)
            self._consolidating.add(session.key)
            try:
                async with lock:
                    snapshot = session.messages[session.last_consolidated:]
                    if snapshot:
                        temp = Session(key=session.key)
                        temp.messages = list(snapshot)
                        if not await self._consolidate_memory(temp, archive_all=True):
                            return OutboundMessage(
                                channel=msg.channel, chat_id=msg.chat_id,
                                content="Memory archival failed, session not cleared. Please try again.",
                            )
            except Exception:
                logger.exception("/new archival failed for {}", session.key)
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="Memory archival failed, session not cleared. Please try again.",
                )
            finally:
                self._consolidating.discard(session.key)
                self._prune_consolidation_lock(session.key, lock)

            session.clear()
            self.sessions.save(session)
            self.sessions.invalidate(session.key)
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="New session started.")
        if cmd == "/clear":
            msg_count = len(session.messages)
            session.clear()
            self.sessions.save(session)
            logger.info("Session cleared for {} (cleared {} messages)", key, msg_count)
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="Conversation history cleared. Let's start fresh!")
        if cmd == "/compact" or cmd.startswith("/compact "):
            arg = msg.content.strip()[8:].strip()
            if arg == "--switch":
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                      content=self._format_compact_model_list())
            elif arg:
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                      content=self._switch_compact_model(arg))
            else:
                if hasattr(self.sessions, "compact_session"):
                    success, message = await self.sessions.compact_session(key)
                    return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=message)
                else:
                    await self._consolidate_memory(session, archive_all=True)
                    return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                          content="Conversation compacted.")
        if cmd == "/skills":
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content=self._list_skills())
        if cmd == "/mcp":
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content=self._list_mcp_servers())
        if cmd == "/model" or cmd.startswith("/model "):
            arg = msg.content.strip()[6:].strip()
            if not arg:
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                      content=self._format_model_list())
            else:
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                      content=self._switch_model(arg))
        if cmd == "/help":
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="nanobot commands:\n\n"
                                          "- /new — Start a new conversation\n"
                                          "- /clear — Clear conversation history\n"
                                          "- /compact — Compact conversation / switch compact model\n"
                                          "- /skills — List available skills\n"
                                          "- /model — Show/switch model\n"
                                          "- /mcp — List configured MCP servers\n"
                                          "- /help — Show available commands")

        unconsolidated = len(session.messages) - session.last_consolidated
        if (unconsolidated >= self.memory_window and session.key not in self._consolidating):
            self._consolidating.add(session.key)
            lock = self._get_consolidation_lock(session.key)

            async def _consolidate_and_unlock():
                try:
                    async with lock:
                        await self._consolidate_memory(session)
                finally:
                    self._consolidating.discard(session.key)
                    self._prune_consolidation_lock(session.key, lock)
                    _task = asyncio.current_task()
                    if _task is not None:
                        self._consolidation_tasks.discard(_task)

            _task = asyncio.create_task(_consolidate_and_unlock())
            self._consolidation_tasks.add(_task)

        self._set_tool_context(msg.channel, msg.chat_id, msg.metadata.get("message_id"))
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()

        history = session.get_history(max_messages=self.memory_window)
        initial_messages = self.context.build_messages(
            history=history,
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel, chat_id=msg.chat_id,
        )

        async def _bus_progress(content: str, *, tool_hint: bool = False) -> None:
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=content, metadata=meta,
            ))

        final_content, _, all_msgs = await self._run_agent_loop(
            initial_messages, on_progress=on_progress or _bus_progress,
        )

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)

        self._save_turn(session, all_msgs, 1 + len(history))
        self.sessions.save(session)

        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool) and message_tool._sent_in_turn:
                return None

        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=final_content,
            metadata=msg.metadata or {},
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

    # -- /model helpers --------------------------------------------------------

    @staticmethod
    def _provider_key(model: str) -> str:
        """Extract provider key from model name (e.g. 'claude-oauth' from 'claude-oauth/claude-sonnet-4-6')."""
        return model.split("/")[0] if "/" in model else "default"

    def _format_model_list(self) -> str:
        """Format current model and available models list."""
        compact = self.compact_model or self.model
        lines = [f"Main: {self.model}", f"Compact: {compact}"]
        if not self._available_models:
            lines.append("\nNo models configured. Add a 'models' list to config.json agents.defaults.")
            return "\n".join(lines)
        lines.append("\nAvailable:")
        for i, m in enumerate(self._available_models, 1):
            marker = "  <-- current" if m == self.model else ""
            lines.append(f"  {i}. {m}{marker}")
        lines.append("\nReply /model <number> to switch.")
        return "\n".join(lines)

    def _switch_model(self, arg: str) -> str:
        """Switch the active model. arg can be a 1-based index or full model name."""
        if not self._available_models:
            return "No models configured. Add a 'models' list to config.json agents.defaults."

        # Resolve target model
        target: str | None = None
        if arg.isdigit():
            idx = int(arg) - 1
            if 0 <= idx < len(self._available_models):
                target = self._available_models[idx]
            else:
                return f"Invalid index. Choose 1-{len(self._available_models)}."
        else:
            # Match by full name or substring
            for m in self._available_models:
                if m == arg or m.lower() == arg.lower():
                    target = m
                    break
            if target is None:
                # Try substring match
                matches = [m for m in self._available_models if arg.lower() in m.lower()]
                if len(matches) == 1:
                    target = matches[0]
                elif len(matches) > 1:
                    return f"Ambiguous match: {', '.join(matches)}"
                else:
                    return f"Model '{arg}' not in available list. Use /model to see options."

        if target == self.model:
            return f"Already using {self.model}."

        # Resolve or create provider
        key = self._provider_key(target)
        if key not in self._providers:
            if not self._provider_factory:
                return "Cannot switch: no provider factory configured."
            try:
                self._providers[key] = self._provider_factory(target)
            except Exception as e:
                return f"Failed to create provider for {target}: {e}"

        old_model = self.model
        self.provider = self._providers[key]
        self.model = target
        self.subagents.provider = self.provider
        self.subagents.model = target
        self._persist_state()
        return f"Switched: {old_model} -> {target}"

    _STATE_FILE = ".state.json"

    def _persist_state(self) -> None:
        """Write model and compact_model to .state.json so they survive restarts."""
        if not self._data_dir:
            return
        try:
            state_path = self._data_dir / self._STATE_FILE
            state: dict[str, str] = {"model": self.model}
            if self.compact_model:
                state["compact_model"] = self.compact_model
            state_path.write_text(json.dumps(state), encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to persist model selection: {}", e)

    @staticmethod
    def load_persisted_state(data_dir: Path) -> dict[str, str]:
        """Read persisted model/compact_model from .state.json."""
        try:
            state_path = data_dir / ".state.json"
            if state_path.is_file():
                return json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            pass
        return {}

    # -- /compact model helpers ------------------------------------------------

    def _format_compact_model_list(self) -> str:
        """Format current compact model and available models list."""
        compact = self.compact_model or self.model
        lines = [f"Compact: {compact}"]
        if not self._available_models:
            lines.append("\nNo models configured. Add a 'models' list to config.json agents.defaults.")
            return "\n".join(lines)
        lines.append("\nAvailable:")
        for i, m in enumerate(self._available_models, 1):
            marker = "  <-- current" if m == compact else ""
            lines.append(f"  {i}. {m}{marker}")
        lines.append("\nReply /compact <number> to switch.")
        return "\n".join(lines)

    def _switch_compact_model(self, arg: str) -> str:
        """Switch the compact model. arg can be a 1-based index or full model name."""
        if not self._available_models:
            return "No models configured. Add a 'models' list to config.json agents.defaults."

        # Resolve target model
        target: str | None = None
        if arg.isdigit():
            idx = int(arg) - 1
            if 0 <= idx < len(self._available_models):
                target = self._available_models[idx]
            else:
                return f"Invalid index. Choose 1-{len(self._available_models)}."
        else:
            for m in self._available_models:
                if m == arg or m.lower() == arg.lower():
                    target = m
                    break
            if target is None:
                matches = [m for m in self._available_models if arg.lower() in m.lower()]
                if len(matches) == 1:
                    target = matches[0]
                elif len(matches) > 1:
                    return f"Ambiguous match: {', '.join(matches)}"
                else:
                    return f"Model '{arg}' not in available list. Use /compact --switch to see options."

        current_compact = self.compact_model or self.model
        if target == current_compact:
            return f"Already using {current_compact} for compact."

        # Resolve or create provider
        key = self._provider_key(target)
        if key not in self._providers:
            if not self._provider_factory:
                return "Cannot switch: no provider factory configured."
            try:
                self._providers[key] = self._provider_factory(target)
            except Exception as e:
                return f"Failed to create provider for {target}: {e}"

        old_compact = current_compact
        self.compact_model = target
        self._compact_provider = self._providers[key]
        self._persist_state()
        return f"Compact model switched: {old_compact} -> {target}"

    # -- end /model helpers ----------------------------------------------------

    _TOOL_RESULT_MAX_CHARS = 500

    def _save_turn(self, session: Session, messages: list[dict], skip: int) -> None:
        """Save new-turn messages into session, truncating large tool results."""
        from datetime import datetime
        for m in messages[skip:]:
            entry = {k: v for k, v in m.items() if k != "reasoning_content"}
            if entry.get("role") == "tool" and isinstance(entry.get("content"), str):
                content = entry["content"]
                if len(content) > self._TOOL_RESULT_MAX_CHARS:
                    entry["content"] = content[:self._TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"
            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)
        session.updated_at = datetime.now()

    async def _consolidate_memory(self, session, archive_all: bool = False) -> bool:
        """Delegate to MemoryStore.consolidate(). Returns True on success."""
        # Always use _compact_provider (not the current switched provider)
        # so memory consolidation works even after /model switching.
        compact_provider = self._compact_provider
        compact_model = self.compact_model or self.model
        return await MemoryStore(self.workspace).consolidate(
            session, compact_provider, compact_model,
            archive_all=archive_all, memory_window=self.memory_window,
        )

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        """Process a message directly (for CLI or cron usage)."""
        await self._connect_mcp()
        msg = InboundMessage(channel=channel, sender_id="user", chat_id=chat_id, content=content)
        response = await self._process_message(msg, session_key=session_key, on_progress=on_progress)
        return response.content if response else ""
