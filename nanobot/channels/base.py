"""Base channel interface for chat platforms."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus

if TYPE_CHECKING:
    from nanobot.session.manager import SessionManager

COMPACT_COOLDOWN_SECONDS = 300  # 5 minutes


class BaseChannel(ABC):
    """
    Abstract base class for chat channel implementations.

    Each channel (Telegram, Discord, etc.) should implement this interface
    to integrate with the nanobot message bus.
    """

    name: str = "base"

    def __init__(self, config: Any, bus: MessageBus, session_manager: SessionManager | None = None):
        """
        Initialize the channel.

        Args:
            config: Channel-specific configuration.
            bus: The message bus for communication.
            session_manager: Optional session manager for slash commands.
        """
        self.config = config
        self.bus = bus
        self.session_manager = session_manager
        self._running = False
        self._compact_cooldowns: dict[str, float] = {}

    @abstractmethod
    async def start(self) -> None:
        """
        Start the channel and begin listening for messages.

        This should be a long-running async task that:
        1. Connects to the chat platform
        2. Listens for incoming messages
        3. Forwards messages to the bus via _handle_message()
        """
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the channel and clean up resources."""
        pass

    @abstractmethod
    async def send(self, msg: OutboundMessage) -> None:
        """
        Send a message through this channel.

        Args:
            msg: The message to send.
        """
        pass

    def is_allowed(self, sender_id: str) -> bool:
        """
        Check if a sender is allowed to use this bot.

        Args:
            sender_id: The sender's identifier.

        Returns:
            True if allowed, False otherwise.
        """
        allow_list = getattr(self.config, "allow_from", [])

        # If no allow list, allow everyone
        if not allow_list:
            return True

        sender_str = str(sender_id)
        if sender_str in allow_list:
            return True
        if "|" in sender_str:
            for part in sender_str.split("|"):
                if part and part in allow_list:
                    return True
        return False

    async def _handle_message(
        self,
        sender_id: str,
        chat_id: str,
        content: str,
        media: list[str] | None = None,
        metadata: dict[str, Any] | None = None
    ) -> None:
        """
        Handle an incoming message from the chat platform.

        This method checks permissions and forwards to the bus.

        Args:
            sender_id: The sender's identifier.
            chat_id: The chat/channel identifier.
            content: Message text content.
            media: Optional list of media URLs.
            metadata: Optional channel-specific metadata.
        """
        if not self.is_allowed(sender_id):
            logger.warning(
                f"Access denied for sender {sender_id} on channel {self.name}. "
                f"Add them to allowFrom list in config to grant access."
            )
            return

        msg = InboundMessage(
            channel=self.name,
            sender_id=str(sender_id),
            chat_id=str(chat_id),
            content=content,
            media=media or [],
            metadata=metadata or {}
        )

        await self.bus.publish_inbound(msg)

    @property
    def is_running(self) -> bool:
        """Check if the channel is running."""
        return self._running

    # ------------------------------------------------------------------
    # Slash command framework
    # ------------------------------------------------------------------

    async def _send_reply(self, chat_id: str, text: str) -> None:
        """Send a text reply to a chat. Subclasses can override for native API."""
        await self.send(OutboundMessage(channel=self.name, chat_id=chat_id, content=text))

    async def _try_handle_command(self, content: str, chat_id: str, sender_id: str = "") -> bool:
        """
        Try to handle a slash command. Returns True if handled, False otherwise.
        Unknown commands return False so the message passes through normally.
        """
        if not content.startswith("/"):
            return False

        if sender_id and not self.is_allowed(sender_id):
            return False

        command = content.split()[0].lower()

        if command == "/clear":
            await self._cmd_clear(chat_id)
            return True
        elif command == "/compact":
            await self._cmd_compact(chat_id)
            return True
        elif command == "/skills":
            await self._cmd_skills(chat_id)
            return True
        elif command == "/models":
            await self._cmd_models(chat_id)
            return True
        elif command == "/help":
            await self._cmd_help(chat_id)
            return True

        return False

    async def _cmd_clear(self, chat_id: str) -> None:
        """Clear conversation history."""
        if not self.session_manager:
            await self._send_reply(chat_id, "Session management is not available.")
            return

        session_key = f"{self.name}:{chat_id}"
        session = self.session_manager.get_or_create(session_key)
        msg_count = len(session.messages)
        session.clear()
        self.session_manager.save(session)

        logger.info(f"Session cleared for {session_key} (cleared {msg_count} messages)")
        await self._send_reply(chat_id, "Conversation history cleared. Let's start fresh!")

    async def _cmd_compact(self, chat_id: str) -> None:
        """Compact conversation history with LLM summary."""
        if not self.session_manager:
            await self._send_reply(chat_id, "Session management is not available.")
            return

        # Cooldown check
        now = time.time()
        last = self._compact_cooldowns.get(chat_id, 0)
        remaining = COMPACT_COOLDOWN_SECONDS - (now - last)

        if remaining > 0:
            minutes = int(remaining // 60)
            seconds = int(remaining % 60)
            await self._send_reply(
                chat_id,
                f"Please wait {minutes}m {seconds}s before using /compact again."
            )
            return

        await self._send_reply(chat_id, "Compacting conversation history...")

        session_key = f"{self.name}:{chat_id}"
        success, message = await self.session_manager.compact_session(session_key)

        if success:
            self._compact_cooldowns[chat_id] = now
            logger.info(f"Session compacted for {session_key}")
            await self._send_reply(chat_id, message)
        else:
            logger.warning(f"Failed to compact session {session_key}: {message}")
            await self._send_reply(chat_id, message)

    async def _cmd_skills(self, chat_id: str) -> None:
        """List available skills."""
        if not self.session_manager:
            await self._send_reply(chat_id, "Session management is not available.")
            return

        try:
            skills_dir = self.session_manager.workspace / "skills"
            if not skills_dir.exists():
                await self._send_reply(chat_id, "No skills directory found.")
                return

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
                    # Parse YAML frontmatter description
                    if content.startswith("---"):
                        end = content.find("---", 3)
                        if end != -1:
                            for line in content[3:end].split("\n"):
                                line = line.strip()
                                if line.startswith("description:"):
                                    description = line[len("description:"):].strip().strip("\"'")
                                    break
                    if not description:
                        for line in content.split("\n"):
                            line = line.strip()
                            if line and not line.startswith("#") and not line.startswith("---"):
                                description = line
                                break
                except Exception:
                    pass

                if len(description) > 80:
                    description = description[:77] + "..."
                entry = f"  {skill_name}" + (f" - {description}" if description else "")
                skills.append(entry)

            if not skills:
                await self._send_reply(chat_id, "No skills found.")
                return

            text = "Available skills:\n" + "\n".join(skills)
            await self._send_reply(chat_id, text)
        except Exception as e:
            logger.error(f"Error listing skills: {e}")
            await self._send_reply(chat_id, f"Error listing skills: {e}")

    async def _cmd_models(self, chat_id: str) -> None:
        """Show current model configuration."""
        if not self.session_manager or not self.session_manager.provider:
            await self._send_reply(chat_id, "Model info not available.")
            return

        main_model = self.session_manager.provider.get_default_model()
        compact_model = self.session_manager.compact_model or main_model
        lines = [
            "Current models:",
            f"  Main: {main_model}",
            f"  Compact: {compact_model}",
        ]
        await self._send_reply(chat_id, "\n".join(lines))

    async def _cmd_help(self, chat_id: str) -> None:
        """Show help message."""
        help_text = (
            "nanobot commands:\n\n"
            "/clear - Clear conversation history\n"
            "/compact - Compact conversation history (5min cooldown)\n"
            "/skills - List available skills\n"
            "/models - Show current model configuration\n"
            "/help - Show this help message\n\n"
            "Send a message to chat!"
        )
        await self._send_reply(chat_id, help_text)
