"""Session management for conversation history."""

import json
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TYPE_CHECKING

from loguru import logger

from nanobot.utils.helpers import ensure_dir, safe_filename

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider

COMPACT_KEEP_RECENT = 5       # Messages to preserve after compact
COMPACT_MAX_MESSAGES = 200    # Max messages to include in summary prompt


@dataclass
class Session:
    """
    A conversation session.
    
    Stores messages in JSONL format for easy reading and persistence.
    """
    
    key: str  # channel:chat_id
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """Add a message to the session."""
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        self.messages.append(msg)
        self.updated_at = datetime.now()
    
    def get_history(self, max_messages: int = 50) -> list[dict[str, Any]]:
        """
        Get message history for LLM context.

        Args:
            max_messages: Maximum messages to return (0 = all).

        Returns:
            List of messages in LLM format.
        """
        # Get recent messages (0 means all)
        if max_messages > 0 and len(self.messages) > max_messages:
            recent = self.messages[-max_messages:]
        else:
            recent = self.messages

        # Convert to LLM format (just role and content)
        return [{"role": m["role"], "content": m["content"]} for m in recent]
    
    def clear(self) -> None:
        """Clear all messages in the session."""
        self.messages = []
        self.updated_at = datetime.now()
    
    def compact(self, summary: str) -> int:
        """
        Compact the session by replacing old messages with a summary.
        
        Args:
            summary: The summary to replace old messages with.
        
        Returns:
            Number of messages removed.
        """
        if len(self.messages) <= COMPACT_KEEP_RECENT:
            return 0  # Too few messages to compact

        old_count = len(self.messages)
        recent = self.messages[-COMPACT_KEEP_RECENT:]
        
        # Replace with summary + recent messages
        self.messages = [
            {
                "role": "system",
                "content": f"[Previous conversation summary]\n{summary}",
                "timestamp": datetime.now().isoformat(),
            }
        ] + recent
        
        self.updated_at = datetime.now()
        return old_count - len(self.messages)


class SessionManager:
    """
    Manages conversation sessions.
    
    Sessions are stored as JSONL files in the sessions directory.
    """
    
    def __init__(self, workspace: Path, provider: "LLMProvider | None" = None, compact_model: str = ""):
        self.workspace = workspace
        self.sessions_dir = ensure_dir(Path.home() / ".nanobot" / "sessions")
        self._cache: dict[str, Session] = {}
        self.provider = provider  # For compact functionality
        self.compact_model = compact_model  # Cheaper model for summarization
    
    def _get_session_path(self, key: str) -> Path:
        """Get the file path for a session."""
        safe_key = safe_filename(key.replace(":", "_"))
        return self.sessions_dir / f"{safe_key}.jsonl"
    
    def get_or_create(self, key: str) -> Session:
        """
        Get an existing session or create a new one.
        
        Args:
            key: Session key (usually channel:chat_id).
        
        Returns:
            The session.
        """
        # Check cache
        if key in self._cache:
            return self._cache[key]
        
        # Try to load from disk
        session = self._load(key)
        if session is None:
            session = Session(key=key)
        
        self._cache[key] = session
        return session
    
    def _load(self, key: str) -> Session | None:
        """Load a session from disk."""
        path = self._get_session_path(key)
        
        if not path.exists():
            return None
        
        try:
            messages = []
            metadata = {}
            created_at = None
            
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    data = json.loads(line)
                    
                    if data.get("_type") == "metadata":
                        metadata = data.get("metadata", {})
                        created_at = datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None
                    else:
                        messages.append(data)
            
            return Session(
                key=key,
                messages=messages,
                created_at=created_at or datetime.now(),
                metadata=metadata
            )
        except Exception as e:
            logger.warning(f"Failed to load session {key}: {e}")
            return None
    
    def save(self, session: Session) -> None:
        """Save a session to disk."""
        path = self._get_session_path(session.key)
        
        with open(path, "w") as f:
            # Write metadata first
            metadata_line = {
                "_type": "metadata",
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "metadata": session.metadata
            }
            f.write(json.dumps(metadata_line) + "\n")
            
            # Write messages
            for msg in session.messages:
                f.write(json.dumps(msg) + "\n")
        
        self._cache[session.key] = session
    
    def delete(self, key: str) -> bool:
        """
        Delete a session.
        
        Args:
            key: Session key.
        
        Returns:
            True if deleted, False if not found.
        """
        # Remove from cache
        self._cache.pop(key, None)
        
        # Remove file
        path = self._get_session_path(key)
        if path.exists():
            path.unlink()
            return True
        return False
    
    def list_sessions(self) -> list[dict[str, Any]]:
        """
        List all sessions.
        
        Returns:
            List of session info dicts.
        """
        sessions = []
        
        for path in self.sessions_dir.glob("*.jsonl"):
            try:
                # Read just the metadata line
                with open(path) as f:
                    first_line = f.readline().strip()
                    if first_line:
                        data = json.loads(first_line)
                        if data.get("_type") == "metadata":
                            sessions.append({
                                "key": path.stem.replace("_", ":"),
                                "created_at": data.get("created_at"),
                                "updated_at": data.get("updated_at"),
                                "path": str(path)
                            })
            except Exception:
                continue
        
        return sorted(sessions, key=lambda x: x.get("updated_at", ""), reverse=True)
    
    async def compact_session(self, key: str) -> tuple[bool, str]:
        """
        Compact a session by summarizing old messages.
        
        Args:
            key: Session key.
        
        Returns:
            Tuple of (success, message).
        """
        if not self.provider:
            return False, "LLM provider not available for compacting."
        
        session = self.get_or_create(key)
        
        if len(session.messages) <= COMPACT_KEEP_RECENT:
            return False, f"Not enough messages to compact (need more than {COMPACT_KEEP_RECENT})."

        # Build summary prompt, capped to avoid exceeding context window
        history = session.get_history(max_messages=0)
        to_summarize = history[:-COMPACT_KEEP_RECENT]
        if len(to_summarize) > COMPACT_MAX_MESSAGES:
            to_summarize = to_summarize[-COMPACT_MAX_MESSAGES:]

        prompt = (
            "Summarize the following conversation concisely, preserving key context, "
            "decisions, and important information. Keep it under 500 words.\n\n"
            "Conversation:\n"
        )
        for msg in to_summarize:
            prompt += f"{msg['role']}: {msg['content']}\n\n"

        # Call LLM to generate summary
        try:
            response = await self.provider.chat(
                messages=[{"role": "user", "content": prompt}],
                model=self.compact_model or self.provider.get_default_model(),
                max_tokens=1000,
            )
            summary = response.content.strip()

            if not summary:
                return False, "Failed to generate summary."

            summarized_count = len(session.messages) - COMPACT_KEEP_RECENT
            session.compact(summary)
            self.save(session)

            return True, f"Summarized {summarized_count} messages, keeping last {COMPACT_KEEP_RECENT}."
        except Exception as e:
            logger.error(f"Error compacting session: {e}")
            return False, f"Error: {e}"
