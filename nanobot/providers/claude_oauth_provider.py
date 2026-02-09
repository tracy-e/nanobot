"""Claude OAuth provider — calls Anthropic Messages API with OAuth token."""

import logging
from typing import Any

import httpx

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from nanobot.providers.claude_oauth_auth import get_claude_oauth_token

logger = logging.getLogger(__name__)


class ClaudeOAuthProvider(LLMProvider):
    """LLM provider that uses Claude Code's OAuth token to call the Anthropic API.

    Reads the OAuth token from macOS Keychain (stored by Claude Code),
    converts between OpenAI-style messages/tools and Anthropic format,
    and calls the Messages API directly.
    """

    API_URL = "https://api.anthropic.com/v1/messages"
    REQUIRED_SYSTEM_PREFIX = (
        "You are Claude Code, Anthropic's official CLI for Claude."
    )
    BETA_HEADER = (
        "claude-code-20250219,oauth-2025-04-20,interleaved-thinking-2025-05-14"
    )
    API_VERSION = "2023-06-01"

    # Tool names that the OAuth API rejects — map to CamelCase alternatives.
    _TOOL_NAME_MAP: dict[str, str] = {
        "read_file": "ReadFile",
        "write_file": "WriteFile",
        "list_dir": "ListDir",
        "exec": "Execute",
    }
    _TOOL_NAME_RMAP: dict[str, str] = {v: k for k, v in _TOOL_NAME_MAP.items()}

    def __init__(self, model: str = "claude-sonnet-4-5-20250929"):
        super().__init__()
        self.model = model
        self._token: str | None = None
        self._client = httpx.AsyncClient(timeout=300.0)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        token = self._get_token()
        system_blocks, anthropic_messages = self._convert_messages(messages)
        anthropic_tools = self._convert_tools(tools) if tools else None

        # Strip provider prefix (e.g. "claude-oauth/claude-sonnet-..." → "claude-sonnet-...")
        resolved_model = model or self.model
        if "/" in resolved_model:
            resolved_model = resolved_model.split("/", 1)[1]

        body: dict[str, Any] = {
            "model": resolved_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_blocks,
            "messages": anthropic_messages,
        }
        if anthropic_tools:
            body["tools"] = anthropic_tools

        headers = {
            "Authorization": f"Bearer {token}",
            "anthropic-version": self.API_VERSION,
            "anthropic-beta": self.BETA_HEADER,
            "content-type": "application/json",
        }

        resp = await self._client.post(self.API_URL, json=body, headers=headers)

        # 401 → refresh token once and retry
        if resp.status_code == 401:
            logger.info("OAuth token expired, refreshing…")
            self._token = None
            token = self._get_token()
            headers["Authorization"] = f"Bearer {token}"
            resp = await self._client.post(self.API_URL, json=body, headers=headers)

        resp.raise_for_status()
        return self._parse_response(resp.json())

    def get_default_model(self) -> str:
        return f"claude-oauth/{self.model}"

    # ------------------------------------------------------------------
    # Token management
    # ------------------------------------------------------------------

    def _get_token(self) -> str:
        if not self._token:
            self._token = get_claude_oauth_token()
        if not self._token:
            raise RuntimeError(
                "No Claude OAuth token found. "
                "Make sure Claude Code is installed and you have logged in."
            )
        return self._token

    # ------------------------------------------------------------------
    # Format conversion: OpenAI → Anthropic
    # ------------------------------------------------------------------

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI tool definitions to Anthropic format."""
        result = []
        for tool in tools:
            fn = tool.get("function", tool)
            name = self._map_tool_name(fn["name"])
            result.append({
                "name": name,
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object"}),
            })
        return result

    def _convert_messages(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Convert OpenAI messages to (system_blocks, anthropic_messages).

        - Extracts role=system messages into content block array.
        - The Claude Code system prefix is always the first block.
        - Converts assistant tool_calls to content blocks with type=tool_use.
        - Converts role=tool messages to user messages with type=tool_result.
        """
        system_parts: list[str] = []
        anthropic_msgs: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "")

            if role == "system":
                system_parts.append(msg.get("content", ""))
                continue

            if role == "user":
                anthropic_msgs.append({
                    "role": "user",
                    "content": self._normalize_content(msg.get("content", "")),
                })
                continue

            if role == "assistant":
                content_blocks: list[dict[str, Any]] = []
                text = msg.get("content")
                if text:
                    content_blocks.append({"type": "text", "text": text})
                for tc in msg.get("tool_calls", []):
                    fn = tc.get("function", {})
                    arguments = fn.get("arguments", {})
                    # arguments may be a JSON string
                    if isinstance(arguments, str):
                        import json
                        try:
                            arguments = json.loads(arguments)
                        except (json.JSONDecodeError, ValueError):
                            arguments = {"raw": arguments}
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": self._map_tool_name(fn.get("name", "")),
                        "input": arguments,
                    })
                if content_blocks:
                    anthropic_msgs.append({
                        "role": "assistant",
                        "content": content_blocks,
                    })
                continue

            if role == "tool":
                # Anthropic expects tool_result inside a user message
                tool_result_block = {
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": msg.get("content", ""),
                }
                # Merge into previous user message if possible
                if (
                    anthropic_msgs
                    and anthropic_msgs[-1]["role"] == "user"
                    and isinstance(anthropic_msgs[-1]["content"], list)
                ):
                    anthropic_msgs[-1]["content"].append(tool_result_block)
                else:
                    anthropic_msgs.append({
                        "role": "user",
                        "content": [tool_result_block],
                    })
                continue

        # Build system as array of content blocks.
        # The Claude Code prefix must be the first block (OAuth requirement).
        system_blocks: list[dict[str, Any]] = [
            {"type": "text", "text": self.REQUIRED_SYSTEM_PREFIX}
        ]
        for part in system_parts:
            if part and not part.startswith(self.REQUIRED_SYSTEM_PREFIX):
                system_blocks.append({"type": "text", "text": part})

        # Ensure messages alternate user/assistant (Anthropic requirement).
        # Merge consecutive same-role messages.
        anthropic_msgs = self._merge_consecutive_roles(anthropic_msgs)

        return system_blocks, anthropic_msgs

    @staticmethod
    def _normalize_content(content: Any) -> str | list[dict[str, Any]]:
        """Keep content as-is if it's a list (multimodal), otherwise ensure str."""
        if isinstance(content, list):
            return content
        return str(content) if content else ""

    @staticmethod
    def _merge_consecutive_roles(
        msgs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Merge consecutive messages with the same role."""
        if not msgs:
            return msgs
        merged: list[dict[str, Any]] = [msgs[0]]
        for msg in msgs[1:]:
            if msg["role"] == merged[-1]["role"]:
                # Merge content
                prev = merged[-1]["content"]
                curr = msg["content"]
                if isinstance(prev, str) and isinstance(curr, str):
                    merged[-1]["content"] = f"{prev}\n\n{curr}"
                else:
                    # Convert to list and extend
                    if isinstance(prev, str):
                        prev = [{"type": "text", "text": prev}]
                    if isinstance(curr, str):
                        curr = [{"type": "text", "text": curr}]
                    merged[-1]["content"] = prev + curr
            else:
                merged.append(msg)
        return merged

    # ------------------------------------------------------------------
    # Response parsing: Anthropic → LLMResponse
    # ------------------------------------------------------------------

    def _parse_response(self, data: dict[str, Any]) -> LLMResponse:
        """Parse Anthropic Messages API response into LLMResponse."""
        content_parts: list[str] = []
        tool_calls: list[ToolCallRequest] = []
        reasoning_parts: list[str] = []

        for block in data.get("content", []):
            block_type = block.get("type")
            if block_type == "text":
                content_parts.append(block.get("text", ""))
            elif block_type == "tool_use":
                tool_calls.append(ToolCallRequest(
                    id=block.get("id", ""),
                    name=self._unmap_tool_name(block.get("name", "")),
                    arguments=block.get("input", {}),
                ))
            elif block_type == "thinking":
                reasoning_parts.append(block.get("thinking", ""))

        usage_data = data.get("usage", {})
        usage = {
            "prompt_tokens": usage_data.get("input_tokens", 0),
            "completion_tokens": usage_data.get("output_tokens", 0),
            "total_tokens": (
                usage_data.get("input_tokens", 0)
                + usage_data.get("output_tokens", 0)
            ),
        }

        finish_reason = data.get("stop_reason", "end_turn")
        # Map Anthropic stop reasons to OpenAI equivalents
        reason_map = {
            "end_turn": "stop",
            "tool_use": "tool_calls",
            "max_tokens": "length",
            "stop_sequence": "stop",
        }
        finish_reason = reason_map.get(finish_reason, finish_reason)

        return LLMResponse(
            content="\n\n".join(content_parts) if content_parts else None,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            reasoning_content=(
                "\n\n".join(reasoning_parts) if reasoning_parts else None
            ),
        )

    # ------------------------------------------------------------------
    # Tool name mapping
    # ------------------------------------------------------------------

    def _map_tool_name(self, name: str) -> str:
        """Map nanobot tool name to Anthropic-safe name."""
        return self._TOOL_NAME_MAP.get(name, name)

    def _unmap_tool_name(self, name: str) -> str:
        """Map Anthropic tool name back to nanobot tool name."""
        return self._TOOL_NAME_RMAP.get(name, name)
