"""MCP client: connects to MCP servers and wraps their tools as native nanobot tools."""

from contextlib import AsyncExitStack
from typing import Any

from loguru import logger

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry


class MCPServerHandle:
    """Holds connection state for one MCP server, supports reconnection."""

    def __init__(self, name: str, cfg):
        self.name = name
        self.cfg = cfg
        self.session = None
        self._stack: AsyncExitStack | None = None

    async def connect(self) -> None:
        """(Re)connect to the MCP server."""
        await self.close()

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        self._stack = AsyncExitStack()
        await self._stack.__aenter__()

        try:
            if self.cfg.command:
                params = StdioServerParameters(
                    command=self.cfg.command, args=self.cfg.args, env=self.cfg.env or None
                )
                read, write = await self._stack.enter_async_context(stdio_client(params))
            elif self.cfg.url:
                import httpx
                from mcp.client.streamable_http import streamable_http_client
                # trust_env=False prevents httpx from reading macOS system
                # proxy settings (via urllib.request.getproxies), which would
                # route localhost traffic through Clash/Surge and break MCP.
                # Use MCP-recommended timeouts (30s connect, 300s read for long tool calls).
                http_client = httpx.AsyncClient(
                    trust_env=False,
                    timeout=httpx.Timeout(30.0, read=300.0),
                )
                read, write, _ = await self._stack.enter_async_context(
                    streamable_http_client(self.cfg.url, http_client=http_client)
                )
            else:
                raise ValueError("no command or url configured")

            self.session = await self._stack.enter_async_context(ClientSession(read, write))
            await self.session.initialize()
            logger.info(f"MCP server '{self.name}': connected")
        except Exception:
            await self.close()
            raise

    async def close(self) -> None:
        self.session = None
        if self._stack:
            try:
                await self._stack.aclose()
            except Exception:
                pass
            self._stack = None


class MCPToolWrapper(Tool):
    """Wraps a single MCP server tool as a nanobot Tool."""

    def __init__(self, handle: MCPServerHandle, server_name: str, tool_def):
        self._handle = handle
        self._original_name = tool_def.name
        self._name = f"mcp_{server_name}_{tool_def.name}"
        self._description = tool_def.description or tool_def.name
        self._parameters = tool_def.inputSchema or {"type": "object", "properties": {}}

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    async def execute(self, **kwargs: Any) -> str:
        from mcp import types

        # Try call; on failure, reconnect once and retry
        for attempt in range(2):
            try:
                if not self._handle.session:
                    await self._handle.connect()
                result = await self._handle.session.call_tool(self._original_name, arguments=kwargs)
                parts = []
                for block in result.content:
                    if isinstance(block, types.TextContent):
                        parts.append(block.text)
                    else:
                        parts.append(str(block))
                return "\n".join(parts) or "(no output)"
            except Exception as e:
                if attempt == 0:
                    logger.warning(f"MCP tool '{self._name}' failed, reconnecting: {e}")
                    try:
                        await self._handle.connect()
                    except Exception as re_err:
                        logger.error(f"MCP reconnect failed for '{self._handle.name}': {re_err}")
                        return f"Error: MCP server '{self._handle.name}' unreachable ({re_err})"
                else:
                    logger.error(f"MCP tool '{self._name}' failed after reconnect: {e}")
                    return f"Error: MCP tool call failed ({type(e).__name__}: {e})"
        return "Error: unexpected state"


async def connect_mcp_servers(
    mcp_servers: dict, registry: ToolRegistry, stack: AsyncExitStack | None = None
) -> list[MCPServerHandle]:
    """Connect to configured MCP servers and register their tools.

    Returns list of MCPServerHandle for lifecycle management (close_mcp).
    """
    handles: list[MCPServerHandle] = []
    for name, cfg in mcp_servers.items():
        try:
            handle = MCPServerHandle(name, cfg)
            await handle.connect()

            tools = await handle.session.list_tools()
            for tool_def in tools.tools:
                wrapper = MCPToolWrapper(handle, name, tool_def)
                registry.register(wrapper)
                logger.debug(f"MCP: registered tool '{wrapper.name}' from server '{name}'")

            handles.append(handle)
            logger.info(f"MCP server '{name}': connected, {len(tools.tools)} tools registered")
        except Exception as e:
            logger.error(f"MCP server '{name}': failed to connect: {e}")
    return handles
