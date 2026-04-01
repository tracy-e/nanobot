"""Built-in slash command handlers."""

from __future__ import annotations

import asyncio
import os
import sys

from nanobot import __version__
from nanobot.bus.events import OutboundMessage
from nanobot.command.router import CommandContext, CommandRouter
from nanobot.utils.helpers import build_status_content


async def cmd_stop(ctx: CommandContext) -> OutboundMessage:
    """Cancel all active tasks and subagents for the session."""
    loop = ctx.loop
    msg = ctx.msg
    tasks = loop._active_tasks.pop(msg.session_key, [])
    cancelled = sum(1 for t in tasks if not t.done() and t.cancel())
    for t in tasks:
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass
    sub_cancelled = await loop.subagents.cancel_by_session(msg.session_key)
    total = cancelled + sub_cancelled
    content = f"Stopped {total} task(s)." if total else "No active task to stop."
    return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)


async def cmd_restart(ctx: CommandContext) -> OutboundMessage:
    """Restart the process in-place via os.execv."""
    msg = ctx.msg

    async def _do_restart():
        await asyncio.sleep(1)
        os.execv(sys.executable, [sys.executable, "-m", "nanobot"] + sys.argv[1:])

    asyncio.create_task(_do_restart())
    return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content="Restarting...")


async def cmd_status(ctx: CommandContext) -> OutboundMessage:
    """Build an outbound status message for a session."""
    loop = ctx.loop
    session = ctx.session or loop.sessions.get_or_create(ctx.key)
    ctx_est = 0
    try:
        ctx_est, _ = loop.memory_consolidator.estimate_session_prompt_tokens(session)
    except Exception:
        pass
    if ctx_est <= 0:
        ctx_est = loop._last_usage.get("prompt_tokens", 0)
    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content=build_status_content(
            version=__version__, model=loop.model,
            start_time=loop._start_time, last_usage=loop._last_usage,
            context_window_tokens=loop.context_window_tokens,
            session_msg_count=len(session.get_history(max_messages=0)),
            context_tokens_estimate=ctx_est,
        ),
        metadata={"render_as": "text"},
    )


async def cmd_new(ctx: CommandContext) -> OutboundMessage:
    """Start a fresh session."""
    loop = ctx.loop
    session = ctx.session or loop.sessions.get_or_create(ctx.key)
    snapshot = session.messages[session.last_consolidated:]
    session.clear()
    loop.sessions.save(session)
    loop.sessions.invalidate(session.key)
    if snapshot:
        loop._schedule_background(loop.memory_consolidator.archive_messages(snapshot))
    return OutboundMessage(
        channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
        content="New session started.",
    )


async def cmd_clear(ctx: CommandContext) -> OutboundMessage:
    """Clear conversation history without archiving."""
    loop = ctx.loop
    session = ctx.session or loop.sessions.get_or_create(ctx.key)
    msg_count = len(session.messages)
    session.clear()
    loop.sessions.save(session)
    from loguru import logger
    logger.info("Session cleared for {} (cleared {} messages)", ctx.key, msg_count)
    return OutboundMessage(
        channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
        content="Conversation history cleared. Let's start fresh!",
    )


async def cmd_skills(ctx: CommandContext) -> OutboundMessage:
    """List available workspace skills."""
    return OutboundMessage(
        channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
        content=ctx.loop._list_skills(),
    )


async def cmd_mcp(ctx: CommandContext) -> OutboundMessage:
    """List configured MCP servers."""
    return OutboundMessage(
        channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
        content=ctx.loop._list_mcp_servers(),
    )


async def cmd_compact(ctx: CommandContext) -> OutboundMessage:
    """Compact conversation or switch compact model."""
    loop = ctx.loop
    session = ctx.session or loop.sessions.get_or_create(ctx.key)
    arg = ctx.args.strip() if ctx.args else ""
    if arg == "--switch":
        content = loop._format_compact_model_list()
    elif arg:
        content = loop._switch_compact_model(arg)
    else:
        snapshot = session.messages[session.last_consolidated:]
        if snapshot:
            loop._schedule_background(loop.memory_consolidator.archive_messages(snapshot))
        content = "Conversation compacted."
    return OutboundMessage(channel=ctx.msg.channel, chat_id=ctx.msg.chat_id, content=content)


async def cmd_model(ctx: CommandContext) -> OutboundMessage:
    """Show or switch the active model."""
    arg = ctx.args.strip() if ctx.args else ""
    if not arg:
        content = ctx.loop._format_model_list()
    else:
        content = ctx.loop._switch_model(arg)
    return OutboundMessage(channel=ctx.msg.channel, chat_id=ctx.msg.chat_id, content=content)


async def cmd_help(ctx: CommandContext) -> OutboundMessage:
    """Return available slash commands."""
    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content=build_help_text(),
        metadata={"render_as": "text"},
    )


def build_help_text() -> str:
    """Build canonical help text shared across channels."""
    lines = [
        "nanobot commands:",
        "/new — Start a new conversation",
        "/clear — Clear conversation history",
        "/compact — Compact conversation / switch compact model",
        "/skills — List available skills",
        "/model — Show/switch model",
        "/mcp — List configured MCP servers",
        "/status — Show bot status",
        "/stop — Stop the current task",
        "/restart — Restart the bot",
        "/help — Show available commands",
    ]
    return "\n".join(lines)


def register_builtin_commands(router: CommandRouter) -> None:
    """Register the default set of slash commands."""
    router.priority("/stop", cmd_stop)
    router.priority("/restart", cmd_restart)
    router.priority("/status", cmd_status)
    router.exact("/new", cmd_new)
    router.exact("/clear", cmd_clear)
    router.exact("/skills", cmd_skills)
    router.exact("/mcp", cmd_mcp)
    router.exact("/status", cmd_status)
    router.exact("/help", cmd_help)
    router.prefix("/compact ", cmd_compact)
    router.prefix("/model ", cmd_model)
    router.exact("/compact", cmd_compact)
    router.exact("/model", cmd_model)
