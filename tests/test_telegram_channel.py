import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.telegram import TelegramChannel
from nanobot.config.schema import TelegramConfig


class _FakeBot:
    def __init__(self) -> None:
        self.sent_messages: list[dict] = []
        self.get_me_calls = 0

    async def get_me(self):
        self.get_me_calls += 1
        return SimpleNamespace(id=999, username="nanobot_test")

    async def set_my_commands(self, commands) -> None:
        self.commands = commands

    async def send_message(self, **kwargs) -> None:
        self.sent_messages.append(kwargs)

    async def send_chat_action(self, **kwargs) -> None:
        pass

    async def get_file(self, file_id: str):
        """Return a fake file that 'downloads' to a path (for reply-to-media tests)."""
        async def _fake_download(path) -> None:
            pass
        return SimpleNamespace(download_to_drive=_fake_download)


class _FakeUpdater:
    def __init__(self, on_start_polling) -> None:
        self._on_start_polling = on_start_polling

    async def start_polling(self, **kwargs) -> None:
        self._on_start_polling()


class _FakeApp:
    def __init__(self, on_start_polling) -> None:
        self.bot = _FakeBot()
        self.updater = _FakeUpdater(on_start_polling)
        self.handlers = []
        self.error_handlers = []

    def add_error_handler(self, handler) -> None:
        self.error_handlers.append(handler)

    def add_handler(self, handler) -> None:
        self.handlers.append(handler)

    async def initialize(self) -> None:
        pass

    async def start(self) -> None:
        pass


def _make_telegram_update(
    *,
    chat_type: str = "group",
    text: str | None = None,
    caption: str | None = None,
    entities=None,
    caption_entities=None,
    reply_to_message=None,
):
    user = SimpleNamespace(id=12345, username="alice", first_name="Alice")
    message = SimpleNamespace(
        chat=SimpleNamespace(type=chat_type, is_forum=False),
        chat_id=-100123,
        text=text,
        caption=caption,
        entities=entities or [],
        caption_entities=caption_entities or [],
        reply_to_message=reply_to_message,
        photo=None,
        voice=None,
        audio=None,
        document=None,
        media_group_id=None,
        message_thread_id=None,
        message_id=1,
    )
    return SimpleNamespace(message=message, effective_user=user)


def test_is_allowed_accepts_legacy_telegram_id_username_formats() -> None:
    channel = TelegramChannel(TelegramConfig(allow_from=["12345", "alice", "67890|bob"]), MessageBus())

    assert channel.is_allowed("12345|carol") is True
    assert channel.is_allowed("99999|alice") is True
    assert channel.is_allowed("67890|bob") is True


@pytest.mark.asyncio
async def test_group_policy_mention_accepts_caption_mention() -> None:
    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"], group_policy="mention"),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)

    handled = []

    async def capture_handle(**kwargs) -> None:
        handled.append(kwargs)

    channel._handle_message = capture_handle
    channel._start_typing = lambda _chat_id: None

    mention = SimpleNamespace(type="mention", offset=0, length=13)
    await channel._on_message(
        _make_telegram_update(caption="@nanobot_test photo", caption_entities=[mention]),
        None,
    )

    assert len(handled) == 1
    assert handled[0]["content"] == "@nanobot_test photo"


@pytest.mark.asyncio
async def test_group_policy_mention_accepts_reply_to_bot() -> None:
    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"], group_policy="mention"),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)

    handled = []

    async def capture_handle(**kwargs) -> None:
        handled.append(kwargs)

    channel._handle_message = capture_handle
    channel._start_typing = lambda _chat_id: None

    reply = SimpleNamespace(from_user=SimpleNamespace(id=999))
    await channel._on_message(_make_telegram_update(text="reply", reply_to_message=reply), None)

    assert len(handled) == 1


@pytest.mark.asyncio
async def test_group_policy_open_accepts_plain_group_message() -> None:
    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"], group_policy="open"),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)

    handled = []

    async def capture_handle(**kwargs) -> None:
        handled.append(kwargs)

    channel._handle_message = capture_handle
    channel._start_typing = lambda _chat_id: None

    await channel._on_message(_make_telegram_update(text="hello group"), None)

    assert len(handled) == 1
    assert channel._app.bot.get_me_calls == 0


@pytest.mark.asyncio
async def test_on_message_reply_to_media_fallback_when_download_fails() -> None:
    """When reply has media but download fails, no media attached and no reply tag."""
    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"], group_policy="open"),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)
    channel._app.bot.get_file = None
    handled = []
    async def capture_handle(**kwargs) -> None:
        handled.append(kwargs)
    channel._handle_message = capture_handle
    channel._start_typing = lambda _chat_id: None

    reply_with_photo = SimpleNamespace(
        text=None,
        caption=None,
        photo=[SimpleNamespace(file_id="x", mime_type="image/jpeg")],
        document=None,
        voice=None,
        audio=None,
        video=None,
        video_note=None,
        animation=None,
    )
    update = _make_telegram_update(text="what is this?", reply_to_message=reply_with_photo)
    await channel._on_message(update, None)

    assert len(handled) == 1
    assert "what is this?" in handled[0]["content"]
    assert handled[0]["media"] == []


@pytest.mark.asyncio
async def test_forward_command_does_not_inject_reply_context() -> None:
    """Slash commands forwarded via _forward_command must not include reply context."""
    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"], group_policy="open"),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)
    handled = []
    async def capture_handle(**kwargs) -> None:
        handled.append(kwargs)
    channel._handle_message = capture_handle

    reply = SimpleNamespace(text="some old message", message_id=2, from_user=SimpleNamespace(id=1))
    update = _make_telegram_update(text="/new", reply_to_message=reply)
    await channel._forward_command(update, None)

    assert len(handled) == 1
    assert handled[0]["content"] == "/new"
