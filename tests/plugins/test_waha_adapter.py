"""WAHA adapter tests: payload mapping, gating reuse, outbound REST shapes.

The transport is faked at the aiohttp session boundary (``_request`` monkeypatched
for outbound, ``aiohttp.ClientSession`` patched for media download); gating and
formatting run through the REAL shared mixin so the tests pin the contract that
WAHA payloads map onto the same bridge-shaped dict the Baileys adapter uses.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageType
from plugins.platforms.waha.adapter import (
    WahaAdapter,
    _mentioned_ids_from_data,
    _standalone_send,
)


def _make_adapter(**extra_overrides):
    extra = {
        "base_url": "http://127.0.0.1:3000",
        "api_key": "secret-key",
        "session": "test-session",
        "dm_policy": "allowlist",
        "allow_from": ["6281234567890@s.whatsapp.net"],
        "group_policy": "allowlist",
        "group_allow_from": ["120363001234567890@g.us"],
        "require_mention": True,
        "observe_unmentioned_group_messages": True,
    }
    extra.update(extra_overrides)
    adapter = object.__new__(WahaAdapter)
    adapter.platform = Platform("waha")
    adapter.config = PlatformConfig(enabled=True, extra=extra)
    adapter._base_url = str(extra["base_url"]).rstrip("/")
    adapter._api_key = str(extra["api_key"])
    adapter._session = str(extra["session"])
    adapter._webhook_port = int(extra.get("webhook_port", 8655))
    adapter._webhook_secret = str(extra.get("webhook_secret", ""))
    adapter._reply_prefix = None
    adapter._dm_policy = "allowlist"
    adapter._allow_from = WahaAdapter._coerce_allow_list(extra["allow_from"])
    adapter._group_policy = "allowlist"
    adapter._group_allow_from = WahaAdapter._coerce_allow_list(extra["group_allow_from"])
    adapter._mention_patterns = adapter._compile_mention_patterns()
    adapter._send_read_receipts = False
    adapter._bot_ids = {"15551230000@s.whatsapp.net"}
    adapter._http_session = MagicMock()
    adapter._running = True
    adapter._message_handler = AsyncMock()
    return adapter


def _dm_payload(body="hello there", **overrides):
    payload = {
        "id": "false_6281234567890@c.us_ABC",
        "timestamp": 1757000000,
        "from": "6281234567890@c.us",
        "chatId": "6281234567890@c.us",
        "fromMe": False,
        "body": body,
        "hasMedia": False,
        "participant": "6281234567890@c.us",
        "sender": {"pushName": "Farel"},
        "_data": {"message": {"conversation": body}},
    }
    payload.update(overrides)
    return payload


def _group_payload(body="just chatting", **overrides):
    payload = _dm_payload(body, **overrides)
    payload["chatId"] = "120363001234567890@g.us"
    payload["from"] = "120363001234567890@g.us"
    payload["participant"] = "6281234567890@c.us"
    return payload


# ---------------------------------------------------------------------------
# payload mapping
# ---------------------------------------------------------------------------

class TestPayloadMapping:
    def test_dm_mapping(self):
        adapter = _make_adapter()
        data = adapter._map_payload(_dm_payload())
        assert data["chatId"] == "6281234567890@c.us"
        assert data["senderId"] == "6281234567890@c.us"
        assert data["isGroup"] is False
        assert data["body"] == "hello there"
        assert data["botIds"] == ["15551230000@s.whatsapp.net"]

    def test_lid_addressing_resolved_to_phone_jid(self):
        """WhatsApp LID-addressed DMs (addressingMode: lid) must surface the phone JID
        from _data.key.remoteJidAlt, or phone-form allowlists silently reject them."""
        adapter = _make_adapter()
        payload = _dm_payload(
            id="false_231580811403454@lid_ABC",
            **{"_data": {"key": {"remoteJid": "231580811403454@lid",
                                 "remoteJidAlt": "6281234567890@s.whatsapp.net",
                                 "fromMe": False, "id": "ABC", "participant": "",
                                 "addressingMode": "lid"}}},
        )
        payload["from"] = "231580811403454@lid"
        payload["chatId"] = "231580811403454@lid"
        payload["participant"] = "231580811403454@lid"
        data = adapter._map_payload(payload)
        assert data["chatId"] == "6281234567890@s.whatsapp.net"
        assert data["senderId"] == "6281234567890@s.whatsapp.net"

    def test_lid_without_alt_kept_as_is(self):
        adapter = _make_adapter()
        payload = _dm_payload()
        payload["from"] = "231580811403454@lid"
        payload["chatId"] = "231580811403454@lid"
        data = adapter._map_payload(payload)
        assert data["chatId"] == "231580811403454@lid"

    def test_group_mapping_uses_participant_as_sender(self):
        adapter = _make_adapter()
        data = adapter._map_payload(_group_payload())
        assert data["isGroup"] is True
        assert data["senderId"] == "6281234567890@c.us"  # participant, not the group JID

    def test_mentioned_ids_extracted_from_engine_data(self):
        payload = _group_payload()
        payload["_data"] = {"message": {
            "extendedTextMessage": {
                "text": "@15551230000 hi",
                "contextInfo": {"mentionedJid": ["15551230000@s.whatsapp.net"]},
            }}}
        assert _mentioned_ids_from_data(payload) == ["15551230000@s.whatsapp.net"]

    def test_reply_to_mapping(self):
        payload = _dm_payload(replyTo={
            "id": "false_15551230000@c.us_XYZ",
            "participant": "15551230000@c.us",
            "body": "earlier message",
        })
        data = _make_adapter()._map_payload(payload)
        assert data["hasQuotedMessage"] is True
        assert data["quotedMessageId"] == "false_15551230000@c.us_XYZ"
        assert data["quotedParticipant"] == "15551230000@c.us"
        assert data["quotedText"] == "earlier message"


# ---------------------------------------------------------------------------
# gating through the REAL shared mixin
# ---------------------------------------------------------------------------

class TestGating:
    def test_allowed_dm_dispatches(self):
        adapter = _make_adapter()
        event = asyncio.run(adapter._build_message_event(_dm_payload()))
        assert event is not None
        assert event.message_type == MessageType.TEXT
        assert event.source.chat_id == "6281234567890@c.us"

    def test_unknown_dm_dropped(self):
        adapter = _make_adapter()
        payload = _dm_payload()
        payload["from"] = payload["chatId"] = "6289999999999@c.us"
        payload["participant"] = "6289999999999@c.us"
        assert asyncio.run(adapter._build_message_event(payload)) is None

    def test_unmentioned_group_message_observed_not_dispatched(self):
        adapter = _make_adapter()
        adapter._session_store = MagicMock()
        session_entry = MagicMock()
        session_entry.session_id = "sess1"
        adapter._session_store.get_or_create_session.return_value = session_entry
        event = asyncio.run(adapter._build_message_event(_group_payload()))
        assert event is None  # not dispatched
        adapter._session_store.append_to_transcript.assert_called_once()
        entry = adapter._session_store.append_to_transcript.call_args.args[1]
        assert entry["observed"] is True

    def test_observed_media_downloaded_and_referenced(self):
        """Observed media is downloaded to the cache and recorded as an inspectable
        reference (no placeholders) — same contract as the Baileys bridge adapter."""
        adapter = _make_adapter()
        adapter._session_store = MagicMock()
        session_entry = MagicMock()
        session_entry.session_id = "sess1"
        adapter._session_store.get_or_create_session.return_value = session_entry

        payload = _group_payload(body="look", hasMedia=True,
                                 media={"url": "http://127.0.0.1:3000/api/files/abc.jpg",
                                        "mimetype": "image/jpeg"})

        class _Resp:
            status = 200
            async def read(self):
                return b"jpeg-bytes"
            async def __aenter__(self):
                return self
            async def __aexit__(self, *exc):
                return False

        session = MagicMock()
        session.get = MagicMock(return_value=_Resp())
        adapter._http_session = session

        asyncio.run(adapter._build_message_event(payload))
        entry = adapter._session_store.append_to_transcript.call_args.args[1]
        assert entry["observed"] is True
        assert "[image: " in entry["content"]
        assert "[If you need a closer look, use vision_analyze with image_url: " in entry["content"]
        assert "look" in entry["content"]

    def test_mentioned_group_message_dispatches(self):
        adapter = _make_adapter()
        payload = _group_payload(body="@15551230000 what's up")
        payload["_data"] = {"message": {"extendedTextMessage": {
            "text": "@15551230000 what's up",
            "contextInfo": {"mentionedJid": ["15551230000@s.whatsapp.net"]}}}}
        event = asyncio.run(adapter._build_message_event(payload))
        assert event is not None

    def test_from_me_skipped_at_webhook_layer(self):
        """fromMe filtering happens in _handle_webhook (bot mode: own messages are echoes)."""
        adapter = _make_adapter(webhook_secret="")
        request = MagicMock()

        async def _json():
            return {"event": "message", "payload": _dm_payload(fromMe=True)}
        request.json = AsyncMock(side_effect=_json)
        request.headers = {}
        asyncio.run(adapter._handle_webhook(request))
        adapter._message_handler.assert_not_called()


# ---------------------------------------------------------------------------
# outbound REST shapes
# ---------------------------------------------------------------------------

class TestOutbound:
    def _capture_adapter(self):
        adapter = _make_adapter()
        captured = []

        async def _fake_request(method, path, payload=None, timeout=30):
            captured.append((method, path, payload))
            return 200, {"key": {"remoteJid": "6281234567890@s.whatsapp.net",
                                 "fromMe": True, "id": f"wamid.out{len(captured)}"}}

        adapter._request = _fake_request
        return adapter, captured

    def test_send_formats_and_posts(self):
        adapter, captured = self._capture_adapter()
        result = asyncio.run(adapter.send("6281234567890@c.us", "# Big\n\nBody **bold**."))
        assert result.success
        method, path, payload = captured[0]
        assert method == "POST" and path == "/api/sendText"
        assert payload["session"] == "test-session"
        assert payload["chatId"] == "6281234567890@c.us"
        assert payload["text"] == "𝐁𝐢𝐠\n\nBody *bold*."

    def test_send_chunks_over_the_cap(self):
        adapter, captured = self._capture_adapter()
        result = asyncio.run(adapter.send("6281234567890@c.us", "a " * 40000))
        assert result.success
        assert len(captured) >= 2
        assert result.continuation_message_ids  # surfaced like the bridge adapter

    def test_edit_uses_chat_message_endpoint(self):
        adapter, captured = self._capture_adapter()
        result = asyncio.run(adapter.edit_message(
            "6281234567890@c.us", "wamid.in1", "## Edited"))
        assert result.success
        method, path, payload = captured[0]
        assert method == "PUT"
        # bare ids are serialized as own-message edits (WAHA rejects bare ids, HTTP 500)
        assert path == ("/api/test-session/chats/6281234567890@c.us/messages/"
                        "true_6281234567890@c.us_wamid.in1")
        assert payload["text"] == "𝑬𝒅𝒊𝒕𝒆𝒅"  # "## Edited" → script-bold H2

    def test_send_returns_serialized_message_id(self):
        """sendText ids come from body.key (NOWEB shape) serialized Baileys-style so the
        stream consumer can edit instead of falling back to fresh sends."""
        adapter, captured = self._capture_adapter()
        result = asyncio.run(adapter.send("6281234567890@c.us", "hello"))
        assert result.success
        assert result.message_id == "true_6281234567890@s.whatsapp.net_wamid.out1"

    def test_edit_accepts_serialized_id_unchanged(self):
        adapter, captured = self._capture_adapter()
        result = asyncio.run(adapter.edit_message(
            "6281234567890@c.us", "true_6281234567890@s.whatsapp.net_wamid.out1", "edited"))
        assert result.success
        method, path, _ = captured[0]
        assert path == ("/api/test-session/chats/6281234567890@c.us/messages/"
                        "true_6281234567890@s.whatsapp.net_wamid.out1")

    def test_media_local_file_sent_as_base64(self, tmp_path):
        adapter, captured = self._capture_adapter()
        img = tmp_path / "pic.png"
        img.write_bytes(b"\x89PNG fake")
        result = asyncio.run(adapter.send_image_file("6281234567890@c.us", str(img), caption="# Cap"))
        assert result.success
        method, path, payload = captured[0]
        assert path == "/api/sendImage"
        assert payload["file"]["mimetype"] == "image/png"
        assert payload["file"]["data"]
        assert payload["caption"] == "𝐂𝐚𝐩"

    def test_media_remote_url_passthrough(self):
        adapter, captured = self._capture_adapter()
        result = asyncio.run(adapter.send_image("6281234567890@c.us", "https://cdn.example.com/x.jpg"))
        assert result.success
        _, _, payload = captured[0]
        assert payload["file"] == {"url": "https://cdn.example.com/x.jpg"}

    def test_typing_and_receipts(self):
        adapter, captured = self._capture_adapter()
        asyncio.run(adapter.send_typing("6281234567890@c.us"))
        paths = [p for _, p, _ in captured]
        assert "/api/startTyping" in paths
        # Read receipts default OFF (privacy) — set the flag explicitly.
        adapter._send_read_receipts = True
        asyncio.run(adapter._send_read_receipt("6281234567890@c.us", "wamid.in1"))
        paths = [p for _, p, _ in captured]
        assert "/api/sendSeen" in paths

    def test_read_receipts_default_off(self):
        adapter, captured = self._capture_adapter()
        asyncio.run(adapter._send_read_receipt("6281234567890@c.us", "wamid.in1"))
        assert captured == []


# ---------------------------------------------------------------------------
# webhook receiver auth
# ---------------------------------------------------------------------------

class TestWebhookAuth:
    def _make_request(self, headers):
        request = MagicMock()

        async def _json():
            return {"event": "message", "payload": {}}
        request.json = AsyncMock(side_effect=_json)
        request.headers = headers
        return request

    def test_secret_mismatch_rejected(self):
        adapter = _make_adapter(webhook_secret="tok-123")
        response = asyncio.run(adapter._handle_webhook(self._make_request({"X-Hermes-Token": "wrong"})))
        assert response.status == 401

    def test_secret_match_accepted(self):
        adapter = _make_adapter(webhook_secret="tok-123")
        response = asyncio.run(adapter._handle_webhook(self._make_request({"X-Hermes-Token": "tok-123"})))
        assert response.status == 200

    def test_non_message_events_acked(self):
        adapter = _make_adapter(webhook_secret="tok-123")
        request = MagicMock()

        async def _json():
            return {"event": "message.ack", "payload": {}}
        request.json = AsyncMock(side_effect=_json)
        request.headers = {"X-Hermes-Token": "tok-123"}
        response = asyncio.run(adapter._handle_webhook(request))
        assert response.status == 200


# ---------------------------------------------------------------------------
# standalone cron delivery
# ---------------------------------------------------------------------------

class TestStandaloneSend:
    def test_standalone_text_formatted(self, monkeypatch):
        monkeypatch.setenv("WAHA_BASE_URL", "http://127.0.0.1:3000")
        monkeypatch.setenv("WAHA_SESSION", "cron-session")
        session_ctx, calls = _fake_aiohttp()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("aiohttp.ClientSession", lambda *a, **kw: session_ctx)
            res = asyncio.run(_standalone_send(_pconfig(), "6281234567890@c.us", "# Big\n\nBody **bold**."))
        assert res["success"] is True
        assert calls[0][0].endswith("/api/sendText")
        assert calls[0][1]["text"] == "𝐁𝐢𝐠\n\nBody *bold*."
        assert calls[0][1]["session"] == "cron-session"


def _pconfig():
    from gateway.config import PlatformConfig
    return PlatformConfig(enabled=True, extra={"base_url": "http://127.0.0.1:3000", "session": "cron-session"})


def _fake_aiohttp():
    """Minimal async context-manager stand-in for aiohttp.ClientSession."""
    calls = []

    class _Resp:
        status = 200

        async def json(self, content_type=None):
            return {"id": "wamid.cron1"}

        async def text(self):
            return ""

    class _Post:
        def __init__(self, url, json=None, timeout=None):
            self.url = url
            self.payload = json
            calls.append((url, json))

        async def __aenter__(self):
            return _Resp()

        async def __aexit__(self, *exc):
            return False

    class _Session:
        def __init__(self, *a, **kw):
            pass

        def post(self, url, json=None, timeout=None):
            return _Post(url, json, timeout)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    return _Session(), calls
