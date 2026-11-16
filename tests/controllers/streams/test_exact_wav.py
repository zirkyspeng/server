"""Tests for the exact-length WAV streams sent to seekable (Marantz) renderers."""

from __future__ import annotations

import struct
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from music_assistant_models.enums import ContentType, MediaType
from music_assistant_models.media_items import AudioFormat

from music_assistant.controllers.streams.controller import (
    StreamsController,
    _exact_length_stream,
)
from music_assistant.helpers.audio import WAV_HEADER_SIZE, create_exact_wave_header

FMT = AudioFormat(content_type=ContentType.WAV, sample_rate=44100, bit_depth=24, channels=2)


async def _source(*chunks: bytes) -> AsyncGenerator[bytes]:
    for chunk in chunks:
        yield chunk


async def _collect(gen: AsyncGenerator[bytes]) -> bytes:
    return b"".join([chunk async for chunk in gen])


def test_exact_wave_header_sizes() -> None:
    """The header announces the exact data size of the given frames."""
    header = create_exact_wave_header(FMT, 1000)
    assert len(header) == WAV_HEADER_SIZE
    assert struct.unpack("<L", header[4:8])[0] == 36 + 6000
    assert struct.unpack("<L", header[40:44])[0] == 6000


async def test_short_source_is_padded_with_silence() -> None:
    """A source ending early still fills the announced body."""
    body = await _collect(_exact_length_stream(_source(b"\x01" * 10), b"HDR", 0, 20))
    assert body == b"HDR" + b"\x01" * 10 + bytes(7)


async def test_long_source_is_truncated() -> None:
    """A source running long never exceeds the announced body."""
    body = await _collect(_exact_length_stream(_source(b"\x01" * 10, b"\x02" * 10), b"H", 0, 15))
    assert body == b"H" + b"\x01" * 10 + b"\x02" * 4


async def test_skip_bytes_aligns_byte_range() -> None:
    """Bytes before the requested offset are dropped across chunk boundaries."""
    body = await _collect(_exact_length_stream(_source(b"ab", b"cdef"), b"", 3, 3))
    assert body == b"def"


def _resolver(streamdetails: object, duration: int | None = 200) -> StreamsController:
    controller = StreamsController.__new__(StreamsController)
    player = MagicMock()
    player.supports_http_time_seek = True
    player.flow_mode = False
    player.supports_gapless = False
    player.config.get_value.side_effect = lambda key, default=None: (
        "flac" if key == "output_codec" else default
    )
    queue_item = SimpleNamespace(
        streamdetails=streamdetails,
        duration=duration,
        media_type=MediaType.TRACK,
        extra_attributes={},
    )
    mass = MagicMock()
    mass.players.get_output_protocol.return_value = player
    mass.player_queues.get_item.return_value = queue_item
    mass.player_queues.get.return_value = None
    controller.mass = mass
    controller._needs_flow_mode = MagicMock(return_value=False)  # type: ignore[method-assign]
    controller._server = MagicMock(base_url="http://ma:8097")
    return controller


def _media() -> MagicMock:
    media = MagicMock()
    media.media_type = MediaType.TRACK
    media.queue_session_id = "s"
    media.queue_item_id = "i"
    media.source_id = "q"
    return media


@pytest.mark.parametrize(
    ("streamdetails", "expected"),
    [
        # transcoded source (Spotify PCM): exact-length verbatim FLAC
        (
            SimpleNamespace(
                path="spotify:track:x",
                duration=200,
                audio_format=AudioFormat(content_type=ContentType.PCM_S32LE),
                decryption_key=None,
                fade_in=False,
            ),
            ".flac?exact=1",
        ),
        # original FLAC over HTTP (Tidal): proxied byte for byte as FLAC
        (
            SimpleNamespace(
                path="https://cdn/x.flac",
                duration=200,
                audio_format=AudioFormat(content_type=ContentType.FLAC),
                decryption_key=None,
                fade_in=False,
            ),
            ".flac",
        ),
    ],
)
async def test_resolve_stream_url_codec(
    monkeypatch: pytest.MonkeyPatch, streamdetails: object, expected: str
) -> None:
    """Only transcoded Marantz tracks are switched to an exact-length FLAC."""
    controller = _resolver(streamdetails)
    monkeypatch.setattr(StreamsController, "base_url", "http://ma:8097", raising=False)
    monkeypatch.setattr(
        "music_assistant.controllers.streams.controller.overlay_active", lambda _q: False
    )
    url = await controller.resolve_stream_url("p", _media())
    assert url.endswith(expected)
