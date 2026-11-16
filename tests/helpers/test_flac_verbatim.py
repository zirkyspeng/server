"""Tests for the exact-length verbatim FLAC sent to seekable (Marantz) renderers."""

from __future__ import annotations

import asyncio
import os
import shutil
from collections.abc import AsyncGenerator

import pytest
from music_assistant_models.enums import ContentType
from music_assistant_models.media_items import AudioFormat

from music_assistant.helpers import flac_verbatim

FMT_24 = AudioFormat(
    content_type=ContentType.PCM_S24LE, sample_rate=44100, bit_depth=24, channels=2
)
FMT_16 = AudioFormat(
    content_type=ContentType.PCM_S16LE, sample_rate=48000, bit_depth=16, channels=2
)


async def _source(data: bytes, chunk_size: int = 10007) -> AsyncGenerator[bytes]:
    for offset in range(0, len(data), chunk_size):
        yield data[offset : offset + chunk_size]


async def _collect(gen: AsyncGenerator[bytes]) -> bytes:
    return b"".join([chunk async for chunk in gen])


def _pcm(audio_format: AudioFormat, frames: int) -> bytes:
    return os.urandom(frames * audio_format.channels * audio_format.bit_depth // 8)


@pytest.mark.parametrize("audio_format", [FMT_24, FMT_16])
async def test_object_has_announced_length(audio_format: AudioFormat) -> None:
    """The produced object is exactly as long as announced, whatever the source does."""
    frames = audio_format.sample_rate * 3 + 123
    full = _pcm(audio_format, frames)
    for pcm in (full, full[: len(full) // 2], full + full[:5000]):
        body = await _collect(flac_verbatim.exact_flac_stream(_source(pcm), audio_format, frames))
        assert len(body) == flac_verbatim.object_length(audio_format, frames)


def test_header_announces_total_samples() -> None:
    """STREAMINFO carries the sample count, which is what the renderer shows as duration."""
    header = flac_verbatim.create_header(FMT_24, 1234567)
    assert header[:4] == b"fLaC"
    assert len(header) == flac_verbatim.FLAC_HEADER_SIZE
    packed = int.from_bytes(header[18:26], "big")
    assert packed >> 44 == 44100
    assert packed & ((1 << 36) - 1) == 1234567


def test_frame_offsets_follow_frame_number_length() -> None:
    """Frame offsets account for the longer frame numbers past frame 127."""
    one = flac_verbatim._frame_size(0, flac_verbatim.FLAC_BLOCK_SIZE, FMT_24)
    two = flac_verbatim._frame_size(128, flac_verbatim.FLAC_BLOCK_SIZE, FMT_24)
    assert two == one + 1
    assert flac_verbatim.frame_offset(200, FMT_24) == (
        flac_verbatim.FLAC_HEADER_SIZE + 128 * one + 72 * two
    )
    offset = flac_verbatim.frame_offset(150, FMT_24) + 10
    assert flac_verbatim.frame_at_offset(offset, FMT_24, 10**7) == 150


async def test_byte_range_matches_full_object() -> None:
    """Frames produced from a later frame on are the bytes of the full object there."""
    frames = 44100 * 20
    pcm = _pcm(FMT_24, frames)
    full = await _collect(flac_verbatim.exact_flac_stream(_source(pcm), FMT_24, frames))
    start = flac_verbatim.frame_offset(150, FMT_24) + 17
    first_frame = flac_verbatim.frame_at_offset(start, FMT_24, frames)
    block_bytes = flac_verbatim.FLAC_BLOCK_SIZE * 6
    part = await _collect(
        flac_verbatim.exact_flac_stream(
            _source(pcm[first_frame * block_bytes :]),
            FMT_24,
            frames,
            first_frame=first_frame,
            skip_bytes=start - flac_verbatim.frame_offset(first_frame, FMT_24),
            include_header=False,
        )
    )
    assert part == full[start:]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
@pytest.mark.parametrize("audio_format", [FMT_24, FMT_16])
async def test_ffmpeg_decodes_the_original_pcm(audio_format: AudioFormat) -> None:
    """A standard decoder gets back the exact PCM that was packed."""
    frames = audio_format.sample_rate * 2 + 999
    pcm = _pcm(audio_format, frames)
    body = await _collect(flac_verbatim.exact_flac_stream(_source(pcm), audio_format, frames))
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-v",
        "error",
        "-i",
        "-",
        "-f",
        f"s{audio_format.bit_depth}le",
        "-",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    decoded, errors = await proc.communicate(body)
    assert errors == b""
    assert decoded == pcm
