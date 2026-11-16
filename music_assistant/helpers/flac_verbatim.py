"""
Exact-length FLAC made of verbatim (uncompressed) frames.

A renderer such as a Marantz/HEOS receiver reads a track's duration from the FLAC
STREAMINFO block, but only shows it when the response announces its length, and it
seeks by byte offset. A compressed FLAC encoded on the fly has no known length and no
predictable byte offsets. Verbatim frames carry the PCM samples as they are, so every
frame has a size known up front: the whole object has an exact length and every byte
offset maps onto one sample position, like a WAV, while the renderer still gets a FLAC
stream it can read the duration of.
"""

from __future__ import annotations

import asyncio
import struct
from collections.abc import AsyncGenerator
from contextlib import aclosing
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from music_assistant_models.media_items import AudioFormat

FLAC_BLOCK_SIZE: Final[int] = 4096
# query marking a stream URL that is served as an exact-length verbatim FLAC
EXACT_FLAC_QUERY: Final[str] = "exact=1"
# "fLaC" marker plus one metadata block header and the 34 byte STREAMINFO body
FLAC_HEADER_SIZE: Final[int] = 4 + 4 + 34
SUPPORTED_BIT_DEPTHS: Final[frozenset[int]] = frozenset({8, 16, 24})

_SAMPLE_RATE_CODES: Final[dict[int, int]] = {
    88200: 1,
    176400: 2,
    192000: 3,
    8000: 4,
    16000: 5,
    22050: 6,
    24000: 7,
    32000: 8,
    44100: 9,
    48000: 10,
    96000: 11,
}
_BIT_DEPTH_CODES: Final[dict[int, int]] = {8: 1, 16: 4, 24: 6}
# frame numbers are UTF-8 coded: the upper bound (exclusive) of each coded length
_FRAME_NUMBER_LENGTHS: Final[tuple[tuple[int, int], ...]] = (
    (0x80, 1),
    (0x800, 2),
    (0x10000, 3),
    (0x200000, 4),
    (0x4000000, 5),
    (0x80000000, 6),
)
# encode this many frames per worker hand-off (about a second of 44.1 kHz audio)
_FRAMES_PER_BATCH: Final[int] = 10


def _crc8_table() -> tuple[int, ...]:
    table = []
    for byte in range(256):
        crc = byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
        table.append(crc)
    return tuple(table)


def _crc16_table() -> tuple[int, ...]:
    table = []
    for byte in range(256):
        crc = byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x8005) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
        table.append(crc)
    return tuple(table)


_CRC8: Final[tuple[int, ...]] = _crc8_table()
_CRC16: Final[tuple[int, ...]] = _crc16_table()


def _crc8(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc = _CRC8[crc ^ byte]
    return crc


def _crc16(data: bytes) -> int:
    crc = 0
    table = _CRC16
    for byte in data:
        crc = ((crc << 8) & 0xFFFF) ^ table[(crc >> 8) ^ byte]
    return crc


def is_supported_format(audio_format: AudioFormat) -> bool:
    """
    Return whether PCM of this format can be packed into verbatim FLAC frames.

    :param audio_format: The PCM format of the audio to pack.
    """
    return audio_format.bit_depth in SUPPORTED_BIT_DEPTHS and 1 <= audio_format.channels <= 8


def _frame_number_length(frame_number: int) -> int:
    for limit, length in _FRAME_NUMBER_LENGTHS:
        if frame_number < limit:
            return length
    raise ValueError(f"FLAC frame number {frame_number} is out of range")


def _encode_frame_number(frame_number: int) -> bytes:
    length = _frame_number_length(frame_number)
    if length == 1:
        return bytes([frame_number])
    tail: list[int] = []
    for _ in range(length - 1):
        tail.insert(0, 0x80 | (frame_number & 0x3F))
        frame_number >>= 6
    lead = ((0xFF << (8 - length)) & 0xFF) | frame_number
    return bytes([lead, *tail])


def _block_size_code(block_size: int) -> tuple[int, bytes]:
    if block_size == FLAC_BLOCK_SIZE:
        return 12, b""
    if block_size <= 256:
        return 6, bytes([block_size - 1])
    return 7, struct.pack(">H", block_size - 1)


def _frame_header(frame_number: int, block_size: int, audio_format: AudioFormat) -> bytes:
    size_code, size_extra = _block_size_code(block_size)
    rate_code = _SAMPLE_RATE_CODES.get(audio_format.sample_rate, 0)
    header = (
        bytes(
            [
                0xFF,
                0xF8,  # sync code, fixed block size
                (size_code << 4) | rate_code,
                ((audio_format.channels - 1) << 4)
                | (_BIT_DEPTH_CODES[audio_format.bit_depth] << 1),
            ]
        )
        + _encode_frame_number(frame_number)
        + size_extra
    )
    return header + bytes([_crc8(header)])


def _frame_size(frame_number: int, block_size: int, audio_format: AudioFormat) -> int:
    """Return the byte size of one verbatim frame."""
    size_extra = len(_block_size_code(block_size)[1])
    header = 4 + _frame_number_length(frame_number) + size_extra + 1
    sample_bytes = audio_format.bit_depth // 8
    return header + audio_format.channels * (1 + block_size * sample_bytes) + 2


def _frame_count(total_frames: int) -> int:
    return -(-total_frames // FLAC_BLOCK_SIZE)


def _full_frame_offset(frame_index: int, audio_format: AudioFormat) -> int:
    """Return the offset, from the end of the header, of a frame preceded by full blocks."""
    offset = 0
    start = 0
    for limit, _length in _FRAME_NUMBER_LENGTHS:
        if frame_index <= start:
            break
        end = min(frame_index, limit)
        # every frame number in this band has the same coded length, so the same size
        offset += (end - start) * _frame_size(start, FLAC_BLOCK_SIZE, audio_format)
        start = end
    return offset


def frame_offset(frame_index: int, audio_format: AudioFormat) -> int:
    """
    Return the byte offset, from the start of the object, at which a frame begins.

    :param frame_index: Index of the frame (every frame but the last holds a full block).
    :param audio_format: The PCM format packed into the frames.
    """
    return FLAC_HEADER_SIZE + _full_frame_offset(frame_index, audio_format)


def object_length(audio_format: AudioFormat, total_frames: int) -> int:
    """
    Return the exact byte length of a verbatim FLAC object holding the given samples.

    :param audio_format: The PCM format packed into the frames.
    :param total_frames: Number of sample frames (samples per channel) in the object.
    """
    if total_frames <= 0:
        return FLAC_HEADER_SIZE
    last = _frame_count(total_frames) - 1
    last_block = total_frames - last * FLAC_BLOCK_SIZE
    return frame_offset(last, audio_format) + _frame_size(last, last_block, audio_format)


def frame_at_offset(byte_offset: int, audio_format: AudioFormat, total_frames: int) -> int:
    """
    Return the index of the frame that holds the given byte offset.

    An offset inside the header maps onto the first frame.

    :param byte_offset: Offset from the start of the object.
    :param audio_format: The PCM format packed into the frames.
    :param total_frames: Number of sample frames in the object.
    """
    low, high = 0, max(_frame_count(total_frames) - 1, 0)
    while low < high:
        middle = (low + high + 1) // 2
        if frame_offset(middle, audio_format) <= byte_offset:
            low = middle
        else:
            high = middle - 1
    return low


def create_header(audio_format: AudioFormat, total_frames: int) -> bytes:
    """
    Return the FLAC marker and STREAMINFO block for an object of the given length.

    :param audio_format: The PCM format packed into the frames.
    :param total_frames: Number of sample frames in the object.
    """
    last_frame = max(_frame_count(total_frames) - 1, 0)
    max_frame_size = _frame_size(last_frame, FLAC_BLOCK_SIZE, audio_format)
    info = (
        struct.pack(">HH", FLAC_BLOCK_SIZE, FLAC_BLOCK_SIZE)
        + (0).to_bytes(3, "big")  # minimum frame size: unknown
        + max_frame_size.to_bytes(3, "big")
        + (
            (audio_format.sample_rate << 44)
            | ((audio_format.channels - 1) << 41)
            | ((audio_format.bit_depth - 1) << 36)
            | total_frames
        ).to_bytes(8, "big")
        + bytes(16)  # MD5 of the audio: unknown up front
    )
    # last-metadata-block flag set on the STREAMINFO block (type 0)
    return b"fLaC" + bytes([0x80]) + len(info).to_bytes(3, "big") + info


def _pcm_to_signed_big_endian(pcm: bytes, sample_bytes: int) -> bytes:
    """Turn little-endian PCM into the big-endian samples a verbatim subframe holds."""
    if sample_bytes == 1:
        # 8 bit PCM (WAV) is unsigned, FLAC samples are signed
        return bytes((byte - 128) & 0xFF for byte in pcm)
    out = bytearray(len(pcm))
    for index in range(sample_bytes):
        out[index::sample_bytes] = pcm[sample_bytes - 1 - index :: sample_bytes]
    return bytes(out)


def encode_frame(frame_number: int, pcm: bytes, audio_format: AudioFormat) -> bytes:
    """
    Encode one block of little-endian interleaved PCM as a verbatim FLAC frame.

    :param frame_number: Number of the frame within the object.
    :param pcm: Little-endian interleaved PCM of at most one block.
    :param audio_format: The format of the PCM.
    """
    sample_bytes = audio_format.bit_depth // 8
    channels = audio_format.channels
    block_align = sample_bytes * channels
    block_size = len(pcm) // block_align
    samples = _pcm_to_signed_big_endian(pcm[: block_size * block_align], sample_bytes)
    subframes = []
    for channel in range(channels):
        channel_bytes = bytearray(block_size * sample_bytes)
        for index in range(sample_bytes):
            channel_bytes[index::sample_bytes] = samples[
                channel * sample_bytes + index :: block_align
            ]
        # subframe header: verbatim type, no wasted bits
        subframes.append(b"\x02" + bytes(channel_bytes))
    frame = _frame_header(frame_number, block_size, audio_format) + b"".join(subframes)
    return frame + struct.pack(">H", _crc16(frame))


def _encode_frames(first_frame_number: int, pcm: bytes, audio_format: AudioFormat) -> bytes:
    block_bytes = FLAC_BLOCK_SIZE * audio_format.channels * (audio_format.bit_depth // 8)
    return b"".join(
        encode_frame(first_frame_number + index, pcm[offset : offset + block_bytes], audio_format)
        for index, offset in enumerate(range(0, len(pcm), block_bytes))
    )


async def exact_flac_stream(
    pcm_input: AsyncGenerator[bytes],
    audio_format: AudioFormat,
    total_frames: int,
    first_frame: int = 0,
    skip_bytes: int = 0,
    include_header: bool = True,
) -> AsyncGenerator[bytes]:
    """
    Yield a verbatim FLAC object of an exact length from a PCM stream.

    A source that ends early is padded with digital silence and one that runs long is
    cut, so the renderer always receives the whole object the response announced.

    :param pcm_input: Little-endian interleaved PCM, starting at the first sample of
        ``first_frame``.
    :param audio_format: The format of the PCM.
    :param total_frames: Number of sample frames in the whole object.
    :param first_frame: Index of the first FLAC frame to produce (a byte range seek).
    :param skip_bytes: Bytes of the produced output to drop before the first one sent.
    :param include_header: Whether to start with the FLAC marker and STREAMINFO.
    """
    block_align = audio_format.channels * (audio_format.bit_depth // 8)
    block_bytes = FLAC_BLOCK_SIZE * block_align
    frame_number = first_frame
    remaining_frames = max(total_frames - first_frame * FLAC_BLOCK_SIZE, 0)
    remaining_pcm = remaining_frames * block_align
    batch_bytes = block_bytes * _FRAMES_PER_BATCH
    buffer = bytearray()

    def _emit(data: bytes) -> bytes:
        nonlocal skip_bytes
        if skip_bytes:
            dropped = min(skip_bytes, len(data))
            skip_bytes -= dropped
            data = data[dropped:]
        return data

    async def _flush(final: bool) -> AsyncGenerator[bytes]:
        nonlocal frame_number, buffer
        usable = len(buffer) if final else (len(buffer) // batch_bytes) * batch_bytes
        if not usable:
            return
        pcm = bytes(buffer[:usable])
        del buffer[:usable]
        frames = await asyncio.to_thread(_encode_frames, frame_number, pcm, audio_format)
        frame_number += -(-len(pcm) // block_bytes)
        if data := _emit(frames):
            yield data

    if include_header and (header := _emit(create_header(audio_format, total_frames))):
        yield header
    async with aclosing(pcm_input):
        async for chunk in pcm_input:
            if remaining_pcm <= 0:
                break
            pcm_chunk = chunk[:remaining_pcm]
            remaining_pcm -= len(pcm_chunk)
            buffer += pcm_chunk
            async for data in _flush(final=False):
                yield data
    if remaining_pcm > 0:
        buffer += bytes(remaining_pcm)
    async for data in _flush(final=True):
        yield data
