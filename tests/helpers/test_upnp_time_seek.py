"""Tests for DLNA time-seek parsing and metadata."""

from __future__ import annotations

import pytest
from music_assistant_models.enums import MediaType
from music_assistant_models.player import PlayerMedia

from music_assistant.helpers.upnp import (
    create_didl_metadata,
    format_time_seek_range,
    parse_time_seek_range,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("npt=128.250-", (128.25, None)),
        ("npt=00:02:08.500-00:03:00.000", (128.5, 180.0)),
    ],
)
def test_parse_time_seek_range(value: str, expected: tuple[float, float | None]) -> None:
    """Both DLNA NPT notations resolve to seconds."""
    assert parse_time_seek_range(value, 300) == expected


@pytest.mark.parametrize("value", ["bytes=10-", "npt=300-", "npt=20-10", "npt=nan-"])
def test_reject_invalid_time_seek_range(value: str) -> None:
    """Malformed and out-of-bounds ranges are rejected."""
    with pytest.raises((ValueError, OverflowError)):
        parse_time_seek_range(value, 300)


def test_format_time_seek_range() -> None:
    """Responses contain the effective start, end and full duration."""
    assert format_time_seek_range(128, 299.999, 300) == "npt=128.000-299.999/300.000"


def test_time_seek_metadata_keeps_full_duration() -> None:
    """A resumed stream exposes the absolute timeline to a time-seek renderer."""
    media = PlayerMedia(
        uri="library://track/1",
        media_type=MediaType.TRACK,
        queue_item_id="item-1",
        duration=300,
        stream_duration=60,
    )

    metadata = create_didl_metadata(media, "http://192.168.1.2/item.flac", supports_time_seek=True)

    assert 'duration="00:05:00"' in metadata
    assert "DLNA.ORG_OP=10" in metadata


def test_regular_metadata_keeps_remaining_stream_duration() -> None:
    """Renderers without HTTP time-seek keep the legacy shortened timeline."""
    media = PlayerMedia(
        uri="library://track/1",
        media_type=MediaType.TRACK,
        queue_item_id="item-1",
        duration=300,
        stream_duration=60,
    )

    metadata = create_didl_metadata(media, "http://192.168.1.2/item.flac")

    assert 'duration="00:01:00"' in metadata
    assert "DLNA.ORG_OP=01" in metadata
