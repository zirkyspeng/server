"""Tests for DLNA players that need software-managed track transitions."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Coroutine
from typing import Any, cast
from unittest.mock import ANY, AsyncMock, MagicMock, call

import pytest
from async_upnp_client.profiles.dlna import TransportState
from music_assistant_models.enums import MediaType, PlaybackState, PlayerFeature
from music_assistant_models.player import PlayerMedia

from music_assistant.providers.dlna.player import DLNAPlayer
from tests.common import MockProvider


def _player(manufacturer: str, duration: int = 20) -> DLNAPlayer:
    """Return a ready DLNA player with a current track."""
    provider = MockProvider("dlna", instance_id="dlna_test")
    device = MagicMock()
    device.manufacturer = manufacturer
    device.has_play_media = True
    device.can_stop = True
    device.transport_state = TransportState.PLAYING
    device.async_set_next_transport_uri = AsyncMock()
    device.async_stop = AsyncMock()
    player = DLNAPlayer(
        provider,  # type: ignore[arg-type]
        "uuid:dlna-player",
        "http://192.168.1.10/description.xml",
        device=device,
    )
    player.set_available(True)
    player.set_current_media(
        uri="http://192.168.1.2/current.mp3",
        media_type=MediaType.TRACK,
        duration=duration,
        clear_all=True,
    )
    player._attr_playback_state = PlaybackState.PLAYING
    player._attr_elapsed_time = 18
    player._attr_elapsed_time_last_updated = time.time()
    return player


async def test_play_media_skips_stop_when_renderer_is_already_idle() -> None:
    """An idle renderer must receive the new URI without a blocking Stop first."""
    player = _player("Marantz")
    assert player.device is not None
    player._attr_playback_state = PlaybackState.IDLE
    player.mass.streams.resolve_stream_url = AsyncMock(  # type: ignore[method-assign]
        return_value="http://192.168.1.2/next.flac"
    )
    player._async_call_avt_fresh = AsyncMock()  # type: ignore[method-assign]
    player.device.async_set_transport_uri = AsyncMock()  # type: ignore[method-assign]
    player.device.async_wait_for_can_play = AsyncMock()  # type: ignore[method-assign]
    player.device.async_play = AsyncMock()  # type: ignore[method-assign]
    media = PlayerMedia(
        uri="library://track/next",
        media_type=MediaType.TRACK,
        title="Next track",
        duration=180,
        queue_item_id="queue-item-next",
    )

    await player.play_media(media)

    stop_mock = cast("AsyncMock", player.device.async_stop)
    stop_mock.assert_not_awaited()
    set_uri_mock = cast(  # type: ignore[redundant-cast]
        "AsyncMock", player.device.async_set_transport_uri
    )
    set_uri_mock.assert_not_awaited()
    fresh_set_uri_mock = cast(  # type: ignore[redundant-cast]
        "AsyncMock", player._async_call_avt_fresh
    )
    assert fresh_set_uri_mock.await_args_list == [
        call(
            "SetAVTransportURI",
            InstanceID=0,
            CurrentURI="http://192.168.1.2/next.flac",
            CurrentURIMetaData=ANY,
        ),
        call("Play", InstanceID=0, Speed="1"),
    ]
    play_mock = cast("AsyncMock", player.device.async_play)  # type: ignore[redundant-cast]
    play_mock.assert_not_awaited()


@pytest.mark.parametrize("duration", [20, 0])
async def test_marantz_starts_enqueued_media_after_the_current_track_stops(
    duration: int,
) -> None:
    """Marantz receives a fresh play command so its duration is populated per track."""
    player = _player("Marantz", duration)
    next_media = PlayerMedia(
        uri="library://track/next",
        media_type=MediaType.TRACK,
        duration=180,
    )
    tasks: list[asyncio.Task[Any]] = []

    def _create_task(target: Coroutine[Any, Any, Any], **_kwargs: Any) -> asyncio.Task[Any]:
        task = asyncio.create_task(target)
        tasks.append(task)
        return task

    player.mass.create_task = _create_task  # type: ignore[assignment]
    player.play_media = AsyncMock()  # type: ignore[method-assign]

    await player.enqueue_next_media(next_media)

    player._attr_playback_state = PlaybackState.IDLE
    await asyncio.gather(*tasks)

    player.play_media.assert_awaited_once_with(next_media)
    assert player.device is not None
    set_next_mock = cast("AsyncMock", player.device.async_set_next_transport_uri)
    set_next_mock.assert_not_awaited()


async def test_marantz_software_next_does_not_override_an_explicit_stop() -> None:
    """Stopping playback manually must not start the queued track."""
    player = _player("Marantz")
    next_media = PlayerMedia(uri="library://track/next", media_type=MediaType.TRACK)
    tasks: list[asyncio.Task[Any]] = []

    def _create_task(target: Coroutine[Any, Any, Any], **_kwargs: Any) -> asyncio.Task[Any]:
        task = asyncio.create_task(target)
        tasks.append(task)
        return task

    player.mass.create_task = _create_task  # type: ignore[assignment]
    player.play_media = AsyncMock()  # type: ignore[method-assign]

    await player.enqueue_next_media(next_media)
    await player.stop()
    player._attr_playback_state = PlaybackState.IDLE
    await asyncio.gather(*tasks)

    player.play_media.assert_not_awaited()


def test_marantz_does_not_advertise_gapless_for_software_transitions() -> None:
    """Software-managed Marantz transitions are enqueued but not gapless."""
    player = _player("Marantz")

    player._set_player_features()

    assert PlayerFeature.ENQUEUE in player.supported_features
    assert PlayerFeature.GAPLESS_PLAYBACK not in player.supported_features


def test_other_dlna_players_keep_native_gapless_enqueuing() -> None:
    """Unrelated DLNA renderers retain native enqueue and gapless features."""
    player = _player("Other")

    player._set_player_features()

    assert PlayerFeature.ENQUEUE in player.supported_features
    assert PlayerFeature.GAPLESS_PLAYBACK in player.supported_features
