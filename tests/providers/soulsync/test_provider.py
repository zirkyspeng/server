"""Tests for the SoulSync provider."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from music_assistant_models.enums import AlbumType, ExternalID, ImageType, MediaType
from music_assistant_models.media_items import (
    Album,
    Artist,
    ItemMapping,
    MediaItemImage,
    ProviderMapping,
    Track,
)

from music_assistant.helpers.api import APICommandHandler, parse_arguments
from music_assistant.providers.soulsync import (
    CONF_SKIP_LOCAL,
    SoulSyncProvider,
    _describe,
    _track_payload,
)
from music_assistant.providers.soulsync.client import (
    BACKGROUND_REQUESTS_PER_MINUTE,
    SoulSyncClient,
)
from music_assistant.providers.soulsync.matching import (
    LibraryIndex,
    WishlistIndex,
    artists_match,
    normalize,
)


def _mapping(domain: str, item_id: str) -> ProviderMapping:
    return ProviderMapping(
        item_id=item_id, provider_domain=domain, provider_instance=f"{domain}--abc"
    )


def _album(domain: str = "tidal", item_id: str = "100", **kwargs: Any) -> Album:
    return Album(
        item_id=item_id,
        provider=f"{domain}--abc",
        name=kwargs.pop("name", "Abbey Road"),
        artists=[
            Artist(
                item_id="1",
                provider=f"{domain}--abc",
                name="The Beatles",
                provider_mappings={_mapping(domain, "1")},
            )
        ],
        provider_mappings={_mapping(domain, item_id)},
        **kwargs,
    )


def _track(domain: str = "tidal", item_id: str = "200", **kwargs: Any) -> Track:
    return Track(
        item_id=item_id,
        provider=f"{domain}--abc",
        name=kwargs.pop("name", "Come Together"),
        artists=[
            Artist(
                item_id="1",
                provider=f"{domain}--abc",
                name="The Beatles",
                provider_mappings={_mapping(domain, "1")},
            )
        ],
        provider_mappings={_mapping(domain, item_id)},
        duration=259,
        track_number=1,
        disc_number=1,
        **kwargs,
    )


def _library() -> LibraryIndex:
    index = LibraryIndex()
    index.set_artists([{"id": 1, "name": "The Beatles", "spotify_artist_id": "sp-beatles"}])
    index.set_albums(
        [
            {
                "id": 10,
                "artist_id": 1,
                "title": "Abbey Road (Remastered 2019)",
                "track_count": 2,
                "tidal_id": None,
            }
        ]
    )
    index.set_album_tracks(
        10,
        [
            {"id": 1000, "album_id": 10, "artist_id": 1, "title": "Come Together"},
            {"id": 1001, "album_id": 10, "artist_id": 1, "title": "Something", "isrc": "GBX1"},
        ],
    )
    return index


def _create_provider() -> SoulSyncProvider:
    with patch.object(SoulSyncProvider, "__init__", lambda *_a, **_kw: None):
        provider = SoulSyncProvider.__new__(SoulSyncProvider)
    provider.config = MagicMock()
    provider.config.values = {}
    provider.config.get_value = lambda key, default=None: (
        True if key == CONF_SKIP_LOCAL else default
    )
    provider.mass = MagicMock()
    provider.mass.get_provider = MagicMock(return_value=None)
    provider.logger = MagicMock()
    provider._library = _library()
    provider._wanted = WishlistIndex()
    provider._library_ready = MagicMock()
    provider._library_ready.is_set = MagicMock(return_value=True)
    provider._client = MagicMock()
    provider._client.post = AsyncMock(return_value=({"created": True}, 201))
    return provider


def test_normalize_drops_editions_accents_and_punctuation() -> None:
    """Edition decorations, accents and punctuation do not change the comparison key."""
    assert normalize("Abbey Road (Remastered 2019)") == normalize("abbey road")
    assert normalize("Hey Jude - 2015 Remaster") == normalize("Hey Jude")
    assert normalize("Café Del Mar") == normalize("Cafe del Mar!")
    assert normalize("Song (feat. Someone)") == normalize("Song")
    # non-latin scripts are kept rather than collapsing to an empty key
    assert normalize("真夜中のドア") == "真夜中のドア"


def test_artists_match_handles_joined_credits() -> None:
    """A joined credit matches any of its artists, an unrelated artist does not."""
    assert artists_match(["Beyoncé"], "Beyonce, JAY-Z")
    assert artists_match(["The Beatles"], "the beatles")
    assert not artists_match(["Muse"], "The Beatles")
    assert not artists_match([], "The Beatles")


def test_library_index_finds_albums_and_tracks() -> None:
    """Albums and tracks are found by title and artist, or by external id."""
    index = _library()
    assert index.find_albums("Abbey Road", ["The Beatles"], [])[0]["id"] == 10
    assert not index.find_albums("Abbey Road", ["Muse"], [])
    assert index.find_track("Come Together", ["The Beatles"], [])["id"] == 1000
    assert index.find_track("Unknown", ["The Beatles"], [("isrc", "gbx1")])["id"] == 1001
    assert index.find_artist(["Someone"], [("spotify", "sp-beatles")])["id"] == 1
    assert index.local_track_count(index.find_albums("Abbey Road", ["The Beatles"], [])) == 2
    # replacing an album's tracks drops the old ones from the title index
    index.set_album_tracks(10, [])
    assert index.find_track("Come Together", ["The Beatles"], []) is None


def test_wishlist_index_matches_ids_names_and_albums() -> None:
    """Wishlist rows are matched by id, by title and artist, and grouped per album."""
    wanted = WishlistIndex()
    wanted.set_wishlist(
        [
            {
                "track_id": "sp1",
                "track_name": "Because",
                "artist_name": "The Beatles",
                "album_name": "Abbey Road",
            }
        ]
    )
    wanted.set_watchlist([{"artist_name": "Muse", "deezer_artist_id": "705"}])
    assert wanted.has_track(["sp1"], "x", [])
    assert wanted.has_track([], "Because", ["The Beatles"])
    assert not wanted.has_track([], "Because", ["Muse"])
    assert wanted.album_track_count("Abbey Road (2019 Remaster)", ["The Beatles"]) == 1
    assert wanted.is_watched("muse", [])
    assert wanted.is_watched("Other", [("deezer", "705")])


def test_track_payload_uses_metadata_source_ids() -> None:
    """A Spotify mapped track keeps its Spotify ids and source."""
    album = _album(
        "spotify",
        "sp-album",
        year=1969,
        album_type=AlbumType.ALBUM,
    )
    album.metadata.images = [
        MediaItemImage(
            type=ImageType.THUMB,
            path="https://img/abbey.jpg",
            provider="spotify",
            remotely_accessible=True,
        )
    ]
    track = _track("spotify", "sp-track", album=album)
    track.external_ids = {(ExternalID.ISRC, "GBAYE0601690")}
    payload = _track_payload(track, album, "spotify", total_tracks=17)
    assert payload["id"] == "sp-track"
    assert payload["source"] == "spotify"
    assert payload["isrc"] == "GBAYE0601690"
    assert payload["duration_ms"] == 259000
    assert payload["album"]["id"] == "sp-album"
    assert payload["album"]["album_type"] == "album"
    assert payload["album"]["release_date"] == "1969"
    assert payload["album"]["total_tracks"] == 17
    assert payload["album"]["images"][0]["url"] == "https://img/abbey.jpg"
    assert payload["artists"] == [{"name": "The Beatles"}]


def test_track_payload_without_metadata_source_uses_neutral_id() -> None:
    """A Tidal-only track gets an id SoulSync can not mistake for another provider's id."""
    album = ItemMapping(
        media_type=MediaType.ALBUM, item_id="100", provider="tidal--abc", name="Abbey Road"
    )
    payload = _track_payload(_track(version="Remastered 2019"), album, None)
    assert payload["id"] == "ma-tidal--abc-200"
    assert "source" not in payload
    assert payload["name"] == "Come Together (Remastered 2019)"
    assert payload["album"]["id"] == "ma-tidal--abc-100"
    assert "album_type" not in payload["album"]


async def test_status_reports_local_and_wanted_items() -> None:
    """Status reflects the local library, the wishlist and the watchlist."""
    provider = _create_provider()
    provider._wanted.set_wishlist(
        [{"track_id": "x", "track_name": "Octopus's Garden", "artist_name": "The Beatles"}]
    )
    local_track = _describe(_track())
    wanted_track = _describe(_track(name="Octopus's Garden", item_id="201"))
    missing_track = _describe(_track(name="Here Comes the Sun", item_id="202"))
    album = _describe(_album())
    artist = _describe(_album().artists[0])
    status = await provider.get_status([local_track, wanted_track, missing_track, album, artist])
    assert status[local_track["uri"]]["state"] == "local"
    assert status[wanted_track["uri"]]["state"] == "wanted"
    assert status[missing_track["uri"]]["state"] == "none"
    assert status[album["uri"]]["state"] == "local"
    assert status[album["uri"]]["local_tracks"] == 2
    assert status[artist["uri"]]["local"] is True
    assert status[artist["uri"]]["watched"] is False


async def test_status_reports_partial_album_on_detail() -> None:
    """An album page compares the local track count with the album's tracklist."""
    provider = _create_provider()
    provider._album_track_total = AsyncMock(return_value=17)  # type: ignore[method-assign]
    album = _describe(_album(), detail=True)
    status = (await provider.get_status([album]))[album["uri"]]
    assert status["state"] == "partial"
    assert status["local_tracks"] == 2
    assert status["total_tracks"] == 17


async def test_add_album_skips_local_and_wanted_tracks() -> None:
    """Only the missing tracks of an album are sent to the wishlist."""
    provider = _create_provider()
    album = _album("spotify", "sp-album")
    tracks = [
        _track("spotify", "sp1", name="Come Together"),
        _track("spotify", "sp2", name="Because"),
        _track("spotify", "sp3", name="Sun King"),
    ]
    provider._wanted.add_track_ids("sp3")
    provider.mass.music.albums.get = AsyncMock(return_value=album)
    provider.mass.music.albums.tracks = AsyncMock(return_value=tracks)
    result = await provider._add_album(album)
    assert result == {"added": 1, "skipped_local": 1, "already_wanted": 1, "failed": 0}
    provider._client.post.assert_awaited_once()
    path, body = provider._client.post.await_args.args
    assert path == "/wishlist"
    assert body["source_type"] == "album"
    assert body["track_data"]["id"] == "sp2"
    assert body["track_data"]["source"] == "spotify"


@pytest.mark.parametrize(
    ("status", "data", "expected"),
    [(201, {"created": True}, "created"), (200, {"created": False}, "updated")],
)
async def test_post_wishlist_reports_outcome(
    status: int, data: dict[str, Any], expected: str
) -> None:
    """A created row and an updated existing row are told apart."""
    provider = _create_provider()
    provider._client.post = AsyncMock(return_value=(data, status))
    assert await provider._post_wishlist({"id": "sp1", "name": "x"}, "manual") == expected
    assert "sp1" in provider._wanted.track_ids


def test_api_commands_parse_frontend_payloads() -> None:
    """The registered API handlers accept the JSON the frontend sends."""
    provider = _create_provider()
    status_handler = APICommandHandler.parse("soulsync/status", provider.get_status)
    add_handler = APICommandHandler.parse("soulsync/add", provider.add_item)
    descriptor = dict(_describe(_track()))
    parsed = parse_arguments(
        status_handler.signature, status_handler.type_hints, {"items": [descriptor]}
    )
    assert parsed["items"][0]["uri"] == descriptor["uri"]
    parsed = parse_arguments(
        add_handler.signature, add_handler.type_hints, {"uri": "spotify://album/1"}
    )
    assert parsed == {"uri": "spotify://album/1"}


async def test_grid_status_does_not_load_album_tracklists() -> None:
    """Only the item details page pays for the album tracklist lookup."""
    provider = _create_provider()
    provider._album_track_total = AsyncMock(return_value=17)  # type: ignore[method-assign]
    album = _describe(_album())
    album["uri"] = "library://album/5"
    status = (await provider.get_status([album]))[album["uri"]]
    assert status["state"] == "local"
    assert status["total_tracks"] is None
    provider._album_track_total.assert_not_awaited()


async def test_client_keeps_budget_for_user_actions() -> None:
    """Background requests stop at their own budget while user requests still go through."""
    client = SoulSyncClient(MagicMock(), "http://soulsync", "key", 1)
    for _ in range(BACKGROUND_REQUESTS_PER_MINUTE):
        await client._throttle(background=True)
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(client._throttle(background=True), timeout=0.05)
    await asyncio.wait_for(client._throttle(background=False), timeout=0.05)
    assert len(client._request_times) == BACKGROUND_REQUESTS_PER_MINUTE + 1
