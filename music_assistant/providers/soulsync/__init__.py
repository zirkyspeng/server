"""
SoulSync plugin provider for Music Assistant.

Adds tracks and albums to the SoulSync wishlist and artists to the SoulSync watchlist,
and reports whether an item is already present in the local (SoulSync) library or
already wanted, so the frontend can badge it.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import suppress
from typing import TYPE_CHECKING, Any, TypedDict, cast

from music_assistant_models.auth import Scope
from music_assistant_models.config_entries import ConfigEntry
from music_assistant_models.enums import AlbumType, ConfigEntryType, ExternalID, MediaType
from music_assistant_models.errors import (
    InvalidDataError,
    MediaNotFoundError,
    ProviderUnavailableError,
    SetupFailedError,
)
from music_assistant_models.media_items import Album, Artist, ItemMapping, Track

from music_assistant.models.plugin import PluginProvider

from .client import SoulSyncClient, SoulSyncError
from .matching import LibraryIndex, WishlistIndex

if TYPE_CHECKING:
    from collections.abc import Callable

    from music_assistant_models.config_entries import ProviderConfig
    from music_assistant_models.enums import ProviderFeature
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant
    from music_assistant.models import ProviderInstanceType
    from music_assistant.models.music_provider import MusicProvider

CONF_URL = "url"
CONF_API_KEY = "api_key"
CONF_PROFILE_ID = "profile_id"
CONF_SKIP_LOCAL = "skip_local_tracks"
CONF_REFRESH_INTERVAL = "refresh_interval"
ENV_API_KEY = "SOULSYNC_API_KEY"
DEFAULT_URL = "http://192.168.1.129:8008"

# metadata sources SoulSync can resolve ids from (core/metadata/registry.py)
METADATA_SOURCES = ("spotify", "deezer")
SUPPORTED_MEDIA_TYPES = (MediaType.TRACK, MediaType.ALBUM, MediaType.ARTIST)
MAX_STATUS_ITEMS = 250
TRACK_FIELDS = (
    "id,album_id,artist_id,title,track_number,isrc,spotify_track_id,deezer_id,"
    "tidal_id,qobuz_id,itunes_track_id,musicbrainz_recording_id"
)


class ItemDescriptor(TypedDict, total=False):
    """Compact description of a media item, as sent by the frontend for status lookups."""

    uri: str
    media_type: str
    name: str
    version: str
    artists: list[str]
    album: str
    album_artists: list[str]
    mappings: list[list[str]]
    external_ids: list[list[str]]
    detail: bool


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Initialize provider(instance) with given configuration."""
    return SoulSyncProvider(mass, manifest, config, set())


class SoulSyncProvider(PluginProvider):
    """Bridge between Music Assistant items and the SoulSync wishlist/watchlist."""

    def __init__(
        self,
        mass: MusicAssistant,
        manifest: ProviderManifest,
        config: ProviderConfig,
        supported_features: set[ProviderFeature],
    ) -> None:
        """Initialize the provider."""
        super().__init__(mass, manifest, config, supported_features)
        self._client: SoulSyncClient
        self._library = LibraryIndex()
        self._wanted = WishlistIndex()
        self._library_ready = asyncio.Event()
        self._refresh_lock = asyncio.Lock()
        self._refresh_task: asyncio.Task[None] | None = None
        self._on_unload: list[Callable[[], None]] = []

    async def get_config_entries(self) -> tuple[ConfigEntry, ...]:
        """Return Config entries to configure this provider."""
        return (
            ConfigEntry(
                key=CONF_URL,
                type=ConfigEntryType.STRING,
                default_value=DEFAULT_URL,
                required=True,
            ),
            ConfigEntry(
                key=CONF_API_KEY,
                type=ConfigEntryType.SECURE_STRING,
                required=False,
            ),
            ConfigEntry(
                key=CONF_PROFILE_ID,
                type=ConfigEntryType.INTEGER,
                default_value=1,
                range=(1, 100),
                advanced=True,
            ),
            ConfigEntry(
                key=CONF_SKIP_LOCAL,
                type=ConfigEntryType.BOOLEAN,
                default_value=True,
            ),
            ConfigEntry(
                key=CONF_REFRESH_INTERVAL,
                type=ConfigEntryType.INTEGER,
                default_value=15,
                range=(5, 1440),
                advanced=True,
            ),
        )

    async def handle_async_init(self) -> None:
        """Create the SoulSync client and verify the connection."""
        api_key = str(self.get_config_value(CONF_API_KEY) or os.environ.get(ENV_API_KEY) or "")
        if not api_key:
            raise SetupFailedError(
                f"A SoulSync API key is required (config or {ENV_API_KEY} environment variable)"
            )
        self._client = SoulSyncClient(
            self.mass.http_session,
            str(self.get_config_value(CONF_URL) or DEFAULT_URL),
            api_key,
            self.get_config_value(CONF_PROFILE_ID, 1, return_type=int),
        )
        try:
            await self._client.get("/wishlist", limit=1)
        except SoulSyncError as err:
            raise SetupFailedError(f"SoulSync refused the API key: {err}") from err
        except ProviderUnavailableError as err:
            raise SetupFailedError(str(err)) from err

    async def loaded_in_mass(self) -> None:
        """Register the API commands and start indexing the SoulSync library."""
        await super().loaded_in_mass()
        self._on_unload.append(
            self.mass.register_api_command(
                "soulsync/status", self.get_status, required_scope=Scope.LIBRARY_READ
            )
        )
        self._on_unload.append(
            self.mass.register_api_command(
                "soulsync/add", self.add_item, required_scope=Scope.LIBRARY_WRITE
            )
        )
        self._on_unload.append(
            self.mass.register_api_command(
                "soulsync/refresh", self.refresh, required_scope=Scope.LIBRARY_WRITE
            )
        )
        self._refresh_task = self.mass.create_task(self._refresh_loop())

    async def unload(self, is_removed: bool = False) -> None:
        """Handle unload/close of the provider."""
        for unload_cb in self._on_unload:
            unload_cb()
        self._on_unload.clear()
        if self._refresh_task:
            self._refresh_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._refresh_task
        await super().unload(is_removed)

    async def get_status(self, items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """
        Return the SoulSync state of the given items, keyed by item uri.

        :param items: Compact item descriptors (uri, media_type, name, artists, ids, ...).
        """
        result: dict[str, dict[str, Any]] = {}
        for descriptor in items[:MAX_STATUS_ITEMS]:
            uri = descriptor.get("uri")
            if not uri:
                continue
            with suppress(ValueError):
                result[uri] = await self._status_for(cast("ItemDescriptor", descriptor))
        return result

    async def add_item(self, uri: str) -> dict[str, Any]:
        """
        Add a track or album to the SoulSync wishlist, or an artist to the watchlist.

        :param uri: Music Assistant uri of the item.
        """
        item = await self.mass.music.get_item_by_uri(uri)
        if isinstance(item, Track):
            result = await self._add_track(item)
        elif isinstance(item, Album):
            result = await self._add_album(item)
        elif isinstance(item, Artist):
            result = await self._add_artist(item)
        else:
            raise InvalidDataError(f"{uri} can not be added to SoulSync")
        await self._refresh_wanted()
        result["status"] = await self._status_for(_describe(item, detail=True))
        return result

    async def refresh(self) -> None:
        """Reload the SoulSync library, wishlist and watchlist indexes."""
        await self._refresh_wanted()
        await self._refresh_library()

    async def _status_for(self, descriptor: ItemDescriptor) -> dict[str, Any]:
        media_type = MediaType(descriptor.get("media_type", ""))
        if media_type not in SUPPORTED_MEDIA_TYPES:
            raise ValueError(media_type)
        name = _full_name(descriptor.get("name", ""), descriptor.get("version"))
        artists = descriptor.get("artists") or []
        mappings = _mapping_ids(descriptor)
        status: dict[str, Any] = {
            "local": False,
            "partial": False,
            "wanted": False,
            "watched": False,
            "local_tracks": None,
            "total_tracks": None,
            "wanted_tracks": 0,
            "indexed": self._library_ready.is_set(),
        }
        # an item already served by a local provider (e.g. plex) is local as well
        status["local"] = any(
            (prov := self.mass.get_provider(instance)) is not None
            and prov.domain != "builtin"
            and not getattr(prov, "is_streaming_provider", True)
            for _, _, instance in mappings
        )
        if media_type == MediaType.ARTIST:
            ids = _artist_ids(descriptor)
            status["local"] = status["local"] or bool(self._library.find_artist([name], ids))
            status["watched"] = self._wanted.is_watched(name, ids)
        elif media_type == MediaType.ALBUM:
            albums = self._library.find_albums(name, artists, _album_ids(descriptor))
            if albums:
                status["local"] = True
                status["local_tracks"] = self._library.local_track_count(albums)
                if descriptor.get("detail") or descriptor.get("uri", "").startswith("library://"):
                    with suppress(MediaNotFoundError, ProviderUnavailableError, InvalidDataError):
                        total = await self._album_track_total(descriptor["uri"])
                        status["total_tracks"] = total
                        status["partial"] = 0 < status["local_tracks"] < total
            status["wanted_tracks"] = self._wanted.album_track_count(name, artists)
            status["wanted"] = status["wanted_tracks"] > 0
        else:
            track_ids = _track_ids(descriptor)
            status["local"] = status["local"] or bool(
                self._library.find_track(name, artists, track_ids)
            )
            status["wanted"] = self._wanted.has_track(
                [item_id for _, item_id, _ in mappings], name, artists
            )
        if status["partial"]:
            status["state"] = "partial"
        elif status["local"]:
            status["state"] = "local"
        elif status["wanted"] or status["watched"]:
            status["state"] = "wanted"
        else:
            status["state"] = "none"
        return status

    async def _album_track_total(self, uri: str) -> int:
        item = await self.mass.music.get_item_by_uri(uri)
        tracks = await self.mass.music.albums.tracks(item.item_id, item.provider)
        return len(tracks)

    async def _add_track(self, track: Track) -> dict[str, Any]:
        track = await self.mass.music.tracks.get(track.item_id, track.provider)
        if self.get_config_value(CONF_SKIP_LOCAL) and self._library.find_track(
            track.name, [x.name for x in track.artists], _track_ids(_describe(track))
        ):
            return {"added": 0, "skipped_local": 1, "already_wanted": 0, "failed": 0}
        album = await self._full_album(track.album)
        source, source_track = await self._metadata_source_track(track)
        payload = _track_payload(source_track or track, album, source)
        outcome = await self._post_wishlist(payload, source_type="manual")
        return {
            "added": int(outcome == "created"),
            "skipped_local": 0,
            "already_wanted": int(outcome == "updated"),
            "failed": int(outcome == "failed"),
        }

    async def _add_album(self, album: Album) -> dict[str, Any]:
        album = await self.mass.music.albums.get(album.item_id, album.provider)
        tracks = await self.mass.music.albums.tracks(album.item_id, album.provider)
        if not tracks:
            raise MediaNotFoundError(f"No tracks found for album {album.name}")
        source, source_album, source_tracks = await self._metadata_source_album(album, tracks)
        payload_album = source_album or album
        candidates = source_tracks or tracks
        artist_names = [x.name for x in album.artists]
        result = {"added": 0, "skipped_local": 0, "already_wanted": 0, "failed": 0}
        skip_local = bool(self.get_config_value(CONF_SKIP_LOCAL))
        for track in candidates:
            names = [x.name for x in track.artists] + artist_names
            if skip_local and self._library.find_track(
                _full_name(track.name, track.version), names, _track_ids(_describe(track))
            ):
                result["skipped_local"] += 1
                continue
            payload = _track_payload(track, payload_album, source, total_tracks=len(candidates))
            if payload["id"] in self._wanted.track_ids:
                result["already_wanted"] += 1
                continue
            outcome = await self._post_wishlist(payload, source_type="album")
            if outcome == "created":
                result["added"] += 1
            elif outcome == "updated":
                result["already_wanted"] += 1
            else:
                result["failed"] += 1
        return result

    async def _add_artist(self, artist: Artist) -> dict[str, Any]:
        artist = await self.mass.music.artists.get(artist.item_id, artist.provider)
        descriptor = _describe(artist)
        if self._wanted.is_watched(artist.name, _artist_ids(descriptor)):
            return {"added": 0, "already_wanted": 1, "failed": 0, "skipped_local": 0}
        source, artist_id = await self._metadata_source_artist(artist)
        if not artist_id:
            raise MediaNotFoundError(f"SoulSync could not identify artist {artist.name}")
        await self._client.post(
            "/watchlist",
            {"artist_id": artist_id, "artist_name": artist.name, "source": source},
        )
        return {"added": 1, "already_wanted": 0, "failed": 0, "skipped_local": 0}

    async def _post_wishlist(self, payload: dict[str, Any], source_type: str) -> str:
        try:
            data, status = await self._client.post(
                "/wishlist",
                {
                    "track_data": payload,
                    "failure_reason": "Added from Music Assistant",
                    "source_type": source_type,
                },
            )
        except SoulSyncError as err:
            self.logger.warning("SoulSync refused %s: %s", payload.get("name"), err)
            return "failed"
        self._wanted.add_track_ids(str(payload["id"]))
        return "created" if status == 201 or data.get("created") else "updated"

    async def _metadata_source_track(self, track: Track) -> tuple[str | None, Track | None]:
        """Return the track as known by a SoulSync metadata source (spotify/deezer)."""
        for mapping in track.provider_mappings:
            if mapping.provider_domain in METADATA_SOURCES and mapping.available:
                return mapping.provider_domain, None
        for provider in self._metadata_providers():
            with suppress(Exception):
                matches = await self.mass.music.tracks.match_provider(track, provider, strict=True)
                if matches:
                    match = await self.mass.music.tracks.get_provider_item(
                        matches[0].item_id, matches[0].provider_instance
                    )
                    return provider.domain, match
        return None, None

    async def _metadata_source_album(
        self, album: Album, tracks: list[Track]
    ) -> tuple[str | None, Album | None, list[Track] | None]:
        """Return the album and its tracks as known by a SoulSync metadata source."""
        for mapping in album.provider_mappings:
            if mapping.provider_domain not in METADATA_SOURCES or not mapping.available:
                continue
            if all(
                any(m.provider_domain == mapping.provider_domain for m in t.provider_mappings)
                for t in tracks
            ):
                return mapping.provider_domain, None, None
            with suppress(Exception):
                source_album = await self.mass.music.albums.get_provider_item(
                    mapping.item_id, mapping.provider_instance
                )
                source_tracks = await self.mass.music.albums.tracks(
                    mapping.item_id, mapping.provider_instance
                )
                if source_tracks:
                    return mapping.provider_domain, source_album, source_tracks
        for provider in self._metadata_providers():
            with suppress(Exception):
                matches = await self.mass.music.albums.match_provider(album, provider, strict=True)
                if not matches:
                    continue
                source_album = await self.mass.music.albums.get_provider_item(
                    matches[0].item_id, matches[0].provider_instance
                )
                source_tracks = await self.mass.music.albums.tracks(
                    matches[0].item_id, matches[0].provider_instance
                )
                if source_tracks:
                    return provider.domain, source_album, source_tracks
        return None, None, None

    async def _metadata_source_artist(self, artist: Artist) -> tuple[str | None, str | None]:
        """Return the (source, id) SoulSync can use to watch the artist."""
        for mapping in artist.provider_mappings:
            if mapping.provider_domain in METADATA_SOURCES and mapping.available:
                return mapping.provider_domain, mapping.item_id
        if mbid := artist.get_external_id(ExternalID.MB_ARTIST):
            return "musicbrainz", mbid
        for provider in self._metadata_providers():
            with suppress(Exception):
                matches = await self.mass.music.artists.match_provider(
                    artist, provider, strict=True
                )
                if matches:
                    return provider.domain, matches[0].item_id
        # let SoulSync's own primary metadata source identify the artist by name
        data, _ = await self._client.post("/search/artists", {"query": artist.name, "limit": 10})
        for candidate in data.get("artists") or []:
            if _full_name(candidate.get("name", "")).casefold() == artist.name.casefold():
                return data.get("source"), str(candidate.get("id"))
        return None, None

    def _metadata_providers(self) -> list[MusicProvider]:
        return [
            provider
            for domain in METADATA_SOURCES
            for provider in self.mass.music.providers
            if provider.domain == domain and provider.available
        ]

    async def _full_album(self, album: Album | ItemMapping | None) -> Album | ItemMapping | None:
        if isinstance(album, ItemMapping):
            with suppress(MediaNotFoundError, ProviderUnavailableError, InvalidDataError):
                return await self.mass.music.albums.get(album.item_id, album.provider)
        return album

    async def _refresh_loop(self) -> None:
        while True:
            try:
                await self.refresh()
            except (SoulSyncError, ProviderUnavailableError) as err:
                self.logger.warning("Could not refresh the SoulSync index: %s", err)
            interval = self.get_config_value(CONF_REFRESH_INTERVAL, 15, return_type=int)
            await asyncio.sleep(interval * 60)

    async def _refresh_wanted(self) -> None:
        wishlist = [row async for row in self._client.iter_pages("/wishlist", "tracks")]
        self._wanted.set_wishlist(wishlist)
        watchlist = await self._client.get("/watchlist")
        self._wanted.set_watchlist(watchlist.get("artists") or [])

    async def _refresh_library(self) -> None:
        async with self._refresh_lock:
            artists = [row async for row in self._client.iter_pages("/library/artists", "artists")]
            albums = [row async for row in self._client.iter_pages("/library/albums", "albums")]
            previous = dict(self._library.albums)
            self._library.set_artists(artists)
            self._library.set_albums(albums)
            # only (re)load the tracks of new or changed albums
            for album in albums:
                old = previous.get(album["id"])
                if (
                    album["id"] in self._library.tracks_by_album
                    and old
                    and old.get("updated_at") == album.get("updated_at")
                    and old.get("track_count") == album.get("track_count")
                ):
                    continue
                with suppress(SoulSyncError):
                    data = await self._client.get(
                        f"/library/albums/{album['id']}/tracks", fields=TRACK_FIELDS
                    )
                    self._library.set_album_tracks(album["id"], data.get("tracks") or [])
            self._library_ready.set()
            self.logger.debug(
                "SoulSync index: %s artists, %s albums, %s tracks, %s wanted tracks",
                len(self._library.artists),
                len(self._library.albums),
                len(self._library.track_rows),
                len(self._wanted.tracks),
            )


def _full_name(name: str, version: str | None = None) -> str:
    return f"{name} ({version})" if version and version.casefold() not in name.casefold() else name


def _describe(item: Track | Album | Artist | ItemMapping, detail: bool = False) -> ItemDescriptor:
    descriptor: ItemDescriptor = {
        "uri": item.uri or "",
        "media_type": item.media_type.value,
        "name": item.name,
        "version": getattr(item, "version", "") or "",
        "artists": [x.name for x in getattr(item, "artists", [])],
        "mappings": [
            [x.provider_domain, x.item_id, x.provider_instance]
            for x in getattr(item, "provider_mappings", [])
        ],
        "external_ids": [[key.value, value] for key, value in getattr(item, "external_ids", [])],
        "detail": detail,
    }
    album = getattr(item, "album", None)
    if album is not None:
        descriptor["album"] = album.name
        descriptor["album_artists"] = [x.name for x in getattr(album, "artists", [])]
    return descriptor


def _mapping_ids(descriptor: ItemDescriptor) -> list[tuple[str, str, str]]:
    return [
        (str(mapping[0]), str(mapping[1]), str(mapping[2] if len(mapping) > 2 else mapping[0]))
        for mapping in descriptor.get("mappings") or []
        if len(mapping) >= 2
    ]


def _external_ids(descriptor: ItemDescriptor) -> dict[str, str]:
    return {
        str(entry[0]): str(entry[1])
        for entry in descriptor.get("external_ids") or []
        if len(entry) == 2
    }


def _provider_ids(descriptor: ItemDescriptor) -> list[tuple[str, str]]:
    return [(domain, item_id) for domain, item_id, _ in _mapping_ids(descriptor)]


def _artist_ids(descriptor: ItemDescriptor) -> list[tuple[str, str]]:
    ids = _provider_ids(descriptor)
    if mbid := _external_ids(descriptor).get(ExternalID.MB_ARTIST.value):
        ids.append(("musicbrainz", mbid))
    return ids


def _album_ids(descriptor: ItemDescriptor) -> list[tuple[str, str]]:
    ids = _provider_ids(descriptor)
    external = _external_ids(descriptor)
    if mbid := external.get(ExternalID.MB_ALBUM.value):
        ids.append(("musicbrainz", mbid))
    if barcode := external.get(ExternalID.BARCODE.value):
        ids.append(("barcode", barcode))
    return ids


def _track_ids(descriptor: ItemDescriptor) -> list[tuple[str, str]]:
    ids = _provider_ids(descriptor)
    external = _external_ids(descriptor)
    if mbid := external.get(ExternalID.MB_RECORDING.value):
        ids.append(("musicbrainz", mbid))
    if isrc := external.get(ExternalID.ISRC.value):
        ids.append(("isrc", isrc))
    return ids


def _track_payload(
    track: Track,
    album: Album | ItemMapping | None,
    source: str | None,
    total_tracks: int | None = None,
) -> dict[str, Any]:
    """Build the (Spotify-shaped) track_data SoulSync stores on a wishlist row."""
    track_id = album_id = None
    if source:
        track_id = next(
            (x.item_id for x in track.provider_mappings if x.provider_domain == source), None
        )
        if isinstance(album, (Album, ItemMapping)):
            album_id = next(
                (
                    x.item_id
                    for x in getattr(album, "provider_mappings", [])
                    if x.provider_domain == source
                ),
                album.item_id if album.provider.startswith(source) else None,
            )
    if not track_id:
        # an id SoulSync can not mistake for a Spotify/Deezer/iTunes id
        source = None
        track_id = f"ma-{track.provider}-{track.item_id}"
    album_artists = [x.name for x in getattr(album, "artists", [])] or [
        x.name for x in track.artists
    ]
    album_payload: dict[str, Any] = {
        "id": album_id or (f"ma-{album.provider}-{album.item_id}" if album else None),
        "name": _full_name(album.name, getattr(album, "version", "")) if album else track.name,
        "artists": [{"name": name} for name in album_artists],
        "images": [],
        "release_date": "",
        "total_tracks": total_tracks,
    }
    if isinstance(album, Album):
        album_type = {
            AlbumType.SINGLE: "single",
            AlbumType.EP: "ep",
            AlbumType.COMPILATION: "compilation",
        }.get(album.album_type, "album" if album.album_type != AlbumType.UNKNOWN else None)
        if album_type:
            album_payload["album_type"] = album_type
        if album.year:
            album_payload["release_date"] = str(album.year)
        if album.image and album.image.remotely_accessible:
            album_payload["images"] = [{"url": album.image.path, "height": 640, "width": 640}]
    payload: dict[str, Any] = {
        "id": track_id,
        "name": _full_name(track.name, track.version),
        "artists": [{"name": x.name} for x in track.artists],
        "album": album_payload,
        "duration_ms": int(track.duration or 0) * 1000,
        "track_number": track.track_number or None,
        "disc_number": track.disc_number or None,
        "explicit": bool(track.metadata.explicit) if track.metadata else False,
        "external_urls": {},
        "popularity": 0,
        "preview_url": None,
    }
    if source:
        payload["source"] = source
    if isrc := track.get_external_id(ExternalID.ISRC):
        payload["isrc"] = isrc
    return payload
