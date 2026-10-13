"""Name normalization and in-memory indexes of the SoulSync library, wishlist and watchlist."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, ClassVar

# parenthesised/bracketed or dash-suffixed release decorations that do not change
# which recording or release is meant ("Remastered 2011", "Deluxe Edition", ...)
_EDITION_WORDS = (
    "remaster",
    "remastered",
    "deluxe",
    "edition",
    "expanded",
    "anniversary",
    "bonus",
    "version",
    "mono",
    "stereo",
    "explicit",
    "clean",
    "single",
)
_EDITION_RE = "|".join(_EDITION_WORDS)
_BRACKETED_EDITION = re.compile(
    rf"\s*[\(\[][^\)\]]*\b(?:{_EDITION_RE})\b[^\)\]]*[\)\]]", re.IGNORECASE
)
_DASHED_EDITION = re.compile(rf"\s+-\s+[^-]*\b(?:{_EDITION_RE})\b.*$", re.IGNORECASE)
_FEATURING = re.compile(r"\s*[\(\[]?\b(?:feat|ft|featuring)\b\.?.*$", re.IGNORECASE)
_NON_WORD = re.compile(r"[\W_]+", re.UNICODE)


def normalize(value: str | None, strip_edition: bool = True) -> str:
    """
    Return a comparison key for a title or name.

    Accents and punctuation are dropped and case is folded; non-latin scripts are kept.

    :param value: The title or name to normalize.
    :param strip_edition: Also drop edition decorations such as "(Remastered 2011)".
    """
    if not value:
        return ""
    text = value
    if strip_edition:
        text = _BRACKETED_EDITION.sub("", text)
        text = _DASHED_EDITION.sub("", text)
    text = _FEATURING.sub("", text) or text
    text = "".join(_strip_latin_accent(char) for char in unicodedata.normalize("NFKC", text))
    text = text.casefold().replace("&", " and ")
    return _NON_WORD.sub("", text)


def artists_match(wanted: list[str], candidate: str | None) -> bool:
    """
    Return True when one of the wanted artist names matches the candidate artist.

    :param wanted: Artist names of the Music Assistant item.
    :param candidate: Artist name (or joined artist names) of the SoulSync entry.
    """
    candidate_key = normalize(candidate, strip_edition=False)
    if not candidate_key:
        return False
    for name in wanted:
        key = normalize(name, strip_edition=False)
        if not key:
            continue
        if key == candidate_key:
            return True
        # joined credits ("A, B") or a credit with a featured artist
        if len(key) >= 3 and key in candidate_key:
            return True
    return False


@dataclass
class LibraryIndex:
    """Lookup tables over the SoulSync library (the user's local files)."""

    artists: dict[int, dict[str, Any]] = field(default_factory=dict)
    albums: dict[int, dict[str, Any]] = field(default_factory=dict)
    tracks_by_album: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    artist_ids_by_name: dict[str, set[int]] = field(default_factory=lambda: defaultdict(set))
    album_ids_by_title: dict[str, set[int]] = field(default_factory=lambda: defaultdict(set))
    track_keys_by_title: dict[str, set[int]] = field(default_factory=lambda: defaultdict(set))
    external_ids: dict[tuple[str, str, str], int] = field(default_factory=dict)
    track_rows: dict[int, dict[str, Any]] = field(default_factory=dict)

    # SoulSync external id columns per entity, keyed by the Music Assistant domain
    ARTIST_ID_COLUMNS: ClassVar[dict[str, str]] = {
        "spotify": "spotify_artist_id",
        "deezer": "deezer_id",
        "tidal": "tidal_id",
        "qobuz": "qobuz_id",
        "apple_music": "itunes_artist_id",
        "musicbrainz": "musicbrainz_id",
    }
    ALBUM_ID_COLUMNS: ClassVar[dict[str, str]] = {
        "spotify": "spotify_album_id",
        "deezer": "deezer_id",
        "tidal": "tidal_id",
        "qobuz": "qobuz_id",
        "apple_music": "itunes_album_id",
        "musicbrainz": "musicbrainz_release_id",
        "barcode": "upc",
    }
    TRACK_ID_COLUMNS: ClassVar[dict[str, str]] = {
        "spotify": "spotify_track_id",
        "deezer": "deezer_id",
        "tidal": "tidal_id",
        "qobuz": "qobuz_id",
        "apple_music": "itunes_track_id",
        "musicbrainz": "musicbrainz_recording_id",
        "isrc": "isrc",
    }

    def set_artists(self, rows: list[dict[str, Any]]) -> None:
        """Replace the artist rows."""
        self.artists = {row["id"]: row for row in rows if row.get("id") is not None}
        self.artist_ids_by_name = defaultdict(set)
        for row in self.artists.values():
            self.artist_ids_by_name[normalize(row.get("name"), strip_edition=False)].add(row["id"])
            self._add_external_ids("artist", row, self.ARTIST_ID_COLUMNS)

    def set_albums(self, rows: list[dict[str, Any]]) -> None:
        """Replace the album rows, dropping cached tracks of albums that disappeared."""
        self.albums = {row["id"]: row for row in rows if row.get("id") is not None}
        self.album_ids_by_title = defaultdict(set)
        for row in self.albums.values():
            self.album_ids_by_title[normalize(row.get("title"))].add(row["id"])
            self._add_external_ids("album", row, self.ALBUM_ID_COLUMNS)
        for album_id in list(self.tracks_by_album):
            if album_id not in self.albums:
                self.set_album_tracks(album_id, [])
                del self.tracks_by_album[album_id]

    def set_album_tracks(self, album_id: int, rows: list[dict[str, Any]]) -> None:
        """Replace the track rows of one album."""
        for old in self.tracks_by_album.get(album_id, []):
            self.track_rows.pop(old["id"], None)
            keys = self.track_keys_by_title.get(normalize(old.get("title")))
            if keys:
                keys.discard(old["id"])
        self.tracks_by_album[album_id] = rows
        for row in rows:
            if row.get("id") is None:
                continue
            self.track_rows[row["id"]] = row
            self.track_keys_by_title[normalize(row.get("title"))].add(row["id"])
            self._add_external_ids("track", row, self.TRACK_ID_COLUMNS)

    def artist_name(self, artist_id: int | None) -> str:
        """Return the name of a SoulSync artist id."""
        if artist_id is None or artist_id not in self.artists:
            return ""
        return str(self.artists[artist_id].get("name") or "")

    def find_artist(self, names: list[str], ids: list[tuple[str, str]]) -> dict[str, Any] | None:
        """Find a library artist by external id or exact (normalized) name."""
        for key, value in ids:
            artist_id = self.external_ids.get(("artist", key, value.casefold()))
            if artist_id in self.artists:
                return self.artists[artist_id]
        for name in names:
            for artist_id in self.artist_ids_by_name.get(normalize(name, strip_edition=False), ()):
                return self.artists[artist_id]
        return None

    def find_albums(
        self, title: str, artist_names: list[str], ids: list[tuple[str, str]]
    ) -> list[dict[str, Any]]:
        """Find library albums by external id, or by title and artist."""
        for key, value in ids:
            album_id = self.external_ids.get(("album", key, value.casefold()))
            if album_id in self.albums:
                return [self.albums[album_id]]
        return [
            self.albums[album_id]
            for album_id in self.album_ids_by_title.get(normalize(title), ())
            if artists_match(artist_names, self.artist_name(self.albums[album_id].get("artist_id")))
        ]

    def find_track(
        self, title: str, artist_names: list[str], ids: list[tuple[str, str]]
    ) -> dict[str, Any] | None:
        """Find a library track by external id, or by title and artist."""
        for key, value in ids:
            track_id = self.external_ids.get(("track", key, value.casefold()))
            if track_id in self.track_rows:
                return self.track_rows[track_id]
        for track_id in self.track_keys_by_title.get(normalize(title), ()):
            row = self.track_rows[track_id]
            album = self.albums.get(int(row.get("album_id") or 0), {})
            if artists_match(artist_names, self.artist_name(row.get("artist_id"))) or (
                artists_match(artist_names, self.artist_name(album.get("artist_id")))
            ):
                return row
        return None

    def local_track_count(self, albums: list[dict[str, Any]]) -> int:
        """Return how many tracks of the given library albums are present."""
        count = 0
        for album in albums:
            if album["id"] in self.tracks_by_album:
                count += len(self.tracks_by_album[album["id"]])
            else:
                count += int(album.get("track_count") or 0)
        return count

    def _add_external_ids(self, kind: str, row: dict[str, Any], columns: dict[str, str]) -> None:
        for key, column in columns.items():
            if value := row.get(column):
                self.external_ids[(kind, key, str(value).casefold())] = row["id"]


@dataclass
class WishlistIndex:
    """Lookup tables over the SoulSync wishlist and artist watchlist."""

    track_ids: set[str] = field(default_factory=set)
    tracks: list[dict[str, Any]] = field(default_factory=list)
    watchlist: list[dict[str, Any]] = field(default_factory=list)
    watchlist_ids: set[tuple[str, str]] = field(default_factory=set)

    def set_wishlist(self, rows: list[dict[str, Any]]) -> None:
        """Replace the wishlist rows."""
        self.tracks = rows
        self.track_ids = {
            str(row[key])
            for row in rows
            for key in ("track_id", "spotify_track_id")
            if row.get(key)
        }

    def set_watchlist(self, rows: list[dict[str, Any]]) -> None:
        """Replace the watchlist rows."""
        self.watchlist = rows
        self.watchlist_ids = set()
        for row in rows:
            for source in ("spotify", "itunes", "deezer", "discogs", "musicbrainz", "amazon"):
                if value := row.get(f"{source}_artist_id"):
                    self.watchlist_ids.add((source, str(value)))

    def add_track_ids(self, *track_ids: str) -> None:
        """Remember freshly added track ids until the next refresh."""
        self.track_ids.update(track_ids)

    def has_track(self, ids: list[str], title: str, artist_names: list[str]) -> bool:
        """Return True when the track is on the wishlist."""
        if any(track_id in self.track_ids for track_id in ids):
            return True
        key = normalize(title)
        return any(
            normalize(row.get("track_name")) == key
            and artists_match(artist_names, row.get("artist_name"))
            for row in self.tracks
        )

    def album_track_count(self, title: str, artist_names: list[str]) -> int:
        """Return how many wishlist tracks belong to the given album."""
        key = normalize(title)
        return sum(
            1
            for row in self.tracks
            if normalize(row.get("album_name")) == key
            and artists_match(artist_names, row.get("artist_name"))
        )

    def is_watched(self, name: str, ids: list[tuple[str, str]]) -> bool:
        """Return True when the artist is on the watchlist."""
        if any(item in self.watchlist_ids for item in ids):
            return True
        key = normalize(name, strip_edition=False)
        return any(
            normalize(row.get("artist_name"), strip_edition=False) == key for row in self.watchlist
        )


def _strip_latin_accent(char: str) -> str:
    # only latin letters lose their accents; kana voicing marks and the like are kept
    if ord(char) >= 0x250:
        return char
    return "".join(
        part for part in unicodedata.normalize("NFKD", char) if unicodedata.category(part) != "Mn"
    )
