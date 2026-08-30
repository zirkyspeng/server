"""Helper(s) to create DIDL Lite metadata for Sonos/DLNA players."""

from __future__ import annotations

from math import isfinite
from typing import TYPE_CHECKING
from unicodedata import normalize
from xml.sax.saxutils import escape as xmlescape

from music_assistant_models.enums import MediaType

from music_assistant.constants import MASS_LOGO_ONLINE
from music_assistant.helpers.audio import get_mime_type

if TYPE_CHECKING:
    from music_assistant.models.player import PlayerMedia


TIME_SEEK_HEADER = "TimeSeekRange.dlna.org"


def _parse_npt_time(value: str) -> float:
    """Parse a DLNA normal-play-time token into seconds."""
    value = value.strip()
    if not value:
        raise ValueError("missing NPT time")
    if ":" not in value:
        seconds = float(value)
    else:
        parts = value.split(":")
        if len(parts) != 3:
            raise ValueError("invalid NPT clock time")
        hours, minutes, seconds_part = parts
        seconds = int(hours) * 3600 + int(minutes) * 60 + float(seconds_part)
    if not isfinite(seconds) or seconds < 0:
        raise ValueError("invalid NPT time")
    return seconds


def parse_time_seek_range(value: str, duration: float) -> tuple[float, float | None]:
    """Parse and validate a DLNA TimeSeekRange request against a known duration."""
    if not value.startswith("npt="):
        raise ValueError("missing npt prefix")
    range_value = value[4:].split(" ", 1)[0]
    if "-" not in range_value:
        raise ValueError("missing NPT range separator")
    start_value, end_value = range_value.split("-", 1)
    start = _parse_npt_time(start_value)
    end = _parse_npt_time(end_value) if end_value else None
    if start >= duration:
        raise OverflowError("NPT start is outside the media duration")
    if end is not None:
        if end > duration:
            raise OverflowError("NPT end is outside the media duration")
        if end <= start:
            raise ValueError("NPT end must be after start")
    return start, end


def format_time_seek_range(start: float, end: float, duration: float) -> str:
    """Format the time-only form of a DLNA TimeSeekRange response."""
    return f"npt={start:.3f}-{end:.3f}/{duration:.3f}"


# XML
def _get_soap_action(command: str) -> str:
    return f"urn:schemas-upnp-org:service:AVTransport:1#{command}"


def _get_body(command: str, arguments: str = "", service: str = "AVTransport") -> str:
    return (
        f'<u:{command} xmlns:u="urn:schemas-upnp-org:service:{service}:1">'
        r"<InstanceID>0</InstanceID>"
        f"{arguments}"
        f"</u:{command}>"
    )


def _get_xml(body: str) -> str:
    return (
        r'<?xml version="1.0"?>'
        r'<s:Envelope s:encodingStyle="http://schemas.xmlsoap.org/soap/encoding/" xmlns:s="http://schemas.xmlsoap.org/soap/envelope/">'
        r"<s:Body>"
        f"{body}"
        r"</s:Body>"
        r"</s:Envelope>"
    )


def get_xml_soap_play() -> tuple[str, str]:
    """Get UPnP xml and soap for Play."""
    command = "Play"
    arguments = r"<Speed>1</Speed>"
    return _get_xml(_get_body(command, arguments)), _get_soap_action(command)


def get_xml_soap_stop() -> tuple[str, str]:
    """Get UPnP xml and soap for Stop."""
    command = "Stop"
    return _get_xml(_get_body(command)), _get_soap_action(command)


def get_xml_soap_pause() -> tuple[str, str]:
    """Get UPnP xml and soap for Pause."""
    command = "Pause"
    return _get_xml(_get_body(command)), _get_soap_action(command)


def get_xml_soap_next() -> tuple[str, str]:
    """Get UPnP xml and soap for Next."""
    command = "Next"
    return _get_xml(_get_body(command)), _get_soap_action(command)


def get_xml_soap_previous() -> tuple[str, str]:
    """Get UPnP xml and soap for Previous."""
    command = "Previous"
    return _get_xml(_get_body(command)), _get_soap_action(command)


def get_xml_soap_transport_info() -> tuple[str, str]:
    """Get UPnP xml and soap for GetTransportInfo."""
    command = "GetTransportInfo"
    return _get_xml(_get_body(command)), _get_soap_action(command)


def get_xml_soap_media_info() -> tuple[str, str]:
    """Get UPnP xml and soap for GetMediaInfo."""
    command = "GetMediaInfo"
    return _get_xml(_get_body(command)), _get_soap_action(command)


def get_xml_soap_set_url(player_media: PlayerMedia) -> tuple[str, str]:
    """Get UPnP xml and soap for SetAVTransportURI."""
    metadata = create_didl_metadata_str(player_media)
    command = "SetAVTransportURI"
    arguments = (
        f"<CurrentURI>{player_media.uri}</CurrentURI>"
        "<CurrentURIMetaData>"
        f"{metadata}"
        "</CurrentURIMetaData>"
    )
    return _get_xml(_get_body(command, arguments)), _get_soap_action(command)


def get_xml_soap_remove_all_tracks() -> tuple[str, str]:
    """Get UPnP xml and soap for RemoveAllTracksFromQueue."""
    command = "RemoveAllTracksFromQueue"
    return _get_xml(_get_body(command)), _get_soap_action(command)


def get_xml_soap_set_next_url(player_media: PlayerMedia) -> tuple[str, str]:
    """Get UPnP xml and soap for SetNextAVTransportURI."""
    metadata = create_didl_metadata_str(player_media)
    command = "SetNextAVTransportURI"
    arguments = (
        f"<NextURI>{player_media.uri}</NextURI><NextURIMetaData>{metadata}</NextURIMetaData>"
    )
    return _get_xml(_get_body(command, arguments)), _get_soap_action(command)


# RemoveTrackFromQueue
def get_xml_soap_remove_track(object_id: str) -> tuple[str, str]:
    """Get UPnP xml and soap for RemoveTrackFromQueue."""
    command = "RemoveTrackFromQueue"
    arguments = f"<ObjectID>{object_id}</ObjectID>"
    return _get_xml(_get_body(command, arguments)), _get_soap_action(command)


# AddURIToQueue
def get_xml_soap_add_uri_to_queue(player_media: PlayerMedia) -> tuple[str, str]:
    """Get UPnP xml and soap for AddURIToQueue."""
    metadata = create_didl_metadata_str(player_media)
    command = "AddURIToQueue"
    arguments = (
        f"<EnqueuedURI>{player_media.uri}</EnqueuedURI>"
        f"<EnqueuedURIMetaData>{metadata}</EnqueuedURIMetaData>"
        "<DesiredFirstTrackNumberEnqueued>1</DesiredFirstTrackNumberEnqueued>"
        "<EnqueueAsNext>0</EnqueueAsNext>"
    )
    return _get_xml(_get_body(command, arguments)), _get_soap_action(command)


# CreateSavedQueue
def get_xml_soap_create_saved_queue(queue_name: str, player_media: PlayerMedia) -> tuple[str, str]:
    """Get UPnP xml and soap for CreateSavedQueue."""
    command = "CreateSavedQueue"
    metadata = create_didl_metadata_str(player_media)
    arguments = (
        f"<Title>{xmlescape(queue_name)}</Title>"
        f"<EnqueuedURI>{player_media.uri}</EnqueuedURI>"
        f"<EnqueuedURIMetaData>{metadata}</EnqueuedURIMetaData>"
    )
    return _get_xml(_get_body(command, arguments)), _get_soap_action(command)


# CreateQueue
def get_xml_soap_create_queue() -> tuple[str, str]:
    """Get UPnP xml and soap for CreateQueue."""
    command = "CreateQueue"
    arguments = (
        "<QueueOwnerID>mass</QueueOwnerID>"
        "<QueueOwnerContext>mass</QueueOwnerContext>"
        "<QueuePolicy>0</QueuePolicy>"
    )
    return _get_xml(_get_body(command, arguments, "Queue")), _get_soap_action(command)


# DIDL-LITE
def create_didl_metadata(
    media: PlayerMedia,
    url: str | None = None,
    *,
    supports_time_seek: bool = False,
    ascii_only: bool = False,
    image_url: str | None = None,
    minimal_profile: bool = False,
) -> str:
    """Create DIDL metadata string from url and PlayerMedia."""
    uri = url or media.uri

    def escape_metadata(data: str) -> str:
        """Escape didl metadata."""
        if ascii_only:
            # Some Marantz firmware truncates decimal Unicode entities to one
            # byte (for example U+2019 becomes control byte 0x19), causing its
            # on-screen metadata parser to discard the whole update.
            for old, new in (
                ("\u2018", "'"),
                ("\u2019", "'"),
                ("\u201c", '"'),
                ("\u201d", '"'),
                ("\u2013", "-"),
                ("\u2014", "-"),
                ("\u2026", "..."),
            ):
                data = data.replace(old, new)
            data = normalize("NFKD", data).encode("ascii", "ignore").decode()
            data = "".join(char if char.isprintable() else " " for char in data)
        data = xmlescape(data)
        # Escape non-ascii to decimal code.
        result = ""
        for char in data:
            unicode_code = ord(char)
            if unicode_code < 128:
                # ascii
                result += char
            else:
                result += f"&#{unicode_code};"
        return result

    ext = uri.split(".")[-1].split("?")[0]
    mime_type = get_mime_type(ext)
    image_url = image_url or media.image_url or MASS_LOGO_ONLINE
    if media.media_type in (MediaType.FLOW_STREAM, MediaType.RADIO) or not media.duration:
        # flow stream, radio or other duration-less stream
        # Use streaming-optimized DLNA flags to prevent buffering
        title = media.title or uri
        return (
            '<DIDL-Lite xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:upnp="urn:schemas-upnp-org:metadata-1-0/upnp/" xmlns="urn:schemas-upnp-org:metadata-1-0/DIDL-Lite/" xmlns:dlna="urn:schemas-dlna-org:metadata-1-0/">'
            f'<item id="flowmode" parentID="0" restricted="1">'
            f"<dc:title>{escape_metadata(title)}</dc:title>"
            f"<upnp:albumArtURI>{escape_metadata(image_url)}</upnp:albumArtURI>"
            f"<dc:queueItemId>{escape_metadata(uri)}</dc:queueItemId>"
            f"<dc:description>Music Assistant</dc:description>"
            "<upnp:class>object.item.audioItem.audioBroadcast</upnp:class>"
            f'<res protocolInfo="http-get:*:{mime_type}:DLNA.ORG_OP=01;DLNA.ORG_CI=0;DLNA.ORG_FLAGS=01700000000000000000000000000000">{escape_metadata(uri)}</res>'
            "</item>"
            "</DIDL-Lite>"
        )

    assert media.queue_item_id is not None  # for type checking

    # For regular tracks with duration, use flags optimized for on-demand content
    # DLNA.ORG_FLAGS=01500000000000000000000000000000 indicates:
    # - Streaming transfer mode (bit 24)
    # - Background transfer mode supported (bit 22)
    # - DLNA v1.5 (bit 20)
    # Time-seek-capable renderers keep an absolute media timeline, including when
    # the first response starts at a resume offset. Other renderers receive only
    # the shortened stream and therefore need its remaining duration.
    stream_duration = int(
        media.duration if supports_time_seek else (media.stream_duration or media.duration or 0)
    )
    duration_str = str(stream_duration // 3600).zfill(2) + ":"
    duration_str += str((stream_duration % 3600) // 60).zfill(2) + ":"
    duration_str += str(stream_duration % 60).zfill(2)

    if minimal_profile:
        # Some Marantz firmware interprets Sonos queue extensions as a signal to
        # defer publishing the transport state. Keep only standard music-track
        # metadata so HEOS can update the display as playback starts.
        return (
            '<DIDL-Lite xmlns="urn:schemas-upnp-org:metadata-1-0/DIDL-Lite/" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/" '
            'xmlns:upnp="urn:schemas-upnp-org:metadata-1-0/upnp/">'
            '<item id="1" parentID="1" restricted="1">'
            f"<dc:title>{escape_metadata(media.title or uri)}</dc:title>"
            f"<dc:creator>{escape_metadata(media.artist or '')}</dc:creator>"
            f"<upnp:artist>{escape_metadata(media.artist or '')}</upnp:artist>"
            f"<upnp:album>{escape_metadata(media.album or '')}</upnp:album>"
            f"<upnp:albumArtURI>{escape_metadata(image_url)}</upnp:albumArtURI>"
            "<upnp:class>object.item.audioItem.musicTrack</upnp:class>"
            f'<res duration="{duration_str}" protocolInfo="http-get:*:{mime_type}:DLNA.ORG_OP={"11" if supports_time_seek else "01"};DLNA.ORG_CI=0;DLNA.ORG_FLAGS=01500000000000000000000000000000">{escape_metadata(uri)}</res>'
            "</item>"
            "</DIDL-Lite>"
        )

    return (
        '<DIDL-Lite xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:upnp="urn:schemas-upnp-org:metadata-1-0/upnp/" xmlns="urn:schemas-upnp-org:metadata-1-0/DIDL-Lite/" xmlns:r="urn:schemas-rinconnetworks-com:metadata-1-0/">'
        f'<item id="{media.queue_item_id or xmlescape(uri)}" restricted="true" parentID="{media.source_id or ""}">'
        f"<dc:title>{escape_metadata(media.title or uri)}</dc:title>"
        f"<dc:creator>{escape_metadata(media.artist or '')}</dc:creator>"
        f"<upnp:album>{escape_metadata(media.album or '')}</upnp:album>"
        f"<upnp:artist>{escape_metadata(media.artist or '')}</upnp:artist>"
        f"<dc:queueItemId>{escape_metadata(media.queue_item_id)}</dc:queueItemId>"
        f"<dc:description>Music Assistant</dc:description>"
        f"<upnp:albumArtURI>{escape_metadata(image_url)}</upnp:albumArtURI>"
        "<upnp:class>object.item.audioItem.musicTrack</upnp:class>"
        f'<res duration="{duration_str}" protocolInfo="http-get:*:{mime_type}:DLNA.ORG_OP={"11" if supports_time_seek else "01"};DLNA.ORG_CI=0;DLNA.ORG_FLAGS=01500000000000000000000000000000">{escape_metadata(uri)}</res>'
        '<desc id="cdudn" nameSpace="urn:schemas-rinconnetworks-com:metadata-1-0/">RINCON_AssociatedZPUDN</desc>'
        "</item>"
        "</DIDL-Lite>"
    )


def create_didl_metadata_str(media: PlayerMedia) -> str:
    """Create (xml-escaped) DIDL metadata string from url and PlayerMedia."""
    return xmlescape(create_didl_metadata(media))
