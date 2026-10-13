"""Minimal client for the SoulSync public REST API (v1)."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import TYPE_CHECKING, Any, cast

import aiohttp
from music_assistant_models.errors import InvalidDataError, ProviderUnavailableError

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

# SoulSync limits /api/v1 to 60 requests per minute per client address; stay below it
# and keep part of the budget free for user actions while the indexer runs
MAX_REQUESTS_PER_MINUTE = 50
BACKGROUND_REQUESTS_PER_MINUTE = 30
PAGE_SIZE = 200
RATE_LIMIT_BACKOFF = 61


class SoulSyncError(InvalidDataError):
    """Raised when SoulSync refuses a request."""

    def __init__(self, message: str, status: int, code: str | None = None) -> None:
        """
        Initialize the error.

        :param message: Error message returned by SoulSync.
        :param status: HTTP status code of the response.
        :param code: SoulSync error code, if provided.
        """
        super().__init__(message)
        self.status = status
        self.code = code


class SoulSyncClient:
    """Talk to the SoulSync /api/v1 endpoints with an API key."""

    def __init__(
        self, session: aiohttp.ClientSession, base_url: str, api_key: str, profile_id: int
    ) -> None:
        """
        Initialize the client.

        :param session: Shared aiohttp session.
        :param base_url: SoulSync base URL, e.g. http://192.168.1.129:8008.
        :param api_key: SoulSync API key.
        :param profile_id: SoulSync profile the wishlist belongs to.
        """
        self._session = session
        self._base_url = base_url.rstrip("/") + "/api/v1"
        self._headers = {"X-API-Key": api_key, "X-Profile-Id": str(profile_id)}
        self._request_times: deque[float] = deque()

    async def get(self, path: str, background: bool = False, **params: Any) -> dict[str, Any]:
        """
        Perform a GET request and return the response data.

        :param path: Endpoint path below /api/v1.
        :param background: Use the smaller request budget of background work.
        :param params: Query parameters.
        """
        data, _ = await self._request("GET", path, params=params, background=background)
        return cast("dict[str, Any]", data)

    async def post(self, path: str, payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
        """
        Perform a POST request and return the response data and HTTP status.

        :param path: Endpoint path below /api/v1.
        :param payload: JSON body.
        """
        data, status = await self._request("POST", path, json=payload)
        return cast("dict[str, Any]", data), cast("int", status)

    async def iter_pages(
        self, path: str, key: str, background: bool = False, **params: Any
    ) -> AsyncGenerator[dict[str, Any]]:
        """
        Iterate over all rows of a paginated listing.

        :param path: Endpoint path below /api/v1.
        :param key: Name of the list in the response data.
        :param background: Use the smaller request budget of background work.
        :param params: Additional query parameters.
        """
        page = 1
        while True:
            data, pagination = await self._request(
                "GET",
                path,
                params={**params, "page": page, "limit": PAGE_SIZE},
                paginated=True,
                background=background,
            )
            for row in data.get(key) or []:
                yield row
            if not pagination or not pagination.get("has_next"):
                return
            page += 1

    async def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        paginated: bool = False,
        background: bool = False,
        retried: bool = False,
    ) -> Any:
        await self._throttle(background)
        try:
            async with self._session.request(
                method,
                self._base_url + path,
                params=params,
                json=json,
                headers=self._headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                try:
                    body = await response.json(content_type=None)
                except ValueError:
                    body = None
                status = response.status
        except (TimeoutError, aiohttp.ClientError) as err:
            raise ProviderUnavailableError(f"SoulSync is unreachable: {err}") from err
        if status == 429 and not retried:
            # another client (or an overlapping restart) used up the shared minute
            await asyncio.sleep(RATE_LIMIT_BACKOFF)
            return await self._request(
                method, path, params, json, paginated, background, retried=True
            )
        if not isinstance(body, dict) or not body.get("success"):
            error = (body or {}).get("error") if isinstance(body, dict) else None
            message = (error or {}).get("message") or f"HTTP {status}"
            raise SoulSyncError(message, status, (error or {}).get("code"))
        if paginated:
            return body.get("data") or {}, body.get("pagination")
        return body.get("data") or {}, status

    async def _throttle(self, background: bool) -> None:
        limit = BACKGROUND_REQUESTS_PER_MINUTE if background else MAX_REQUESTS_PER_MINUTE
        while True:
            now = time.monotonic()
            while self._request_times and now - self._request_times[0] > 60:
                self._request_times.popleft()
            if len(self._request_times) < limit:
                self._request_times.append(now)
                return
            # no lock is held while waiting, so a user action is never queued
            # behind background requests that wait for a smaller budget
            await asyncio.sleep(60 - (now - self._request_times[-limit]) + 0.1)
