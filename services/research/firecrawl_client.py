"""Minimal Firecrawl wrapper for datasheet markdown ingestion."""

from __future__ import annotations

import json
import os
import time
from typing import Any
from urllib import error, request


class FirecrawlError(RuntimeError):
    """Base error for Firecrawl fetch failures."""


class FirecrawlHTTPError(FirecrawlError):
    """Firecrawl request failed with an HTTP status code."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = int(status_code)


class FirecrawlClient:
    """HTTP client for Firecrawl scrape endpoint."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_url: str = "https://api.firecrawl.dev/v1/scrape",
        timeout_s: float = 30.0,
        retry_attempts: int = 2,
    ) -> None:
        self._api_key = (api_key or os.getenv("FIRECRAWL_API_KEY", "")).strip()
        self._api_url = api_url
        self._timeout_s = max(5.0, float(timeout_s))
        self._retry_attempts = max(1, int(retry_attempts))

    @property
    def enabled(self) -> bool:
        return bool(self._api_key)

    def fetch_markdown(self, url: str, *, max_pages: int = 1, max_markdown_chars: int = 20000) -> str:
        if not self.enabled:
            raise FirecrawlError("firecrawl_missing_key")
        payload = {
            "url": url,
            "formats": ["markdown"],
            "maxAge": 0,
            "maxPages": max(1, int(max_pages)),
        }
        data = json.dumps(payload).encode("utf-8")
        body = ""
        for attempt in range(1, self._retry_attempts + 1):
            req = request.Request(
                self._api_url,
                data=data,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "pyPiBot-research/1.0",
                },
                method="POST",
            )
            try:
                with request.urlopen(req, timeout=self._timeout_s) as response:
                    body = response.read().decode("utf-8")
                break
            except error.HTTPError as exc:
                status = int(exc.code)
                if status in {401, 403}:
                    raise FirecrawlHTTPError(status, "firecrawl_auth_failed") from exc
                if attempt >= self._retry_attempts:
                    raise FirecrawlHTTPError(status, "firecrawl_http_error") from exc
                time.sleep(0.2 * attempt)
            except error.URLError as exc:
                if attempt >= self._retry_attempts:
                    raise FirecrawlError("firecrawl_timeout_or_network_error") from exc
                time.sleep(0.2 * attempt)
        response_payload = json.loads(body)
        markdown = str((response_payload.get("data") or {}).get("markdown") or "").strip()
        if len(markdown) > max_markdown_chars:
            return markdown[:max_markdown_chars] + "…"
        return markdown
