"""Minimal Firecrawl wrapper for datasheet markdown ingestion."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib import request


class FirecrawlClient:
    """HTTP client for Firecrawl scrape endpoint."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_url: str = "https://api.firecrawl.dev/v1/scrape",
        timeout_s: float = 30.0,
    ) -> None:
        self._api_key = (api_key or os.getenv("FIRECRAWL_API_KEY", "")).strip()
        self._api_url = api_url
        self._timeout_s = max(5.0, float(timeout_s))

    @property
    def enabled(self) -> bool:
        return bool(self._api_key)

    def fetch_markdown(self, url: str, *, max_pages: int = 1, max_markdown_chars: int = 20000) -> str:
        payload = {
            "url": url,
            "formats": ["markdown"],
            "maxAge": 0,
            "maxPages": max(1, int(max_pages)),
        }
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self._api_url,
            data=data,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=self._timeout_s) as response:
            body = response.read().decode("utf-8")
        response_payload = json.loads(body)
        markdown = str((response_payload.get("data") or {}).get("markdown") or "").strip()
        if len(markdown) > max_markdown_chars:
            return markdown[:max_markdown_chars] + "…"
        return markdown
