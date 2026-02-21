"""SQLite-backed trusted domain allowlist storage."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import ipaddress
from pathlib import Path
import sqlite3
import threading
from typing import Iterable
from urllib.parse import urlparse

from config import ConfigController


_BLOCKED_HOSTS = {"localhost", "127.0.0.1", "::1"}


@dataclass(frozen=True)
class TrustedDomain:
    domain: str
    added_at: str
    added_by: str


class TrustedDomainStore:
    """Manage persistent trusted domains for research allowlisting."""

    def __init__(self, db_path: Path | None = None) -> None:
        if db_path is None:
            config = ConfigController.get_instance().get_config()
            var_dir = Path(config.get("var_dir", "./var/")).expanduser()
            db_path = var_dir / "user_profiles.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self._db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._initialize_db()

    @property
    def db_path(self) -> Path:
        return self._db_path

    def _initialize_db(self) -> None:
        cursor = self._conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS trusted_domains (
                domain TEXT PRIMARY KEY,
                added_at TIMESTAMP,
                added_by TEXT DEFAULT 'user'
            )
            """
        )
        self._conn.commit()

    def list_domains(self) -> list[TrustedDomain]:
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT domain, COALESCE(added_at, ''), COALESCE(added_by, 'user')
            FROM trusted_domains
            ORDER BY domain ASC
            """
        )
        rows = cursor.fetchall()
        return [TrustedDomain(domain=row[0], added_at=row[1], added_by=row[2]) for row in rows]

    def get_domain_set(self) -> set[str]:
        return {item.domain for item in self.list_domains()}

    def add_domain(self, domain_or_url: str, *, added_by: str = "user") -> str | None:
        domain = normalize_trusted_domain(domain_or_url)
        if domain is None:
            return None
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO trusted_domains (domain, added_at, added_by)
                VALUES (?, ?, ?)
                ON CONFLICT(domain) DO NOTHING
                """,
                (domain, now, added_by or "user"),
            )
            self._conn.commit()
        return domain

    def remove_domain(self, domain_or_url: str) -> bool:
        domain = normalize_trusted_domain(domain_or_url)
        if domain is None:
            return False
        with self._lock:
            cursor = self._conn.execute("DELETE FROM trusted_domains WHERE domain = ?", (domain,))
            self._conn.commit()
        return cursor.rowcount > 0


def normalize_trusted_domain(domain_or_url: str) -> str | None:
    raw = str(domain_or_url or "").strip().lower()
    if not raw or "*" in raw:
        return None

    parsed = urlparse(raw)
    if parsed.scheme == "file":
        return None

    candidate = raw
    if "://" in raw:
        if parsed.scheme not in {"http", "https"}:
            return None
        candidate = parsed.hostname or ""
    elif "/" in raw or "?" in raw or "#" in raw:
        parsed = urlparse(f"https://{raw}")
        candidate = parsed.hostname or ""

    candidate = candidate.strip(". ").lower()
    if not candidate or candidate in _BLOCKED_HOSTS or candidate.endswith(".local"):
        return None
    if candidate.endswith(".internal"):
        return None

    try:
        ip = ipaddress.ip_address(candidate)
    except ValueError:
        ip = None

    if ip is not None:
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return None
        return candidate

    if " " in candidate:
        return None
    return candidate


def merge_allowlists(static_domains: Iterable[str], trusted_domains: Iterable[str]) -> set[str]:
    merged: set[str] = set()
    for domain in [*static_domains, *trusted_domains]:
        normalized = normalize_trusted_domain(str(domain))
        if normalized:
            merged.add(normalized)
    return merged
