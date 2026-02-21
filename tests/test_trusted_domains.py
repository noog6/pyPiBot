"""Tests for trusted research domain persistence and validation."""

from __future__ import annotations

from storage.trusted_domains import TrustedDomainStore, normalize_trusted_domain


def test_normalize_trusted_domain_strips_to_base_domain() -> None:
    assert normalize_trusted_domain("HTTPS://Slashdot.ORG/story?id=1") == "slashdot.org"


def test_normalize_trusted_domain_rejects_unsafe_hosts() -> None:
    assert normalize_trusted_domain("file:///etc/passwd") is None
    assert normalize_trusted_domain("localhost") is None
    assert normalize_trusted_domain("127.0.0.1") is None
    assert normalize_trusted_domain("10.0.0.5") is None
    assert normalize_trusted_domain("*.example.com") is None


def test_remove_domain_round_trip(tmp_path) -> None:
    store = TrustedDomainStore(db_path=tmp_path / "trusted.db")
    assert store.add_domain("example.com") == "example.com"
    assert store.remove_domain("https://example.com/path") is True
    assert "example.com" not in store.get_domain_set()
