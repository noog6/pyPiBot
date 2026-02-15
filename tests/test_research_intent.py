"""Tests for research intent detection helpers."""

from __future__ import annotations

from services.research.intent import has_research_intent


def test_has_research_intent_matches_expected_phrases() -> None:
    assert has_research_intent("Can you look up this pinout?") is True
    assert has_research_intent("Please search online for this board") is True
    assert has_research_intent("find datasheet for ads1015") is True
    assert has_research_intent("what does the datasheet say about gain") is True


def test_has_research_intent_ignores_non_research_requests() -> None:
    assert has_research_intent("hello theo") is False
    assert has_research_intent("tell me a joke") is False
