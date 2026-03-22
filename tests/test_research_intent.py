"""Tests for research intent detection helpers."""

from __future__ import annotations

from services.research.intent import has_research_intent


def test_has_research_intent_matches_expected_phrases() -> None:
    assert has_research_intent("Can you look up this pinout?") is True
    assert has_research_intent("Please search online for this board") is True
    assert has_research_intent("find datasheet for ads1015") is True
    assert has_research_intent("what does the datasheet say about gain") is True


def test_has_research_intent_matches_general_web_queries() -> None:
    assert has_research_intent("Can you search the web for Waveshare Servo HAT voltage?") is True
    assert has_research_intent("Please find specs for PCA9685 board") is True
    assert has_research_intent("what is the pinout for ads1015") is True


def test_has_research_intent_prefers_motion_for_short_look_up_directional_phrases() -> None:
    assert has_research_intent("look up") is False
    assert has_research_intent("look up please") is False
    assert has_research_intent("look up now") is False
    assert has_research_intent("look up a bit") is False
    assert has_research_intent("look up and then right") is False


def test_has_research_intent_prefers_motion_for_positional_look_up_tails() -> None:
    assert has_research_intent("look up to the ceiling") is False
    assert has_research_intent("look up at the ceiling") is False
    assert has_research_intent("look up at me") is False
    assert has_research_intent("look up to the left") is False


def test_has_research_intent_keeps_clear_lookup_queries() -> None:
    assert has_research_intent("look up the weather") is True
    assert has_research_intent("look up Toronto weather") is True
    assert has_research_intent("look up Sectigo") is True
    assert has_research_intent("look up what this means") is True


def test_has_research_intent_treats_bare_can_you_look_up_as_non_research() -> None:
    assert has_research_intent("can you look up") is False
    assert has_research_intent("can you look up please") is False
    assert has_research_intent("can you look up Toronto weather") is True


def test_has_research_intent_ignores_non_research_requests() -> None:
    assert has_research_intent("hello theo") is False
    assert has_research_intent("tell me a joke") is False


def test_has_research_intent_does_not_conflate_lookup_with_look_up() -> None:
    assert has_research_intent("lookup Toronto weather") is False
