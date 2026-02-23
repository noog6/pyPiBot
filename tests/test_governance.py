"""Tests for governance runtime threshold behavior."""

from ai.governance import GovernanceLayer, ToolSpec


def _build_layer() -> GovernanceLayer:
    tool_specs = {
        "perform_research": ToolSpec(
            tier=1,
            reversible=False,
            cost_hint="med",
            safety_tags=("network", "research"),
            confirm_required=False,
        )
    }
    config = {
        "governance": {
            "autonomy_level": "act-with-bounds",
            "risk_threshold": 0.95,
            "guarded_thresholds": {
                "max_cost_score": 0.8,
                "min_rate_limit_remaining": 2,
                "privacy_flag_requires_confirmation": True,
            },
            "budgets": {"tool_calls_per_minute": 0, "expensive_calls_per_day": 0},
        }
    }
    return GovernanceLayer(tool_specs, config)


def test_tier1_guarded_tool_runs_without_confirmation_in_normal_runtime_context() -> None:
    layer = _build_layer()
    action = layer.build_action_packet("perform_research", "call_1", {"query": "weather"})

    decision = layer.decide_tool_call(
        action,
        runtime_context={
            "cost_score": 0.4,
            "rate_limit_remaining": 10,
            "privacy_flag": False,
        },
    )

    assert decision.status == "approved"
    assert decision.needs_confirmation is False


def test_tier1_guarded_tool_requires_confirmation_when_runtime_threshold_exceeded() -> None:
    layer = _build_layer()
    action = layer.build_action_packet("perform_research", "call_2", {"query": "weather"})

    decision = layer.decide_tool_call(
        action,
        runtime_context={
            "cost_score": 0.85,
            "rate_limit_remaining": 1,
            "privacy_flag": True,
        },
    )

    assert decision.status == "needs_confirmation"
    assert decision.needs_confirmation is True
    assert decision.reason == "tier1_guarded_threshold_exceeded"


def test_build_normalized_idempotency_key_collapses_string_whitespace() -> None:
    from ai.governance import build_normalized_idempotency_key

    key_with_space = build_normalized_idempotency_key(
        "update_user_profile",
        {"name": "Sprinkles 2", "favorites": ["ice cream"]},
    )
    key_without_space = build_normalized_idempotency_key(
        "update_user_profile",
        {"name": "Sprinkles2", "favorites": ["icecream"]},
    )

    assert key_with_space == key_without_space
