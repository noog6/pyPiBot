"""Realtime API controller for OpenAI realtime connections."""

from __future__ import annotations

import asyncio
import audioop
import base64
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
import hashlib
from io import BytesIO
import json
import logging
import os
import random
import re
import signal
import threading
import time
import uuid
from typing import Any, Callable, Iterator
from urllib import request
from urllib.parse import urlparse

from config import ConfigController
from core.alert_policy import Alert, AlertPolicy
from core.budgeting import RollingWindowBudget
from core.logging import (
    configure_websocket_library_logging,
    logger,
    log_error,
    log_info,
    log_session_updated,
    log_tool_call,
    log_warning,
    log_ws_event,
)
from interaction import (
    AsyncMicrophone,
    AudioPlayer,
    InteractionState,
    InteractionStateManager,
)
from ai.tools import function_map, render_memory_cards_for_assistant, tools
from ai.function_call_accumulator import FunctionCallAccumulator
from ai.realtime.event_router import EventRouter
from ai.realtime.confirmation import (
    ConfirmationCoordinator,
    ConfirmationReminderDecision,
    ConfirmationState,
    ConfirmationTransitionDecision,
    normalize_confirmation_decision,
)
from ai.realtime.confirmation_service import ConfirmationService
from ai.realtime.confirmation_runtime import ConfirmationRuntime
from ai.realtime.battery_injection_policy import BatteryInjectionPolicy
from ai.realtime.injection_bus import InjectionBus
from ai.realtime.injections import InjectionCoordinator
from ai.realtime.input_audio_events import InputAudioEventHandlers
from ai.realtime.lifecycle_state import LifecycleStateCoordinator
from ai.realtime.memory_runtime import MemoryRuntime
from ai.realtime.research_runtime import ResearchRuntime
from ai.realtime.response_create_runtime import ResponseCreateRuntime
from ai.realtime.response_lifecycle import ResponseLifecycleTracker
from ai.realtime.response_terminal_handlers import ResponseTerminalHandlers
from ai.realtime.asr_trust import build_utterance_trust_snapshot, should_clarify
from ai.realtime.runtime_tasks import RuntimeTaskRegistry
from ai.realtime.shutdown import ShutdownCoordinator
from ai.realtime.transport import RealtimeTransport
from ai.realtime.types import CanonicalResponseState, PendingResponseCreate, UtteranceContext
from ai.realtime import preference_recall_runtime
from ai.interaction_lifecycle_controller import (
    InteractionLifecycleController,
    InteractionLifecycleState,
    LifecycleDecisionAction,
)
from ai.interaction_lifecycle_policy import (
    InteractionLifecyclePolicy,
    ResponseCreateDecisionAction,
    ServerAutoCreatedDecisionAction,
)
from ai.reflection import ReflectionCoordinator, ReflectionContext
from ai.stimuli_coordinator import StimuliCoordinator
from ai.governance import (
    ActionPacket,
    GovernanceLayer,
    build_normalized_idempotency_key,
    build_tool_specs,
    normalize_tool_arguments,
    normalized_decision_payload,
)
from ai.orchestration import OrchestrationPhase, OrchestrationState
from ai.event_bus import Event, EventBus
from ai.event_injector import EventInjector
from ai.micro_ack_manager import MicroAckCategory, MicroAckConfig, MicroAckContext, MicroAckManager
from ai.utils import (
    PREFIX_PADDING_MS,
    RUN_TIME_TABLE_LOG_JSON,
    SILENCE_DURATION_MS,
    SILENCE_THRESHOLD,
    build_session_instructions,
)
from motion import (
    MotionController,
    gesture_attention_snap,
    gesture_curious_tilt,
    gesture_idle,
    gesture_nod,
)
from motion.gesture_library import DEFAULT_GESTURES
from services.profile_manager import ProfileManager
from services.reflection_manager import ReflectionManager
from services.memory_manager import (
    MemoryBrief,
    MemoryManager,
    MemoryScope,
)
from services.research import ResearchRequest, build_openai_service_or_null, has_research_intent
from services.research.grounding import (
    build_research_grounding_explanation,
    build_unverified_sources_only_response,
    get_content_fetch_state,
    requires_unverified_sources_only_response,
)
from services.research.research_transcript import write_research_transcript
from storage import StorageController
from storage.trusted_domains import TrustedDomainStore, merge_allowlists, normalize_trusted_domain

RESPONSE_DONE_REFLECTION_PROMPT = """Summarize what changed. Any anomalies? Should anything be remembered?
Return ONLY valid JSON with keys: summary, remember_memory.
Rules:
- summary: one-line run summary.
- remember_memory: null OR an object with keys {{content, tags, importance, confidence}}.
- tags: optional list of short strings.
- importance: integer 1-5.
- confidence: float 0.0-1.0 for how certain this should be stored as durable memory.

Context:
User input: {user_input}
Assistant reply: {assistant_reply}
Tool calls: {tool_calls}
Response metadata: {response_metadata}
"""

ALLOWED_OUTBOUND_HOSTS = {"api.openai.com"}
ALLOWED_OUTBOUND_SCHEMES = {"https", "wss"}


_EMAIL_REDACT_RE = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_REDACT_RE = re.compile(r"(?:(?<=\s)|^)(?:\+?\d[\d()\-\s]{7,}\d)(?=\s|$)")
_PREFERENCE_RECALL_MARKERS = (
    "prefer",
    "preferred",
    "preference",
    "favorite",
    "favourite",
    "what do i use",
    "remind me",
    "which",
)
_PREFERENCE_RECALL_DOMAINS = {
    "editor",
    "ide",
    "tool",
    "workflow",
    "language",
    "theme",
    "music",
    "drink",
    "food",
}
_PREFERENCE_QUERY_CANONICAL_BY_DOMAIN = {
    "editor": ("preferred editor", "favorite editor", "user preference"),
    "ide": ("preferred editor", "favorite editor", "user preference"),
}
_PREFERENCE_QUERY_FALLBACK_CANONICAL = ("user preference", "favorite preference", "preferred setting")
_PREFERENCE_FALLBACK_MARKERS = ("remember", "recall", "search", "memory", "memories")
_MEMORY_RECALL_ANSWER_MARKERS = (
    "relevant memory",
    "i found a saved preference",
    "i don't have that saved yet",
    "i don’t have that saved yet",
    "what i remember",
)
_MEMORY_REASSURANCE_SUFFIX_PATTERNS = (
    re.compile(r"^(?:if you want,?\s+)?i can remember .* next time\.?$", re.IGNORECASE),
    re.compile(r"^(?:if you want,?\s+)?i can save .* memory\.?$", re.IGNORECASE),
    re.compile(r"^want me to (?:save|remember|pin|rename|update) .*\?$", re.IGNORECASE),
    re.compile(r"^would you like me to (?:save|remember|pin|rename|update) .*\?$", re.IGNORECASE),
)
_MEMORY_RECALL_CONCISE_FOLLOWUP = "Want me to save or update that memory?"
_MICRO_ACK_CONFIRMATION_ALLOWLIST: frozenset[str] = frozenset(
    {
        "watchdog_confirmation_pending",
        "watchdog_permission_pending",
        "watchdog_approval_pending",
    }
)
_MICRO_ACK_CONFIRMATION_DECLINE_GUARD_S = 5.0
_PREFERENCE_KEYWORD_STOPWORDS = {
    "which",
    "what",
    "who",
    "when",
    "where",
    "why",
    "how",
    "do",
    "i",
    "my",
    "you",
    "the",
    "a",
    "an",
    "is",
    "are",
    "to",
    "for",
    "of",
    "about",
    "prefer",
    "preferred",
    "preference",
    "favorite",
    "favourite",
}
_PREFERENCE_QUERY_NOISE_TOKENS = {
    "hey",
    "theo",
    "remember",
    "memories",
    "memory",
    "check",
    "anything",
    "related",
}
_PREFERENCE_RECALL_VARIANT_NOISE_TOKENS = _PREFERENCE_QUERY_NOISE_TOKENS | {
    "user",
    "you",
    "your",
}


@dataclass
class PendingAction:
    action: ActionPacket
    staging: dict[str, Any]
    original_intent: str
    created_at: float
    idempotency_key: str | None = None
    retry_count: int = 0
    max_retries: int = 2


@dataclass
class SensorAggregationWindow:
    source: str
    severity: str
    trigger: str
    first_seen_monotonic: float
    last_seen_monotonic: float
    total_events: int = 0
    dropped_events: int = 0
    payload_count: int = 0
    latest_metadata: dict[str, Any] | None = None


@dataclass
class PendingServerAutoResponse:
    turn_id: str
    response_id: str
    canonical_key: str
    created_at_ms: int
    active: bool = True
    cancelled_for_upgrade: bool = False
    pre_audio_hold: bool = False
    upgrade_chain_id: str = ""


@dataclass
class ServerAutoPreAudioHoldRecord:
    turn_id: str
    response_id: str
    hold_started_at: float
    hold_released_at: float | None = None
    last_reason: str = ""


@dataclass
class ResponseGatingVerdict:
    action: str
    reason: str
    decided_at: float


@dataclass(frozen=True)
class NormalizedConfirmationDecision:
    status: str
    reason: str
    action_summary: str
    approved: bool
    needs_confirmation: bool
    confirm_required: bool
    confirm_reason: str | None
    confirm_prompt: str | None
    idempotency_key: str
    cooldown_seconds: float
    dry_run_supported: bool
    decision_source: str = "tier_default"
    thresholds: dict[str, Any] | None = None
    max_reminders: int | None = None
    reminder_schedule_seconds: tuple[float, ...] = ()


@dataclass(frozen=True)
class PendingMicroAckMarker:
    category: MicroAckCategory | str
    priority: int
    reason: str


class ServerAutoArbitrationOutcome(str, Enum):
    ALLOW = "allow"
    CANCEL_PRE_AUDIO = "cancel_pre_audio"
    DEFER = "defer"


@dataclass(frozen=True)
class StartupDependencyOutcome:
    component: str
    dependency_class: str
    status: str
    detail: str | None = None


class RealtimeAPIStartupError(RuntimeError):
    """Typed startup error for classified RealtimeAPI dependency failures."""

    def __init__(self, message: str, *, outcome: StartupDependencyOutcome) -> None:
        super().__init__(message)
        self.outcome = outcome


class _DisabledMicrophone:
    """Fallback microphone implementation used when input startup is degraded."""

    def __init__(self) -> None:
        self.is_recording = False
        self.is_receiving = False

    def start_recording(self) -> None:
        self.is_recording = True

    def stop_recording(self) -> None:
        self.is_recording = False

    def start_receiving(self) -> None:
        self.is_receiving = True
        self.is_recording = False

    def stop_receiving(self) -> None:
        self.is_receiving = False

    def get_audio_data(self) -> bytes | None:
        return None

    def drain_queue(self, max_items: int = 9999) -> int:
        _ = max_items
        return 0

    def close(self) -> None:
        return None


@dataclass
class PendingConfirmationToken:
    id: str
    kind: str
    tool_name: str | None
    request: ResearchRequest | None
    pending_action: PendingAction | None
    created_at: float
    expiry_ts: float | None
    retry_count: int = 0
    max_retries: int = 2
    prompt_sent: bool = False
    reminder_sent: bool = False
    metadata: dict[str, Any] | None = None


@dataclass
class IntentLedgerEntry:
    tool_name: str
    normalized_intent: str
    idempotency_key: str | None
    state: str
    updated_at: float
    pending_at: float | None = None
    approved_at: float | None = None
    denied_at: float | None = None
    timeout_at: float | None = None
    failure_at: float | None = None
    executed_at: float | None = None


@dataclass
class ConversationEfficiencyState:
    ack_count: int = 0
    substantive_count: int = 0
    duplicate_prevented: int = 0
    dropped_queued_creates: int = 0
    estimated_token_overhead: int = 0
    estimated_audio_overhead_ms: int = 0
    silent_turn_incidents: int = 0
    substantive_count_by_canonical: dict[str, int] | None = None
    duplicate_alerted_canonical_keys: set[str] | None = None


def _validate_outbound_endpoint(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_OUTBOUND_SCHEMES:
        raise RuntimeError(
            f"Blocked outbound endpoint with non-TLS scheme: {parsed.scheme or 'missing'}"
        )
    if not parsed.hostname:
        raise RuntimeError("Blocked outbound endpoint with missing hostname.")
    if parsed.hostname not in ALLOWED_OUTBOUND_HOSTS:
        raise RuntimeError(f"Blocked outbound endpoint to untrusted host: {parsed.hostname}")
    if parsed.username or parsed.password:
        raise RuntimeError("Blocked outbound endpoint with embedded credentials.")


def _require_websockets() -> Any:
    import importlib
    import importlib.util

    if importlib.util.find_spec("websockets") is None:
        raise RuntimeError("websockets is required for RealtimeAPI")

    websockets = importlib.import_module("websockets")
    return websockets


def _resolve_websocket_exceptions(websockets: Any) -> tuple[type[BaseException], type[BaseException]]:
    connection_closed = getattr(websockets, "ConnectionClosed", None)
    connection_closed_error = getattr(websockets, "ConnectionClosedError", None)
    exceptions_module = getattr(websockets, "exceptions", None)
    if exceptions_module is not None:
        connection_closed = getattr(exceptions_module, "ConnectionClosed", connection_closed)
        connection_closed_error = getattr(
            exceptions_module, "ConnectionClosedError", connection_closed_error
        )
    if connection_closed is None or connection_closed_error is None:
        raise RuntimeError("Unsupported websockets version: missing ConnectionClosed errors.")
    return connection_closed, connection_closed_error


def _format_rate_limit_field(value: Any, *, none_token: str = "n/a") -> str:
    if value is None:
        return none_token
    return str(value)


def _format_rate_limit_duration(value: Any, *, none_token: str = "n/a") -> str:
    formatted = _format_rate_limit_field(value, none_token=none_token)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{formatted}s"
    return formatted


def parse_rate_limits(event: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Parse and normalize realtime API rate_limit buckets from an event payload.

    Buckets are optional; absence is treated as normal unless payloads are malformed
    or regressions persist based on session policy.
    """

    raw_rate_limits = event.get("rate_limits")

    def _redact_sample_entry(entry: Any) -> Any:
        if not isinstance(entry, dict):
            return {"type": type(entry).__name__}
        return {
            "name": entry.get("name"),
            "remaining": entry.get("remaining"),
            "limit": entry.get("limit"),
            "reset_seconds": entry.get("reset_seconds"),
        }

    meta: dict[str, Any] = {
        "event_id": event.get("event_id"),
        "present_names": [],
        "unknown_names": [],
        "entry_count": 0,
        "malformed_count": 0,
        "rate_limits_is_list": isinstance(raw_rate_limits, list),
        "sample_entries": [],
    }
    if not isinstance(raw_rate_limits, list):
        return {}, meta

    meta["entry_count"] = len(raw_rate_limits)
    meta["sample_entries"] = [_redact_sample_entry(entry) for entry in raw_rate_limits[:2]]
    rl_map: dict[str, dict[str, Any]] = {}
    present_names: set[str] = set()
    unknown_names: set[str] = set()
    allowed_names = {"requests", "tokens"}

    for entry in raw_rate_limits:
        if not isinstance(entry, dict):
            meta["malformed_count"] += 1
            continue
        name = str(entry.get("name", "")).strip().lower()
        if not name:
            meta["malformed_count"] += 1
            unknown_names.add("<missing>")
            continue
        present_names.add(name)
        if name in allowed_names:
            rl_map[name] = entry
        else:
            unknown_names.add(name)

    meta["present_names"] = sorted(present_names)
    meta["unknown_names"] = sorted(unknown_names)
    return rl_map, meta


def log_runtime(function_or_name: str, duration: float) -> None:
    jsonl_file = RUN_TIME_TABLE_LOG_JSON
    os.makedirs(os.path.dirname(jsonl_file), exist_ok=True)
    time_record = {
        "timestamp": datetime.now().isoformat(),
        "function": function_or_name,
        "duration": f"{duration:.4f}",
    }
    with open(jsonl_file, "a", encoding="utf-8") as file:
        json.dump(time_record, file)
        file.write("\n")

    logger.info("⏰ %s() took %.4f seconds", function_or_name, duration)


def base64_encode_audio(audio_bytes: bytes) -> str:
    return base64.b64encode(audio_bytes).decode("utf-8")


def _validate_tool_specs(tool_specs: dict[str, Any], tool_defs: list[dict[str, Any]]) -> None:
    required_fields = (
        "tier",
        "reversible",
        "cost_hint",
        "safety_tags",
        "confirm_required",
        "confirm_reason",
        "confirm_prompt",
        "cooldown_seconds",
        "dry_run_supported",
        "governance_tier",
        "side_effects",
        "sensitivity",
        "default_confirmation",
    )
    tool_names = {tool.get("name") for tool in tool_defs}
    missing_specs = sorted(tool_names - set(tool_specs.keys()))
    if missing_specs:
        log_warning(
            "Governance tool_specs missing entries for: %s",
            ", ".join(missing_specs),
        )
    unknown_specs = sorted(set(tool_specs.keys()) - tool_names)
    if unknown_specs:
        log_warning(
            "Governance tool_specs include unknown tools: %s",
            ", ".join(unknown_specs),
        )
    for tool_name, spec in tool_specs.items():
        if not isinstance(spec, dict):
            log_warning("Governance tool_specs for %s should be a mapping.", tool_name)
            continue
        missing_fields = [field for field in required_fields if field not in spec]
        if missing_fields:
            log_warning(
                "Governance tool_specs for %s missing fields: %s",
                tool_name,
                ", ".join(missing_fields),
            )
        tier = spec.get("tier")
        if tier is None:
            log_warning("Governance tool_specs for %s missing tier value.", tool_name)
        elif int(tier) not in (0, 1, 2, 3):
            log_warning(
                "Governance tool_specs for %s has unexpected tier %s.",
                tool_name,
                tier,
            )


class RealtimeAPI:
    """Realtime OpenAI API client."""

    def __init__(self, prompts: list[str] | None = None) -> None:
        self.prompts = prompts
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("Please set the OPENAI_API_KEY environment variable.")
        config = ConfigController.get_instance().get_config()
        audio_cfg = config.get("audio") or {}
        input_cfg = audio_cfg.get("input") or {}
        output_cfg = audio_cfg.get("output") or {}
        startup_cfg = config.get("startup") or {}
        self._audio_input_device_name = input_cfg.get("device_name")
        self._audio_output_device_name = output_cfg.get("device_name")
        self._allow_audio_startup_degraded = bool(startup_cfg.get("allow_audio_input_failure", False))
        self.exit_event = asyncio.Event()
        self.mic = self._initialize_microphone(
            allow_failure=self._allow_audio_startup_degraded,
        )
        self.audio_player: AudioPlayer | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.ready_event = threading.Event()
        self._shutdown = ShutdownCoordinator(
            close_timeout_s=float(config.get("websocket_close_timeout_s", 5.0))
        )
        self._runtime_tasks = RuntimeTaskRegistry()

        self.assistant_reply = ""
        self._assistant_reply_accum = ""
        self._assistant_reply_response_id: str | None = None
        self._audio_accum = bytearray()
        self._audio_accum_response_id: str | None = None
        self._audio_accum_bytes_target = 9600
        self.response_in_progress = False
        self._response_in_flight = False
        self._response_create_queue: deque[dict[str, Any]] = deque()
        self._pending_response_create: PendingResponseCreate | None = None
        self._response_create_enqueue_seq = 0
        self._response_create_queued_creates_total = 0
        self._response_create_drains_total = 0
        self._response_create_max_qdepth = 0
        self._response_create_turn_counter = 0
        self._current_response_turn_id: str | None = None
        self._utterance_context: UtteranceContext | None = None
        self._queued_confirmation_reminder_keys: set[str] = set()
        self._pending_confirmation_prompt_latches: set[str] = set()
        self._pending_response_create_origins: deque[dict[str, str]] = deque(maxlen=64)
        self._audio_playback_busy = False
        self.function_call: dict[str, Any] | None = None
        self.function_call_args = ""
        self.mic_send_suppress_until = 0.0
        # Tracks whether to start mic receiving on the first audio delta for the current response.
        self._mic_receive_on_first_audio = False
        self.rate_limits: dict[str, Any] | None = None
        self.rate_limits_supports_tokens = False
        self.rate_limits_supports_requests = False
        self.rate_limits_last_present_names: set[str] = set()
        self.rate_limits_last_event_id = ""
        self._rate_limits_regression_missing_counts: dict[str, int] = {
            "tokens": 0,
            "requests": 0,
        }
        self.response_start_time: float | None = None
        self.websocket = None
        self.profile_manager = ProfileManager.get_instance()
        self._memory_manager = MemoryManager.get_instance()
        self.state_manager = InteractionStateManager()
        self.state_manager.set_gesture_handler(self._handle_state_gesture)
        self.state_manager.set_earcon_handler(self._handle_state_earcon)
        self._speaking_started = False
        self._gesture_last_fired: dict[str, float] = {}
        self._last_gesture_time = 0.0
        self._last_interaction_state = self.state_manager.state
        self._gesture_global_cooldown_s = 10.0
        self._pending_image_stimulus: dict[str, Any] | None = None
        self._pending_image_flush_after_playback = False
        self._last_user_input_text: str | None = None
        self._last_user_input_time: float | None = None
        self._last_user_input_source: str | None = None
        self._pending_turn_memory_brief: MemoryBrief | None = None
        self._last_user_battery_query_time: float | None = None
        self._last_outgoing_event_type: str | None = None
        self._tool_call_records: list[dict[str, Any]] = []
        self._last_tool_call_results: list[dict[str, Any]] = []
        self._last_response_metadata: dict[str, Any] = {}
        self._injection_response_cooldown_s = float(
            config.get("injection_response_cooldown_s", 0.0)
        )
        self._max_injection_responses_per_minute = int(
            config.get("max_injection_responses_per_minute", 0)
        )
        self._injection_response_timestamps: deque[float] = deque()
        self._injection_response_triggers = config.get("injection_response_triggers") or {}
        self._injection_response_trigger_timestamps: dict[str, deque[float]] = {}
        self._sensor_event_aggregation_window_s = min(
            5.0,
            max(0.0, float(config.get("sensor_event_aggregation_window_s", 3.0))),
        )
        self._sensor_event_aggregate_sources: set[str] = {
            "battery",
            "imu",
            "camera",
            "ops",
            "health",
        }
        self._sensor_event_aggregation_windows: dict[str, SensorAggregationWindow] = {}
        self._sensor_event_aggregation_tasks: dict[str, asyncio.Task[None]] = {}
        self._sensor_event_aggregation_lock = asyncio.Lock()
        self._sensor_event_aggregation_metrics: dict[str, int] = {
            "dropped": 0,
            "coalesced": 0,
            "immediate": 0,
        }
        battery_cfg = config.get("battery") or {}
        battery_response_cfg = battery_cfg.get("response") or {}
        self._battery_response_enabled = bool(battery_response_cfg.get("enabled", True))
        self._battery_response_allow_warning = bool(
            battery_response_cfg.get("allow_warning", True)
        )
        self._battery_response_allow_critical = bool(
            battery_response_cfg.get("allow_critical", True)
        )
        self._battery_response_require_transition = bool(
            battery_response_cfg.get("require_transition", False)
        )
        self._battery_query_context_window_s = float(
            battery_response_cfg.get("query_context_window_s", 45.0)
        )
        self._image_response_mode = str(config.get("image_response_mode", "respond")).lower()
        self._image_response_enabled = self._image_response_mode != "catalog_only"
        self._reflection_enabled = bool(config.get("reflection_enabled", False))
        self._reflection_min_interval_s = float(config.get("reflection_min_interval_s", 300.0))
        memory_cfg = config.get("memory") or {}
        self._auto_memory_enabled = bool(memory_cfg.get("auto_memory_enabled", False))
        self._require_confirmation_for_auto_memory = bool(
            memory_cfg.get("require_confirmation_for_auto_memory", False)
        )
        self._auto_memory_min_confidence = float(memory_cfg.get("auto_memory_min_confidence", 0.75))
        ops_cfg = config.get("ops") or {}
        budgets_cfg = ops_cfg.get("budgets") or {}
        self._ai_call_budget = RollingWindowBudget(
            int(budgets_cfg.get("ai_calls_per_minute", 0)),
            60.0,
            name="ai_calls",
        )
        self._alert_policy = AlertPolicy.from_config(config)
        governance_cfg = config.get("governance") or {}
        _validate_tool_specs(governance_cfg.get("tool_specs") or {}, tools)
        tool_specs = build_tool_specs(
            governance_cfg.get("tool_specs") or {},
            registered_tool_names=(tool.get("name") for tool in tools),
        )
        self._governance = GovernanceLayer(tool_specs, config)
        self._debug_governance_decisions = bool(governance_cfg.get("debug_decisions", False))
        self._pending_action: PendingAction | None = None
        self._confirmation_state = ConfirmationState.IDLE
        self._pending_confirmation_token: PendingConfirmationToken | None = None
        self._awaiting_confirmation_completion = False
        self._approval_timeout_s = float(config.get("approval_timeout_s", 30.0))
        realtime_cfg = config.get("realtime") or {}
        rate_limits_cfg = realtime_cfg.get("rate_limits") or {}
        self._rate_limits_strict = bool(rate_limits_cfg.get("strict", False))
        self._rate_limits_regression_warning_threshold = max(
            1,
            int(rate_limits_cfg.get("regression_warning_threshold", 3)),
        )
        self._rate_limits_debug_samples = bool(rate_limits_cfg.get("debug_samples", False))
        self._confirmation_awaiting_decision_timeout_s = float(
            realtime_cfg.get("confirmation_awaiting_decision_timeout_s", 20.0)
        )
        self._research_permission_awaiting_decision_timeout_s = float(
            (config.get("research") or {}).get("permission_awaiting_decision_timeout_s", 60.0)
        )
        self._confirmation_late_decision_grace_s = float(
            realtime_cfg.get("confirmation_late_decision_grace_s", 15.0)
        )
        self._confirmation_decision_expiry_grace_s = max(
            0.0,
            float(realtime_cfg.get("confirmation_decision_expiry_grace_s", 1.5)),
        )
        self._confirmation_unclear_max_reprompts = max(
            0,
            int(realtime_cfg.get("confirmation_unclear_max_reprompts", 1)),
        )
        self._confirmation_reminder_interval_s = max(
            0.0,
            float(realtime_cfg.get("confirmation_reminder_interval_s", 6.0)),
        )
        self._confirmation_reminder_max_count = max(
            0,
            int(realtime_cfg.get("confirmation_reminder_max_count", 2)),
        )
        self._confirmation_reminder_tracker: dict[str, dict[str, Any]] = {}
        self._confirmation_token_created_at: float | None = None
        self._confirmation_last_activity_at: float | None = None
        self._confirmation_speech_active = False
        self._confirmation_asr_pending = False
        self._confirmation_pause_started_at: float | None = None
        self._confirmation_paused_accum_s = 0.0
        self._confirmation_timeout_check_log_interval_s = float(
            realtime_cfg.get("confirmation_timeout_check_log_interval_s", 1.0)
        )
        self._confirmation_timeout_check_last_logged_at: dict[str, float] = {}
        self._confirmation_timeout_check_last_pause_reason: dict[str, str] = {}
        self._confirmation_coordinator = ConfirmationCoordinator(
            reminder_interval_s=self._confirmation_reminder_interval_s,
            reminder_max_count=self._confirmation_reminder_max_count,
            awaiting_decision_timeout_s=self._confirmation_awaiting_decision_timeout_s,
            research_permission_timeout_s=self._research_permission_awaiting_decision_timeout_s,
            timeout_check_log_interval_s=self._confirmation_timeout_check_log_interval_s,
            on_transition=self._log_confirmation_transition,
            on_timeout_check=self._log_confirmation_timeout_check,
        )
        self._confirmation_last_closed_token: dict[str, Any] | None = None
        self._confirmation_timeout_debounce_window_s = float(
            config.get("confirmation_timeout_debounce_window_s", 5.0)
        )
        self._confirmation_timeout_markers: dict[str, float] = {}
        self._confirmation_timeout_causes: dict[str, str] = {}
        self._recent_confirmation_outcomes: dict[str, dict[str, Any]] = {}
        self._confirmation_transition_lock: asyncio.Lock | None = None
        self._confirmation_service = ConfirmationService(
            awaiting_timeout_s=self._confirmation_awaiting_decision_timeout_s,
            late_decision_grace_s=self._confirmation_late_decision_grace_s,
        )
        self._confirmation_runtime = ConfirmationRuntime(self)
        self._research_runtime = ResearchRuntime(self)
        self._memory_runtime = MemoryRuntime(self)
        self._response_create_runtime = ResponseCreateRuntime(self)
        self._websocket_close_timeout_s = float(config.get("websocket_close_timeout_s", 5.0))
        self._tool_definitions = {tool["name"]: tool for tool in tools}
        self._storage = StorageController.get_instance()
        self._presented_actions: set[str] = set()
        self._stop_words = [
            word.strip().lower()
            for word in (config.get("stop_words") or [])
            if isinstance(word, str) and word.strip()
        ]
        self._stop_word_cooldown_s = float(config.get("stop_word_cooldown_s", 0.0))
        logging_cfg = config.get("logging") or {}
        user_transcripts_cfg = logging_cfg.get("user_transcripts") or {}
        partials_cfg = user_transcripts_cfg.get("partials") or {}
        redact_cfg = user_transcripts_cfg.get("redact") or {}
        self._log_user_transcripts_enabled = bool(user_transcripts_cfg.get("enabled", False))
        self._log_user_transcript_partials_enabled = bool(partials_cfg.get("enabled", False))
        self._log_user_transcript_partials_min_chars_delta = max(
            1, int(partials_cfg.get("min_chars_delta", 8))
        )
        self._log_user_transcript_redact_enabled = bool(redact_cfg.get("enabled", True))
        self._last_logged_partial_user_transcript = ""
        memory_retrieval_cfg = config.get("memory_retrieval") or {}
        self._memory_retrieval_enabled = bool(memory_retrieval_cfg.get("enabled", True))
        self._memory_retrieval_max_memories = int(memory_retrieval_cfg.get("max_memories", 3))
        self._memory_retrieval_max_chars = int(memory_retrieval_cfg.get("max_chars", 450))
        self._memory_retrieval_cooldown_s = float(memory_retrieval_cfg.get("cooldown_s", 10.0))
        self._memory_retrieval_scope = str(memory_retrieval_cfg.get("scope", MemoryScope.USER_GLOBAL.value))
        self._memory_retrieval_min_user_chars = int(memory_retrieval_cfg.get("min_user_chars", 12))
        self._memory_retrieval_min_user_tokens = int(memory_retrieval_cfg.get("min_user_tokens", 3))
        memory_hydration_cfg = config.get("memory_hydration") or {}
        self._startup_memory_digest_enabled = bool(memory_hydration_cfg.get("startup_digest_enabled", True))
        self._startup_memory_digest_max_items = int(memory_hydration_cfg.get("startup_digest_max_items", 2))
        self._startup_memory_digest_max_chars = int(memory_hydration_cfg.get("startup_digest_max_chars", 280))
        self._tool_execution_disabled_until = 0.0
        self._reflection_coordinator = ReflectionCoordinator(
            api_key=self.api_key,
            enabled=self._reflection_enabled,
            min_interval_s=self._reflection_min_interval_s,
            storage=self._storage,
            budget=self._ai_call_budget,
        )
        self._reflection_enqueued = False
        self._response_done_reflection_task: asyncio.Task | None = None
        self._stimuli_coordinator = StimuliCoordinator(
            debounce_window_s=float(config.get("injection_debounce_window_s", 0.4)),
            cooldown_s=self._injection_response_cooldown_s,
            emit_callback=self.maybe_request_response,
        )
        self._gesture_cooldowns_s = {
            "gesture_attention_snap": 10.0,
            "gesture_curious_tilt": 6.0,
            "gesture_nod": 8.0,
            "gesture_idle": 8.0,
        }
        self.orchestration_state = OrchestrationState()
        self.event_bus = EventBus()
        self._event_injector = EventInjector(
            self.event_bus,
            ready_event=self.ready_event,
            is_ready=self.is_ready_for_injections,
            inject_callback=self.inject_event,
        )
        self._session_connection_attempts = 0
        self._session_connections = 0
        self._session_reconnects = 0
        self._session_failures = 0
        self._session_connected = False
        self._last_connect_time: float | None = None
        self._last_disconnect_reason: str | None = None
        self._last_failure_reason: str | None = None
        self._memory_retrieval_error_throttle_s = 60.0
        self._memory_retrieval_last_error_log_at = 0.0
        self._memory_retrieval_suppressed_errors = 0
        preference_recall_cfg = memory_retrieval_cfg.get("preference_recall") or {}
        self._preference_recall_cooldown_s = float(preference_recall_cfg.get("cooldown_s", self._memory_retrieval_cooldown_s))
        self._preference_recall_max_attempts = max(1, int(preference_recall_cfg.get("max_attempts", 1)))
        self._preference_recall_min_semantic_score = float(preference_recall_cfg.get("min_semantic_score", 0.15))
        self._preference_recall_min_lexical_score = float(preference_recall_cfg.get("min_lexical_score", 0.1))
        self._preference_recall_allow_low_score_debug = bool(preference_recall_cfg.get("allow_low_score_debug", False))
        self._preference_recall_cache: dict[str, dict[str, Any]] = {}
        self._pending_preference_recall_trace: dict[str, Any] | None = None
        self._preference_recall_skip_logged_turn_ids: set[str] = set()
        self._preference_recall_handled_logged_turn_ids: set[str] = set()
        self._response_schedule_logged_turn_ids: set[str] = set()
        self._turn_diagnostic_timestamps: dict[str, dict[str, float]] = {}

        research_cfg = config.get("research") or {}
        self._research_enabled = bool(research_cfg.get("enabled", False))
        self._research_permission_required = bool(research_cfg.get("permission_required", True))
        self._research_mode = str(research_cfg.get("research_mode", "")).strip().lower()
        if self._research_mode not in {"auto", "ask", "disabled", "ask_on_assistant_or_unknown"}:
            self._research_mode = (
                "ask_on_assistant_or_unknown" if self._research_permission_required else "auto"
            )
        self._research_provider = str(research_cfg.get("provider", "null")).strip().lower()
        firecrawl_cfg = research_cfg.get("firecrawl") or {}
        self._research_firecrawl_enabled = bool(firecrawl_cfg.get("enabled", False))
        self._research_firecrawl_allowlist_mode = str(
            firecrawl_cfg.get("allowlist_mode", "public")
        ).strip().lower()
        self._research_firecrawl_allowlist_domains = {
            domain.strip().lower()
            for domain in (firecrawl_cfg.get("allowlist_domains") or [])
            if isinstance(domain, str) and domain.strip()
        }
        self._trusted_domain_store = TrustedDomainStore()
        self._trusted_research_domains = self._trusted_domain_store.get_domain_set()
        self._research_firecrawl_allowlist_domains = merge_allowlists(
            self._research_firecrawl_allowlist_domains,
            self._trusted_research_domains,
        )
        self._research_service = build_openai_service_or_null(config)
        self._pending_research_request: ResearchRequest | None = None
        self._prior_research_permission_marker: dict[str, Any] | None = None
        self._prior_research_permission_grace_s = float(
            research_cfg.get("prior_permission_grace_s", 8.0)
        )
        self._research_permission_outcome_ttl_s = float(
            research_cfg.get("permission_outcome_ttl_s", 20.0)
        )
        self._research_permission_outcomes: dict[str, dict[str, Any]] = {}
        self._research_suppressed_fingerprints: dict[str, str] = {}
        self._research_pending_call_ids: set[str] = set()
        self._deferred_research_tool_call: dict[str, Any] | None = None
        self._tool_call_dedupe_ttl_s = float(research_cfg.get("tool_call_dedupe_ttl_s", 30.0))
        self._research_spoken_response_dedupe_ttl_s = float(
            research_cfg.get("spoken_response_dedupe_ttl_s", 60.0)
        )
        self._spoken_research_response_ids: dict[str, float] = {}
        self._response_create_debug_trace = bool(
            research_cfg.get("debug_response_create_trace", False)
        )
        self._last_response_create_ts: float | None = None
        self._response_done_serial = 0
        self._active_response_id: str | None = None
        self._active_response_confirmation_guarded = False
        self._active_response_preference_guarded = False
        self._active_response_origin = "unknown"
        self._active_response_consumes_canonical_slot = True
        self._preference_recall_suppressed_turns: set[str] = set()
        self._preference_recall_suppressed_input_event_keys: set[str] = set()
        self._preference_recall_locked_input_event_keys: set[str] = set()
        self._preference_recall_followup_enabled = bool(preference_recall_cfg.get("followup_enabled", False))
        self._pending_preference_memory_context_by_canonical_key: dict[str, dict[str, Any]] = {}
        self._pending_preference_memory_context_by_turn_id: dict[str, dict[str, Any]] = {}
        self._pending_preference_memory_context_by_response_id: dict[str, dict[str, Any]] = {}
        self._pending_preference_memory_context_response_id_by_turn_id: dict[str, str] = {}
        self._pending_server_auto_input_event_keys: deque[str] = deque(maxlen=64)
        self._pending_server_auto_response_by_response_id: dict[str, PendingServerAutoResponse] = {}
        self._pending_server_auto_response_id_by_turn_id: dict[str, str] = {}
        # Legacy mirror kept for compatibility with tests and older call sites.
        self._pending_server_auto_response_by_turn_id: dict[str, PendingServerAutoResponse] = {}
        self._cancelled_response_ids: set[str] = set()
        self._suppressed_audio_response_ids: set[str] = set()
        self._cancelled_deliverable_logged_ids: set[str] = set()
        self._cancelled_response_timing_by_id: dict[str, dict[str, Any]] = {}
        self._response_status_by_id: dict[str, str] = {}
        self._provisional_response_ids: set[str] = set()
        self._provisional_response_ids_completed_empty: set[str] = set()
        self._stale_response_drop_window_by_id: dict[str, dict[str, Any]] = {}
        self._stale_response_drop_window_s = 3.0
        self._stale_response_map: dict[str, dict[str, Any]] = {}
        self._stale_response_map_ttl_s = 15.0
        self._active_server_auto_input_event_key: str | None = None
        self._current_input_event_key: str | None = None
        self._active_input_event_key_by_turn_id: dict[str, str] = {}
        self._input_event_key_counter = 0
        self._synthetic_input_event_counter = 0
        self._response_created_canonical_keys: set[str] = set()
        self._empty_response_retry_canonical_keys: set[str] = set()
        self._empty_response_retry_counts: dict[str, int] = {}
        self._empty_response_retry_fallback_emitted: set[str] = set()
        self._empty_response_retry_max_attempts = 2
        self._response_delivery_ledger: dict[str, str] = {}
        self._response_id_by_canonical_key: dict[str, str] = {}
        self._canonical_response_state_by_key: dict[str, CanonicalResponseState] = {}
        self._canonical_response_lifecycle_state: dict[str, dict[str, bool]] = {}
        self._lifecycle_canonical_timeline: dict[str, deque[str]] = {}
        self._interaction_lifecycle_controller = InteractionLifecycleController()
        self._interaction_lifecycle_policy = InteractionLifecyclePolicy()
        self._response_obligations: dict[str, dict[str, Any]] = {}
        self._canonical_invariant_logged: set[str] = set()
        self._already_scheduled_for_input_event_key: set[str] = set()
        self._tool_followup_state_by_canonical_key: dict[str, str] = {}
        self._active_response_input_event_key: str | None = None
        self._active_response_canonical_key: str | None = None
        self._response_gating_verdict_by_input_event_key: dict[str, ResponseGatingVerdict] = {}
        self._latest_partial_transcript_by_turn_id: dict[str, str] = {}
        self._server_auto_audio_waiters_by_turn_id: dict[str, asyncio.Event] = {}
        self._server_auto_audio_defer_tasks_by_turn_id: dict[str, asyncio.Task[Any]] = {}
        self._server_auto_pre_audio_hold_by_turn_id: dict[str, ServerAutoPreAudioHoldRecord] = {}
        self._server_auto_pre_audio_hold_phase_by_key: dict[tuple[str, str], str] = {}
        self._audio_response_started_ids: set[str] = set()
        self._server_auto_audio_deferral_timeout_ms = max(
            0,
            int((config.get("realtime") or {}).get("server_auto_audio_deferral_timeout_ms", 225)),
        )
        self._last_vision_frame_sent_at_monotonic: float | None = None
        self._suppressed_topics: set[str] = set()
        self._battery_redline_percent = float(battery_response_cfg.get("redline_percent", 10.0))
        self._load_topic_suppression_preferences()
        self._pending_alternate_intent_override: dict[str, Any] | None = None
        self._last_executed_tool_call: dict[str, Any] | None = None
        self._intent_ledger: dict[str, IntentLedgerEntry] = {}
        self._intent_state_ttl_s = float(research_cfg.get("intent_state_ttl_s", 300.0))
        self._intent_execution_cooldown_s = float(
            research_cfg.get("intent_execution_cooldown_s", self._tool_call_dedupe_ttl_s)
        )
        self._intent_denial_cooldown_s = float(
            research_cfg.get("intent_denial_cooldown_s", 45.0)
        )

        realtime_cfg = config.get("realtime") or {}
        self._stale_response_map_ttl_s = max(
            1.0,
            float(realtime_cfg.get("stale_response_map_ttl_s", self._stale_response_map_ttl_s)),
        )
        self._debug_vad = bool(realtime_cfg.get("debug_vad", False))
        self._awaiting_confirmation_allowed_sources = (
            self._load_awaiting_confirmation_source_policy(realtime_cfg)
        )
        self._startup_injection_gate_timeout_s = max(
            0.0,
            float(realtime_cfg.get("startup_injection_gate_timeout_s", 5.0)),
        )
        self._startup_injections = InjectionCoordinator(
            gate_timeout_s=self._startup_injection_gate_timeout_s,
            loop_getter=lambda: self.loop,
            emit_injected_event=self._emit_injected_event,
            emit_system_context_payload=self._emit_system_context_payload,
        )
        self._battery_injection_policy = BatteryInjectionPolicy(self)
        self._injection_bus = InjectionBus(self, self._startup_injections)
        self._utterance_counter = 0
        self._active_utterance: dict[str, Any] | None = None
        self._utterance_info_summary: dict[str, bool] = {}
        self._utterance_info_summary_emitted = False
        self._recent_input_levels: deque[dict[str, float]] = deque(maxlen=256)
        self._transcript_response_watchdog_timeout_s = max(
            0.5,
            float(realtime_cfg.get("transcript_response_watchdog_timeout_s", 3.0)),
        )
        self._transcript_response_watchdog_tasks: dict[str, asyncio.Task] = {}
        self._transcript_response_outcome_logged_keys: set[str] = set()
        self._lifecycle_trace_transcript_delta_sample_n = max(
            1,
            int(realtime_cfg.get("lifecycle_trace_transcript_delta_sample_n", 20)),
        )
        self._lifecycle_trace_transcript_delta_inactivity_ms = max(
            250,
            int(realtime_cfg.get("lifecycle_trace_transcript_delta_inactivity_ms", 750)),
        )
        self._lifecycle_trace_item_added_unknown_threshold = max(
            1,
            int(realtime_cfg.get("lifecycle_trace_item_added_unknown_threshold", 3)),
        )
        self._lifecycle_trace_item_added_unknown_window_s = max(
            1.0,
            float(realtime_cfg.get("lifecycle_trace_item_added_unknown_window_s", 10.0)),
        )
        self._lifecycle_trace_item_added_unknown_cooldown_s = max(
            1.0,
            float(realtime_cfg.get("lifecycle_trace_item_added_unknown_cooldown_s", 30.0)),
        )
        self._lifecycle_trace_item_added_unknown_debug = bool(
            realtime_cfg.get("lifecycle_trace_item_added_unknown_debug", False)
        )
        self._lifecycle_trace_transcript_delta_state: dict[str, dict[str, float | int]] = {}
        self._lifecycle_trace_item_added_unknown_events: deque[dict[str, Any]] = deque()
        self._lifecycle_trace_item_added_unknown_last_escalation_ts = 0.0
        self._response_id_by_output_item_id: dict[str, str] = {}
        self._minimum_non_confirmation_duration_ms = int(
            realtime_cfg.get("minimum_non_confirmation_duration_ms", 120)
        )
        asr_cfg = config.get("asr") or {}
        verify_cfg = asr_cfg.get("verify_on_risk") or {}
        self._asr_verify_on_risk_enabled = bool(verify_cfg.get("enabled", True))
        self._asr_verify_min_confidence = float(verify_cfg.get("min_confidence", 0.65))
        self._asr_verify_short_utterance_ms = int(verify_cfg.get("short_utterance_ms", 450))
        self._asr_verify_max_clarify_per_turn = max(1, int(verify_cfg.get("max_clarify_per_turn", 1)))
        self._asr_clarify_count_by_turn: dict[str, int] = {}
        self._asr_clarify_asked_input_event_keys: set[str] = set()
        self._utterance_trust_snapshot_by_input_event_key: dict[str, dict[str, Any]] = {}
        self._micro_ack_channel_mode = str(realtime_cfg.get("micro_ack_channel_mode", "text_and_audio")).strip().lower()
        if self._micro_ack_channel_mode not in {"text_only", "text_and_audio"}:
            self._micro_ack_channel_mode = "text_and_audio"
        self._micro_ack_runtime_mode = str(
            realtime_cfg.get("micro_ack_runtime_mode", realtime_cfg.get("micro_ack_quiet_mode", "normal"))
        ).strip().lower()
        if self._micro_ack_runtime_mode not in {"normal", "quiet", "verbose"}:
            self._micro_ack_runtime_mode = "normal"
        self._micro_ack_channel_policy = self._load_micro_ack_channel_policy(realtime_cfg)
        self._vad_turn_detection = self._resolve_vad_turn_detection(config)
        self._micro_ack_suppress_until_ts = 0.0
        self._micro_ack_near_ready_suppress_ms = max(
            0,
            int(realtime_cfg.get("micro_ack_near_ready_suppress_ms", 0)),
        )
        self._pending_micro_ack_by_turn_channel: dict[tuple[str, str], PendingMicroAckMarker] = {}
        self._micro_ack_manager = self._build_micro_ack_manager(realtime_cfg)
        self._conversation_efficiency_by_turn: dict[str, ConversationEfficiencyState] = {}
        self._conversation_efficiency_logged_turns: set[str] = set()
        self._silent_turn_incident_count = 0
        self._response_lifecycle = ResponseLifecycleTracker(self)
        self._response_terminal_handlers = ResponseTerminalHandlers(self)
        self._input_audio_events = InputAudioEventHandlers(self)
        self._transport: RealtimeTransport | None = None
        self._event_router = EventRouter(
            fallback=self._handle_unknown_event,
            on_exception=self._on_event_handler_exception,
        )
        self._function_call_accumulator = FunctionCallAccumulator(
            on_function_call_item=self._on_function_call_item_added,
            on_assistant_message_item=self._on_assistant_output_item_added,
            on_arguments_done=self.handle_function_call,
        )
        self._configure_event_router()

    def _configure_event_router(self) -> None:
        self._event_router.register("response.created", self._handle_response_created_event)
        self._event_router.register("response.output_audio.delta", self._handle_response_output_audio_delta_event)
        self._event_router.register("response.done", self._handle_response_done_event)
        self._event_router.register(
            "conversation.item.input_audio_transcription.completed",
            self._handle_input_audio_transcription_completed_event,
        )
        self._event_router.register(
            "conversation.item.input_audio_transcription.failed",
            self._handle_input_audio_transcription_failed_event,
        )
        for event_type in {
            "conversation.item.added",
            "response.text.delta",
            "response.output_audio.done",
            "response.output_audio_transcript.delta",
            "response.output_audio_transcript.done",
            "response.completed",
        }:
            self._event_router.register(event_type, self._handle_response_lifecycle_event)
        self._event_router.register(
            "input_audio_buffer.speech_started",
            self._input_audio_events.handle_input_audio_buffer_speech_started,
        )
        self._event_router.register(
            "input_audio_buffer.speech_stopped",
            self._input_audio_events.handle_input_audio_buffer_speech_stopped,
        )
        self._event_router.register(
            "input_audio_buffer.committed",
            self._input_audio_events.handle_input_audio_buffer_committed,
        )
        self._event_router.register(
            "conversation.item.input_audio_transcription.delta",
            self._input_audio_events.handle_input_audio_transcription_partial,
        )
        self._event_router.register(
            "conversation.item.input_audio_transcription.partial",
            self._input_audio_events.handle_input_audio_transcription_partial,
        )
        self._event_router.register("response.output_item.added", self._handle_output_item_added_event)
        self._event_router.register(
            "response.function_call_arguments.delta",
            self._handle_function_call_arguments_delta_event,
        )
        self._event_router.register(
            "response.function_call_arguments.done",
            self._handle_function_call_arguments_done_event,
        )
        for event_type in {"session.updated", "rate_limits.updated"}:
            self._event_router.register(event_type, self._handle_session_or_rate_limit_event)
        for event_type in {
            "error",
            "conversation.item.created",
            "response.audio_transcript.done",
            "response.output_text.delta",
            "response.output_text.done",
            "response.text.done",
        }:
            self._event_router.register(event_type, self._handle_error_or_system_event)

    def _on_event_handler_exception(self, event_type: str, exc: Exception) -> None:
        logger.debug("event_handler_exception event=%s error=%s", event_type, exc)

    async def _handle_unknown_event(self, event: dict[str, Any], websocket: Any) -> None:
        _ = websocket
        logger.debug("Unhandled realtime event type=%s", str(event.get("type") or "unknown"))

    def _response_trace_by_id(self) -> dict[str, dict[str, str]]:
        trace_by_id = getattr(self, "_response_trace_context_by_id", None)
        if isinstance(trace_by_id, dict):
            return trace_by_id
        trace_by_id = {}
        self._response_trace_context_by_id = trace_by_id
        return trace_by_id

    def _record_response_trace_context(self, response_id: str, **fields: str) -> dict[str, str]:
        response_key = str(response_id or "").strip()
        if not response_key:
            return {}
        trace_by_id = self._response_trace_by_id()
        current = trace_by_id.setdefault(response_key, {})
        for key, value in fields.items():
            normalized_value = str(value or "").strip()
            if normalized_value:
                current[key] = normalized_value
        return current

    def _response_item_id_map(self) -> dict[str, str]:
        mapping = getattr(self, "_response_id_by_output_item_id", None)
        if isinstance(mapping, dict):
            return mapping
        mapping = {}
        self._response_id_by_output_item_id = mapping
        return mapping

    def _remember_output_item_response_id(self, *, item_id: str, response_id: str) -> None:
        normalized_item_id = str(item_id or "").strip()
        normalized_response_id = str(response_id or "").strip()
        if not normalized_item_id or not normalized_response_id:
            return
        mapping = self._response_item_id_map()
        mapping[normalized_item_id] = normalized_response_id
        if len(mapping) <= 2048:
            return
        stale_item_ids = list(mapping.keys())[: len(mapping) - 2048]
        for stale_item_id in stale_item_ids:
            mapping.pop(stale_item_id, None)

    def _response_id_for_output_item(self, item_id: str) -> str:
        normalized_item_id = str(item_id or "").strip()
        if not normalized_item_id:
            return ""
        return str(self._response_item_id_map().get(normalized_item_id) or "").strip()

    def _classify_conversation_item_added_category(self, *, item_type: str, item_role: str) -> str:
        normalized_type = str(item_type or "").strip().lower()
        normalized_role = str(item_role or "").strip().lower()
        if normalized_type == "message" and normalized_role == "user":
            return "user_message"
        if normalized_type in {"function_call_output", "tool_result"}:
            return "tool_result"
        if normalized_type == "message" and normalized_role == "system":
            return "system_injection"
        return "unknown"

    def _emit_response_lifecycle_trace(
        self,
        *,
        event_type: str,
        response_id: str,
        turn_id: str,
        input_event_key: str,
        canonical_key: str,
        origin: str,
        active_input_event_key: str,
        active_canonical_key: str,
        payload_summary: str = "",
        item_id: str = "",
        item_type: str = "",
        item_role: str = "",
    ) -> None:
        now = time.monotonic()
        response_key = str(response_id or "").strip() or "unknown"
        resolved_turn_id = str(turn_id or "").strip() or "unknown"
        resolved_input_key = str(input_event_key or "").strip() or "unknown"
        resolved_canonical_key = str(canonical_key or "").strip() or "unknown"
        resolved_origin = str(origin or "").strip() or "unknown"
        active_input_key = str(active_input_event_key or "").strip() or "none"
        active_key = str(active_canonical_key or "").strip() or "none"
        normalized_event_type = str(event_type or "").strip() or "unknown"
        trace_flags: list[str] = []
        if normalized_event_type == "response.output_audio_transcript.delta":
            self._prune_lifecycle_trace_transcript_delta_state(now=now)
            transcript_state = self._lifecycle_trace_transcript_delta_state.setdefault(
                response_key,
                {"seq": 0, "last_seen": now},
            )
            seq = int(transcript_state.get("seq", 0)) + 1
            transcript_state["seq"] = seq
            transcript_state["last_seen"] = now
            is_first = seq == 1
            sampled = (seq % self._lifecycle_trace_transcript_delta_sample_n) == 0
            if is_first or sampled:
                trace_flags.append(
                    f"seq={seq} sampled={str(sampled).lower()} first={str(is_first).lower()} last=false"
                )
                logger.debug(
                    "response_lifecycle_trace response_id=%s event_type=%s turn_id=%s input_event_key=%s canonical_key=%s "
                    "origin=%s active_input_event_key=%s active_canonical_key=%s %s",
                    response_key,
                    normalized_event_type,
                    resolved_turn_id,
                    resolved_input_key,
                    resolved_canonical_key,
                    resolved_origin,
                    active_input_key,
                    active_key,
                    trace_flags[-1],
                )
        elif normalized_event_type == "conversation.item.added":
            resolved_item_id = str(item_id or "").strip() or "unknown"
            resolved_item_type = str(item_type or "").strip() or "unknown"
            resolved_item_role = str(item_role or "").strip() or "unknown"
            logger.debug(
                "response_lifecycle_trace response_id=%s event_type=%s turn_id=%s input_event_key=%s canonical_key=%s "
                "origin=%s active_input_event_key=%s active_canonical_key=%s item_id=%s item_type=%s item_role=%s",
                response_key,
                normalized_event_type,
                resolved_turn_id,
                resolved_input_key,
                resolved_canonical_key,
                resolved_origin,
                active_input_key,
                active_key,
                resolved_item_id,
                resolved_item_type,
                resolved_item_role,
            )
            if response_key == "unknown":
                category = self._classify_conversation_item_added_category(
                    item_type=resolved_item_type,
                    item_role=resolved_item_role,
                )
                if category != "unknown":
                    logger.debug(
                        "response_lifecycle_trace_item_added_non_response category=%s item_id=%s item_type=%s item_role=%s",
                        category,
                        resolved_item_id,
                        resolved_item_type,
                        resolved_item_role,
                    )
                else:
                    self._maybe_escalate_unknown_item_added(
                        now=now,
                        item_id=resolved_item_id,
                        item_type=resolved_item_type,
                        item_role=resolved_item_role,
                    )
        else:
            logger.info(
                "response_lifecycle_trace response_id=%s event_type=%s turn_id=%s input_event_key=%s canonical_key=%s "
                "origin=%s active_input_event_key=%s active_canonical_key=%s",
                response_key,
                normalized_event_type,
                resolved_turn_id,
                resolved_input_key,
                resolved_canonical_key,
                resolved_origin,
                active_input_key,
                active_key,
            )
        if payload_summary:
            logger.debug(
                "response_lifecycle_trace_detail response_id=%s event_type=%s turn_id=%s input_event_key=%s "
                "canonical_key=%s origin=%s active_input_event_key=%s active_canonical_key=%s payload=%s",
                response_key,
                normalized_event_type,
                resolved_turn_id,
                resolved_input_key,
                resolved_canonical_key,
                resolved_origin,
                active_input_key,
                active_key,
                payload_summary,
            )

    def _prune_lifecycle_trace_transcript_delta_state(self, *, now: float) -> None:
        state = self._lifecycle_trace_transcript_delta_state
        if not state:
            return
        inactivity_s = self._lifecycle_trace_transcript_delta_inactivity_ms / 1000.0
        stale_response_ids = [
            response_id
            for response_id, entry in state.items()
            if now - float(entry.get("last_seen", now)) > inactivity_s
        ]
        for response_id in stale_response_ids:
            state.pop(response_id, None)
        if len(state) <= 256:
            return
        oldest = sorted(
            state.items(),
            key=lambda pair: float(pair[1].get("last_seen", now)),
        )
        for response_id, _ in oldest[: len(state) - 256]:
            state.pop(response_id, None)

    def _maybe_escalate_unknown_item_added(self, *, now: float, item_id: str, item_type: str, item_role: str) -> None:
        events = self._lifecycle_trace_item_added_unknown_events
        window_s = self._lifecycle_trace_item_added_unknown_window_s
        while events and now - float((events[0] or {}).get("ts", now)) > window_s:
            events.popleft()
        events.append(
            {
                "ts": now,
                "item_id": str(item_id or "").strip() or "unknown",
                "item_type": str(item_type or "").strip() or "unknown",
                "item_role": str(item_role or "").strip() or "unknown",
            }
        )
        if len(events) < self._lifecycle_trace_item_added_unknown_threshold:
            return
        last_escalation = self._lifecycle_trace_item_added_unknown_last_escalation_ts
        if last_escalation > 0.0 and now - last_escalation < self._lifecycle_trace_item_added_unknown_cooldown_s:
            return
        self._lifecycle_trace_item_added_unknown_last_escalation_ts = now
        logger.info(
            "response_lifecycle_trace_unknown_item_added_spike event_type=conversation.item.added response_id=unknown "
            "count=%s window_s=%s cooldown_s=%s",
            len(events),
            self._lifecycle_trace_item_added_unknown_window_s,
            self._lifecycle_trace_item_added_unknown_cooldown_s,
        )
        if bool(getattr(self, "_lifecycle_trace_item_added_unknown_debug", False)):
            details = [
                "%s:%s:%s" % (
                    str(entry.get("item_id") or "unknown"),
                    str(entry.get("item_type") or "unknown"),
                    str(entry.get("item_role") or "unknown"),
                )
                for entry in list(events)
            ]
            logger.debug(
                "response_lifecycle_trace_unknown_item_added_spike_items count=%s items=%s",
                len(events),
                ",".join(details),
            )

    async def _handle_response_lifecycle_event(self, event: dict[str, Any], websocket: Any) -> None:
        event_type = str(event.get("type") or "unknown")
        response_id = self._response_id_from_event(event)
        if not self._should_process_response_event_ingress(event, source="lifecycle"):
            return
        if not response_id:
            response_id = str(getattr(self, "_active_response_id", "") or "").strip()
        trace_context = self._response_trace_by_id().get(response_id, {}) if response_id else {}
        stale_context = self._stale_response_context(response_id) if response_id else {}
        active_turn_id = str(self._current_turn_id_or_unknown() or "").strip() or "unknown"
        active_input_event_key = str(getattr(self, "_active_response_input_event_key", "") or "").strip()
        active_canonical_key = str(getattr(self, "_active_response_canonical_key", "") or "").strip()
        turn_id = str(
            trace_context.get("turn_id")
            or stale_context.get("turn_id")
            or active_turn_id
            or "unknown"
        ).strip() or "unknown"
        input_event_key = str(
            trace_context.get("input_event_key")
            or stale_context.get("input_event_key")
            or active_input_event_key
            or "unknown"
        ).strip() or "unknown"
        canonical_key = str(
            trace_context.get("canonical_key")
            or stale_context.get("canonical_key")
            or active_canonical_key
            or self._canonical_utterance_key(turn_id=turn_id, input_event_key=input_event_key)
        ).strip() or "unknown"
        origin = str(
            trace_context.get("origin")
            or stale_context.get("origin")
            or getattr(self, "_active_response_origin", "unknown")
            or "unknown"
        ).strip() or "unknown"
        item = event.get("item")
        item_type = str(item.get("type") or "") if isinstance(item, dict) else ""
        item_role = str(item.get("role") or "") if isinstance(item, dict) else ""
        item_id = str(item.get("id") or "") if isinstance(item, dict) else ""
        if event_type == "response.output_item.added" and response_id and item_id:
            self._remember_output_item_response_id(item_id=item_id, response_id=response_id)
        if event_type == "conversation.item.added" and not response_id and item_id:
            mapped_response_id = self._response_id_for_output_item(item_id)
            if mapped_response_id:
                response_id = mapped_response_id
                trace_context = self._response_trace_by_id().get(response_id, {})
                stale_context = self._stale_response_context(response_id)
        payload_summary = (
            f"has_item={isinstance(item, dict)} item_type={item_type or 'unknown'} item_role={item_role or 'unknown'} "
            f"has_delta={bool('delta' in event)} has_transcript={bool('transcript' in event)}"
        )
        if response_id:
            self._record_response_trace_context(
                response_id,
                turn_id=turn_id,
                input_event_key=input_event_key,
                canonical_key=canonical_key,
                origin=origin,
            )
            if self._is_cancelled_or_superseded_response_id(response_id):
                self._remember_stale_response_context(
                    response_id=response_id,
                    canonical_key=canonical_key,
                    origin=origin,
                    turn_id=turn_id,
                    input_event_key=input_event_key,
                    upgrade_chain_id=self._upgrade_chain_id_from_response(response_id),
                )
        self._emit_response_lifecycle_trace(
            event_type=event_type,
            response_id=response_id,
            turn_id=turn_id,
            input_event_key=input_event_key,
            canonical_key=canonical_key,
            origin=origin,
            active_input_event_key=active_input_event_key,
            active_canonical_key=active_canonical_key,
            payload_summary=payload_summary,
            item_id=item_id,
            item_type=item_type,
            item_role=item_role,
        )
        logger.debug(
            "realtime_lifecycle_event_received event_type=%s has_item=%s has_delta=%s",
            event_type,
            isinstance(event.get("item"), dict),
            "delta" in event,
        )
        await self._handle_event_legacy(event, websocket)

    def _extract_assistant_text_from_content(self, content: Any) -> tuple[str, list[str]]:
        if not isinstance(content, list):
            return "", []
        part_types: list[str] = []
        extracted_parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = str(part.get("type") or "")
            part_types.append(part_type)

            candidate_texts: list[str] = []
            if part_type in {"output_text", "text"}:
                text_value = part.get("text")
                if isinstance(text_value, str) and text_value:
                    candidate_texts.append(text_value)
            transcript_value = part.get("transcript")
            if isinstance(transcript_value, str) and transcript_value:
                candidate_texts.append(transcript_value)

            for candidate in candidate_texts:
                if candidate:
                    extracted_parts.append(candidate)

        return "".join(extracted_parts), part_types

    async def _handle_output_item_added_event(self, event: dict[str, Any], websocket: Any) -> None:
        _ = websocket
        item = event.get("item")
        if isinstance(item, dict):
            item_type = str(item.get("type") or "").strip().lower()
            if item_type in {"function_call", "function_call_output", "tool_call", "tool_result"}:
                self._mark_active_canonical_deliverable_observed(reason=f"response.output_item.added:{item_type}")
        await self._function_call_accumulator.handle_output_item_added(event)

    async def _handle_function_call_arguments_delta_event(self, event: dict[str, Any], websocket: Any) -> None:
        _ = websocket
        self._mark_active_canonical_deliverable_observed(reason="response.function_call_arguments.delta")
        self._function_call_accumulator.handle_function_call_arguments_delta(event)
        self.function_call_args = self._function_call_accumulator.arguments_buffer

    async def _handle_function_call_arguments_done_event(self, event: dict[str, Any], websocket: Any) -> None:
        self._mark_active_canonical_deliverable_observed(reason="response.function_call_arguments.done")
        self.function_call_args = self._function_call_accumulator.arguments_buffer
        await self._function_call_accumulator.handle_function_call_arguments_done(event, websocket)
        self._function_call_accumulator.reset_arguments_buffer()

    def _on_function_call_item_added(self, item: dict[str, Any]) -> None:
        self.function_call = item
        self.function_call_args = ""

    def _on_assistant_output_item_added(self, item: dict[str, Any]) -> None:
        if self._is_active_response_guarded():
            return
        if not self._allow_text_output_state_transition(
            response_id=getattr(self, "_active_response_id", None),
            event_type="response.output_item.added",
        ):
            return
        if item.get("type") != "message" or item.get("role") != "assistant":
            return
        extracted_text, part_types = self._extract_assistant_text_from_content(item.get("content", []))
        logger.debug(
            "assistant_content_event_received event_type=response.output_item.added part_types=%s extracted_chars=%s",
            ",".join(part_types),
            len(extracted_text),
        )
        if not extracted_text:
            return
        self._mark_utterance_info_summary(deliverable_seen=True)
        self._record_active_canonical_deliverable_class(text=extracted_text, reason="response.output_item.added")
        self._cancel_micro_ack(turn_id=self._current_turn_id_or_unknown(), reason="response_started")
        self._mark_first_assistant_utterance_observed_if_needed(extracted_text)
        self._append_assistant_reply_text(extracted_text)
        self.state_manager.update_state(InteractionState.SPEAKING, "text output")
        current_turn_id = self._current_turn_id_or_unknown()
        current_input_event_key = str(getattr(self, "_active_response_input_event_key", "") or "").strip()
        if not current_input_event_key:
            current_input_event_key = str(getattr(self, "_current_input_event_key", "") or "").strip()
        self._set_response_delivery_state(
            turn_id=current_turn_id,
            input_event_key=current_input_event_key,
            state="delivered",
        )

    async def _handle_session_or_rate_limit_event(self, event: dict[str, Any], websocket: Any) -> None:
        await self._handle_event_legacy(event, websocket)

    async def _handle_error_or_system_event(self, event: dict[str, Any], websocket: Any) -> None:
        await self._handle_event_legacy(event, websocket)

    def _runtime_task_registry(self) -> RuntimeTaskRegistry:
        registry = getattr(self, "_runtime_tasks", None)
        if not isinstance(registry, RuntimeTaskRegistry):
            registry = RuntimeTaskRegistry()
            self._runtime_tasks = registry
        return registry

    def _shutdown_coordinator(self) -> ShutdownCoordinator:
        coordinator = getattr(self, "_shutdown", None)
        if not isinstance(coordinator, ShutdownCoordinator):
            coordinator = ShutdownCoordinator(
                close_timeout_s=float(getattr(self, "_websocket_close_timeout_s", 5.0) or 5.0)
            )
            self._shutdown = coordinator
        return coordinator

    def _startup_injection_coordinator(self) -> InjectionCoordinator:
        bus = getattr(self, "_injection_bus", None)
        if isinstance(bus, InjectionBus):
            return bus.coordinator
        coordinator = getattr(self, "_startup_injections", None)
        if not isinstance(coordinator, InjectionCoordinator):
            gate_timeout_s = float(getattr(self, "_startup_injection_gate_timeout_s", 5.0) or 0.0)
            coordinator = InjectionCoordinator(
                gate_timeout_s=gate_timeout_s,
                loop_getter=lambda: getattr(self, "loop", None),
                emit_injected_event=self._emit_injected_event,
                emit_system_context_payload=self._emit_system_context_payload,
            )
            coordinator.released = bool(self.__dict__.get("_startup_injection_gate_released", False))
            coordinator.first_assistant_utterance_observed = bool(
                self.__dict__.get("_startup_first_assistant_utterance_observed", False)
            )
            queue = self.__dict__.get("_startup_injection_queue")
            if isinstance(queue, list):
                coordinator.queue.extend(queue)
            elif isinstance(queue, deque):
                coordinator.queue = queue
            coordinator.timeout_task = self.__dict__.get("_startup_injection_timeout_task")
            self._startup_injections = coordinator
        self._injection_bus = InjectionBus(self, coordinator)
        return coordinator

    def _injection_bus_instance(self) -> InjectionBus:
        bus = getattr(self, "_injection_bus", None)
        if isinstance(bus, InjectionBus):
            return bus
        bus = InjectionBus(self, self._startup_injection_coordinator())
        self._injection_bus = bus
        return bus

    @property
    def _startup_injection_gate_released(self) -> bool:
        """Legacy read-only alias for coordinator gate state."""
        return self._startup_injection_coordinator().released

    @property
    def _startup_injection_queue(self) -> list[dict[str, Any]]:
        """Legacy read-only alias for coordinator queue snapshot."""
        return list(self._startup_injection_coordinator().queue)

    @property
    def _startup_injection_timeout_task(self) -> Any:
        """Legacy read-only alias for coordinator timeout task."""
        return self._startup_injection_coordinator().timeout_task

    @staticmethod
    def _load_micro_ack_channel_policy(realtime_cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
        raw = realtime_cfg.get("micro_ack_channels")
        if not isinstance(raw, dict):
            raw = {}
        defaults: dict[str, dict[str, Any]] = {
            "voice": {"enabled": True, "cooldown_ms": 10000, "speak": True},
            "text": {"enabled": True, "cooldown_ms": 1000, "speak": False},
        }
        policy = {channel: values.copy() for channel, values in defaults.items()}
        for channel, candidate in raw.items():
            if not isinstance(channel, str) or not isinstance(candidate, dict):
                continue
            merged = policy.get(channel, {"enabled": True, "cooldown_ms": 10000, "speak": channel == "voice"})
            if "enabled" in candidate:
                merged["enabled"] = bool(candidate.get("enabled"))
            if "cooldown_ms" in candidate:
                merged["cooldown_ms"] = max(0, int(candidate.get("cooldown_ms", 0)))
            if "speak" in candidate:
                merged["speak"] = bool(candidate.get("speak"))
            policy[channel] = merged
        return policy

    def _build_micro_ack_manager(self, realtime_cfg: dict[str, Any]) -> MicroAckManager:
        raw_category_cooldowns = realtime_cfg.get("micro_ack_category_cooldown_ms")
        category_cooldown_ms: dict[str, int] = {}
        if isinstance(raw_category_cooldowns, dict):
            for category, cooldown_ms in raw_category_cooldowns.items():
                category_key = str(category).strip()
                if not category_key:
                    continue
                category_cooldown_ms[category_key] = max(0, int(cooldown_ms))

        cfg = MicroAckConfig(
            enabled=bool(realtime_cfg.get("micro_ack_enabled", True)),
            delay_ms=max(150, int(realtime_cfg.get("micro_ack_delay_ms", 450))),
            expected_wait_threshold_ms=max(
                350,
                int(realtime_cfg.get("micro_ack_expected_wait_threshold_ms", 700)),
            ),
            long_wait_second_ack_ms=max(
                0,
                int(realtime_cfg.get("micro_ack_long_wait_second_ack_ms", 4000)),
            ),
            global_cooldown_ms=max(
                1000,
                int(realtime_cfg.get("micro_ack_global_cooldown_ms", 10000)),
            ),
            global_cooldown_scope=str(realtime_cfg.get("micro_ack_global_cooldown_scope", "channel")).strip().lower(),
            per_turn_max=max(1, int(realtime_cfg.get("micro_ack_per_turn_max", 1))),
            channel_enabled={
                channel: bool(values.get("enabled", True))
                for channel, values in self._micro_ack_channel_policy.items()
            },
            channel_cooldown_ms={
                channel: max(0, int(values.get("cooldown_ms", 0)))
                for channel, values in self._micro_ack_channel_policy.items()
            },
            category_cooldown_ms=category_cooldown_ms,
        )
        return MicroAckManager(
            config=cfg,
            on_emit=self._emit_micro_ack,
            on_log=self._log_micro_ack_event,
            suppression_reason=self._micro_ack_suppression_reason,
        )

    def _log_micro_ack_event(
        self,
        event: str,
        turn_id: str,
        reason: str,
        delay_ms: int | None,
        category: str | None,
        channel: str | None,
        intent: str | None,
        action: str | None,
        tool_call_id: str | None,
        dedupe_fingerprint: str | None,
        suppression_source: str | None,
    ) -> None:
        run_id = self._current_run_id() or ""
        category_value = category or ""
        channel_value = channel or ""
        intent_value = intent or ""
        action_value = action or ""
        tool_call_id_value = tool_call_id or ""
        dedupe_fingerprint_value = dedupe_fingerprint or ""
        suppression_source_value = suppression_source or ""
        if event == "scheduled":
            logger.info(
                "micro_ack_scheduled run_id=%s turn_id=%s reason=%s delay_ms=%s category=%s channel=%s intent=%s action=%s tool_call_id=%s dedupe_fp=%s",
                run_id,
                turn_id,
                reason,
                delay_ms if delay_ms is not None else "",
                category_value,
                channel_value,
                intent_value,
                action_value,
                tool_call_id_value,
                dedupe_fingerprint_value,
            )
            return
        if event == "emitted":
            self._record_conversation_micro_ack(turn_id=turn_id)
            logger.info(
                "micro_ack_emitted run_id=%s turn_id=%s phrase_id=%s category=%s channel=%s intent=%s action=%s tool_call_id=%s dedupe_fp=%s",
                run_id,
                turn_id,
                reason,
                category_value,
                channel_value,
                intent_value,
                action_value,
                tool_call_id_value,
                dedupe_fingerprint_value,
            )
            return
        if event == "cancelled":
            logger.info(
                "micro_ack_cancelled run_id=%s turn_id=%s reason=%s category=%s channel=%s intent=%s action=%s tool_call_id=%s",
                run_id,
                turn_id,
                reason,
                category_value,
                channel_value,
                intent_value,
                action_value,
                tool_call_id_value,
            )
            return
        if event in {"emitted", "cancelled", "suppressed"} and channel_value:
            self._clear_pending_micro_ack_marker(turn_id=turn_id, channel=channel_value, reason=event)
        logger.debug(
            "micro_ack_suppressed run_id=%s turn_id=%s reason=%s category=%s channel=%s intent=%s action=%s tool_call_id=%s dedupe_fp=%s suppression_source=%s",
            run_id,
            turn_id,
            reason,
            category_value,
            channel_value,
            intent_value,
            action_value,
            tool_call_id_value,
            dedupe_fingerprint_value,
            suppression_source_value,
        )

    def _micro_ack_suppression_reason(self) -> str | None:
        manager = getattr(self, "_micro_ack_manager", None)
        if manager is None:
            return "disabled"
        baseline_reason = manager.suppression_baseline_reason()
        if baseline_reason:
            return baseline_reason
        if time.monotonic() < float(getattr(self, "_micro_ack_suppress_until_ts", 0.0) or 0.0):
            return "confirmation_decline_guard"
        pending_reason = str(getattr(self, "_pending_micro_ack_reason", "") or "").strip().lower()
        if (
            (self._has_active_confirmation_token() or self._is_awaiting_confirmation_phase())
            and pending_reason not in _MICRO_ACK_CONFIRMATION_ALLOWLIST
        ):
            return "confirmation_pending"
        suppress_for_tool_followup, _ = self._should_suppress_pending_micro_ack_for_tool_followup()
        if suppress_for_tool_followup:
            return "tool_followup_imminent"
        current_state = getattr(getattr(self, "state_manager", None), "state", None)
        if current_state == InteractionState.LISTENING:
            return "listening_state"
        if bool(getattr(self, "_confirmation_speech_active", False)):
            return "speech_active"
        return None

    def _should_suppress_pending_micro_ack_for_tool_followup(self) -> tuple[bool, str | None]:
        turn_input_candidates: list[tuple[str, str]] = []

        active_turn_id = str(self._current_turn_id_or_unknown() or "").strip()
        if active_turn_id:
            active_input_event_key = str(self._active_input_event_key_for_turn(active_turn_id) or "").strip()
            if active_input_event_key:
                turn_input_candidates.append((active_turn_id, active_input_event_key))

        markers = getattr(self, "_pending_micro_ack_by_turn_channel", None)
        if isinstance(markers, dict):
            for marker_turn_id, _channel in markers:
                normalized_turn_id = str(marker_turn_id or "").strip()
                if not normalized_turn_id:
                    continue
                candidate_input_event_key = str(self._active_input_event_key_for_turn(normalized_turn_id) or "").strip()
                if not candidate_input_event_key:
                    continue
                candidate_pair = (normalized_turn_id, candidate_input_event_key)
                if candidate_pair not in turn_input_candidates:
                    turn_input_candidates.append(candidate_pair)

        for candidate_turn_id, candidate_input_event_key in turn_input_candidates:
            suppress_for_tool_followup, tool_followup_call_id = self._should_suppress_micro_ack_for_tool_followup(
                turn_id=candidate_turn_id,
                input_event_key=candidate_input_event_key,
            )
            if suppress_for_tool_followup:
                return True, tool_followup_call_id
        return False, None

    def _micro_ack_correlation_metadata(self) -> dict[str, Any]:
        pending = getattr(self, "_pending_response_create", None)
        if isinstance(pending, PendingResponseCreate):
            metadata = self._extract_response_create_metadata(pending.event)
            if metadata:
                return metadata
        last_metadata = getattr(self, "_last_response_metadata", None)
        if isinstance(last_metadata, dict):
            response_payload = last_metadata.get("response")
            if isinstance(response_payload, dict):
                metadata = response_payload.get("metadata")
                if isinstance(metadata, dict):
                    return metadata
        return {}

    @staticmethod
    def _micro_ack_category_priority(category: MicroAckCategory | str) -> int:
        if isinstance(category, MicroAckCategory):
            normalized = category.value
        else:
            normalized = str(category or "").strip().lower()
        priorities = {
            MicroAckCategory.START_OF_WORK.value: 1,
            MicroAckCategory.LATENCY_MASK.value: 2,
            MicroAckCategory.FAILURE_FALLBACK.value: 3,
            MicroAckCategory.SAFETY_GATE.value: 4,
        }
        return priorities.get(normalized, 0)

    def _clear_pending_micro_ack_marker(
        self,
        *,
        turn_id: str,
        channel: str | None = None,
        reason: str,
    ) -> None:
        markers = getattr(self, "_pending_micro_ack_by_turn_channel", None)
        if not isinstance(markers, dict) or not markers:
            return
        if channel is not None:
            markers.pop((turn_id, channel), None)
            return
        for key in [pending_key for pending_key in markers if pending_key[0] == turn_id]:
            markers.pop(key, None)

    def _clear_stale_pending_micro_ack_markers_for_turn_transition(self, *, next_turn_id: str) -> None:
        markers = getattr(self, "_pending_micro_ack_by_turn_channel", None)
        if not isinstance(markers, dict) or not markers:
            return
        for key in [pending_key for pending_key in markers if pending_key[0] != next_turn_id]:
            markers.pop(key, None)

    def _maybe_schedule_micro_ack(
        self,
        *,
        turn_id: str,
        category: MicroAckCategory,
        channel: str,
        intent: str | None = None,
        action: str | None = None,
        tool_call_id: str | None = None,
        reason: str,
        expected_delay_ms: int | None = None,
    ) -> None:
        manager = getattr(self, "_micro_ack_manager", None)
        if manager is None or self.loop is None:
            return
        active_input_event_key = self._active_input_event_key_for_turn(turn_id)
        if not active_input_event_key:
            return
        interaction_state = getattr(getattr(self, "state_manager", None), "state", None)
        if interaction_state == InteractionState.IDLE:
            return
        delivery_state = self._response_delivery_state(turn_id=turn_id, input_event_key=active_input_event_key)
        if delivery_state in {"done", "cancelled"}:
            return
        canonical_key = self._canonical_utterance_key(turn_id=turn_id, input_event_key=active_input_event_key)
        lifecycle_state = self._lifecycle_controller().state_for(canonical_key)
        if lifecycle_state in {
            InteractionLifecycleState.DONE,
            InteractionLifecycleState.CANCELLED,
            InteractionLifecycleState.REPLACED,
        }:
            return
        suppress_for_tool_followup, tool_followup_call_id = self._should_suppress_micro_ack_for_tool_followup(
            turn_id=turn_id,
            input_event_key=active_input_event_key,
        )
        if suppress_for_tool_followup:
            logger.debug(
                "micro_ack_suppressed run_id=%s turn_id=%s reason=tool_followup_imminent tool_call_id=%s",
                self._current_run_id() or "",
                turn_id,
                tool_followup_call_id or "",
            )
            return
        near_ready_reason = self._micro_ack_near_ready_suppression_reason(
            turn_id=turn_id,
            category=category,
        )
        if near_ready_reason:
            self._suppress_pending_micro_ack_near_ready(
                turn_id=turn_id,
                category=category,
                channel=channel,
                reason=near_ready_reason,
            )
            return
        markers = getattr(self, "_pending_micro_ack_by_turn_channel", None)
        if not isinstance(markers, dict):
            markers = {}
            self._pending_micro_ack_by_turn_channel = markers
        marker_key = (turn_id, channel)
        incoming_priority = self._micro_ack_category_priority(category)
        existing_marker = markers.get(marker_key)
        if isinstance(existing_marker, PendingMicroAckMarker):
            if incoming_priority <= existing_marker.priority:
                return
            manager.cancel(turn_id=turn_id, reason=f"micro_ack_replaced:{reason}")
            markers.pop(marker_key, None)
        metadata = self._micro_ack_correlation_metadata()
        metadata_intent = str(metadata.get("intent") or metadata.get("normalized_intent") or "").strip() or None
        metadata_action = str(metadata.get("action") or metadata.get("trigger") or "").strip() or None
        metadata_tool_call_id = str(metadata.get("tool_call_id") or "").strip() or None
        canonical_key = str(getattr(self, "_active_response_canonical_key", "") or "").strip() or canonical_key

        self._pending_micro_ack_reason = reason
        context = MicroAckContext(
            category=category,
            channel=channel,
            run_id=self._current_run_id() or None,
            session_id=str(getattr(self, "_active_session_id", "") or "").strip() or None,
            turn_id=turn_id,
            intent=intent or metadata_intent,
            action=action or metadata_action or canonical_key,
            tool_call_id=tool_call_id or metadata_tool_call_id,
        )
        markers[marker_key] = PendingMicroAckMarker(
            category=category,
            priority=incoming_priority,
            reason=reason,
        )
        try:
            manager.maybe_schedule(
                context=context,
                reason=reason,
                loop=self.loop,
                expected_delay_ms=expected_delay_ms,
            )
        finally:
            self._pending_micro_ack_reason = None

    def _should_suppress_micro_ack_for_tool_followup(
        self,
        *,
        turn_id: str,
        input_event_key: str,
    ) -> tuple[bool, str | None]:
        normalized_turn_id = str(turn_id or "").strip()
        normalized_input_event_key = str(input_event_key or "").strip()
        if not normalized_turn_id:
            return False, None

        def _iter_candidates() -> Iterator[dict[str, Any]]:
            pending = getattr(self, "_pending_response_create", None)
            if pending is not None and isinstance(getattr(pending, "event", None), dict):
                yield pending.event
            for queued in list(getattr(self, "_response_create_queue", deque()) or ()):
                if isinstance(queued, dict) and isinstance(queued.get("event"), dict):
                    yield queued["event"]

        active_response_id = str(getattr(self, "_active_response_id", "") or "").strip()
        for response_create_event in _iter_candidates():
            response_metadata = self._extract_response_create_metadata(response_create_event)
            if str(response_metadata.get("tool_followup", "")).strip().lower() not in {"true", "1", "yes"}:
                continue
            candidate_turn_id = str(response_metadata.get("turn_id") or "").strip()
            parent_turn_id = str(response_metadata.get("parent_turn_id") or "").strip()
            if candidate_turn_id != normalized_turn_id and parent_turn_id != normalized_turn_id:
                continue
            parent_input_event_key = str(response_metadata.get("parent_input_event_key") or "").strip()
            if parent_input_event_key and normalized_input_event_key and parent_input_event_key != normalized_input_event_key:
                continue
            blocked_by_response_id = str(response_metadata.get("blocked_by_response_id") or "").strip()
            blocked_by_active_response = bool(
                blocked_by_response_id and active_response_id and blocked_by_response_id == active_response_id
            )
            canonical_key = self._canonical_utterance_key(
                turn_id=candidate_turn_id or normalized_turn_id,
                input_event_key=str(response_metadata.get("input_event_key") or "").strip(),
            )
            followup_state = self._tool_followup_state(canonical_key=canonical_key)
            if followup_state in {"scheduled", "scheduled_release", "creating", "created", "released_on_response_done"}:
                return True, str(response_metadata.get("tool_call_id") or "").strip() or None
            if not blocked_by_active_response and followup_state != "blocked_active_response":
                return True, str(response_metadata.get("tool_call_id") or "").strip() or None
        return False, None

    def _allow_text_output_state_transition(self, *, response_id: str | None, event_type: str) -> bool:
        normalized_response_id = str(response_id or "").strip()
        active_response_id = str(getattr(self, "_active_response_id", "") or "").strip()
        if not normalized_response_id or not active_response_id or normalized_response_id != active_response_id:
            logger.debug(
                "text_output_transition_suppressed run_id=%s event_type=%s reason=response_binding_missing response_id=%s active_response_id=%s",
                self._current_run_id() or "",
                event_type,
                normalized_response_id or "unknown",
                active_response_id or "unknown",
            )
            return False
        if self._response_status(normalized_response_id) != "active":
            logger.debug(
                "text_output_transition_suppressed run_id=%s event_type=%s reason=response_not_active response_id=%s status=%s",
                self._current_run_id() or "",
                event_type,
                normalized_response_id,
                self._response_status(normalized_response_id),
            )
            return False
        return True

    def _micro_ack_near_ready_suppression_reason(
        self,
        *,
        turn_id: str,
        category: MicroAckCategory | str,
    ) -> str | None:
        suppress_ms = int(getattr(self, "_micro_ack_near_ready_suppress_ms", 0) or 0)
        if suppress_ms <= 0:
            return None
        normalized_category = (
            category.value if isinstance(category, MicroAckCategory) else str(category or "").strip().lower()
        )
        if normalized_category == MicroAckCategory.SAFETY_GATE.value:
            return None
        pending = getattr(self, "_pending_response_create", None)
        if isinstance(pending, PendingResponseCreate) and pending.turn_id == turn_id:
            age_ms = (time.monotonic() - pending.created_at) * 1000.0
            if age_ms >= suppress_ms:
                return "response_create_bound"
        last_response_create_ts = getattr(self, "_last_response_create_ts", None)
        if isinstance(last_response_create_ts, (int, float)) and bool(getattr(self, "_response_in_flight", False)):
            elapsed_ms = (time.monotonic() - float(last_response_create_ts)) * 1000.0
            if elapsed_ms <= suppress_ms:
                return "response_create_recent"
        return None

    def _suppress_pending_micro_ack_near_ready(
        self,
        *,
        turn_id: str,
        category: MicroAckCategory | str,
        channel: str,
        reason: str,
    ) -> None:
        normalized_category = (
            category.value if isinstance(category, MicroAckCategory) else str(category or "").strip().lower()
        )
        if normalized_category == MicroAckCategory.SAFETY_GATE.value:
            return
        logger.info(
            "micro_ack_suppressed_near_ready run_id=%s turn_id=%s category=%s channel=%s reason=%s",
            self._current_run_id() or "",
            turn_id,
            normalized_category,
            channel,
            reason,
        )
        manager = getattr(self, "_micro_ack_manager", None)
        if manager is not None and hasattr(manager, "cancel_matching"):
            manager.cancel_matching(
                turn_id=turn_id,
                reason="near_ready",
                matcher=lambda context: (
                    context.category.value
                    if isinstance(context.category, MicroAckCategory)
                    else str(context.category or "").strip().lower()
                ) != MicroAckCategory.SAFETY_GATE.value,
            )
        elif manager is not None:
            manager.cancel(turn_id=turn_id, reason="near_ready")
        markers = getattr(self, "_pending_micro_ack_by_turn_channel", None)
        if isinstance(markers, dict):
            for marker_key, marker in list(markers.items()):
                marker_turn_id, _ = marker_key
                if marker_turn_id != turn_id or not isinstance(marker, PendingMicroAckMarker):
                    continue
                marker_category = (
                    marker.category.value
                    if isinstance(marker.category, MicroAckCategory)
                    else str(marker.category or "").strip().lower()
                )
                if marker_category == MicroAckCategory.SAFETY_GATE.value:
                    continue
                markers.pop(marker_key, None)

    @staticmethod
    def _micro_ack_category_for_reason(reason: str) -> MicroAckCategory:
        normalized_reason = str(reason or "").strip().lower()
        if normalized_reason == "speech_stopped":
            return MicroAckCategory.START_OF_WORK
        if normalized_reason == "transcript_finalized":
            return MicroAckCategory.LATENCY_MASK
        if normalized_reason.startswith("watchdog_"):
            if any(
                marker in normalized_reason
                for marker in ("confirm", "approval", "permission", "policy", "safety")
            ):
                return MicroAckCategory.SAFETY_GATE
            if any(marker in normalized_reason for marker in ("error", "failed", "failure", "timeout")):
                return MicroAckCategory.FAILURE_FALLBACK
            return MicroAckCategory.LATENCY_MASK
        return MicroAckCategory.LATENCY_MASK

    def _cancel_micro_ack(self, *, turn_id: str, reason: str) -> None:
        self._clear_pending_micro_ack_marker(turn_id=turn_id, reason=reason)
        manager = getattr(self, "_micro_ack_manager", None)
        if manager is None:
            return
        manager.cancel(turn_id=turn_id, reason=reason)

    def _extend_micro_ack_decline_guard(self, *, now: float | None = None) -> None:
        current_until = float(getattr(self, "_micro_ack_suppress_until_ts", 0.0) or 0.0)
        start = max(current_until, now if now is not None else time.monotonic())
        self._micro_ack_suppress_until_ts = start + _MICRO_ACK_CONFIRMATION_DECLINE_GUARD_S

    def _clear_micro_ack_decline_guard(self) -> None:
        self._micro_ack_suppress_until_ts = 0.0

    def _emit_micro_ack(self, context: MicroAckContext, phrase_id: str, phrase: str) -> None:
        websocket = getattr(self, "websocket", None)
        if websocket is None or self.loop is None:
            return
        if not self._micro_ack_phrase_is_non_substantive(phrase):
            logger.info(
                "micro_ack_discarded reason=substantive_content turn_id=%s phrase_id=%s tool_call_id=%s",
                context.turn_id,
                phrase_id,
                context.tool_call_id or "none",
            )
            self._schedule_tool_followup_after_micro_ack_discard(context=context)
            return
        speak = self._should_speak_micro_ack(context.channel)

        async def _emit() -> None:
            await self.send_assistant_message(
                phrase,
                websocket,
                speak=speak,
                response_metadata={
                    "trigger": "micro_ack",
                    "micro_ack": "true",
                    "consumes_canonical_slot": "false",
                    "micro_ack_turn_id": context.turn_id,
                    "micro_ack_phrase_id": phrase_id,
                    "micro_ack_category": context.category,
                    "micro_ack_channel": context.channel,
                    "micro_ack_intent": context.intent or "",
                    "micro_ack_action": context.action or "",
                    "micro_ack_tool_call_id": context.tool_call_id or "",
                },
            )

        self.loop.create_task(_emit())

    def _micro_ack_phrase_is_non_substantive(self, phrase: str) -> bool:
        normalized_phrase = str(phrase or "").strip()
        if not normalized_phrase:
            return False
        token_count = len([token for token in re.split(r"\s+", normalized_phrase) if token])
        if token_count > 8:
            return False
        normalized = re.sub(r"\s+", " ", normalized_phrase).lower()
        substantive_markers = (
            "queued",
            "queueing",
            "started",
            "completed",
            "done",
            "i have",
            "i've",
            "i will",
            "i can",
            "moving",
            "turned",
            "action",
            "request",
            "because",
            "here's",
        )
        if any(marker in normalized for marker in substantive_markers):
            return False
        return True

    def _schedule_tool_followup_after_micro_ack_discard(self, *, context: MicroAckContext) -> None:
        tool_call_id = str(getattr(context, "tool_call_id", "") or "").strip()
        if not tool_call_id:
            return
        response_create_event, canonical_key = self._build_tool_followup_response_create_event(call_id=tool_call_id)

        async def _emit_followup() -> None:
            await self._send_response_create(
                self.websocket,
                response_create_event,
                origin="tool_output",
                record_ai_call=True,
            )

        logger.info(
            "micro_ack_discard_followup_scheduled turn_id=%s tool_call_id=%s canonical_key=%s",
            context.turn_id,
            tool_call_id,
            canonical_key,
        )
        self.loop.create_task(_emit_followup())

    def _conversation_efficiency_state(self, *, turn_id: str) -> ConversationEfficiencyState:
        normalized_turn_id = str(turn_id or "").strip() or "turn-unknown"
        efficiency = getattr(self, "_conversation_efficiency_by_turn", None)
        if not isinstance(efficiency, dict):
            efficiency = {}
            self._conversation_efficiency_by_turn = efficiency
        state = efficiency.get(normalized_turn_id)
        if state is None:
            state = ConversationEfficiencyState(
                substantive_count_by_canonical={},
                duplicate_alerted_canonical_keys=set(),
            )
            efficiency[normalized_turn_id] = state
        if state.substantive_count_by_canonical is None:
            state.substantive_count_by_canonical = {}
        if state.duplicate_alerted_canonical_keys is None:
            state.duplicate_alerted_canonical_keys = set()
        return state

    def _record_conversation_micro_ack(self, *, turn_id: str) -> None:
        state = self._conversation_efficiency_state(turn_id=turn_id)
        state.ack_count += 1
        state.estimated_token_overhead += 6
        state.estimated_audio_overhead_ms += 900

    def _record_duplicate_create_attempt(self, *, turn_id: str, canonical_key: str, reason: str) -> None:
        state = self._conversation_efficiency_state(turn_id=turn_id)
        state.duplicate_prevented += 1
        logger.debug(
            "conversation_efficiency_duplicate run_id=%s turn_id=%s canonical_key=%s reason=%s duplicate_prevented=%s",
            self._current_run_id() or "",
            turn_id,
            canonical_key or "unknown",
            reason,
            state.duplicate_prevented,
        )

    def _record_dropped_queued_creates(self, *, turn_id: str, dropped_count: int) -> None:
        if dropped_count <= 0:
            return
        state = self._conversation_efficiency_state(turn_id=turn_id)
        state.dropped_queued_creates += int(dropped_count)
        state.estimated_token_overhead += int(dropped_count) * 4

    def _record_substantive_response(self, *, turn_id: str, canonical_key: str) -> None:
        normalized_key = str(canonical_key or "").strip() or "canonical-unknown"
        state = self._conversation_efficiency_state(turn_id=turn_id)
        counts = state.substantive_count_by_canonical or {}
        next_count = int(counts.get(normalized_key, 0)) + 1
        counts[normalized_key] = next_count
        state.substantive_count_by_canonical = counts
        state.substantive_count += 1
        if next_count > 1:
            state.estimated_token_overhead += 120
            state.estimated_audio_overhead_ms += 2500
            alerted_keys = state.duplicate_alerted_canonical_keys or set()
            if normalized_key not in alerted_keys:
                alerted_keys.add(normalized_key)
                state.duplicate_alerted_canonical_keys = alerted_keys
                self._emit_alert(
                    Alert(
                        key="conversation_efficiency_substantive_duplicates",
                        message=(
                            "Multiple substantive responses emitted for one canonical user input."
                        ),
                        severity="warning",
                        metadata={
                            "turn_id": turn_id,
                            "canonical_key": normalized_key,
                            "substantive_count": next_count,
                        },
                    )
                )

    def _recent_silent_turn_lifecycle_markers(self, *, canonical_key: str) -> str:
        normalized_canonical_key = str(canonical_key or "").strip()
        if not normalized_canonical_key:
            return ""
        timeline_store = getattr(self, "_lifecycle_canonical_timeline", None)
        timeline = timeline_store.get(normalized_canonical_key) if isinstance(timeline_store, dict) else None
        if not isinstance(timeline, deque) or not timeline:
            return ""
        marker_entries: list[str] = []
        decision_markers = (
            ("response.created", "response_created_"),
            ("audio_started", "audio_delta_"),
            ("response.done", "transition_done:response_done"),
        )
        for marker_name, decision_prefix in decision_markers:
            matched_entry = ""
            for timeline_entry in reversed(timeline):
                if f"decision={decision_prefix}" in timeline_entry:
                    matched_entry = timeline_entry
                    break
            if matched_entry:
                marker_entries.append(f"{marker_name}={matched_entry}")
        return " ".join(marker_entries)

    def _record_silent_turn_incident(
        self,
        *,
        turn_id: str,
        canonical_key: str,
        origin: str,
        response_id: str | None,
    ) -> None:
        state = self._conversation_efficiency_state(turn_id=turn_id)
        state.silent_turn_incidents += 1
        self._silent_turn_incident_count = int(getattr(self, "_silent_turn_incident_count", 0)) + 1
        lifecycle_markers = self._recent_silent_turn_lifecycle_markers(canonical_key=canonical_key)
        marker_suffix = f" lifecycle_markers={lifecycle_markers}" if lifecycle_markers else ""
        logger.info(
            "silent_turn_incident run_id=%s turn_id=%s canonical_key=%s origin=%s response_id=%s total_incidents=%s%s",
            self._current_run_id() or "",
            str(turn_id or "").strip() or "turn-unknown",
            str(canonical_key or "").strip() or "canonical-unknown",
            str(origin or "").strip() or "unknown",
            str(response_id or "").strip() or "none",
            self._silent_turn_incident_count,
            marker_suffix,
        )

    def _log_turn_conversation_efficiency(
        self,
        *,
        turn_id: str,
        canonical_key: str,
        close_reason: str,
    ) -> None:
        normalized_turn_id = str(turn_id or "").strip() or "turn-unknown"
        if normalized_turn_id in self._conversation_efficiency_logged_turns:
            return
        efficiency = getattr(self, "_conversation_efficiency_by_turn", None)
        if not isinstance(efficiency, dict):
            efficiency = {}
            self._conversation_efficiency_by_turn = efficiency
        state = efficiency.get(normalized_turn_id)
        if state is None:
            return
        self._conversation_efficiency_logged_turns.add(normalized_turn_id)
        logger.info(
            "conversation_efficiency run_id=%s turn_id=%s canonical_key=%s ack_count=%s substantive_count=%s "
            "duplicate_prevented=%s estimated_token_overhead=%s estimated_audio_overhead_ms=%s dropped_queued_creates=%s "
            "silent_turn_incidents=%s silent_turn_incidents_total=%s close_reason=%s",
            self._current_run_id() or "",
            normalized_turn_id,
            str(canonical_key or "").strip() or "canonical-unknown",
            state.ack_count,
            state.substantive_count,
            state.duplicate_prevented,
            state.estimated_token_overhead,
            state.estimated_audio_overhead_ms,
            state.dropped_queued_creates,
            state.silent_turn_incidents,
            int(getattr(self, "_silent_turn_incident_count", 0)),
            close_reason,
        )

    def _should_speak_micro_ack(self, channel: str) -> bool:
        if self._micro_ack_runtime_mode == "quiet":
            return False
        if self._micro_ack_channel_mode == "text_only":
            return False
        channel_policy = self._micro_ack_channel_policy.get(channel, {})
        if not bool(channel_policy.get("enabled", True)):
            return False
        if self._micro_ack_runtime_mode == "verbose":
            return True
        return bool(channel_policy.get("speak", channel == "voice"))

    def _mark_transcript_response_outcome(
        self,
        *,
        input_event_key: str,
        turn_id: str,
        outcome: str,
        reason: str | None = None,
        details: str | None = None,
    ) -> None:
        normalized_key = str(input_event_key or "").strip()
        if not normalized_key:
            return
        delivery_state = self._response_delivery_state(turn_id=turn_id, input_event_key=normalized_key)
        if delivery_state in {"delivered", "done"} and reason == "audio_playback_busy":
            reason = "already_handled"
            details = f"canonical_delivery_state={delivery_state}"
        watchdog_tasks = getattr(self, "_transcript_response_watchdog_tasks", None)
        if not isinstance(watchdog_tasks, dict):
            watchdog_tasks = {}
            self._transcript_response_watchdog_tasks = watchdog_tasks
        task = watchdog_tasks.pop(normalized_key, None)
        if task is not None and not task.done():
            task.cancel()
        if outcome != "response_not_scheduled":
            return
        logged_keys = getattr(self, "_transcript_response_outcome_logged_keys", None)
        if not isinstance(logged_keys, set):
            logged_keys = set()
            self._transcript_response_outcome_logged_keys = logged_keys
        if normalized_key in logged_keys:
            return
        logged_keys.add(normalized_key)
        normalized_reason = str(reason or "unknown").strip() or "unknown"
        log_method = logger.info
        if normalized_reason in {"same_turn_already_owned", "active_response_in_flight"}:
            log_method = logger.debug
        if (
            normalized_reason == "already_handled"
            and isinstance(details, str)
            and (
                details.strip() == "canonical delivery terminal state"
                or details.strip().startswith("canonical_delivery_state=")
            )
        ):
            log_method = logger.debug
        log_method(
            "response_not_scheduled run_id=%s turn_id=%s input_event_key=%s reason=%s details=%s",
            self._current_run_id() or "",
            turn_id,
            normalized_key,
            normalized_reason,
            details or "",
        )
        self._log_response_site_debug(
            site="response_not_scheduled",
            turn_id=turn_id,
            input_event_key=normalized_key,
            canonical_key=self._canonical_utterance_key(turn_id=turn_id, input_event_key=normalized_key),
            origin=str(getattr(self, "_active_response_origin", "") or "unknown").strip() or "unknown",
            trigger=normalized_reason,
        )

    async def _watch_transcript_response_outcome(
        self,
        *,
        turn_id: str,
        input_event_key: str,
        timeout_s: float,
    ) -> None:
        try:
            await asyncio.sleep(timeout_s)
        except asyncio.CancelledError:
            return
        suppression_until = float(getattr(self, "_preference_recall_response_suppression_until", 0.0) or 0.0)
        pending = getattr(self, "_pending_response_create", None)
        policy_decision = self._lifecycle_policy().decide_watchdog_timeout(
            suppressed_by_turn=turn_id in getattr(self, "_preference_recall_suppressed_turns", set()),
            suppressed_by_input_event=input_event_key in getattr(self, "_preference_recall_suppressed_input_event_keys", set()),
            suppression_window_active=suppression_until > time.monotonic(),
            response_in_flight=self._is_active_response_blocking(),
            active_response_origin=str(getattr(self, "_active_response_origin", "unknown") or "unknown"),
            active_response_id=str(getattr(self, "_active_response_id", None) or "unknown"),
            delivery_state_terminal=self._response_delivery_state(turn_id=turn_id, input_event_key=input_event_key) in {"delivered", "done"},
            audio_playback_busy=bool(getattr(self, "_audio_playback_busy", False)),
            has_pending_response_create=pending is not None,
            pending_origin=str(getattr(pending, "origin", "unknown") if pending is not None else "unknown"),
            pending_reason=str(getattr(pending, "reason", "unknown") if pending is not None else "unknown"),
            listening_state_gate=getattr(getattr(self, "state_manager", None), "state", None) == InteractionState.LISTENING,
        )
        self._mark_transcript_response_outcome(
            input_event_key=input_event_key,
            turn_id=turn_id,
            outcome="response_not_scheduled",
            reason=policy_decision.reason_code,
            details=policy_decision.details,
        )
        if policy_decision.should_schedule_micro_ack:
            self._maybe_schedule_micro_ack(
                turn_id=turn_id,
                category=self._micro_ack_category_for_reason(f"watchdog_{policy_decision.reason_code}"),
                channel="voice",
                action=self._canonical_utterance_key(turn_id=turn_id, input_event_key=input_event_key),
                reason=f"watchdog_{policy_decision.reason_code}",
                expected_delay_ms=900,
            )

    def _start_transcript_response_watchdog(self, *, turn_id: str, input_event_key: str) -> None:
        normalized_key = str(input_event_key or "").strip()
        if not normalized_key:
            return
        timeout_s = max(0.5, float(getattr(self, "_transcript_response_watchdog_timeout_s", 3.0) or 3.0))
        watchdog_tasks = getattr(self, "_transcript_response_watchdog_tasks", None)
        if not isinstance(watchdog_tasks, dict):
            watchdog_tasks = {}
            self._transcript_response_watchdog_tasks = watchdog_tasks
        existing = watchdog_tasks.pop(normalized_key, None)
        if existing is not None and not existing.done():
            existing.cancel()
        logged_keys = getattr(self, "_transcript_response_outcome_logged_keys", None)
        if not isinstance(logged_keys, set):
            logged_keys = set()
            self._transcript_response_outcome_logged_keys = logged_keys
        logged_keys.discard(normalized_key)
        watchdog_tasks[normalized_key] = self._runtime_task_registry().spawn(
            f"watchdog.{normalized_key}",
            self._watch_transcript_response_outcome(
                turn_id=turn_id,
                input_event_key=normalized_key,
                timeout_s=timeout_s,
            )
        )

    def _create_microphone(self) -> AsyncMicrophone:
        return AsyncMicrophone(
            input_device_name=self._audio_input_device_name,
            debug_list_devices=False,
        )

    def _initialize_microphone(self, *, allow_failure: bool) -> AsyncMicrophone | _DisabledMicrophone:
        try:
            return self._create_microphone()
        except Exception as exc:
            outcome = StartupDependencyOutcome(
                component="audio_input",
                dependency_class="required",
                status="degraded" if allow_failure else "fatal",
                detail=str(exc),
            )
            if not allow_failure:
                raise RealtimeAPIStartupError(
                    f"audio input initialization failed: {exc}",
                    outcome=outcome,
                ) from exc
            logger.warning(
                "startup component=%s dependency_class=%s status=%s detail=%s",
                outcome.component,
                outcome.dependency_class,
                outcome.status,
                outcome.detail,
            )
            return _DisabledMicrophone()

    def _resolve_vad_turn_detection(self, config: dict[str, Any]) -> dict[str, float | int | bool]:
        realtime_cfg = config.get("realtime") or {}
        profile_name = str(realtime_cfg.get("vad_profile", "default"))
        profiles = realtime_cfg.get("vad_profiles") or {}
        profile_cfg = profiles.get(profile_name) or profiles.get("default") or {}

        threshold = float(
            profile_cfg.get("threshold", config.get("silence_threshold", SILENCE_THRESHOLD))
        )
        prefix_padding_ms = int(
            profile_cfg.get("prefix_padding_ms", config.get("prefix_padding_ms", PREFIX_PADDING_MS))
        )
        silence_duration_ms = int(
            profile_cfg.get(
                "silence_duration_ms",
                config.get("silence_duration_ms", SILENCE_DURATION_MS),
            )
        )

        return {
            "profile": profile_name,
            "threshold": threshold,
            "prefix_padding_ms": prefix_padding_ms,
            "silence_duration_ms": silence_duration_ms,
            "create_response": True,
            "interrupt_response": True,
        }

    def get_event_bus(self) -> EventBus:
        return self.event_bus

    def _allow_ai_call(self, reason: str, *, bypass: bool = False) -> bool:
        if bypass:
            return True
        if self._ai_call_budget.allow():
            return True
        self._emit_alert(
            Alert(
                key="budget_ai_calls",
                message=f"AI call budget exhausted ({reason}).",
                severity="warning",
                metadata={"reason": reason},
            )
        )
        return False

    def _record_ai_call(self) -> None:
        self._ai_call_budget.record()

    def _emit_alert(self, alert: Alert) -> None:
        alert_policy = getattr(self, "_alert_policy", None)
        event_bus = getattr(self, "event_bus", None)
        if alert_policy is None or event_bus is None:
            return
        alert_policy.emit(event_bus, alert)

    def is_ready_for_injections(self, with_reason: bool = False) -> bool | tuple[bool, str]:
        injector_ready, injector_reason = self._can_accept_external_stimulus(
            "event_injector",
            "probe",
        )

        ready = False
        reason = "not_ready"
        if not self.ready_event.is_set():
            reason = "ready_event_not_set"
        elif self.websocket is None:
            reason = "websocket_unavailable"
        elif self.loop is None:
            reason = "loop_unavailable"
        elif not self.loop.is_running():
            reason = "loop_not_running"
        elif not injector_ready:
            reason = injector_reason
        else:
            ready = True
            reason = "ready"

        if with_reason:
            return ready, reason
        return ready

    def _load_awaiting_confirmation_source_policy(
        self,
        realtime_cfg: dict[str, Any],
    ) -> set[tuple[str, str | None]]:
        raw_entries = realtime_cfg.get(
            "awaiting_confirmation_allowed_sources",
            ["battery:critical", "imu:critical"],
        )
        policy: set[tuple[str, str | None]] = set()
        if not isinstance(raw_entries, list):
            log_warning(
                "realtime.awaiting_confirmation_allowed_sources should be a list; got %s.",
                type(raw_entries).__name__,
            )
            return policy
        for entry in raw_entries:
            if not isinstance(entry, str) or not entry.strip():
                continue
            source, _, kind = entry.strip().lower().partition(":")
            if not source:
                continue
            policy.add((source, kind or None))
        return policy

    def _can_accept_external_stimulus(
        self,
        source: str,
        kind: str,
        *,
        priority: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[bool, str]:
        return self._injection_bus_instance().can_accept_external_stimulus(
            source,
            kind,
            priority=priority,
            metadata=metadata,
        )

    def _normalize_external_stimulus_kind(
        self,
        kind: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        return self._injection_bus_instance().normalize_external_stimulus_kind(kind, metadata)

    def _is_allowed_awaiting_confirmation_stimulus(
        self,
        source: str,
        kind: str,
        priority: str,
    ) -> bool:
        if priority not in {"critical", "high"}:
            return False
        for allowed_source, allowed_kind in self._awaiting_confirmation_allowed_sources:
            if source != allowed_source:
                continue
            if allowed_kind in (None, "*", kind):
                return True
        return False

    def _log_suppressed_stimulus(self, source: str, kind: str, reason: str) -> None:
        phase = getattr(self.orchestration_state, "phase", None)
        phase_name = getattr(phase, "value", str(phase))
        logger.info(
            "Suppressed external stimulus source=%s kind=%s phase=%s reason=%s",
            source,
            kind,
            phase_name,
            reason,
        )

    def _startup_injection_gate_active(self) -> bool:
        return self._injection_bus_instance().startup_injection_gate_active()

    def _startup_gate_is_critical_allowed(self, source: str, kind: str, priority: str) -> bool:
        return self._injection_bus_instance().startup_gate_is_critical_allowed(source, kind, priority)

    def _maybe_defer_startup_injection(
        self,
        *,
        source: str,
        kind: str,
        priority: str,
        payload: dict[str, Any],
    ) -> bool:
        return self._injection_bus_instance().maybe_defer_startup_injection(
            source=source,
            kind=kind,
            priority=priority,
            payload=payload,
        )

    def _release_startup_injection_gate(self, *, reason: str) -> None:
        self._injection_bus_instance().release_startup_injection_gate(reason=reason)

    async def _startup_injection_timeout_release(self) -> None:
        await self._injection_bus_instance().startup_injection_timeout_release()

    def _ensure_startup_injection_timeout_task(self) -> None:
        self._injection_bus_instance().ensure_startup_injection_timeout_task()

    def get_session_health(self) -> dict[str, Any]:
        injection_ready_result = self.is_ready_for_injections(with_reason=True)
        if isinstance(injection_ready_result, tuple):
            injection_ready, injection_ready_reason = injection_ready_result
        else:
            injection_ready = bool(injection_ready_result)
            injection_ready_reason = "ready" if injection_ready else "not_ready"
        injection_ready = bool(injection_ready)
        session_ready = bool(self.ready_event.is_set())
        retrieval_metrics = self._memory_manager.get_retrieval_health_metrics(
            scope=self._memory_retrieval_scope,
            session_id=self._memory_manager.get_active_session_id(),
        )
        return {
            "connected": self._session_connected,
            "ready": injection_ready,
            "injection_ready": injection_ready,
            "injection_ready_reason": str(injection_ready_reason),
            "session_ready": session_ready,
            "connection_attempts": self._session_connection_attempts,
            "connections": self._session_connections,
            "reconnects": self._session_reconnects,
            "failures": self._session_failures,
            "last_connect_time": self._last_connect_time or 0.0,
            "last_disconnect_reason": self._last_disconnect_reason or "",
            "last_failure_reason": self._last_failure_reason or "",
            "memory_retrieval": retrieval_metrics,
            "rate_limits": {
                "supports_tokens": int(self.rate_limits_supports_tokens),
                "supports_requests": int(self.rate_limits_supports_requests),
                "last_present_names": sorted(self.rate_limits_last_present_names),
                "last_event_id": self.rate_limits_last_event_id,
            },
            "sensor_event_aggregation": dict(self._sensor_event_aggregation_metrics),
            "silent_turn_incidents": int(getattr(self, "_silent_turn_incident_count", 0)),
        }

    def _note_connection_attempt(self) -> None:
        self._session_connection_attempts += 1

    def _note_connected(self) -> None:
        self._session_connections += 1
        self._session_connected = True
        self._last_connect_time = time.time()
        self._last_disconnect_reason = None

    def _note_disconnect(self, reason: str) -> None:
        self._session_connected = False
        self._last_disconnect_reason = reason

    def _note_failure(self, reason: str) -> None:
        self._session_failures += 1
        self._last_failure_reason = reason

    def _note_reconnect(self) -> None:
        self._session_reconnects += 1

    def _track_outgoing_event(
        self,
        event: dict[str, Any],
        *,
        origin: str | None = None,
    ) -> None:
        event_type = event.get("type")
        if not event_type:
            return
        self._last_outgoing_event_type = str(event_type)
        if event_type == "response.create":
            self._queue_response_origin(origin, event)

    def _queue_response_origin(self, origin: str | None, event: dict[str, Any] | None = None) -> None:
        normalized_origin = str(origin).strip() if origin else "unknown"
        response = event.get("response") if isinstance(event, dict) else None
        metadata = response.get("metadata") if isinstance(response, dict) else None
        metadata_turn_id = str(metadata.get("turn_id") or "").strip() if isinstance(metadata, dict) else ""
        metadata_input_event_key = str(metadata.get("input_event_key") or "").strip() if isinstance(metadata, dict) else ""
        consumes_canonical_slot = self._response_consumes_canonical_slot(metadata)
        is_micro_ack = isinstance(metadata, dict) and str(metadata.get("micro_ack", "")).strip().lower() == "true"
        pending_origin = {
            "origin": "micro_ack" if is_micro_ack else normalized_origin,
            "micro_ack": "true" if is_micro_ack else "false",
            "consumes_canonical_slot": "true" if consumes_canonical_slot else "false",
            "turn_id": metadata_turn_id,
            "input_event_key": metadata_input_event_key,
        }
        if metadata_turn_id and metadata_input_event_key:
            self._bind_active_input_event_key_for_turn(
                turn_id=metadata_turn_id,
                input_event_key=metadata_input_event_key,
                cause="queue_response_origin",
                origin=normalized_origin,
            )
        self._pending_response_create_origins.append(pending_origin)

    def _response_consumes_canonical_slot(self, metadata: dict[str, Any] | None) -> bool:
        if not isinstance(metadata, dict):
            return True
        explicit_value = metadata.get("consumes_canonical_slot")
        if explicit_value is not None:
            normalized = str(explicit_value).strip().lower()
            return normalized in {"1", "true", "yes"}
        return str(metadata.get("micro_ack", "")).strip().lower() != "true"

    def _consume_response_origin(self, event: dict[str, Any] | None = None) -> str:
        self._active_response_consumes_canonical_slot = True
        self._last_consumed_response_origin_context = {}
        if self._pending_response_create_origins:
            response = event.get("response") if isinstance(event, dict) else None
            metadata = response.get("metadata") if isinstance(response, dict) else None
            is_micro_ack_response = (
                isinstance(metadata, dict)
                and str(metadata.get("micro_ack", "")).strip().lower() == "true"
            )
            if is_micro_ack_response:
                preferred_index = next(
                    (
                        idx
                        for idx, pending in enumerate(self._pending_response_create_origins)
                        if isinstance(pending, dict) and str(pending.get("micro_ack", "")).strip().lower() == "true"
                    ),
                    0,
                )
            else:
                preferred_index = next(
                    (
                        idx
                        for idx, pending in enumerate(self._pending_response_create_origins)
                        if not isinstance(pending, dict) or str(pending.get("micro_ack", "")).strip().lower() != "true"
                    ),
                    None,
                )
                if preferred_index is None:
                    self._active_response_consumes_canonical_slot = self._response_consumes_canonical_slot(metadata)
                    return "server_auto"
            pending = self._pending_response_create_origins[preferred_index]
            del self._pending_response_create_origins[preferred_index]
            if not isinstance(pending, dict):
                self._active_response_consumes_canonical_slot = True
                return str(pending or "unknown")
            self._last_consumed_response_origin_context = {
                "turn_id": str(pending.get("turn_id") or "").strip(),
                "input_event_key": str(pending.get("input_event_key") or "").strip(),
            }
            self._active_response_consumes_canonical_slot = (
                str(pending.get("consumes_canonical_slot", "")).strip().lower() == "true"
            )
            return str(pending.get("origin") or "unknown")
        response = event.get("response") if isinstance(event, dict) else None
        metadata = response.get("metadata") if isinstance(response, dict) else None
        self._active_response_consumes_canonical_slot = self._response_consumes_canonical_slot(metadata)
        return "server_auto"

    def inject_event(self, event: Event) -> None:
        allowed, reason = self._can_accept_external_stimulus(
            event.source,
            event.kind,
            priority=event.priority,
            metadata=event.metadata,
        )
        if not allowed:
            self._log_suppressed_stimulus(
                event.source,
                self._normalize_external_stimulus_kind(event.kind, event.metadata),
                reason,
            )
            return
        message, request_response = self._format_event_for_injection(event)
        bypass_response_suppression = False
        safety_override = False
        if event.source == "battery":
            request_response = self._should_request_battery_response(event, fallback=request_response)
            bypass_response_suppression = request_response and self._is_battery_query_context_active()
            safety_override = request_response and self._is_battery_safety_override(event)
        elif event.request_response is not None:
            request_response = event.request_response
        if event.source in {"battery", "imu", "camera", "ops", "health"} and event.request_response is None:
            request_response = bool(request_response) and safety_override
        queued_payload = {
            "type": "event",
            "event": event,
            "message": message,
            "request_response": request_response,
            "bypass_response_suppression": bypass_response_suppression,
            "safety_override": safety_override,
            "injection_metadata": {
                "source": event.source,
                "kind": event.kind,
                "priority": event.priority,
                "severity": str((event.metadata or {}).get("severity") or "").strip().lower(),
            },
        }
        severity = str((event.metadata or {}).get("severity") or event.kind or "").strip().lower()
        priority = str(event.priority or "").strip().lower()
        if self._maybe_defer_startup_injection(
            source=event.source,
            kind=severity,
            priority=priority,
            payload=queued_payload,
        ):
            self._ensure_startup_injection_timeout_task()
            return
        self._emit_injected_event(queued_payload)

    def _emit_injected_event(self, payload: dict[str, Any]) -> None:
        event = payload.get("event")
        if isinstance(event, Event):
            self._log_injection_event(event, bool(payload.get("request_response", True)))
        self._send_text_message(
            str(payload.get("message") or ""),
            request_response=bool(payload.get("request_response", True)),
            bypass_response_suppression=bool(payload.get("bypass_response_suppression", False)),
            safety_override=bool(payload.get("safety_override", False)),
            injection_metadata=payload.get("injection_metadata") if isinstance(payload.get("injection_metadata"), dict) else None,
        )

    def _log_injection_event(self, event: Event, request_response: bool) -> None:
        metadata = event.metadata or {}
        parts = [
            "injection:",
            f"source={event.source}",
        ]
        if event.kind:
            parts.append(f"kind={event.kind}")
        severity = metadata.get("severity")
        if severity:
            parts.append(f"severity={severity}")
        reason = metadata.get("reason") or metadata.get("event_type")
        if reason:
            parts.append(f"reason={reason}")
        parts.append(f"create_response={'true' if request_response else 'false'}")
        log_info(" ".join(parts))

    def _send_text_message(
        self,
        message: str,
        request_response: bool = True,
        *,
        bypass_response_suppression: bool = False,
        safety_override: bool = False,
        injection_metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.loop:
            logger.debug("Unable to send message; event loop unavailable.")
            return

        coordinator = self._shutdown_coordinator()
        if coordinator.is_shutdown_requested():
            logger.info(
                "queued_message_dropped_during_shutdown reason=shutdown_requested websocket_state=unknown"
            )
            return

        future = asyncio.run_coroutine_threadsafe(
            self.send_text_message_to_conversation(
                message,
                request_response=request_response,
                bypass_response_suppression=bypass_response_suppression,
                safety_override=safety_override,
                injection_metadata=injection_metadata,
            ),
            self.loop,
        )

        def _on_complete(task) -> None:
            try:
                task.result()
            except Exception as exc:
                try:
                    websocket_state = asyncio.run_coroutine_threadsafe(
                        coordinator.websocket_close_state(),
                        self.loop,
                    ).result(timeout=0.2)
                except Exception:
                    websocket_state = "unknown"

                is_shutdown = coordinator.is_shutdown_requested()
                if is_shutdown or websocket_state in {"closing", "closed"}:
                    logger.info(
                        "queued_message_dropped_during_shutdown reason=websocket_%s "
                        "shutdown_requested=%s exception=%s",
                        websocket_state,
                        is_shutdown,
                        type(exc).__name__,
                    )
                    return

                logger.warning(
                    "queued_message_send_failed reason=unexpected_send_error "
                    "shutdown_requested=%s websocket_state=%s exception=%s",
                    is_shutdown,
                    websocket_state,
                    type(exc).__name__,
                )

        future.add_done_callback(_on_complete)

    def inject_system_context(self, payload: dict[str, Any]) -> None:
        """Inject non-chat startup context into conversation state."""

        source = str(payload.get("source") or "system_context").strip().lower()
        queued_payload = {
            "type": "system_context",
            "payload": payload,
        }
        if self._maybe_defer_startup_injection(
            source=source,
            kind="context",
            priority="normal",
            payload=queued_payload,
        ):
            self._ensure_startup_injection_timeout_task()
            return
        self._emit_system_context_payload(queued_payload)

    def _emit_system_context_payload(self, queued_payload: dict[str, Any]) -> None:
        payload = queued_payload.get("payload")
        if not isinstance(payload, dict):
            return

        if not self.loop:
            logger.debug("Unable to send system context; event loop unavailable.")
            return
        future = asyncio.run_coroutine_threadsafe(
            self.send_system_context(payload),
            self.loop,
        )

        def _on_complete(task) -> None:
            try:
                task.result()
            except Exception as exc:
                logger.warning("Failed to inject system context: %s", exc)

        future.add_done_callback(_on_complete)

    async def send_system_context(self, payload: dict[str, Any]) -> None:
        """Send system context JSON as a system-role message without requesting a response."""

        if not self.websocket:
            logger.debug("Unable to send system context; websocket unavailable.")
            return
        context_text = json.dumps(payload, sort_keys=True)
        event = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": context_text,
                    }
                ],
            },
        }
        log_ws_event("Outgoing", event)
        self._track_outgoing_event(event, origin="system_context")
        transport = self._get_or_create_transport()
        await transport.send_json(self.websocket, event)

    def _format_event_for_injection(self, event: Event) -> tuple[str, bool]:
        if event.content:
            default_response = event.source not in {"battery", "imu", "camera", "ops", "health"}
            return event.content, default_response
        if event.source == "imu":
            metadata = event.metadata
            return (
                "IMU event: "
                f"{metadata.get('event_type')} ({metadata.get('severity')}) "
                f"details={metadata.get('details')}",
                False,
            )
        if event.source == "battery":
            metadata = event.metadata
            percent = float(metadata.get("percent_of_range", 0.0)) * 100
            voltage = float(metadata.get("voltage", 0.0))
            severity = str(metadata.get("severity", "info"))
            event_type = str(metadata.get("event_type", "status"))
            transition = str(metadata.get("transition", "steady"))
            delta_percent = float(metadata.get("delta_percent", 0.0))
            rapid_drop = bool(metadata.get("rapid_drop", False))

            if event_type == "clear":
                return (
                    "Battery warning cleared: "
                    f"{voltage:.2f}V ({percent:.1f}% of range) transition={transition}",
                    False,
                )

            return (
                "Battery voltage: "
                f"{voltage:.2f}V ({percent:.1f}% of range) "
                f"severity={severity} transition={transition} "
                f"delta_percent={delta_percent:.1f} rapid_drop={str(rapid_drop).lower()}",
                False,
            )
        default_response = event.source not in {"battery", "imu", "camera", "ops", "health"}
        return f"{event.source} event: {event.metadata}", default_response


    def _should_request_battery_response(self, event: Event, *, fallback: bool = False) -> bool:
        return self._battery_policy().should_request_response(event, fallback=fallback)

    def _is_battery_status_query(self, text: str) -> bool:
        return self._battery_policy().is_battery_status_query(text)

    def _is_battery_query_context_active(self) -> bool:
        return self._battery_policy().is_query_context_active()

    def _handle_state_gesture(self, state: InteractionState) -> None:
        """Hook for gesture cues on state transitions."""
        previous_state = self._last_interaction_state
        self._last_interaction_state = state

        if state == InteractionState.SPEAKING:
            logger.debug("Gesture cue skipped for state: %s", state.value)
            return

        gesture_name: str | None = None
        delay_ms = 0
        if state == InteractionState.LISTENING:
            gesture_name = "gesture_attention_snap"
        elif state == InteractionState.THINKING:
            gesture_name = "gesture_curious_tilt"
            delay_ms = random.randint(150, 300)
        elif state == InteractionState.IDLE and previous_state == InteractionState.SPEAKING:
            gesture_name = "gesture_nod"

        if not gesture_name:
            logger.debug("Gesture cue ignored for state: %s", state.value)
            return

        try:
            controller = MotionController.get_instance()
        except Exception as exc:
            logger.warning("Gesture cue skipped: motion controller unavailable (%s).", exc)
            return

        if not controller.is_control_loop_alive():
            logger.info(
                "Gesture cue skipped: motion controller not running (state=%s).",
                state.value,
            )
            return

        try:
            servos = controller.servo_registry.get_servos()
        except Exception as exc:
            logger.warning("Gesture cue skipped: servo registry unavailable (%s).", exc)
            return

        if not servos:
            logger.warning("Gesture cue skipped: servo registry not ready.")
            return

        if controller.is_moving():
            logger.debug("Gesture cue skipped: motion active for state %s.", state.value)
            return

        with controller._queue_lock:
            queued_actions = list(controller.action_queue)

        if queued_actions:
            logger.debug(
                "Gesture cue skipped: action queue not empty (%s items).",
                len(queued_actions),
            )
            return

        now = time.monotonic()
        global_elapsed = now - self._last_gesture_time
        if global_elapsed < self._gesture_global_cooldown_s:
            logger.debug(
                "Gesture cue skipped: global cooldown %.2fs remaining for %s.",
                self._gesture_global_cooldown_s - global_elapsed,
                state.value,
            )
            return

        per_cooldown = self._gesture_cooldowns_s.get(gesture_name, 0.0)
        last_fired = self._gesture_last_fired.get(gesture_name, 0.0)
        per_elapsed = now - last_fired
        if per_elapsed < per_cooldown:
            logger.debug(
                "Gesture cue skipped: %s cooldown %.2fs remaining for %s.",
                gesture_name,
                per_cooldown - per_elapsed,
                state.value,
            )
            return

        action = None
        if gesture_name == "gesture_attention_snap":
            action = gesture_attention_snap(delay_ms=delay_ms)
        elif gesture_name == "gesture_curious_tilt":
            action = gesture_curious_tilt(delay_ms=delay_ms)
        elif gesture_name == "gesture_nod":
            action = gesture_nod(delay_ms=delay_ms)
        elif gesture_name == "gesture_idle":
            action = gesture_idle(delay_ms=delay_ms)

        if action is None:
            logger.warning("Gesture cue skipped: no action built for %s.", gesture_name)
            return

        controller.add_action_to_queue(action)
        self._gesture_last_fired[gesture_name] = now
        self._last_gesture_time = now
        logger.info(
            "Gesture cue emitted: state=%s gesture=%s delay_ms=%s global_cd=%.2fs "
            "gesture_cd=%.2fs",
            state.value,
            gesture_name,
            delay_ms,
            self._gesture_global_cooldown_s,
            per_cooldown,
        )

    def _handle_state_earcon(self, state: InteractionState) -> None:
        """Hook for earcon cues on state transitions."""
        logger.debug("Earcon cue for state: %s", state.value)

    def _record_user_input(self, text: str, *, source: str) -> None:
        clean_text = text.strip()
        if not clean_text:
            return
        memory_intent_subtype = self._classify_memory_intent(clean_text)
        memory_intent = memory_intent_subtype != "none"
        self._last_user_input_text = clean_text
        self._last_user_input_time = time.monotonic()
        self._last_user_input_source = source
        if self._is_battery_status_query(clean_text):
            self._last_user_battery_query_time = self._last_user_input_time
        self._update_topic_suppression_from_user_text(clean_text)
        self._mark_preference_recall_candidate(clean_text, source=source)
        self._prepare_turn_memory_brief(
            clean_text,
            source=source,
            memory_intent=memory_intent,
            memory_intent_subtype=memory_intent_subtype,
        )

    def _is_memory_intent(self, text: str) -> bool:
        return self._get_memory_runtime().is_memory_intent(text)

    def _classify_memory_intent(self, text: str) -> str:
        return self._get_memory_runtime().classify_memory_intent(text)

    def _get_memory_runtime(self) -> MemoryRuntime:
        runtime = getattr(self, "_memory_runtime", None)
        if runtime is None:
            runtime = MemoryRuntime(self)
            self._memory_runtime = runtime
        return runtime

    def _extract_preference_keywords(self, text: str) -> list[str]:
        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_+-]{1,}", text.lower())
        keywords: list[str] = []
        for token in tokens:
            if token in _PREFERENCE_KEYWORD_STOPWORDS:
                continue
            if token not in keywords:
                keywords.append(token)
        return keywords

    def _build_preference_recall_query(self, user_text: str, *, keywords: list[str]) -> str:
        canonical = list(_PREFERENCE_QUERY_FALLBACK_CANONICAL)
        domain_hits = [domain for domain in _PREFERENCE_RECALL_DOMAINS if domain in user_text]
        for domain in _PREFERENCE_RECALL_DOMAINS:
            if domain in user_text:
                canonical = list(_PREFERENCE_QUERY_CANONICAL_BY_DOMAIN.get(domain, canonical))
                break

        entity_tokens: list[str] = []
        for marker in ("prefer", "preferred", "favorite", "favourite"):
            for match in re.finditer(rf"\b{marker}\b(?P<entity>[^?.!,;:]{{0,40}})", user_text):
                entity = " ".join(match.group("entity").split())
                for token in self._extract_preference_keywords(entity):
                    if token not in entity_tokens:
                        entity_tokens.append(token)

        normalized_keywords: list[str] = []
        for keyword in keywords:
            normalized = keyword.strip().lower()
            if not normalized or normalized in _PREFERENCE_QUERY_NOISE_TOKENS:
                continue
            if normalized not in normalized_keywords:
                normalized_keywords.append(normalized)

        preference_cue_tokens: list[str] = []
        if any(marker in user_text for marker in ("favorite", "favourite", "prefer", "preferred")):
            preference_cue_tokens.extend(["favorite", "preferred"])

        domain_tokens: list[str] = []
        for domain in domain_hits:
            if domain not in domain_tokens:
                domain_tokens.append(domain)
        for token in entity_tokens:
            if token in _PREFERENCE_RECALL_DOMAINS and token not in domain_tokens:
                domain_tokens.append(token)

        keyword_entities = [
            token
            for token in normalized_keywords
            if token not in domain_tokens and token not in preference_cue_tokens
        ]

        ordered_parts: list[str] = []
        # Keep semantically important preference cues first so low token limits do not drop
        # either the requested domain (editor) or concrete entity (vim).
        for item in domain_tokens + keyword_entities + preference_cue_tokens + canonical + normalized_keywords + entity_tokens:
            normalized = item.strip().lower()
            if normalized and normalized not in ordered_parts:
                ordered_parts.append(normalized)
        if not ordered_parts:
            return "user preference"
        return " ".join(ordered_parts[:8])

    def _build_preference_recall_query_variants(self, query: str) -> list[tuple[str, str]]:
        normalized_tokens = [
            token
            for token in self._extract_preference_keywords(query)
            if token not in _PREFERENCE_RECALL_VARIANT_NOISE_TOKENS
        ]
        if not normalized_tokens:
            candidate = " ".join((query or "").strip().lower().split())
            return [(candidate or "user preference", "canonical")]
        domain_tokens = [token for token in normalized_tokens if token in _PREFERENCE_RECALL_DOMAINS]
        marker_tokens = [token for token in normalized_tokens if token in {"favorite", "preferred", "preference", "prefer"}]
        value_tokens = [token for token in normalized_tokens if token not in domain_tokens and token not in marker_tokens]

        variants: list[tuple[str, str]] = []

        def _append_variant(parts: list[str], variant_class: str) -> None:
            candidate = " ".join(part.strip().lower() for part in parts if part).strip()
            if candidate and all(existing != candidate for existing, _ in variants):
                variants.append((candidate, variant_class))

        if domain_tokens:
            _append_variant(domain_tokens[:2], "domain_only")
            _append_variant(["preferred"] + domain_tokens + value_tokens, "canonical")
            _append_variant(["favorite"] + domain_tokens + value_tokens, "canonical")
            for domain in domain_tokens:
                for canonical in _PREFERENCE_QUERY_CANONICAL_BY_DOMAIN.get(domain, ()):
                    canonical_tokens = [
                        token.strip().lower()
                        for token in canonical.split()
                        if token.strip().lower() not in _PREFERENCE_RECALL_VARIANT_NOISE_TOKENS
                    ]
                    _append_variant(canonical_tokens, "canonical")
        _append_variant(normalized_tokens[:8], "canonical")
        if domain_tokens:
            _append_variant(domain_tokens + marker_tokens + value_tokens, "expanded")
        if value_tokens:
            _append_variant(value_tokens + domain_tokens, "expanded")
        _append_variant(domain_tokens + ["preference"], "canonical")
        return variants

    def _strict_preference_domain_query(self, query: str) -> str | None:
        query_tokens = [
            token
            for token in self._extract_preference_keywords(query)
            if token not in _PREFERENCE_RECALL_VARIANT_NOISE_TOKENS
        ]
        domain_tokens = [token for token in query_tokens if token in _PREFERENCE_RECALL_DOMAINS]
        if not domain_tokens:
            return None
        return " ".join(domain_tokens[:2]).strip().lower() or None

    def _preference_recall_memories_from_payload(self, payload: dict[str, Any] | None) -> list[dict[str, Any]]:
        if not isinstance(payload, dict):
            return []
        memories = payload.get("memories")
        if isinstance(memories, list):
            return [item for item in memories if isinstance(item, dict)]
        items = payload.get("items")
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
        return []

    def _preference_recall_fallback_query(self, query: str) -> str | None:
        query_tokens = self._extract_preference_keywords(query)
        if not query_tokens:
            return None
        marker_present = any(marker in query for marker in _PREFERENCE_FALLBACK_MARKERS)
        if not marker_present:
            return None
        fallback_tokens = [token for token in query_tokens if token in _PREFERENCE_RECALL_DOMAINS]
        if not fallback_tokens:
            fallback_tokens = query_tokens[:1]
        fallback_query = " ".join(fallback_tokens[:2]).strip().lower()
        return fallback_query or None

    def _filter_preference_recall_payload_for_user(self, payload: dict[str, Any], *, query: str) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return {"memories": [], "memory_cards": [], "memory_cards_text": ""}
        cards = payload.get("memory_cards")
        if not isinstance(cards, list):
            return payload
        min_semantic = float(getattr(self, "_preference_recall_min_semantic_score", 0.15))
        min_lexical = float(getattr(self, "_preference_recall_min_lexical_score", 0.1))
        allow_low_score_debug = bool(getattr(self, "_preference_recall_allow_low_score_debug", False))
        query_tokens = set(self._extract_preference_keywords(query))
        filtered_cards: list[dict[str, Any]] = []
        filtered_memories: list[dict[str, Any]] = []
        dropped_count = 0
        memories = self._preference_recall_memories_from_payload(payload)

        for idx, card in enumerate(cards):
            if not isinstance(card, dict):
                continue
            why = str(card.get("why_relevant", ""))
            semantic_match = re.search(r"semantic similarity=([0-9]*\.?[0-9]+)", why.lower())
            lexical_match = re.search(r"lexical score=([0-9]*\.?[0-9]+)", why.lower())
            semantic_score = float(semantic_match.group(1)) if semantic_match else 0.0
            lexical_score = float(lexical_match.group(1)) if lexical_match else 0.0
            memory_text = str(card.get("memory", ""))
            memory_tokens = set(self._extract_preference_keywords(memory_text))
            overlaps = bool(query_tokens and query_tokens.intersection(memory_tokens))
            keep = (
                overlaps
                or semantic_score >= min_semantic
                or lexical_score >= min_lexical
                or "lexical exact match" in why.lower()
            )
            if keep or allow_low_score_debug:
                filtered_cards.append(card)
                if idx < len(memories):
                    filtered_memories.append(memories[idx])
            else:
                dropped_count += 1

        if dropped_count > 0:
            logger.debug(
                "preference_recall_cards_filtered run_id=%s dropped=%s query=%s min_semantic=%.2f min_lexical=%.2f",
                self._current_run_id() or "",
                dropped_count,
                query,
                min_semantic,
                min_lexical,
            )
        filtered_payload = dict(payload)
        filtered_payload["memory_cards"] = filtered_cards
        filtered_payload["memories"] = filtered_memories
        filtered_payload["memory_cards_text"] = render_memory_cards_for_assistant(
            filtered_cards,
            total_memories=len(filtered_memories),
        )
        return filtered_payload

    async def _run_preference_recall_with_fallbacks(
        self,
        *,
        recall_fn: Any,
        source: str,
        resolved_turn_id: str,
        query: str,
    ) -> tuple[dict[str, Any], bool]:
        scope = str(getattr(self, "_memory_retrieval_scope", MemoryScope.USER_GLOBAL.value))
        recall_query_variants = self._build_preference_recall_query_variants(query)
        fallback_query = self._preference_recall_fallback_query(query)
        strict_domain_query = self._strict_preference_domain_query(query)
        max_attempts = int(getattr(self, "_preference_recall_max_attempts", 1))
        if fallback_query and max_attempts > 1:
            fallback_variant = (fallback_query, "canonical")
            if all(existing != fallback_query for existing, _ in recall_query_variants):
                recall_query_variants.append(fallback_variant)

        planned_attempts = recall_query_variants[:max_attempts]
        if (
            strict_domain_query
            and max_attempts > 0
            and all(existing_query != strict_domain_query for existing_query, _ in planned_attempts)
        ):
            strict_variant = (strict_domain_query, "domain_only")
            if len(planned_attempts) >= max_attempts:
                planned_attempts = planned_attempts[: max_attempts - 1]
            planned_attempts.append(strict_variant)

        last_empty_reason = "no_ranked_matches"
        for attempt_index, (candidate_query, variant_class) in enumerate(planned_attempts):
            attempted_variants = planned_attempts[: attempt_index + 1]
            query_lineage = ";".join(
                f"{index}:{attempted_query}|{attempted_variant_class}"
                for index, (attempted_query, attempted_variant_class) in enumerate(attempted_variants)
            )
            logger.debug(
                "preference_recall_query_variant run_id=%s resolved_turn_id=%s query_variant_index=%s variant_class=%s query=%s max_attempts=%s",
                self._current_run_id() or "",
                resolved_turn_id,
                attempt_index,
                variant_class,
                candidate_query,
                max_attempts,
            )
            payload = await recall_fn(query=candidate_query, limit=3, scope=scope)
            payload = self._filter_preference_recall_payload_for_user(
                payload if isinstance(payload, dict) else {},
                query=candidate_query,
            )
            payload_keys = sorted(payload.keys()) if isinstance(payload, dict) else []
            memories = self._preference_recall_memories_from_payload(payload)
            cards = payload.get("memory_cards") if isinstance(payload, dict) else None
            cards_list = cards if isinstance(cards, list) else []
            cards_count = len(cards_list)
            trace_present = isinstance(payload.get("trace"), dict) if isinstance(payload, dict) else False
            trace = payload.get("trace") if trace_present else {}
            candidate_counts = trace.get("candidate_counts") if isinstance(trace.get("candidate_counts"), dict) else {}
            retrieval_backend = str(trace.get("retrieval_mode", "unknown"))
            filters_applied = [
                "min_semantic_score",
                "min_lexical_score",
                "token_overlap",
            ]
            memory_cards_text = (
                str(payload.get("memory_cards_text", "")).strip() if isinstance(payload, dict) else ""
            )
            payload_returned_count = payload.get("returned_count") if isinstance(payload, dict) else None
            derived_returned_count = len(memories) + cards_count
            returned_count = payload_returned_count if isinstance(payload_returned_count, int) and payload_returned_count >= 0 else derived_returned_count
            hit = bool(returned_count > 0 or derived_returned_count > 0)

            if hit and not memory_cards_text:
                best_memory = ""
                if cards_list and isinstance(cards_list[0], dict):
                    best_memory = str(cards_list[0].get("memory", "")).strip()
                if not best_memory and memories and isinstance(memories[0], dict):
                    best_memory = str(memories[0].get("content", "")).strip()
                if best_memory:
                    memory_cards_text = f'Relevant memory:\n- "{best_memory}"'
                    payload["memory_cards_text"] = memory_cards_text

            empty_reason = "none"
            if not hit:
                if not payload_keys:
                    empty_reason = "invalid_payload"
                elif not trace_present:
                    empty_reason = "no_hit_trace_unavailable"
                elif candidate_counts.get("lexical_candidates", 0) == 0 and candidate_counts.get("semantic_candidates", 0) == 0:
                    empty_reason = "no_candidates"
                elif not cards_count and memories:
                    empty_reason = "filtered_by_card_thresholds"
                else:
                    empty_reason = "no_ranked_matches"
            if isinstance(payload, dict):
                payload["hit"] = hit
                payload["returned_count"] = returned_count
                payload["empty_reason"] = empty_reason
            last_empty_reason = empty_reason

            logger.info(
                "preference_recall_tool_result run_id=%s resolved_turn_id=%s query=%s payload_keys=%s "
                "memories_count=%s cards_count=%s memory_cards_text_len=%s scope=%s attempt=%s source=%s "
                "retrieval_backend=%s filters_applied=%s candidate_count=%s returned_count=%s empty_reason=%s query_lineage=%s",
                self._current_run_id() or "",
                resolved_turn_id,
                candidate_query,
                ",".join(payload_keys),
                len(memories),
                cards_count,
                len(memory_cards_text),
                scope,
                attempt_index,
                source,
                retrieval_backend,
                ",".join(filters_applied),
                candidate_counts.get("combined_candidates", candidate_counts.get("lexical_candidates", 0)),
                returned_count,
                empty_reason,
                query_lineage,
            )
            logger.debug(
                "preference_recall_attempt_result run_id=%s resolved_turn_id=%s query_variant_index=%s variant_class=%s hit=%s",
                self._current_run_id() or "",
                resolved_turn_id,
                attempt_index,
                variant_class,
                hit,
            )
            self._preference_recall_cache[candidate_query] = {"timestamp": time.monotonic(), "payload": payload}
            if hit:
                if attempt_index > 0:
                    logger.info(
                        "preference_recall_fallback_hit run_id=%s resolved_turn_id=%s primary_query=%s fallback_query=%s scope=%s",
                        self._current_run_id() or "",
                        resolved_turn_id,
                        query,
                        candidate_query,
                        scope,
                    )
                return payload, True
        return {
            "memories": [],
            "memory_cards": [],
            "memory_cards_text": "",
            "hit": False,
            "returned_count": 0,
            "empty_reason": last_empty_reason,
        }, True

    def _is_preference_recall_intent(self, text: str) -> tuple[bool, list[str]]:
        normalized = " ".join((text or "").lower().split())
        if not normalized:
            return False, []
        memory_intent_subtype = self._classify_memory_intent(normalized)
        if memory_intent_subtype == "preference_recall":
            return True, self._extract_preference_keywords(normalized)
        has_marker = any(marker in normalized for marker in _PREFERENCE_RECALL_MARKERS)
        has_domain = any(domain in normalized for domain in _PREFERENCE_RECALL_DOMAINS)
        if not has_marker or not has_domain:
            return False, []
        return True, self._extract_preference_keywords(normalized)

    def _build_preference_query_fingerprint(self, *, query: str, source: str) -> str:
        payload = json.dumps(
            {
                "query": " ".join((query or "").lower().split()),
                "source": str(source or "").strip().lower(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _sanitize_memory_cards_text_for_user(self, text: str) -> str:
        """Strip retrieval-ID phrasing from memory card text before user delivery."""

        cleaned_lines: list[str] = []
        for raw_line in (text or "").splitlines():
            line = str(raw_line).strip()
            if not line:
                continue
            line = re.sub(r"\bmemory\s*#\s*\w+\b", "memory", line, flags=re.IGNORECASE)
            line = re.sub(r"\bmemory\s+id\s*[:#-]?\s*\w+\b", "memory", line, flags=re.IGNORECASE)
            line = re.sub(r"\bID\s*[:#-]?\s*\w+\b", "", line)
            line = " ".join(line.split())
            if line:
                cleaned_lines.append(line)
        return "\n".join(cleaned_lines)

    def _is_memory_recall_answer_text(self, text: str) -> bool:
        normalized = " ".join(str(text or "").strip().lower().split())
        if not normalized:
            return False
        return any(marker in normalized for marker in _MEMORY_RECALL_ANSWER_MARKERS)

    def _matches_memory_reassurance_suffix(self, line: str) -> bool:
        normalized = " ".join(str(line or "").strip().split())
        if not normalized:
            return False
        return any(pattern.match(normalized) for pattern in _MEMORY_REASSURANCE_SUFFIX_PATTERNS)

    def _normalize_memory_recall_answer(self, text: str) -> str:
        if not self._is_memory_recall_answer_text(text):
            return text
        raw_lines = [str(line).strip() for line in str(text or "").splitlines()]
        lines = [line for line in raw_lines if line]
        if len(lines) < 2:
            return text
        suffix_start = len(lines)
        for index in range(len(lines) - 1, -1, -1):
            if self._matches_memory_reassurance_suffix(lines[index]):
                suffix_start = index
                continue
            break
        if suffix_start >= len(lines):
            return text
        suffix_count = len(lines) - suffix_start
        if suffix_count < 2:
            return text
        normalized_lines = lines[:suffix_start] + [_MEMORY_RECALL_CONCISE_FOLLOWUP]
        return "\n".join(normalized_lines)

    def _append_assistant_reply_text(
        self,
        text: str,
        *,
        allow_separator: bool = True,
        response_id: str | None = None,
    ) -> None:
        """Append assistant text while optionally inserting a boundary separator."""

        segment = str(text or "")
        if not segment:
            return
        active_response_id = str(getattr(self, "_active_response_id", "") or "").strip()
        normalized_response_id = str(response_id or "").strip()
        if normalized_response_id and active_response_id and normalized_response_id != active_response_id:
            logger.debug(
                "assistant_text_delta_discarded reason=inactive_response response_id=%s active_response_id=%s",
                normalized_response_id,
                active_response_id,
            )
            return
        buffer_response_id = str(getattr(self, "_assistant_reply_response_id", "") or "").strip()
        if normalized_response_id and buffer_response_id and normalized_response_id != buffer_response_id:
            self.assistant_reply = ""
            self._assistant_reply_accum = ""
        if normalized_response_id:
            self._assistant_reply_response_id = normalized_response_id
        elif active_response_id and not buffer_response_id:
            self._assistant_reply_response_id = active_response_id
        prior = str(getattr(self, "_assistant_reply_accum", "") or "")
        needs_separator = (
            allow_separator
            and bool(prior)
            and not prior.endswith((" ", "\n", "\t"))
            and not segment.startswith((" ", "\n", "\t", ".", ",", "!", "?", ":", ";"))
        )
        if needs_separator:
            self.assistant_reply += " "
            self._assistant_reply_accum += " "
        self.assistant_reply += segment
        self._assistant_reply_accum += segment

    def _memory_pin_followup_needed(
        self,
        *,
        cards: list[dict[str, Any]],
        memories: list[dict[str, Any]],
        query: str,
    ) -> bool:
        confidence_needs_help = any(
            str(card.get("confidence", "")).strip().lower() in {"medium", "low"}
            for card in cards
            if isinstance(card, dict)
        )
        likely_to_repeat = "prefer" in query or "favorite" in query or "favourite" in query
        if not likely_to_repeat:
            likely_to_repeat = any(
                "preference" in {str(tag).strip().lower() for tag in memory.get("tags", []) if isinstance(tag, str)}
                for memory in memories
                if isinstance(memory, dict)
            )
        return confidence_needs_help or likely_to_repeat

    def _mark_preference_recall_candidate(self, text: str, *, source: str) -> None:
        matched, keywords = self._is_preference_recall_intent(text)
        if not matched:
            self._pending_preference_recall_trace = None
            return
        query = self._build_preference_recall_query(text.lower(), keywords=keywords)
        self._pending_preference_recall_trace = {
            "intent": "preference_recall",
            "decision": "skipped_tool",
            "reason": "model_did_not_request_tool",
            "source": source,
            "query_fingerprint": self._build_preference_query_fingerprint(query=query, source=source),
            "query": query,
        }

    def _clear_preference_recall_candidate(self) -> None:
        self._pending_preference_recall_trace = None

    def _current_turn_id_or_unknown(self) -> str:
        turn_id = str(getattr(self, "_current_response_turn_id", "") or "").strip()
        return turn_id or "turn-unknown"

    def _current_utterance_seq(self) -> int:
        active = getattr(self, "_active_utterance", None)
        if isinstance(active, dict):
            utterance_id = active.get("utterance_id")
            if isinstance(utterance_id, int):
                return utterance_id
        return int(getattr(self, "_utterance_counter", 0) or 0)

    def _build_utterance_context(
        self,
        *,
        turn_id: str | None = None,
        input_event_key: str | None = None,
        utterance_seq: int | None = None,
    ) -> UtteranceContext:
        return self._lifecycle_state_coordinator().build_utterance_context(
            run_id=self._current_run_id(),
            turn_id=str(turn_id or self._current_turn_id_or_unknown()).strip() or "turn-unknown",
            input_event_key=str(
                input_event_key
                if input_event_key is not None
                else getattr(self, "_current_input_event_key", "")
                or ""
            ).strip(),
            utterance_seq=int(utterance_seq or self._current_utterance_seq() or 0),
        )

    def _lifecycle_state_coordinator(self) -> LifecycleStateCoordinator:
        coordinator = getattr(self, "_lifecycle_state_coordinator_instance", None)
        if not isinstance(coordinator, LifecycleStateCoordinator):
            coordinator = LifecycleStateCoordinator()
            self._lifecycle_state_coordinator_instance = coordinator
        return coordinator

    @contextmanager
    def _utterance_context_scope(
        self,
        *,
        turn_id: str | None = None,
        input_event_key: str | None = None,
        utterance_seq: int | None = None,
        restore_on_exit: bool = False,
    ) -> Iterator[UtteranceContext]:
        prior_context = getattr(self, "_utterance_context", None)
        prior_turn_id = getattr(self, "_current_response_turn_id", None)
        prior_input_event_key = getattr(self, "_current_input_event_key", None)
        context = self._build_utterance_context(
            turn_id=turn_id,
            input_event_key=input_event_key,
            utterance_seq=utterance_seq,
        )
        self._clear_stale_pending_micro_ack_markers_for_turn_transition(next_turn_id=context.turn_id)
        self._utterance_context = context
        self._current_response_turn_id = context.turn_id
        self._current_input_event_key = context.input_event_key or None
        try:
            yield context
        finally:
            if restore_on_exit:
                self._utterance_context = prior_context
                self._current_response_turn_id = prior_turn_id
                self._current_input_event_key = prior_input_event_key

    def _active_input_event_key_for_turn(self, turn_id: str) -> str:
        active_by_turn = getattr(self, "_active_input_event_key_by_turn_id", None)
        if not isinstance(active_by_turn, dict):
            active_by_turn = {}
            self._active_input_event_key_by_turn_id = active_by_turn
        return self._lifecycle_state_coordinator().active_input_event_key_for_turn(active_by_turn, turn_id=turn_id)

    def _bind_active_input_event_key_for_turn(
        self,
        *,
        turn_id: str,
        input_event_key: str | None,
        cause: str = "unspecified",
        response_id: str | None = None,
        origin: str | None = None,
    ) -> None:
        active_by_turn = getattr(self, "_active_input_event_key_by_turn_id", None)
        if not isinstance(active_by_turn, dict):
            active_by_turn = {}
            self._active_input_event_key_by_turn_id = active_by_turn
        normalized_turn_id = str(turn_id or "").strip() or "turn-unknown"
        old_active_key = str(active_by_turn.get(normalized_turn_id) or "").strip()
        requested_key = str(input_event_key or "").strip()
        should_block_tool_rebind = (
            bool(old_active_key)
            and not old_active_key.startswith("tool:")
            and requested_key.startswith("tool:")
        )
        resolved_key = old_active_key if should_block_tool_rebind else requested_key
        self._lifecycle_state_coordinator().bind_active_input_event_key_for_turn(
            active_by_turn,
            turn_id=normalized_turn_id,
            input_event_key=resolved_key,
        )
        if should_block_tool_rebind:
            logger.info(
                "active_key_transition run_id=%s turn_id=%s old_active_key=%s new_active_key=%s requested_key=%s cause=%s response_id=%s origin=%s blocked_tool_rebind=true",
                self._current_run_id() or "",
                normalized_turn_id,
                old_active_key or "none",
                resolved_key or "none",
                requested_key or "none",
                cause,
                str(response_id or "").strip() or "unknown",
                str(origin or "").strip() or "unknown",
            )
            return
        if old_active_key != resolved_key:
            logger.info(
                "active_key_transition run_id=%s turn_id=%s old_active_key=%s new_active_key=%s cause=%s response_id=%s origin=%s blocked_tool_rebind=false",
                self._current_run_id() or "",
                normalized_turn_id,
                old_active_key or "none",
                resolved_key or "none",
                cause,
                str(response_id or "").strip() or "unknown",
                str(origin or "").strip() or "unknown",
            )

    def _log_response_binding_event(self, *, response_key: str, turn_id: str, origin: str) -> None:
        logger.info(
            "response_binding run_id=%s active_key=%s response_key=%s turn_id=%s origin=%s",
            self._current_run_id() or "",
            self._active_input_event_key_for_turn(turn_id) or "unknown",
            str(response_key or "").strip() or "unknown",
            turn_id or "turn-unknown",
            origin or "unknown",
        )

    def _log_parent_binding_snapshot(
        self,
        *,
        turn_id: str,
        response_id: str | None,
        origin: str,
        input_event_key: str,
        response_key: str,
        response_metadata: dict[str, Any] | None,
    ) -> None:
        metadata = response_metadata if isinstance(response_metadata, dict) else {}
        logger.info(
            "parent_binding_snapshot run_id=%s turn_id=%s response_id=%s origin=%s input_event_key=%s active_key=%s response_key=%s active_input_event_key=%s active_canonical_key=%s parent_input_event_key=%s parent_turn_id=%s tool_call_id=%s",
            self._current_run_id() or "",
            turn_id or "turn-unknown",
            str(response_id or "").strip() or "unknown",
            origin or "unknown",
            input_event_key or "unknown",
            self._active_input_event_key_for_turn(turn_id) or "unknown",
            response_key or "unknown",
            str(getattr(self, "_active_response_input_event_key", "") or "").strip() or "none",
            str(getattr(self, "_active_response_canonical_key", "") or "").strip() or "none",
            str(metadata.get("parent_input_event_key") or "").strip() or "none",
            str(metadata.get("parent_turn_id") or "").strip() or "none",
            str(metadata.get("tool_call_id") or "").strip() or "none",
        )

    def _clear_stale_pending_server_auto_for_turn(
        self,
        *,
        turn_id: str,
        active_input_event_key: str,
        reason: str,
    ) -> None:
        pending_server_auto_keys = getattr(self, "_pending_server_auto_input_event_keys", None)
        if not isinstance(pending_server_auto_keys, deque) or not pending_server_auto_keys:
            return
        normalized_active_key = str(active_input_event_key or "").strip()
        if not normalized_active_key:
            return
        kept: deque[str] = deque(maxlen=pending_server_auto_keys.maxlen)
        dropped = 0
        for queued_key in pending_server_auto_keys:
            normalized = str(queued_key or "").strip()
            if normalized and normalized != normalized_active_key:
                dropped += 1
                continue
            kept.append(normalized)
        if dropped <= 0:
            return
        self._pending_server_auto_input_event_keys = kept
        self._clear_pending_response_contenders(
            turn_id=turn_id,
            input_event_key=normalized_active_key,
            reason=reason,
        )

    def _invalidate_provisional_tool_followups_for_turn(
        self,
        *,
        turn_id: str,
        provisional_parent_input_event_key: str,
        reason: str,
    ) -> None:
        normalized_turn_id = str(turn_id or "").strip() or "turn-unknown"
        normalized_parent_key = str(provisional_parent_input_event_key or "").strip()
        if not normalized_parent_key or not self._is_synthetic_input_event_key(normalized_parent_key):
            return

        def _is_provisional_tool_followup(event: dict[str, Any], *, fallback_turn_id: str) -> tuple[bool, str]:
            metadata = self._extract_response_create_metadata(event)
            if str(metadata.get("tool_followup", "")).strip().lower() not in {"true", "1", "yes"}:
                return False, ""
            event_turn_id = str(metadata.get("parent_turn_id") or metadata.get("turn_id") or fallback_turn_id or "").strip() or fallback_turn_id
            parent_input_event_key = str(metadata.get("parent_input_event_key") or "").strip()
            if event_turn_id != normalized_turn_id or parent_input_event_key != normalized_parent_key:
                return False, ""
            canonical_key = self._canonical_utterance_key(
                turn_id=str(metadata.get("turn_id") or fallback_turn_id or "").strip() or fallback_turn_id,
                input_event_key=str(metadata.get("input_event_key") or "").strip(),
            )
            return True, canonical_key

        dropped_pending = 0
        pending = getattr(self, "_pending_response_create", None)
        if pending is not None and isinstance(getattr(pending, "event", None), dict):
            is_match, canonical_key = _is_provisional_tool_followup(
                pending.event,
                fallback_turn_id=str(getattr(pending, "turn_id", "") or normalized_turn_id),
            )
            if is_match:
                if canonical_key:
                    self._set_tool_followup_state(
                        canonical_key=canonical_key,
                        state="dropped",
                        reason=f"transcript_final_handoff_provisional_lineage:{reason}",
                    )
                self._pending_response_create = None
                dropped_pending = 1

        dropped_queue = 0
        queue = getattr(self, "_response_create_queue", None)
        if isinstance(queue, deque) and queue:
            retained: deque[dict[str, Any]] = deque()
            for queued in queue:
                if not isinstance(queued, dict) or not isinstance(queued.get("event"), dict):
                    retained.append(queued)
                    continue
                is_match, canonical_key = _is_provisional_tool_followup(
                    queued["event"],
                    fallback_turn_id=str(queued.get("turn_id") or normalized_turn_id),
                )
                if not is_match:
                    retained.append(queued)
                    continue
                if canonical_key:
                    self._set_tool_followup_state(
                        canonical_key=canonical_key,
                        state="dropped",
                        reason=f"transcript_final_handoff_provisional_lineage:{reason}",
                    )
                dropped_queue += 1
            self._response_create_queue = retained

        if dropped_pending or dropped_queue:
            self._sync_pending_response_create_queue()
            logger.info(
                "provisional_tool_followup_invalidated run_id=%s turn_id=%s parent_input_event_key=%s reason=%s dropped_pending=%s dropped_queue=%s",
                self._current_run_id() or "",
                normalized_turn_id,
                normalized_parent_key,
                str(reason or "unknown").strip() or "unknown",
                dropped_pending,
                dropped_queue,
            )

    def _record_pending_server_auto_response(
        self,
        *,
        turn_id: str,
        response_id: str | None,
        canonical_key: str,
        upgrade_chain_id: str = "",
    ) -> None:
        normalized_turn_id = str(turn_id or "").strip() or "turn-unknown"
        normalized_response_id = str(response_id or "").strip()
        if not normalized_response_id:
            return
        by_response_id, response_id_by_turn_id, legacy_by_turn_id = self._pending_server_auto_response_stores()
        prior_response_id = str(response_id_by_turn_id.get(normalized_turn_id) or "").strip()
        if prior_response_id and prior_response_id != normalized_response_id:
            prior_pending = by_response_id.get(prior_response_id)
            if isinstance(prior_pending, PendingServerAutoResponse):
                prior_pending.active = False
        pending = by_response_id.get(normalized_response_id)
        if not isinstance(pending, PendingServerAutoResponse):
            pending = PendingServerAutoResponse(
                turn_id=normalized_turn_id,
                response_id=normalized_response_id,
                canonical_key=str(canonical_key or "").strip(),
                created_at_ms=int(time.time() * 1000),
                active=True,
                pre_audio_hold=False,
                upgrade_chain_id=str(upgrade_chain_id or "").strip(),
            )
            by_response_id[normalized_response_id] = pending
        else:
            pending.turn_id = normalized_turn_id
            pending.canonical_key = str(canonical_key or "").strip()
            pending.active = True
            pending.cancelled_for_upgrade = False
            pending.pre_audio_hold = False
            pending.upgrade_chain_id = str(upgrade_chain_id or pending.upgrade_chain_id or "").strip()
        response_id_by_turn_id[normalized_turn_id] = normalized_response_id
        legacy_by_turn_id[normalized_turn_id] = pending

    def _pending_server_auto_response_stores(
        self,
    ) -> tuple[dict[str, PendingServerAutoResponse], dict[str, str], dict[str, PendingServerAutoResponse]]:
        by_response_id = getattr(self, "_pending_server_auto_response_by_response_id", None)
        if not isinstance(by_response_id, dict):
            by_response_id = {}
            self._pending_server_auto_response_by_response_id = by_response_id
        response_id_by_turn_id = getattr(self, "_pending_server_auto_response_id_by_turn_id", None)
        if not isinstance(response_id_by_turn_id, dict):
            response_id_by_turn_id = {}
            self._pending_server_auto_response_id_by_turn_id = response_id_by_turn_id
        legacy_by_turn_id = getattr(self, "_pending_server_auto_response_by_turn_id", None)
        if not isinstance(legacy_by_turn_id, dict):
            legacy_by_turn_id = {}
            self._pending_server_auto_response_by_turn_id = legacy_by_turn_id
        for legacy_turn_id, legacy_pending in tuple(legacy_by_turn_id.items()):
            if not isinstance(legacy_pending, PendingServerAutoResponse):
                continue
            legacy_response_id = str(legacy_pending.response_id or "").strip()
            normalized_turn_id = str(legacy_turn_id or "").strip() or str(legacy_pending.turn_id or "").strip() or "turn-unknown"
            if not legacy_response_id:
                continue
            by_response_id.setdefault(legacy_response_id, legacy_pending)
            response_id_by_turn_id.setdefault(normalized_turn_id, legacy_response_id)
            legacy_by_turn_id[normalized_turn_id] = by_response_id[legacy_response_id]
        return by_response_id, response_id_by_turn_id, legacy_by_turn_id

    def _pending_server_auto_response_mutation_allowed(
        self,
        *,
        pending: PendingServerAutoResponse,
        turn_id: str,
        mutation: str,
        expected_response_id: str | None = None,
        expected_upgrade_chain_id: str | None = None,
    ) -> bool:
        normalized_turn_id = str(turn_id or "").strip() or "turn-unknown"
        active_response_id = str(getattr(self, "_active_response_id", "") or "").strip()
        pending_response_id = str(pending.response_id or "").strip()
        pending_turn_id = str(getattr(pending, "turn_id", "") or "").strip() or "turn-unknown"
        expected_response_id_normalized = str(expected_response_id or "").strip()
        expected_upgrade_chain_id_normalized = str(expected_upgrade_chain_id or "").strip()
        pending_upgrade_chain_id = str(getattr(pending, "upgrade_chain_id", "") or "").strip()

        lineage_mismatch = False
        if pending_turn_id != normalized_turn_id:
            lineage_mismatch = True
        if expected_response_id_normalized and pending_response_id != expected_response_id_normalized:
            lineage_mismatch = True
        if (
            expected_upgrade_chain_id_normalized
            and pending_upgrade_chain_id
            and pending_upgrade_chain_id != expected_upgrade_chain_id_normalized
        ):
            lineage_mismatch = True
        if lineage_mismatch:
            logger.info(
                "pending_server_auto_mutation_rejected run_id=%s turn_id=%s mutation=%s reason=lineage_mismatch pending_response_id=%s active_response_id=%s expected_response_id=%s pending_chain_id=%s expected_chain_id=%s",
                self._current_run_id() or "",
                normalized_turn_id,
                mutation,
                pending_response_id or "none",
                active_response_id or "none",
                expected_response_id_normalized or "none",
                pending_upgrade_chain_id or "none",
                expected_upgrade_chain_id_normalized or "none",
            )
            return False

        if mutation == "replaced" and not pending.active and not pending.cancelled_for_upgrade:
            logger.info(
                "pending_server_auto_mutation_ignored run_id=%s turn_id=%s mutation=%s reason=idempotent_duplicate pending_response_id=%s active_response_id=%s",
                self._current_run_id() or "",
                normalized_turn_id,
                mutation,
                pending_response_id or "none",
                active_response_id or "none",
            )
            return False

        if active_response_id and pending_response_id == active_response_id:
            return True
        if mutation in {"replaced", "cancel_and_replace", "cancelled"}:
            # Transcript-final replacement cleanups are allowed when lineage still
            # matches even if active ownership has already moved or response.done
            # has cleared active ownership.
            return True
        logger.info(
            "pending_server_auto_mutation_rejected run_id=%s turn_id=%s mutation=%s reason=lineage_mismatch pending_response_id=%s active_response_id=%s",
            self._current_run_id() or "",
            normalized_turn_id,
            mutation,
            pending_response_id or "none",
            active_response_id or "none",
        )
        return False

    def _mark_response_provisional(self, *, response_id: str | None) -> None:
        normalized_response_id = str(response_id or "").strip()
        if not normalized_response_id:
            return
        provisional_ids = getattr(self, "_provisional_response_ids", None)
        if not isinstance(provisional_ids, set):
            provisional_ids = set()
            self._provisional_response_ids = provisional_ids
        provisional_ids.add(normalized_response_id)

    def _is_provisional_response(self, *, response_id: str | None) -> bool:
        normalized_response_id = str(response_id or "").strip()
        if not normalized_response_id:
            return False
        provisional_ids = getattr(self, "_provisional_response_ids", None)
        return isinstance(provisional_ids, set) and normalized_response_id in provisional_ids

    def _mark_provisional_response_completed_empty(self, *, response_id: str | None) -> None:
        normalized_response_id = str(response_id or "").strip()
        if not normalized_response_id:
            return
        completed = getattr(self, "_provisional_response_ids_completed_empty", None)
        if not isinstance(completed, set):
            completed = set()
            self._provisional_response_ids_completed_empty = completed
        completed.add(normalized_response_id)

    def _is_provisional_response_completed_empty(self, *, response_id: str | None) -> bool:
        normalized_response_id = str(response_id or "").strip()
        if not normalized_response_id:
            return False
        completed = getattr(self, "_provisional_response_ids_completed_empty", None)
        return isinstance(completed, set) and normalized_response_id in completed

    def _set_server_auto_pre_audio_hold(
        self,
        *,
        turn_id: str,
        enabled: bool,
        reason: str,
        response_id: str | None = None,
    ) -> None:
        normalized_turn_id = str(turn_id or "").strip() or "turn-unknown"
        normalized_response_id = str(response_id or "").strip()
        if not normalized_response_id:
            pending = self._pending_server_auto_response_for_turn(turn_id=normalized_turn_id)
            if pending is not None:
                normalized_response_id = str(pending.response_id or "").strip()
        if not normalized_response_id:
            normalized_response_id = str(getattr(self, "_active_response_id", "") or "").strip()
        if not normalized_response_id:
            logger.debug(
                "server_auto_pre_audio_hold_transition_rejected run_id=%s turn_id=%s response_id=unknown transition=missing_response_id reason=%s",
                self._current_run_id() or "",
                normalized_turn_id,
                reason,
            )
            return
        phase_store = getattr(self, "_server_auto_pre_audio_hold_phase_by_key", None)
        if not isinstance(phase_store, dict):
            phase_store = {}
            self._server_auto_pre_audio_hold_phase_by_key = phase_store
        hold_phase = "held" if enabled else "released"
        hold_key = (normalized_turn_id, normalized_response_id)
        prior_phase = str(phase_store.get(hold_key) or "not_started")
        duplicate_transition = (
            (prior_phase == "held" and hold_phase == "held")
            or (prior_phase == "released" and hold_phase == "released")
        )
        if duplicate_transition:
            logger.debug(
                "server_auto_pre_audio_hold_transition_duplicate run_id=%s turn_id=%s response_id=%s phase=%s reason=%s",
                self._current_run_id() or "",
                normalized_turn_id,
                normalized_response_id,
                hold_phase,
                reason,
            )
            return
        transition_allowed = (
            (prior_phase == "not_started" and hold_phase == "held")
            or (prior_phase == "held" and hold_phase == "released")
        )
        if not transition_allowed:
            logger.debug(
                "server_auto_pre_audio_hold_transition_rejected run_id=%s turn_id=%s response_id=%s prior_phase=%s requested_phase=%s reason=%s",
                self._current_run_id() or "",
                normalized_turn_id,
                normalized_response_id,
                prior_phase,
                hold_phase,
                reason,
            )
            return
        phase_store[hold_key] = hold_phase
        hold_store = getattr(self, "_server_auto_pre_audio_hold_by_turn_id", None)
        if not isinstance(hold_store, dict):
            hold_store = {}
            self._server_auto_pre_audio_hold_by_turn_id = hold_store
        now = time.time()
        pending = self._pending_server_auto_response_for_turn(turn_id=normalized_turn_id)
        if pending is not None and str(pending.response_id or "").strip() == normalized_response_id:
            pending.pre_audio_hold = bool(enabled)
        if hold_phase == "held":
            hold_store[normalized_turn_id] = ServerAutoPreAudioHoldRecord(
                turn_id=normalized_turn_id,
                response_id=normalized_response_id,
                hold_started_at=now,
                hold_released_at=None,
                last_reason=reason,
            )
        else:
            active_hold_for_turn = any(
                phase == "held" and hold_turn_id == normalized_turn_id
                for (hold_turn_id, _hold_response_id), phase in phase_store.items()
            )
            existing_record = hold_store.get(normalized_turn_id)
            if active_hold_for_turn:
                if isinstance(existing_record, ServerAutoPreAudioHoldRecord):
                    existing_record.last_reason = reason
                else:
                    hold_store[normalized_turn_id] = ServerAutoPreAudioHoldRecord(
                        turn_id=normalized_turn_id,
                        response_id=normalized_response_id,
                        hold_started_at=now,
                        hold_released_at=None,
                        last_reason=reason,
                    )
            else:
                prior_hold_started_at = (
                    existing_record.hold_started_at
                    if isinstance(existing_record, ServerAutoPreAudioHoldRecord)
                    and existing_record.response_id == normalized_response_id
                    else now
                )
                hold_store[normalized_turn_id] = ServerAutoPreAudioHoldRecord(
                    turn_id=normalized_turn_id,
                    response_id=normalized_response_id,
                    hold_started_at=prior_hold_started_at,
                    hold_released_at=now,
                    last_reason=reason,
                )
            existing_record = hold_store.get(normalized_turn_id)
            if (
                isinstance(existing_record, ServerAutoPreAudioHoldRecord)
                and existing_record.response_id == normalized_response_id
            ):
                existing_record.hold_released_at = now
                existing_record.last_reason = reason
        logger.info(
            "server_auto_pre_audio_hold run_id=%s turn_id=%s response_id=%s enabled=%s phase=%s reason=%s",
            self._current_run_id() or "",
            normalized_turn_id,
            normalized_response_id,
            str(bool(enabled)).lower(),
            hold_phase,
            reason,
        )

    def _server_auto_pre_audio_hold_active(self, *, turn_id: str, response_id: str | None = None) -> bool:
        normalized_turn_id = str(turn_id or "").strip() or "turn-unknown"
        normalized_response_id = str(response_id or "").strip()
        phase_store = getattr(self, "_server_auto_pre_audio_hold_phase_by_key", None)
        if normalized_response_id and isinstance(phase_store, dict):
            return str(phase_store.get((normalized_turn_id, normalized_response_id)) or "not_started") == "held"
        hold_store = getattr(self, "_server_auto_pre_audio_hold_by_turn_id", None)
        if not isinstance(hold_store, dict):
            return False
        hold_record = hold_store.get(normalized_turn_id)
        if not isinstance(hold_record, ServerAutoPreAudioHoldRecord):
            return False
        if normalized_response_id and hold_record.response_id != normalized_response_id:
            return False
        return hold_record.hold_released_at is None

    def _server_auto_pre_audio_hold_released_for_other_response(self, *, turn_id: str, response_id: str) -> bool:
        phase_store = getattr(self, "_server_auto_pre_audio_hold_phase_by_key", None)
        if not isinstance(phase_store, dict):
            return False
        normalized_turn_id = str(turn_id or "").strip() or "turn-unknown"
        normalized_response_id = str(response_id or "").strip()
        for (candidate_turn_id, candidate_response_id), phase in phase_store.items():
            if candidate_turn_id != normalized_turn_id:
                continue
            if candidate_response_id == normalized_response_id:
                continue
            if str(phase or "") == "released":
                return True
        return False

    def _pending_server_auto_response_for_turn(self, *, turn_id: str) -> PendingServerAutoResponse | None:
        normalized_turn_id = str(turn_id or "").strip() or "turn-unknown"
        by_response_id, response_id_by_turn_id, legacy_by_turn_id = self._pending_server_auto_response_stores()
        response_id = str(response_id_by_turn_id.get(normalized_turn_id) or "").strip()
        pending = by_response_id.get(response_id) if response_id else None
        if pending is None:
            legacy_pending = legacy_by_turn_id.get(normalized_turn_id)
            if isinstance(legacy_pending, PendingServerAutoResponse):
                legacy_response_id = str(legacy_pending.response_id or "").strip()
                if legacy_response_id:
                    by_response_id.setdefault(legacy_response_id, legacy_pending)
                    response_id_by_turn_id[normalized_turn_id] = legacy_response_id
                    pending = by_response_id.get(legacy_response_id)
        if not isinstance(pending, PendingServerAutoResponse):
            return None
        legacy_by_turn_id[normalized_turn_id] = pending
        return pending

    def _mark_pending_server_auto_response_cancelled(
        self,
        *,
        turn_id: str,
        reason: str,
    ) -> PendingServerAutoResponse | None:
        pending = self._pending_server_auto_response_for_turn(turn_id=turn_id)
        if pending is None:
            return None
        if not self._pending_server_auto_response_mutation_allowed(
            pending=pending,
            turn_id=turn_id,
            mutation="cancelled",
        ):
            return None
        pending.active = False
        pending.cancelled_for_upgrade = True
        self._set_server_auto_pre_audio_hold(
            turn_id=turn_id,
            enabled=False,
            reason=f"pending_cancelled:{reason}",
            response_id=pending.response_id,
        )
        cancelled_ids = getattr(self, "_cancelled_response_ids", None)
        if not isinstance(cancelled_ids, set):
            cancelled_ids = set()
            self._cancelled_response_ids = cancelled_ids
        cancelled_ids.add(pending.response_id)
        self._set_response_status(response_id=pending.response_id, status="cancelled")
        trace_context = self._response_trace_by_id().get(pending.response_id, {})
        self._remember_stale_response_context(
            response_id=pending.response_id,
            canonical_key=str(pending.canonical_key or trace_context.get("canonical_key") or "").strip(),
            origin=str(trace_context.get("origin") or "server_auto").strip() or "server_auto",
            turn_id=str(turn_id or trace_context.get("turn_id") or "unknown").strip() or "unknown",
            input_event_key=self._active_input_event_key_for_turn(turn_id),
            upgrade_chain_id=str(getattr(pending, "upgrade_chain_id", "") or trace_context.get("upgrade_chain_id") or "").strip(),
        )
        self._clear_cancelled_response_blocking_state(
            response_id=pending.response_id,
            reason=reason,
        )
        log_label = (
            "server_auto_cancelled_for_empty_transcript"
            if str(reason or "").strip() == "empty_transcript"
            else "server_auto_cancelled_for_upgrade"
        )
        logger.info(
            "%s run_id=%s turn_id=%s response_id=%s pending_owner_response_id=%s reason=%s",
            log_label,
            self._current_run_id() or "",
            str(turn_id or "").strip() or "turn-unknown",
            pending.response_id,
            pending.response_id,
            reason,
        )
        return pending

    def _quarantine_cancelled_response_id(
        self,
        *,
        response_id: str | None,
        turn_id: str,
        input_event_key: str | None,
        origin: str,
        reason: str,
    ) -> None:
        normalized_response_id = str(response_id or "").strip()
        if not normalized_response_id:
            return
        cancelled_ids = getattr(self, "_cancelled_response_ids", None)
        if not isinstance(cancelled_ids, set):
            cancelled_ids = set()
            self._cancelled_response_ids = cancelled_ids
        cancelled_ids.add(normalized_response_id)
        self._stale_response_ids().add(normalized_response_id)
        self._set_response_status(response_id=normalized_response_id, status="cancelled")
        self._record_cancel_issued_timing(normalized_response_id)
        canonical_key = self._canonical_utterance_key(
            turn_id=turn_id,
            input_event_key=input_event_key,
        )
        self._remember_stale_response_context(
            response_id=normalized_response_id,
            canonical_key=canonical_key,
            origin=str(origin or "unknown").strip() or "unknown",
            turn_id=str(turn_id or "unknown").strip() or "unknown",
            input_event_key=str(input_event_key or "unknown").strip() or "unknown",
        )
        self._suppress_cancelled_response_audio(normalized_response_id)
        self._clear_cancelled_response_blocking_state(
            response_id=normalized_response_id,
            reason=reason,
        )

    def should_cancel_and_replace(
        self,
        *,
        server_auto_state: PendingServerAutoResponse | None,
        transcript_final_state: dict[str, Any] | None,
        pref_ctx_state: dict[str, Any] | None,
    ) -> bool:
        _ = transcript_final_state
        _ = pref_ctx_state
        if not isinstance(server_auto_state, PendingServerAutoResponse) or not server_auto_state.active:
            return False
        canonical_key = str(server_auto_state.canonical_key or "").strip()
        if canonical_key and self._canonical_first_audio_started(canonical_key):
            return False
        return True

    def _clear_canonical_terminal_delivery_state(self, *, canonical_key: str) -> None:
        normalized_key = str(canonical_key or "").strip()
        if not normalized_key:
            return
        self._canonical_response_state_mutate(
            canonical_key=normalized_key,
            turn_id=self._current_turn_id_or_unknown(),
            input_event_key=None,
            mutator=lambda record: (
                setattr(record, "cancel_sent", False),
                setattr(record, "done", False),
            ),
        )

    def _mark_canonical_cancelled_for_upgrade(
        self,
        *,
        canonical_key: str,
        turn_id: str,
        response_id: str,
    ) -> None:
        normalized_key = str(canonical_key or "").strip()
        if not normalized_key:
            return
        self._canonical_response_state_mutate(
            canonical_key=normalized_key,
            turn_id=turn_id,
            input_event_key=None,
            mutator=lambda record: (
                setattr(record, "created", True),
                setattr(record, "cancel_sent", True),
                setattr(record, "done", False),
                setattr(record, "response_id", str(response_id or "")),
            ),
        )
        self._lifecycle_controller().on_cancel_sent(normalized_key)

    def _record_stale_response_drop(self, *, response_id: str, event_type: str) -> None:
        normalized_response_id = str(response_id or "").strip() or "unknown"
        normalized_event_type = str(event_type or "").strip() or "unknown"
        now = time.monotonic()
        window_s = max(0.5, float(getattr(self, "_stale_response_drop_window_s", 3.0) or 3.0))
        store = getattr(self, "_stale_response_drop_window_by_id", None)
        if not isinstance(store, dict):
            store = {}
            self._stale_response_drop_window_by_id = store
        state = store.get(normalized_response_id)
        if not isinstance(state, dict) or (now - float(state.get("started_at", now))) > window_s:
            state = {
                "started_at": now,
                "counts": {},
                "summary_emitted": False,
            }
            store[normalized_response_id] = state
        counts = state.get("counts")
        if not isinstance(counts, dict):
            counts = {}
            state["counts"] = counts
        counts[normalized_event_type] = int(counts.get(normalized_event_type, 0)) + 1
        chain_id = self._upgrade_chain_id_from_response(normalized_response_id)
        if counts[normalized_event_type] == 1:
            logger.debug(
                "dropped_stale_response_event response_id=%s event_type=%s upgrade_chain_id=%s",
                normalized_response_id,
                normalized_event_type,
                chain_id or "none",
            )
            return
        if state.get("summary_emitted", False):
            return
        state["summary_emitted"] = True
        logger.info(
            "dropped_stale_response_event_summary response_id=%s counts=%s window_s=%s upgrade_chain_id=%s",
            normalized_response_id,
            json.dumps(counts, sort_keys=True),
            int(window_s),
            chain_id or "none",
        )

    def _stale_response_ids(self) -> set[str]:
        stale_ids = getattr(self, "_stale_response_ids_set", None)
        if not isinstance(stale_ids, set):
            stale_ids = set()
            self._stale_response_ids_set = stale_ids
        return stale_ids

    def _prune_stale_response_map(self, *, now: float | None = None) -> None:
        store = getattr(self, "_stale_response_map", None)
        if not isinstance(store, dict) or not store:
            return
        ts = time.monotonic() if now is None else float(now)
        stale_ids = [
            response_id
            for response_id, entry in store.items()
            if ts > float(entry.get("expires_at_monotonic", 0.0))
        ]
        for response_id in stale_ids:
            store.pop(response_id, None)

    def _remember_stale_response_context(
        self,
        *,
        response_id: str,
        canonical_key: str,
        origin: str,
        turn_id: str,
        input_event_key: str,
        upgrade_chain_id: str = "",
    ) -> None:
        normalized_response_id = str(response_id or "").strip()
        if not normalized_response_id:
            return
        now = time.monotonic()
        self._prune_stale_response_map(now=now)
        store = getattr(self, "_stale_response_map", None)
        if not isinstance(store, dict):
            store = {}
            self._stale_response_map = store
        ttl_s = max(1.0, float(getattr(self, "_stale_response_map_ttl_s", 15.0) or 15.0))
        existing = store.get(normalized_response_id)
        state = dict(existing) if isinstance(existing, dict) else {}
        state["canonical_key"] = str(canonical_key or state.get("canonical_key") or "").strip()
        state["origin"] = str(origin or state.get("origin") or "unknown").strip() or "unknown"
        state["turn_id"] = str(turn_id or state.get("turn_id") or "unknown").strip() or "unknown"
        state["input_event_key"] = str(
            input_event_key or state.get("input_event_key") or "unknown"
        ).strip() or "unknown"
        state["upgrade_chain_id"] = str(state.get("upgrade_chain_id") or "").strip()
        if str(upgrade_chain_id or "").strip():
            state["upgrade_chain_id"] = str(upgrade_chain_id or "").strip()
        state["expires_at_monotonic"] = now + ttl_s
        store[normalized_response_id] = state

    def _stale_response_context(self, response_id: str) -> dict[str, str]:
        normalized_response_id = str(response_id or "").strip()
        if not normalized_response_id:
            return {}
        self._prune_stale_response_map()
        store = getattr(self, "_stale_response_map", None)
        if not isinstance(store, dict):
            return {}
        state = store.get(normalized_response_id)
        if not isinstance(state, dict):
            return {}
        return {
            "canonical_key": str(state.get("canonical_key") or "").strip(),
            "origin": str(state.get("origin") or "").strip(),
            "turn_id": str(state.get("turn_id") or "").strip(),
            "input_event_key": str(state.get("input_event_key") or "").strip(),
            "upgrade_chain_id": str(state.get("upgrade_chain_id") or "").strip(),
        }

    def _clear_stale_response_context(self, response_id: str) -> None:
        normalized_response_id = str(response_id or "").strip()
        if not normalized_response_id:
            return
        store = getattr(self, "_stale_response_map", None)
        if isinstance(store, dict):
            store.pop(normalized_response_id, None)

    def _is_stale_response_event(self, event: dict[str, Any]) -> bool:
        response_id = self._response_id_from_event(event)
        return bool(response_id) and response_id in self._stale_response_ids()

    def _should_drop_stale_response_event(self, event: dict[str, Any]) -> bool:
        event_type = str(event.get("type") or "").strip()
        if event_type not in {
            "response.output_audio.delta",
            "response.output_audio.done",
            "response.output_audio_transcript.delta",
            "response.output_audio_transcript.done",
            "response.output_text.delta",
            "response.output_text.done",
            "response.text.delta",
            "response.text.done",
            "response.done",
        }:
            return False
        if not self._is_stale_response_event(event):
            return False
        response_id = self._response_id_from_event(event)
        if event_type in {"response.done", "response.completed"}:
            self._clear_stale_response_context(response_id)
        self._record_stale_response_drop(
            response_id=response_id,
            event_type=event_type,
        )
        return True

    def _mark_pending_server_auto_response_replaced(self, *, turn_id: str) -> None:
        pending = self._pending_server_auto_response_for_turn(turn_id=turn_id)
        if pending is None:
            logger.info(
                "pending_server_auto_mutation_ignored run_id=%s turn_id=%s mutation=%s reason=already_cleaned pending_response_id=none active_response_id=%s",
                self._current_run_id() or "",
                str(turn_id or "").strip() or "turn-unknown",
                "replaced",
                str(getattr(self, "_active_response_id", "") or "").strip() or "none",
            )
            return
        if not self._pending_server_auto_response_mutation_allowed(
            pending=pending,
            turn_id=turn_id,
            mutation="replaced",
            expected_response_id=pending.response_id,
            expected_upgrade_chain_id=getattr(pending, "upgrade_chain_id", ""),
        ):
            return
        pending.active = False
        pending.cancelled_for_upgrade = False
        self._set_server_auto_pre_audio_hold(
            turn_id=turn_id,
            enabled=False,
            reason="pending_replaced",
            response_id=pending.response_id,
        )

    def _should_defer_provisional_server_auto_tool_call(self) -> bool:
        active_origin = str(getattr(self, "_active_response_origin", "") or "").strip().lower()
        if active_origin != "server_auto":
            return False
        turn_id = str(getattr(self, "_current_response_turn_id", "") or "").strip()
        active_response_id = str(getattr(self, "_active_response_id", "") or "").strip()
        # Prefer the currently bound active response key over any legacy server-auto key.
        # This prevents stale synthetic keys from forcing execution deferment after
        # transcript-final handoff has already established canonical ownership.
        active_input_event_key = str(getattr(self, "_active_response_input_event_key", "") or "").strip() or str(
            getattr(self, "_active_server_auto_input_event_key", "") or ""
        ).strip()
        if not turn_id or not active_response_id or not self._is_synthetic_input_event_key(active_input_event_key):
            return False
        return self._server_auto_pre_audio_hold_active(turn_id=turn_id, response_id=active_response_id)

    def _is_cancelled_response_event(self, event: dict[str, Any]) -> bool:
        response_id = self._response_id_from_event(event)
        return self._response_status(response_id) == "cancelled"

    def _response_status(self, response_id: str | None) -> str:
        normalized_response_id = str(response_id or "").strip()
        if not normalized_response_id:
            return "active"
        status_by_id = getattr(self, "_response_status_by_id", None)
        if isinstance(status_by_id, dict):
            status = str(status_by_id.get(normalized_response_id) or "").strip().lower()
            if status in {"active", "cancelled", "terminal_done"}:
                return status
        cancelled_ids = getattr(self, "_cancelled_response_ids", None)
        if isinstance(cancelled_ids, set) and normalized_response_id in cancelled_ids:
            return "cancelled"
        suppressed_audio_ids = getattr(self, "_suppressed_audio_response_ids", None)
        if isinstance(suppressed_audio_ids, set) and normalized_response_id in suppressed_audio_ids:
            return "cancelled"
        superseded_ids = getattr(self, "_superseded_response_ids", None)
        if isinstance(superseded_ids, set) and normalized_response_id in superseded_ids:
            return "cancelled"
        if str(getattr(self, "_active_response_id", "") or "").strip() == normalized_response_id:
            return "active"
        return "active"

    def _set_response_status(self, *, response_id: str | None, status: str) -> None:
        normalized_response_id = str(response_id or "").strip()
        normalized_status = str(status or "").strip().lower()
        if not normalized_response_id or normalized_status not in {"active", "cancelled", "terminal_done"}:
            return
        status_by_id = getattr(self, "_response_status_by_id", None)
        if not isinstance(status_by_id, dict):
            status_by_id = {}
            self._response_status_by_id = status_by_id
        prior_status = str(status_by_id.get(normalized_response_id) or "").strip().lower()
        if prior_status == "cancelled" and normalized_status == "terminal_done":
            return
        status_by_id[normalized_response_id] = normalized_status

    def _is_cancelled_or_superseded_response_id(self, response_id: str | None) -> bool:
        return self._response_status(response_id) == "cancelled"

    def _upgrade_chain_id_from_response(self, response_id: str | None) -> str:
        normalized_response_id = str(response_id or "").strip()
        if not normalized_response_id:
            return ""
        trace_context = self._response_trace_by_id().get(normalized_response_id, {})
        if isinstance(trace_context, dict):
            trace_chain_id = str(trace_context.get("upgrade_chain_id") or "").strip()
            if trace_chain_id:
                return trace_chain_id
        stale_context = self._stale_response_context(normalized_response_id)
        return str(stale_context.get("upgrade_chain_id") or "").strip()

    def _ensure_upgrade_chain_id(self, *, turn_id: str, input_event_key: str, response_id: str | None) -> str:
        normalized_response_id = str(response_id or "").strip()
        existing = self._upgrade_chain_id_from_response(normalized_response_id)
        if existing:
            return existing
        run_id = str(self._current_run_id() or "").strip() or "run-unknown"
        normalized_turn_id = str(turn_id or "").strip() or "turn-unknown"
        normalized_input_event_key = str(input_event_key or "").strip() or (normalized_response_id or "unknown")
        return f"{run_id}:{normalized_turn_id}:{normalized_input_event_key}"

    def _set_upgrade_chain_trace_context(
        self,
        *,
        response_id: str | None,
        chain_id: str,
        turn_id: str,
        input_event_key: str,
        canonical_key: str,
        origin: str,
    ) -> None:
        normalized_response_id = str(response_id or "").strip()
        normalized_chain_id = str(chain_id or "").strip()
        if not normalized_response_id or not normalized_chain_id:
            return
        self._record_response_trace_context(
            normalized_response_id,
            upgrade_chain_id=normalized_chain_id,
            turn_id=turn_id,
            input_event_key=input_event_key,
            canonical_key=canonical_key,
            origin=origin,
        )

    def _should_process_response_event_ingress(self, event: dict[str, Any], *, source: str) -> bool:
        event_type = str(event.get("type") or "unknown")
        response_id = self._response_id_from_event(event)
        if self._should_drop_stale_response_event(event):
            return False
        cancelled_ids = getattr(self, "_cancelled_response_ids", set())
        if (
            event_type == "response.completed"
            and response_id
            and isinstance(cancelled_ids, set)
            and response_id in cancelled_ids
        ):
            self._log_cancelled_deliverable_once(response_id, source_event=event_type)
            self._clear_cancelled_response_tracking(response_id)
            self._log_lifecycle_coherence(
                stage="cancelled_provisional_terminalization",
                turn_id=self._current_turn_id_or_unknown(),
                response_id=response_id,
                canonical_key=str(getattr(self, "_active_response_canonical_key", "") or "").strip(),
            )
            return False
        if not self._is_cancelled_response_event(event):
            return True
        if event_type in {"response.output_audio.delta", "response.output_audio.done"}:
            self._record_cancelled_audio_race_transition(
                response_id=response_id,
                event_type=event_type,
            )
            if event_type == "response.output_audio.delta":
                logger.debug(
                    "audio_delta_suppressed run_id=%s response_id=%s event_type=response.output_audio.delta",
                    self._current_run_id() or "",
                    response_id or "unknown",
                )
        if event_type in {"response.done", "response.completed"}:
            self._set_response_status(response_id=response_id, status="terminal_done")
            self._log_cancelled_deliverable_once(response_id, source_event=event_type)
            if event_type == "response.completed":
                self._clear_cancelled_response_tracking(response_id)
            self._log_lifecycle_coherence(
                stage="cancelled_provisional_terminalization",
                turn_id=self._current_turn_id_or_unknown(),
                response_id=response_id,
                canonical_key=str(getattr(self, "_active_response_canonical_key", "") or "").strip(),
            )
        else:
            logger.debug(
                "cancelled_response_event_suppressed response_id=%s event_type=%s source=%s",
                response_id or "unknown",
                event_type or "unknown",
                source,
            )
        return False

    def _log_lifecycle_coherence(
        self,
        *,
        stage: str,
        turn_id: str,
        response_id: str | None = None,
        canonical_key: str | None = None,
    ) -> None:
        violations: list[str] = []
        normalized_turn_id = str(turn_id or "").strip() or "turn-unknown"
        normalized_response_id = str(response_id or "").strip()
        normalized_canonical_key = str(canonical_key or "").strip()
        pending = self._pending_server_auto_response_for_turn(turn_id=normalized_turn_id)
        if stage in {"transcript_final_rebind", "replacement_scheduled", "replacement_created"}:
            active_turn_map = getattr(self, "_active_input_event_key_by_turn_id", {})
            if isinstance(active_turn_map, dict) and normalized_turn_id not in active_turn_map:
                violations.append("missing_active_input_event_key_for_turn")
        if stage == "transcript_final_rebind":
            active_key = str(getattr(self, "_active_response_canonical_key", "") or "").strip()
            if normalized_canonical_key and active_key and active_key != normalized_canonical_key:
                violations.append("active_canonical_pointer_mismatch")
        if stage == "replacement_scheduled":
            if pending is not None and pending.active:
                violations.append("pending_server_auto_still_active_after_replacement_schedule")
        if stage == "replacement_created":
            if pending is None:
                violations.append("pending_server_auto_missing_on_replacement_created")
        if stage == "cancelled_provisional_terminalization" and normalized_response_id:
            if self._response_status(normalized_response_id) != "terminal_done":
                violations.append("cancelled_response_not_terminal_done")
        if violations:
            logger.warning(
                "lifecycle_coherence_violation run_id=%s stage=%s turn_id=%s response_id=%s canonical_key=%s violations=%s",
                self._current_run_id() or "",
                stage,
                normalized_turn_id,
                normalized_response_id or "none",
                normalized_canonical_key or "none",
                ",".join(violations),
            )
            return
        logger.debug(
            "lifecycle_coherence_ok run_id=%s stage=%s turn_id=%s response_id=%s canonical_key=%s",
            self._current_run_id() or "",
            stage,
            normalized_turn_id,
            normalized_response_id or "none",
            normalized_canonical_key or "none",
        )

    def _suppress_cancelled_response_audio(self, response_id: str | None) -> None:
        normalized_response_id = str(response_id or "").strip()
        if not normalized_response_id:
            return
        suppressed_audio_ids = getattr(self, "_suppressed_audio_response_ids", None)
        if not isinstance(suppressed_audio_ids, set):
            suppressed_audio_ids = set()
            self._suppressed_audio_response_ids = suppressed_audio_ids
        suppressed_audio_ids.add(normalized_response_id)
        self._set_response_status(response_id=normalized_response_id, status="cancelled")

        accum_response_id = str(getattr(self, "_audio_accum_response_id", "") or "").strip()
        if accum_response_id == normalized_response_id:
            self._audio_accum.clear()
            self._audio_accum_response_id = None

        player = getattr(self, "audio_player", None)
        if player is not None:
            cancel_current = getattr(player, "cancel_current_response", None)
            if callable(cancel_current):
                cancel_current()
            else:
                flush = getattr(player, "flush", None)
                if callable(flush):
                    flush()

    def _response_id_from_event(self, event: dict[str, Any] | None) -> str:
        if not isinstance(event, dict):
            return ""
        response_id = str(event.get("response_id") or "").strip()
        if not response_id and isinstance(event.get("response"), dict):
            response_id = str((event.get("response") or {}).get("id") or "").strip()
        return response_id

    def _log_cancelled_deliverable_once(self, response_id: str, source_event: str) -> None:
        normalized_response_id = str(response_id or "").strip()
        if not normalized_response_id:
            return
        logged_ids = getattr(self, "_cancelled_deliverable_logged_ids", None)
        if not isinstance(logged_ids, set):
            logged_ids = set()
            self._cancelled_deliverable_logged_ids = logged_ids
        if normalized_response_id in logged_ids:
            logger.debug(
                "cancelled_deliverable_log_suppressed response_id=%s source_event=%s",
                normalized_response_id,
                source_event,
            )
            return
        logger.info(
            "deliverable_selected response_id=%s selected=false reason=cancelled",
            normalized_response_id,
        )
        logged_ids.add(normalized_response_id)

    def _cancelled_response_timing_state(self, response_id: str) -> dict[str, Any] | None:
        normalized_response_id = str(response_id or "").strip()
        if not normalized_response_id:
            return None
        store = getattr(self, "_cancelled_response_timing_by_id", None)
        if not isinstance(store, dict):
            store = {}
            self._cancelled_response_timing_by_id = store
        state = store.get(normalized_response_id)
        if not isinstance(state, dict):
            state = {
                "cancel_issued_at": None,
                "first_audio_delta_seen_at": None,
                "output_audio_done_at": None,
                "race_logged": False,
            }
            store[normalized_response_id] = state
        return state

    def _record_cancel_issued_timing(self, response_id: str) -> None:
        state = self._cancelled_response_timing_state(response_id)
        if state is None:
            return
        state["cancel_issued_at"] = float(time.time())
        state["first_audio_delta_seen_at"] = None
        state["output_audio_done_at"] = None
        state["race_logged"] = False

    def _record_cancelled_audio_race_transition(self, *, response_id: str, event_type: str) -> None:
        state = self._cancelled_response_timing_state(response_id)
        if state is None:
            return
        now = float(time.time())
        if event_type == "response.output_audio.delta" and state.get("first_audio_delta_seen_at") is None:
            state["first_audio_delta_seen_at"] = now
        if event_type == "response.output_audio.done" and state.get("output_audio_done_at") is None:
            state["output_audio_done_at"] = now
        cancel_issued_at = state.get("cancel_issued_at")
        if cancel_issued_at is None or bool(state.get("race_logged", False)):
            return
        first_audio_delta_seen_at = state.get("first_audio_delta_seen_at")
        output_audio_done_at = state.get("output_audio_done_at")
        delta_after_cancel_ms: str
        if isinstance(first_audio_delta_seen_at, (int, float)):
            delta_after_cancel_ms = str(int((float(first_audio_delta_seen_at) - float(cancel_issued_at)) * 1000))
        else:
            delta_after_cancel_ms = "na"
        logger.info(
            "cancel_audio_race_observed response_id=%s cancel_issued_at=%s first_audio_delta_seen_at=%s output_audio_done_at=%s delta_after_cancel_ms=%s",
            str(response_id or "").strip() or "unknown",
            f"{float(cancel_issued_at):.6f}",
            f"{float(first_audio_delta_seen_at):.6f}" if isinstance(first_audio_delta_seen_at, (int, float)) else "na",
            f"{float(output_audio_done_at):.6f}" if isinstance(output_audio_done_at, (int, float)) else "na",
            delta_after_cancel_ms,
        )
        state["race_logged"] = True

    def _clear_cancelled_response_tracking(self, response_id: str) -> None:
        normalized_response_id = str(response_id or "").strip()
        if not normalized_response_id:
            return
        cancelled_ids = getattr(self, "_cancelled_response_ids", None)
        if isinstance(cancelled_ids, set):
            cancelled_ids.discard(normalized_response_id)
        logged_ids = getattr(self, "_cancelled_deliverable_logged_ids", None)
        if isinstance(logged_ids, set):
            logged_ids.discard(normalized_response_id)
        timing_store = getattr(self, "_cancelled_response_timing_by_id", None)
        if isinstance(timing_store, dict):
            timing_store.pop(normalized_response_id, None)
        suppressed_audio_ids = getattr(self, "_suppressed_audio_response_ids", None)
        if isinstance(suppressed_audio_ids, set):
            suppressed_audio_ids.discard(normalized_response_id)
        status_by_id = getattr(self, "_response_status_by_id", None)
        if isinstance(status_by_id, dict):
            status_by_id.pop(normalized_response_id, None)
        self._clear_stale_response_context(normalized_response_id)

    def _canonical_utterance_key(self, *, turn_id: str, input_event_key: str | None) -> str:
        return self._lifecycle_state_coordinator().canonical_utterance_key(
            run_id=self._current_run_id(),
            turn_id=turn_id,
            input_event_key=input_event_key,
        )

    def _lifecycle_controller(self) -> InteractionLifecycleController:
        controller = getattr(self, "_interaction_lifecycle_controller", None)
        if not isinstance(controller, InteractionLifecycleController):
            controller = InteractionLifecycleController()
            self._interaction_lifecycle_controller = controller
        return controller

    def _lifecycle_policy(self) -> InteractionLifecyclePolicy:
        policy = getattr(self, "_interaction_lifecycle_policy", None)
        if not isinstance(policy, InteractionLifecyclePolicy):
            policy = InteractionLifecyclePolicy()
            self._interaction_lifecycle_policy = policy
        return policy

    def _is_active_response_blocking(self) -> bool:
        # Invariant: only one response.create is allowed in flight; additional
        # create attempts must queue until response.done clears flight state.
        if not bool(getattr(self, "_response_in_flight", False)):
            return False
        if str(getattr(self, "_active_response_origin", "") or "").strip().lower() == "micro_ack":
            return False
        active_response_id = str(getattr(self, "_active_response_id", "") or "").strip()
        if not active_response_id:
            pending_origins = getattr(self, "_pending_response_create_origins", None)
            if isinstance(pending_origins, deque) and pending_origins:
                if all(
                    isinstance(pending, dict)
                    and str(pending.get("micro_ack", "")).strip().lower() == "true"
                    for pending in pending_origins
                ):
                    return False
            return True
        cancelled_ids = getattr(self, "_cancelled_response_ids", None)
        if isinstance(cancelled_ids, set) and active_response_id in cancelled_ids:
            return False
        return True

    def _clear_cancelled_response_blocking_state(self, *, response_id: str, reason: str) -> None:
        normalized_response_id = str(response_id or "").strip()
        if not normalized_response_id:
            return
        if str(getattr(self, "_active_response_id", "") or "").strip() != normalized_response_id:
            return
        self._response_in_flight = False
        self.response_in_progress = False
        self._active_response_id = None
        self._active_response_origin = "unknown"
        self._active_response_input_event_key = None
        self._active_response_canonical_key = None
        self._active_server_auto_input_event_key = None
        self._active_response_confirmation_guarded = False
        self._active_response_preference_guarded = False
        logger.info(
            "cancelled_response_unblocked_scheduler run_id=%s response_id=%s reason=%s",
            self._current_run_id() or "",
            normalized_response_id,
            reason or "unknown",
        )

    def _is_synthetic_input_event_key(self, input_event_key: str | None) -> bool:
        normalized = str(input_event_key or "").strip().lower()
        return normalized.startswith("synthetic_")

    def _rebind_active_response_correlation_key(
        self,
        *,
        turn_id: str,
        replacement_input_event_key: str,
        cause: str = "transcript_final_rebind",
    ) -> None:
        normalized_replacement = str(replacement_input_event_key or "").strip()
        if not normalized_replacement:
            return
        if str(getattr(self, "_active_response_origin", "")).strip().lower() != "server_auto":
            return
        if not bool(getattr(self, "_response_in_flight", False)):
            return
        active_key = str(getattr(self, "_active_server_auto_input_event_key", "") or "").strip()
        if not self._is_synthetic_input_event_key(active_key):
            return
        old_canonical_key = self._canonical_utterance_key(turn_id=turn_id, input_event_key=active_key)
        new_canonical_key = self._canonical_utterance_key(turn_id=turn_id, input_event_key=normalized_replacement)
        if old_canonical_key == new_canonical_key:
            return

        def _update_local_active_response_pointers() -> None:
            if str(getattr(self, "_active_response_input_event_key", "") or "").strip() == active_key:
                self._active_response_input_event_key = normalized_replacement
            if str(getattr(self, "_active_response_canonical_key", "") or "").strip() == old_canonical_key:
                self._active_response_canonical_key = new_canonical_key
            self._active_server_auto_input_event_key = normalized_replacement

        lifecycle = self._lifecycle_controller()
        new_lifecycle_state = lifecycle.state_for(new_canonical_key)
        if new_lifecycle_state in {
            InteractionLifecycleState.AUDIO_STARTED,
            InteractionLifecycleState.DONE,
        }:
            _update_local_active_response_pointers()
            self._log_lifecycle_event(
                turn_id=turn_id,
                input_event_key=normalized_replacement,
                canonical_key=new_canonical_key,
                origin="server_auto",
                response_id=getattr(self, "_active_response_id", None),
                decision=(
                    "transition_rebind_skipped:new_key_already_active"
                    f":new_state={new_lifecycle_state.value}"
                    f":cause={str(cause or 'unknown').strip() or 'unknown'}"
                ),
            )
            logger.debug(
                "[RESPTRACE] response_key_rebind_skipped run_id=%s turn_id=%s old_input_event_key=%s "
                "new_input_event_key=%s old_canonical_key=%s new_canonical_key=%s new_state=%s",
                self._current_run_id() or "",
                turn_id,
                active_key,
                normalized_replacement,
                old_canonical_key,
                new_canonical_key,
                new_lifecycle_state.value,
            )
            return

        state_store = self._canonical_response_state_store()
        old_state = state_store.get(old_canonical_key)
        if isinstance(old_state, CanonicalResponseState):
            if new_canonical_key not in state_store:
                rebound_state = replace(old_state)
                rebound_state.turn_id = str(turn_id or "").strip() or rebound_state.turn_id
                rebound_state.input_event_key = normalized_replacement
                state_store[new_canonical_key] = rebound_state
            del state_store[old_canonical_key]
            self._sync_legacy_response_state_mirrors()

        lifecycle_state = getattr(self, "_canonical_response_lifecycle_state", None)
        if isinstance(lifecycle_state, dict) and old_canonical_key in lifecycle_state:
            old_lifecycle = lifecycle_state.pop(old_canonical_key)
            if new_canonical_key not in lifecycle_state:
                lifecycle_state[new_canonical_key] = old_lifecycle
        old_lifecycle_state = lifecycle.state_for(old_canonical_key)
        should_mark_replaced = old_lifecycle_state in {
            InteractionLifecycleState.CANCELLED,
            InteractionLifecycleState.DONE,
            InteractionLifecycleState.REPLACED,
        }
        if should_mark_replaced:
            lifecycle.on_replaced(old_canonical_key, new_canonical_key)
        else:
            lifecycle.on_key_rebound(old_canonical_key, new_canonical_key)
        transition_decision = "transition_replaced" if should_mark_replaced else "transition_key_rebound"
        self._log_lifecycle_event(
            turn_id=turn_id,
            input_event_key=normalized_replacement,
            canonical_key=new_canonical_key,
            origin="server_auto",
            response_id=getattr(self, "_active_response_id", None),
            decision=f"{transition_decision}:cause={str(cause or 'unknown').strip() or 'unknown'}",
        )

        _update_local_active_response_pointers()
        self._set_upgrade_chain_trace_context(
            response_id=getattr(self, "_active_response_id", None),
            chain_id=self._ensure_upgrade_chain_id(
                turn_id=turn_id,
                input_event_key=normalized_replacement,
                response_id=getattr(self, "_active_response_id", None),
            ),
            turn_id=turn_id,
            input_event_key=normalized_replacement,
            canonical_key=new_canonical_key,
            origin="server_auto",
        )
        logger.debug(
            "[RESPTRACE] response_key_rebound run_id=%s turn_id=%s old_input_event_key=%s new_input_event_key=%s "
            "old_canonical_key=%s new_canonical_key=%s",
            self._current_run_id() or "",
            turn_id,
            active_key,
            normalized_replacement,
            old_canonical_key,
            new_canonical_key,
        )
        self._log_lifecycle_coherence(
            stage="transcript_final_rebind",
            turn_id=turn_id,
            response_id=str(getattr(self, "_active_response_id", "") or "").strip(),
            canonical_key=new_canonical_key,
        )

    def _resptrace_suppression_reason(self, *, turn_id: str, input_event_key: str) -> str:
        suppression_until = float(getattr(self, "_preference_recall_response_suppression_until", 0.0) or 0.0)
        suppression_window_active = suppression_until > time.monotonic()
        suppressed_turns = getattr(self, "_preference_recall_suppressed_turns", None)
        if not isinstance(suppressed_turns, set):
            suppressed_turns = set()
            self._preference_recall_suppressed_turns = suppressed_turns
        suppressed_input_event_keys = getattr(self, "_preference_recall_suppressed_input_event_keys", None)
        if not isinstance(suppressed_input_event_keys, set):
            suppressed_input_event_keys = set()
            self._preference_recall_suppressed_input_event_keys = suppressed_input_event_keys
        if input_event_key and input_event_key in suppressed_input_event_keys:
            return "input_event_suppressed"
        if turn_id in suppressed_turns:
            return "turn_suppressed"
        if suppression_window_active:
            return "suppression_window_active"
        return "none"

    def _response_obligation_key(self, *, turn_id: str, input_event_key: str | None) -> str:
        return self._lifecycle_state_coordinator().response_obligation_key(
            run_id=self._current_run_id(),
            turn_id=turn_id,
            input_event_key=input_event_key,
        )

    def _canonical_response_state_store(self) -> dict[str, CanonicalResponseState]:
        state_store = getattr(self, "_canonical_response_state_by_key", None)
        if not isinstance(state_store, dict):
            state_store = {}
            self._canonical_response_state_by_key = state_store
        return state_store

    def _canonical_response_state(self, canonical_key: str) -> CanonicalResponseState | None:
        normalized = str(canonical_key or "").strip()
        if not normalized:
            return None
        state = self._canonical_response_state_store().get(normalized)
        return state if isinstance(state, CanonicalResponseState) else None

    def _canonical_response_state_mutate(
        self,
        *,
        canonical_key: str,
        turn_id: str,
        input_event_key: str | None,
        mutator: Callable[[CanonicalResponseState], None],
    ) -> CanonicalResponseState:
        normalized_canonical_key = str(canonical_key or "").strip()
        if not normalized_canonical_key:
            normalized_canonical_key = self._canonical_utterance_key(
                turn_id=turn_id,
                input_event_key=input_event_key,
            )
        state_store = self._canonical_response_state_store()
        prior_state = state_store.get(normalized_canonical_key)
        if not isinstance(prior_state, CanonicalResponseState):
            prior_state = CanonicalResponseState()
        next_state = replace(prior_state)
        resolved_turn_id = str(turn_id or "").strip()
        resolved_input_event_key = str(input_event_key or "").strip()
        if resolved_turn_id:
            next_state.turn_id = resolved_turn_id
        if resolved_input_event_key:
            next_state.input_event_key = resolved_input_event_key
        mutator(next_state)
        state_store[normalized_canonical_key] = next_state
        self._sync_legacy_response_state_mirrors()
        self._debug_assert_canonical_state_invariants(
            canonical_key=normalized_canonical_key,
            state=next_state,
        )
        return next_state

    def _sync_legacy_response_state_mirrors(self) -> None:
        # TODO: Remove these legacy mirrors after callers/tests fully migrate to canonical state accessors.
        state_store = self._canonical_response_state_store()
        created_keys: set[str] = set()
        delivery_ledger: dict[str, str] = {}
        response_id_by_key: dict[str, str] = {}
        obligations: dict[str, dict[str, Any]] = {}
        for canonical_key, state in state_store.items():
            if not isinstance(state, CanonicalResponseState):
                continue
            if state.created:
                created_keys.add(canonical_key)
            if state.done:
                delivery_ledger[canonical_key] = "done"
            elif state.cancel_sent:
                delivery_ledger[canonical_key] = "cancelled"
            elif state.audio_started:
                delivery_ledger[canonical_key] = "delivered"
            elif state.created:
                delivery_ledger[canonical_key] = "created"
            if state.response_id:
                response_id_by_key[canonical_key] = state.response_id
            if state.obligation_present:
                obligations[canonical_key] = dict(state.obligation or {
                    "turn_id": state.turn_id,
                    "input_event_key": state.input_event_key,
                    "source": state.origin,
                    "created_at": time.monotonic(),
                })
        self._response_created_canonical_keys = created_keys
        self._response_delivery_ledger = delivery_ledger
        self._response_id_by_canonical_key = response_id_by_key
        self._response_obligations = obligations

    def _debug_assert_canonical_state_invariants(
        self,
        *,
        canonical_key: str,
        state: CanonicalResponseState,
    ) -> None:
        if not logger.isEnabledFor(10):
            return
        invalid: list[str] = []
        if state.done and not state.created:
            invalid.append("done_without_created")
        if state.audio_started and not state.created:
            invalid.append("audio_without_created")
        if state.cancel_sent and not state.created:
            invalid.append("cancel_without_created")
        if not invalid:
            return
        run_id = str(self._current_run_id() or "").strip() or "run-unknown"
        logged = getattr(self, "_canonical_invariant_logged", None)
        if not isinstance(logged, set):
            logged = set()
            self._canonical_invariant_logged = logged
        for item in invalid:
            invariant_key = f"{run_id}:{canonical_key}:{item}"
            if invariant_key in logged:
                continue
            logged.add(invariant_key)
            logger.debug(
                "canonical_state_invariant_failed run_id=%s canonical_key=%s reason=%s state=%s",
                run_id,
                canonical_key,
                item,
                {
                    "created": state.created,
                    "audio_started": state.audio_started,
                    "deliverable_observed": state.deliverable_observed,
                    "done": state.done,
                    "cancel_sent": state.cancel_sent,
                    "origin": state.origin,
                    "response_id": state.response_id,
                    "obligation_present": state.obligation_present,
                    "input_event_key": state.input_event_key,
                    "turn_id": state.turn_id,
                },
            )

    def _active_canonical_key_for_deliverable_marker(self) -> str:
        active_key = str(getattr(self, "_active_response_canonical_key", "") or "").strip()
        if active_key:
            return active_key
        response_id = str(getattr(self, "_active_response_id", "") or "").strip()
        if response_id:
            trace_context = self._response_trace_by_id().get(response_id, {})
            if isinstance(trace_context, dict):
                traced_key = str(trace_context.get("canonical_key") or "").strip()
                if traced_key:
                    return traced_key
        turn_id = str(self._current_turn_id_or_unknown() or "").strip()
        input_event_key = str(getattr(self, "_active_response_input_event_key", "") or "").strip()
        if turn_id and input_event_key:
            return self._canonical_utterance_key(turn_id=turn_id, input_event_key=input_event_key)
        return ""

    def _mark_active_canonical_deliverable_observed(self, *, reason: str) -> None:
        canonical_key = self._active_canonical_key_for_deliverable_marker()
        if not canonical_key:
            logger.debug("canonical_deliverable_marker_skipped reason=%s marker_reason=%s", "canonical_key_unavailable", reason)
            return

        self._canonical_response_state_mutate(
            canonical_key=canonical_key,
            turn_id=self._current_turn_id_or_unknown(),
            input_event_key=getattr(self, "_active_response_input_event_key", None),
            mutator=lambda record: (
                setattr(record, "created", True),
                setattr(record, "deliverable_observed", True),
                setattr(record, "origin", str(getattr(self, "_active_response_origin", "unknown") or "unknown")),
                setattr(record, "response_id", str(getattr(self, "_active_response_id", "") or "")),
            ),
        )
        logger.debug(
            "canonical_deliverable_marker_set canonical_key=%s reason=%s",
            canonical_key,
            reason,
        )

    def _classify_deliverable_text(self, text: str) -> str:
        normalized = re.sub(r"\s+", " ", str(text or "")).strip().lower()
        if not normalized:
            return "unknown"
        progress_markers = (
            "checking",
            "retrieving",
            "one moment",
            "let me",
            "hold on",
            "give me a second",
            "looking that up",
            "i'm checking",
            "i am checking",
        )
        if any(marker in normalized for marker in progress_markers):
            return "progress"
        return "final"

    def _record_active_canonical_deliverable_class(self, *, text: str, reason: str) -> None:
        canonical_key = self._active_canonical_key_for_deliverable_marker()
        if not canonical_key:
            return
        observed_class = self._classify_deliverable_text(text)
        if observed_class == "unknown":
            return

        def _mutator(record: CanonicalResponseState) -> None:
            prior_class = str(getattr(record, "deliverable_class", "unknown") or "unknown").strip().lower() or "unknown"
            if prior_class == "final":
                return
            if prior_class == observed_class:
                return
            if prior_class == "progress" and observed_class != "final":
                return
            record.deliverable_class = observed_class

        self._canonical_response_state_mutate(
            canonical_key=canonical_key,
            turn_id=self._current_turn_id_or_unknown(),
            input_event_key=getattr(self, "_active_response_input_event_key", None),
            mutator=_mutator,
        )
        logger.debug(
            "canonical_deliverable_class_set canonical_key=%s class=%s reason=%s",
            canonical_key,
            observed_class,
            reason,
        )

    def _response_delivery_state(self, *, turn_id: str, input_event_key: str | None) -> str | None:
        key = self._canonical_utterance_key(turn_id=turn_id, input_event_key=input_event_key)
        state = self._canonical_response_state(key)
        if isinstance(state, CanonicalResponseState):
            blocked_reason = str(getattr(state, "blocked_reason", "") or "").strip().lower()
            if blocked_reason:
                return blocked_reason
            if state.done:
                return "done"
            if state.cancel_sent:
                return "cancelled"
            if state.audio_started:
                return "delivered"
            if state.created:
                return "created"
            return None
        ledger = getattr(self, "_response_delivery_ledger", None)
        if not isinstance(ledger, dict):
            return None
        value = ledger.get(key)
        return str(value).strip().lower() if isinstance(value, str) else None

    def _set_response_delivery_state(
        self,
        *,
        turn_id: str,
        input_event_key: str | None,
        state: str,
    ) -> None:
        normalized_state = str(state or "").strip().lower()
        if not normalized_state:
            return
        key = self._canonical_utterance_key(turn_id=turn_id, input_event_key=input_event_key)

        def _mutate(record: CanonicalResponseState) -> None:
            if normalized_state == "created":
                record.created = True
                record.done = False
                record.cancel_sent = False
            elif normalized_state == "delivered":
                record.created = True
                record.audio_started = True
                record.done = False
            elif normalized_state == "done":
                record.created = True
                record.done = True
            elif normalized_state == "cancelled":
                record.cancel_sent = True
            elif normalized_state == "blocked_empty_transcript":
                record.blocked_reason = "blocked_empty_transcript"
                record.done = True
                record.cancel_sent = True
                record.created = False
                record.audio_started = False
                record.deliverable_observed = False

        self._canonical_response_state_mutate(
            canonical_key=key,
            turn_id=turn_id,
            input_event_key=input_event_key,
            mutator=_mutate,
        )

    def _mark_empty_transcript_blocked(self, *, turn_id: str, input_event_key: str, reason: str) -> None:
        normalized_turn_id = str(turn_id or "").strip() or "turn-unknown"
        normalized_input_event_key = str(input_event_key or "").strip()
        if not normalized_input_event_key:
            return
        self._set_response_delivery_state(
            turn_id=normalized_turn_id,
            input_event_key=normalized_input_event_key,
            state="blocked_empty_transcript",
        )
        self._clear_pending_response_contenders(
            turn_id=normalized_turn_id,
            input_event_key=normalized_input_event_key,
            reason="empty_transcript_blocked",
        )
        logger.info(
            "transcript_empty_blocked run_id=%s turn_id=%s input_event_key=%s reason=%s",
            self._current_run_id() or "",
            normalized_turn_id,
            normalized_input_event_key,
            reason,
        )

    def _is_response_already_delivered(self, *, turn_id: str, input_event_key: str | None) -> bool:
        return self._response_delivery_state(turn_id=turn_id, input_event_key=input_event_key) == "delivered"

    def _single_flight_block_reason(self, *, turn_id: str, input_event_key: str | None) -> str | None:
        canonical_key = self._canonical_utterance_key(turn_id=turn_id, input_event_key=input_event_key)
        obligation_state = self._canonical_response_state(canonical_key)
        obligation_present = bool(
            isinstance(obligation_state, CanonicalResponseState)
            and obligation_state.obligation_present
        )
        decision = self._lifecycle_controller().decide_response_create_allow(canonical_key, origin="assistant_message")
        if decision.action is LifecycleDecisionAction.CANCEL:
            if "state=done" in decision.reason:
                return "already_done"
            if obligation_present:
                return None
            return "already_created"
        if decision.action is LifecycleDecisionAction.DEFER:
            return decision.reason
        state = self._response_delivery_state(turn_id=turn_id, input_event_key=input_event_key)
        if state == "created":
            return "already_created"
        if state in {"delivered", "done"}:
            return "already_done"
        return None

    def _log_response_create_blocked(
        self,
        *,
        turn_id: str,
        origin: str,
        input_event_key: str | None,
        canonical_key: str,
        block_reason: str,
    ) -> None:
        existing_state = self._canonical_response_state(canonical_key)
        existing_response_id = str(existing_state.response_id if isinstance(existing_state, CanonicalResponseState) else "").strip()
        self._log_lifecycle_event(
            turn_id=turn_id,
            input_event_key=input_event_key,
            canonical_key=canonical_key,
            origin=origin,
            response_id=existing_response_id,
            decision=f"guard_blocked:{block_reason}",
        )
        self._debug_dump_canonical_key_timeline(
            canonical_key=canonical_key,
            trigger=f"response_create_blocked:{block_reason}",
        )

    def _log_response_site_debug(
        self,
        *,
        site: str,
        turn_id: str,
        input_event_key: str | None,
        canonical_key: str | None,
        origin: str,
        trigger: str,
    ) -> None:
        normalized_input_event_key = str(input_event_key or "").strip()
        resolved_canonical_key = str(canonical_key or "").strip() or self._canonical_utterance_key(
            turn_id=turn_id,
            input_event_key=normalized_input_event_key,
        )
        obligation_key = self._response_obligation_key(
            turn_id=turn_id,
            input_event_key=normalized_input_event_key,
        )
        obligation_state = self._canonical_response_state(obligation_key)
        obligation_present = bool(
            isinstance(obligation_state, CanonicalResponseState)
            and obligation_state.obligation_present
        )
        suppression_reason = self._resptrace_suppression_reason(
            turn_id=turn_id,
            input_event_key=normalized_input_event_key,
        )
        logger.debug(
            "%s run_id=%s turn_id=%s input_event_key=%s canonical_key=%s origin=%s trigger=%s "
            "queue_len=%s pending_server_auto=%s suppression_active=%s obligation_present=%s response_done_serial=%s",
            site,
            self._current_run_id() or "",
            turn_id,
            normalized_input_event_key or "unknown",
            resolved_canonical_key,
            origin,
            trigger,
            len(getattr(self, "_response_create_queue", deque()) or ()),
            len(getattr(self, "_pending_server_auto_input_event_keys", deque()) or ()),
            suppression_reason != "none",
            obligation_present,
            getattr(self, "_response_done_serial", 0),
        )

    def _set_response_obligation(self, *, turn_id: str, input_event_key: str, source: str) -> None:
        obligation_key = self._response_obligation_key(turn_id=turn_id, input_event_key=input_event_key)
        active_response_input_event_key = str(
            getattr(self, "_active_response_input_event_key", "") or ""
        ).strip() or "none"
        active_server_auto_input_event_key = str(
            getattr(self, "_active_server_auto_input_event_key", "") or ""
        ).strip() or "none"
        updated_state, obligation_present_before = self._lifecycle_state_coordinator().transition_response_obligation(
            state_store=self._canonical_response_state_store(),
            obligation_key=obligation_key,
            turn_id=turn_id,
            input_event_key=input_event_key,
            source=source,
            present=True,
        )
        self._sync_legacy_response_state_mirrors()
        self._debug_assert_canonical_state_invariants(canonical_key=obligation_key, state=updated_state)
        total_obligations = sum(
            1
            for state in self._canonical_response_state_store().values()
            if isinstance(state, CanonicalResponseState) and state.obligation_present
        )
        logger.info(
            "response_obligation_set run_id=%s turn_id=%s input_event_key=%s source=%s",
            self._current_run_id() or "",
            turn_id,
            input_event_key or "unknown",
            source,
        )
        logger.debug(
            "[RESPTRACE] obligation_set run_id=%s turn_id=%s input_event_key=%s canonical_key=%s source=%s "
            "obligation_present_before=%s obligation_present_after=%s total_obligations=%s obligation_key=%s "
            "active_response_input_event_key=%s active_server_auto_input_event_key=%s",
            self._current_run_id() or "",
            turn_id,
            input_event_key or "unknown",
            self._canonical_utterance_key(turn_id=turn_id, input_event_key=input_event_key),
            source,
            obligation_present_before,
            updated_state.obligation_present,
            total_obligations,
            obligation_key,
            active_response_input_event_key,
            active_server_auto_input_event_key,
        )

    def _clear_response_obligation(
        self,
        *,
        turn_id: str,
        input_event_key: str | None,
        reason: str,
        origin: str,
    ) -> None:
        obligation_key = self._response_obligation_key(turn_id=turn_id, input_event_key=input_event_key)
        prior_state = self._canonical_response_state(obligation_key)
        if not isinstance(prior_state, CanonicalResponseState) or not prior_state.obligation_present:
            return
        active_response_input_event_key = str(
            getattr(self, "_active_response_input_event_key", "") or ""
        ).strip() or "none"
        active_server_auto_input_event_key = str(
            getattr(self, "_active_server_auto_input_event_key", "") or ""
        ).strip() or "none"
        updated_state, obligation_present_before = self._lifecycle_state_coordinator().transition_response_obligation(
            state_store=self._canonical_response_state_store(),
            obligation_key=obligation_key,
            turn_id=turn_id,
            input_event_key=input_event_key,
            source=origin,
            present=False,
        )
        self._sync_legacy_response_state_mirrors()
        self._debug_assert_canonical_state_invariants(canonical_key=obligation_key, state=updated_state)
        normalized_input_event_key = str(input_event_key or "").strip() or "unknown"
        total_obligations = sum(
            1
            for state in self._canonical_response_state_store().values()
            if isinstance(state, CanonicalResponseState) and state.obligation_present
        )
        logger.info(
            "response_obligation_cleared run_id=%s turn_id=%s input_event_key=%s origin=%s reason=%s",
            self._current_run_id() or "",
            turn_id,
            normalized_input_event_key,
            origin,
            reason,
        )
        logger.debug(
            "[RESPTRACE] obligation_cleared run_id=%s turn_id=%s input_event_key=%s canonical_key=%s origin=%s "
            "obligation_present_before=%s obligation_present_after=%s total_obligations=%s obligation_key=%s "
            "active_response_input_event_key=%s active_server_auto_input_event_key=%s reason=%s",
            self._current_run_id() or "",
            turn_id,
            normalized_input_event_key,
            self._canonical_utterance_key(turn_id=turn_id, input_event_key=input_event_key),
            origin,
            obligation_present_before,
            updated_state.obligation_present,
            total_obligations,
            obligation_key,
            active_response_input_event_key,
            active_server_auto_input_event_key,
            reason,
        )

    def _mark_input_event_key_scheduled(self, *, input_event_key: str) -> None:
        normalized_key = str(input_event_key or "").strip()
        if not normalized_key:
            return
        scheduled_keys = getattr(self, "_already_scheduled_for_input_event_key", None)
        if not isinstance(scheduled_keys, set):
            scheduled_keys = set()
            self._already_scheduled_for_input_event_key = scheduled_keys
        scheduled_keys.add(normalized_key)

    def _is_input_event_key_already_scheduled(self, *, input_event_key: str) -> bool:
        normalized_key = str(input_event_key or "").strip()
        if not normalized_key:
            return False
        scheduled_keys = getattr(self, "_already_scheduled_for_input_event_key", None)
        return isinstance(scheduled_keys, set) and normalized_key in scheduled_keys

    def _next_synthetic_input_event_key(self, origin: str) -> str:
        counter = int(getattr(self, "_synthetic_input_event_counter", 0)) + 1
        self._synthetic_input_event_counter = counter
        normalized_origin = str(origin or "unknown").strip().lower() or "unknown"
        return f"synthetic_{normalized_origin}_{counter}"

    def _ensure_response_create_correlation(
        self,
        *,
        response_create_event: dict[str, Any],
        origin: str,
        turn_id: str,
    ) -> str:
        response_payload = response_create_event.setdefault("response", {})
        metadata = response_payload.setdefault("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
            response_payload["metadata"] = metadata
        input_event_key = str(
            metadata.get("input_event_key")
            or getattr(self, "_current_input_event_key", "")
            or ""
        ).strip()
        if not input_event_key:
            input_event_key = self._next_synthetic_input_event_key(origin)
            metadata["input_event_key"] = input_event_key
        metadata["turn_id"] = str(metadata.get("turn_id") or "").strip() or str(turn_id or "").strip() or "turn-unknown"
        metadata["input_event_key"] = str(metadata.get("input_event_key") or "").strip() or input_event_key
        return input_event_key

    def _tool_followup_input_event_key(self, *, call_id: str) -> str:
        return f"tool:{str(call_id or '').strip() or 'unknown'}"

    def _tool_followup_state(self, *, canonical_key: str) -> str:
        normalized_canonical_key = str(canonical_key or "").strip()
        if not normalized_canonical_key:
            return "new"
        state_store = getattr(self, "_tool_followup_state_by_canonical_key", None)
        if not isinstance(state_store, dict):
            state_store = {}
            self._tool_followup_state_by_canonical_key = state_store
        state = str(state_store.get(normalized_canonical_key) or "").strip().lower()
        return state or "new"

    def _set_tool_followup_state(self, *, canonical_key: str, state: str, reason: str) -> None:
        normalized_canonical_key = str(canonical_key or "").strip()
        normalized_state = str(state or "").strip().lower() or "new"
        if not normalized_canonical_key:
            return
        state_store = getattr(self, "_tool_followup_state_by_canonical_key", None)
        if not isinstance(state_store, dict):
            state_store = {}
            self._tool_followup_state_by_canonical_key = state_store
        prior_state = str(state_store.get(normalized_canonical_key) or "new").strip().lower() or "new"
        log_method = logger.info
        if normalized_state == "blocked_active_response":
            log_method = logger.debug
        if prior_state == normalized_state:
            log_method(
                "tool_followup_state canonical_key=%s state=%s reason=%s",
                normalized_canonical_key,
                normalized_state,
                reason,
            )
            return
        state_store[normalized_canonical_key] = normalized_state
        log_method(
            "tool_followup_state canonical_key=%s state=%s reason=%s prior_state=%s",
            normalized_canonical_key,
            normalized_state,
            reason,
            prior_state,
        )
        if normalized_state in {"done", "dropped"}:
            self._clear_stale_assistant_message_creates_for_tool_followup(
                canonical_key=normalized_canonical_key,
                state=normalized_state,
            )

    def _clear_stale_assistant_message_creates_for_tool_followup(
        self,
        *,
        canonical_key: str,
        state: str,
    ) -> None:
        if ":tool:" not in str(canonical_key or ""):
            return
        normalized_canonical_key = str(canonical_key or "").strip()
        if not normalized_canonical_key:
            return

        dropped_pending = False
        pending = getattr(self, "_pending_response_create", None)
        if pending is not None and str(getattr(pending, "origin", "") or "").strip().lower() == "assistant_message":
            pending_metadata = self._extract_response_create_metadata(getattr(pending, "event", {}) or {})
            pending_turn_id = str(pending_metadata.get("turn_id") or pending.turn_id or "").strip() or pending.turn_id
            pending_input_event_key = str(pending_metadata.get("input_event_key") or "").strip()
            pending_canonical_key = self._canonical_utterance_key(
                turn_id=pending_turn_id,
                input_event_key=pending_input_event_key,
            )
            if pending_canonical_key == normalized_canonical_key:
                self._pending_response_create = None
                dropped_pending = True

        dropped_queue = 0
        queue = getattr(self, "_response_create_queue", None)
        if isinstance(queue, deque) and queue:
            retained: deque[dict[str, Any]] = deque()
            for queued in queue:
                origin = str(queued.get("origin") or "").strip().lower() if isinstance(queued, dict) else ""
                if origin != "assistant_message":
                    retained.append(queued)
                    continue
                metadata = self._extract_response_create_metadata(queued.get("event") or {})
                queued_turn_id = str(queued.get("turn_id") or "").strip()
                queued_input_event_key = str(metadata.get("input_event_key") or "").strip()
                queued_canonical_key = self._canonical_utterance_key(
                    turn_id=queued_turn_id,
                    input_event_key=queued_input_event_key,
                )
                if queued_canonical_key == normalized_canonical_key:
                    dropped_queue += 1
                    continue
                retained.append(queued)
            self._response_create_queue = retained

        if dropped_pending or dropped_queue:
            self._sync_pending_response_create_queue()
            logger.info(
                "tool_followup_cleanup_stale_creates canonical_key=%s state=%s dropped_pending=%s dropped_queue=%s",
                normalized_canonical_key,
                state,
                str(dropped_pending).lower(),
                dropped_queue,
            )

    @staticmethod
    def _tool_call_id_from_input_event_key(input_event_key: str | None) -> str:
        normalized_input_event_key = str(input_event_key or "").strip()
        if not normalized_input_event_key.startswith("tool:"):
            return ""
        return normalized_input_event_key.partition("tool:")[2].strip()

    def _evaluate_tool_lineage_guard(
        self,
        *,
        origin: str,
        turn_id: str,
        input_event_key: str,
        response_metadata: dict[str, Any] | None = None,
    ) -> tuple[bool, str, str, str, str]:
        normalized_input_event_key = str(input_event_key or "").strip()
        if not normalized_input_event_key.startswith("tool:"):
            return True, "not_tool_lineage", "", "not_applicable", ""
        metadata = response_metadata if isinstance(response_metadata, dict) else {}
        lineage_origin = self._canonical_response_create_origin(
            origin=origin,
            response_metadata=metadata,
        )
        canonical_key = self._canonical_utterance_key(
            turn_id=turn_id,
            input_event_key=normalized_input_event_key,
        )
        parent_state = self._tool_followup_state(canonical_key=canonical_key)
        call_id = (
            str(metadata.get("tool_call_id") or "").strip()
            or self._tool_call_id_from_input_event_key(normalized_input_event_key)
            or "unknown"
        )
        allow_suppressed = str(metadata.get("tool_lineage_allow_suppressed", "")).strip().lower() in {
            "1",
            "true",
            "yes",
        }
        allowed = allow_suppressed or parent_state != "dropped"
        reason = "explicit_allowlist" if allow_suppressed else f"tool_followup_state_{parent_state}"
        logger.info(
            "derived_response_lineage_eval origin=%s canonical_key=%s parent_state=%s allowed=%s",
            lineage_origin,
            canonical_key,
            parent_state,
            str(allowed).lower(),
        )
        if lineage_origin == "micro_ack":
            logger.info(
                "micro_ack_lineage_guard outcome=%s reason=%s",
                "allow" if allowed else "deny",
                reason,
            )
        elif lineage_origin == "assistant_message":
            logger.info(
                "assistant_message_lineage_guard outcome=%s reason=%s",
                "allow" if allowed else "deny",
                reason,
            )
        if not allowed:
            logger.info(
                "suppressed_tool_lineage_block origin=%s canonical_key=%s call_id=%s reason=%s",
                lineage_origin,
                canonical_key,
                call_id,
                reason,
            )
        return allowed, reason, canonical_key, parent_state, call_id

    def _is_low_risk_reversible_gesture_tool(self, *, tool_name: str | None) -> bool:
        normalized_tool_name = str(tool_name or "").strip().lower()
        if not normalized_tool_name.startswith("gesture_"):
            return False
        return normalized_tool_name in {
            "gesture_idle",
            "gesture_nod",
            "gesture_no",
            "gesture_look_around",
            "gesture_look_up",
            "gesture_look_left",
            "gesture_look_right",
            "gesture_look_down",
            "gesture_look_center",
            "gesture_curious_tilt",
            "gesture_attention_snap",
        }

    def _build_tool_followup_response_create_event(
        self,
        *,
        call_id: str,
        response_create_event: dict[str, Any] | None = None,
        tool_name: str | None = None,
        tool_result_has_distinct_info: bool = False,
    ) -> tuple[dict[str, Any], str]:
        event: dict[str, Any] = response_create_event or {"type": "response.create"}
        response_payload = event.setdefault("response", {})
        if not isinstance(response_payload, dict):
            response_payload = {}
            event["response"] = response_payload
        metadata = response_payload.setdefault("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
            response_payload["metadata"] = metadata
        tool_call_id = str(call_id or "").strip() or "unknown"
        turn_id = self._current_turn_id_or_unknown()
        parent_input_event_key = self._parent_input_event_key_for_tool_followup(turn_id=turn_id)
        tool_input_event_key = self._tool_followup_input_event_key(call_id=tool_call_id)
        metadata["turn_id"] = turn_id
        metadata["input_event_key"] = tool_input_event_key
        metadata["parent_turn_id"] = turn_id
        if parent_input_event_key:
            metadata["parent_input_event_key"] = parent_input_event_key
        metadata["tool_followup"] = "true"
        metadata["tool_call_id"] = tool_call_id
        normalized_tool_name = str(tool_name or "").strip().lower()
        if normalized_tool_name:
            metadata["tool_name"] = normalized_tool_name
        if self._is_low_risk_reversible_gesture_tool(tool_name=normalized_tool_name):
            metadata["tool_followup_suppress_if_parent_covered"] = "true"
            metadata["tool_followup_status_only"] = "true"
            response_payload["instructions"] = (
                "Gesture follow-up only: acknowledge gesture completion in one short sentence. "
                "Do not restate or re-answer semantic memory/preferences content already covered by the parent response. "
                "Do not narrate environment/vision context in gesture-only followups."
            )
        if tool_result_has_distinct_info:
            metadata["tool_result_has_distinct_info"] = "true"
        canonical_key = self._canonical_utterance_key(turn_id=turn_id, input_event_key=tool_input_event_key)
        return event, canonical_key

    def _parent_input_event_key_for_tool_followup(self, *, turn_id: str) -> str:
        parent_input_event_key = self._active_input_event_key_for_turn(turn_id)
        if parent_input_event_key and not parent_input_event_key.startswith("tool:"):
            return parent_input_event_key

        active_response_turn_id = str(getattr(self, "_current_response_turn_id", "") or "").strip()
        active_response_input_event_key = str(getattr(self, "_active_response_input_event_key", "") or "").strip()
        if (
            active_response_turn_id == str(turn_id or "").strip()
            and active_response_input_event_key
            and not active_response_input_event_key.startswith("tool:")
        ):
            return active_response_input_event_key

        return parent_input_event_key

    def _tool_result_has_distinct_followup_info(self, *, tool_name: str, result: Any) -> bool:
        normalized_tool_name = str(tool_name or "").strip().lower()
        if normalized_tool_name == "perform_research" and isinstance(result, dict):
            sources = result.get("sources")
            findings = result.get("findings")
            summary = str(result.get("summary") or "").strip()
            if summary:
                return True
            if isinstance(sources, list) and len(sources) > 0:
                return True
            if isinstance(findings, list) and len(findings) > 0:
                return True
        if isinstance(result, dict):
            distinct_marker = result.get("tool_result_has_distinct_info")
            if isinstance(distinct_marker, bool):
                return distinct_marker
        return False

    def _should_suppress_tool_followup_after_turn_deliverable(
        self,
        *,
        turn_id: str,
        parent_input_event_key: str | None = None,
    ) -> bool:
        normalized_turn_id = str(turn_id or "").strip()
        if not normalized_turn_id:
            return False
        normalized_parent_key = str(parent_input_event_key or "").strip()
        for state in self._canonical_response_state_store().values():
            if not isinstance(state, CanonicalResponseState):
                continue
            if str(state.turn_id or "").strip() != normalized_turn_id:
                continue
            input_event_key = str(state.input_event_key or "").strip()
            if normalized_parent_key and input_event_key != normalized_parent_key:
                continue
            if input_event_key.startswith("tool:"):
                continue
            if str(state.origin or "").strip().lower() == "micro_ack":
                continue
            if str(getattr(state, "deliverable_class", "unknown") or "unknown").strip().lower() == "final":
                return True
        return False

    def _canonical_state_for_response_id(
        self,
        *,
        response_id: str | None,
    ) -> tuple[str, CanonicalResponseState] | None:
        normalized_response_id = str(response_id or "").strip()
        if not normalized_response_id:
            return None
        for canonical_key, state in self._canonical_response_state_store().items():
            if not isinstance(state, CanonicalResponseState):
                continue
            if str(state.response_id or "").strip() != normalized_response_id:
                continue
            return canonical_key, state
        return None

    def _resolve_parent_state_for_tool_followup(
        self,
        *,
        response_metadata: dict[str, Any],
        blocked_by_response_id: str | None,
    ) -> tuple[str, CanonicalResponseState] | None:
        state_entry: tuple[str, CanonicalResponseState] | None = None
        resolved_from = "none"
        normalized_blocked_by = str(blocked_by_response_id or "").strip()
        if normalized_blocked_by:
            candidate = self._canonical_state_for_response_id(response_id=normalized_blocked_by)
            if candidate:
                state_entry = candidate
                resolved_from = "response_id"
        parent_turn_id = str(response_metadata.get("parent_turn_id") or response_metadata.get("turn_id") or "").strip()
        parent_input_event_key = str(response_metadata.get("parent_input_event_key") or "").strip()
        if not state_entry and parent_turn_id and parent_input_event_key and not parent_input_event_key.startswith("tool:"):
            parent_canonical_key = self._canonical_utterance_key(
                turn_id=parent_turn_id,
                input_event_key=parent_input_event_key,
            )
            candidate = self._canonical_response_state_store().get(parent_canonical_key)
            if isinstance(candidate, CanonicalResponseState):
                state_entry = (parent_canonical_key, candidate)
                resolved_from = "parent_key"
        if not state_entry and parent_turn_id:
            for canonical_key, candidate in self._canonical_response_state_store().items():
                if not isinstance(candidate, CanonicalResponseState):
                    continue
                if str(candidate.turn_id or "").strip() != parent_turn_id:
                    continue
                candidate_input_event_key = str(candidate.input_event_key or "").strip()
                if not candidate_input_event_key or candidate_input_event_key.startswith("tool:"):
                    continue
                candidate_origin = str(candidate.origin or "").strip().lower()
                if candidate_origin in {"", "micro_ack", "tool_output"}:
                    continue
                state_entry = (canonical_key, candidate)
                resolved_from = "turn_scan"
                break
        tool_call_id = str(response_metadata.get("tool_call_id") or "").strip()
        resolved_parent_response_id = "none"
        resolved_parent_canonical_key = "none"
        parent_covered = False
        if state_entry:
            resolved_parent_canonical_key = state_entry[0]
            resolved_parent_response_id = str(state_entry[1].response_id or "").strip() or "none"
            parent_origin = str(state_entry[1].origin or "").strip().lower()
            if parent_origin not in {"micro_ack", "tool_output"} and bool(state_entry[1].done):
                parent_covered, coverage_source, deliverable_observed, deliverable_class, terminal_selected, terminal_reason = self._parent_response_coverage_state(
                    parent_state=state_entry[1],
                )
                logger.info(
                    "parent_coverage_source_of_truth run_id=%s parent_response_id=%s covered=%s source=%s canonical_observed=%s canonical_class=%s terminal_selected=%s terminal_reason=%s",
                    self._current_run_id() or "",
                    resolved_parent_response_id,
                    str(parent_covered).lower(),
                    coverage_source,
                    str(deliverable_observed).lower(),
                    deliverable_class or "unknown",
                    str(terminal_selected).lower(),
                    terminal_reason,
                )
        logger.info(
            "tool_followup_parent_resolution run_id=%s turn_id=%s tool_call_id=%s parent_input_event_key=%s blocked_by_response_id=%s resolved_parent_response_id=%s resolved_parent_canonical_key=%s resolved_from=%s parent_covered=%s",
            self._current_run_id() or "",
            parent_turn_id or str(response_metadata.get("turn_id") or "").strip() or "turn-unknown",
            tool_call_id or "unknown",
            parent_input_event_key or "none",
            normalized_blocked_by or "none",
            resolved_parent_response_id,
            resolved_parent_canonical_key,
            resolved_from,
            str(parent_covered).lower(),
        )
        return state_entry

    def _should_suppress_queued_tool_followup_release(
        self,
        *,
        response_metadata: dict[str, Any],
        blocked_by_response_id: str | None,
    ) -> tuple[bool, tuple[str, CanonicalResponseState] | None, str]:
        suppressible = str(response_metadata.get("tool_followup_suppress_if_parent_covered", "")).strip().lower() in {"true", "1", "yes"}
        tool_name = str(response_metadata.get("tool_name") or "").strip().lower()
        has_distinct_info = str(response_metadata.get("tool_result_has_distinct_info", "")).strip().lower() in {"true", "1", "yes"}
        if not suppressible:
            return False, None, "not_suppressible"
        if has_distinct_info:
            return False, None, "distinct_info"
        if not self._is_low_risk_reversible_gesture_tool(tool_name=tool_name):
            return False, None, "non_gesture_tool"

        state_entry = self._resolve_parent_state_for_tool_followup(
            response_metadata=response_metadata,
            blocked_by_response_id=blocked_by_response_id,
        )
        if not state_entry:
            return False, None, "parent_unresolved"
        _, parent_state = state_entry
        parent_origin = str(parent_state.origin or "").strip().lower()
        if parent_origin in {"micro_ack", "tool_output"}:
            return False, state_entry, "parent_origin_excluded"
        if not bool(parent_state.done):
            return False, state_entry, "parent_not_done"
        parent_covered, coverage_source, deliverable_observed, deliverable_class, terminal_selected, terminal_reason = self._parent_response_coverage_state(
            parent_state=parent_state,
        )
        logger.info(
            "parent_coverage_source_of_truth run_id=%s parent_response_id=%s covered=%s source=%s canonical_observed=%s canonical_class=%s terminal_selected=%s terminal_reason=%s",
            self._current_run_id() or "",
            str(getattr(parent_state, "response_id", "") or "").strip() or "none",
            str(parent_covered).lower(),
            coverage_source,
            str(deliverable_observed).lower(),
            deliverable_class or "unknown",
            str(terminal_selected).lower(),
            terminal_reason,
        )
        if not parent_covered:
            return False, state_entry, "parent_not_deliverable"
        return True, state_entry, "parent_covered_tool_result"

    def _should_drop_tool_followup_at_create_seam(
        self,
        *,
        turn_id: str,
        response_metadata: dict[str, Any],
        canonical_key: str,
        drain_trigger: str,
    ) -> bool:
        is_tool_followup = str(response_metadata.get("tool_followup", "")).strip().lower() in {"true", "1", "yes"}
        if not is_tool_followup:
            return False
        blocked_by_response_id = str(response_metadata.get("blocked_by_response_id") or "").strip()
        should_drop, parent_entry, reason = self._should_suppress_queued_tool_followup_release(
            response_metadata=response_metadata,
            blocked_by_response_id=blocked_by_response_id or None,
        )
        parent_state = parent_entry[1] if parent_entry else None
        canonical_deliverable_state = "none"
        terminal_deliverable_state = "none"
        coverage_source = "none"
        if parent_state is not None:
            parent_covered, coverage_source, deliverable_observed, deliverable_class, terminal_selected, terminal_reason = self._parent_response_coverage_state(
                parent_state=parent_state,
            )
            canonical_deliverable_state = (
                f"done={str(bool(getattr(parent_state, 'done', False))).lower()},"
                f"deliverable_observed={str(deliverable_observed).lower()},"
                f"deliverable_class={deliverable_class or 'unknown'}"
            )
            terminal_deliverable_state = (
                f"selected={str(terminal_selected).lower()},"
                f"reason={terminal_reason}"
            )
            logger.info(
                "parent_coverage_source_of_truth run_id=%s parent_response_id=%s covered=%s source=%s canonical_observed=%s canonical_class=%s terminal_selected=%s terminal_reason=%s",
                self._current_run_id() or "",
                str(getattr(parent_state, "response_id", "") or "").strip() or "none",
                str(parent_covered).lower(),
                coverage_source,
                str(deliverable_observed).lower(),
                deliverable_class or "unknown",
                str(terminal_selected).lower(),
                terminal_reason,
            )
        logger.info(
            "create_seam_parent_coverage_eval run_id=%s turn_id=%s canonical_key=%s tool_call_id=%s drop_decision=%s reason=%s resolved_parent_response_id=%s resolved_parent_origin=%s resolved_parent_canonical_deliverable_state=%s resolved_parent_terminal_deliverable_state=%s coverage_source=%s",
            self._current_run_id() or "",
            turn_id,
            canonical_key,
            str(response_metadata.get("tool_call_id") or "").strip() or "unknown",
            str(should_drop).lower(),
            reason,
            str(getattr(parent_state, "response_id", "") or "").strip() or "none",
            str(getattr(parent_state, "origin", "") or "").strip() or "none",
            canonical_deliverable_state,
            terminal_deliverable_state,
            coverage_source,
        )
        if not should_drop:
            return False
        self._set_tool_followup_state(
            canonical_key=canonical_key,
            state="dropped",
            reason=f"parent_covered_tool_result seam={drain_trigger}",
        )
        logger.info(
            "tool_followup_create_suppressed turn_id=%s canonical_key=%s reason=parent_covered_tool_result seam=%s",
            turn_id,
            canonical_key,
            drain_trigger,
        )
        return True

    def _turn_has_pending_tool_followup(self, *, turn_id: str) -> bool:
        normalized_turn_id = str(turn_id or "").strip()
        if not normalized_turn_id:
            return False
        followup_states = getattr(self, "_tool_followup_state_by_canonical_key", None)
        if not isinstance(followup_states, dict):
            return False
        pending_states = {
            "scheduled",
            "blocked_active_response",
            "scheduled_release",
            "released_on_response_done",
            "creating",
            "created",
        }
        turn_prefix = f":{normalized_turn_id}:"
        for canonical_key, raw_state in followup_states.items():
            normalized_canonical_key = str(canonical_key or "").strip()
            if ":tool:" not in normalized_canonical_key:
                continue
            if turn_prefix not in normalized_canonical_key:
                continue
            state = str(raw_state or "").strip().lower() or "new"
            if state in pending_states:
                return True
        return False

    def _response_done_deliverable_decision(
        self,
        *,
        turn_id: str,
        origin: str,
        delivery_state_before_done: str | None,
        active_response_was_provisional: bool,
        done_canonical_key: str,
    ) -> tuple[bool, str]:
        if delivery_state_before_done == "cancelled":
            return False, "cancelled"
        if str(origin or "").strip().lower() == "micro_ack":
            return False, "micro_ack_non_deliverable"
        if active_response_was_provisional and self._is_empty_response_done(canonical_key=done_canonical_key):
            return False, "provisional_empty_non_deliverable"
        if (
            str(origin or "").strip().lower() == "server_auto"
            and self._turn_has_pending_tool_followup(turn_id=turn_id)
        ):
            return False, "tool_followup_precedence"
        return True, "normal"

    def _terminal_deliverable_selection_store(self) -> dict[str, dict[str, Any]]:
        store = getattr(self, "_terminal_deliverable_selection_by_response_id", None)
        if not isinstance(store, dict):
            store = {}
            self._terminal_deliverable_selection_by_response_id = store
        return store

    def _apply_terminal_deliverable_selection(
        self,
        *,
        canonical_key: str,
        response_id: str,
        turn_id: str,
        input_event_key: str | None,
        selected: bool,
        selection_reason: str,
    ) -> None:
        normalized_response_id = str(response_id or "").strip()
        normalized_canonical_key = str(canonical_key or "").strip()
        if normalized_response_id:
            self._terminal_deliverable_selection_store()[normalized_response_id] = {
                "selected": bool(selected),
                "reason": str(selection_reason or "unknown").strip().lower() or "unknown",
                "canonical_key": normalized_canonical_key,
            }

        if not normalized_canonical_key:
            return

        normalized_reason = str(selection_reason or "").strip().lower()

        def _mutator(record: CanonicalResponseState) -> None:
            record.done = True
            if normalized_response_id:
                record.response_id = normalized_response_id
            if selected:
                record.created = True
                record.deliverable_observed = True
                if str(record.deliverable_class or "unknown").strip().lower() == "unknown":
                    record.deliverable_class = "final"
            elif normalized_reason in {"micro_ack_non_deliverable", "cancelled", "provisional_empty_non_deliverable"}:
                if str(record.deliverable_class or "unknown").strip().lower() == "unknown":
                    record.deliverable_class = "non_deliverable"

        self._canonical_response_state_mutate(
            canonical_key=normalized_canonical_key,
            turn_id=turn_id,
            input_event_key=input_event_key,
            mutator=_mutator,
        )
        logger.info(
            "terminal_deliverable_selection_applied run_id=%s turn_id=%s canonical_key=%s response_id=%s selected=%s reason=%s",
            self._current_run_id() or "",
            turn_id,
            normalized_canonical_key,
            normalized_response_id or "none",
            str(bool(selected)).lower(),
            normalized_reason or "unknown",
        )

    def _parent_response_coverage_state(
        self,
        *,
        parent_state: CanonicalResponseState,
    ) -> tuple[bool, str, bool, str, bool, str]:
        deliverable_class = str(getattr(parent_state, "deliverable_class", "") or "").strip().lower()
        deliverable_observed = bool(getattr(parent_state, "deliverable_observed", False))
        canonical_covered = deliverable_observed or deliverable_class in {"progress", "final"}
        normalized_response_id = str(getattr(parent_state, "response_id", "") or "").strip()
        terminal_selected = False
        terminal_reason = "unknown"
        if normalized_response_id:
            selection_entry = self._terminal_deliverable_selection_store().get(normalized_response_id)
            if isinstance(selection_entry, dict):
                terminal_selected = bool(selection_entry.get("selected", False))
                terminal_reason = str(selection_entry.get("reason") or "unknown").strip().lower() or "unknown"
        covered = canonical_covered or terminal_selected
        if canonical_covered:
            source = "canonical"
        elif terminal_selected:
            source = "terminal_selection"
        else:
            source = "none"
        return covered, source, deliverable_observed, deliverable_class, terminal_selected, terminal_reason

    def _turn_has_final_deliverable(self, *, turn_id: str) -> bool:
        normalized_turn_id = str(turn_id or "").strip()
        if not normalized_turn_id:
            return False
        for state in self._canonical_response_state_store().values():
            if not isinstance(state, CanonicalResponseState):
                continue
            if str(state.turn_id or "").strip() != normalized_turn_id:
                continue
            if str(getattr(state, "deliverable_class", "unknown") or "unknown").strip().lower() == "final":
                return True
        return False

    def _assistant_message_same_turn_owner_reason(
        self,
        *,
        turn_id: str,
        input_event_key: str | None,
        canonical_key: str,
    ) -> str | None:
        normalized_turn_id = str(turn_id or "").strip()
        if not normalized_turn_id:
            return None

        if self._turn_has_pending_tool_followup(turn_id=normalized_turn_id):
            return "tool_followup_owned"

        if self._turn_has_final_deliverable(turn_id=normalized_turn_id):
            return "terminal_deliverable_owned"

        lifecycle_state = self._lifecycle_controller().state_for(canonical_key)
        if lifecycle_state in {
            InteractionLifecycleState.DONE,
            InteractionLifecycleState.REPLACED,
            InteractionLifecycleState.CANCELLED,
        }:
            return f"canonical_{lifecycle_state.value}"

        if self._is_active_response_blocking() and bool(getattr(self, "_active_response_consumes_canonical_slot", True)):
            active_origin = str(getattr(self, "_active_response_origin", "") or "").strip().lower()
            active_turn_id = str(getattr(self, "_current_response_turn_id", "") or "").strip()
            active_input_event_key = str(getattr(self, "_active_response_input_event_key", "") or "").strip()
            active_canonical_key = str(getattr(self, "_active_response_canonical_key", "") or "").strip()
            if not active_canonical_key and active_turn_id and active_input_event_key:
                active_canonical_key = self._canonical_utterance_key(
                    turn_id=active_turn_id,
                    input_event_key=active_input_event_key,
                )
            if active_origin != "micro_ack" and active_canonical_key and active_canonical_key == canonical_key:
                return "active_response_owned"

        return None

    def _release_blocked_tool_followups_for_response_done(self, *, response_id: str) -> None:
        normalized_response_id = str(response_id or "").strip()
        if not normalized_response_id:
            return

        def _iter_candidates() -> list[tuple[str, dict[str, Any], str, str]]:
            candidates: list[tuple[str, dict[str, Any], str, str]] = []
            pending = getattr(self, "_pending_response_create", None)
            if pending is not None and isinstance(getattr(pending, "event", None), dict):
                candidates.append(("pending", pending.event, str(getattr(pending, "origin", "") or "unknown"), str(getattr(pending, "turn_id", "") or "turn-unknown")))
            for queued in list(getattr(self, "_response_create_queue", deque()) or ()):
                if not isinstance(queued, dict):
                    continue
                event = queued.get("event")
                if not isinstance(event, dict):
                    continue
                candidates.append(("queue", event, str(queued.get("origin") or "unknown"), str(queued.get("turn_id") or "turn-unknown")))
            return candidates

        released_canonical_keys: set[str] = set()
        for source, response_create_event, origin, turn_id in _iter_candidates():
            response_metadata = self._extract_response_create_metadata(response_create_event)
            is_tool_followup = str(response_metadata.get("tool_followup", "")).strip().lower() in {"true", "1", "yes"}
            if not is_tool_followup:
                continue
            blocked_by_response_id = str(response_metadata.get("blocked_by_response_id") or "").strip()
            if blocked_by_response_id and blocked_by_response_id != normalized_response_id:
                continue
            input_event_key = str(response_metadata.get("input_event_key") or "").strip()
            canonical_key = self._canonical_utterance_key(turn_id=turn_id, input_event_key=input_event_key)
            if canonical_key in released_canonical_keys:
                continue
            if self._tool_followup_state(canonical_key=canonical_key) != "blocked_active_response":
                continue
            should_drop, parent_entry, reason = self._should_suppress_queued_tool_followup_release(
                response_metadata=response_metadata,
                blocked_by_response_id=normalized_response_id,
            )
            parent_state = parent_entry[1] if parent_entry else None
            logger.info(
                "queue_release_parent_eval run_id=%s turn_id=%s canonical_key=%s tool_call_id=%s release_decision=%s reason=%s resolved_parent_response_id=%s resolved_parent_origin=%s resolved_parent_deliverable_state=%s",
                self._current_run_id() or "",
                turn_id,
                canonical_key,
                str(response_metadata.get("tool_call_id") or "").strip() or "unknown",
                "drop" if should_drop else "release",
                reason,
                str(getattr(parent_state, "response_id", "") or "").strip() or "none",
                str(getattr(parent_state, "origin", "") or "").strip() or "none",
                (
                    f"done={str(bool(getattr(parent_state, 'done', False))).lower()},"
                    f"deliverable_observed={str(bool(getattr(parent_state, 'deliverable_observed', False))).lower()},"
                    f"deliverable_class={str(getattr(parent_state, 'deliverable_class', 'unknown') or 'unknown').strip().lower()}"
                )
                if parent_state is not None
                else "none",
            )
            if should_drop:
                self._set_tool_followup_state(
                    canonical_key=canonical_key,
                    state="dropped",
                    reason=f"parent_covered_tool_result response_id={normalized_response_id}",
                )
                logger.info(
                    "tool_followup_release_suppressed turn_id=%s origin=%s reason=parent_covered_tool_result response_id=%s",
                    turn_id,
                    origin,
                    normalized_response_id,
                )
                continue
            response_metadata["tool_followup_release"] = "true"
            self._set_tool_followup_state(
                canonical_key=canonical_key,
                state="scheduled_release",
                reason=f"response_done response_id={normalized_response_id}",
            )
            logger.info(
                "response_create_scheduled turn_id=%s origin=%s reason=release_after_response_done",
                turn_id,
                origin,
            )
            released_canonical_keys.add(canonical_key)

    def _response_has_safety_override(self, response_create_event: dict[str, Any]) -> bool:
        metadata = self._extract_response_create_metadata(response_create_event)
        return str(metadata.get("safety_override", "")).strip().lower() in {"true", "1", "yes"}

    def _empty_response_retry_idempotency_key(self, *, canonical_key: str) -> str:
        return self._response_lifecycle_tracker().empty_response_retry_idempotency_key(canonical_key=canonical_key)

    @staticmethod
    def _strip_empty_retry_suffix_lineage(input_event_key: str | None) -> str:
        return ResponseLifecycleTracker.strip_empty_retry_suffix_lineage(input_event_key)

    def _canonical_key_for_empty_retry_origin(self, *, turn_id: str, input_event_key: str | None) -> str:
        return self._response_lifecycle_tracker().canonical_key_for_empty_retry_origin(
            turn_id=turn_id,
            input_event_key=input_event_key,
        )

    async def _emit_empty_response_retry_exhausted_fallback(
        self,
        *,
        websocket: Any,
        turn_id: str,
        input_event_key: str,
        canonical_key: str,
        origin: str,
    ) -> None:
        if websocket is None:
            return
        exhausted_registry = getattr(self, "_empty_response_retry_fallback_emitted", None)
        if not isinstance(exhausted_registry, set):
            exhausted_registry = set()
            self._empty_response_retry_fallback_emitted = exhausted_registry
        if canonical_key in exhausted_registry:
            return
        exhausted_registry.add(canonical_key)
        await self.send_assistant_message(
            "Sorry—I’m having trouble generating a response right now.",
            websocket,
            speak=False,
        )
        self._lifecycle_controller().on_response_done(canonical_key)
        self._set_response_delivery_state(
            turn_id=turn_id,
            input_event_key=input_event_key,
            state="done",
        )
        self._log_lifecycle_event(
            turn_id=turn_id,
            input_event_key=input_event_key,
            canonical_key=canonical_key,
            origin=origin,
            response_id=None,
            decision="transition_terminal:empty_response_retry_exhausted",
        )
        self._debug_dump_canonical_key_timeline(
            canonical_key=canonical_key,
            trigger="empty_response_retry_exhausted",
        )

    def _is_empty_response_done(self, *, canonical_key: str) -> bool:
        return self._response_lifecycle_tracker().is_empty_response_done(canonical_key=canonical_key)

    async def _maybe_schedule_empty_response_retry(
        self,
        *,
        websocket: Any,
        turn_id: str,
        canonical_key: str,
        input_event_key: str,
        origin: str,
        delivery_state_before_done: str | None,
    ) -> None:
        await self._response_lifecycle_tracker().maybe_schedule_empty_response_retry(
            websocket=websocket,
            turn_id=turn_id,
            canonical_key=canonical_key,
            input_event_key=input_event_key,
            origin=origin,
            delivery_state_before_done=delivery_state_before_done,
        )

    def _response_lifecycle_tracker(self) -> ResponseLifecycleTracker:
        tracker = getattr(self, "_response_lifecycle", None)
        if not isinstance(tracker, ResponseLifecycleTracker):
            tracker = ResponseLifecycleTracker(self)
            self._response_lifecycle = tracker
        return tracker


    def _response_terminal_handlers_module(self) -> ResponseTerminalHandlers:
        handlers = getattr(self, "_response_terminal_handlers", None)
        if not isinstance(handlers, ResponseTerminalHandlers):
            handlers = ResponseTerminalHandlers(self)
            self._response_terminal_handlers = handlers
        return handlers

    def _response_is_explicit_multipart(self, metadata: dict[str, Any] | None) -> bool:
        if not isinstance(metadata, dict):
            return False
        return str(metadata.get("explicit_multipart", "")).strip().lower() in {"true", "1", "yes"}

    def _should_upgrade_with_audio_started(self, metadata: dict[str, Any] | None, *, origin: str) -> bool:
        if not isinstance(metadata, dict):
            return False
        if str(metadata.get("transcript_upgrade_replacement", "")).strip().lower() in {"true", "1", "yes"}:
            return True
        return str(origin or "").strip().lower() == "upgraded_response"

    def _canonical_lifecycle_state(self, canonical_key: str) -> dict[str, bool]:
        lifecycle_state = getattr(self, "_canonical_response_lifecycle_state", None)
        if not isinstance(lifecycle_state, dict):
            lifecycle_state = {}
            self._canonical_response_lifecycle_state = lifecycle_state
        normalized_canonical_key = str(canonical_key or "").strip()
        if not normalized_canonical_key:
            return {}
        existing = lifecycle_state.get(normalized_canonical_key)
        if not isinstance(existing, dict):
            existing = {}
            lifecycle_state[normalized_canonical_key] = existing
        return existing

    def _append_lifecycle_timeline_event(self, *, canonical_key: str, entry: str) -> None:
        normalized_canonical_key = str(canonical_key or "").strip()
        if not normalized_canonical_key:
            return
        timeline_store = getattr(self, "_lifecycle_canonical_timeline", None)
        if not isinstance(timeline_store, dict):
            timeline_store = {}
            self._lifecycle_canonical_timeline = timeline_store
        timeline = timeline_store.get(normalized_canonical_key)
        if not isinstance(timeline, deque):
            timeline = deque(maxlen=64)
            timeline_store[normalized_canonical_key] = timeline
        timeline.append(entry)

    def _debug_dump_canonical_key_timeline(self, *, canonical_key: str, trigger: str) -> None:
        normalized_canonical_key = str(canonical_key or "").strip()
        if not normalized_canonical_key:
            return
        timeline_store = getattr(self, "_lifecycle_canonical_timeline", None)
        timeline = timeline_store.get(normalized_canonical_key) if isinstance(timeline_store, dict) else None
        if not isinstance(timeline, deque):
            return
        logger.debug(
            "lifecycle_timeline_dump run_id=%s canonical_key=%s trigger=%s events=%s",
            self._current_run_id() or "",
            normalized_canonical_key,
            trigger,
            json.dumps(list(timeline), separators=(",", ":")),
        )

    def _log_lifecycle_event(
        self,
        *,
        turn_id: str,
        input_event_key: str | None,
        canonical_key: str,
        origin: str,
        response_id: str | None,
        decision: str,
        level: int = logging.INFO,
        upgrade_chain_id: str | None = None,
    ) -> None:
        resolved_turn_id = str(turn_id or "").strip() or "turn-unknown"
        resolved_input_event_key = str(input_event_key or "").strip() or "unknown"
        resolved_canonical_key = str(canonical_key or "").strip() or self._canonical_utterance_key(
            turn_id=resolved_turn_id,
            input_event_key=resolved_input_event_key,
        )
        resolved_origin = str(origin or "").strip() or "unknown"
        resolved_response_id = str(response_id or "").strip() or "none"
        resolved_decision = str(decision or "").strip() or "unknown"
        resolved_upgrade_chain_id = str(upgrade_chain_id or "").strip() or self._upgrade_chain_id_from_response(response_id)
        resolved_level = level
        if (
            resolved_level >= logging.INFO
            and resolved_decision.startswith("audio_delta_allow:")
        ):
            resolved_level = logging.DEBUG
        logger.log(
            resolved_level,
            "lifecycle_event run_id=%s turn_id=%s input_event_key=%s canonical_key=%s origin=%s response_id=%s decision=%s upgrade_chain_id=%s",
            self._current_run_id() or "",
            resolved_turn_id,
            resolved_input_event_key,
            resolved_canonical_key,
            resolved_origin,
            resolved_response_id,
            resolved_decision,
            resolved_upgrade_chain_id or "none",
        )
        self._append_lifecycle_timeline_event(
            canonical_key=resolved_canonical_key,
            entry=(
                f"decision={resolved_decision};origin={resolved_origin};"
                f"response_id={resolved_response_id};turn_id={resolved_turn_id};"
                f"input_event_key={resolved_input_event_key};upgrade_chain_id={resolved_upgrade_chain_id or 'none'}"
            ),
        )

    def _canonical_first_audio_started(self, canonical_key: str) -> bool:
        if self._lifecycle_controller().audio_started(canonical_key):
            return True
        unified_state = self._canonical_response_state(canonical_key)
        if isinstance(unified_state, CanonicalResponseState) and unified_state.audio_started:
            return True
        state = self._canonical_lifecycle_state(canonical_key)
        return bool(state.get("first_audio_started", False))

    def _load_topic_suppression_preferences(self) -> None:
        profile_manager = getattr(self, "profile_manager", None)
        if profile_manager is None:
            return
        try:
            profile = profile_manager.load_active_profile()
        except Exception:
            return
        prefs = profile.preferences if isinstance(profile.preferences, dict) else {}
        suppressed = prefs.get("suppressed_topics")
        if not isinstance(suppressed, list):
            return
        self._suppressed_topics = {str(topic).strip().lower() for topic in suppressed if str(topic).strip()}

    def _persist_topic_suppression_preferences(self) -> None:
        profile_manager = getattr(self, "profile_manager", None)
        if profile_manager is None:
            return
        try:
            profile = profile_manager.load_active_profile()
            prefs = dict(profile.preferences or {})
            prefs["suppressed_topics"] = sorted(self._suppressed_topics)
            profile_manager.update_active_profile_fields(preferences=prefs)
        except Exception as exc:
            logger.debug("topic_suppression_persist_failed error=%s", exc)

    def _update_topic_suppression_from_user_text(self, text: str) -> None:
        normalized = " ".join((text or "").strip().lower().split())
        if not normalized:
            return
        if "stop talking about battery" in normalized:
            if "battery" not in self._suppressed_topics:
                self._suppressed_topics.add("battery")
                self._persist_topic_suppression_preferences()
                logger.info("topic_suppression_updated topic=battery suppressed=true")
            return
        if "resume talking about battery" in normalized or "you can talk about battery" in normalized:
            if "battery" in self._suppressed_topics:
                self._suppressed_topics.discard("battery")
                self._persist_topic_suppression_preferences()
                logger.info("topic_suppression_updated topic=battery suppressed=false")

    def _is_battery_safety_override(self, event: Event) -> bool:
        return self._battery_policy().is_safety_override(event)

    def _battery_policy(self) -> BatteryInjectionPolicy:
        policy = getattr(self, "_battery_injection_policy", None)
        if isinstance(policy, BatteryInjectionPolicy):
            return policy
        policy = BatteryInjectionPolicy(self)
        self._battery_injection_policy = policy
        return policy

    def _next_input_event_key(self) -> str:
        counter = int(getattr(self, "_input_event_key_counter", 0)) + 1
        self._input_event_key_counter = counter
        return f"input_event_{counter}"

    def _resolve_input_event_key(self, event: dict[str, Any]) -> str:
        candidate_keys = (
            event.get("item_id"),
            event.get("event_id"),
            event.get("id"),
        )
        for candidate in candidate_keys:
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        return self._next_input_event_key()

    def _clear_pending_response_contenders(
        self,
        *,
        turn_id: str,
        input_event_key: str | None,
        reason: str,
    ) -> None:
        normalized_turn_id = str(turn_id or "").strip()
        normalized_input_event_key = str(input_event_key or "").strip()
        if not normalized_turn_id and not normalized_input_event_key:
            return

        removed_pending = 0
        pending = getattr(self, "_pending_response_create", None)
        if isinstance(pending, PendingResponseCreate):
            pending_metadata = self._extract_response_create_metadata(pending.event)
            pending_input_key = str(pending_metadata.get("input_event_key") or "").strip()
            if (
                (normalized_turn_id and pending.turn_id == normalized_turn_id)
                or (normalized_input_event_key and pending_input_key == normalized_input_event_key)
            ):
                self._pending_response_create = None
                removed_pending += 1

        response_create_queue = getattr(self, "_response_create_queue", None)
        if not isinstance(response_create_queue, deque):
            response_create_queue = deque()
            self._response_create_queue = response_create_queue

        kept: deque[dict[str, Any]] = deque()
        dropped = 0
        for queued in response_create_queue:
            queued_turn_id = str(queued.get("turn_id") or "").strip()
            queued_metadata = self._extract_response_create_metadata(queued.get("event") or {})
            queued_input_key = str(queued_metadata.get("input_event_key") or "").strip()
            if (
                (normalized_turn_id and queued_turn_id == normalized_turn_id)
                or (normalized_input_event_key and queued_input_key == normalized_input_event_key)
            ):
                dropped += 1
                continue
            kept.append(queued)
        if dropped:
            self._response_create_queue = kept
        if removed_pending or dropped:
            self._sync_pending_response_create_queue()
            logger.info(
                "response_contenders_cleared run_id=%s turn_id=%s input_event_key=%s removed_pending=%s removed_queued=%s reason=%s",
                self._current_run_id() or "",
                normalized_turn_id or "unknown",
                normalized_input_event_key or "unknown",
                removed_pending,
                dropped,
                reason,
            )

    def _clear_all_pending_response_creates(self, *, reason: str) -> None:
        removed_pending = 1 if self._pending_response_create is not None else 0
        removed_queued = len(self._response_create_queue)
        self._pending_response_create = None
        self._response_create_queue.clear()
        self._sync_pending_response_create_queue()
        if removed_pending or removed_queued:
            logger.info(
                "response_contenders_cleared run_id=%s turn_id=all input_event_key=all removed_pending=%s removed_queued=%s reason=%s",
                self._current_run_id() or "",
                removed_pending,
                removed_queued,
                reason,
            )

    def _drop_suppressed_scheduled_response_creates(self, *, turn_id: str, origin: str) -> None:
        normalized_origin = str(origin or "").strip().lower()
        if normalized_origin != "server_auto":
            return
        if not isinstance(getattr(self, "_queued_confirmation_reminder_keys", None), set):
            self._queued_confirmation_reminder_keys = set()
        pending = getattr(self, "_pending_response_create", None)
        if isinstance(pending, PendingResponseCreate) and pending.turn_id == turn_id:
            pending_origin = str(getattr(pending, "origin", "") or "").strip().lower()
            if pending_origin == normalized_origin:
                self._pending_response_create = None
                self._sync_pending_response_create_queue()
                self._record_dropped_queued_creates(turn_id=turn_id, dropped_count=1)
                logger.info(
                    "response_schedule_drop run_id=%s turn_id=%s origin=%s reason=preference_recall_suppressed",
                    self._current_run_id() or "",
                    turn_id,
                    normalized_origin,
                )

        response_create_queue = getattr(self, "_response_create_queue", None)
        if not isinstance(response_create_queue, deque):
            response_create_queue = deque()
            self._response_create_queue = response_create_queue
        if not response_create_queue:
            return
        kept: deque[dict[str, Any]] = deque()
        dropped = 0
        for queued in response_create_queue:
            queued_turn_id = str(queued.get("turn_id") or "").strip()
            queued_origin = str(queued.get("origin") or "").strip().lower()
            if queued_turn_id == turn_id and queued_origin == normalized_origin:
                dropped += 1
                continue
            kept.append(queued)
        if dropped:
            self._response_create_queue = kept
            self._sync_pending_response_create_queue()
            self._record_dropped_queued_creates(turn_id=turn_id, dropped_count=dropped)
            logger.info(
                "response_schedule_drop run_id=%s turn_id=%s origin=%s dropped=%s reason=preference_recall_suppressed",
                self._current_run_id() or "",
                turn_id,
                normalized_origin,
                dropped,
            )

    async def _suppress_preference_recall_server_auto_response(self, websocket: Any) -> None:
        return await preference_recall_runtime._suppress_preference_recall_server_auto_response(self, websocket)

    def _emit_preference_recall_skip_trace_if_needed(self, *, turn_id: str | None) -> None:
        return preference_recall_runtime._emit_preference_recall_skip_trace_if_needed(self, turn_id=turn_id)

    async def _maybe_handle_preference_recall_intent(
        self,
        user_text: str,
        websocket: Any,
        *,
        source: str,
    ) -> bool:
        return await preference_recall_runtime._maybe_handle_preference_recall_intent(
            self,
            user_text,
            websocket,
            source=source,
        )

    def _is_active_response_guarded(self) -> bool:
        return bool(
            getattr(self, "_active_response_confirmation_guarded", False)
            or getattr(self, "_active_response_preference_guarded", False)
        )


    def _should_skip_turn_memory_retrieval(self, user_text: str) -> bool:
        return self._get_memory_runtime().should_skip_turn_memory_retrieval(user_text)

    def _prepare_turn_memory_brief(
        self,
        user_text: str,
        *,
        source: str,
        memory_intent: bool = False,
        memory_intent_subtype: str = "none",
    ) -> None:
        self._get_memory_runtime().prepare_turn_memory_brief(
            user_text,
            source=source,
            memory_intent=memory_intent,
            memory_intent_subtype=memory_intent_subtype,
        )

    def _consume_pending_memory_brief_note(self) -> str | None:
        return self._get_memory_runtime().consume_pending_memory_brief_note()

    def _build_startup_memory_digest_note(self) -> str | None:
        return self._get_memory_runtime().build_startup_memory_digest_note()

    async def _send_memory_brief_note(self, websocket: Any, note_text: str) -> None:
        note_event = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "system",
                "content": [{"type": "input_text", "text": note_text}],
            },
        }
        log_ws_event("Outgoing", note_event)
        self._track_outgoing_event(note_event, origin="memory_brief")
        transport = self._get_or_create_transport()
        await transport.send_json(websocket, note_event)

    def _find_stop_word(self, text: str) -> str | None:
        return preference_recall_runtime._find_stop_word(self, text)

    def _set_pending_preference_memory_context(
        self,
        *,
        turn_id: str,
        input_event_key: str,
        memory_context: dict[str, Any],
    ) -> None:
        normalized_turn_id = str(turn_id or "").strip() or "turn-unknown"
        canonical_key = self._canonical_utterance_key(turn_id=turn_id, input_event_key=input_event_key)
        response_id = self._resolve_pending_preference_memory_context_response_id(
            turn_id=normalized_turn_id,
            canonical_key=canonical_key,
        )
        store = getattr(self, "_pending_preference_memory_context_by_canonical_key", None)
        if not isinstance(store, dict):
            store = {}
            self._pending_preference_memory_context_by_canonical_key = store
        turn_store = getattr(self, "_pending_preference_memory_context_by_turn_id", None)
        if not isinstance(turn_store, dict):
            turn_store = {}
            self._pending_preference_memory_context_by_turn_id = turn_store
        response_store = getattr(self, "_pending_preference_memory_context_by_response_id", None)
        if not isinstance(response_store, dict):
            response_store = {}
            self._pending_preference_memory_context_by_response_id = response_store
        response_pointer_store = getattr(self, "_pending_preference_memory_context_response_id_by_turn_id", None)
        if not isinstance(response_pointer_store, dict):
            response_pointer_store = {}
            self._pending_preference_memory_context_response_id_by_turn_id = response_pointer_store

        payload = dict(memory_context)
        record = {
            "payload": payload,
            "owner_turn_id": normalized_turn_id,
            "owner_canonical_key": canonical_key,
            "owner_response_id": response_id,
        }
        store[canonical_key] = record
        turn_store[normalized_turn_id] = record
        if response_id:
            response_store[response_id] = record
            response_pointer_store[normalized_turn_id] = response_id
        prompt_note = str(payload.get("prompt_note") or "").strip()
        logger.debug(
            "pref_recall_context_attached key=%s len=%s hit=%s",
            canonical_key,
            len(prompt_note),
            str(bool(payload.get("hit", False))).lower(),
        )
        logger.debug(
            "pref_recall_context_attached_debug run_id=%s turn_id=%s stored_under=%s store_scope=%s text_len=%s hit=%s returned_count=%s",
            self._current_run_id() or "",
            normalized_turn_id,
            canonical_key,
            "turn+key+response",
            len(prompt_note),
            str(bool(payload.get("hit", False))).lower(),
            int(payload.get("returned_count") or 0),
        )

    def _resolve_pending_preference_memory_context_response_id(
        self,
        *,
        turn_id: str,
        canonical_key: str,
    ) -> str:
        response_pointer_store = getattr(self, "_pending_preference_memory_context_response_id_by_turn_id", None)
        if not isinstance(response_pointer_store, dict):
            response_pointer_store = {}
            self._pending_preference_memory_context_response_id_by_turn_id = response_pointer_store
        by_canonical_key = getattr(self, "_response_id_by_canonical_key", None)
        mapped_response_id = ""
        if isinstance(by_canonical_key, dict):
            mapped_response_id = str(by_canonical_key.get(canonical_key) or "").strip()
        pointer_response_id = str(response_pointer_store.get(turn_id) or "").strip()
        return mapped_response_id or pointer_response_id

    def _coerce_pending_preference_memory_context_record(
        self,
        *,
        raw: dict[str, Any],
        fallback_turn_id: str,
        fallback_canonical_key: str,
        fallback_response_id: str,
    ) -> dict[str, Any]:
        payload = raw.get("payload") if isinstance(raw.get("payload"), dict) else raw
        owner_turn_id = str(raw.get("owner_turn_id") or fallback_turn_id or "").strip() or fallback_turn_id
        owner_canonical_key = (
            str(raw.get("owner_canonical_key") or fallback_canonical_key or "").strip() or fallback_canonical_key
        )
        owner_response_id = str(raw.get("owner_response_id") or fallback_response_id or "").strip()
        return {
            "payload": dict(payload) if isinstance(payload, dict) else {},
            "owner_turn_id": owner_turn_id,
            "owner_canonical_key": owner_canonical_key,
            "owner_response_id": owner_response_id,
        }

    def _record_matches_pending_preference_memory_context_owner(
        self,
        *,
        record: dict[str, Any],
        turn_id: str,
        canonical_key: str,
        response_id: str,
    ) -> bool:
        owner_turn_id = str(record.get("owner_turn_id") or "").strip()
        owner_canonical_key = str(record.get("owner_canonical_key") or "").strip()
        owner_response_id = str(record.get("owner_response_id") or "").strip()
        if owner_turn_id and owner_turn_id != turn_id:
            return False
        if owner_canonical_key and owner_canonical_key != canonical_key:
            return False
        if response_id and owner_response_id and owner_response_id != response_id:
            return False
        return True

    def _resolve_pending_preference_memory_context_payload(
        self,
        *,
        turn_id: str,
        input_event_key: str,
        consume: bool,
    ) -> tuple[dict[str, Any] | None, str]:
        normalized_turn_id = str(turn_id or "").strip() or "turn-unknown"
        store = getattr(self, "_pending_preference_memory_context_by_canonical_key", None)
        if not isinstance(store, dict):
            store = {}
            self._pending_preference_memory_context_by_canonical_key = store
        turn_store = getattr(self, "_pending_preference_memory_context_by_turn_id", None)
        if not isinstance(turn_store, dict):
            turn_store = {}
            self._pending_preference_memory_context_by_turn_id = turn_store
        response_store = getattr(self, "_pending_preference_memory_context_by_response_id", None)
        if not isinstance(response_store, dict):
            response_store = {}
            self._pending_preference_memory_context_by_response_id = response_store
        response_pointer_store = getattr(self, "_pending_preference_memory_context_response_id_by_turn_id", None)
        if not isinstance(response_pointer_store, dict):
            response_pointer_store = {}
            self._pending_preference_memory_context_response_id_by_turn_id = response_pointer_store

        base_canonical_key = self._canonical_utterance_key(
            turn_id=normalized_turn_id,
            input_event_key=input_event_key,
        )
        response_id = self._resolve_pending_preference_memory_context_response_id(
            turn_id=normalized_turn_id,
            canonical_key=base_canonical_key,
        )

        candidate_keys: list[str] = []

        def _add_candidate(key: str | None) -> None:
            canonical = self._canonical_utterance_key(turn_id=normalized_turn_id, input_event_key=key)
            if canonical not in candidate_keys:
                candidate_keys.append(canonical)

        _add_candidate(input_event_key)
        _add_candidate(self._active_input_event_key_for_turn(normalized_turn_id))
        _add_candidate(str(getattr(self, "_active_server_auto_input_event_key", "") or "").strip())
        _add_candidate(str(getattr(self, "_current_input_event_key", "") or "").strip())

        payload_candidates: list[tuple[str, dict[str, Any], str]] = []
        response_candidate_ids = [response_id]
        pointer_response_id = str(response_pointer_store.get(normalized_turn_id) or "").strip()
        if pointer_response_id and pointer_response_id not in response_candidate_ids:
            response_candidate_ids.append(pointer_response_id)
        for candidate_response_id in response_candidate_ids:
            if not candidate_response_id:
                continue
            payload = response_store.get(candidate_response_id)
            if isinstance(payload, dict):
                payload_candidates.append((candidate_response_id, payload, "response"))
        for candidate_key in candidate_keys:
            payload = store.get(candidate_key)
            if isinstance(payload, dict):
                payload_candidates.append((candidate_key, payload, "key"))
        turn_payload = turn_store.get(normalized_turn_id)
        if isinstance(turn_payload, dict):
            payload_candidates.append((normalized_turn_id, turn_payload, "turn"))

        if not payload_candidates:
            return None, ""

        def _rank(item: tuple[str, dict[str, Any], str]) -> tuple[int, int, int, int]:
            scope = item[2]
            payload = item[1].get("payload") if isinstance(item[1].get("payload"), dict) else item[1]
            return (
                1 if scope == "response" else 0,
                int(bool(payload.get("hit", False))),
                int(payload.get("returned_count") or 0),
                len(str(payload.get("prompt_note") or "")),
            )

        chosen_key, raw_chosen_payload, chosen_scope = max(payload_candidates, key=_rank)
        chosen_record = self._coerce_pending_preference_memory_context_record(
            raw=raw_chosen_payload,
            fallback_turn_id=normalized_turn_id,
            fallback_canonical_key=base_canonical_key,
            fallback_response_id=response_id,
        )
        if not self._record_matches_pending_preference_memory_context_owner(
            record=chosen_record,
            turn_id=normalized_turn_id,
            canonical_key=base_canonical_key,
            response_id=response_id,
        ):
            return None, ""

        chosen_payload = chosen_record.get("payload") if isinstance(chosen_record.get("payload"), dict) else {}
        owner_turn_id = str(chosen_record.get("owner_turn_id") or normalized_turn_id)
        owner_canonical_key = str(chosen_record.get("owner_canonical_key") or base_canonical_key)
        owner_response_id = str(chosen_record.get("owner_response_id") or "").strip()

        store[owner_canonical_key] = chosen_record
        turn_store[owner_turn_id] = chosen_record
        if owner_response_id:
            response_store[owner_response_id] = chosen_record
            response_pointer_store[owner_turn_id] = owner_response_id

        if consume:
            for candidate_key in candidate_keys:
                store.pop(candidate_key, None)
            store.pop(owner_canonical_key, None)
            turn_store.pop(normalized_turn_id, None)
            if owner_turn_id != normalized_turn_id:
                turn_store.pop(owner_turn_id, None)
            if owner_response_id:
                response_store.pop(owner_response_id, None)
                response_pointer_store.pop(owner_turn_id, None)
            response_pointer_store.pop(normalized_turn_id, None)
        return dict(chosen_payload), f"{chosen_scope}:{chosen_key}"

    def _has_pending_preference_memory_context(self, *, turn_id: str, input_event_key: str) -> bool:
        payload, _ = self._resolve_pending_preference_memory_context_payload(
            turn_id=turn_id,
            input_event_key=input_event_key,
            consume=False,
        )
        return isinstance(payload, dict)

    def _consume_pending_preference_memory_context_note(
        self,
        *,
        turn_id: str,
        input_event_key: str,
    ) -> str | None:
        payload, _ = self._resolve_pending_preference_memory_context_payload(
            turn_id=turn_id,
            input_event_key=input_event_key,
            consume=True,
        )
        if not isinstance(payload, dict):
            return None
        prompt_note = str(payload.get("prompt_note") or "").strip()
        return prompt_note or None

    def _peek_pending_preference_memory_context_payload(
        self,
        *,
        turn_id: str,
        input_event_key: str,
    ) -> dict[str, Any] | None:
        payload, _ = self._resolve_pending_preference_memory_context_payload(
            turn_id=turn_id,
            input_event_key=input_event_key,
            consume=False,
        )
        return payload if isinstance(payload, dict) else None

    def _tool_execution_cooldown_remaining(self) -> float:
        remaining = self._tool_execution_disabled_until - time.monotonic()
        return remaining if remaining > 0 else 0.0

    async def _handle_stop_word(
        self,
        text: str,
        websocket: Any,
        *,
        source: str,
    ) -> bool:
        stop_word = self._find_stop_word(text)
        if not stop_word:
            return False
        now = time.monotonic()
        if self._stop_word_cooldown_s > 0:
            self._tool_execution_disabled_until = max(
                self._tool_execution_disabled_until,
                now + self._stop_word_cooldown_s,
            )
        else:
            self._tool_execution_disabled_until = max(self._tool_execution_disabled_until, now)
        if self._pending_action:
            await self._reject_tool_call(
                self._pending_action.action,
                f"Stop word '{stop_word}' detected; tool execution paused.",
                websocket,
                staging=self._pending_action.staging,
                status="cancelled",
            )
            self._clear_pending_action()
        else:
            cooldown_msg = ""
            if self._stop_word_cooldown_s > 0:
                cooldown_msg = f" for {int(self._stop_word_cooldown_s)}s"
            await self.send_assistant_message(
                f"Stop word '{stop_word}' detected. Pending actions cancelled and tool use paused{cooldown_msg}.",
                websocket,
            )
        return True

    def _extract_transcript(self, event: dict[str, Any]) -> str | None:
        transcript = event.get("transcript")
        if isinstance(transcript, str) and transcript.strip():
            return transcript
        item = event.get("item", {})
        item_transcript = item.get("transcript")
        if isinstance(item_transcript, str) and item_transcript.strip():
            return item_transcript
        for content in item.get("content", []) or []:
            if isinstance(content, dict):
                content_transcript = content.get("transcript")
                if isinstance(content_transcript, str) and content_transcript.strip():
                    return content_transcript
                if content.get("type") == "input_text" and content.get("text"):
                    return content["text"]
        return None

    def _current_run_id(self) -> str | None:
        storage = getattr(self, "_storage", None)
        if storage is None:
            return None
        try:
            storage_info = storage.get_storage_info()
        except Exception:
            return None
        run_id = getattr(storage_info, "run_id", None)
        return str(run_id) if run_id else None

    def _redact_user_transcript_text(self, text: str) -> str:
        redacted = _EMAIL_REDACT_RE.sub("<redacted_email>", text)
        return _PHONE_REDACT_RE.sub("<redacted_phone>", redacted)

    def _should_log_partial_user_transcript(self, transcript: str) -> bool:
        cleaned = transcript.strip()
        if not cleaned:
            return False
        previous = getattr(self, "_last_logged_partial_user_transcript", "")
        if not previous:
            return True
        if cleaned == previous:
            return False
        delta = abs(len(cleaned) - len(previous))
        return delta >= self._log_user_transcript_partials_min_chars_delta

    def _log_user_transcript(self, transcript: str, *, final: bool, event_type: str) -> None:
        if not self._log_user_transcripts_enabled:
            return
        cleaned = " ".join((transcript or "").split())
        if not cleaned:
            return
        if not final:
            if not self._log_user_transcript_partials_enabled:
                return
            if not self._should_log_partial_user_transcript(cleaned):
                return
            self._last_logged_partial_user_transcript = cleaned
        else:
            self._last_logged_partial_user_transcript = ""

        log_text = (
            self._redact_user_transcript_text(cleaned)
            if self._log_user_transcript_redact_enabled
            else cleaned
        )
        escaped = log_text.replace("\\", "\\\\").replace('"', '\\"')
        meta = {
            "event_type": event_type,
            "run_id": self._current_run_id(),
            "source": "input_audio_transcription",
        }
        logger.info(
            '[USER] transcript %s: "%s" meta=%s',
            "final" if final else "partial",
            escaped,
            json.dumps(meta, sort_keys=True),
        )

    def _asr_meta_from_event(self, event: dict[str, Any]) -> dict[str, Any]:
        item = event.get("item") if isinstance(event.get("item"), dict) else {}
        content = item.get("content") if isinstance(item.get("content"), list) else []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("metadata"), dict):
                return dict(block.get("metadata") or {})
        metadata = event.get("metadata")
        if isinstance(metadata, dict):
            return dict(metadata)
        return {}

    def _log_utterance_trust_snapshot(
        self,
        *,
        transcript: str,
        event: dict[str, Any],
        turn_id: str,
        input_event_key: str,
    ) -> dict[str, Any]:
        duration_ms = None
        if isinstance(getattr(self, "_active_utterance", None), dict):
            duration_ms = self._active_utterance.get("duration_ms")
        snapshot_obj = build_utterance_trust_snapshot(
            run_id=self._current_run_id() or "",
            turn_id=turn_id,
            input_event_key=input_event_key,
            transcript_text=transcript,
            utterance_duration_ms=duration_ms,
            asr_meta=self._asr_meta_from_event(event),
            short_utterance_ms=self._asr_verify_short_utterance_ms,
        )
        snapshot = {
            "run_id": snapshot_obj.run_id,
            "turn_id": snapshot_obj.turn_id,
            "input_event_key": snapshot_obj.input_event_key,
            "transcript_text": snapshot_obj.transcript_text,
            "utterance_duration_ms": snapshot_obj.utterance_duration_ms,
            "word_count": snapshot_obj.word_count,
            "asr_confidence": snapshot_obj.asr_confidence,
            "asr_avg_logprob": snapshot_obj.asr_avg_logprob,
            "asr_no_speech_prob": snapshot_obj.asr_no_speech_prob,
            "short_utterance": snapshot_obj.short_utterance,
            "very_short_text": snapshot_obj.very_short_text,
            "contains_rare_terms": snapshot_obj.contains_rare_terms,
            "likely_proper_noun": snapshot_obj.likely_proper_noun,
            "topic_anchors": snapshot_obj.topic_anchors,
            "visual_question": snapshot_obj.visual_question,
            "vad_profile": str(self._vad_turn_detection.get("profile") or "unknown"),
            "vad_threshold": self._vad_turn_detection.get("threshold"),
            "vad_prefix_padding_ms": self._vad_turn_detection.get("prefix_padding_ms"),
            "vad_silence_duration_ms": self._vad_turn_detection.get("silence_duration_ms"),
        }
        self._utterance_trust_snapshot_by_input_event_key[input_event_key] = snapshot
        flags = [
            key for key in (
                "short_utterance",
                "very_short_text",
                "contains_rare_terms",
                "visual_question",
            ) if snapshot.get(key)
        ]
        logger.info(
            "UTTERANCE_TRUST run_id=%s turn_id=%s input_event_key=%s dur_ms=%s words=%s asr_conf=%s flags=%s anchors=%s",
            snapshot["run_id"],
            snapshot["turn_id"],
            snapshot["input_event_key"],
            snapshot["utterance_duration_ms"],
            snapshot["word_count"],
            snapshot["asr_confidence"],
            ",".join(flags) if flags else "none",
            ",".join(snapshot["topic_anchors"]),
        )
        return snapshot

    def _gating_verdict_key(self, *, turn_id: str, input_event_key: str) -> str:
        return self._canonical_utterance_key(turn_id=turn_id, input_event_key=input_event_key)

    def _get_response_gating_verdict(self, *, turn_id: str, input_event_key: str) -> ResponseGatingVerdict | None:
        store = getattr(self, "_response_gating_verdict_by_input_event_key", None)
        if not isinstance(store, dict):
            return None
        return store.get(self._gating_verdict_key(turn_id=turn_id, input_event_key=input_event_key))

    def _set_response_gating_verdict(self, *, turn_id: str, input_event_key: str, action: str, reason: str) -> ResponseGatingVerdict:
        store = getattr(self, "_response_gating_verdict_by_input_event_key", None)
        if not isinstance(store, dict):
            store = {}
            self._response_gating_verdict_by_input_event_key = store
        key = self._gating_verdict_key(turn_id=turn_id, input_event_key=input_event_key)
        existing = store.get(key)
        if isinstance(existing, ResponseGatingVerdict) and existing.action == "CLARIFY":
            return existing
        verdict = ResponseGatingVerdict(action=action, reason=reason, decided_at=time.monotonic())
        store[key] = verdict
        return verdict

    def _pending_tool_followup_likely(self) -> bool:
        pending_token = getattr(self, "_pending_confirmation_token", None)
        pending_action = getattr(pending_token, "pending_action", None)
        if pending_action is not None and getattr(pending_action, "action", None) is not None:
            return True
        if isinstance(getattr(self, "_deferred_research_tool_call", None), dict):
            return True
        return getattr(self, "_pending_research_request", None) is not None

    def _upgrade_likely_for_server_auto_turn(self, *, turn_id: str, input_event_key: str) -> tuple[bool, str]:
        partial_store = getattr(self, "_latest_partial_transcript_by_turn_id", None)
        partial_text = ""
        if isinstance(partial_store, dict):
            partial_text = str(partial_store.get(turn_id) or "").strip()
        memory_intent_likely = bool(partial_text) and self._is_memory_intent(partial_text)
        verify_clarify_likely = False
        if bool(getattr(self, "_asr_verify_on_risk_enabled", False)) and partial_text:
            vision_state = self.get_vision_state()
            verify_clarify_likely, _ = should_clarify(
                transcript_text=partial_text,
                snapshot=build_utterance_trust_snapshot(
                    run_id=self._current_run_id() or "",
                    turn_id=turn_id,
                    input_event_key=input_event_key,
                    transcript_text=partial_text,
                    utterance_duration_ms=getattr(self, "_asr_verify_short_utterance_ms", 450),
                    asr_meta={},
                    short_utterance_ms=self._asr_verify_short_utterance_ms,
                ),
                min_confidence=self._asr_verify_min_confidence,
                camera_available=bool(vision_state.get("available", False) or vision_state.get("can_capture", False)),
                camera_recent=bool(vision_state.get("available", False)),
            )
        pending_tool_followup_likely = self._pending_tool_followup_likely()
        reasons = [
            reason
            for reason, enabled in (
                ("memory_intent_likely", memory_intent_likely),
                ("verify_on_risk_clarify_likely", verify_clarify_likely),
                ("pending_tool_followup", pending_tool_followup_likely),
            )
            if enabled
        ]
        return bool(reasons), ",".join(reasons) if reasons else "none"

    def _should_delay_server_auto_until_transcript_final(self, *, turn_id: str, input_event_key: str) -> bool:
        verdict = self._get_response_gating_verdict(turn_id=turn_id, input_event_key=input_event_key)
        if not isinstance(verdict, ResponseGatingVerdict):
            return False
        if str(verdict.reason or "").strip().lower() != "awaiting_transcript_final":
            return False
        utterance = getattr(self, "_active_utterance", None)
        if not isinstance(utterance, dict):
            return True
        duration_ms = utterance.get("duration_ms")
        if isinstance(duration_ms, (int, float)) and duration_ms <= 2200:
            return True
        transcript_text = str(utterance.get("transcript") or "").strip()
        if transcript_text:
            return len(transcript_text.split()) <= 8
        return True

    def _start_audio_response_if_needed(self, *, response_id: str | None) -> None:
        normalized_response_id = str(response_id or "").strip()
        if not normalized_response_id:
            return
        started_ids = getattr(self, "_audio_response_started_ids", None)
        if not isinstance(started_ids, set):
            started_ids = set()
            self._audio_response_started_ids = started_ids
        if normalized_response_id in started_ids:
            return
        if self.audio_player:
            self.audio_player.start_response()
        started_ids.add(normalized_response_id)

    def _signal_server_auto_transcript_final(self, *, turn_id: str) -> None:
        pending = self._pending_server_auto_response_for_turn(turn_id=turn_id)
        self._set_server_auto_pre_audio_hold(
            turn_id=turn_id,
            enabled=False,
            reason="transcript_final_linked",
            response_id=(pending.response_id if pending is not None else None),
        )
        waiter = getattr(self, "_server_auto_audio_waiters_by_turn_id", {}).get(turn_id)
        if isinstance(waiter, asyncio.Event):
            waiter.set()

    def _should_start_deferred_server_auto_audio(self, *, turn_id: str, input_event_key: str, response_id: str) -> bool:
        if self._server_auto_pre_audio_hold_active(turn_id=turn_id, response_id=response_id):
            return False
        if self._is_cancelled_or_superseded_response_id(response_id):
            return False
        if str(getattr(self, "_active_response_id", "") or "").strip() != response_id:
            return False
        verdict = self._get_response_gating_verdict(turn_id=turn_id, input_event_key=input_event_key)
        if verdict is None or verdict.action in {"NOOP", "ANSWER"}:
            return True
        return False

    def _schedule_server_auto_audio_deferral(
        self,
        *,
        turn_id: str,
        input_event_key: str,
        response_id: str,
    ) -> None:
        waiters = getattr(self, "_server_auto_audio_waiters_by_turn_id", None)
        if not isinstance(waiters, dict):
            waiters = {}
            self._server_auto_audio_waiters_by_turn_id = waiters
        tasks = getattr(self, "_server_auto_audio_defer_tasks_by_turn_id", None)
        if not isinstance(tasks, dict):
            tasks = {}
            self._server_auto_audio_defer_tasks_by_turn_id = tasks
        existing = tasks.pop(turn_id, None)
        if existing is not None:
            existing.cancel()
        waiter = asyncio.Event()
        waiters[turn_id] = waiter

        async def _wait_for_upgrade_decision() -> None:
            micro_ack_sent_during_hold = False
            try:
                while True:
                    pending = self._pending_server_auto_response_for_turn(turn_id=turn_id)
                    active_pending_response_id = str(pending.response_id or "").strip() if pending is not None else ""
                    if not active_pending_response_id or active_pending_response_id != response_id:
                        logger.debug(
                            "server_auto_audio_deferral_waiter_stale run_id=%s turn_id=%s response_id=%s active_pending_response_id=%s reason=pending_mismatch",
                            self._current_run_id() or "",
                            turn_id,
                            response_id,
                            active_pending_response_id or "unknown",
                        )
                        break
                    if self._server_auto_pre_audio_hold_released_for_other_response(
                        turn_id=turn_id,
                        response_id=response_id,
                    ):
                        logger.debug(
                            "server_auto_audio_deferral_waiter_stale run_id=%s turn_id=%s response_id=%s reason=hold_released_for_other_response",
                            self._current_run_id() or "",
                            turn_id,
                            response_id,
                        )
                        break
                    timed_out = False
                    try:
                        await asyncio.wait_for(waiter.wait(), timeout=self._server_auto_audio_deferral_timeout_ms / 1000.0)
                    except asyncio.TimeoutError:
                        timed_out = True
                    if self._server_auto_pre_audio_hold_active(turn_id=turn_id, response_id=response_id):
                        if self._is_cancelled_or_superseded_response_id(response_id) or str(getattr(self, "_active_response_id", "") or "").strip() != response_id:
                            self._set_server_auto_pre_audio_hold(
                                turn_id=turn_id,
                                enabled=False,
                                reason="proceed_without_transcript_final",
                                response_id=response_id,
                            )
                            break
                        if timed_out and not micro_ack_sent_during_hold:
                            self._maybe_schedule_micro_ack(
                                turn_id=turn_id,
                                category=self._micro_ack_category_for_reason("transcript_finalized"),
                                channel="voice",
                                action=self._canonical_utterance_key(turn_id=turn_id, input_event_key=input_event_key),
                                reason="transcript_finalized",
                                expected_delay_ms=700,
                            )
                            micro_ack_sent_during_hold = True
                        logger.info(
                            "server_auto_audio_start_deferred run_id=%s turn_id=%s response_id=%s timeout_ms=%s reasons=awaiting_transcript_final_pre_audio_hold",
                            self._current_run_id() or "",
                            turn_id,
                            response_id,
                            self._server_auto_audio_deferral_timeout_ms,
                        )
                        waiter.clear()
                        continue
                    if self._should_start_deferred_server_auto_audio(
                        turn_id=turn_id,
                        input_event_key=input_event_key,
                        response_id=response_id,
                    ):
                        self._start_audio_response_if_needed(response_id=response_id)
                    break
            finally:
                waiters.pop(turn_id, None)
                tasks.pop(turn_id, None)

        tasks[turn_id] = asyncio.create_task(_wait_for_upgrade_decision())

    def get_vision_state(self, now: float | None = None) -> dict[str, Any]:
        ts = time.monotonic() if now is None else float(now)
        camera = getattr(self, "camera_controller", None)
        can_capture = bool(camera)
        queued_frame_count = 0
        if camera is not None:
            pending = getattr(camera, "_pending_images", None)
            if pending is not None:
                try:
                    queued_frame_count = len(pending)
                except Exception:
                    queued_frame_count = 0
        last_sent = getattr(self, "_last_vision_frame_sent_at_monotonic", None)
        last_frame_age_ms = None
        if isinstance(last_sent, (int, float)):
            last_frame_age_ms = max(0, int((ts - float(last_sent)) * 1000))
        available = queued_frame_count > 0 or (last_frame_age_ms is not None and last_frame_age_ms <= 5000)
        return {
            "available": bool(available),
            "last_frame_age_ms": last_frame_age_ms,
            "queued_frame_count": queued_frame_count,
            "can_capture": can_capture,
        }

    async def _maybe_verify_on_risk_clarify(
        self,
        *,
        transcript: str,
        websocket: Any,
        turn_id: str,
        input_event_key: str,
        snapshot: dict[str, Any],
    ) -> bool:
        if not self._asr_verify_on_risk_enabled:
            return False
        if input_event_key in self._asr_clarify_asked_input_event_keys:
            return False
        if self._asr_clarify_count_by_turn.get(turn_id, 0) >= self._asr_verify_max_clarify_per_turn:
            return False
        vision_state = self.get_vision_state()
        normalized_transcript = " ".join((transcript or "").lower().split())
        known_domain = self._is_memory_intent(transcript) or any(
            marker in normalized_transcript
            for marker in ("look center", "look left", "look right", "look up", "look down", "favorite", "favourite", "prefer")
        )
        should_confirm, reason = should_clarify(
            transcript_text=transcript,
            snapshot=build_utterance_trust_snapshot(
                run_id=snapshot.get("run_id", ""),
                turn_id=turn_id,
                input_event_key=input_event_key,
                transcript_text=transcript,
                utterance_duration_ms=snapshot.get("utterance_duration_ms"),
                asr_meta={
                    "confidence": snapshot.get("asr_confidence"),
                    "avg_logprob": snapshot.get("asr_avg_logprob"),
                    "no_speech_prob": snapshot.get("asr_no_speech_prob"),
                },
                short_utterance_ms=self._asr_verify_short_utterance_ms,
            ),
            min_confidence=self._asr_verify_min_confidence,
            known_domain=known_domain,
            camera_available=bool(vision_state.get("available", False) or vision_state.get("can_capture", False)),
            camera_recent=bool(vision_state.get("available", False)),
        )
        if not should_confirm:
            self._set_response_gating_verdict(
                turn_id=turn_id,
                input_event_key=input_event_key,
                action="ANSWER",
                reason="verify_clear",
            )
            return False
        self._set_response_gating_verdict(
            turn_id=turn_id,
            input_event_key=input_event_key,
            action="CLARIFY",
            reason=reason,
        )
        self._asr_clarify_asked_input_event_keys.add(input_event_key)
        self._asr_clarify_count_by_turn[turn_id] = self._asr_clarify_count_by_turn.get(turn_id, 0) + 1
        pending = self._pending_server_auto_response_for_turn(turn_id=turn_id)
        if isinstance(pending, PendingServerAutoResponse) and pending.active and pending.response_id:
            cancel_event = {"type": "response.cancel", "response_id": pending.response_id}
            self._record_cancel_issued_timing(pending.response_id)
            self._stale_response_ids().add(pending.response_id)
            self._mark_pending_server_auto_response_cancelled(turn_id=turn_id, reason="verify_clarify")
            self._suppress_cancelled_response_audio(pending.response_id)
            transport = self._get_or_create_transport()
            await transport.send_json(websocket, cancel_event)
        self.assistant_reply = ""
        self._assistant_reply_accum = ""
        self._assistant_reply_response_id = None
        clarify_key = f"{input_event_key}:clarify"
        clarify_text = (
            "I can’t see right now. Want me to take a quick look with the camera?"
            if reason == "visual_unavailable"
            else "I heard you, but I’m not sure what you mean yet. Could you be a bit more specific?"
            if reason in {"low_semantic_confidence", "short_utterance"}
            else f"Quick check: did you ask \"{snapshot.get('transcript_text') or transcript.strip()}\"?"
        )
        if reason in {"low_semantic_confidence", "short_utterance"}:
            logger.info(
                "short_utterance_policy_applied run_id=%s turn_id=%s input_event_key=%s reason=%s words=%s",
                self._current_run_id() or "",
                turn_id,
                input_event_key,
                reason,
                snapshot.get("word_count", "unknown"),
            )
            logger.info(
                "ambiguous_utterance_detected run_id=%s turn_id=%s input_event_key=%s reason=%s asr_conf=%s",
                self._current_run_id() or "",
                turn_id,
                input_event_key,
                reason,
                snapshot.get("asr_confidence"),
            )
            if bool(vision_state.get("available", False) or vision_state.get("can_capture", False)):
                logger.info(
                    "context_enrichment_suppressed run_id=%s turn_id=%s input_event_key=%s reason=%s",
                    self._current_run_id() or "",
                    turn_id,
                    input_event_key,
                    reason,
                )
        await self.send_assistant_message(
            clarify_text,
            websocket,
            response_metadata={
                "trigger": "asr_verify_on_risk",
                "reason": reason,
                "input_event_key": clarify_key,
                "clarify_mode": "bounded",
            },
        )
        logger.info(
            "asr_verify_on_risk_clarify run_id=%s turn_id=%s input_event_key=%s reason=%s",
            self._current_run_id() or "",
            turn_id,
            input_event_key,
            reason,
        )
        logger.info(
            "clarification_response_selected run_id=%s turn_id=%s input_event_key=%s reason=%s",
            self._current_run_id() or "",
            turn_id,
            input_event_key,
            reason,
        )
        if reason in {"low_semantic_confidence", "short_utterance"}:
            logger.info(
                "low_semantic_confidence_fallback run_id=%s turn_id=%s input_event_key=%s reason=%s",
                self._current_run_id() or "",
                turn_id,
                input_event_key,
                reason,
            )
        return True

    def _has_camera_tool_result_for_turn(self, turn_id: str) -> bool:
        records = getattr(self, "_tool_call_records", None)
        if not isinstance(records, list) or not turn_id:
            return False
        normalized_turn_id = str(turn_id)
        for record in records:
            if not isinstance(record, dict):
                continue
            if str(record.get("turn_id") or "") != normalized_turn_id:
                continue
            tool_name = str(record.get("name") or "").strip().lower()
            if "camera" in tool_name:
                return True
        return False

    def _normalize_verify_clarify_message(
        self,
        *,
        message: str,
        metadata: dict[str, Any],
    ) -> str:
        trigger = str(metadata.get("trigger") or "").strip().lower()
        reason = str(metadata.get("reason") or "").strip().lower()
        if trigger != "asr_verify_on_risk" or reason != "visual_unavailable":
            return message
        turn_id = str(metadata.get("turn_id") or self._current_turn_id_or_unknown())
        if self._has_camera_tool_result_for_turn(turn_id):
            return message
        return "I can’t see right now. Want me to take a quick look with the camera?"

    def _is_bounded_clarify_mode(self, metadata: dict[str, Any]) -> bool:
        trigger = str(metadata.get("trigger") or "").strip().lower()
        reason = str(metadata.get("reason") or "").strip().lower()
        return trigger == "asr_verify_on_risk" and reason in {"low_semantic_confidence", "short_utterance"}

    def _bounded_clarify_response_create_event(
        self,
        *,
        message: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        response_create_event = {"type": "response.create", "response": {"metadata": metadata}}
        if not self._is_bounded_clarify_mode(metadata):
            return response_create_event
        response_payload = response_create_event.setdefault("response", {})
        response_payload["instructions"] = (
            "You are in bounded clarify mode. Speak exactly this one sentence and nothing else: "
            f"{message!r}. Do not add scene, IMU, memory, or environment commentary."
        )
        return response_create_event

    def _maybe_enqueue_reflection(self, trigger: str) -> None:
        if self._reflection_enqueued:
            logger.debug(
                "Reflection enqueue skipped: already enqueued (trigger=%s).",
                trigger,
            )
            return
        context = ReflectionContext(
            last_user_input=self._last_user_input_text,
            assistant_reply=self._assistant_reply_accum,
            tool_calls=list(self._tool_call_records),
            response_metadata={
                **self._last_response_metadata,
                "trigger": trigger,
                "last_user_input_source": self._last_user_input_source,
                "last_user_input_time": self._last_user_input_time,
            },
        )
        logger.debug(
            "Reflection enqueue requested (trigger=%s, tool_calls=%d, reply_len=%d).",
            trigger,
            len(self._tool_call_records),
            len(self._assistant_reply_accum),
        )
        self._last_tool_call_results = list(self._tool_call_records)
        self._reflection_coordinator.enqueue_reflection(context)
        self._reflection_enqueued = True

    def _clip_text(self, text: str | None, limit: int = 1200) -> str:
        if not text:
            return "(none)"
        return text if len(text) <= limit else f"{text[:limit]}…"

    def _find_gesture_definition(self, gesture_name: str) -> dict[str, Any] | None:
        for definition in DEFAULT_GESTURES:
            if definition.name == gesture_name:
                total_duration = sum(frame.duration_ms for frame in definition.frames)
                return {
                    "name": definition.name,
                    "total_duration_ms": total_duration,
                    "frame_count": len(definition.frames),
                }
        return None

    def _stage_action(self, action: ActionPacket) -> dict[str, Any]:
        tool_def = self._tool_definitions.get(action.tool_name)
        errors: list[str] = []
        warnings: list[str] = []
        bounds_checks: list[dict[str, Any]] = []

        if tool_def is None:
            warnings.append("Tool definition not found; skipped schema validation.")
        else:
            schema = tool_def.get("parameters") or {}
            properties = schema.get("properties") or {}
            required = set(schema.get("required") or [])
            provided = set(action.tool_args.keys())

            missing = sorted(required - provided)
            if missing:
                errors.append(f"Missing required args: {', '.join(missing)}.")

            extra = sorted(provided - set(properties.keys()))
            if extra:
                warnings.append(f"Unknown args will be ignored: {', '.join(extra)}.")

            for key, spec in properties.items():
                if key not in action.tool_args:
                    continue
                value = action.tool_args[key]
                expected = spec.get("type")
                if expected and not self._validate_arg_type(expected, value):
                    errors.append(f"Arg '{key}' expected type {expected}.")
                minimum = spec.get("minimum")
                maximum = spec.get("maximum")
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if minimum is not None or maximum is not None:
                        within = True
                        if minimum is not None and value < minimum:
                            within = False
                        if maximum is not None and value > maximum:
                            within = False
                        bounds_checks.append(
                            {
                                "field": key,
                                "value": value,
                                "min": minimum,
                                "max": maximum,
                                "within": within,
                            }
                        )
                        if not within:
                            errors.append(
                                f"Arg '{key}' out of bounds ({minimum}..{maximum})."
                            )

        motion_info = None
        if action.tool_name.startswith("gesture_"):
            gesture = self._find_gesture_definition(action.tool_name)
            motion_info = {
                "gesture": action.tool_name,
                "duration_ms": gesture["total_duration_ms"] if gesture else None,
                "frame_count": gesture["frame_count"] if gesture else None,
                "delay_ms": action.tool_args.get("delay_ms", 0),
                "intensity": action.tool_args.get("intensity", 1.0),
                "safe_servo_bounds": {"pan": [-90.0, 90.0], "tilt": [-45.0, 45.0]},
            }

        explanation = self._describe_staged_action(action, motion_info)

        return {
            "valid": not errors,
            "errors": errors,
            "warnings": warnings,
            "bounds_checks": bounds_checks,
            "explanation": explanation,
            "motion": motion_info,
        }

    def _validate_arg_type(self, expected: str, value: Any) -> bool:
        if expected == "string":
            return isinstance(value, str)
        if expected == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if expected == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if expected == "boolean":
            return isinstance(value, bool)
        if expected == "array":
            return isinstance(value, list)
        if expected == "object":
            return isinstance(value, dict)
        return True

    def _normalize_dry_run_flag(self, value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y"}
        return bool(value)

    def _extract_dry_run_flag(self, args: dict[str, Any]) -> bool:
        if "dry_run" not in args:
            return False
        return self._normalize_dry_run_flag(args.pop("dry_run"))

    def _build_tool_runtime_context(self, action: ActionPacket) -> dict[str, Any]:
        cost_score_map = {"cheap": 0.2, "med": 0.5, "expensive": 0.9}
        context: dict[str, Any] = {
            "cost_score": cost_score_map.get(str(getattr(action, "cost", "med")).lower(), 0.5),
        }
        requests_bucket = None
        if isinstance(self.rate_limits, dict):
            requests_bucket = self.rate_limits.get("requests")
        if isinstance(requests_bucket, dict):
            remaining = requests_bucket.get("remaining")
            if isinstance(remaining, (int, float)):
                context["rate_limit_remaining"] = float(remaining)

        privacy_keys = {"contains_pii", "sensitive", "private", "privacy_sensitive"}
        context["privacy_flag"] = any(
            self._normalize_dry_run_flag(getattr(action, "tool_args", {}).get(key))
            for key in privacy_keys
            if key in getattr(action, "tool_args", {})
        )
        return context

    def _normalize_confirmation_decision(
        self,
        tool_name: str,
        args: dict[str, Any],
        governance_decision: Any,
        runtime_context: dict[str, Any] | None,
    ) -> NormalizedConfirmationDecision:
        """Normalize governance decision fields used by confirmation plumbing."""
        decision = GovernanceLayer.coerce_decision_payload(governance_decision)
        action_summary_fallback = f"tool={tool_name} requires confirmation"
        normalized_payload = normalized_decision_payload(
            decision,
            action_summary_fallback=action_summary_fallback,
        )
        idempotency_key = str(
            normalized_payload["idempotency_key"]
            or build_normalized_idempotency_key(tool_name, args)
        )
        return NormalizedConfirmationDecision(
            status=str(decision.status),
            reason=str(decision.reason),
            action_summary=str(normalized_payload["action_summary"]),
            approved=str(decision.status) == "approved",
            needs_confirmation=bool(normalized_payload["confirm_required"]),
            confirm_required=bool(normalized_payload["confirm_required"]),
            confirm_reason=(
                str(normalized_payload["confirm_reason"])
                if normalized_payload["confirm_reason"] is not None
                else None
            ),
            confirm_prompt=(
                str(normalized_payload["confirm_prompt"])
                if normalized_payload["confirm_prompt"] is not None
                else None
            ),
            idempotency_key=idempotency_key,
            cooldown_seconds=float(normalized_payload["cooldown_seconds"]),
            dry_run_supported=bool(normalized_payload["dry_run_supported"]),
            decision_source=str(normalized_payload.get("decision_source") or "tier_default"),
            thresholds=self._compact_governance_thresholds(normalized_payload.get("thresholds") or runtime_context),
            max_reminders=self._coerce_optional_int(normalized_payload.get("max_reminders")),
            reminder_schedule_seconds=self._coerce_schedule_seconds(
                normalized_payload.get("reminder_schedule_seconds")
            ),
        )

    def _compact_governance_thresholds(self, thresholds: dict[str, Any] | None) -> dict[str, Any]:
        payload = dict(thresholds or {})
        return {key: value for key, value in payload.items() if value is not None}

    def _coerce_optional_int(self, value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _coerce_schedule_seconds(self, value: Any) -> tuple[float, ...]:
        if isinstance(value, str):
            raw_values: list[Any] = [part.strip() for part in value.split(",")]
        elif isinstance(value, (list, tuple)):
            raw_values = list(value)
        else:
            return ()
        parsed: list[float] = []
        for item in raw_values:
            try:
                parsed.append(max(0.0, float(item)))
            except (TypeError, ValueError):
                continue
        return tuple(parsed)

    def _describe_staged_action(
        self, action: ActionPacket, motion_info: dict[str, Any] | None
    ) -> str:
        if action.tool_name.startswith("read_") or action.tool_name.startswith("get_"):
            return f"Would read data via {action.tool_name}."
        if action.tool_name.startswith("recall_"):
            return f"Would fetch stored memories via {action.tool_name}."
        if action.tool_name.startswith("gesture_") and motion_info:
            duration = motion_info.get("duration_ms")
            return (
                f"Would queue gesture '{motion_info['gesture']}' "
                f"for ~{duration}ms within safe servo bounds."
            )
        return f"Would call {action.tool_name} with the proposed arguments."

    def _build_approval_prompt(
        self,
        action: ActionPacket,
        *,
        action_summary: str | None = None,
        confirm_prompt: str | None = None,
        confirm_reason: str | None = None,
    ) -> str:
        tool_metadata = self._governance.describe_tool(action.tool_name)
        dry_run_supported = bool(tool_metadata.get("dry_run_supported"))
        service = getattr(self, "_confirmation_service", None)
        if service is None:
            service = ConfirmationService(awaiting_timeout_s=20.0, late_decision_grace_s=15.0)
        prompt = service.build_prompt(
            action_summary=str(action_summary or ""),
            confirm_reason=str(confirm_reason or ""),
            dry_run_supported=dry_run_supported,
            confirm_prompt=confirm_prompt,
        )
        if confirm_prompt:
            normalized_override = " ".join(str(confirm_prompt).split()).strip()
            if prompt == normalized_override:
                logger.info(
                    "Approval prompt override accepted | tool=%s details=%s",
                    action.tool_name,
                    json.dumps({"confirm_prompt": normalized_override}, sort_keys=True),
                )
            else:
                logger.info(
                    "Approval prompt override ignored due to contract violation | tool=%s details=%s",
                    action.tool_name,
                    json.dumps({"confirm_prompt": normalized_override}, sort_keys=True),
                )
        return prompt

    def _log_structured_noop_event(
        self,
        *,
        outcome: str,
        reason: str,
        tool_name: str | None,
        action_id: str | None,
        token_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> None:
        run_id = self._current_run_id() or ""
        logger.info(
            "NO_OP_EVENT run_id=%s idempotency_key=%s reason=%s outcome=%s token=%s tool=%s action=%s",
            run_id,
            idempotency_key or "",
            reason,
            outcome,
            token_id or "",
            tool_name or "",
            action_id or "",
        )

    async def _emit_final_noop_user_text(
        self,
        websocket: Any,
        *,
        outcome: str,
        reason: str,
    ) -> bool:
        if self._has_active_confirmation_token():
            logger.info(
                "NO_OP_USER_TEXT_SKIPPED reason=token_not_resolved outcome=%s path_reason=%s",
                outcome,
                reason,
            )
            return False
        send_method = getattr(self, "send_assistant_message", None)
        default_send_method = getattr(type(self), "send_assistant_message", None)
        is_default_send_method = (
            callable(send_method)
            and hasattr(send_method, "__func__")
            and send_method.__func__ is default_send_method
        )
        if is_default_send_method:
            logger.info(
                "NO_OP_USER_TEXT_SKIPPED reason=default_send_method outcome=%s path_reason=%s",
                outcome,
                reason,
            )
            return False
        await self.send_assistant_message("No action taken.", websocket)
        return True

    async def _send_noop_tool_output(
        self,
        websocket: Any,
        *,
        call_id: str,
        status: str,
        message: str,
        tool_name: str,
        reason: str,
        category: str,
        action: ActionPacket | None = None,
        staging: dict[str, Any] | None = None,
        extra_fields: dict[str, Any] | None = None,
        include_response_create: bool = False,
        response_origin: str = "tool_output",
    ) -> None:
        payload: dict[str, Any] = {
            "status": status,
            "message": message,
            "tool": tool_name,
            "reason": reason,
            "no_op": {
                "executed": False,
                "category": category,
            },
            "confirmation": {
                "pending": self._has_active_confirmation_token() or self._is_awaiting_confirmation_phase(),
                "token": getattr(getattr(self, "_pending_confirmation_token", None), "id", ""),
            },
        }
        if action is not None:
            governance = getattr(self, "_governance", None)
            payload["tool_metadata"] = (
                governance.describe_tool(action.tool_name)
                if governance is not None and hasattr(governance, "describe_tool")
                else {}
            )
            payload["action_packet"] = action.to_payload()
            payload["staging"] = staging
            payload["summary"] = action.summary()
        if extra_fields:
            payload.update(extra_fields)
        function_call_output = {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(payload),
            },
        }
        log_ws_event("Outgoing", function_call_output)
        self._track_outgoing_event(function_call_output)
        transport = self._get_or_create_transport()
        await transport.send_json(websocket, function_call_output)
        if include_response_create:
            response_create_event, tool_followup_canonical_key = self._build_tool_followup_response_create_event(
                call_id=call_id,
                response_create_event={"type": "response.create"},
            )
            logger.info(
                "tool_followup_response_scheduled call_id=%s canonical_key=%s",
                call_id,
                tool_followup_canonical_key,
            )
            await self._send_response_create(websocket, response_create_event, origin=response_origin)

    def _format_action_packet(self, action: ActionPacket) -> str:
        alternatives = ", ".join(action.alternatives) if action.alternatives else "(none)"
        return (
            "Action packet:\n"
            f"- what: {action.what}\n"
            f"- why: {action.why}\n"
            f"- impact: {action.impact}\n"
            f"- rollback: {action.rollback}\n"
            f"- cost: {action.cost}\n"
            f"- confidence: {action.confidence:.2f}\n"
            f"- alternatives: {alternatives}"
        )

    def _normalize_tool_intent(self, tool_name: str, args: dict[str, Any]) -> str:
        normalized_args = normalize_tool_arguments(args)
        payload = json.dumps(
            {"tool": tool_name, "args": normalized_args},
            sort_keys=True,
            separators=(",", ":"),
        )
        return payload

    def _prune_intent_ledger(self, *, now: float | None = None) -> None:
        now_ts = time.monotonic() if now is None else float(now)
        ttl_s = max(0.0, float(getattr(self, "_intent_state_ttl_s", 300.0)))
        expired = [
            key
            for key, entry in self._intent_ledger.items()
            if now_ts - float(entry.updated_at) > ttl_s
        ]
        for key in expired:
            self._intent_ledger.pop(key, None)

    def _intent_ledger_key(self, tool_name: str, args: dict[str, Any]) -> str:
        normalized = self._normalize_tool_intent(tool_name, args)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _idempotency_key_for_action(self, action: ActionPacket | None) -> str | None:
        if action is None:
            return None
        token = getattr(self, "_pending_confirmation_token", None)
        pending = getattr(token, "pending_action", None) if token is not None else None
        if pending is not None and getattr(pending, "action", None) is action:
            return pending.idempotency_key
        fallback_pending = getattr(self, "_pending_action", None)
        if fallback_pending is not None and getattr(fallback_pending, "action", None) is action:
            return fallback_pending.idempotency_key
        return None

    def _record_intent_state(
        self,
        tool_name: str,
        args: dict[str, Any],
        state: str,
        *,
        idempotency_key: str | None = None,
    ) -> None:
        now = time.monotonic()
        self._prune_intent_ledger(now=now)
        key = self._intent_ledger_key(tool_name, args)
        normalized = self._normalize_tool_intent(tool_name, args)
        entry = self._intent_ledger.get(key)
        if entry is None:
            entry = IntentLedgerEntry(
                tool_name=tool_name,
                normalized_intent=normalized,
                idempotency_key=idempotency_key,
                state=state,
                updated_at=now,
            )
        if idempotency_key:
            entry.idempotency_key = idempotency_key
        entry.state = state
        entry.updated_at = now
        if state == "pending":
            entry.pending_at = now
        elif state == "approved":
            entry.approved_at = now
        elif state == "denied":
            entry.denied_at = now
        elif state == "timeout":
            entry.timeout_at = now
        elif state == "failure":
            entry.failure_at = now
        elif state == "executed":
            entry.executed_at = now
        self._intent_ledger[key] = entry

    def _evaluate_intent_guard(
        self,
        tool_name: str,
        args: dict[str, Any],
        *,
        phase: str,
        idempotency_key: str | None = None,
    ) -> tuple[bool, str | None, str | None]:
        now = time.monotonic()
        self._prune_intent_ledger(now=now)
        normalized_intent = self._normalize_tool_intent(tool_name, args)
        entry = None
        normalized_idempotency_key = str(idempotency_key or "").strip()
        if normalized_idempotency_key:
            matching_entries = [
                existing
                for existing in self._intent_ledger.values()
                if str(getattr(existing, "idempotency_key", "") or "") == normalized_idempotency_key
            ]
            if matching_entries:
                entry = max(matching_entries, key=lambda existing: float(existing.updated_at))

        if entry is None:
            key = self._intent_ledger_key(tool_name, args)
            entry = self._intent_ledger.get(key)

        if entry is None:
            entry = next(
                (
                    existing
                    for existing in self._intent_ledger.values()
                    if existing.normalized_intent == normalized_intent
                ),
                None,
            )
        if entry is None:
            return False, None, None

        denial_at = entry.denied_at
        denial_cooldown_s = max(0.0, float(getattr(self, "_intent_denial_cooldown_s", 45.0)))
        if (
            phase == "confirmation_prompt"
            and isinstance(denial_at, (int, float))
            and now - float(denial_at) <= denial_cooldown_s
        ):
            logger.info(
                "INTENT_GUARD_HIT reason=blocked_recent_denial phase=%s tool=%s remaining_s=%.2f",
                phase,
                tool_name,
                max(0.0, denial_cooldown_s - (now - float(denial_at))),
            )
            return (
                True,
                "blocked_recent_denial",
                "I can't re-open that confirmation yet because this exact request was just denied. Please wait briefly or change the request.",
            )

        timeout_at = entry.timeout_at
        if (
            phase == "confirmation_prompt"
            and isinstance(timeout_at, (int, float))
            and now - float(timeout_at) <= denial_cooldown_s
        ):
            logger.info(
                "INTENT_GUARD_HIT reason=blocked_recent_timeout phase=%s tool=%s remaining_s=%.2f",
                phase,
                tool_name,
                max(0.0, denial_cooldown_s - (now - float(timeout_at))),
            )
            return (
                True,
                "blocked_recent_timeout",
                "I can't re-open that confirmation yet because this exact request just timed out. Please wait briefly or change the request.",
            )

        executed_at = entry.executed_at
        execution_cooldown_s = max(
            0.0,
            float(getattr(self, "_intent_execution_cooldown_s", getattr(self, "_tool_call_dedupe_ttl_s", 30.0))),
        )
        if (
            phase == "execution"
            and isinstance(executed_at, (int, float))
            and now - float(executed_at) <= execution_cooldown_s
        ):
            logger.info(
                "INTENT_GUARD_HIT reason=blocked_duplicate_execution phase=%s tool=%s remaining_s=%.2f",
                phase,
                tool_name,
                max(0.0, execution_cooldown_s - (now - float(executed_at))),
            )
            return True, "blocked_duplicate_execution", "Duplicate execution blocked by intent cooldown; reuse recent results."

        return False, None, None

    async def _handle_intent_guard_block(
        self,
        action: ActionPacket,
        websocket: Any,
        *,
        status: str,
        reason: str,
        message: str,
        staging: dict[str, Any] | None = None,
        include_assistant_message: bool = True,
    ) -> None:
        if include_assistant_message:
            await self.send_assistant_message(message, websocket)
        await self._send_noop_tool_output(
            websocket,
            call_id=action.id,
            status=status,
            message=message,
            tool_name=action.tool_name,
            reason=reason,
            category="suppression",
            action=action,
            staging=staging,
        )
        self._presented_actions.discard(action.id)
        self.function_call = None
        self.function_call_args = ""

    def _build_tool_call_fingerprint(self, tool_name: str, args: dict[str, Any]) -> str:
        key = build_normalized_idempotency_key(tool_name, args)
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def _record_confirmation_timeout(self, action: ActionPacket, cause: str) -> None:
        fingerprint = self._build_tool_call_fingerprint(action.tool_name, action.tool_args)
        service = getattr(self, "_confirmation_service", None)
        if service is not None:
            service.timeout_markers[fingerprint] = time.monotonic()
            service.timeout_causes[fingerprint] = cause
            return
        self._confirmation_timeout_markers[fingerprint] = time.monotonic()
        self._confirmation_timeout_causes[fingerprint] = cause

    def _record_recent_confirmation_outcome(self, idempotency_key: str | None, outcome: str) -> None:
        normalized_key = str(idempotency_key or "").strip()
        if not normalized_key:
            return
        payload = {
            "outcome": str(outcome),
            "timestamp": time.monotonic(),
        }
        service = getattr(self, "_confirmation_service", None)
        if service is not None:
            service.recent_outcomes[normalized_key] = payload
            return
        self._recent_confirmation_outcomes[normalized_key] = payload

    def _suppressed_confirmation_outcome(
        self,
        *,
        idempotency_key: str | None,
        cooldown_seconds: float,
    ) -> str | None:
        if cooldown_seconds <= 0:
            return None
        normalized_key = str(idempotency_key or "").strip()
        if not normalized_key:
            return None
        service = getattr(self, "_confirmation_service", None)
        outcomes = service.recent_outcomes if service is not None else getattr(self, "_recent_confirmation_outcomes", {})
        payload = outcomes.get(normalized_key)
        if not isinstance(payload, dict):
            return None
        outcome = str(payload.get("outcome") or "").strip()
        if outcome not in {"rejected", "expired"}:
            return None
        timestamp = payload.get("timestamp")
        if not isinstance(timestamp, (int, float)):
            return None
        if time.monotonic() - float(timestamp) <= float(cooldown_seconds):
            return outcome
        outcomes.pop(normalized_key, None)
        return None

    def _is_suppressed_after_confirmation_timeout(
        self, tool_name: str, args: dict[str, Any]
    ) -> bool:
        now = time.monotonic()
        service = getattr(self, "_confirmation_service", None)
        timeout_markers = service.timeout_markers if service is not None else self._confirmation_timeout_markers
        timeout_causes = service.timeout_causes if service is not None else self._confirmation_timeout_causes
        expired = [
            marker
            for marker, ts in timeout_markers.items()
            if now - ts > self._confirmation_timeout_debounce_window_s
        ]
        for marker in expired:
            timeout_markers.pop(marker, None)
            timeout_causes.pop(marker, None)
        fingerprint = self._build_tool_call_fingerprint(tool_name, args)
        ts = timeout_markers.get(fingerprint)
        if ts is None:
            return False
        return now - ts <= self._confirmation_timeout_debounce_window_s

    def _is_duplicate_tool_call(self, tool_name: str, args: dict[str, Any]) -> bool:
        if not self._last_executed_tool_call:
            return False
        now = time.monotonic()
        age = now - float(self._last_executed_tool_call.get("timestamp", 0.0))
        if age > self._tool_call_dedupe_ttl_s:
            return False
        fingerprint = self._build_tool_call_fingerprint(tool_name, args)
        return fingerprint == self._last_executed_tool_call.get("fingerprint")

    def _record_executed_tool_call(self, tool_name: str, args: dict[str, Any]) -> None:
        self._last_executed_tool_call = {
            "tool": tool_name,
            "fingerprint": self._build_tool_call_fingerprint(tool_name, args),
            "timestamp": time.monotonic(),
        }

    def _mark_prior_research_permission_granted(self, request: ResearchRequest) -> None:
        context_value = getattr(request, "context", None)
        context = context_value if isinstance(context_value, dict) else {}
        query = str(getattr(request, "prompt", ""))
        source = context.get("source")
        fingerprint = self._build_research_request_fingerprint(request)
        query_fingerprint = self._build_research_query_fingerprint(query)
        self._prior_research_permission_marker = {
            "granted_at": time.monotonic(),
            "prompt": request.prompt,
            "query": query,
            "source": source,
            "fingerprint": fingerprint,
            "query_fingerprint": query_fingerprint,
        }
        self._record_research_permission_outcome(fingerprint, approved=True)
        self._record_research_permission_outcome(query_fingerprint, approved=True)

    def _normalize_research_query_text(self, query: str) -> str:
        return " ".join((query or "").strip().lower().split())

    def _normalize_research_source_text(self, source: str | None) -> str:
        return str(source or "").strip().lower()

    def _build_research_fingerprint(self, *, query: str, source: str | None) -> str:
        return self._get_research_runtime().build_research_fingerprint(query=query, source=source)

    def _get_research_runtime(self) -> ResearchRuntime:
        runtime = getattr(self, "_research_runtime", None)
        if runtime is None:
            runtime = ResearchRuntime(self)
            self._research_runtime = runtime
        return runtime

    def _build_research_query_fingerprint(self, query: str) -> str:
        return self._build_research_fingerprint(query=query, source=None)

    def _build_research_request_fingerprint(self, request: ResearchRequest) -> str:
        context_value = getattr(request, "context", None)
        context = context_value if isinstance(context_value, dict) else {}
        return self._build_research_fingerprint(
            query=str(getattr(request, "prompt", "")),
            source=context.get("source"),
        )

    def _build_research_args_fingerprint(self, args: dict[str, Any]) -> str:
        context = args.get("context") if isinstance(args.get("context"), dict) else {}
        return self._build_research_fingerprint(
            query=str(args.get("query") or args.get("prompt") or ""),
            source=context.get("source"),
        )

    def _is_research_fingerprint_equivalent_for_grace_window(
        self,
        action_args: dict[str, Any],
        marker: dict[str, Any],
    ) -> bool:
        marker_query = self._normalize_research_query_text(str(marker.get("query") or marker.get("prompt") or ""))
        action_query = self._normalize_research_query_text(
            str(action_args.get("query") or action_args.get("prompt") or "")
        )
        if not marker_query or marker_query != action_query:
            return False
        marker_source = self._normalize_research_source_text(marker.get("source"))
        action_context = (
            action_args.get("context") if isinstance(action_args.get("context"), dict) else {}
        )
        action_source = self._normalize_research_source_text(action_context.get("source"))
        if marker_source and action_source:
            return marker_source == action_source
        return True

    def _prune_research_permission_outcomes(self, now: float | None = None) -> None:
        self._get_research_runtime().prune_research_permission_outcomes(now=now)

    def _record_research_permission_outcome(self, fingerprint: str, *, approved: bool) -> None:
        self._get_research_runtime().record_research_permission_outcome(fingerprint, approved=approved)

    def _get_research_permission_outcome(self, fingerprint: str) -> bool | None:
        self._prune_research_permission_outcomes()
        payload = self._research_permission_outcomes.get(fingerprint)
        if payload is None:
            return None
        return bool(payload.get("approved"))

    def _consume_prior_research_permission_marker_if_fresh(self, action: ActionPacket) -> bool:
        if action.tool_name != "perform_research":
            return False
        action_fingerprint = self._build_research_args_fingerprint(action.tool_args)
        action_query_fingerprint = self._build_research_query_fingerprint(
            str(action.tool_args.get("query") or action.tool_args.get("prompt") or "")
        )
        outcome = self._get_research_permission_outcome(action_fingerprint)
        if outcome is True:
            return True
        query_only_outcome = self._get_research_permission_outcome(action_query_fingerprint)
        if query_only_outcome is True:
            return True
        marker = getattr(self, "_prior_research_permission_marker", None)
        if marker is None:
            return False
        granted_at = marker.get("granted_at")
        if not isinstance(granted_at, (int, float)):
            self._prior_research_permission_marker = None
            return False
        if time.monotonic() - float(granted_at) > self._prior_research_permission_grace_s:
            self._prior_research_permission_marker = None
            return False
        marker_fingerprint = marker.get("fingerprint")
        if isinstance(marker_fingerprint, str) and marker_fingerprint == action_fingerprint:
            self._prior_research_permission_marker = None
            return True
        marker_query_fingerprint = marker.get("query_fingerprint")
        if (
            isinstance(marker_query_fingerprint, str)
            and marker_query_fingerprint == action_query_fingerprint
        ):
            self._prior_research_permission_marker = None
            return True
        if not self._is_research_fingerprint_equivalent_for_grace_window(action.tool_args, marker):
            return False
        self._prior_research_permission_marker = None
        return True

    def _prune_spoken_research_response_ids(self, now: float) -> None:
        if not self._spoken_research_response_ids:
            return
        ttl = self._research_spoken_response_dedupe_ttl_s
        expired = [
            research_id
            for research_id, ts in self._spoken_research_response_ids.items()
            if now - ts > ttl
        ]
        for research_id in expired:
            self._spoken_research_response_ids.pop(research_id, None)

    def _mark_or_suppress_research_spoken_response(self, research_id: str | None) -> bool:
        if not research_id:
            return False
        now = time.monotonic()
        self._prune_spoken_research_response_ids(now)
        existing_ts = self._spoken_research_response_ids.get(research_id)
        if existing_ts is not None:
            age_ms = (now - existing_ts) * 1000.0
            logger.info(
                "[Research] Suppressing duplicate spoken response (research_id=%s, age_ms=%.1f)",
                research_id,
                age_ms,
            )
            return True
        self._spoken_research_response_ids[research_id] = now
        return False

    def _extract_research_id(self, result: Any) -> str | None:
        if not isinstance(result, dict):
            return None
        research_id = result.get("research_id")
        return str(research_id) if research_id else None

    def _next_response_turn_id(self) -> str:
        self._response_create_turn_counter += 1
        return f"turn_{self._response_create_turn_counter}"

    def _resolve_response_create_turn_id(
        self,
        *,
        origin: str,
        response_create_event: dict[str, Any],
    ) -> str:
        metadata = self._extract_response_create_metadata(response_create_event)
        metadata_turn_id = metadata.get("turn_id")
        if isinstance(metadata_turn_id, str) and metadata_turn_id.strip():
            turn_id = metadata_turn_id.strip()
            with self._utterance_context_scope(turn_id=turn_id):
                return turn_id

        normalized_origin = str(origin or "unknown").strip().lower()
        if normalized_origin == "assistant_message":
            micro_ack_turn_id = metadata.get("micro_ack_turn_id")
            if isinstance(micro_ack_turn_id, str) and micro_ack_turn_id.strip():
                turn_id = micro_ack_turn_id.strip()
                with self._utterance_context_scope(turn_id=turn_id):
                    return turn_id
            if self._current_response_turn_id:
                return self._current_response_turn_id
            turn_id = self._next_response_turn_id()
            with self._utterance_context_scope(turn_id=turn_id):
                return turn_id
        if normalized_origin == "tool_output":
            if self._current_response_turn_id:
                return self._current_response_turn_id
            if self._pending_response_create is not None:
                return self._pending_response_create.turn_id
            turn_id = self._next_response_turn_id()
            with self._utterance_context_scope(turn_id=turn_id):
                return turn_id
        if self._current_response_turn_id:
            return self._current_response_turn_id
        turn_id = self._next_response_turn_id()
        with self._utterance_context_scope(turn_id=turn_id):
            return turn_id

    def _response_create_priority(self, origin: str) -> int:
        normalized_origin = str(origin or "").strip().lower()
        if normalized_origin == "upgraded_response":
            return 2
        if normalized_origin in {"assistant_message", "clarify", "server_auto", "tool_output"}:
            return 1
        return 1

    def _response_create_queue_priority(
        self,
        *,
        origin: str,
        canonical_key: str,
    ) -> int:
        if ":tool:" in str(canonical_key or ""):
            return 3
        return self._response_create_priority(origin)

    def _next_response_create_enqueue_seq(self) -> int:
        self._response_create_enqueue_seq = int(getattr(self, "_response_create_enqueue_seq", 0) or 0) + 1
        return self._response_create_enqueue_seq

    def _sync_pending_response_create_queue(self) -> None:
        queue = getattr(self, "_response_create_queue", None)
        if not isinstance(queue, deque):
            queue = deque()
            self._response_create_queue = queue
        pending = self._pending_response_create
        self._queued_confirmation_reminder_keys.clear()
        if pending is None:
            for queued in queue:
                queued_key = self._queued_response_reminder_key(queued)
                if queued_key:
                    self._queued_confirmation_reminder_keys.add(queued_key)
            self._response_create_max_qdepth = max(
                int(getattr(self, "_response_create_max_qdepth", 0) or 0),
                len(queue),
            )
            return
        queued_item: dict[str, Any] = {
            "websocket": pending.websocket,
            "event": pending.event,
            "origin": pending.origin,
            "turn_id": pending.turn_id,
            "record_ai_call": pending.record_ai_call,
            "debug_context": pending.debug_context,
            "memory_brief_note": pending.memory_brief_note,
            "enqueued_done_serial": pending.enqueued_done_serial,
            "enqueue_seq": pending.enqueue_seq,
        }
        if pending.queued_reminder_key is not None:
            queued_item["queued_reminder_key"] = pending.queued_reminder_key

        pending_metadata = self._extract_response_create_metadata(pending.event)
        pending_canonical_key = self._canonical_utterance_key(
            turn_id=pending.turn_id,
            input_event_key=str(pending_metadata.get("input_event_key") or "").strip(),
        )
        kept: deque[dict[str, Any]] = deque()
        for queued in queue:
            metadata = self._extract_response_create_metadata(queued.get("event") or {})
            queued_canonical = self._canonical_utterance_key(
                turn_id=str(queued.get("turn_id") or "").strip(),
                input_event_key=str(metadata.get("input_event_key") or "").strip(),
            )
            if queued_canonical == pending_canonical_key:
                continue
            kept.append(queued)
        queue.clear()
        queue.appendleft(queued_item)
        queue.extend(kept)
        for queued in queue:
            queued_key = self._queued_response_reminder_key(queued)
            if queued_key:
                self._queued_confirmation_reminder_keys.add(queued_key)
        self._response_create_max_qdepth = max(
            int(getattr(self, "_response_create_max_qdepth", 0) or 0),
            len(queue),
        )

    def _schedule_pending_response_create(
        self,
        *,
        websocket: Any,
        response_create_event: dict[str, Any],
        origin: str,
        reason: str,
        record_ai_call: bool,
        debug_context: dict[str, Any] | None,
        memory_brief_note: str | None,
    ) -> bool:
        return self._response_create_runtime.schedule_pending_response_create(
            websocket=websocket,
            response_create_event=response_create_event,
            origin=origin,
            reason=reason,
            record_ai_call=record_ai_call,
            debug_context=debug_context,
            memory_brief_note=memory_brief_note,
        )

    async def _send_response_create(
        self,
        websocket: Any,
        response_create_event: dict[str, Any],
        *,
        origin: str,
        utterance_context: UtteranceContext | None = None,
        record_ai_call: bool = False,
        debug_context: dict[str, Any] | None = None,
        memory_brief_note: str | None = None,
    ) -> bool:
        return await self._response_create_runtime.send_response_create(
            websocket,
            response_create_event,
            origin=origin,
            utterance_context=utterance_context,
            record_ai_call=record_ai_call,
            debug_context=debug_context,
            memory_brief_note=memory_brief_note,
        )

    async def _drain_response_create_queue(self, source_trigger: str | None = None) -> None:
        await self._response_create_runtime.drain_response_create_queue(source_trigger=source_trigger)


    def _extract_confirmation_reminder_dedupe_key(
        self,
        response_create_event: dict[str, Any],
    ) -> str | None:
        metadata = self._extract_response_create_metadata(response_create_event)
        if self._extract_response_create_trigger(metadata) != "confirmation_reminder":
            return None
        confirmation_token = str(metadata.get("confirmation_token") or "").strip()
        if confirmation_token:
            return f"token:{confirmation_token}"
        idempotency_key = str(
            metadata.get("confirmation_idempotency_key")
            or metadata.get("idempotency_key")
            or ""
        ).strip()
        if idempotency_key:
            return f"idempotency:{idempotency_key}"
        return None

    def _queued_response_reminder_key(self, queued: dict[str, Any]) -> str | None:
        queued_key = queued.get("queued_reminder_key")
        if isinstance(queued_key, str) and queued_key.strip():
            return queued_key.strip()
        queued_event = queued.get("event")
        if isinstance(queued_event, dict):
            return self._extract_confirmation_reminder_dedupe_key(queued_event)
        return None

    def _dedupe_queued_confirmation_reminders(self) -> None:
        if len(self._response_create_queue) <= 1:
            return
        queued_items = list(self._response_create_queue)
        newest_index_by_key: dict[str, int] = {}
        for index, queued in enumerate(queued_items):
            reminder_key = self._queued_response_reminder_key(queued)
            if reminder_key is not None:
                newest_index_by_key[reminder_key] = index
        if not newest_index_by_key:
            return
        deduped_queue: deque[dict[str, Any]] = deque()
        dropped_count = 0
        for index, queued in enumerate(queued_items):
            reminder_key = self._queued_response_reminder_key(queued)
            if reminder_key is not None and newest_index_by_key.get(reminder_key) != index:
                dropped_count += 1
                logger.info(
                    "Dropping duplicate queued confirmation reminder response.create dedupe_key=%s origin=%s (newer entry kept).",
                    reminder_key,
                    queued.get("origin"),
                )
                continue
            deduped_queue.append(queued)
        if dropped_count <= 0:
            return
        self._response_create_queue = deduped_queue
        self._queued_confirmation_reminder_keys = {
            key
            for key in (self._queued_response_reminder_key(item) for item in self._response_create_queue)
            if key is not None
        }

    def _is_stale_queued_response_create(self, queued: dict[str, Any]) -> bool:
        origin = str(queued.get("origin") or "").strip().lower()
        if origin not in {"tool_output", "assistant_message"}:
            return False
        enqueued_serial = queued.get("enqueued_done_serial")
        if not isinstance(enqueued_serial, int):
            return False
        completed_distance = self._response_done_serial - enqueued_serial
        if origin == "assistant_message":
            if completed_distance < 1:
                return False
            response_metadata = self._extract_response_create_metadata(queued.get("event") or {})
            approval_flow = str(response_metadata.get("approval_flow", "")).strip().lower()
            if approval_flow in {"true", "1", "yes"} and self._has_active_confirmation_token():
                return False
            if self._has_active_confirmation_token() and self._is_token_approval_flow_metadata(response_metadata):
                return False
            return True
        if completed_distance < 2:
            return False
        # If two or more responses have already completed since this tool follow-up
        # was queued, it is stale and tends to replay old tool answers.
        return True

    def _extract_response_create_metadata(self, response_create_event: dict[str, Any]) -> dict[str, Any]:
        response_payload = response_create_event.get("response") if isinstance(response_create_event, dict) else None
        metadata = response_payload.get("metadata") if isinstance(response_payload, dict) else None
        if not isinstance(metadata, dict):
            return {}
        return metadata

    def _canonical_response_create_origin(
        self,
        *,
        origin: str | None,
        response_metadata: dict[str, Any] | None = None,
    ) -> str:
        normalized_origin = str(origin or "unknown").strip().lower() or "unknown"
        metadata = response_metadata if isinstance(response_metadata, dict) else {}
        is_micro_ack = str(metadata.get("micro_ack", "")).strip().lower() in {"1", "true", "yes"}
        if is_micro_ack:
            return "micro_ack"
        return normalized_origin

    def _extract_response_create_trigger(self, metadata: dict[str, Any]) -> str:
        trigger = metadata.get("trigger") if isinstance(metadata, dict) else None
        if isinstance(trigger, str) and trigger.strip():
            return trigger.strip().lower()
        return "unknown"

    def _is_preference_recall_lock_blocked(
        self,
        *,
        turn_id: str,
        input_event_key: str,
        normalized_origin: str,
        response_metadata: dict[str, Any],
    ) -> bool:
        if normalized_origin == "preference_recall":
            return False
        trigger = self._extract_response_create_trigger(response_metadata)
        if trigger == "preference_recall":
            return False
        locked_input_event_keys = getattr(self, "_preference_recall_locked_input_event_keys", set())
        if input_event_key and input_event_key in locked_input_event_keys:
            logger.debug(
                "preference_recall_default_response_blocked run_id=%s turn_id=%s input_event_key=%s reason=locked",
                self._current_run_id() or "",
                turn_id,
                input_event_key,
            )
            return True
        return False

    def _confirmation_hold_components(self) -> tuple[bool, bool, bool, bool]:
        token = getattr(self, "_pending_confirmation_token", None)
        phase = getattr(self.orchestration_state, "phase", None)
        token_active = token is not None or getattr(self, "_pending_action", None) is not None
        awaiting_phase = token is not None or phase == OrchestrationPhase.AWAITING_CONFIRMATION
        hold_active = phase == OrchestrationPhase.AWAITING_CONFIRMATION or token_active or awaiting_phase
        return token_active, awaiting_phase, hold_active, phase == OrchestrationPhase.AWAITING_CONFIRMATION

    def _build_confirmation_transition_decision(
        self,
        *,
        reason: str,
        include_reminder_gate: bool = False,
        was_confirmation_guarded: bool = False,
    ) -> ConfirmationTransitionDecision:
        return self._confirmation_runtime.build_confirmation_transition_decision(
            reason=reason,
            include_reminder_gate=include_reminder_gate,
            was_confirmation_guarded=was_confirmation_guarded,
        )

    def _is_awaiting_confirmation_phase(self) -> bool:
        _, awaiting_phase, _, _ = self._confirmation_hold_components()
        return awaiting_phase

    def _is_user_confirmation_trigger(self, trigger: str, metadata: dict[str, Any]) -> bool:
        return self._lifecycle_state_coordinator().is_user_confirmation_trigger(trigger, metadata)

    def _can_release_queued_response_create(self, trigger: str, metadata: dict[str, Any]) -> bool:
        return self._lifecycle_state_coordinator().can_release_queued_response_create(
            has_active_confirmation_token=self._has_active_confirmation_token(),
            is_awaiting_confirmation_phase=self._is_awaiting_confirmation_phase(),
            trigger=trigger,
            metadata=metadata,
        )

    def _drop_response_create_for_terminal_state(
        self,
        *,
        turn_id: str,
        input_event_key: str,
        origin: str,
        response_metadata: dict[str, Any] | None = None,
    ) -> bool:
        normalized_input_event_key = str(input_event_key or "").strip()
        retry_reason = ""
        if isinstance(response_metadata, dict):
            retry_reason = str(response_metadata.get("retry_reason") or "").strip().lower()

        is_empty_retry = retry_reason == "empty_response_done" or normalized_input_event_key.endswith("__empty_retry")
        if is_empty_retry and self._turn_has_final_deliverable(turn_id=turn_id):
            logger.info(
                "response_dropped_terminal_state run_id=%s turn_id=%s canonical_key=%s prior_state=%s",
                self._current_run_id() or "",
                turn_id,
                self._canonical_utterance_key(turn_id=turn_id, input_event_key=str(input_event_key or "").strip()),
                "turn_final_deliverable_empty_retry",
            )
            self._mark_transcript_response_outcome(
                input_event_key=str(input_event_key or "").strip(),
                turn_id=turn_id,
                outcome="response_not_scheduled",
                reason="canonical_delivery_terminal_state",
                details=(
                    "canonical delivery terminal state "
                    f"origin={str(origin or '').strip().lower() or 'unknown'} "
                    "prior_state=turn_final_deliverable_empty_retry "
                    f"canonical_key={self._canonical_utterance_key(turn_id=turn_id, input_event_key=str(input_event_key or '').strip())}"
                ),
            )
            return True

        terminal_state_canonical_key = self._canonical_utterance_key(
            turn_id=turn_id,
            input_event_key=normalized_input_event_key,
        )
        origin_canonical_key = terminal_state_canonical_key
        if retry_reason == "empty_response_done":
            origin_canonical_key = self._canonical_key_for_empty_retry_origin(
                turn_id=turn_id,
                input_event_key=normalized_input_event_key,
            )

        delivery_state = self._response_delivery_state(
            turn_id=turn_id,
            input_event_key=normalized_input_event_key,
        )
        if delivery_state == "blocked_empty_transcript":
            logger.info(
                "response_dropped_terminal_state run_id=%s turn_id=%s canonical_key=%s prior_state=%s",
                self._current_run_id() or "",
                turn_id,
                terminal_state_canonical_key,
                delivery_state,
            )
            self._mark_transcript_response_outcome(
                input_event_key=normalized_input_event_key,
                turn_id=turn_id,
                outcome="response_not_scheduled",
                reason="empty_transcript_blocked",
                details=(
                    "canonical delivery terminal state "
                    f"origin={str(origin or '').strip().lower() or 'unknown'} "
                    f"prior_state={delivery_state} "
                    f"canonical_key={terminal_state_canonical_key} "
                    f"origin_canonical_key={origin_canonical_key}"
                ),
            )
            return True

        prior_state = self._lifecycle_controller().state_for(terminal_state_canonical_key)
        if prior_state not in {
            InteractionLifecycleState.DONE,
            InteractionLifecycleState.REPLACED,
            InteractionLifecycleState.CANCELLED,
        }:
            return False
        prior_state_value = prior_state.value
        logger.info(
            "response_dropped_terminal_state run_id=%s turn_id=%s canonical_key=%s prior_state=%s",
            self._current_run_id() or "",
            turn_id,
            terminal_state_canonical_key,
            prior_state_value,
        )
        self._mark_transcript_response_outcome(
            input_event_key=normalized_input_event_key,
            turn_id=turn_id,
            outcome="response_not_scheduled",
            reason="canonical_delivery_terminal_state",
            details=(
                "canonical delivery terminal state "
                f"origin={str(origin or '').strip().lower() or 'unknown'} "
                f"prior_state={prior_state_value} "
                f"canonical_key={terminal_state_canonical_key} "
                f"origin_canonical_key={origin_canonical_key}"
            ),
        )
        return True

    def _is_user_approved_interrupt_response(self, response_payload: dict[str, Any]) -> bool:
        metadata = response_payload.get("metadata") if isinstance(response_payload, dict) else None
        if not isinstance(metadata, dict):
            return False
        value = metadata.get("user_approved_interrupt")
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes"}
        return False

    def _is_truthy_metadata_value(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes"}
        return False

    def _has_explicit_confirmation_decision(self, metadata: dict[str, Any]) -> bool:
        decision_keys = ("confirmation_decision", "user_confirmation_decision", "decision")
        for key in decision_keys:
            decision_value = metadata.get(key)
            if isinstance(decision_value, str):
                if self._parse_confirmation_decision(decision_value) in {"yes", "no", "cancel"}:
                    return True
        if self._is_truthy_metadata_value(metadata.get("confirmation_accepted")):
            return True
        if self._is_truthy_metadata_value(metadata.get("confirmation_rejected")):
            return True
        last_user_input_text = getattr(self, "_last_user_input_text", "") or ""
        decision = self._parse_confirmation_decision(last_user_input_text)
        return decision in {"yes", "no", "cancel"}

    def _expire_confirmation_awaiting_decision_timeout(self) -> PendingConfirmationToken | None:
        return self._confirmation_runtime.expire_confirmation_awaiting_decision_timeout()

    def _log_confirmation_timeout_check(
        self,
        token: PendingConfirmationToken,
        *,
        remaining_s: float,
        pause_reason: str | None,
    ) -> None:
        now = time.monotonic()
        interval = max(0.25, float(getattr(self, "_confirmation_timeout_check_log_interval_s", 1.0)))
        token_id = token.id
        last_logged = self._confirmation_timeout_check_last_logged_at.get(token_id)
        normalized_reason = pause_reason or "none"
        prior_reason = self._confirmation_timeout_check_last_pause_reason.get(token_id)
        pause_reason_changed = prior_reason != normalized_reason
        should_log = pause_reason_changed or last_logged is None or (now - last_logged) >= interval
        if not should_log:
            return
        self._confirmation_timeout_check_last_logged_at[token_id] = now
        self._confirmation_timeout_check_last_pause_reason[token_id] = normalized_reason
        if pause_reason_changed and prior_reason is not None:
            logger.info(
                "CONFIRMATION_PAUSE_REASON_CHANGED token=%s kind=%s paused_reason=%s remaining_s=%.1f",
                token.id,
                token.kind,
                normalized_reason,
                remaining_s,
            )
            return
        logger.debug(
            "CONFIRMATION_TIMEOUT_CHECK token=%s kind=%s remaining_s=%.1f paused_reason=%s",
            token.id,
            token.kind,
            remaining_s,
            normalized_reason,
        )

    async def _maybe_handle_confirmation_decision_timeout(
        self,
        websocket: Any,
        *,
        source_event: str,
    ) -> bool:
        return await self._confirmation_runtime.maybe_handle_confirmation_decision_timeout(
            websocket,
            source_event=source_event,
        )

    def _should_guard_confirmation_response(self, origin: str, response_payload: dict[str, Any]) -> bool:
        if not self._has_active_confirmation_token() and not self._is_awaiting_confirmation_phase():
            return False
        metadata = response_payload.get("metadata") if isinstance(response_payload, dict) else None
        if not isinstance(metadata, dict):
            metadata = {}
        approval_flow = str(metadata.get("approval_flow", "")).strip().lower()
        if approval_flow in {"true", "1", "yes"}:
            return False
        if self._has_explicit_confirmation_decision(metadata):
            return False
        normalized_origin = str(origin).strip().lower()
        return normalized_origin in {"", "unknown", "server_auto", "assistant_message", "tool_output"}

    def _recover_confirmation_guard_microphone(self, trigger: str) -> None:
        transition = self._build_confirmation_transition_decision(
            reason=f"recover_mic:{trigger}",
            was_confirmation_guarded=True,
        )
        if not transition.recover_mic or self._audio_playback_busy:
            return
        mic = getattr(self, "mic", None)
        if mic is None or getattr(mic, "is_recording", False):
            return
        token = getattr(self, "_pending_confirmation_token", None)
        pending_tool = getattr(getattr(token, "pending_action", None), "action", None)
        pending_tool = getattr(pending_tool, "tool_name", "none")
        logger.info(
            "CONFIRMATION_GUARD_RECOVERY_MIC_RESTART phase=%s pending_tool=%s trigger=%s",
            getattr(self.orchestration_state, "phase", None),
            pending_tool,
            trigger,
        )
        mic.start_recording()

    async def _add_no_tools_follow_up_instruction(self, websocket: Any) -> None:
        instruction_event = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Research already completed; answer using the tool results; do not call perform_research again.",
                    }
                ],
            },
        }
        log_ws_event("Outgoing", instruction_event)
        self._track_outgoing_event(instruction_event)
        transport = self._get_or_create_transport()
        await transport.send_json(websocket, instruction_event)

    def _log_confirmation_transition(
        self,
        from_state: ConfirmationState,
        to_state: ConfirmationState,
        *,
        reason: str,
        token: PendingConfirmationToken | None = None,
    ) -> None:
        logger.info(
            "CONFIRMATION_FSM transition=%s->%s token=%s kind=%s reason=%s",
            from_state.value,
            to_state.value,
            token.id if token else "none",
            token.kind if token else "none",
            reason,
        )

    def _set_confirmation_state(self, state: ConfirmationState, *, reason: str) -> None:
        self._confirmation_runtime.set_confirmation_state(state, reason=reason)

    def _get_confirmation_timeout_s(self, token: PendingConfirmationToken | None) -> float:
        return self._confirmation_runtime.get_confirmation_timeout_s(token)

    def _confirmation_pause_reason(self) -> str | None:
        return self._confirmation_runtime.confirmation_pause_reason()

    def _is_confirmation_ttl_paused(self) -> bool:
        return self._confirmation_runtime.is_confirmation_ttl_paused()

    def _approval_expired_for_action(
        self,
        action: ActionPacket,
        *,
        now: float,
        decision: str,
    ) -> bool:
        if action.expiry_ts is None:
            return False
        if self._is_confirmation_ttl_paused():
            return False
        expiry_ts = float(action.expiry_ts)
        if now <= expiry_ts:
            return False
        grace_s = max(0.0, float(getattr(self, "_confirmation_decision_expiry_grace_s", 1.5)))
        if decision in {"yes", "no", "cancel"} and (now - expiry_ts) <= grace_s:
            return False
        return True

    async def _confirmation_transition_guard(self) -> asyncio.Lock:
        return await self._confirmation_runtime.confirmation_transition_guard()

    def _refresh_confirmation_pause(self) -> None:
        self._confirmation_runtime.refresh_confirmation_pause()

    def _mark_confirmation_activity(self, *, reason: str, now: float | None = None) -> None:
        self._confirmation_runtime.mark_confirmation_activity(reason=reason, now=now)

    def _confirmation_effective_elapsed_s(self, now: float | None = None) -> float:
        return self._confirmation_runtime.confirmation_effective_elapsed_s(now=now)

    def _confirmation_remaining_seconds(self, *, now: float | None = None) -> float:
        return self._confirmation_runtime.confirmation_remaining_seconds(now=now)

    def _create_confirmation_token(
        self,
        *,
        kind: str,
        tool_name: str | None,
        request: ResearchRequest | None = None,
        pending_action: PendingAction | None = None,
        expiry_ts: float | None = None,
        max_retries: int = 2,
        metadata: dict[str, Any] | None = None,
    ) -> PendingConfirmationToken:
        return self._confirmation_runtime.create_confirmation_token(
            token_cls=PendingConfirmationToken,
            kind=kind,
            tool_name=tool_name,
            request=request,
            pending_action=pending_action,
            expiry_ts=expiry_ts,
            max_retries=max_retries,
            metadata=metadata,
        )

    def _sync_confirmation_legacy_fields(self) -> None:
        self._confirmation_runtime.sync_confirmation_legacy_fields()

    def _close_confirmation_token(self, *, outcome: str) -> None:
        self._confirmation_runtime.close_confirmation_token(outcome=outcome)

    def _clear_pending_action(self) -> None:
        token = getattr(self, "_pending_confirmation_token", None)
        if token and token.pending_action is not None:
            token.pending_action = None
            if token.kind == "tool_governance":
                self._close_confirmation_token(outcome="cleared_pending_action")
                return
        if self._pending_action:
            self._presented_actions.discard(self._pending_action.action.id)
        self._pending_action = None

    def _has_active_confirmation_token(self) -> bool:
        token_active, _, _, _ = self._confirmation_hold_components()
        return token_active

    def _confirmation_reminder_key(self, token: PendingConfirmationToken | None) -> str | None:
        if token is not None:
            if token.pending_action is not None and token.pending_action.idempotency_key:
                return str(token.pending_action.idempotency_key)
            if token.kind in {"research_permission", "research_budget"} and token.request is not None:
                return f"{token.kind}:{self._build_research_request_fingerprint(token.request)}"
            return f"{token.kind}:{token.id}"
        pending_action = getattr(self, "_pending_action", None)
        if pending_action is not None:
            action_packet = getattr(pending_action, "action", None)
            action_id = getattr(action_packet, "id", None)
            return f"legacy:{action_id or id(pending_action)}"
        return None

    def _confirmation_prompt_latch_key(self, token: PendingConfirmationToken | None) -> str | None:
        if token is None:
            return None
        idempotency_key = str(
            getattr(getattr(token, "pending_action", None), "idempotency_key", "") or ""
        ).strip()
        if idempotency_key:
            return f"idempotency:{idempotency_key}"
        return f"token:{token.id}"

    def _is_confirmation_prompt_latched(self, token: PendingConfirmationToken | None) -> bool:
        latch_key = self._confirmation_prompt_latch_key(token)
        if latch_key is None:
            return False
        return latch_key in self._pending_confirmation_prompt_latches

    def _clear_queued_confirmation_reminder_markers(
        self,
        token: PendingConfirmationToken | None,
    ) -> None:
        if token is None:
            return
        reminder_keys = getattr(self, "_queued_confirmation_reminder_keys", None)
        if not isinstance(reminder_keys, set):
            reminder_keys = set()
            self._queued_confirmation_reminder_keys = reminder_keys
        reminder_keys.discard(f"token:{token.id}")
        idempotency_key = str(
            getattr(getattr(token, "pending_action", None), "idempotency_key", "") or ""
        ).strip()
        if idempotency_key:
            reminder_keys.discard(f"idempotency:{idempotency_key}")

    def _confirmation_reminder_schedule(
        self,
        token: PendingConfirmationToken | None,
    ) -> tuple[int, tuple[float, ...], float]:
        metadata = token.metadata if token is not None and isinstance(token.metadata, dict) else {}
        is_tool_governance = token is not None and str(token.kind or "") == "tool_governance"
        configured_max = self._coerce_optional_int(metadata.get("max_reminders"))
        if configured_max is not None:
            max_reminders = max(0, configured_max)
        elif is_tool_governance:
            max_reminders = 1
        else:
            max_reminders = int(self._confirmation_reminder_max_count)
        configured_schedule = self._coerce_schedule_seconds(metadata.get("reminder_schedule_seconds"))
        schedule = configured_schedule
        if not schedule and is_tool_governance:
            schedule = (8.0,)
        min_interval_s = max(0.0, float(self._confirmation_reminder_interval_s))
        if max_reminders > 0 and not schedule:
            schedule = tuple(min_interval_s * index for index in range(max_reminders))
        if max_reminders <= 0:
            return 0, (), min_interval_s
        return max_reminders, schedule[:max_reminders], min_interval_s

    def _evaluate_confirmation_reminder(
        self,
        token: PendingConfirmationToken | None,
        *,
        reason: str,
    ) -> tuple[bool, str | None, int, float | None, str | None, float | None]:
        key = self._confirmation_reminder_key(token)
        now = time.monotonic()
        if key is None:
            logger.info(
                "CONFIRMATION_REMINDER_SUPPRESSED reason=%s suppress_reason=no_confirmation_key",
                reason,
            )
            return False, None, 0, None, "no_confirmation_key", None
        _max_reminders, schedule, _min_interval_s = self._confirmation_reminder_schedule(token)
        coordinator = getattr(self, "_confirmation_coordinator", None)
        if coordinator is not None:
            coordinator.pending_token = token
            coordinator.token_created_at = (
                float(token.created_at) if token is not None and isinstance(token.created_at, (int, float)) else now
            )
            coordinator.reminder_tracker = getattr(self, "_confirmation_reminder_tracker", {})
            decision = coordinator.evaluate_reminder(key=key, schedule=schedule, now=now)
            self._confirmation_reminder_tracker = coordinator.reminder_tracker
            return (
                decision.allowed,
                decision.key,
                decision.sent_count,
                decision.sent_at,
                decision.suppress_reason,
                decision.now,
            )
        max_reminders, _schedule, min_interval_s = self._confirmation_reminder_schedule(token)
        if max_reminders <= 0:
            return False, key, 0, None, "max_count", now
        reminder_tracker = getattr(self, "_confirmation_reminder_tracker", None)
        if not isinstance(reminder_tracker, dict):
            reminder_tracker = {}
            self._confirmation_reminder_tracker = reminder_tracker
        entry = reminder_tracker.get(key)
        sent_count = int(entry.get("count", 0)) if isinstance(entry, dict) else 0
        last_sent_at = float(entry.get("last_sent_at", 0.0)) if isinstance(entry, dict) else None
        token_created_at = (
            float(token.created_at)
            if token is not None and isinstance(token.created_at, (int, float))
            else now
        )
        elapsed_s = max(0.0, now - token_created_at)
        if sent_count >= max_reminders:
            return False, key, sent_count, last_sent_at, "max_count", now
        if sent_count < len(schedule) and elapsed_s < float(schedule[sent_count]):
            return False, key, sent_count, last_sent_at, "schedule", now
        if last_sent_at is not None and (now - last_sent_at) < min_interval_s:
            return False, key, sent_count, last_sent_at, "interval", now
        return True, key, sent_count + 1, now, None, now

    def _allow_confirmation_reminder(
        self,
        token: PendingConfirmationToken | None,
        *,
        reason: str,
    ) -> tuple[bool, str | None, int, float | None, str | None]:
        allowed, key, sent_count, sent_at, suppress_reason, now = self._evaluate_confirmation_reminder(
            token,
            reason=reason,
        )
        if not allowed:
            return False, key, sent_count, sent_at, suppress_reason
        coordinator = getattr(self, "_confirmation_coordinator", None)
        if coordinator is not None:
            coordinator.pending_token = token
            coordinator.reminder_tracker = getattr(self, "_confirmation_reminder_tracker", {})
            coordinator.mark_reminder_sent(
                ConfirmationReminderDecision(
                    allowed=True,
                    key=key,
                    sent_count=sent_count,
                    sent_at=now,
                    suppress_reason=None,
                    now=now,
                ),
                reason=reason,
            )
            entry = coordinator.reminder_tracker.get(key or "") if key is not None else None
            if isinstance(entry, dict):
                entry["last_run_id"] = self._current_run_id()
            self._confirmation_reminder_tracker = coordinator.reminder_tracker
        return True, key, sent_count, now, None

    def _should_send_response_done_fallback_reminder(self) -> bool:
        token = getattr(self, "_pending_confirmation_token", None)
        if token is None:
            return False
        allowed, key, _sent_count, _sent_at, suppress_reason, now = self._evaluate_confirmation_reminder(
            token,
            reason="response_done_fallback",
        )
        if not allowed:
            return False
        reminder_tracker = getattr(self, "_confirmation_reminder_tracker", None)
        if not isinstance(reminder_tracker, dict):
            reminder_tracker = {}
            self._confirmation_reminder_tracker = reminder_tracker
        entry = reminder_tracker.get(key) if key is not None else None
        if isinstance(entry, dict):
            last_sent_at = entry.get("last_sent_at")
            if isinstance(last_sent_at, (int, float)) and now is not None and (now - float(last_sent_at)) < 0.25:
                logger.info(
                    "CONFIRMATION_REMINDER_SUPPRESSED reason=response_done_fallback suppress_reason=duplicate_burst key=%s",
                    key or "none",
                )
                return False
        return suppress_reason is None

    def _is_guarded_server_auto_reminder_allowed(self, *, reason: str) -> bool:
        normalized_origin = str(getattr(self, "_active_response_origin", "") or "").strip().lower()
        if normalized_origin != "server_auto":
            return True
        token = getattr(self, "_pending_confirmation_token", None)
        allowed, key, sent_count, _sent_at, suppress_reason, _now = self._evaluate_confirmation_reminder(
            token,
            reason=reason,
        )
        if not allowed:
            logger.info(
                "CONFIRMATION_GUARDED_COMPLETION decision=server_auto_suppressed origin=%s reason=%s suppress_reason=%s key=%s sent_count=%s",
                normalized_origin,
                reason,
                suppress_reason or "unknown",
                key or "none",
                sent_count,
            )
            return False
        logger.info(
            "CONFIRMATION_GUARDED_COMPLETION decision=reminder_allowed origin=%s reason=%s key=%s next_count=%s",
            normalized_origin,
            reason,
            key or "none",
            sent_count,
        )
        return True

    def _is_token_approval_flow_metadata(self, metadata: dict[str, Any]) -> bool:
        approval_flow = str(metadata.get("approval_flow", "")).strip().lower()
        if approval_flow in {"true", "1", "yes"}:
            return True
        token = getattr(self, "_pending_confirmation_token", None)
        if token is None:
            return False
        token_id = str(metadata.get("confirmation_token", "")).strip()
        return token_id != "" and token_id == token.id

    def _classify_pending_confirmation_intent(
        self,
        transcript: str,
        *,
        pending_action: PendingAction | None,
    ) -> tuple[str | None, str | None]:
        if pending_action is None:
            return None, None
        normalized_text = str(transcript or "").strip()
        if not normalized_text:
            return None, None
        pending_packet = getattr(pending_action, "action", None)
        pending_tool = str(getattr(pending_packet, "tool_name", "") or "")
        if not pending_tool:
            return None, None
        pending_args = getattr(pending_packet, "tool_args", {})
        pending_intent = self._normalize_tool_intent(pending_tool, pending_args)
        pending_idempotency_key = str(
            pending_action.idempotency_key or build_normalized_idempotency_key(pending_tool, pending_args)
        ).strip()

        incoming_tool = "perform_research" if has_research_intent(normalized_text) else ""
        if not incoming_tool:
            return "intent_switched", None
        incoming_args = {"query": normalized_text}
        incoming_intent = self._normalize_tool_intent(incoming_tool, incoming_args)
        incoming_idempotency_key = build_normalized_idempotency_key(incoming_tool, incoming_args)
        if (
            incoming_intent == pending_intent
            or (pending_idempotency_key and incoming_idempotency_key == pending_idempotency_key)
        ):
            return "intent_match", incoming_idempotency_key
        return "intent_switched", incoming_idempotency_key

    async def _maybe_emit_confirmation_reminder(
        self,
        websocket: Any,
        *,
        reason: str,
        parsed_intent_category: str | None = None,
        parsed_intent_idempotency_key: str | None = None,
    ) -> None:
        token = getattr(self, "_pending_confirmation_token", None)
        transition = self._build_confirmation_transition_decision(
            reason=f"reminder:{reason}",
            include_reminder_gate=True,
        )
        if not transition.emit_reminder:
            token_active, awaiting_phase, _, _ = self._confirmation_hold_components()
            logger.info(
                "CONFIRMATION_REMINDER_SUPPRESSED reason=%s suppress_reason=inactive_confirmation_context token_active=%s awaiting_phase=%s",
                reason,
                token_active,
                awaiting_phase,
            )
            return
        if token is not None and not self._is_confirmation_prompt_latched(token):
            logger.info(
                "CONFIRMATION_REMINDER_SUPPRESSED reason=%s suppress_reason=prompt_not_sent token=%s",
                reason,
                token.id,
            )
            return

        pending_action = getattr(token, "pending_action", None) if token is not None else self._pending_action
        pending_idempotency_key = str(getattr(pending_action, "idempotency_key", "") or "").strip()
        incoming_idempotency_key = str(parsed_intent_idempotency_key or "").strip()
        if parsed_intent_category == "intent_switched":
            logger.info(
                "CONFIRMATION_REMINDER_SUPPRESSED reason=%s suppress_reason=intent_switched pending_idempotency_key=%s incoming_idempotency_key=%s",
                reason,
                pending_idempotency_key or "none",
                incoming_idempotency_key or "none",
            )
            return

        allowed, key, next_sent_count, sent_at, suppress_reason = self._allow_confirmation_reminder(
            token,
            reason=reason,
        )
        if not allowed:
            logger.info(
                "CONFIRMATION_REMINDER_SUPPRESSED reason=%s suppress_reason=%s key=%s sent_count=%s",
                reason,
                suppress_reason or "unknown",
                key or "none",
                next_sent_count,
            )
            return
        if token is not None:
            token.reminder_sent = True
        if self._confirmation_state == ConfirmationState.PENDING_PROMPT:
            self._set_confirmation_state(ConfirmationState.AWAITING_DECISION, reason="reminder_sent")
        logger.info(
            "CONFIRMATION_REMINDER_SENT reason=%s key=%s sent_count=%s sent_at=%.3f run_id=%s token_id=%s idempotency_key=%s",
            reason,
            key or "none",
            next_sent_count,
            sent_at or 0.0,
            self._current_run_id() or "none",
            token.id if token is not None else "none",
            getattr(getattr(token, "pending_action", None), "idempotency_key", None) or "none",
        )
        metadata = token.metadata if token is not None and isinstance(token.metadata, dict) else {}
        reminder_summary = " ".join(str(metadata.get("action_summary") or "").split()).strip()
        if not reminder_summary and token is not None and token.pending_action is not None:
            reminder_summary = " ".join(str(token.pending_action.action.summary() or "").split()).strip()
        if reminder_summary.endswith("."):
            reminder_summary = reminder_summary[:-1]
        if not reminder_summary:
            reminder_summary = "confirm or deny the pending request"
        reminder_message = "Please reply with: yes or no."
        logger.info(
            "CONFIRMATION_NON_DECISION_RESPONSE_SUPPRESSED origin=%s response_id=%s reason=%s",
            self._active_response_origin,
            self._active_response_id or "unknown",
            reason,
        )
        await self.send_assistant_message(
            reminder_message,
            websocket,
            response_metadata={
                "trigger": "confirmation_reminder",
                "approval_flow": "true",
                "confirmation_pending": "true",
                "confirmation_reason": reason,
                "confirmation_options": "yes,no",
                "confirmation_token": token.id if token is not None else "",
            },
        )

    async def _send_confirmation_reminder(self, websocket: Any, *, reason: str) -> None:
        """Backward-compatible alias for reminder emission.

        This shim keeps existing tests/callers working while routing reminder
        emissions through `_maybe_emit_confirmation_reminder`.
        """
        await self._maybe_emit_confirmation_reminder(websocket, reason=reason)

    def _parse_confirmation_decision(self, text: str) -> str:
        return normalize_confirmation_decision(text)

    def _detect_alternate_intent_while_confirmation_pending(
        self,
        text: str,
        *,
        pending_action: PendingAction,
    ) -> bool:
        pending_tool_name = str(getattr(getattr(pending_action, "action", None), "tool_name", "") or "")
        if not has_research_intent(text):
            return False
        return pending_tool_name != "perform_research"

    def _consume_pending_alternate_intent_override(
        self,
        *,
        function_name: str,
        args: dict[str, Any],
        pending_action: PendingAction,
        token: PendingConfirmationToken | None,
    ) -> bool:
        marker = getattr(self, "_pending_alternate_intent_override", None)
        if not isinstance(marker, dict):
            return False
        detected_at = marker.get("detected_at")
        if not isinstance(detected_at, (int, float)) or (time.monotonic() - float(detected_at)) > 15.0:
            self._pending_alternate_intent_override = None
            return False
        expected_token = str(marker.get("token_id") or "")
        current_token = str(getattr(token, "id", "") or "")
        if expected_token and expected_token != current_token:
            self._pending_alternate_intent_override = None
            return False
        pending_intent = self._normalize_tool_intent(
            pending_action.action.tool_name,
            getattr(pending_action.action, "tool_args", {}),
        )
        incoming_intent = self._normalize_tool_intent(function_name, args)
        if incoming_intent == pending_intent:
            return False
        self._pending_alternate_intent_override = None
        logger.info(
            "Function call outcome: bypassed pending confirmation suppression via alternate intent marker | incoming=%s pending=%s token=%s",
            function_name,
            pending_action.action.tool_name,
            current_token or "legacy",
        )
        return True

    async def _maybe_apply_late_confirmation_decision(self, text: str, websocket: Any) -> bool:
        transition = self._build_confirmation_transition_decision(reason="late_resolve")
        if not transition.allow_response_transition:
            return False
        decision = self._parse_confirmation_decision(text)
        if decision not in {"yes", "no", "cancel"}:
            return False
        service = getattr(self, "_confirmation_service", None)
        marker = service.last_closed if service is not None else getattr(self, "_confirmation_last_closed_token", None)
        if not isinstance(marker, dict):
            return False
        closed_at = marker.get("closed_at")
        if not isinstance(closed_at, (int, float)):
            return False
        grace_s = max(0.0, float(getattr(self, "_confirmation_late_decision_grace_s", 15.0)))
        age_s = max(0.0, time.monotonic() - float(closed_at))
        if age_s > grace_s:
            return False
        last_token = marker.get("token")
        if not isinstance(last_token, PendingConfirmationToken):
            return False
        if marker.get("kind") != "research_permission" or marker.get("outcome") != "awaiting_decision_timeout":
            return False
        request = last_token.request
        if request is None:
            return False
        logger.info(
            "CONFIRMATION_GRACE_DECISION_APPLIED token=%s kind=%s decision=%s age_s=%.1f grace_s=%.1f close_reason=%s",
            last_token.id,
            marker.get("kind"),
            decision,
            age_s,
            grace_s,
            "late_resolve_applied",
        )
        if decision == "yes":
            self._mark_prior_research_permission_granted(request)
            await self._dispatch_research_request(request, websocket)
            return True
        fingerprint = self._build_research_request_fingerprint(request)
        self._record_research_permission_outcome(fingerprint, approved=False)
        await self.send_assistant_message(
            "Understood — I won't perform web research right now.",
            websocket,
        )
        self.orchestration_state.transition(
            OrchestrationPhase.IDLE,
            reason="research permission rejected",
        )
        return True

    def _sample_audio_levels(self, audio_bytes: bytes) -> tuple[float | None, float | None]:
        if not audio_bytes:
            return None, None
        try:
            rms = audioop.rms(audio_bytes, 2) / 32768.0
            peak = audioop.max(audio_bytes, 2) / 32768.0
        except audioop.error:
            return None, None
        return rms, peak

    def _refresh_utterance_audio_levels(self) -> None:
        if not self._active_utterance:
            return
        start_ts = self._active_utterance.get("t_start")
        if start_ts is None:
            return
        samples = [s for s in self._recent_input_levels if s["t"] >= start_ts]
        if not samples:
            return
        rms_values = [s["rms"] for s in samples if s.get("rms") is not None]
        peak_values = [s["peak"] for s in samples if s.get("peak") is not None]
        if rms_values:
            self._active_utterance["rms_estimate"] = sum(rms_values) / len(rms_values)
        if peak_values:
            self._active_utterance["peak_estimate"] = max(peak_values)

    def _log_utterance_envelope(self, event_name: str) -> None:
        if not self._debug_vad or not self._active_utterance:
            return
        env = self._active_utterance
        logger.debug(
            "VAD_UTTERANCE event=%s id=%s t_start=%.3f t_stop=%s duration_ms=%s rms=%s peak=%s transcript=%s len=%s confirmation_candidate=%s decision=%s suppressed=%s",
            event_name,
            env.get("utterance_id"),
            env.get("t_start", 0.0),
            f"{env.get('t_stop'):.3f}" if env.get("t_stop") is not None else "n/a",
            int(env.get("duration_ms")) if env.get("duration_ms") is not None else "n/a",
            f"{env.get('rms_estimate'):.4f}" if env.get("rms_estimate") is not None else "n/a",
            f"{env.get('peak_estimate'):.4f}" if env.get("peak_estimate") is not None else "n/a",
            self._clip_text(env.get("transcript"), limit=60),
            env.get("transcript_len", 0),
            env.get("confirmation_candidate", False),
            env.get("decision", "unclear"),
            env.get("suppressed", False),
        )

    def _reset_utterance_info_summary(self) -> None:
        self._utterance_info_summary = {
            "speech_started_seen": False,
            "speech_stopped_seen": False,
            "commit_seen": False,
            "transcript_present": False,
            "asr_error_present": False,
            "response_created_seen": False,
            "response_done_seen": False,
            "deliverable_seen": False,
        }
        self._utterance_info_summary_emitted = False

    def _mark_utterance_info_summary(self, **updates: bool) -> None:
        if not isinstance(getattr(self, "_utterance_info_summary", None), dict):
            self._reset_utterance_info_summary()
        for key, value in updates.items():
            if key in self._utterance_info_summary:
                self._utterance_info_summary[key] = bool(value)

    def _emit_utterance_info_summary(self, *, anchor: str) -> None:
        if getattr(self, "_utterance_info_summary_emitted", False):
            return
        summary = dict(getattr(self, "_utterance_info_summary", {}) or {})
        if not summary:
            self._reset_utterance_info_summary()
            summary = dict(self._utterance_info_summary)
        self._utterance_info_summary_emitted = True
        logger.info(
            "UTTERANCE_INFO_SUMMARY run_id=%s anchor=%s speech_started_seen=%s speech_stopped_seen=%s "
            "commit_seen=%s transcript_present=%s asr_error_present=%s response_created_seen=%s "
            "response_done_seen=%s deliverable_seen=%s",
            self._current_run_id() or "",
            anchor,
            summary.get("speech_started_seen", False),
            summary.get("speech_stopped_seen", False),
            summary.get("commit_seen", False),
            summary.get("transcript_present", False),
            summary.get("asr_error_present", False),
            summary.get("response_created_seen", False),
            summary.get("response_done_seen", False),
            summary.get("deliverable_seen", False),
        )

    def _summary_response_created_seen_for_canonical(
        self,
        *,
        turn_id: str,
        canonical_key: str,
    ) -> bool:
        normalized_canonical_key = str(canonical_key or "").strip()
        if not normalized_canonical_key:
            return False
        active_canonical_key = str(getattr(self, "_active_response_canonical_key", "") or "").strip()
        if active_canonical_key and active_canonical_key == normalized_canonical_key:
            return True
        pending = self._pending_server_auto_response_for_turn(turn_id=turn_id)
        if isinstance(pending, PendingServerAutoResponse):
            pending_canonical_key = str(pending.canonical_key or "").strip()
            if pending_canonical_key and pending_canonical_key == normalized_canonical_key:
                return True
        return False

    def _is_noise_like_transcript(self, transcript: str) -> bool:
        normalized = " ".join(re.sub(r"[^\w\s]", " ", transcript.lower()).split())
        if not normalized:
            return True
        return normalized in {"uh", "um", "hmm", "mm", "ah", "er"}

    def _should_suppress_short_utterance(self, transcript: str | None, duration_ms: float | None) -> bool:
        duration = float(duration_ms) if duration_ms is not None else 0.0
        text = transcript or ""
        decision = self._parse_confirmation_decision(text)
        if decision in {"yes", "no", "cancel"}:
            return False
        if duration >= self._minimum_non_confirmation_duration_ms:
            return False
        return self._is_noise_like_transcript(text)

    async def _suppress_guardrail_response(self, websocket: Any, transcript: str | None) -> None:
        logger.info(
            "VAD_GUARDRAIL_SUPPRESSED min_duration_ms=%s transcript=%s",
            self._minimum_non_confirmation_duration_ms,
            self._clip_text(transcript, limit=80),
        )
        cancel_event = {"type": "response.cancel"}
        log_ws_event("Outgoing", cancel_event)
        self._track_outgoing_event(cancel_event, origin="vad_guardrail")
        transport = self._get_or_create_transport()
        await transport.send_json(websocket, cancel_event)

    async def _maybe_handle_approval_response(
        self, text: str, websocket: Any
    ) -> bool:
        return await self._confirmation_runtime.maybe_handle_approval_response(text, websocket)

    async def _maybe_handle_research_permission_response(self, text: str, websocket: Any) -> bool:
        return await self._get_research_runtime().maybe_handle_research_permission_response(text, websocket)
    async def _maybe_handle_research_budget_response(self, text: str, websocket: Any) -> bool:
        token = getattr(self, "_pending_confirmation_token", None)
        if token is None or token.kind != "research_budget":
            return False
        request = token.request
        if request is None:
            return False
        decision = self._parse_confirmation_decision(text)
        if decision in {"yes", "approve", "proceed"}:
            self._set_confirmation_state(ConfirmationState.RESOLVING, reason="research_budget_accepted")
            token_id = token.id
            self._close_confirmation_token(outcome="approved")
            approved_request = replace(
                request,
                context={**request.context, "over_budget_approved": True},
            )
            logger.info(
                "RESEARCH_DISPATCH_PREP token=%s approved_context=true",
                token_id,
            )
            if await self._dispatch_deferred_research_tool_call(websocket, token_id=token_id):
                return True
            await self._dispatch_research_request(approved_request, websocket)
            return True
        if decision in {"no", "cancel", "reject", "deny"}:
            self._set_confirmation_state(ConfirmationState.RESOLVING, reason="research_budget_rejected")
            self._close_confirmation_token(outcome="rejected")
            await self.send_assistant_message(
                "Okay — I won't run web research while budget is 0. You can raise research.budget.daily_limit in config.",
                websocket,
            )
            return True
        return True

    async def _maybe_process_research_intent(self, text: str, websocket: Any, *, source: str) -> bool:
        return await self._get_research_runtime().maybe_process_research_intent(
            text,
            websocket,
            source=source,
        )

    def _research_budget_remaining(self) -> int | None:
        service = getattr(self, "_research_service", None)
        getter = getattr(service, "get_budget_remaining", None)
        if not callable(getter):
            return None
        try:
            value = getter()
        except Exception:  # pragma: no cover - defensive fallback
            return None
        if value is None:
            return None
        return int(value)

    def _research_can_run_now(self) -> bool:
        service = getattr(self, "_research_service", None)
        checker = getattr(service, "can_run_research_now", None)
        if not callable(checker):
            return True
        try:
            return bool(checker())
        except Exception:  # pragma: no cover - defensive fallback
            return True

    def _classify_research_initiator(self, source: str | None) -> str:
        normalized = str(source or "").strip().lower()
        if normalized in {"input_audio_transcription", "text_message", "startup_prompt"}:
            return "user_initiated"
        return "assistant_initiated"

    def should_request_research_permission(
        self,
        request: ResearchRequest,
        *,
        source: str | None,
        domains_known: bool,
    ) -> tuple[bool, str]:
        mode = getattr(self, "_research_mode", "auto")
        initiator = self._classify_research_initiator(source)
        if mode == "disabled":
            return True, "research_mode_disabled"
        if mode == "ask":
            return True, "research_mode_forced_ask"
        if not domains_known:
            return True, "unknown_or_unallowlisted_domains"
        if mode == "ask_on_assistant_or_unknown" and initiator == "assistant_initiated":
            return True, "assistant_initiated"

        for candidate_url in self._extract_research_urls(request.prompt):
            parsed = urlparse(candidate_url)
            host = (parsed.hostname or "").lower().strip()
            if host and not self._is_research_domain_allowlisted(host):
                return True, "non_allowlisted_domain"
            if parsed.path.lower().endswith(".pdf"):
                return True, "pdf_ingestion"

        if self._research_provider == "openai" and self._research_firecrawl_enabled and not domains_known:
            return True, "paid_provider_firecrawl"

        return False, "allowlisted_low_risk"

    def _extract_research_urls(self, prompt: str) -> list[str]:
        return re.findall(r"https?://[^\s)\]>\"']+", str(prompt or ""), flags=re.IGNORECASE)

    def _extract_primary_research_domain(self, prompt: str) -> str | None:
        domains = self._extract_research_domains_from_prompt(prompt)
        return domains[0] if domains else None

    def _extract_research_domains_from_prompt(self, prompt: str) -> list[str]:
        domains: list[str] = []
        seen: set[str] = set()
        for candidate_url in self._extract_research_urls(prompt):
            host = normalize_trusted_domain((urlparse(candidate_url).hostname or "").lower().strip())
            if not host or host in seen:
                continue
            seen.add(host)
            domains.append(host)
        return domains

    async def _discover_research_domains(self, request: ResearchRequest) -> list[str]:
        discovered = self._extract_research_domains_from_prompt(request.prompt)
        if discovered:
            return discovered[:2]

        service = getattr(self, "_research_service", None)
        discover_fn = getattr(service, "discover_domains", None)
        if not callable(discover_fn):
            return []

        try:
            raw_domains = await asyncio.to_thread(discover_fn, request)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[Research] domain_discovery_failed error=%s", exc)
            return []

        domains: list[str] = []
        seen: set[str] = set()
        for raw_domain in raw_domains or []:
            normalized = normalize_trusted_domain(str(raw_domain))
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            domains.append(normalized)
            if len(domains) >= 2:
                break
        return domains

    def _build_research_permission_prompt(self, *, domains: list[str], domains_known: bool) -> str:
        if domains_known:
            if len(domains) >= 2:
                domain_summary = f"{domains[0]} and {domains[1]}"
            else:
                domain_summary = f"{domains[0]} (and possibly one other source)"
            return (
                "I need your permission before a web lookup due to research mode policy. "
                f"I can look this up on {domain_summary}. Proceed? (yes/no)"
            )

        return (
            "I need your permission before a web lookup due to research mode policy. "
            "I can look this up on the web, but I don't know the source domains yet—"
            "want me to preview domains first, or proceed? (preview/proceed/yes/no)"
        )

    def _is_domain_preview_request(self, text: str) -> bool:
        lowered = str(text or "").strip().lower()
        return "preview" in lowered and "domain" in lowered

    def _research_request_domains_allowlisted(self, request: ResearchRequest) -> bool:
        urls = self._extract_research_urls(request.prompt)
        if not urls:
            return True
        for candidate_url in urls:
            host = (urlparse(candidate_url).hostname or "").lower().strip()
            if host and not self._is_research_domain_allowlisted(host):
                return False
        return True

    def _is_research_domain_allowlisted(self, host: str) -> bool:
        mode = getattr(self, "_research_firecrawl_allowlist_mode", "public")
        normalized_host = normalize_trusted_domain(host)
        if not normalized_host:
            return False
        if normalized_host in getattr(self, "_trusted_research_domains", set()):
            logger.info("[Allowlist] domain_trusted domain=%s", normalized_host)
            return True
        if mode == "off":
            return True
        if mode == "explicit":
            allowlist = getattr(self, "_research_firecrawl_allowlist_domains", set())
            if not allowlist:
                return False
            return any(
                normalized_host == allowed or normalized_host.endswith(f".{allowed}")
                for allowed in allowlist
            )
        return True

    async def _dispatch_research_request(self, request: ResearchRequest, websocket: Any) -> None:
        self.orchestration_state.transition(
            OrchestrationPhase.PLAN,
            reason="research dispatch",
        )
        self._research_pending_call_ids.clear()
        logger.info("[Research] Dispatched")
        research_id = f"research_{uuid.uuid4().hex}"
        packet = None
        try:
            packet = await asyncio.to_thread(self._research_service.request_research, request)
        except Exception as exc:  # noqa: BLE001 - keep runtime resilient to provider failures
            logger.warning("[Research] Request failed: %s", exc)
        finally:
            storage = getattr(self, "_storage", None)
            if storage is not None:
                storage_info = storage.get_storage_info()
                _ = write_research_transcript(
                    run_dir=storage_info.run_dir,
                    run_id=storage_info.run_id,
                    request=request,
                    packet=packet,
                    research_id=research_id,
                )

        if packet is None:
            await self.send_assistant_message(
                "I wasn't able to complete that research request this time. Please try again.",
                websocket,
            )
            return

        realtime_payload = packet.to_realtime_payload()
        if packet.status == "disabled" or not self._research_enabled:
            await self.send_assistant_message(
                "Research is currently disabled in config, so I'll answer without web lookup.",
                websocket,
            )
            return

        grounding_explanation = build_research_grounding_explanation(packet)
        status, skip_reason, failure_name = get_content_fetch_state(packet)
        logger.info(
            "[Research] response_grounding research_id=%s content_fetch_status=%s skip_reason=%s failure=%s sources_count=%s",
            research_id,
            status,
            skip_reason,
            failure_name,
            len(packet.sources),
        )

        if requires_unverified_sources_only_response(packet):
            message = build_unverified_sources_only_response(packet)
        else:
            summary = (realtime_payload["answer_summary"] or "").strip()
            if summary:
                message = f"{summary}\n\n{grounding_explanation}"
            else:
                message = grounding_explanation

        await self.send_assistant_message(message, websocket)

    async def _dispatch_deferred_research_tool_call(self, websocket: Any, *, token_id: str) -> bool:
        pending = self._deferred_research_tool_call
        if not isinstance(pending, dict):
            logger.debug("research_dispatch_attempt outcome=skipped(reason=no_pending_tool token=%s)", token_id)
            return False
        pending_token = str(pending.get("token_id") or "")
        if pending_token != token_id:
            logger.debug(
                "research_dispatch_attempt outcome=skipped(reason=token_mismatch pending_token=%s token=%s)",
                pending_token or "none",
                token_id,
            )
            return False

        function_name = str(pending.get("tool_name") or "perform_research")
        call_id = str(pending.get("call_id") or "") or f"deferred_{uuid.uuid4().hex}"
        args = pending.get("args") if isinstance(pending.get("args"), dict) else {}
        self._deferred_research_tool_call = None
        logger.info("RESEARCH_DISPATCH token=%s call_id=%s source=deferred_tool", token_id, call_id)
        logger.debug("research_dispatch_attempt outcome=dispatched token=%s call_id=%s", token_id, call_id)
        try:
            action = self._governance.build_action_packet(
                function_name,
                call_id,
                args,
                reason="deferred_research_confirmation_approved",
            )
            staging = self._stage_action(action)
            if not staging.get("valid", True):
                await self._reject_tool_call(
                    action,
                    "Argument validation failed.",
                    websocket,
                    staging=staging,
                    status="invalid_arguments",
                )
                return True
            self.orchestration_state.transition(
                OrchestrationPhase.ACT,
                reason="deferred perform_research approved",
            )
            await self._execute_action(action, staging, websocket)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "research_dispatch_attempt outcome=error token=%s call_id=%s error=%s",
                token_id,
                call_id,
                exc,
            )
            await self.send_assistant_message(
                "I got your approval, but I couldn't dispatch the deferred research call. Please try again.",
                websocket,
            )
            return True

    def _build_response_done_prompt(self, trigger: str) -> str:
        tool_calls = self._clip_text(
            json.dumps(self._tool_call_records, ensure_ascii=False)
        )
        response_metadata = self._clip_text(
            json.dumps(
                {
                    **self._last_response_metadata,
                    "trigger": trigger,
                    "last_user_input_source": self._last_user_input_source,
                    "last_user_input_time": self._last_user_input_time,
                },
                ensure_ascii=False,
            )
        )
        return RESPONSE_DONE_REFLECTION_PROMPT.format(
            user_input=self._clip_text(self._last_user_input_text),
            assistant_reply=self._clip_text(self._assistant_reply_accum),
            tool_calls=tool_calls,
            response_metadata=response_metadata,
        )

    def _call_openai_prompt(self, prompt: str) -> str:
        endpoint = "https://api.openai.com/v1/chat/completions"
        _validate_outbound_endpoint(endpoint)
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a concise internal auditor."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 220,
        }
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            endpoint,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=30) as response:
            body = response.read().decode("utf-8")
        response_payload = json.loads(body)
        return response_payload["choices"][0]["message"]["content"].strip()

    def _parse_response_done_payload(self, raw_response: str) -> dict[str, Any] | None:
        try:
            payload = json.loads(raw_response)
        except json.JSONDecodeError:
            logger.debug("response.done reflection returned non-JSON payload.")
            return None
        if not isinstance(payload, dict):
            logger.debug("response.done reflection payload is not an object.")
            return None
        return payload

    def _is_low_signal_auto_memory_input(self) -> bool:
        text = (self._last_user_input_text or "").strip().lower()
        if len(text) < 12:
            return True
        if len(text.split()) < 3:
            return True
        low_signal_patterns = (
            "ok",
            "okay",
            "thanks",
            "thank you",
            "yes",
            "no",
            "cool",
            "nice",
            "got it",
        )
        return text in low_signal_patterns

    def _should_store_auto_memory(self, *, confidence: float, content: str) -> bool:
        if not getattr(self, "_auto_memory_enabled", False):
            logger.debug("Skipping auto-memory: disabled by config.")
            return False
        if getattr(self, "_require_confirmation_for_auto_memory", False):
            logger.info("Skipping auto-memory: require_confirmation_for_auto_memory is enabled.")
            return False
        if self._is_low_signal_auto_memory_input():
            logger.debug("Skipping auto-memory: low-signal user input.")
            return False
        if len(content.strip()) < 16:
            logger.debug("Skipping auto-memory: candidate memory content too short.")
            return False
        threshold = float(getattr(self, "_auto_memory_min_confidence", 0.75))
        if confidence < threshold:
            logger.debug(
                "Skipping auto-memory: confidence %.2f below threshold %.2f.",
                confidence,
                threshold,
            )
            return False
        return True

    async def _run_response_done_reflection(self, trigger: str) -> None:
        if not self.api_key:
            logger.debug("Skipping response.done reflection: missing API key.")
            return
        if not self._allow_ai_call("response_done_reflection"):
            logger.info("Skipping response.done reflection: AI call budget exhausted.")
            return
        prompt = self._build_response_done_prompt(trigger)
        try:
            raw_response = await asyncio.to_thread(self._call_openai_prompt, prompt)
        except Exception as exc:  # noqa: BLE001 - guard background call
            logger.warning("response.done reflection failed: %s", exc)
            return
        self._record_ai_call()
        payload = self._parse_response_done_payload(raw_response)
        if not payload:
            return
        summary = payload.get("summary")
        if isinstance(summary, str) and summary.strip():
            logger.info("response.done summary: %s", summary.strip())
        remember_payload = payload.get("remember_memory")
        if not isinstance(remember_payload, dict):
            return
        content = remember_payload.get("content")
        if not isinstance(content, str) or not content.strip():
            return
        tags = remember_payload.get("tags")
        if not isinstance(tags, list):
            tags = None
        importance = remember_payload.get("importance", 3)
        if not isinstance(importance, int):
            importance = 3
        confidence = remember_payload.get("confidence", 0.0)
        if not isinstance(confidence, (int, float)):
            confidence = 0.0
        confidence = max(0.0, min(float(confidence), 1.0))
        if not self._should_store_auto_memory(confidence=confidence, content=content):
            return
        remember_func = function_map.get("remember_memory")
        if remember_func is None:
            logger.warning("response.done reflection skipped: remember_memory unavailable.")
            return
        result = await remember_func(
            content=content.strip(),
            tags=tags,
            importance=importance,
            source="auto_reflection",
        )
        logger.info(
            "response.done remember_memory stored: memory_id=%s source=auto_reflection confidence=%.2f",
            result.get("memory_id"),
            confidence,
        )

    def _enqueue_response_done_reflection(self, trigger: str) -> None:
        if self._response_done_reflection_task and not self._response_done_reflection_task.done():
            logger.debug("response.done reflection skipped: task already running.")
            return

        async def _runner() -> None:
            try:
                await self._run_response_done_reflection(trigger)
            except Exception as exc:  # noqa: BLE001 - safety for background task
                logger.warning("response.done reflection task failed: %s", exc)

        self._response_done_reflection_task = asyncio.create_task(_runner())

    def _on_playback_complete(self) -> None:
        if self.exit_event.is_set():
            logger.info("Playback complete during shutdown -> skipping mic restart")
            return
        logger.info("Playback complete -> restarting mic")
        self._audio_playback_busy = False

        self.mic.stop_receiving()
        self.mic_send_suppress_until = time.monotonic() + 1.2

        if self.websocket:
            try:
                transport = self._get_or_create_transport()
                asyncio.create_task(
                    transport.send_json(self.websocket, {"type": "input_audio_buffer.clear"})
                )
            except Exception:
                logger.exception("Failed to send input_audio_buffer.clear")

        self.mic.start_recording()
        if self._response_create_queue:
            self._response_create_queue_drain_source = "playback_complete"
            self._runtime_task_registry().spawn(
                "response_queue_drain.playback_complete",
                self._drain_response_create_queue(),
            )
        if self._pending_image_flush_after_playback and self._pending_image_stimulus:
            self._pending_image_flush_after_playback = False
            self._schedule_pending_image_flush("playback complete")

    async def run(self) -> None:
        websockets = _require_websockets()
        _, ConnectionClosedError = _resolve_websocket_exceptions(websockets)
        self._transport = RealtimeTransport(
            connect_fn=websockets.connect,
            validate_outbound_endpoint=_validate_outbound_endpoint,
        )

        self.loop = asyncio.get_running_loop()
        self.ready_event.clear()
        self._event_injector.start()
        pending_disconnect_reason: str | None = None

        def _playback_complete_from_thread() -> None:
            if self.loop:
                self.loop.call_soon_threadsafe(self._on_playback_complete)

        self.audio_player = AudioPlayer(
            on_playback_complete=_playback_complete_from_thread,
            output_device_name=self._audio_output_device_name,
        )
        configure_websocket_library_logging()

        try:
            while True:
                try:
                    self._note_connection_attempt()
                    url = "wss://api.openai.com/v1/realtime?model=gpt-realtime"
                    headers = {"Authorization": f"Bearer {self.api_key}"}

                    async with self._transport.connect(
                        url=url,
                        headers=headers,
                        close_timeout=120,
                        ping_interval=30,
                        ping_timeout=10,
                    ) as websocket:
                        log_info("✅ Connected to the server.", style="bold green")
                        self._note_connected()
                        pending_disconnect_reason = None

                        await self.initialize_session(websocket)
                        ws_task = asyncio.create_task(self.process_ws_messages(websocket))

                        logger.info(
                            "Conversation started. Speak freely, and the assistant will respond."
                        )

                        self.websocket = websocket
                        try:
                            self.loop.add_signal_handler(signal.SIGTERM, self.shutdown_handler)
                            self.loop.add_signal_handler(signal.SIGINT, self.shutdown_handler)
                        except (NotImplementedError, RuntimeError):
                            logger.debug("Signal handlers unavailable; relying on task cancellation.")

                        if self.prompts:
                            await self.send_initial_prompts(websocket)

                        if not self.mic.is_recording:
                            self.mic.start_recording()
                            if self.prompts:
                                logger.info("Recording started after startup prompts.")
                            else:
                                logger.info("Recording started. Listening for speech...")

                        await self.send_audio_loop(websocket)

                        await ws_task

                    break
                except ConnectionClosedError as exc:
                    pending_disconnect_reason = str(exc)
                    self._note_disconnect(pending_disconnect_reason)
                    if "keepalive ping timeout" in str(exc):
                        logger.warning(
                            "WebSocket connection lost due to keepalive ping timeout. Reconnecting..."
                        )
                        self._note_reconnect()
                        await asyncio.sleep(1)
                        continue
                    logger.exception("WebSocket connection closed unexpectedly.")
                    self._note_failure(str(exc))
                    break
                except Exception as exc:
                    logger.exception("An unexpected error occurred: %s", exc)
                    pending_disconnect_reason = str(exc)
                    self._note_failure(str(exc))
                    break
                finally:
                    if self.audio_player:
                        self.audio_player.close()
                    self.mic.stop_recording()
                    self.mic.close()
                    self.websocket = None
                    self.ready_event.clear()
                    if pending_disconnect_reason:
                        self._note_disconnect(pending_disconnect_reason)
                    elif self._session_connected:
                        self._note_disconnect("clean shutdown")
        finally:
            logger.info(
                "response_create_queue_stats queued_creates_total=%s drains_total=%s max_qdepth=%s",
                int(getattr(self, "_response_create_queued_creates_total", 0) or 0),
                int(getattr(self, "_response_create_drains_total", 0) or 0),
                int(getattr(self, "_response_create_max_qdepth", 0) or 0),
            )
            self._runtime_task_registry().cancel_all("run_finally")
            await self._runtime_task_registry().await_all(timeout_s=1.0)
            self._event_injector.stop()

    async def initialize_session(self, websocket: Any) -> None:
        profile_context = self.profile_manager.get_profile_context()
        reflection_manager = ReflectionManager.get_instance()
        recent_lessons = reflection_manager.get_recent_lessons()
        lessons_block = None
        if recent_lessons:
            lesson_lines = "\n".join(f"- {lesson}" for lesson in recent_lessons)
            lessons_block = (
                "Lessons learned:\n"
                f"{lesson_lines}\n"
                "Use these lessons when planning next actions."
            )
        instructions = build_session_instructions(
            profile_context.to_instruction_block(),
            lessons_block,
        )
        logger.info(
            "Using VAD profile=%s threshold=%.2f prefix_padding_ms=%s silence_duration_ms=%s",
            self._vad_turn_detection["profile"],
            self._vad_turn_detection["threshold"],
            self._vad_turn_detection["prefix_padding_ms"],
            self._vad_turn_detection["silence_duration_ms"],
        )
        session_update = {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "model": "gpt-realtime",
                "output_modalities": ["audio"],
                "audio": {
                    "input": {
                        "transcription": {
                            "model": "gpt-4o-mini-transcribe",
                        },
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": self._vad_turn_detection["threshold"],
                            "prefix_padding_ms": self._vad_turn_detection["prefix_padding_ms"],
                            "silence_duration_ms": self._vad_turn_detection["silence_duration_ms"],
                            "create_response": self._vad_turn_detection["create_response"],
                            "interrupt_response": self._vad_turn_detection["interrupt_response"],
                        },
                    },
                    "output": {
                        "voice": "ballad",
                    },
                },
                "instructions": instructions,
                "tools": tools,
            },
        }
        log_ws_event("Outgoing", session_update)
        transport = self._get_or_create_transport()
        await transport.send_json(websocket, session_update)
        startup_digest_note = self._build_startup_memory_digest_note()
        if startup_digest_note:
            await self._send_memory_brief_note(websocket, startup_digest_note)

    def _get_or_create_transport(self) -> RealtimeTransport:
        transport = getattr(self, "_transport", None)
        if transport is not None:
            return transport
        websockets = _require_websockets()
        transport = RealtimeTransport(
            connect_fn=websockets.connect,
            validate_outbound_endpoint=_validate_outbound_endpoint,
        )
        self._transport = transport
        return transport

    async def process_ws_messages(self, websocket: Any) -> None:
        websockets = _require_websockets()
        ConnectionClosed, _ = _resolve_websocket_exceptions(websockets)
        while True:
            try:
                transport = self._transport or self._get_or_create_transport()
                event = await transport.recv_json(websocket)
                log_ws_event("Incoming", event)
                await self.handle_event(event, websocket)
            except asyncio.CancelledError:
                log_info("WebSocket receive loop cancelled.")
                self._note_disconnect("websocket loop cancelled")
                break
            except ConnectionClosed:
                log_warning("⚠️ WebSocket connection lost.")
                self._note_disconnect("websocket connection closed")
                self._response_in_flight = False
                self.response_in_progress = False
                self._active_response_id = None
                self._response_create_queue_drain_source = "websocket_close"
                await self._drain_response_create_queue(source_trigger="websocket_close")
                break

    async def _recover_from_event_handler_error(self, event_type: str, websocket: Any) -> None:
        self._active_response_confirmation_guarded = False
        self._awaiting_confirmation_completion = False

        token = getattr(self, "_pending_confirmation_token", None)
        phase = getattr(getattr(self, "orchestration_state", None), "phase", None)
        in_confirmation_or_research_flow = bool(
            (token is not None and getattr(token, "kind", "") in {"research_permission", "research_budget", "tool_governance"})
            or phase == OrchestrationPhase.AWAITING_CONFIRMATION
            or getattr(self, "_pending_research_request", None) is not None
        )

        if in_confirmation_or_research_flow:
            if token is not None:
                self._close_confirmation_token(outcome="event_handler_error")
            if phase != OrchestrationPhase.IDLE:
                self.orchestration_state.transition(
                    OrchestrationPhase.IDLE,
                    reason="event handler error recovery",
                )
            try:
                await self.send_assistant_message(
                    "Sorry—I hit an internal error while handling that confirmation/research request, so I cancelled it. Please try again.",
                    websocket,
                )
            except Exception:
                logger.exception(
                    "Failed to send confirmation/research fallback after event handler error event=%s",
                    event_type,
                )

    def _arbitrate_server_auto_response_created(
        self,
        *,
        turn_id: str,
        input_event_key: str,
        canonical_key: str,
        origin: str,
    ) -> tuple[ServerAutoArbitrationOutcome, str]:
        # Migration (phase 1): branch logic moved into InteractionLifecyclePolicy.
        normalized_origin = str(origin or "").strip().lower()
        normalized_turn_id = str(turn_id or "").strip()
        normalized_canonical_key = str(canonical_key or "").strip()
        normalized_input_event_key = str(input_event_key or "").strip()
        suppression_until = float(getattr(self, "_preference_recall_response_suppression_until", 0.0) or 0.0)
        suppressed_turns = getattr(self, "_preference_recall_suppressed_turns", None)
        if not isinstance(suppressed_turns, set):
            suppressed_turns = set()
            self._preference_recall_suppressed_turns = suppressed_turns
        suppressed_input_event_keys = getattr(self, "_preference_recall_suppressed_input_event_keys", None)
        if not isinstance(suppressed_input_event_keys, set):
            suppressed_input_event_keys = set()
            self._preference_recall_suppressed_input_event_keys = suppressed_input_event_keys

        suppression_by_input_event = bool(
            normalized_input_event_key and normalized_input_event_key in suppressed_input_event_keys
        )
        suppression_by_turn = normalized_turn_id in suppressed_turns and (
            not normalized_input_event_key or normalized_input_event_key.startswith("synthetic_server_auto_")
        )
        obligation_key = self._response_obligation_key(
            turn_id=normalized_turn_id,
            input_event_key=normalized_input_event_key,
        )
        obligation_state = self._canonical_response_state(obligation_key)
        obligation_replacement = bool(
            isinstance(obligation_state, CanonicalResponseState)
            and obligation_state.obligation_present
        )
        policy_decision = self._lifecycle_policy().decide_server_auto_created(
            normalized_origin=normalized_origin,
            has_turn_id=bool(normalized_turn_id),
            has_canonical_key=bool(normalized_canonical_key),
            suppression_by_turn=suppression_by_turn,
            suppression_window_active=suppression_until > time.monotonic(),
            suppression_by_input_event=suppression_by_input_event,
            obligation_replacement=obligation_replacement,
        )
        logger.info(
            "arbitration_decision surface=server_auto_created action=%s reason_code=%s selected_candidate_id=%s turn_id=%s canonical_key=%s",
            policy_decision.action.value,
            policy_decision.reason_code,
            policy_decision.selected_candidate_id,
            normalized_turn_id or "unknown",
            normalized_canonical_key or "unknown",
        )
        if policy_decision.action is ServerAutoCreatedDecisionAction.CANCEL_PRE_AUDIO:
            return ServerAutoArbitrationOutcome.CANCEL_PRE_AUDIO, policy_decision.reason_code
        if policy_decision.action is ServerAutoCreatedDecisionAction.DEFER:
            return ServerAutoArbitrationOutcome.DEFER, policy_decision.reason_code
        return ServerAutoArbitrationOutcome.ALLOW, policy_decision.reason_code

    async def handle_event(self, event: dict[str, Any], websocket: Any) -> None:
        event_type = str(event.get("type") or "")
        if not isinstance(getattr(self, "_event_router", None), EventRouter):
            self._event_router = EventRouter(
                fallback=self._handle_unknown_event,
                on_exception=self._on_event_handler_exception,
            )
            self._configure_event_router()
        await self._maybe_handle_confirmation_decision_timeout(
            websocket,
            source_event=event_type or "unknown",
        )
        try:
            await self._event_router.dispatch(event, websocket)
        except Exception:
            token = getattr(self, "_pending_confirmation_token", None)
            phase = getattr(getattr(self, "orchestration_state", None), "phase", None)
            logger.exception(
                "Unhandled event handler exception event=%s token_id=%s token_kind=%s phase=%s",
                event_type,
                getattr(token, "id", "none"),
                getattr(token, "kind", "none"),
                getattr(phase, "value", str(phase)),
            )
            logger.error("EVENT_HANDLER_ERROR event=%s", event_type)
            await self._recover_from_event_handler_error(event_type, websocket)

    async def _handle_response_created_event(self, event: dict[str, Any], websocket: Any) -> None:
        origin = self._consume_response_origin(event)
        self._mark_utterance_info_summary(response_created_seen=True)
        log_info(f"response.created: origin={origin}")
        response = event.get("response") or {}
        response_metadata = response.get("metadata") if isinstance(response, dict) else None
        metadata_turn_id = str(response_metadata.get("turn_id") or "").strip() if isinstance(response_metadata, dict) else ""
        metadata_input_event_key = str(response_metadata.get("input_event_key") or "").strip() if isinstance(response_metadata, dict) else ""
        metadata_upgrade_chain_id = str(response_metadata.get("upgrade_chain_id") or "").strip() if isinstance(response_metadata, dict) else ""
        pending_origin_context = getattr(self, "_last_consumed_response_origin_context", {})
        if not isinstance(pending_origin_context, dict):
            pending_origin_context = {}
        pending_turn_id = str(pending_origin_context.get("turn_id") or "").strip()
        pending_input_event_key = str(pending_origin_context.get("input_event_key") or "").strip()
        response_id = response.get("id")
        self._active_response_id = str(response_id) if response_id else None
        self._set_response_status(response_id=self._active_response_id, status="active")
        self._active_response_origin = str(origin)
        self._active_response_input_event_key = None
        self._active_response_canonical_key = None
        pending_confirmation_active = self._has_active_confirmation_token() or self._is_awaiting_confirmation_phase()
        self._active_response_preference_guarded = False
        turn_id = metadata_turn_id or pending_turn_id or self._current_turn_id_or_unknown()
        self._cancel_micro_ack(turn_id=turn_id, reason="response_created")
        if origin == "server_auto":
            expected_input_event_key = self._active_input_event_key_for_turn(turn_id)
            input_event_key = metadata_input_event_key or pending_input_event_key or None
            pending_server_auto_keys = getattr(self, "_pending_server_auto_input_event_keys", None)
            if not isinstance(pending_server_auto_keys, deque):
                pending_server_auto_keys = deque(maxlen=64)
                self._pending_server_auto_input_event_keys = pending_server_auto_keys
            while not input_event_key and pending_server_auto_keys:
                queued_key = str(pending_server_auto_keys.popleft() or "").strip()
                if not queued_key:
                    continue
                if expected_input_event_key and queued_key != expected_input_event_key:
                    logger.info(
                        "server_auto_response_mismatch_discard run_id=%s turn_id=%s active_key=%s response_key=%s",
                        self._current_run_id() or "",
                        turn_id,
                        expected_input_event_key,
                        queued_key,
                    )
                    continue
                input_event_key = queued_key
                break
            if not input_event_key:
                current_input_event_key = str(getattr(self, "_current_input_event_key", "") or "").strip()
                if expected_input_event_key and current_input_event_key and current_input_event_key != expected_input_event_key:
                    logger.info(
                        "server_auto_response_mismatch_discard run_id=%s turn_id=%s active_key=%s response_key=%s",
                        self._current_run_id() or "",
                        turn_id,
                        expected_input_event_key,
                        current_input_event_key,
                    )
                elif current_input_event_key:
                    input_event_key = current_input_event_key
            if expected_input_event_key and input_event_key and input_event_key != expected_input_event_key:
                logger.info(
                    "server_auto_response_mismatch_discard run_id=%s turn_id=%s active_key=%s response_key=%s",
                    self._current_run_id() or "",
                    turn_id,
                    expected_input_event_key,
                    input_event_key,
                )
                input_event_key = ""
            if not input_event_key and expected_input_event_key:
                logger.info(
                    "server_auto_response_stale_ignored run_id=%s turn_id=%s active_key=%s response_id=%s",
                    self._current_run_id() or "",
                    turn_id,
                    expected_input_event_key,
                    self._active_response_id or "unknown",
                )
                cancel_event = {"type": "response.cancel"}
                log_ws_event("Outgoing", cancel_event)
                self._track_outgoing_event(cancel_event, origin="server_auto_binding_guard")
                self._lifecycle_controller().on_cancel_sent(
                    self._canonical_utterance_key(turn_id=turn_id, input_event_key=expected_input_event_key)
                )
                transport = self._get_or_create_transport()
                await transport.send_json(websocket, cancel_event)
                return
            if not input_event_key:
                input_event_key = self._next_synthetic_input_event_key("server_auto")
                logger.info(
                    "response_created_key_synthesized run_id=%s turn_id=%s origin=%s reason=no_metadata_or_pending",
                    self._current_run_id() or "",
                    turn_id,
                    origin,
                )
            self._active_server_auto_input_event_key = input_event_key
            self._set_response_gating_verdict(
                turn_id=turn_id,
                input_event_key=input_event_key,
                action="NOOP",
                reason="awaiting_transcript_final",
            )
            self._log_response_binding_event(
                response_key=input_event_key,
                turn_id=turn_id,
                origin=origin,
            )
            if input_event_key:
                self._mark_transcript_response_outcome(
                    input_event_key=input_event_key,
                    turn_id=turn_id,
                    outcome="response_created",
                )
            canonical_key = self._canonical_utterance_key(turn_id=turn_id, input_event_key=input_event_key)
            resolved_input_event_key = input_event_key
            self._log_response_site_debug(
                site="server_auto_response_linked",
                turn_id=turn_id,
                input_event_key=input_event_key,
                canonical_key=canonical_key,
                origin=origin,
                trigger="response_created",
            )
            if not input_event_key:
                logger.info(
                    "memory_intent_decision_path run_id=%s turn_id=%s decision_path=speculative_server_auto",
                    self._current_run_id() or "",
                    turn_id,
                )
        else:
            current_input_event_key = metadata_input_event_key or pending_input_event_key or str(getattr(self, "_current_input_event_key", "") or "").strip()
            if not current_input_event_key:
                current_input_event_key = self._next_synthetic_input_event_key(origin)
                logger.info(
                    "response_created_key_synthesized run_id=%s turn_id=%s origin=%s reason=no_metadata_or_pending",
                    self._current_run_id() or "",
                    turn_id,
                    origin,
                )
            self._log_response_binding_event(
                response_key=current_input_event_key,
                turn_id=turn_id,
                origin=origin,
            )
            if current_input_event_key:
                self._mark_transcript_response_outcome(
                    input_event_key=current_input_event_key,
                    turn_id=turn_id,
                    outcome="response_created",
                )
            self._active_server_auto_input_event_key = None
            canonical_key = self._canonical_utterance_key(turn_id=turn_id, input_event_key=current_input_event_key)
            resolved_input_event_key = current_input_event_key
        with self._utterance_context_scope(
            turn_id=turn_id,
            input_event_key=resolved_input_event_key,
        ) as utterance_context:
            turn_id = utterance_context.turn_id
            resolved_input_event_key = utterance_context.input_event_key
            canonical_key = utterance_context.canonical_key
        if str((response_metadata or {}).get("tool_followup", "")).strip().lower() in {"true", "1", "yes"}:
            self._set_tool_followup_state(
                canonical_key=canonical_key,
                state="created",
                reason="response_created",
            )
        if str((response_metadata or {}).get("transcript_upgrade_replacement", "")).strip().lower() in {"true", "1", "yes"}:
            self._log_lifecycle_coherence(
                stage="replacement_created",
                turn_id=turn_id,
                response_id=self._active_response_id,
                canonical_key=canonical_key,
            )
        if origin != "server_auto":
            current_input_event_key = resolved_input_event_key
        upgrade_chain_id = metadata_upgrade_chain_id or self._ensure_upgrade_chain_id(
            turn_id=turn_id,
            input_event_key=resolved_input_event_key,
            response_id=self._active_response_id,
        )
        self._set_upgrade_chain_trace_context(
            response_id=self._active_response_id,
            chain_id=upgrade_chain_id,
            turn_id=turn_id,
            input_event_key=resolved_input_event_key,
            canonical_key=canonical_key,
            origin=origin,
        )
        arbitration_outcome, arbitration_reason_code = self._arbitrate_server_auto_response_created(
            turn_id=turn_id,
            input_event_key=resolved_input_event_key,
            canonical_key=canonical_key,
            origin=origin,
        )
        if origin == "server_auto":
            logger.info(
                "server_auto_arbitration outcome=%s reason=%s canonical_key=%s",
                arbitration_outcome.value,
                arbitration_reason_code,
                canonical_key,
            )
        created_keys_size_before = len(getattr(self, "_response_created_canonical_keys", set()) or ())
        consumes_canonical_slot = bool(getattr(self, "_active_response_consumes_canonical_slot", True))
        if consumes_canonical_slot:
            self._canonical_response_state_mutate(
                canonical_key=canonical_key,
                turn_id=turn_id,
                input_event_key=resolved_input_event_key,
                mutator=lambda record: (
                    setattr(record, "created", True),
                    setattr(record, "origin", origin),
                ),
            )
        created_keys_size_after = len(getattr(self, "_response_created_canonical_keys", set()) or ())
        logger.debug(
            "[RESPTRACE] response_created run_id=%s turn_id=%s origin=%s response_id=%s resolved_input_event_key=%s "
            "canonical_key=%s consumes_canonical_slot=%s created_keys_size_before=%s created_keys_size_after=%s "
            "active_key_for_turn=%s",
            self._current_run_id() or "",
            turn_id,
            origin,
            self._active_response_id or "unknown",
            str(resolved_input_event_key or "").strip() or "unknown",
            canonical_key,
            consumes_canonical_slot,
            created_keys_size_before,
            created_keys_size_after,
            self._active_input_event_key_for_turn(turn_id) or "unknown",
        )
        lifecycle_canonical_key = canonical_key
        if not consumes_canonical_slot:
            lifecycle_canonical_key = f"{canonical_key}:non_consuming:{self._active_response_id or 'unknown'}"
        lifecycle_state = self._canonical_lifecycle_state(lifecycle_canonical_key)
        lifecycle_state["server_auto_arbitration"] = arbitration_outcome.value
        lifecycle_created_decision = self._lifecycle_controller().on_response_created(
            lifecycle_canonical_key,
            origin=origin,
        )
        if consumes_canonical_slot and not (
            origin == "server_auto"
            and lifecycle_created_decision.action is LifecycleDecisionAction.DEFER
        ):
            self._record_substantive_response(
                turn_id=turn_id,
                canonical_key=canonical_key,
            )
        self._log_lifecycle_event(
            turn_id=turn_id,
            input_event_key=resolved_input_event_key,
            canonical_key=lifecycle_canonical_key,
            origin=origin,
            response_id=self._active_response_id,
            decision=f"response_created_{lifecycle_created_decision.action.value}:{lifecycle_created_decision.reason}",
        )
        if lifecycle_created_decision.action is LifecycleDecisionAction.CANCEL:
            self._debug_dump_canonical_key_timeline(
                canonical_key=lifecycle_canonical_key,
                trigger="response_created_cancelled",
            )
            cancel_event = {"type": "response.cancel"}
            log_ws_event("Outgoing", cancel_event)
            self._track_outgoing_event(cancel_event, origin="interaction_lifecycle_controller")
            self._lifecycle_controller().on_cancel_sent(lifecycle_canonical_key)
            self._log_lifecycle_event(
                turn_id=turn_id,
                input_event_key=resolved_input_event_key,
                canonical_key=lifecycle_canonical_key,
                origin=origin,
                response_id=self._active_response_id,
                decision="transition_cancel_sent:interaction_lifecycle_controller",
            )
            if websocket is not None:
                transport = self._get_or_create_transport()
                await transport.send_json(websocket, cancel_event)
            return
        if lifecycle_created_decision.action is LifecycleDecisionAction.DEFER:
            self._record_pending_server_auto_response(
                turn_id=turn_id,
                response_id=self._active_response_id,
                canonical_key=lifecycle_canonical_key,
                upgrade_chain_id=upgrade_chain_id,
            )
            self._mark_response_provisional(response_id=self._active_response_id)
        self._active_response_input_event_key = str(resolved_input_event_key or "").strip() or None
        self._bind_active_input_event_key_for_turn(
            turn_id=turn_id,
            input_event_key=self._active_response_input_event_key,
            cause="response_created",
            response_id=self._active_response_id,
            origin=origin,
        )
        self._active_response_canonical_key = lifecycle_canonical_key
        self._log_parent_binding_snapshot(
            turn_id=turn_id,
            response_id=self._active_response_id,
            origin=origin,
            input_event_key=str(resolved_input_event_key or ""),
            response_key=str(resolved_input_event_key or ""),
            response_metadata=response_metadata,
        )
        if self._active_response_id:
            self._record_response_trace_context(
                self._active_response_id,
                turn_id=turn_id,
                input_event_key=self._active_response_input_event_key or "",
                canonical_key=lifecycle_canonical_key,
                origin=origin,
                upgrade_chain_id=upgrade_chain_id,
            )
        self._emit_response_lifecycle_trace(
            event_type="response.created",
            response_id=str(self._active_response_id or ""),
            turn_id=turn_id,
            input_event_key=str(self._active_response_input_event_key or ""),
            canonical_key=lifecycle_canonical_key,
            origin=origin,
            active_input_event_key=str(self._active_response_input_event_key or ""),
            active_canonical_key=lifecycle_canonical_key,
            payload_summary=(
                "arbitration=%s consumes_canonical_slot=%s" % (arbitration_outcome.value, consumes_canonical_slot)
            ),
        )
        if consumes_canonical_slot:
            self._canonical_lifecycle_state(canonical_key).setdefault("first_audio_started", False)
            self._set_response_delivery_state(
                turn_id=turn_id,
                input_event_key=resolved_input_event_key,
                state="created",
            )
            self._canonical_response_state_mutate(
                canonical_key=canonical_key,
                turn_id=turn_id,
                input_event_key=resolved_input_event_key,
                mutator=lambda record: (
                    setattr(record, "response_id", str(self._active_response_id or "")),
                    setattr(record, "origin", origin),
                ),
            )
        active_input_event_key = str(getattr(self, "_active_server_auto_input_event_key", "") or "").strip()
        obligation_input_event_key = active_input_event_key if origin == "server_auto" else current_input_event_key
        if arbitration_outcome is ServerAutoArbitrationOutcome.CANCEL_PRE_AUDIO:
            replacement_reason = arbitration_reason_code
            self._active_response_preference_guarded = True
            if consumes_canonical_slot:
                self._canonical_response_state_mutate(
                    canonical_key=canonical_key,
                    turn_id=turn_id,
                    input_event_key=resolved_input_event_key,
                    mutator=lambda record: setattr(record, "created", False),
                )
                lifecycle_state["cancel_requested_pre_audio"] = True
            self._mark_transcript_response_outcome(
                input_event_key=active_input_event_key,
                turn_id=turn_id,
                outcome="response_not_scheduled",
                reason=replacement_reason,
                details="guarded server_auto response cancelled",
            )
            self._set_response_delivery_state(
                turn_id=turn_id,
                input_event_key=active_input_event_key,
                state="cancelled",
            )
            logger.info(
                "PREFERENCE_RECALL_RESPONSE_GUARDED origin=%s response_id=%s",
                origin,
                self._active_response_id or "unknown",
            )
            self._log_response_site_debug(
                site="PREFERENCE_RECALL_RESPONSE_GUARDED",
                turn_id=turn_id,
                input_event_key=active_input_event_key,
                canonical_key=canonical_key,
                origin=origin,
                trigger=replacement_reason,
            )
            logger.info(
                "server_auto_guard_pre_audio_cancel response_id=%s canonical_key=%s",
                self._active_response_id or "unknown",
                canonical_key,
            )
            cancel_event = {"type": "response.cancel"}
            log_ws_event("Outgoing", cancel_event)
            self._track_outgoing_event(cancel_event, origin="preference_recall_guard")
            self._lifecycle_controller().on_cancel_sent(canonical_key)
            self._log_lifecycle_event(
                turn_id=turn_id,
                input_event_key=active_input_event_key,
                canonical_key=canonical_key,
                origin=origin,
                response_id=self._active_response_id,
                decision=f"guard_cancel_sent:{replacement_reason}",
            )
            transport = self._get_or_create_transport()
            await transport.send_json(websocket, cancel_event)
        elif arbitration_outcome is ServerAutoArbitrationOutcome.DEFER:
            self._active_response_preference_guarded = True
            logger.info(
                "server_auto_guard_pre_audio_defer response_id=%s canonical_key=%s",
                self._active_response_id or "unknown",
                canonical_key,
            )
            cancel_event = {"type": "response.cancel"}
            log_ws_event("Outgoing", cancel_event)
            self._track_outgoing_event(cancel_event, origin="server_auto_arbitration_defer")
            self._lifecycle_controller().on_cancel_sent(canonical_key)
            transport = self._get_or_create_transport()
            await transport.send_json(websocket, cancel_event)
            return
        else:
            if consumes_canonical_slot:
                logger.debug(
                    "[RESPTRACE] response_created_clear_obligation run_id=%s turn_id=%s origin=%s "
                    "obligation_input_event_key=%s canonical_key=%s",
                    self._current_run_id() or "",
                    turn_id,
                    origin,
                    str(obligation_input_event_key or "").strip() or "unknown",
                    canonical_key,
                )
                self._clear_response_obligation(
                    turn_id=turn_id,
                    input_event_key=obligation_input_event_key,
                    reason="response_created",
                    origin=origin,
                )
            if origin == "server_auto":
                self._clear_stale_pending_server_auto_for_turn(
                    turn_id=turn_id,
                    active_input_event_key=active_input_event_key,
                    reason="response_created_for_active_input",
                )
        if pending_confirmation_active:
            log_info(f"response.created consumed by confirmation flow; origin={origin}")
            self._active_response_confirmation_guarded = self._should_guard_confirmation_response(
                origin,
                response,
            )
            if self._active_response_confirmation_guarded:
                self._mark_confirmation_activity(reason="guarded_response_created")
                logger.info(
                    "CONFIRMATION_NON_DECISION_RESPONSE_GUARDED origin=%s response_id=%s",
                    self._active_response_origin,
                    self._active_response_id or "unknown",
                )
        else:
            self._active_response_confirmation_guarded = False
            self.orchestration_state.transition(
                OrchestrationPhase.PLAN,
                reason="response created",
            )
        deferred_audio_start = False
        if origin == "server_auto":
            upgrade_likely, likely_reasons = self._upgrade_likely_for_server_auto_turn(
                turn_id=turn_id,
                input_event_key=resolved_input_event_key,
            )
            should_delay_for_transcript_final = self._should_delay_server_auto_until_transcript_final(
                turn_id=turn_id,
                input_event_key=resolved_input_event_key,
            )
            if (upgrade_likely or should_delay_for_transcript_final) and self._active_response_id and resolved_input_event_key:
                deferred_audio_start = True
                defer_reason = likely_reasons if upgrade_likely else "awaiting_transcript_final_short_utterance"
                self._set_server_auto_pre_audio_hold(
                    turn_id=turn_id,
                    enabled=should_delay_for_transcript_final,
                    reason=("awaiting_transcript_final" if should_delay_for_transcript_final else "upgrade_likely"),
                    response_id=self._active_response_id,
                )
                logger.info(
                    "server_auto_audio_start_deferred run_id=%s turn_id=%s response_id=%s timeout_ms=%s reasons=%s",
                    self._current_run_id() or "",
                    turn_id,
                    self._active_response_id,
                    self._server_auto_audio_deferral_timeout_ms,
                    defer_reason,
                )
                self._schedule_server_auto_audio_deferral(
                    turn_id=turn_id,
                    input_event_key=resolved_input_event_key,
                    response_id=self._active_response_id,
                )
        if not deferred_audio_start:
            self._start_audio_response_if_needed(response_id=self._active_response_id)
        self._audio_accum.clear()
        self._audio_accum_response_id = None
        self._mic_receive_on_first_audio = True
        self.response_in_progress = True
        self._response_in_flight = True
        self._speaking_started = False
        self.assistant_reply = ""
        self._assistant_reply_accum = ""
        self._assistant_reply_response_id = self._active_response_id
        self._tool_call_records = []
        self._last_tool_call_results = []
        self._last_response_metadata = {}
        self._reflection_enqueued = False
        if (
            self.state_manager.state != InteractionState.LISTENING
            or self._is_user_approved_interrupt_response(response)
        ):
            self.state_manager.update_state(InteractionState.THINKING, "response created")
        else:
            logger.info(
                "Skipping THINKING transition for response.created while still listening."
            )

    async def _handle_response_output_audio_delta_event(self, event: dict[str, Any], websocket: Any) -> None:
        _ = websocket
        response_id = self._response_id_from_event(event)
        if not self._should_process_response_event_ingress(event, source="audio_delta"):
            return
        if self._is_active_response_guarded():
            return
        active_input_event_key = str(getattr(self, "_active_response_input_event_key", "") or "").strip()
        active_turn_id = self._current_turn_id_or_unknown()
        if str(getattr(self, "_active_response_origin", "") or "").strip().lower() == "server_auto" and active_input_event_key:
            verdict = self._get_response_gating_verdict(turn_id=active_turn_id, input_event_key=active_input_event_key)
            if verdict is None or verdict.action == "NOOP":
                return
            if verdict.action in {"CLARIFY", "UPGRADE"}:
                if response_id:
                    self._quarantine_cancelled_response_id(
                        response_id=response_id,
                        turn_id=active_turn_id,
                        input_event_key=active_input_event_key,
                        origin=str(getattr(self, "_active_response_origin", "unknown") or "unknown"),
                        reason=f"audio_delta_{verdict.action.lower()}",
                    )
                    cancel_event = {"type": "response.cancel", "response_id": response_id}
                    transport = self._get_or_create_transport()
                    await transport.send_json(websocket, cancel_event)
                return
        self._mark_utterance_info_summary(deliverable_seen=True)
        active_canonical_key = str(getattr(self, "_active_response_canonical_key", "") or "").strip()
        if active_canonical_key:
            lifecycle_state = self._canonical_lifecycle_state(active_canonical_key)
            if lifecycle_state.get("cancel_requested_pre_audio", False):
                return
            arbitration_state = str(
                lifecycle_state.get(
                    "server_auto_arbitration",
                    ServerAutoArbitrationOutcome.ALLOW.value,
                )
                or ""
            ).strip().lower()
            if arbitration_state and arbitration_state != ServerAutoArbitrationOutcome.ALLOW.value:
                return
        if active_canonical_key:
            audio_decision = self._lifecycle_controller().on_audio_delta(active_canonical_key)
            self._log_lifecycle_event(
                turn_id=self._current_turn_id_or_unknown(),
                input_event_key=getattr(self, "_active_response_input_event_key", None),
                canonical_key=active_canonical_key,
                origin=getattr(self, "_active_response_origin", "unknown"),
                response_id=getattr(self, "_active_response_id", None),
                decision=f"audio_delta_{audio_decision.action.value}:{audio_decision.reason}",
            )
            if audio_decision.action is LifecycleDecisionAction.CANCEL:
                self._debug_dump_canonical_key_timeline(
                    canonical_key=active_canonical_key,
                    trigger="audio_delta_cancelled",
                )
                return
        self._cancel_micro_ack(turn_id=self._current_turn_id_or_unknown(), reason="response_started")
        if active_canonical_key:
            self._canonical_lifecycle_state(active_canonical_key)["first_audio_started"] = True
            self._canonical_response_state_mutate(
                canonical_key=active_canonical_key,
                turn_id=self._current_turn_id_or_unknown(),
                input_event_key=getattr(self, "_active_response_input_event_key", None),
                mutator=lambda record: (
                    setattr(record, "created", True),
                    setattr(record, "audio_started", True),
                    setattr(record, "origin", str(getattr(self, "_active_response_origin", "unknown") or "unknown")),
                    setattr(record, "response_id", str(getattr(self, "_active_response_id", "") or "")),
                ),
            )
        self._audio_playback_busy = True
        audio_data = base64.b64decode(event["delta"])
        if not str(getattr(self, "_audio_accum_response_id", "") or "").strip():
            self._audio_accum_response_id = str(response_id or "").strip() or None
        self._audio_accum.extend(audio_data)

        if self._mic_receive_on_first_audio and not self.mic.is_receiving:
            self._mic_receive_on_first_audio = False
            log_info("Starting mic receiving on first audio delta.")
            self.mic.start_receiving()

        if not self._speaking_started:
            self._speaking_started = True
            self.state_manager.update_state(InteractionState.SPEAKING, "audio output")

        if len(self._audio_accum) >= self._audio_accum_bytes_target:
            if self.audio_player:
                self.audio_player.play_audio(bytes(self._audio_accum))
            self._audio_accum.clear()
            self._audio_accum_response_id = None

    async def _handle_response_done_event(self, event: dict[str, Any], websocket: Any) -> None:
        _ = websocket
        response_id = self._response_id_from_event(event)
        if not self._should_process_response_event_ingress(event, source="done"):
            return
        self._set_response_status(response_id=response_id, status="terminal_done")
        await self.handle_response_done(event)

    async def _cancel_and_replace_pending_server_auto_on_transcript_final(
        self,
        *,
        websocket: Any,
        turn_id: str,
        input_event_key: str,
        origin_label: str = "upgraded_response",
        memory_intent_subtype: str = "none",
    ) -> bool:
        cancel_replace_started_at = time.monotonic()
        self._cancel_micro_ack(turn_id=turn_id, reason="upgrade_selected")
        pending = self._pending_server_auto_response_for_turn(turn_id=turn_id)
        pending_active = bool(isinstance(pending, PendingServerAutoResponse) and pending.active)
        if pending_active and pending is not None:
            pending_active = self._pending_server_auto_response_mutation_allowed(
                pending=pending,
                turn_id=turn_id,
                mutation="cancel_and_replace",
            )
        old_response_id = pending.response_id if pending_active and pending is not None else ""
        action = "no_pending"
        if pending_active and old_response_id:
            action = "cancel_and_replace"
        elif pending is not None:
            action = "replace_only"
        upgrade_chain_id = str(getattr(pending, "upgrade_chain_id", "") or "").strip() if pending is not None else ""
        if not upgrade_chain_id:
            upgrade_chain_id = self._upgrade_chain_id_from_response(old_response_id)
        logger.info(
            "upgrade_flow_snapshot run_id=%s turn_id=%s old_response_id=%s pending_owner_response_id=%s pending_server_auto_active=%s action=%s upgrade_chain_id=%s",
            self._current_run_id() or "",
            turn_id,
            old_response_id or "none",
            old_response_id or "none",
            str(pending_active).lower(),
            action,
            upgrade_chain_id or "none",
        )
        if pending is None:
            return False
        replacement_canonical_key = self._canonical_utterance_key(turn_id=turn_id, input_event_key=input_event_key)
        if str(pending.canonical_key or "").strip() == replacement_canonical_key:
            logger.info(
                "upgrade_flow_fallback outcome=keep_server_auto reason=same_canonical_key run_id=%s turn_id=%s canonical_key=%s",
                self._current_run_id() or "",
                turn_id,
                replacement_canonical_key,
            )
            return False
        old_pending_canonical_key = str(pending.canonical_key or "").strip()
        if old_pending_canonical_key:
            self._lifecycle_controller().on_replaced(old_pending_canonical_key, replacement_canonical_key)
            self._log_lifecycle_event(
                turn_id=turn_id,
                input_event_key=input_event_key,
                canonical_key=replacement_canonical_key,
                origin=origin_label,
                response_id=old_response_id or None,
                decision="transition_replaced:cause=replacement_create",
            )
        if pending_active and old_response_id:
            cancel_event = {"type": "response.cancel", "response_id": old_response_id}
            self._record_cancel_issued_timing(old_response_id)
            self._stale_response_ids().add(old_response_id)
            if str(getattr(self, "_assistant_reply_response_id", "") or "").strip() == old_response_id:
                self.assistant_reply = ""
                self._assistant_reply_accum = ""
                self._assistant_reply_response_id = None
            log_ws_event("Outgoing", cancel_event)
            self._track_outgoing_event(cancel_event, origin="server_auto_upgrade")
            self._mark_pending_server_auto_response_cancelled(
                turn_id=turn_id,
                reason="transcript_final_upgrade",
            )
            self._suppress_cancelled_response_audio(old_response_id)
            self._mark_canonical_cancelled_for_upgrade(
                canonical_key=pending.canonical_key,
                turn_id=turn_id,
                response_id=old_response_id,
            )
            transport = self._get_or_create_transport()
            await transport.send_json(websocket, cancel_event)
            if str(getattr(self, "_active_response_id", "") or "").strip() == old_response_id:
                self._response_in_flight = False
                self.response_in_progress = False

        replacement_event = {"type": "response.create", "response": {"metadata": {}}}
        metadata = replacement_event["response"]["metadata"]
        metadata["turn_id"] = str(turn_id or "").strip() or "turn-unknown"
        metadata["input_event_key"] = str(input_event_key or "").strip()
        metadata["safety_override"] = "true"
        metadata["transcript_upgrade_replacement"] = "true"
        normalized_memory_intent_subtype = str(memory_intent_subtype or "none").strip().lower() or "none"
        if normalized_memory_intent_subtype != "none":
            metadata["memory_intent_subtype"] = normalized_memory_intent_subtype
        if upgrade_chain_id:
            metadata["upgrade_chain_id"] = upgrade_chain_id
        self._clear_canonical_terminal_delivery_state(canonical_key=replacement_canonical_key)
        pref_payload = self._peek_pending_preference_memory_context_payload(
            turn_id=turn_id,
            input_event_key=input_event_key,
        )
        pref_prompt_note = str(pref_payload.get("prompt_note") or "").strip() if isinstance(pref_payload, dict) else ""
        pref_hit = bool(isinstance(pref_payload, dict) and pref_payload.get("hit", False))
        logger.info(
            "replacement_response_scheduled run_id=%s turn_id=%s canonical_key=%s input_event_key=%s pref_ctx_len=%s pref_hit=%s upgrade_chain_id=%s",
            self._current_run_id() or "",
            turn_id,
            replacement_canonical_key,
            input_event_key,
            len(pref_prompt_note),
            str(pref_hit).lower(),
            upgrade_chain_id or "none",
        )
        replacement_sent = await self._send_response_create(
            websocket,
            replacement_event,
            origin=origin_label,
            utterance_context=UtteranceContext(
                turn_id=turn_id,
                input_event_key=input_event_key,
                canonical_key=replacement_canonical_key,
                utterance_seq=self._current_utterance_seq(),
            ),
            record_ai_call=True,
            memory_brief_note=pref_prompt_note or None,
        )
        if not replacement_sent:
            self._cancel_micro_ack(turn_id=turn_id, reason="upgrade_blocked")
            return False
        turn_timestamps_store = getattr(self, "_turn_diagnostic_timestamps", None)
        if not isinstance(turn_timestamps_store, dict):
            turn_timestamps_store = {}
            self._turn_diagnostic_timestamps = turn_timestamps_store
        turn_timestamps = turn_timestamps_store.setdefault(turn_id, {})
        turn_timestamps["cancel_replace_ms"] = (time.monotonic() - cancel_replace_started_at) * 1000.0
        self._mark_pending_server_auto_response_replaced(turn_id=turn_id)
        self._log_lifecycle_coherence(
            stage="replacement_scheduled",
            turn_id=turn_id,
            response_id=old_response_id,
            canonical_key=replacement_canonical_key,
        )
        return True

    async def _handle_input_audio_transcription_completed_event(
        self,
        event: dict[str, Any],
        websocket: Any,
    ) -> None:
        event_type = str(event.get("type") or "unknown")
        transcript = self._extract_transcript(event)
        self._mark_utterance_info_summary(transcript_present=bool(transcript))
        self._log_user_transcript(transcript or "", final=True, event_type=event_type)
        input_event_key = self._resolve_input_event_key(event)
        resolved_turn_id = self._current_turn_id_or_unknown()
        with self._utterance_context_scope(
            turn_id=resolved_turn_id,
            input_event_key=input_event_key,
        ) as utterance_context:
            resolved_turn_id = utterance_context.turn_id
            input_event_key = utterance_context.input_event_key
        active_by_turn = getattr(self, "_active_input_event_key_by_turn_id", None)
        if not isinstance(active_by_turn, dict):
            active_by_turn = {}
            self._active_input_event_key_by_turn_id = active_by_turn
        active_by_turn[resolved_turn_id] = input_event_key
        self._lifecycle_controller().on_transcript_final(
            self._canonical_utterance_key(turn_id=resolved_turn_id, input_event_key=input_event_key)
        )
        self._rebind_active_response_correlation_key(
            turn_id=resolved_turn_id,
            replacement_input_event_key=input_event_key,
            cause="transcript_final_rebind",
        )
        transcript_canonical_key = self._canonical_utterance_key(
            turn_id=resolved_turn_id,
            input_event_key=input_event_key,
        )
        pending_server_auto = self._pending_server_auto_response_for_turn(turn_id=resolved_turn_id)
        pending_server_auto_key = str(getattr(pending_server_auto, "canonical_key", "") or "").strip()
        pending_server_auto_active = bool(
            isinstance(pending_server_auto, PendingServerAutoResponse) and pending_server_auto.active
        )
        logger.info(
            "transcript_final_handoff run_id=%s turn_id=%s old_key=%s new_key=%s prior_response_id=%s pending_server_auto_active=%s canonical_owner_before=%s canonical_owner_after=%s",
            self._current_run_id() or "",
            resolved_turn_id,
            pending_server_auto_key or "none",
            transcript_canonical_key,
            str(getattr(pending_server_auto, "response_id", "") or "none"),
            str(pending_server_auto_active).lower(),
            pending_server_auto_key or "none",
            transcript_canonical_key,
        )
        self._clear_stale_pending_server_auto_for_turn(
            turn_id=resolved_turn_id,
            active_input_event_key=input_event_key,
            reason="new_transcript_final",
        )
        pending_server_auto_input_event_key = ""
        pending_server_auto_response_id = str(getattr(pending_server_auto, "response_id", "") or "").strip()
        if pending_server_auto_response_id:
            trace = self._response_trace_by_id().get(pending_server_auto_response_id, {})
            pending_server_auto_input_event_key = str(trace.get("input_event_key") or "").strip()
        if not pending_server_auto_input_event_key:
            pending_server_auto_input_event_key = str(getattr(self, "_active_server_auto_input_event_key", "") or "").strip()
        if not pending_server_auto_input_event_key:
            pending_server_auto_input_event_key = str(getattr(pending_server_auto, "canonical_key", "") or "").strip()
            if pending_server_auto_input_event_key and ":" in pending_server_auto_input_event_key:
                pending_server_auto_input_event_key = pending_server_auto_input_event_key.split(":")[-1]
        if pending_server_auto_input_event_key:
            self._invalidate_provisional_tool_followups_for_turn(
                turn_id=resolved_turn_id,
                provisional_parent_input_event_key=pending_server_auto_input_event_key,
                reason="transcript_final_handoff",
            )
        memory_intent_subtype = self._classify_memory_intent(transcript or "")
        memory_intent = memory_intent_subtype != "none"
        logger.info(
            "memory_intent_classification run_id=%s source=input_audio_transcription input_event_key=%s memory_intent=%s memory_intent_subtype=%s",
            self._current_run_id() or "",
            input_event_key,
            str(memory_intent).lower(),
            memory_intent_subtype,
        )
        if memory_intent:
            self._set_response_obligation(
                turn_id=self._current_turn_id_or_unknown(),
                input_event_key=input_event_key,
                source="input_audio_transcription",
            )
            logger.info(
                "response_obligation_eval run_id=%s turn_id=%s input_event_key=%s obligation_state=%s response_created_seen=%s response_done_seen=%s action=%s reason=%s",
                self._current_run_id() or "",
                resolved_turn_id,
                input_event_key,
                "open",
                str(
                    self._summary_response_created_seen_for_canonical(
                        turn_id=resolved_turn_id,
                        canonical_key=transcript_canonical_key,
                    )
                ).lower(),
                str(self._response_delivery_state(turn_id=resolved_turn_id, input_event_key=input_event_key) == "done").lower(),
                "schedule",
                "memory_intent_transcript_final",
            )
        pending_server_auto_keys = getattr(self, "_pending_server_auto_input_event_keys", None)
        if not isinstance(pending_server_auto_keys, deque):
            pending_server_auto_keys = deque(maxlen=64)
            self._pending_server_auto_input_event_keys = pending_server_auto_keys
        active_input_event_key = str(getattr(self, "_active_server_auto_input_event_key", "") or "").strip()
        has_active_server_auto = str(getattr(self, "_active_response_origin", "")).strip().lower() == "server_auto"
        if not (has_active_server_auto and active_input_event_key and active_input_event_key == input_event_key):
            pending_server_auto_keys.append(input_event_key)
        logger.info(
            "input_audio_transcription_linked run_id=%s input_event_key=%s pending_server_auto=%s",
            self._current_run_id() or "",
            input_event_key,
            len(pending_server_auto_keys),
        )
        self._start_transcript_response_watchdog(
            turn_id=resolved_turn_id,
            input_event_key=input_event_key,
        )
        turn_timestamps_store = getattr(self, "_turn_diagnostic_timestamps", None)
        if not isinstance(turn_timestamps_store, dict):
            turn_timestamps_store = {}
            self._turn_diagnostic_timestamps = turn_timestamps_store
        turn_timestamps = turn_timestamps_store.setdefault(resolved_turn_id, {})
        turn_timestamps["transcript_final"] = time.monotonic()
        self._signal_server_auto_transcript_final(turn_id=resolved_turn_id)
        partial_store = getattr(self, "_latest_partial_transcript_by_turn_id", None)
        if isinstance(partial_store, dict):
            partial_store.pop(resolved_turn_id, None)
        confirmation_active = self._has_active_confirmation_token() or self._is_awaiting_confirmation_phase()
        if confirmation_active:
            self._confirmation_asr_pending = False
            self._mark_confirmation_activity(reason="transcription_completed")
        if self._active_utterance is not None:
            self._active_utterance["transcript"] = transcript or ""
            self._active_utterance["transcript_len"] = len((transcript or "").strip())
            decision = self._parse_confirmation_decision(transcript or "")
            self._active_utterance["confirmation_candidate"] = decision in {"yes", "no"}
            self._active_utterance["decision"] = decision
            duration_ms = self._active_utterance.get("duration_ms")
            suppressed = self._should_suppress_short_utterance(transcript, duration_ms)
            self._active_utterance["suppressed"] = suppressed
            if suppressed:
                await self._suppress_guardrail_response(websocket, transcript)
            self._log_utterance_envelope(event_type)
            if suppressed:
                return
        trust_snapshot = self._log_utterance_trust_snapshot(
            transcript=transcript or "",
            event=event,
            turn_id=resolved_turn_id,
            input_event_key=input_event_key,
        )
        transcript_word_count = int(trust_snapshot.get("word_count") or 0)
        if not transcript or transcript_word_count <= 0:
            self._cancel_micro_ack(turn_id=resolved_turn_id, reason="transcript_completed_empty")
            summary_canonical_key = self._canonical_utterance_key(
                turn_id=resolved_turn_id,
                input_event_key=input_event_key,
            )
            self._mark_utterance_info_summary(
                response_created_seen=self._summary_response_created_seen_for_canonical(
                    turn_id=resolved_turn_id,
                    canonical_key=summary_canonical_key,
                ),
                response_done_seen=False,
                deliverable_seen=False,
            )
            self._emit_utterance_info_summary(anchor="transcript_completed_empty")
            pending = self._pending_server_auto_response_for_turn(turn_id=resolved_turn_id)
            if isinstance(pending, PendingServerAutoResponse) and pending.active and pending.response_id:
                cancel_event = {"type": "response.cancel", "response_id": pending.response_id}
                self._record_cancel_issued_timing(pending.response_id)
                self._stale_response_ids().add(pending.response_id)
                self._mark_pending_server_auto_response_cancelled(turn_id=resolved_turn_id, reason="empty_transcript")
                self._suppress_cancelled_response_audio(pending.response_id)
                transport = self._get_or_create_transport()
                await transport.send_json(websocket, cancel_event)
            self._mark_empty_transcript_blocked(
                turn_id=resolved_turn_id,
                input_event_key=input_event_key,
                reason="transcript_missing_or_zero_words",
            )
            self._clear_response_obligation(
                turn_id=resolved_turn_id,
                input_event_key=input_event_key,
                reason="empty_transcript",
                origin="input_audio_transcription",
            )
            self._set_response_gating_verdict(
                turn_id=resolved_turn_id,
                input_event_key=input_event_key,
                action="CLARIFY",
                reason="empty_transcript_blocked",
            )
            self._mark_transcript_response_outcome(
                input_event_key=input_event_key,
                turn_id=resolved_turn_id,
                outcome="response_not_scheduled",
                reason="empty_transcript_blocked",
                details="transcript_missing_or_zero_words",
            )
            if hasattr(self, "state_manager") and self.state_manager is not None:
                self.state_manager.update_state(InteractionState.LISTENING, "empty transcript blocked")
            return
        if transcript and not confirmation_active and await self._maybe_verify_on_risk_clarify(
            transcript=transcript,
            websocket=websocket,
            turn_id=resolved_turn_id,
            input_event_key=input_event_key,
            snapshot=trust_snapshot,
        ):
            return
        if transcript and not confirmation_active:
            self._set_response_gating_verdict(
                turn_id=resolved_turn_id,
                input_event_key=input_event_key,
                action="ANSWER",
                reason="transcript_final",
            )
        if transcript:
            decision_path = "canonical_transcript"
            if memory_intent:
                transcript_upgrade_candidate = bool(
                    pending_server_auto is not None
                    and pending_server_auto_key
                    and pending_server_auto_key != transcript_canonical_key
                )
                decision_path = "upgraded_response" if transcript_upgrade_candidate else "canonical_transcript"
                logger.info(
                    "memory_intent_decision_path run_id=%s turn_id=%s input_event_key=%s decision_path=%s",
                    self._current_run_id() or "",
                    resolved_turn_id,
                    input_event_key,
                    decision_path,
                )
            if decision_path != "upgraded_response":
                self._maybe_schedule_micro_ack(
                    turn_id=resolved_turn_id,
                    category=self._micro_ack_category_for_reason("transcript_finalized"),
                    channel="voice",
                    action=self._canonical_utterance_key(turn_id=resolved_turn_id, input_event_key=input_event_key),
                    reason="transcript_finalized",
                    expected_delay_ms=700,
                )
            else:
                self._cancel_micro_ack(turn_id=resolved_turn_id, reason="upgrade_selected")
            self._record_user_input(transcript, source="input_audio_transcription")
            if await self._maybe_handle_approval_response(transcript, websocket):
                return
            if await self._handle_stop_word(
                transcript,
                websocket,
                source="input_audio_transcription",
            ):
                return
            if await self._maybe_handle_research_permission_response(transcript, websocket):
                return
            if await self._maybe_handle_research_budget_response(transcript, websocket):
                return
            if await self._maybe_apply_late_confirmation_decision(transcript, websocket):
                return
            if await self._maybe_handle_preference_recall_intent(
                transcript,
                websocket,
                source="input_audio_transcription",
            ):
                return
            if decision_path == "upgraded_response":
                pending = self._pending_server_auto_response_for_turn(turn_id=resolved_turn_id)
                replacement_canonical_key = self._canonical_utterance_key(
                    turn_id=resolved_turn_id,
                    input_event_key=input_event_key,
                )
                pending_is_active = bool(isinstance(pending, PendingServerAutoResponse) and pending.active)
                can_cancel = self.should_cancel_and_replace(
                    server_auto_state=pending,
                    transcript_final_state={
                        "turn_id": resolved_turn_id,
                        "input_event_key": input_event_key,
                    },
                    pref_ctx_state=self._peek_pending_preference_memory_context_payload(
                        turn_id=resolved_turn_id,
                        input_event_key=input_event_key,
                    ),
                )
                logger.info(
                    "canonical_response_schedule_eval run_id=%s turn_id=%s input_event_key=%s synthetic_key=%s transcript_key=%s action=%s reason=%s",
                    self._current_run_id() or "",
                    resolved_turn_id,
                    input_event_key,
                    str(getattr(pending, "canonical_key", "") or "none"),
                    replacement_canonical_key,
                    "replace" if pending is not None else "skip",
                    "pending_server_auto_present" if pending is not None else "no_pending_server_auto",
                )
                if pending_is_active and not can_cancel:
                    fallback_reason = "audio_already_started"
                    if str(getattr(pending, "canonical_key", "") or "").strip() == replacement_canonical_key:
                        fallback_reason = "same_canonical_key"
                    logger.info(
                        "canonical_response_schedule_eval run_id=%s turn_id=%s input_event_key=%s synthetic_key=%s transcript_key=%s action=%s reason=%s",
                        self._current_run_id() or "",
                        resolved_turn_id,
                        input_event_key,
                        str(getattr(pending, "canonical_key", "") or "none"),
                        replacement_canonical_key,
                        "skip",
                        fallback_reason,
                    )
                    logger.info(
                        "upgrade_flow_fallback outcome=keep_server_auto reason=%s run_id=%s turn_id=%s input_event_key=%s",
                        fallback_reason,
                        self._current_run_id() or "",
                        resolved_turn_id,
                        input_event_key,
                    )
                    return
                self._set_response_gating_verdict(
                    turn_id=resolved_turn_id,
                    input_event_key=input_event_key,
                    action="UPGRADE",
                    reason="transcript_upgrade",
                )
                if await self._cancel_and_replace_pending_server_auto_on_transcript_final(
                    websocket=websocket,
                    turn_id=resolved_turn_id,
                    input_event_key=input_event_key,
                    origin_label="upgraded_response",
                    memory_intent_subtype=memory_intent_subtype,
                ):
                    return
            if await self._maybe_process_research_intent(
                transcript,
                websocket,
                source="input_audio_transcription",
            ):
                return
        elif confirmation_active and self._has_active_confirmation_token():
            token = self._pending_confirmation_token
            metadata = token.metadata if token is not None and isinstance(token.metadata, dict) else {}
            metadata["empty_transcript_reprompted"] = True
            if token is not None:
                token.metadata = metadata

    async def _handle_input_audio_transcription_failed_event(
        self,
        event: dict[str, Any],
        websocket: Any,
    ) -> None:
        _ = websocket
        error_payload = event.get("error")
        self._mark_utterance_info_summary(asr_error_present=True, transcript_present=False)
        logger.info(
            "input_audio_transcription_failed run_id=%s error=%s",
            self._current_run_id() or "",
            json.dumps(error_payload, sort_keys=True) if isinstance(error_payload, dict) else str(error_payload or "unknown"),
        )
        self._emit_utterance_info_summary(anchor="transcript_failed")

    async def _handle_event_legacy(self, event: dict[str, Any], websocket: Any) -> None:
        event_type = event.get("type")
        response_id = self._response_id_from_event(event)
        if not self._should_process_response_event_ingress(event, source="legacy"):
            return
        if event_type in {"response.done", "response.completed"} and response_id:
            self._set_response_status(response_id=response_id, status="terminal_done")
        if event_type == "response.output_item.added":
            await self._handle_output_item_added_event(event, websocket)
        elif event_type == "response.function_call_arguments.delta":
            await self._handle_function_call_arguments_delta_event(event, websocket)
        elif event_type == "response.function_call_arguments.done":
            await self._handle_function_call_arguments_done_event(event, websocket)
        elif event_type == "response.text.delta":
            if self._is_active_response_guarded():
                return
            if not self._allow_text_output_state_transition(response_id=response_id, event_type=event_type):
                return
            self._mark_utterance_info_summary(deliverable_seen=True)
            self._cancel_micro_ack(turn_id=self._current_turn_id_or_unknown(), reason="response_started")
            delta = event.get("delta", "")
            self._record_active_canonical_deliverable_class(text=delta, reason=event_type)
            self._mark_first_assistant_utterance_observed_if_needed(delta)
            self._append_assistant_reply_text(delta, allow_separator=False, response_id=response_id)
            self.state_manager.update_state(InteractionState.SPEAKING, "text output")
        elif event_type == "response.output_text.delta":
            if self._is_active_response_guarded():
                return
            if not self._allow_text_output_state_transition(response_id=response_id, event_type=event_type):
                return
            self._mark_utterance_info_summary(deliverable_seen=True)
            self._cancel_micro_ack(turn_id=self._current_turn_id_or_unknown(), reason="response_started")
            delta = event.get("delta", "")
            self._record_active_canonical_deliverable_class(text=delta, reason=event_type)
            logger.debug(
                "assistant_content_event_received event_type=response.output_text.delta delta_len=%s",
                len(delta),
            )
            self._mark_first_assistant_utterance_observed_if_needed(delta)
            self._append_assistant_reply_text(delta, allow_separator=False, response_id=response_id)
            self.state_manager.update_state(InteractionState.SPEAKING, "text output")
        elif event_type in {"response.output_text.done", "response.text.done"}:
            if self._is_active_response_guarded():
                return
            if self.state_manager.state == InteractionState.SPEAKING and not self._audio_playback_busy:
                self.state_manager.update_state(InteractionState.IDLE, "text output done")
        elif event_type == "response.output_audio.done":
            await self.handle_audio_response_done()
            self.state_manager.update_state(InteractionState.IDLE, "audio output done")
        elif event_type == "response.output_audio_transcript.delta":
            if self._is_active_response_guarded():
                return
            self._mark_utterance_info_summary(deliverable_seen=True)
            delta = event.get("delta", "")
            self._mark_first_assistant_utterance_observed_if_needed(delta)
            self._append_assistant_reply_text(delta, allow_separator=False, response_id=response_id)
        elif event_type == "response.output_audio_transcript.done":
            await self.handle_transcribe_response_done()
            self.state_manager.update_state(
                InteractionState.IDLE,
                "audio transcript done",
            )
        elif event_type == "conversation.item.added":
            if self._is_active_response_guarded():
                return
            if not self._allow_text_output_state_transition(response_id=response_id, event_type=event_type):
                return
            item = event.get("item", {})
            if not isinstance(item, dict):
                return
            if item.get("type") != "message" or item.get("role") != "assistant":
                return
            extracted_text, part_types = self._extract_assistant_text_from_content(item.get("content", []))
            logger.debug(
                "assistant_content_event_received event_type=conversation.item.added part_types=%s extracted_chars=%s",
                ",".join(part_types),
                len(extracted_text),
            )
            if not extracted_text:
                return
            self._cancel_micro_ack(turn_id=self._current_turn_id_or_unknown(), reason="response_started")
            self._record_active_canonical_deliverable_class(text=extracted_text, reason="conversation.item.added")
            self._mark_first_assistant_utterance_observed_if_needed(extracted_text)
            self._append_assistant_reply_text(extracted_text, response_id=response_id)
            self.state_manager.update_state(InteractionState.SPEAKING, "text output")
            current_turn_id = self._current_turn_id_or_unknown()
            current_input_event_key = str(getattr(self, "_current_input_event_key", "") or "").strip()
            self._set_response_delivery_state(
                turn_id=current_turn_id,
                input_event_key=current_input_event_key,
                state="delivered",
            )
        elif event_type == "response.completed":
            await self.handle_response_completed(event)
        elif event_type == "error":
            await self.handle_error(event, websocket)
        elif event_type == "rate_limits.updated":
            expected_buckets = {"requests", "tokens"}
            rl, rl_meta = parse_rate_limits(event)
            self.rate_limits = rl

            present_names = set(rl_meta.get("present_names", []))
            session_id = "unknown"
            memory_manager = getattr(self, "_memory_manager", None)
            if memory_manager is not None:
                session_id_value = memory_manager.get_active_session_id()
                if session_id_value:
                    session_id = str(session_id_value)

            self.rate_limits_last_present_names = set(present_names)
            self.rate_limits_last_event_id = str(rl_meta.get("event_id") or "")
            self.rate_limits_supports_tokens = self.rate_limits_supports_tokens or ("tokens" in present_names)
            self.rate_limits_supports_requests = self.rate_limits_supports_requests or (
                "requests" in present_names
            )

            if "tokens" in present_names:
                self._rate_limits_regression_missing_counts["tokens"] = 0
            elif self.rate_limits_supports_tokens:
                self._rate_limits_regression_missing_counts["tokens"] += 1

            if "requests" in present_names:
                self._rate_limits_regression_missing_counts["requests"] = 0
            elif self.rate_limits_supports_requests:
                self._rate_limits_regression_missing_counts["requests"] += 1

            malformed_rate_limits = (
                (not rl_meta.get("rate_limits_is_list", True))
                or bool(rl_meta.get("malformed_count", 0))
            )
            if not present_names:
                logger.warning(
                    "Realtime API rate_limits.updated has no buckets: event_id=%s entry_count=%s "
                    "malformed_count=%s unknown_names=%s",
                    self.rate_limits_last_event_id or "unknown",
                    rl_meta.get("entry_count", 0),
                    rl_meta.get("malformed_count", 0),
                    rl_meta.get("unknown_names", []),
                )
            elif malformed_rate_limits:
                logger.warning(
                    "Realtime API rate_limits.updated malformed entries detected: event_id=%s present_names=%s "
                    "entry_count=%s malformed_count=%s unknown_names=%s",
                    self.rate_limits_last_event_id or "unknown",
                    sorted(present_names),
                    rl_meta.get("entry_count", 0),
                    rl_meta.get("malformed_count", 0),
                    rl_meta.get("unknown_names", []),
                )

            for bucket in sorted(expected_buckets):
                seen_in_session = bool(getattr(self, f"rate_limits_supports_{bucket}", False))
                missing_count = self._rate_limits_regression_missing_counts[bucket]
                if bucket in present_names or not seen_in_session or missing_count <= 0:
                    continue
                if self._rate_limits_strict and missing_count == 1:
                    logger.warning(
                        "Realtime API rate_limits.updated regression: bucket=%s missing_count=%s "
                        "session_id=%s strict=%s",
                        bucket,
                        missing_count,
                        session_id,
                        self._rate_limits_strict,
                    )
                elif (not self._rate_limits_strict) and (
                    missing_count == self._rate_limits_regression_warning_threshold
                ):
                    logger.warning(
                        "Realtime API rate_limits.updated regression: bucket=%s missing_count=%s "
                        "threshold=%s session_id=%s strict=%s",
                        bucket,
                        missing_count,
                        self._rate_limits_regression_warning_threshold,
                        session_id,
                        self._rate_limits_strict,
                    )

            parts = [f"present_names={sorted(present_names)}"]
            requests_bucket = rl.get("requests", {})
            if "requests" in rl:
                parts.append(
                    "requests %s/%s reset=%s"
                    % (
                        _format_rate_limit_field(requests_bucket.get("remaining")),
                        _format_rate_limit_field(requests_bucket.get("limit")),
                        _format_rate_limit_duration(requests_bucket.get("reset_seconds")),
                    )
                )
            tokens_bucket = rl.get("tokens", {})
            if "tokens" in rl:
                parts.append(
                    "tokens %s/%s reset=%s"
                    % (
                        _format_rate_limit_field(tokens_bucket.get("remaining")),
                        _format_rate_limit_field(tokens_bucket.get("limit")),
                        _format_rate_limit_duration(tokens_bucket.get("reset_seconds")),
                    )
                )
            logger.info("Rate limits: %s", " | ".join(parts))

            if self._rate_limits_debug_samples and logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Rate limits debug: event_id=%s entry_count=%s malformed_count=%s "
                    "unknown_names=%s sample_entries=%s",
                    self.rate_limits_last_event_id or "unknown",
                    rl_meta.get("entry_count", 0),
                    rl_meta.get("malformed_count", 0),
                    rl_meta.get("unknown_names", []),
                    rl_meta.get("sample_entries", []),
                )
        elif event_type == "session.updated":
            log_session_updated(event, full_payload=True)
            if not self.ready_event.is_set():
                logger.info("Realtime API ready to accept injections.")
                self.ready_event.set()
            self._ensure_startup_injection_timeout_task()

    async def handle_output_item_added(self, event: dict[str, Any]) -> None:
        await self._function_call_accumulator.handle_output_item_added(event)

    async def handle_function_call(self, event: dict[str, Any], websocket: Any) -> None:
        if self.function_call:
            function_name = self.function_call.get("name")
            call_id = self.function_call.get("call_id")
            suppression_notice_sent = False
            try:
                args = json.loads(self.function_call_args) if self.function_call_args else {}
                args_parsed = True
            except json.JSONDecodeError:
                args = {}
                args_parsed = False
            logger.info(
                "Function call received | tool=%s call_id=%s args_parsed=%s",
                function_name,
                call_id,
                args_parsed,
            )
            if self._should_defer_provisional_server_auto_tool_call():
                logger.info(
                    "Function call deferred | tool=%s call_id=%s reason=provisional_server_auto_pre_audio_hold",
                    function_name,
                    call_id,
                )
                await self._send_noop_tool_output(
                    websocket,
                    call_id=call_id,
                    status="deferred",
                    message=(
                        "No action taken. Tool call deferred while transcript-final ownership is still provisional; "
                        "retry after transcript-final handoff."
                    ),
                    tool_name=function_name,
                    reason="provisional_server_auto_pre_audio_hold",
                    category="suppression",
                    include_response_create=False,
                )
                self.function_call = None
                self.function_call_args = ""
                return
            token = getattr(self, "_pending_confirmation_token", None)
            research_fingerprint = (
                self._build_research_args_fingerprint(args)
                if function_name == "perform_research"
                else None
            )
            if function_name == "perform_research" and research_fingerprint is not None:
                research_outcome = self._get_research_permission_outcome(research_fingerprint)
                if research_outcome is False:
                    already_suppressed = research_fingerprint in self._research_suppressed_fingerprints
                    self._research_suppressed_fingerprints[research_fingerprint] = "blocked_by_research_permission"
                    if already_suppressed:
                        logger.info(
                            "Function call outcome: stable suppression replay | tool=%s call_id=%s",
                            function_name,
                            call_id,
                        )
                    else:
                        logger.info(
                            "Function call outcome: suppressed by denied research permission | tool=%s call_id=%s",
                            function_name,
                            call_id,
                        )
                    self._log_structured_noop_event(
                        outcome="blocked_by_research_permission",
                        reason="research_permission_denied",
                        tool_name=function_name,
                        action_id=str(call_id) if call_id is not None else None,
                        token_id=getattr(token, "id", None),
                        idempotency_key=build_normalized_idempotency_key(function_name, args),
                    )
                    await self._send_noop_tool_output(
                        websocket,
                        call_id=call_id,
                        status="blocked_by_research_permission",
                        message="No action taken. Research request was denied for this query; do not retry until user re-approves.",
                        tool_name=function_name,
                        reason="research_permission_denied",
                        category="suppression",
                    )
                    await self._emit_final_noop_user_text(
                        websocket,
                        outcome="blocked_by_research_permission",
                        reason="research_permission_denied",
                    )
                    self.function_call = None
                    self.function_call_args = ""
                    return
            if (
                function_name == "perform_research"
                and ((token is not None and token.kind in {"research_permission", "research_budget"}) or self._pending_research_request is not None)
            ):
                if call_id in self._research_pending_call_ids:
                    logger.debug(
                        "Function call outcome: repeated pending-research suppression | tool=%s call_id=%s",
                        function_name,
                        call_id,
                    )
                else:
                    self._research_pending_call_ids.add(str(call_id))
                    deferred_token_id = token.id if token is not None else ""
                    previous_call_id = None
                    if isinstance(self._deferred_research_tool_call, dict):
                        previous_call_id = str(self._deferred_research_tool_call.get("call_id") or "") or None
                    self._deferred_research_tool_call = {
                        "token_id": deferred_token_id,
                        "tool_name": function_name,
                        "call_id": str(call_id),
                        "args": args,
                        "stored_at": time.monotonic(),
                    }
                    logger.info(
                        "Function call outcome: suppressed pending research confirmation | tool=%s call_id=%s token=%s stored=latest replaced_call_id=%s",
                        function_name,
                        call_id,
                        deferred_token_id or "legacy",
                        previous_call_id or "none",
                    )
                self._log_structured_noop_event(
                    outcome="waiting_for_permission",
                    reason="permission_pending",
                    tool_name=function_name,
                    action_id=str(call_id) if call_id is not None else None,
                    token_id=getattr(token, "id", None),
                    idempotency_key=build_normalized_idempotency_key(function_name, args),
                )
                await self._send_noop_tool_output(
                    websocket,
                    call_id=call_id,
                    status="waiting_for_permission",
                    message="No action taken. Permission required; ask user first.",
                    tool_name=function_name,
                    reason="permission_pending",
                    category="suppression",
                    extra_fields={"token": token.id if token is not None else ""},
                )
                self.function_call = None
                self.function_call_args = ""
                return
            pending_action = token.pending_action if token is not None else self._pending_action
            if pending_action is not None:
                pending_tool = pending_action.action.tool_name
                pending_intent = self._normalize_tool_intent(
                    pending_tool,
                    getattr(pending_action.action, "tool_args", {}),
                )
                incoming_intent = self._normalize_tool_intent(function_name, args)
                if function_name == pending_tool and incoming_intent != pending_intent:
                    previous_action_id = pending_action.action.id
                    previous_token_id = token.id if token is not None else None
                    logger.info(
                        "Function call outcome: replacing pending confirmation intent | tool=%s old_call=%s new_call=%s",
                        function_name,
                        previous_action_id,
                        call_id,
                    )
                    if token is not None and token.pending_action is not None:
                        self._close_confirmation_token(outcome="replaced")
                    else:
                        self._clear_pending_action()
                    self._log_structured_noop_event(
                        outcome="cancelled",
                        reason="replaced_by_new_intent",
                        tool_name=function_name,
                        action_id=previous_action_id,
                        token_id=previous_token_id,
                        idempotency_key=getattr(pending_action, "idempotency_key", None),
                    )
                    if not suppression_notice_sent:
                        await self._emit_final_noop_user_text(
                            websocket,
                            outcome="cancelled",
                            reason="replaced_by_new_intent",
                        )
                        suppression_notice_sent = True
                    if self.orchestration_state.phase == OrchestrationPhase.AWAITING_CONFIRMATION:
                        self.orchestration_state.transition(
                            OrchestrationPhase.IDLE,
                            reason="confirmation replaced",
                        )
                    pending_action = None
                else:
                    if not self._consume_pending_alternate_intent_override(
                        function_name=function_name,
                        args=args,
                        pending_action=pending_action,
                        token=token,
                    ):
                        logger.info(
                            "Function call outcome: suppressed pending confirmation | incoming=%s pending=%s",
                            function_name,
                            pending_tool,
                        )
                        await self._send_awaiting_confirmation_output(call_id, websocket)
                        self.function_call = None
                        self.function_call_args = ""
                        return
            if self._is_duplicate_tool_call(function_name, args):
                logger.info(
                    "Function call outcome: skipped duplicate | tool=%s call_id=%s",
                    function_name,
                    call_id,
                )
                self._log_structured_noop_event(
                    outcome="redundant",
                    reason="duplicate_tool_call",
                    tool_name=function_name,
                    action_id=str(call_id) if call_id is not None else None,
                    token_id=getattr(token, "id", None),
                    idempotency_key=build_normalized_idempotency_key(function_name, args),
                )
                await self._send_noop_tool_output(
                    websocket,
                    call_id=call_id,
                    status="redundant",
                    message="No action taken. Duplicate tool call ignored; use previous tool results.",
                    tool_name=function_name,
                    reason="duplicate_tool_call",
                    category="suppression",
                )
                await self._emit_final_noop_user_text(
                    websocket,
                    outcome="redundant",
                    reason="duplicate_tool_call",
                )
                self.function_call = None
                self.function_call_args = ""
                return
            if self._is_suppressed_after_confirmation_timeout(function_name, args):
                logger.info(
                    "Function call outcome: suppressed confirmation-timeout debounce | tool=%s call_id=%s",
                    function_name,
                    call_id,
                )
                self._log_structured_noop_event(
                    outcome="suppressed_after_confirmation_timeout",
                    reason="confirmation_timeout_debounce",
                    tool_name=function_name,
                    action_id=str(call_id) if call_id is not None else None,
                    token_id=getattr(token, "id", None),
                    idempotency_key=build_normalized_idempotency_key(function_name, args),
                )
                await self._send_noop_tool_output(
                    websocket,
                    call_id=call_id,
                    status="suppressed_after_confirmation_timeout",
                    message="No action taken. Tool call suppressed after confirmation timeout; retry shortly.",
                    tool_name=function_name,
                    reason="confirmation_timeout_debounce",
                    category="suppression",
                )
                await self._emit_final_noop_user_text(
                    websocket,
                    outcome="suppressed_after_confirmation_timeout",
                    reason="confirmation_timeout_debounce",
                )
                self.function_call = None
                self.function_call_args = ""
                return
            dry_run_requested = self._extract_dry_run_flag(args)
            action = self._governance.build_action_packet(
                function_name,
                call_id,
                args,
                reason=f"function_call {function_name}",
            )
            staging = self._stage_action(action)
            if not staging["valid"]:
                await self._reject_tool_call(
                    action,
                    "Argument validation failed.",
                    websocket,
                    staging=staging,
                    status="invalid_arguments",
                )
                return
            runtime_context = self._build_tool_runtime_context(action)
            if hasattr(self._governance, "decide_tool_call"):
                governance_decision = self._governance.decide_tool_call(
                    action,
                    dry_run_requested=dry_run_requested,
                    runtime_cooldown_seconds=self._tool_execution_cooldown_remaining(),
                    runtime_context=runtime_context,
                )
            else:
                governance_decision = self._governance.review(action)
            confirmation_decision = self._normalize_confirmation_decision(
                function_name,
                args,
                governance_decision,
                runtime_context,
            )
            if confirmation_decision.cooldown_seconds > 0 and not confirmation_decision.needs_confirmation:
                await self._reject_tool_call(
                    action,
                    (
                        f"Tool execution paused for "
                        f"{confirmation_decision.cooldown_seconds:.0f}s due to stop word."
                    ),
                    websocket,
                    status="cancelled",
                )
                return
            if dry_run_requested:
                if not confirmation_decision.dry_run_supported:
                    await self._reject_tool_call(
                        action,
                        confirmation_decision.reason,
                        websocket,
                        staging=staging,
                        status="denied",
                    )
                    return
                await self._send_dry_run_output(action, staging, websocket)
                return
            approved_via_prior_permission = (
                confirmation_decision.needs_confirmation
                and self._consume_prior_research_permission_marker_if_fresh(action)
            )
            should_execute = confirmation_decision.approved or approved_via_prior_permission
            if should_execute:
                blocked, blocked_status, blocked_message = self._evaluate_intent_guard(
                    action.tool_name,
                    action.tool_args,
                    phase="execution",
                    idempotency_key=confirmation_decision.idempotency_key,
                )
                if blocked:
                    await self._handle_intent_guard_block(
                        action,
                        websocket,
                        status=blocked_status or "blocked",
                        reason=blocked_status or "intent_guard",
                        message=blocked_message or "Tool execution blocked by intent guard.",
                        staging=staging,
                    )
                    return
                decision_reason = (
                    "approved_via_prior_research_permission"
                    if approved_via_prior_permission
                    else confirmation_decision.reason
                )
                final_execution_decision = "execute"
                if self._debug_governance_decisions:
                    log_info(
                        f"🛡️ Governance decision: {confirmation_decision.status} ({confirmation_decision.reason}) {action.summary()}"
                    )
                if approved_via_prior_permission and self._debug_governance_decisions:
                    log_info(
                        "🛡️ Governance decision: approved "
                        f"(approved_via_prior_research_permission) {action.summary()}"
                    )
                if self._debug_governance_decisions:
                    logger.info(
                        "Function call outcome: executing tool | tool=%s call_id=%s decision_reason=%s idempotency_key=%s",
                        function_name,
                        call_id,
                        decision_reason,
                        confirmation_decision.idempotency_key,
                    )
                self._record_intent_state(
                    action.tool_name,
                    action.tool_args,
                    "approved",
                    idempotency_key=confirmation_decision.idempotency_key,
                )
                self.orchestration_state.transition(
                    OrchestrationPhase.ACT,
                    reason=f"function_call {function_name}",
                )
                await self._execute_action(
                    action,
                    staging,
                    websocket,
                    idempotency_key=confirmation_decision.idempotency_key,
                )
            elif confirmation_decision.needs_confirmation:
                suppressed_outcome = self._suppressed_confirmation_outcome(
                    idempotency_key=confirmation_decision.idempotency_key,
                    cooldown_seconds=float(confirmation_decision.cooldown_seconds or 0.0),
                )
                if suppressed_outcome is not None:
                    guidance = (
                        "Please wait briefly before asking again, or change the request details."
                        if suppressed_outcome == "rejected"
                        else "That confirmation recently expired. Please wait briefly, then ask again if needed."
                    )
                    message = f"No action taken. {guidance}"
                    await self._handle_intent_guard_block(
                        action,
                        websocket,
                        status="suppressed_recent_confirmation_outcome",
                        reason=f"recent_confirmation_{suppressed_outcome}",
                        message=message,
                        staging=staging,
                        include_assistant_message=False,
                    )
                    return
                blocked, blocked_status, blocked_message = self._evaluate_intent_guard(
                    action.tool_name,
                    action.tool_args,
                    phase="confirmation_prompt",
                    idempotency_key=confirmation_decision.idempotency_key,
                )
                if blocked:
                    suppression_reason = blocked_status or ""
                    await self._handle_intent_guard_block(
                        action,
                        websocket,
                        status=blocked_status or "blocked",
                        reason=blocked_status or "intent_guard",
                        message=blocked_message or "Tool confirmation blocked by intent guard.",
                        staging=staging,
                        include_assistant_message=suppression_reason
                        not in {"blocked_recent_denial", "blocked_recent_timeout"},
                    )
                    return
                final_execution_decision = "request_confirmation"
                if self._debug_governance_decisions:
                    log_info(
                        f"🛡️ Governance decision: {confirmation_decision.status} ({confirmation_decision.reason}) {action.summary()}"
                    )
                action.requires_confirmation = True
                action.expiry_ts = time.monotonic() + self._approval_timeout_s
                pending_action = PendingAction(
                    action=action,
                    staging=staging,
                    original_intent=self._last_user_input_text or "",
                    created_at=time.monotonic(),
                    idempotency_key=confirmation_decision.idempotency_key,
                )
                self._record_intent_state(
                    action.tool_name,
                    action.tool_args,
                    "pending",
                    idempotency_key=confirmation_decision.idempotency_key,
                )
                token = self._create_confirmation_token(
                    kind="tool_governance",
                    tool_name=action.tool_name,
                    pending_action=pending_action,
                    expiry_ts=action.expiry_ts,
                    max_retries=pending_action.max_retries,
                    metadata={
                        "approval_flow": True,
                        "max_reminders": confirmation_decision.max_reminders,
                        "reminder_schedule_seconds": list(confirmation_decision.reminder_schedule_seconds),
                        "action_summary": confirmation_decision.action_summary,
                    },
                )
                self._pending_action = pending_action
                self._sync_confirmation_legacy_fields()
                await self._request_tool_confirmation(
                    action,
                    str(confirmation_decision.confirm_reason or confirmation_decision.reason),
                    websocket,
                    staging,
                    action_summary=confirmation_decision.action_summary,
                    confirm_prompt=(
                        str(confirmation_decision.confirm_prompt)
                        if confirmation_decision.confirm_prompt is not None
                        else None
                    ),
                    confirm_reason=(
                        str(confirmation_decision.confirm_reason)
                        if confirmation_decision.confirm_reason is not None
                        else None
                    ),
                )
                token.prompt_sent = True
                self._set_confirmation_state(ConfirmationState.AWAITING_DECISION, reason="tool_prompt_sent")
            else:
                self._record_intent_state(
                    action.tool_name,
                    action.tool_args,
                    "denied",
                    idempotency_key=confirmation_decision.idempotency_key,
                )
                final_execution_decision = "reject"
                if self._debug_governance_decisions:
                    log_info(
                        f"🛡️ Governance decision: {confirmation_decision.status} ({confirmation_decision.reason}) {action.summary()}"
                    )
                await self._reject_tool_call(
                    action,
                    confirmation_decision.reason,
                    websocket,
                    staging=staging,
                    status="denied",
                )
            logger.info(
                "Governance review summary | call_id=%s tool=%s initial_status=%s "
                "initial_reason=%s decision_source=%s thresholds=%s "
                "confirm_required=%s confirm_reason=%s idempotency_key=%s "
                "prior_permission_override=%s final_execution_decision=%s",
                call_id,
                function_name,
                confirmation_decision.status,
                confirmation_decision.reason,
                confirmation_decision.decision_source,
                json.dumps(confirmation_decision.thresholds or {}, separators=(",", ":"), sort_keys=True),
                confirmation_decision.confirm_required,
                confirmation_decision.confirm_reason,
                confirmation_decision.idempotency_key,
                approved_via_prior_permission,
                final_execution_decision,
            )

    async def execute_function_call(
        self,
        function_name: str,
        call_id: str,
        args: dict[str, Any],
        websocket: Any,
        *,
        action: ActionPacket | None = None,
        staging: dict[str, Any] | None = None,
        force_no_tools_followup: bool = False,
        inject_no_tools_instruction: bool = False,
    ) -> None:
        if function_name in function_map:
            try:
                result = await function_map[function_name](**args)
                log_tool_call(function_name, args, result)
                logger.info(
                    "tool_result_received run_id=%s turn_id=%s call_id=%s",
                    self._current_run_id() or "",
                    self._current_turn_id_or_unknown(),
                    call_id,
                )
            except Exception as exc:
                error_message = f"Error executing function '{function_name}': {exc}"
                log_error(error_message)
                result = {"error": error_message}
                self._record_intent_state(function_name, args, "failure")
                await self.send_error_message_to_assistant(error_message, websocket)
        else:
            error_message = (
                f"Function '{function_name}' not found. Add to function_map in tools.py."
            )
            log_error(error_message)
            result = {"error": error_message}
            self._record_intent_state(function_name, args, "failure")
            await self.send_error_message_to_assistant(error_message, websocket)

        self._tool_call_records.append(
            {
                "name": function_name,
                "call_id": call_id,
                "args": args,
                "result": result,
                "turn_id": self._current_turn_id_or_unknown(),
                "action_packet": action.to_payload() if action else None,
                "staging": staging,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
        self._last_tool_call_results = list(self._tool_call_records)
        if function_name == "recall_memories":
            self._clear_preference_recall_candidate()

        output_payload: dict[str, Any] = {"result": result}
        if action:
            output_payload["action_packet"] = action.to_payload()
        if staging is not None:
            output_payload["staging"] = staging

        if function_name == "perform_research" and isinstance(result, dict):
            metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
            status = str(metadata.get("content_fetch_status") or "skipped")
            skip_reason = metadata.get("content_fetch_skip_reason")
            failure_name = metadata.get("content_fetch_error")
            sources = result.get("sources") if isinstance(result.get("sources"), list) else []
            logger.info(
                "[Research] response_grounding research_id=%s content_fetch_status=%s skip_reason=%s failure=%s sources_count=%s",
                result.get("research_id"),
                status,
                skip_reason,
                failure_name,
                len(sources),
            )

        function_call_output = {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(output_payload),
            },
        }
        log_ws_event("Outgoing", function_call_output)
        self._track_outgoing_event(function_call_output)
        transport = self._get_or_create_transport()
        await transport.send_json(websocket, function_call_output)
        logger.info(
            "tool_result_enqueued_to_realtime call_id=%s conversation_item_id=%s",
            call_id,
            None,
        )
        self._mark_utterance_info_summary(deliverable_seen=True)
        if inject_no_tools_instruction:
            await self._add_no_tools_follow_up_instruction(websocket)
        research_id = self._extract_research_id(result) if function_name == "perform_research" else None
        if self._mark_or_suppress_research_spoken_response(research_id):
            self.function_call = None
            self.function_call_args = ""
            return
        response_create_event, tool_followup_canonical_key = self._build_tool_followup_response_create_event(
            call_id=call_id,
            response_create_event={"type": "response.create"},
            tool_name=function_name,
            tool_result_has_distinct_info=self._tool_result_has_distinct_followup_info(
                tool_name=function_name,
                result=result,
            ),
        )
        if force_no_tools_followup:
            response_payload = response_create_event.setdefault("response", {})
            if not isinstance(response_payload, dict):
                response_payload = {}
                response_create_event["response"] = response_payload
            response_payload["tool_choice"] = "none"
        logger.info(
            "tool_followup_response_scheduled call_id=%s canonical_key=%s",
            call_id,
            tool_followup_canonical_key,
        )
        await self._send_response_create(
            websocket,
            response_create_event,
            origin="tool_output",
            debug_context={
                "tool_name": function_name,
                "call_id": call_id,
                "research_id": research_id,
            },
        )

        self.function_call = None
        self.function_call_args = ""

    async def _execute_action(
        self,
        action: ActionPacket,
        staging: dict[str, Any],
        websocket: Any,
        *,
        idempotency_key: str | None = None,
        force_no_tools_followup: bool = False,
        inject_no_tools_instruction: bool = False,
    ) -> None:
        if not staging.get("valid", True):
            await self._reject_tool_call(
                action,
                "Argument validation failed.",
                websocket,
                staging=staging,
                status="invalid_arguments",
            )
            return
        blocked, blocked_status, blocked_message = self._evaluate_intent_guard(
            action.tool_name,
            action.tool_args,
            phase="execution",
            idempotency_key=idempotency_key,
        )
        if blocked:
            await self._handle_intent_guard_block(
                action,
                websocket,
                status=blocked_status or "blocked",
                reason=blocked_status or "intent_guard",
                message=blocked_message or "Tool execution blocked by intent guard.",
                staging=staging,
            )
            return
        if action.tier > 1 and action.id not in self._presented_actions:
            await self._present_action(
                action,
                staging,
                websocket,
                reason="pre-execution presentation",
            )
        await self.execute_function_call(
            action.tool_name,
            action.id,
            action.tool_args,
            websocket,
            action=action,
            staging=staging,
            force_no_tools_followup=force_no_tools_followup,
            inject_no_tools_instruction=inject_no_tools_instruction,
        )
        self._record_executed_tool_call(action.tool_name, action.tool_args)
        self._record_intent_state(
            action.tool_name,
            action.tool_args,
            "executed",
            idempotency_key=idempotency_key,
        )
        self._governance.record_execution(action)
        self._presented_actions.discard(action.id)

    async def _request_tool_confirmation(
        self,
        action: ActionPacket,
        reason: str,
        websocket: Any,
        staging: dict[str, Any],
        *,
        action_summary: str | None = None,
        confirm_prompt: str | None = None,
        confirm_reason: str | None = None,
    ) -> None:
        summary = action.summary()
        tool_metadata = self._governance.describe_tool(action.tool_name)
        logger.info(
            "Entering AWAITING_CONFIRMATION tool=%s args=%s",
            action.tool_name,
            json.dumps(action.tool_args, sort_keys=True),
        )
        self.orchestration_state.transition(
            OrchestrationPhase.AWAITING_CONFIRMATION,
            reason=f"needs confirmation for {action.tool_name}",
        )
        self._awaiting_confirmation_completion = False
        token = getattr(self, "_pending_confirmation_token", None)
        if token is not None:
            service = getattr(self, "_confirmation_service", None)
            if service is not None:
                service.start_pending(token, token.pending_action, time.monotonic())
        message = self._build_approval_prompt(
            action,
            action_summary=action_summary,
            confirm_prompt=confirm_prompt,
            confirm_reason=confirm_reason,
        )
        latch_key = self._confirmation_prompt_latch_key(token)
        if token is not None and latch_key is not None and latch_key in self._pending_confirmation_prompt_latches:
            logger.info(
                "CONFIRMATION_PROMPT_SUPPRESSED reason=latched token=%s latch_key=%s",
                token.id,
                latch_key,
            )
            await self._maybe_emit_confirmation_reminder(websocket, reason="confirmation_prompt_latched")
            return
        response_metadata = {"trigger": "confirmation_prompt", "approval_flow": "true"}
        if token is not None:
            response_metadata["confirmation_token"] = token.id
        await self.send_assistant_message(
            message,
            websocket,
            response_metadata=response_metadata,
        )
        if latch_key is not None:
            self._pending_confirmation_prompt_latches.add(latch_key)
        function_call_output = {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": action.id,
                "output": json.dumps(
                    {
                        "status": "awaiting_confirmation",
                        "message": message,
                        "tool": action.tool_name,
                        "tool_metadata": tool_metadata,
                        "action_packet": action.to_payload(),
                        "staging": staging,
                        "summary": summary,
                        "reason": reason,
                    }
                ),
            },
        }
        log_ws_event("Outgoing", function_call_output)
        self._track_outgoing_event(function_call_output)
        transport = self._get_or_create_transport()
        await transport.send_json(websocket, function_call_output)
        # Do not request another model response here.
        # send_assistant_message() above already produced the spoken approval prompt,
        # and a second response.create can get deferred during AWAITING_CONFIRMATION,
        # then replayed later as an unrelated extra spoken response.
        self._presented_actions.add(action.id)
        self.function_call = None
        self.function_call_args = ""

    async def _send_awaiting_confirmation_output(self, call_id: str, websocket: Any) -> None:
        if not self._pending_action:
            return
        pending = self._pending_action
        action = pending.action
        self._log_structured_noop_event(
            outcome="awaiting_confirmation",
            reason="pending_confirmation",
            tool_name=action.tool_name,
            action_id=action.id,
            token_id=getattr(getattr(self, "_pending_confirmation_token", None), "id", None),
            idempotency_key=getattr(pending, "idempotency_key", None),
        )
        await self._send_noop_tool_output(
            websocket,
            call_id=call_id,
            status="awaiting_confirmation",
            message="No action taken. Awaiting explicit yes/no confirmation for pending action.",
            tool_name=action.tool_name,
            reason="pending_confirmation",
            category="suppression",
            action=action,
            staging=pending.staging,
        )

    async def _present_action(
        self,
        action: ActionPacket,
        staging: dict[str, Any],
        websocket: Any,
        *,
        reason: str | None = None,
    ) -> None:
        summary = action.summary()
        function_call_output = {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": action.id,
                "output": json.dumps(
                    {
                        "status": "presented",
                        "tool": action.tool_name,
                        "tool_metadata": self._governance.describe_tool(action.tool_name),
                        "action_packet": action.to_payload(),
                        "staging": staging,
                        "summary": summary,
                        "reason": reason,
                    }
                ),
            },
        }
        log_ws_event("Outgoing", function_call_output)
        self._track_outgoing_event(function_call_output)
        transport = self._get_or_create_transport()
        await transport.send_json(websocket, function_call_output)
        response_create_event, tool_followup_canonical_key = self._build_tool_followup_response_create_event(
            call_id=action.id,
            response_create_event={"type": "response.create"},
        )
        logger.info(
            "tool_followup_response_scheduled call_id=%s canonical_key=%s",
            action.id,
            tool_followup_canonical_key,
        )
        await self._send_response_create(websocket, response_create_event, origin="tool_output")
        self._presented_actions.add(action.id)

    async def _send_dry_run_output(
        self,
        action: ActionPacket,
        staging: dict[str, Any],
        websocket: Any,
    ) -> None:
        summary = action.summary()
        dry_run_payload = {
            "requested": True,
            "summary": summary,
            "valid": staging.get("valid", True),
            "staging": staging,
        }
        function_call_output = {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": action.id,
                "output": json.dumps(
                    {
                        "status": "dry_run",
                        "message": "Dry-run requested; no execution performed.",
                        "tool": action.tool_name,
                        "tool_metadata": self._governance.describe_tool(action.tool_name),
                        "action_packet": action.to_payload(),
                        "staging": staging,
                        "summary": summary,
                        "dry_run": dry_run_payload,
                    }
                ),
            },
        }
        log_ws_event("Outgoing", function_call_output)
        self._track_outgoing_event(function_call_output)
        transport = self._get_or_create_transport()
        await transport.send_json(websocket, function_call_output)
        response_create_event, tool_followup_canonical_key = self._build_tool_followup_response_create_event(
            call_id=action.id,
            response_create_event={"type": "response.create"},
        )
        logger.info(
            "tool_followup_response_scheduled call_id=%s canonical_key=%s",
            action.id,
            tool_followup_canonical_key,
        )
        await self._send_response_create(websocket, response_create_event, origin="tool_output")
        self.function_call = None
        self.function_call_args = ""

    async def _reject_tool_call(
        self,
        action: ActionPacket,
        reason: str,
        websocket: Any,
        *,
        staging: dict[str, Any] | None = None,
        status: str = "denied",
        include_assistant_message: bool = True,
    ) -> None:
        packet = self._format_action_packet(action)
        message = f"No action taken.\n{packet}\nReason: {reason}."
        self._log_structured_noop_event(
            outcome=status,
            reason=reason,
            tool_name=action.tool_name,
            action_id=action.id,
            token_id=getattr(getattr(self, "_pending_confirmation_token", None), "id", None),
            idempotency_key=self._idempotency_key_for_action(action),
        )
        if include_assistant_message:
            await self.send_assistant_message(message, websocket)
        await self._send_noop_tool_output(
            websocket,
            call_id=action.id,
            status=status,
            message=message,
            tool_name=action.tool_name,
            reason=reason,
            category="rejection",
            action=action,
            staging=staging,
            include_response_create=not include_assistant_message,
        )
        self._presented_actions.discard(action.id)
        self.function_call = None
        self.function_call_args = ""

    async def send_image_to_assistant(self, new_image: Any) -> None:
        if self.websocket:
            allowed, reason = self._can_accept_external_stimulus(
                "camera",
                "image",
                priority="normal",
            )
            if not allowed:
                self._log_suppressed_stimulus("camera", "image", reason)
                return
            bytes_buffer = BytesIO()
            new_image.save(bytes_buffer, format="JPEG", quality=55, optimize=True)
            encoded_image = base64.b64encode(bytes_buffer.getvalue()).decode("utf-8")
            image_item = {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{encoded_image}",
                        }
                    ],
                },
            }
            log_ws_event("Image", image_item)
            transport = self._get_or_create_transport()
            await transport.send_json(self.websocket, image_item)
            self._last_vision_frame_sent_at_monotonic = time.monotonic()
            if self._image_response_enabled:
                await self._stimuli_coordinator.enqueue(
                    trigger="image_message",
                    metadata={"image_format": "jpeg"},
                    priority=self._get_injection_priority("image_message"),
                )
            else:
                logger.debug("Skipping image injection response (mode=%s).", self._image_response_mode)
        else:
            log_warning("Unable to send image to assistant, websocket not available")

    async def send_assistant_message(
        self,
        message: str,
        websocket: Any,
        *,
        speak: bool = True,
        response_metadata: dict[str, Any] | None = None,
        utterance_context: UtteranceContext | None = None,
    ) -> None:
        metadata = {"origin": "assistant_message"}
        if response_metadata:
            metadata.update(response_metadata)
        context_hint = utterance_context or getattr(self, "_utterance_context", None) or self._build_utterance_context()
        metadata.setdefault("turn_id", context_hint.turn_id)
        if context_hint.input_event_key:
            metadata.setdefault("input_event_key", context_hint.input_event_key)
        message = self._normalize_verify_clarify_message(message=message, metadata=metadata)
        assistant_item = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": message}],
            },
        }
        log_ws_event("Outgoing", assistant_item)
        transport = self._get_or_create_transport()
        await transport.send_json(websocket, assistant_item)
        if not speak:
            return
        trigger_reason = str(metadata.get("trigger") or "").strip().lower()
        explicit_parent_key = str(metadata.get("input_event_key") or "").strip()
        background_behavior_allowed = str(metadata.get("background_behavior_allowed", "")).strip().lower() in {
            "1",
            "true",
            "yes",
        }
        if not explicit_parent_key and not background_behavior_allowed:
            logger.info(
                "assistant_message_orphan_guarded run_id=%s turn_id=%s origin=assistant_message",
                self._current_run_id() or "",
                metadata.get("turn_id") or "turn-unknown",
            )
            return
        token = getattr(self, "_pending_confirmation_token", None)
        if token is not None or self._is_awaiting_confirmation_phase():
            metadata.setdefault("approval_flow", "true")
            if token is not None:
                metadata.setdefault("confirmation_token", token.id)
        trace_turn_id = str(metadata.get("turn_id") or self._current_turn_id_or_unknown())
        trace_input_event_key = str(metadata.get("input_event_key") or "").strip()
        trace_context = self._build_utterance_context(
            turn_id=trace_turn_id,
            input_event_key=trace_input_event_key,
            utterance_seq=context_hint.utterance_seq,
        )
        trace_canonical_key = trace_context.canonical_key
        trigger_reason = str(metadata.get("trigger") or "assistant_message")
        obligation_present = self._response_obligation_key(
            turn_id=trace_turn_id,
            input_event_key=trace_input_event_key,
        ) in getattr(self, "_response_obligations", {})
        logger.debug(
            "[RESPTRACE] response_create_request run_id=%s turn_id=%s origin=%s trigger_reason=%s input_event_key=%s "
            "canonical_key=%s sent_now=false scheduled=false replaced=false queue_len=%s pending_server_auto_len=%s "
            "suppression_active=%s suppression_reason=%s obligation_present=%s response_done_serial=%s",
            self._current_run_id() or "",
            trace_turn_id,
            "assistant_message",
            trigger_reason,
            trace_input_event_key or "unknown",
            trace_canonical_key,
            len(getattr(self, "_response_create_queue", deque()) or ()),
            len(getattr(self, "_pending_server_auto_input_event_keys", deque()) or ()),
            bool(trace_turn_id in getattr(self, "_preference_recall_suppressed_turns", set())),
            self._resptrace_suppression_reason(turn_id=trace_turn_id, input_event_key=trace_input_event_key),
            obligation_present,
            getattr(self, "_response_done_serial", 0),
        )
        await self._send_response_create(
            websocket,
            self._bounded_clarify_response_create_event(message=message, metadata=metadata),
            origin="assistant_message",
            utterance_context=trace_context,
        )

    async def send_error_message_to_assistant(self, error_message: str, websocket: Any) -> None:
        error_item = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": error_message}],
            },
        }
        log_ws_event("Outgoing", error_item)
        transport = self._get_or_create_transport()
        await transport.send_json(websocket, error_item)

    async def handle_transcribe_response_done(self) -> None:
        await self._response_terminal_handlers_module().handle_transcribe_response_done()

    async def handle_audio_response_done(self) -> None:
        await self._response_terminal_handlers_module().handle_audio_response_done()

    async def handle_response_done(self, event: dict[str, Any] | None = None) -> None:
        await self._response_terminal_handlers_module().handle_response_done(event)

    async def handle_response_completed(self, event: dict[str, Any] | None = None) -> None:
        await self._response_terminal_handlers_module().handle_response_completed(event)

    async def handle_error(self, event: dict[str, Any], websocket: Any) -> None:
        error_message = event.get("error", {}).get("message", "")
        if "no active response found" in error_message.lower():
            logger.debug(
                "response_cancel_noop run_id=%s turn_id=%s reason=no_active_response",
                self._current_run_id() or "",
                self._current_turn_id_or_unknown(),
            )
            logger.debug("Ignoring cancellation error with no active response: %s", error_message)
            return

        log_error(f"Error: {error_message}")
        if "buffer is empty" in error_message:
            logger.info("Received 'buffer is empty' error, no audio data sent.")
        elif "Conversation already has an active response" in error_message:
            logger.warning("Received 'active response' error despite response.create serialization.")
            self.response_in_progress = True
            self._response_in_flight = True
        else:
            logger.error("Unhandled error: %s", error_message)

    async def handle_speech_stopped(self, websocket: Any) -> None:
        self.mic.stop_recording()
        logger.info("Speech ended, processing...")
        self.response_start_time = time.perf_counter()
        self._response_start_monotonic = time.monotonic()

    async def send_initial_prompts(self, websocket: Any) -> None:
        logger.info("Sending %s prompts: %s", len(self.prompts), self.prompts)
        if self.prompts and len(self.prompts) == 1:
            startup_prompt = self.prompts[0]
            if await self._maybe_process_research_intent(
                startup_prompt,
                websocket,
                source="startup_prompt",
            ):
                self._record_user_input(startup_prompt, source="startup_prompt")
                return

        content = [{"type": "input_text", "text": prompt} for prompt in self.prompts]
        if self.prompts:
            self._record_user_input(self.prompts[-1], source="startup_prompt")
        event = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": content,
            },
        }
        log_ws_event("Outgoing", event)
        self._track_outgoing_event(event)
        transport = self._get_or_create_transport()
        await transport.send_json(websocket, event)

        if not self._allow_ai_call("startup_prompt"):
            logger.info("Skipping startup response: AI call budget exhausted.")
            return
        memory_brief_note = self._consume_pending_memory_brief_note()
        response_create_event = {"type": "response.create"}
        await self._send_response_create(
            websocket,
            response_create_event,
            origin="prompt",
            record_ai_call=True,
            memory_brief_note=memory_brief_note,
        )

    async def send_text_message_to_conversation(
        self,
        text_message: str,
        request_response: bool = True,
        *,
        bypass_response_suppression: bool = False,
        safety_override: bool = False,
        injection_metadata: dict[str, Any] | None = None,
    ) -> None:
        self._record_user_input(text_message, source="text_message")
        if await self._handle_stop_word(text_message, self.websocket, source="text_message"):
            return
        if await self._maybe_handle_approval_response(text_message, self.websocket):
            return
        if await self._maybe_handle_research_permission_response(text_message, self.websocket):
            return
        if await self._maybe_handle_research_budget_response(text_message, self.websocket):
            return
        if await self._maybe_apply_late_confirmation_decision(text_message, self.websocket):
            return
        if await self._maybe_handle_preference_recall_intent(
            text_message,
            self.websocket,
            source="text_message",
        ):
            return
        if await self._maybe_process_research_intent(
            text_message,
            self.websocket,
            source="text_message",
        ):
            return
        self.orchestration_state.transition(
            OrchestrationPhase.SENSE,
            reason="text message",
        )
        text_event = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text_message}],
            },
        }
        log_ws_event("Outgoing", text_event)
        self._track_outgoing_event(text_event)
        transport = self._get_or_create_transport()
        await transport.send_json(self.websocket, text_event)
        if request_response:
            enqueue_metadata = {
                "text_length": len(text_message),
                "bypass_limits": bypass_response_suppression,
                "safety_override": safety_override,
            }
            if injection_metadata:
                enqueue_metadata["source"] = str(injection_metadata.get("source") or "").strip().lower()
                enqueue_metadata["kind"] = str(injection_metadata.get("kind") or "").strip().lower()
                enqueue_metadata["priority"] = str(injection_metadata.get("priority") or "").strip().lower()
                enqueue_metadata["severity"] = str(injection_metadata.get("severity") or "").strip().lower()
            await self._stimuli_coordinator.enqueue(
                trigger="text_message",
                metadata=enqueue_metadata,
                priority=self._get_injection_priority("text_message"),
            )

    def _mark_first_assistant_utterance_observed_if_needed(self, delta: str) -> None:
        coordinator = self._startup_injection_coordinator()
        if delta and not coordinator.first_assistant_utterance_observed:
            coordinator.first_assistant_utterance_observed = True
            self._release_startup_injection_gate(reason="first_turn_complete")

    def _is_critical_sensor_trigger(self, source: str, severity: str) -> bool:
        normalized_source = str(source).strip().lower()
        normalized_severity = str(severity).strip().lower()
        return normalized_severity == "critical" and normalized_source in {"battery", "imu"}

    def _sensor_aggregation_key(self, source: str, severity: str) -> str:
        normalized_source = str(source).strip().lower() or "unknown"
        normalized_severity = str(severity).strip().lower() or "unknown"
        return f"{normalized_source}:{normalized_severity}"

    async def _should_defer_sensor_response(self, trigger: str, metadata: dict[str, Any]) -> bool:
        if trigger != "text_message":
            self._sensor_event_aggregation_metrics["immediate"] += 1
            return False

        source = str(metadata.get("source") or "").strip().lower()
        severity = str(metadata.get("severity") or "").strip().lower()
        if source not in self._sensor_event_aggregate_sources:
            self._sensor_event_aggregation_metrics["immediate"] += 1
            return False
        if self._is_critical_sensor_trigger(source, severity):
            self._sensor_event_aggregation_metrics["immediate"] += 1
            return False

        if self._sensor_event_aggregation_window_s <= 0.0:
            self._sensor_event_aggregation_metrics["immediate"] += 1
            return False

        aggregate_key = self._sensor_aggregation_key(source, severity)
        now = time.monotonic()
        async with self._sensor_event_aggregation_lock:
            if metadata.get("sensor_aggregate_summary"):
                return False
            payload_count = int(metadata.get("event_count") or 1)
            window = self._sensor_event_aggregation_windows.get(aggregate_key)
            if window is None:
                self._sensor_event_aggregation_windows[aggregate_key] = SensorAggregationWindow(
                    source=source,
                    severity=severity or "unknown",
                    trigger=trigger,
                    first_seen_monotonic=now,
                    last_seen_monotonic=now,
                    total_events=payload_count,
                    dropped_events=0,
                    payload_count=1,
                    latest_metadata=dict(metadata),
                )
                task = self._sensor_event_aggregation_tasks.get(aggregate_key)
                if task is None or task.done():
                    self._sensor_event_aggregation_tasks[aggregate_key] = self._runtime_task_registry().spawn(
                        f"sensor_aggregate_flush.{aggregate_key}",
                        self._flush_sensor_aggregation_window(aggregate_key),
                    )
            else:
                window.last_seen_monotonic = now
                window.total_events += payload_count
                window.payload_count += 1
                window.latest_metadata = dict(metadata)
                window.dropped_events += payload_count
                self._sensor_event_aggregation_metrics["dropped"] += payload_count
            return True

    async def _flush_sensor_aggregation_window(self, aggregate_key: str) -> None:
        await asyncio.sleep(self._sensor_event_aggregation_window_s)
        async with self._sensor_event_aggregation_lock:
            window = self._sensor_event_aggregation_windows.pop(aggregate_key, None)
            self._sensor_event_aggregation_tasks.pop(aggregate_key, None)
        if window is None:
            return
        summary_metadata = {
            "source": window.source,
            "severity": window.severity,
            "sensor_aggregate_summary": True,
            "event_count": max(1, int(window.total_events)),
            "dropped_events": max(0, int(window.dropped_events)),
            "window_s": self._sensor_event_aggregation_window_s,
            "bypass_limits": bool((window.latest_metadata or {}).get("bypass_limits", False)),
            "safety_override": bool((window.latest_metadata or {}).get("safety_override", False)),
        }
        self._sensor_event_aggregation_metrics["coalesced"] += 1
        await self.maybe_request_response(window.trigger, summary_metadata)

    async def maybe_request_response(self, trigger: str, metadata: dict[str, Any]) -> None:
        await self._injection_bus_instance().maybe_request_response(trigger, metadata)


    def _get_injection_priority(self, trigger: str) -> int:
        trigger_config = self._injection_response_triggers.get(trigger, {})
        return int(trigger_config.get("priority", 0))

    def _queue_pending_image_stimulus(self, trigger: str, metadata: dict[str, Any]) -> None:
        self._pending_image_stimulus = {"trigger": trigger, "metadata": metadata}
        self._pending_image_flush_after_playback = False
        logger.info("Queued pending image stimulus while response is in progress.")

    def _schedule_pending_image_flush(self, reason: str) -> None:
        if not self._pending_image_stimulus:
            return
        if self.loop and self.loop.is_running():
            self._runtime_task_registry().spawn(
                f"pending_image_flush.{reason.replace(' ', '_')}",
                self._flush_pending_image_stimulus(reason),
            )
        else:
            logger.warning(
                "Unable to flush pending image stimulus after %s: event loop unavailable.",
                reason,
            )

    async def _flush_pending_image_stimulus(self, reason: str) -> None:
        if not self._pending_image_stimulus:
            return
        if self._has_active_confirmation_token():
            logger.info(
                "Deferring pending image stimulus flush after %s: awaiting confirmation unresolved.",
                reason,
            )
            return
        if self.response_in_progress:
            logger.info(
                "Deferring pending image stimulus flush after %s: response in progress.",
                reason,
            )
            return
        pending = self._pending_image_stimulus
        self._pending_image_stimulus = None
        self._pending_image_flush_after_playback = False
        logger.info("Flushing pending image stimulus after %s.", reason)
        await self.maybe_request_response(pending["trigger"], pending["metadata"])

    async def send_audio_loop(self, websocket: Any) -> None:
        try:
            while not self.exit_event.is_set():
                if self.mic.is_receiving:
                    await asyncio.sleep(0.05)
                    continue

                now = time.monotonic()
                if now < self.mic_send_suppress_until:
                    self.mic.drain_queue()
                    await asyncio.sleep(0.05)
                    continue

                audio_data = self.mic.get_audio_data()
                if audio_data:
                    rms, peak = self._sample_audio_levels(audio_data)
                    self._recent_input_levels.append(
                        {"t": time.monotonic(), "rms": rms, "peak": peak}
                    )
                    base64_audio = base64_encode_audio(audio_data)
                    if base64_audio:
                        audio_event = {
                            "type": "input_audio_buffer.append",
                            "audio": base64_audio,
                        }
                        transport = self._get_or_create_transport()
                        await transport.send_json(websocket, audio_event)
                        logger.debug(
                            "input_audio_append_sent bytes=%s rms=%s peak=%s",
                            len(audio_data),
                            rms,
                            peak,
                        )
                    else:
                        logger.debug("Failed to encode audio data for sending")

                await asyncio.sleep(0.03)

        except asyncio.CancelledError:
            logger.info("Audio send loop cancelled. Closing the connection.")
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Closing the connection.")
        finally:
            self.exit_event.set()
            await self._close_websocket("audio loop exiting", websocket=websocket)

    def shutdown_handler(self) -> None:
        if not self._shutdown_coordinator().request_shutdown():
            return

        logger.info("Received shutdown signal. Initiating graceful shutdown...")
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self._request_shutdown)

    def _request_shutdown(self) -> None:
        self.exit_event.set()
        coordinator = self._shutdown_coordinator()
        self._runtime_task_registry().cancel_all("request_shutdown")
        close_task = None
        if self.websocket:
            close_task = self.loop.create_task(self._close_websocket("signal"))
        coordinator.cancel_tasks(
            asyncio.all_tasks(self.loop),
            exclude_current=True,
            exclude_tasks=[close_task] if close_task else None,
        )

    async def _close_websocket(
        self,
        reason: str,
        *,
        websocket: Any | None = None,
        timeout_s: float | None = None,
    ) -> None:
        await self._shutdown_coordinator().close_websocket(
            websocket or self.websocket,
            reason=reason,
            timeout_s=timeout_s,
        )
