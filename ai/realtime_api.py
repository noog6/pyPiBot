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
from ai.tools import function_map, tools
from ai.interaction_lifecycle_controller import (
    InteractionLifecycleController,
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
    render_realtime_memory_brief_item,
    render_startup_memory_digest_item,
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
    "which",
)
_MEMORY_INTENT_MARKERS = (
    "remember",
    "memory",
    "memories",
    "recall",
    "do you know",
    "what's my name",
    "what is my name",
    "my name",
    "prefer",
    "preferred",
    "preference",
    "favorite",
    "favourite",
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
_PREFERENCE_TAG_FALLBACK_QUERIES = ("preference", "favorite", "preferred")
_MICRO_ACK_CONFIRMATION_ALLOWLIST: frozenset[str] = frozenset(
    {
        "watchdog_confirmation_pending",
        "watchdog_permission_pending",
        "watchdog_approval_pending",
    }
)
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


@dataclass
class PendingAction:
    action: ActionPacket
    staging: dict[str, Any]
    original_intent: str
    created_at: float
    idempotency_key: str | None = None
    retry_count: int = 0
    max_retries: int = 2


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
class UtteranceContext:
    turn_id: str
    input_event_key: str
    canonical_key: str
    utterance_seq: int


@dataclass
class CanonicalResponseState:
    created: bool = False
    audio_started: bool = False
    done: bool = False
    cancel_sent: bool = False
    origin: str = "unknown"
    response_id: str = ""
    obligation_present: bool = False
    input_event_key: str = ""
    turn_id: str = ""
    obligation: dict[str, Any] | None = None


class ConfirmationState(str, Enum):
    IDLE = "idle"
    PENDING_PROMPT = "pending_prompt"
    AWAITING_DECISION = "awaiting_decision"
    RESOLVING = "resolving"
    COMPLETED = "completed"


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
class PendingResponseCreate:
    websocket: Any
    event: dict[str, Any]
    origin: str
    turn_id: str
    created_at: float
    reason: str
    record_ai_call: bool = False
    debug_context: dict[str, Any] | None = None
    memory_brief_note: str | None = None
    queued_reminder_key: str | None = None
    enqueued_done_serial: int = 0


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
        self._shutdown_lock = threading.Lock()
        self._shutdown_requested = False

        self.assistant_reply = ""
        self._assistant_reply_accum = ""
        self._audio_accum = bytearray()
        self._audio_accum_bytes_target = 9600
        self.response_in_progress = False
        self._response_in_flight = False
        self._response_create_queue: deque[dict[str, Any]] = deque()
        self._pending_response_create: PendingResponseCreate | None = None
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
        self._warned_missing_rate_limit_buckets: set[str] = set()
        self._last_rate_limit_present_buckets: set[str] = set()
        self.response_start_time: float | None = None
        self.websocket = None
        self._ws_close_lock = asyncio.Lock()
        self._ws_close_started = False
        self._ws_close_done = False
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
        self._confirmation_last_closed_token: dict[str, Any] | None = None
        self._confirmation_timeout_debounce_window_s = float(
            config.get("confirmation_timeout_debounce_window_s", 5.0)
        )
        self._confirmation_timeout_markers: dict[str, float] = {}
        self._confirmation_timeout_causes: dict[str, str] = {}
        self._recent_confirmation_outcomes: dict[str, dict[str, Any]] = {}
        self._confirmation_transition_lock: asyncio.Lock | None = None
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
        self._missing_requests_bucket_warning_interval = 3
        self._missing_requests_bucket_consecutive_updates = 0
        self._last_connect_time: float | None = None
        self._last_disconnect_reason: str | None = None
        self._last_failure_reason: str | None = None
        self._memory_retrieval_error_throttle_s = 60.0
        self._memory_retrieval_last_error_log_at = 0.0
        self._memory_retrieval_suppressed_errors = 0
        self._preference_recall_cooldown_s = self._memory_retrieval_cooldown_s
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
        self._pending_server_auto_input_event_keys: deque[str] = deque(maxlen=64)
        self._active_server_auto_input_event_key: str | None = None
        self._current_input_event_key: str | None = None
        self._active_input_event_key_by_turn_id: dict[str, str] = {}
        self._input_event_key_counter = 0
        self._synthetic_input_event_counter = 0
        self._response_created_canonical_keys: set[str] = set()
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
        self._active_response_input_event_key: str | None = None
        self._active_response_canonical_key: str | None = None
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
        self._debug_vad = bool(realtime_cfg.get("debug_vad", False))
        self._awaiting_confirmation_allowed_sources = (
            self._load_awaiting_confirmation_source_policy(realtime_cfg)
        )
        self._utterance_counter = 0
        self._active_utterance: dict[str, Any] | None = None
        self._recent_input_levels: deque[dict[str, float]] = deque(maxlen=256)
        self._transcript_response_watchdog_timeout_s = max(
            0.5,
            float(realtime_cfg.get("transcript_response_watchdog_timeout_s", 3.0)),
        )
        self._transcript_response_watchdog_tasks: dict[str, asyncio.Task] = {}
        self._transcript_response_outcome_logged_keys: set[str] = set()
        self._minimum_non_confirmation_duration_ms = int(
            realtime_cfg.get("minimum_non_confirmation_duration_ms", 120)
        )
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
        self._micro_ack_manager = self._build_micro_ack_manager(realtime_cfg)

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
    ) -> None:
        run_id = self._current_run_id() or ""
        category_value = category or ""
        channel_value = channel or ""
        intent_value = intent or ""
        action_value = action or ""
        tool_call_id_value = tool_call_id or ""
        if event == "scheduled":
            logger.info(
                "micro_ack_scheduled run_id=%s turn_id=%s reason=%s delay_ms=%s category=%s channel=%s intent=%s action=%s tool_call_id=%s",
                run_id,
                turn_id,
                reason,
                delay_ms if delay_ms is not None else "",
                category_value,
                channel_value,
                intent_value,
                action_value,
                tool_call_id_value,
            )
            return
        if event == "emitted":
            logger.info(
                "micro_ack_emitted run_id=%s turn_id=%s phrase_id=%s category=%s channel=%s intent=%s action=%s tool_call_id=%s",
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
        logger.debug(
            "micro_ack_suppressed run_id=%s turn_id=%s reason=%s category=%s channel=%s intent=%s action=%s tool_call_id=%s",
            run_id,
            turn_id,
            reason,
            category_value,
            channel_value,
            intent_value,
            action_value,
            tool_call_id_value,
        )

    def _micro_ack_suppression_reason(self) -> str | None:
        manager = getattr(self, "_micro_ack_manager", None)
        if manager is None:
            return "disabled"
        baseline_reason = manager.suppression_baseline_reason()
        if baseline_reason:
            return baseline_reason
        pending_reason = str(getattr(self, "_pending_micro_ack_reason", "") or "").strip().lower()
        if (
            (self._has_active_confirmation_token() or self._is_awaiting_confirmation_phase())
            and pending_reason not in _MICRO_ACK_CONFIRMATION_ALLOWLIST
        ):
            return "confirmation_pending"
        current_state = getattr(getattr(self, "state_manager", None), "state", None)
        if current_state == InteractionState.LISTENING:
            return "listening_state"
        if bool(getattr(self, "_confirmation_speech_active", False)):
            return "speech_active"
        return None

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
        metadata = self._micro_ack_correlation_metadata()
        metadata_intent = str(metadata.get("intent") or metadata.get("normalized_intent") or "").strip() or None
        metadata_action = str(metadata.get("action") or metadata.get("trigger") or "").strip() or None
        metadata_tool_call_id = str(metadata.get("tool_call_id") or "").strip() or None
        canonical_key = str(getattr(self, "_active_response_canonical_key", "") or "").strip() or None

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
        try:
            manager.maybe_schedule(
                context=context,
                reason=reason,
                loop=self.loop,
                expected_delay_ms=expected_delay_ms,
            )
        finally:
            self._pending_micro_ack_reason = None

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
        manager = getattr(self, "_micro_ack_manager", None)
        if manager is None:
            return
        manager.cancel(turn_id=turn_id, reason=reason)

    def _emit_micro_ack(self, context: MicroAckContext, phrase_id: str, phrase: str) -> None:
        websocket = getattr(self, "websocket", None)
        if websocket is None or self.loop is None:
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
        logger.info(
            "response_not_scheduled run_id=%s turn_id=%s input_event_key=%s reason=%s details=%s",
            self._current_run_id() or "",
            turn_id,
            normalized_key,
            reason or "unknown",
            details or "",
        )
        self._log_response_site_debug(
            site="response_not_scheduled",
            turn_id=turn_id,
            input_event_key=normalized_key,
            canonical_key=self._canonical_utterance_key(turn_id=turn_id, input_event_key=normalized_key),
            origin=str(getattr(self, "_active_response_origin", "") or "unknown").strip() or "unknown",
            trigger=reason or "unknown",
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
            response_in_flight=bool(getattr(self, "_response_in_flight", False)),
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
        watchdog_tasks[normalized_key] = asyncio.create_task(
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
        self._alert_policy.emit(self.event_bus, alert)

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
        self._expire_confirmation_awaiting_decision_timeout()
        phase = getattr(self.orchestration_state, "phase", None)
        phase_name = getattr(phase, "value", str(phase))
        normalized_source = str(source).lower()
        normalized_kind = self._normalize_external_stimulus_kind(kind, metadata)
        normalized_priority = (priority or "").lower()

        pending_action = getattr(self, "_pending_confirmation_token", None)
        response_in_progress = bool(getattr(self, "response_in_progress", False))

        if (
            pending_action is not None
            and self._is_awaiting_confirmation_phase()
            and not self._is_allowed_awaiting_confirmation_stimulus(
                normalized_source,
                normalized_kind,
                normalized_priority,
            )
        ):
            return False, "awaiting_confirmation_policy"

        if response_in_progress:
            return False, "response_in_progress"

        state = getattr(self.state_manager, "state", None)
        if state not in (InteractionState.IDLE, InteractionState.LISTENING):
            state_name = getattr(state, "value", str(state))
            return False, f"interaction_state={state_name}"

        return True, f"phase={phase_name}"

    def _normalize_external_stimulus_kind(
        self,
        kind: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        if metadata:
            severity = metadata.get("severity")
            if isinstance(severity, str) and severity.strip():
                return severity.strip().lower()
        return str(kind).strip().lower() or "unknown"

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

    def get_session_health(self) -> dict[str, Any]:
        retrieval_metrics = self._memory_manager.get_retrieval_health_metrics(
            scope=self._memory_retrieval_scope,
            session_id=self._memory_manager.get_active_session_id(),
        )
        return {
            "connected": self._session_connected,
            "ready": self.ready_event.is_set(),
            "connection_attempts": self._session_connection_attempts,
            "connections": self._session_connections,
            "reconnects": self._session_reconnects,
            "failures": self._session_failures,
            "last_connect_time": self._last_connect_time or 0.0,
            "last_disconnect_reason": self._last_disconnect_reason or "",
            "last_failure_reason": self._last_failure_reason or "",
            "memory_retrieval": retrieval_metrics,
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
        consumes_canonical_slot = self._response_consumes_canonical_slot(metadata)
        pending_origin = {
            "origin": normalized_origin,
            "micro_ack": "true"
            if isinstance(metadata, dict) and str(metadata.get("micro_ack", "")).strip().lower() == "true"
            else "false",
            "consumes_canonical_slot": "true" if consumes_canonical_slot else "false",
        }
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
        self._log_injection_event(event, request_response)
        self._send_text_message(
            message,
            request_response=request_response,
            bypass_response_suppression=bypass_response_suppression,
            safety_override=safety_override,
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
    ) -> None:
        if not self.loop:
            logger.debug("Unable to send message; event loop unavailable.")
            return
        future = asyncio.run_coroutine_threadsafe(
            self.send_text_message_to_conversation(
                message,
                request_response=request_response,
                bypass_response_suppression=bypass_response_suppression,
                safety_override=safety_override,
            ),
            self.loop,
        )

        def _on_complete(task) -> None:
            try:
                task.result()
            except Exception as exc:
                logger.warning("Failed to send queued message: %s", exc)

        future.add_done_callback(_on_complete)

    def inject_system_context(self, payload: dict[str, Any]) -> None:
        """Inject non-chat startup context into conversation state."""

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
        await self.websocket.send(json.dumps(event))

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
        metadata = event.metadata or {}
        severity = str(metadata.get("severity", "info"))
        event_type = str(metadata.get("event_type", "status"))
        transition = str(metadata.get("transition", "steady"))

        if "battery" in getattr(self, "_suppressed_topics", set()) and not self._is_battery_safety_override(event):
            logger.info("battery_response_suppressed reason=topic_suppression")
            return False

        if event_type == "clear" or severity == "info":
            return self._is_battery_query_context_active()
        if not self._battery_response_enabled:
            return self._is_battery_query_context_active()

        if self._is_battery_query_context_active():
            return True

        if severity == "critical" and self._battery_response_allow_critical:
            if self._battery_response_require_transition:
                return not transition.startswith("steady_")
            return True

        if severity == "warning" and self._battery_response_allow_warning:
            if self._battery_response_require_transition:
                return transition in {"enter_warning", "enter_critical", "delta_drop"}
            return transition in {"enter_warning", "enter_critical", "delta_drop"}

        return fallback and not transition.startswith("steady_")

    def _is_battery_status_query(self, text: str) -> bool:
        lowered = text.strip().lower()
        if not lowered:
            return False
        query_tokens = (
            "battery",
            "battery level",
            "charge",
            "charging",
            "voltage",
            "power level",
            "low battery",
            "how's battery",
            "hows battery",
            "how is battery",
        )
        return any(token in lowered for token in query_tokens)

    def _is_battery_query_context_active(self) -> bool:
        if self._last_user_battery_query_time is None:
            return False
        return (
            time.monotonic() - self._last_user_battery_query_time
            <= self._battery_query_context_window_s
        )

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
        memory_intent = self._is_memory_intent(clean_text)
        self._last_user_input_text = clean_text
        self._last_user_input_time = time.monotonic()
        self._last_user_input_source = source
        if self._is_battery_status_query(clean_text):
            self._last_user_battery_query_time = self._last_user_input_time
        self._update_topic_suppression_from_user_text(clean_text)
        self._mark_preference_recall_candidate(clean_text, source=source)
        self._prepare_turn_memory_brief(clean_text, source=source, memory_intent=memory_intent)

    def _is_memory_intent(self, text: str) -> bool:
        normalized = " ".join((text or "").lower().split())
        if not normalized:
            return False
        return any(marker in normalized for marker in _MEMORY_INTENT_MARKERS)

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
        entity_phrases: list[str] = []
        for domain in _PREFERENCE_RECALL_DOMAINS:
            if domain in user_text:
                canonical = list(_PREFERENCE_QUERY_CANONICAL_BY_DOMAIN.get(domain, canonical))
                break

        for marker in ("prefer", "preferred", "favorite", "favourite"):
            for match in re.finditer(rf"\\b{marker}\\b(?P<entity>[^?.!,;:]{{0,40}})", user_text):
                entity = " ".join(match.group("entity").split())
                if entity:
                    entity_phrases.append(entity)

        if "editor" in user_text:
            canonical.extend(("editor", "user preferred editor"))

        ordered_parts: list[str] = []
        for item in canonical + entity_phrases + keywords:
            normalized = item.strip().lower()
            if normalized and normalized not in ordered_parts:
                ordered_parts.append(normalized)
        return ", ".join(ordered_parts)

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

    async def _run_preference_recall_with_fallbacks(
        self,
        *,
        recall_fn: Any,
        source: str,
        resolved_turn_id: str,
        query: str,
    ) -> tuple[dict[str, Any], bool]:
        scope = str(getattr(self, "_memory_retrieval_scope", MemoryScope.USER_GLOBAL.value))
        recall_queries = [query]
        if "editor" in query:
            recall_queries.append("editor")
        recall_queries.extend(_PREFERENCE_TAG_FALLBACK_QUERIES)

        for attempt_index, candidate_query in enumerate(recall_queries):
            payload = await recall_fn(query=candidate_query, limit=3, scope=scope)
            payload_keys = sorted(payload.keys()) if isinstance(payload, dict) else []
            memories = self._preference_recall_memories_from_payload(payload)
            cards = payload.get("memory_cards") if isinstance(payload, dict) else None
            cards_count = len(cards) if isinstance(cards, list) else 0
            logger.info(
                "preference_recall_tool_result run_id=%s resolved_turn_id=%s query=%s payload_keys=%s "
                "memories_count=%s cards_count=%s memory_cards_text_len=%s scope=%s attempt=%s source=%s",
                self._current_run_id() or "",
                resolved_turn_id,
                candidate_query,
                ",".join(payload_keys),
                len(memories),
                cards_count,
                len(str(payload.get("memory_cards_text", "")).strip()) if isinstance(payload, dict) else 0,
                scope,
                attempt_index,
                source,
            )
            self._preference_recall_cache[candidate_query] = {"timestamp": time.monotonic(), "payload": payload}
            cards_list = cards if isinstance(cards, list) else []
            memory_cards_text = (
                str(payload.get("memory_cards_text", "")).strip() if isinstance(payload, dict) else ""
            )
            hit = bool(memory_cards_text) or bool(cards_list) or bool(memories)
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
        return {"memories": []}, True

    def _is_preference_recall_intent(self, text: str) -> tuple[bool, list[str]]:
        normalized = " ".join((text or "").lower().split())
        if not normalized:
            return False, []
        if self._is_memory_intent(normalized):
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
        resolved_turn_id = str(turn_id or self._current_turn_id_or_unknown()).strip() or "turn-unknown"
        resolved_input_event_key = str(
            input_event_key
            if input_event_key is not None
            else getattr(self, "_current_input_event_key", "")
            or ""
        ).strip()
        resolved_utterance_seq = int(utterance_seq or self._current_utterance_seq() or 0)
        canonical_key = self._canonical_utterance_key(
            turn_id=resolved_turn_id,
            input_event_key=resolved_input_event_key,
        )
        return UtteranceContext(
            turn_id=resolved_turn_id,
            input_event_key=resolved_input_event_key,
            canonical_key=canonical_key,
            utterance_seq=resolved_utterance_seq,
        )

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
        return str(active_by_turn.get(turn_id) or "").strip()

    def _log_response_binding_event(self, *, response_key: str, turn_id: str, origin: str) -> None:
        logger.info(
            "response_binding run_id=%s active_key=%s response_key=%s turn_id=%s origin=%s",
            self._current_run_id() or "",
            self._active_input_event_key_for_turn(turn_id) or "unknown",
            str(response_key or "").strip() or "unknown",
            turn_id or "turn-unknown",
            origin or "unknown",
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

    def _canonical_utterance_key(self, *, turn_id: str, input_event_key: str | None) -> str:
        resolved_turn_id = str(turn_id or "").strip() or "turn-unknown"
        resolved_input_event_key = str(input_event_key or "").strip() or f"synthetic:{resolved_turn_id}"
        run_id = str(self._current_run_id() or "").strip() or "run-unknown"
        return f"{run_id}:{resolved_turn_id}:{resolved_input_event_key}"

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

    def _is_synthetic_input_event_key(self, input_event_key: str | None) -> bool:
        normalized = str(input_event_key or "").strip().lower()
        return normalized.startswith("synthetic_")

    def _rebind_active_response_correlation_key(
        self,
        *,
        turn_id: str,
        replacement_input_event_key: str,
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
        self._lifecycle_controller().on_replaced(old_canonical_key, new_canonical_key)
        self._log_lifecycle_event(
            turn_id=turn_id,
            input_event_key=normalized_replacement,
            canonical_key=new_canonical_key,
            origin="server_auto",
            response_id=getattr(self, "_active_response_id", None),
            decision="transition_replaced",
        )

        if str(getattr(self, "_active_response_input_event_key", "") or "").strip() == active_key:
            self._active_response_input_event_key = normalized_replacement
        if str(getattr(self, "_active_response_canonical_key", "") or "").strip() == old_canonical_key:
            self._active_response_canonical_key = new_canonical_key
        self._active_server_auto_input_event_key = normalized_replacement
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
        return self._canonical_utterance_key(turn_id=turn_id, input_event_key=input_event_key)

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
                    "done": state.done,
                    "cancel_sent": state.cancel_sent,
                    "origin": state.origin,
                    "response_id": state.response_id,
                    "obligation_present": state.obligation_present,
                    "input_event_key": state.input_event_key,
                    "turn_id": state.turn_id,
                },
            )

    def _response_delivery_state(self, *, turn_id: str, input_event_key: str | None) -> str | None:
        key = self._canonical_utterance_key(turn_id=turn_id, input_event_key=input_event_key)
        state = self._canonical_response_state(key)
        if isinstance(state, CanonicalResponseState):
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

        self._canonical_response_state_mutate(
            canonical_key=key,
            turn_id=turn_id,
            input_event_key=input_event_key,
            mutator=_mutate,
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
        prior_state = self._canonical_response_state(obligation_key)
        obligation_present_before = bool(
            isinstance(prior_state, CanonicalResponseState) and prior_state.obligation_present
        )
        active_response_input_event_key = str(
            getattr(self, "_active_response_input_event_key", "") or ""
        ).strip() or "none"
        active_server_auto_input_event_key = str(
            getattr(self, "_active_server_auto_input_event_key", "") or ""
        ).strip() or "none"

        def _mutate(record: CanonicalResponseState) -> None:
            record.obligation_present = True
            record.origin = str(source or "").strip() or record.origin
            record.obligation = {
                "turn_id": turn_id,
                "input_event_key": input_event_key,
                "source": source,
                "created_at": time.monotonic(),
            }

        updated_state = self._canonical_response_state_mutate(
            canonical_key=obligation_key,
            turn_id=turn_id,
            input_event_key=input_event_key,
            mutator=_mutate,
        )
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
        obligation_present_before = prior_state.obligation_present
        active_response_input_event_key = str(
            getattr(self, "_active_response_input_event_key", "") or ""
        ).strip() or "none"
        active_server_auto_input_event_key = str(
            getattr(self, "_active_server_auto_input_event_key", "") or ""
        ).strip() or "none"

        def _mutate(record: CanonicalResponseState) -> None:
            record.obligation_present = False
            record.obligation = None

        updated_state = self._canonical_response_state_mutate(
            canonical_key=obligation_key,
            turn_id=turn_id,
            input_event_key=input_event_key,
            mutator=_mutate,
        )
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
        metadata.setdefault("turn_id", turn_id)
        return input_event_key

    def _response_has_safety_override(self, response_create_event: dict[str, Any]) -> bool:
        metadata = self._extract_response_create_metadata(response_create_event)
        return str(metadata.get("safety_override", "")).strip().lower() in {"true", "1", "yes"}

    def _response_is_explicit_multipart(self, metadata: dict[str, Any] | None) -> bool:
        if not isinstance(metadata, dict):
            return False
        return str(metadata.get("explicit_multipart", "")).strip().lower() in {"true", "1", "yes"}

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
        logger.log(
            level,
            "lifecycle_event run_id=%s turn_id=%s input_event_key=%s canonical_key=%s origin=%s response_id=%s decision=%s",
            self._current_run_id() or "",
            resolved_turn_id,
            resolved_input_event_key,
            resolved_canonical_key,
            resolved_origin,
            resolved_response_id,
            resolved_decision,
        )
        self._append_lifecycle_timeline_event(
            canonical_key=resolved_canonical_key,
            entry=(
                f"decision={resolved_decision};origin={resolved_origin};"
                f"response_id={resolved_response_id};turn_id={resolved_turn_id};"
                f"input_event_key={resolved_input_event_key}"
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
        metadata = event.metadata or {}
        severity = str(metadata.get("severity", "")).strip().lower()
        if severity != "critical":
            return False
        percent = float(metadata.get("percent_of_range", 1.0)) * 100.0
        return percent <= self._battery_redline_percent

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
            logger.info(
                "response_schedule_drop run_id=%s turn_id=%s origin=%s dropped=%s reason=preference_recall_suppressed",
                self._current_run_id() or "",
                turn_id,
                normalized_origin,
                dropped,
            )

    async def _suppress_preference_recall_server_auto_response(self, websocket: Any) -> None:
        turn_id = self._current_turn_id_or_unknown()
        input_event_key = str(getattr(self, "_current_input_event_key", "") or "").strip()
        pending_before = 1 if isinstance(getattr(self, "_pending_response_create", None), PendingResponseCreate) else 0
        queue_before = len(getattr(self, "_response_create_queue", deque()) or ())
        suppressed_turns = getattr(self, "_preference_recall_suppressed_turns", None)
        if not isinstance(suppressed_turns, set):
            suppressed_turns = set()
            self._preference_recall_suppressed_turns = suppressed_turns
        suppressed_turns.add(turn_id)
        suppressed_input_event_keys = getattr(self, "_preference_recall_suppressed_input_event_keys", None)
        if not isinstance(suppressed_input_event_keys, set):
            suppressed_input_event_keys = set()
            self._preference_recall_suppressed_input_event_keys = suppressed_input_event_keys
        if input_event_key:
            suppressed_input_event_keys.add(input_event_key)
        self._clear_pending_response_contenders(
            turn_id=turn_id,
            input_event_key=input_event_key,
            reason="preference_recall_suppressed",
        )
        self._drop_suppressed_scheduled_response_creates(turn_id=turn_id, origin="server_auto")
        logger.info(
            "preference_recall_response_suppressed run_id=%s turn_id=%s input_event_key=%s reason=handled_preference_recall",
            self._current_run_id() or "",
            turn_id,
            input_event_key or "unknown",
        )
        self._log_response_site_debug(
            site="preference_recall_response_suppressed",
            turn_id=turn_id,
            input_event_key=input_event_key,
            canonical_key=self._canonical_utterance_key(turn_id=turn_id, input_event_key=input_event_key),
            origin="server_auto",
            trigger="handled_preference_recall",
        )
        pending_after = 1 if isinstance(getattr(self, "_pending_response_create", None), PendingResponseCreate) else 0
        queue_after = len(getattr(self, "_response_create_queue", deque()) or ())
        active_server_auto_key = str(getattr(self, "_active_server_auto_input_event_key", "") or "").strip() or "unknown"
        canonical_key = self._canonical_utterance_key(turn_id=turn_id, input_event_key=input_event_key)
        logger.debug(
            "[RESPTRACE] suppression_applied run_id=%s turn_id=%s input_event_key=%s canonical_key=%s "
            "suppressed_turns_count=%s suppressed_keys_count=%s removed_pending_count=%s removed_queued_count=%s "
            "active_server_auto_key=%s",
            self._current_run_id() or "",
            turn_id,
            input_event_key or "unknown",
            canonical_key,
            len(suppressed_turns),
            len(suppressed_input_event_keys),
            max(0, pending_before - pending_after),
            max(0, queue_before - queue_after),
            active_server_auto_key,
        )
        if str(getattr(self, "_active_response_origin", "")).strip().lower() != "server_auto":
            return
        active_key = str(getattr(self, "_active_server_auto_input_event_key", "") or "").strip()
        if input_event_key and active_key and active_key != input_event_key:
            return
        if not bool(getattr(self, "_response_in_flight", False)):
            return
        cancel_event = {"type": "response.cancel"}
        log_ws_event("Outgoing", cancel_event)
        self._track_outgoing_event(cancel_event, origin="preference_recall_guard")
        try:
            await websocket.send(json.dumps(cancel_event))
        except Exception as exc:
            message = str(exc)
            if "no active response found" in message.lower():
                logger.info(
                    "response_cancel_noop run_id=%s turn_id=%s reason=no_active_response",
                    self._current_run_id() or "",
                    turn_id,
                )
                logger.debug(
                    "response_cancel_noop_detail run_id=%s turn_id=%s error=%s",
                    self._current_run_id() or "",
                    turn_id,
                    message,
                )
                return
            logger.exception(
                "response_cancel_failed run_id=%s turn_id=%s origin=preference_recall_guard",
                self._current_run_id() or "",
                turn_id,
            )

    def _emit_preference_recall_skip_trace_if_needed(self, *, turn_id: str | None) -> None:
        pending = self._pending_preference_recall_trace if isinstance(self._pending_preference_recall_trace, dict) else None
        if not pending:
            return
        resolved_turn_id = str(turn_id or "").strip() or "turn-unknown"
        if resolved_turn_id in self._preference_recall_skip_logged_turn_ids:
            return
        if pending.get("intent") != "preference_recall":
            return
        recall_invoked = any(
            isinstance(record, dict)
            and record.get("name") == "recall_memories"
            and (
                str(record.get("turn_id") or "").strip() in {"", resolved_turn_id}
            )
            for record in (self._tool_call_records or [])
        )
        if recall_invoked:
            self._clear_preference_recall_candidate()
            return
        self._preference_recall_skip_logged_turn_ids.add(resolved_turn_id)
        logger.info(
            "preference_recall_decision_trace intent=%s decision=%s reason=%s query_fingerprint=%s run_id=%s turn_id=%s source=%s",
            pending.get("intent", "preference_recall"),
            pending.get("decision", "skipped_tool"),
            pending.get("reason", "model_did_not_request_tool"),
            pending.get("query_fingerprint", ""),
            self._current_run_id() or "",
            resolved_turn_id,
            pending.get("source", "unknown"),
        )
        self._clear_preference_recall_candidate()

    async def _maybe_handle_preference_recall_intent(self, text: str, websocket: Any, *, source: str) -> bool:
        matched, keywords = self._is_preference_recall_intent(text)
        if not matched:
            return False

        resolved_turn_id = self._current_turn_id_or_unknown()
        replacement_input_event_key = str(getattr(self, "_current_input_event_key", "") or "").strip()
        locked_input_event_keys = getattr(self, "_preference_recall_locked_input_event_keys", None)
        if not isinstance(locked_input_event_keys, set):
            locked_input_event_keys = set()
            self._preference_recall_locked_input_event_keys = locked_input_event_keys
        if replacement_input_event_key:
            locked_input_event_keys.add(replacement_input_event_key)
            logger.debug(
                "preference_recall_lock_set run_id=%s input_event_key=%s reason=intent_matched",
                self._current_run_id() or "",
                replacement_input_event_key,
            )

        turn_timestamps_store = getattr(self, "_turn_diagnostic_timestamps", None)
        if not isinstance(turn_timestamps_store, dict):
            turn_timestamps_store = {}
            self._turn_diagnostic_timestamps = turn_timestamps_store
        turn_timestamps = turn_timestamps_store.setdefault(resolved_turn_id, {})
        turn_timestamps["preference_recall_start"] = time.monotonic()
        self._mark_preference_recall_candidate(text, source=source)
        try:
            query = self._build_preference_recall_query(text.lower(), keywords=keywords)
            now = time.monotonic()
            cooldown_s = max(0.0, float(getattr(self, "_preference_recall_cooldown_s", 0.0)))
            cached = self._preference_recall_cache.get(query)
            result_payload: dict[str, Any] | None = None
            recall_invoked = False
            if (
                cached
                and cooldown_s > 0.0
                and now - float(cached.get("timestamp", 0.0)) < cooldown_s
            ):
                result_payload = cached.get("payload") if isinstance(cached.get("payload"), dict) else None
                logger.info(
                    "Preference recall reused cached result source=%s query=%s cooldown_s=%.2f",
                    source,
                    query,
                    cooldown_s,
                )
            if result_payload is None:
                recall_fn = function_map.get("recall_memories")
                if recall_fn is None:
                    if isinstance(self._pending_preference_recall_trace, dict):
                        self._pending_preference_recall_trace["reason"] = "policy_skip"
                    logger.warning("Preference recall intent matched but recall_memories tool unavailable.")
                    return False
                recall_invoked = True
                tool_call_records = getattr(self, "_tool_call_records", None)
                if isinstance(tool_call_records, list):
                    tool_call_records.append(
                        {
                            "name": "recall_memories",
                            "source": "preference_recall",
                            "turn_id": self._current_turn_id_or_unknown(),
                            "query": query,
                        }
                    )
                if isinstance(self._pending_preference_recall_trace, dict):
                    self._pending_preference_recall_trace["decision"] = "invoked_tool"
                    self._pending_preference_recall_trace["reason"] = "preference_intent_matched"
                result_payload, recall_invoked = await self._run_preference_recall_with_fallbacks(
                    recall_fn=recall_fn,
                    source=source,
                    resolved_turn_id=resolved_turn_id,
                    query=query,
                )
                logger.info(
                    "Preference recall executed source=%s query=%s keywords=%s",
                    source,
                    query,
                    ",".join(keywords),
                )

            memories = self._preference_recall_memories_from_payload(result_payload)
            cards = result_payload.get("memory_cards") if isinstance(result_payload, dict) else None
            cards_list = cards if isinstance(cards, list) else []
            memory_cards_text = ""
            if isinstance(result_payload, dict):
                memory_cards_text = self._sanitize_memory_cards_text_for_user(
                    str(result_payload.get("memory_cards_text", "")).strip()
                )
            hit = (
                bool(memory_cards_text)
                or bool(cards_list)
                or (isinstance(memories, list) and len(memories) > 0)
            )
            await self._suppress_preference_recall_server_auto_response(websocket)
            # preference_recall is an internal tool pathway: suppress server_auto output,
            # then guarantee a replacement assistant response for this input_event_key.
            if replacement_input_event_key and self._is_input_event_key_already_scheduled(
                input_event_key=replacement_input_event_key
            ):
                logger.info(
                    "preference_recall_replacement_skip run_id=%s turn_id=%s input_event_key=%s reason=already_scheduled_for_input_event_key",
                    self._current_run_id() or "",
                    resolved_turn_id,
                    replacement_input_event_key,
                )
                self._clear_preference_recall_candidate()
                return True
            if not hit:
                await self.send_assistant_message(
                    "I don’t have that saved yet. If you share it, I can remember it for next time.",
                    websocket,
                    response_metadata={
                        "turn_id": resolved_turn_id,
                        "input_event_key": replacement_input_event_key,
                        "trigger": "preference_recall",
                    },
                )
                self._mark_input_event_key_scheduled(input_event_key=replacement_input_event_key)
                turn_timestamps["preference_recall_end"] = time.monotonic()
                handled_logged_turn_ids = getattr(self, "_preference_recall_handled_logged_turn_ids", None)
                if not isinstance(handled_logged_turn_ids, set):
                    handled_logged_turn_ids = set()
                    self._preference_recall_handled_logged_turn_ids = handled_logged_turn_ids
                if resolved_turn_id not in handled_logged_turn_ids:
                    handled_logged_turn_ids.add(resolved_turn_id)
                    logger.info(
                        "preference_recall_handled run_id=%s resolved_turn_id=%s source=%s query=%s",
                        self._current_run_id() or "",
                        resolved_turn_id,
                        source,
                        query,
                    )
                    logger.debug(
                        "turn_diagnostic_timestamps run_id=%s turn_id=%s transcript_final_ts=%s preference_recall_start_ts=%s preference_recall_end_ts=%s response_schedule_ts=%s",
                        self._current_run_id() or "",
                        resolved_turn_id,
                        turn_timestamps.get("transcript_final"),
                        turn_timestamps.get("preference_recall_start"),
                        turn_timestamps.get("preference_recall_end"),
                        turn_timestamps.get("response_schedule"),
                    )
                self._clear_preference_recall_candidate()
                return True

            response_lines: list[str] = []
            if recall_invoked:
                response_lines.append("I’m checking what I remember.")
            if memory_cards_text:
                response_lines.append(memory_cards_text)
            else:
                first_memory = memories[0] if isinstance(memories[0], dict) else {}
                memory_text = str(first_memory.get("content", "")).strip()
                if not memory_text:
                    memory_text = "I found a saved preference, but it does not include enough detail to quote yet."
                response_lines.append(f'Relevant memory: "{memory_text}"')
                response_lines.append("Why it's relevant: " + '"It matches your preference question."')
                response_lines.append("Confidence: medium")

            if self._memory_pin_followup_needed(cards=cards_list, memories=memories, query=query):
                response_lines.append("Want me to pin or rename this memory so it’s easier to recall later?")

            await self.send_assistant_message(
                "\n".join(response_lines),
                websocket,
                response_metadata={
                    "turn_id": resolved_turn_id,
                    "input_event_key": replacement_input_event_key,
                    "trigger": "preference_recall",
                },
            )
            self._mark_input_event_key_scheduled(input_event_key=replacement_input_event_key)
            turn_timestamps["preference_recall_end"] = time.monotonic()
            handled_logged_turn_ids = getattr(self, "_preference_recall_handled_logged_turn_ids", None)
            if not isinstance(handled_logged_turn_ids, set):
                handled_logged_turn_ids = set()
                self._preference_recall_handled_logged_turn_ids = handled_logged_turn_ids
            if resolved_turn_id not in handled_logged_turn_ids:
                handled_logged_turn_ids.add(resolved_turn_id)
                logger.info(
                    "preference_recall_handled run_id=%s resolved_turn_id=%s source=%s query=%s",
                    self._current_run_id() or "",
                    resolved_turn_id,
                    source,
                    query,
                )
                logger.debug(
                    "turn_diagnostic_timestamps run_id=%s turn_id=%s transcript_final_ts=%s preference_recall_start_ts=%s preference_recall_end_ts=%s response_schedule_ts=%s",
                    self._current_run_id() or "",
                    resolved_turn_id,
                    turn_timestamps.get("transcript_final"),
                    turn_timestamps.get("preference_recall_start"),
                    turn_timestamps.get("preference_recall_end"),
                    turn_timestamps.get("response_schedule"),
                )
            self._clear_preference_recall_candidate()
            return True
        finally:
            if replacement_input_event_key:
                locked_input_event_keys.discard(replacement_input_event_key)
                logger.debug(
                    "preference_recall_lock_cleared run_id=%s input_event_key=%s reason=completed",
                    self._current_run_id() or "",
                    replacement_input_event_key,
                )

    def _is_active_response_guarded(self) -> bool:
        return bool(
            getattr(self, "_active_response_confirmation_guarded", False)
            or getattr(self, "_active_response_preference_guarded", False)
        )


    def _should_skip_turn_memory_retrieval(self, user_text: str) -> bool:
        text = user_text.strip()
        if len(text) < int(getattr(self, "_memory_retrieval_min_user_chars", 12)):
            return True
        if len(text.split()) < int(getattr(self, "_memory_retrieval_min_user_tokens", 3)):
            return True
        noisy = {"ok", "okay", "thanks", "thank you", "yes", "no", "cool", "nice", "got it"}
        return text.lower() in noisy

    def _prepare_turn_memory_brief(self, user_text: str, *, source: str, memory_intent: bool = False) -> None:
        if self._should_skip_turn_memory_retrieval(user_text):
            self._pending_turn_memory_brief = None
            return
        if not getattr(self, "_memory_retrieval_enabled", False):
            self._pending_turn_memory_brief = None
            return
        manager = getattr(self, "_memory_manager", None)
        if manager is None:
            self._pending_turn_memory_brief = None
            return
        try:
            self._pending_turn_memory_brief = manager.retrieve_for_turn(
                latest_user_utterance=user_text,
                user_id=manager.get_active_user_id(),
                max_memories=int(getattr(self, "_memory_retrieval_max_memories", 3)),
                max_chars=int(getattr(self, "_memory_retrieval_max_chars", 450)),
                cooldown_s=float(getattr(self, "_memory_retrieval_cooldown_s", 10.0)),
                bypass_cooldown=memory_intent,
                scope=str(getattr(self, "_memory_retrieval_scope", MemoryScope.USER_GLOBAL.value)),
                session_id=manager.get_active_session_id(),
            )
            retrieval_debug = manager.get_last_turn_retrieval_debug_metadata()
            if retrieval_debug:
                if memory_intent:
                    logger.info(
                        "turn_memory_retrieval_cooldown_bypassed source=%s cooldown_bypassed_reason=memory_intent",
                        source,
                    )
                semantic_runtime_health = manager.get_semantic_runtime_health()
                semantic_streak = int(semantic_runtime_health.get("query_embedding_not_ready_streak", 0))
                semantic_health_suffix = (
                    " semantic_runtime_ready=%s semantic_runtime_streak=%s semantic_runtime_last_error=%s "
                    "semantic_runtime_readiness_last_transition_at=%s semantic_runtime_readiness_age_ms=%s "
                    "semantic_runtime_readiness_transition_count=%s"
                    % (
                        semantic_runtime_health.get("ready"),
                        semantic_streak,
                        semantic_runtime_health.get("last_error_code"),
                        semantic_runtime_health.get("readiness_last_transition_at"),
                        semantic_runtime_health.get("readiness_age_ms"),
                        semantic_runtime_health.get("readiness_transition_count"),
                    )
                )
                logger.info(
                    "Turn memory retrieval audit source=%s mode=%s lexical_candidates=%s semantic_candidates=%s semantic_scored=%s candidates_without_ready_embedding=%s candidates_below_influence_threshold=%s candidates_semantic_applied=%s selected=%s fallback_reason=%s latency_ms=%s truncated=%s truncation_count=%s dedupe_count=%s semantic_provider=%s semantic_model=%s semantic_query_timeout_ms=%s semantic_query_timeout_ms_used=%s semantic_query_duration_ms=%s semantic_query_embed_elapsed_ms=%s semantic_timeout_source=%s semantic_result_status=%s semantic_error_code=%s semantic_error_class=%s semantic_failure_class=%s canary_last_error_code=%s canary_last_latency_ms=%s canary_last_checked_age_ms=%s semantic_provider_last_error_code=%s timeout_backoff_until_remaining_ms=%s semantic_scoring_skipped_reason=%s query_fingerprint_hash=%s query_fingerprint_length=%s%s",
                    source,
                    retrieval_debug.get("mode"),
                    retrieval_debug.get("lexical_candidate_count"),
                    retrieval_debug.get("semantic_candidate_count"),
                    retrieval_debug.get("semantic_scored_count"),
                    retrieval_debug.get("candidates_without_ready_embedding"),
                    retrieval_debug.get("candidates_below_influence_threshold"),
                    retrieval_debug.get("candidates_semantic_applied"),
                    retrieval_debug.get("selected_count"),
                    retrieval_debug.get("fallback_reason"),
                    retrieval_debug.get("latency_ms"),
                    retrieval_debug.get("truncated"),
                    retrieval_debug.get("truncation_count"),
                    retrieval_debug.get("dedupe_count"),
                    retrieval_debug.get("semantic_provider"),
                    retrieval_debug.get("semantic_model"),
                    retrieval_debug.get("semantic_query_timeout_ms"),
                    retrieval_debug.get("semantic_query_timeout_ms_used"),
                    retrieval_debug.get("semantic_query_duration_ms"),
                    retrieval_debug.get("semantic_query_embed_elapsed_ms"),
                    retrieval_debug.get("semantic_timeout_source"),
                    retrieval_debug.get("semantic_result_status"),
                    retrieval_debug.get("semantic_error_code"),
                    retrieval_debug.get("semantic_error_class"),
                    retrieval_debug.get("semantic_failure_class"),
                    retrieval_debug.get("canary_last_error_code"),
                    retrieval_debug.get("canary_last_latency_ms"),
                    retrieval_debug.get("canary_last_checked_age_ms"),
                    retrieval_debug.get("semantic_provider_last_error_code"),
                    retrieval_debug.get("timeout_backoff_until_remaining_ms"),
                    retrieval_debug.get("semantic_scoring_skipped_reason"),
                    retrieval_debug.get("query_fingerprint_hash"),
                    retrieval_debug.get("query_fingerprint_length"),
                    semantic_health_suffix,
                )
        except Exception as exc:  # pragma: no cover - defensive fail-open
            self._pending_turn_memory_brief = None
            now = time.monotonic()
            throttle_s = max(1.0, float(getattr(self, "_memory_retrieval_error_throttle_s", 60.0)))
            should_log = (
                self._memory_retrieval_last_error_log_at <= 0.0
                or (now - self._memory_retrieval_last_error_log_at) >= throttle_s
            )
            if should_log:
                suppressed = int(getattr(self, "_memory_retrieval_suppressed_errors", 0))
                logger.warning(
                    "Turn memory retrieval failed for source=%s suppressed_since_last=%s: %s",
                    source,
                    suppressed,
                    exc,
                )
                self._memory_retrieval_last_error_log_at = now
                self._memory_retrieval_suppressed_errors = 0
            else:
                self._memory_retrieval_suppressed_errors += 1

    def _consume_pending_memory_brief_note(self) -> str | None:
        brief = getattr(self, "_pending_turn_memory_brief", None)
        self._pending_turn_memory_brief = None
        if brief is None or not brief.items:
            return None
        lines = [
            "Turn memory brief (retrieved long-term context; do not quote verbatim unless asked):"
        ]
        for index, item in enumerate(brief.items, start=1):
            lines.append(render_realtime_memory_brief_item(index=index, item=item))
        if brief.truncated:
            lines.append("Additional relevant memories were omitted due to retrieval limits.")
        return "\n".join(lines)

    def _build_startup_memory_digest_note(self) -> str | None:
        if not getattr(self, "_startup_memory_digest_enabled", True):
            return None
        manager = getattr(self, "_memory_manager", None)
        if manager is None:
            return None
        try:
            digest = manager.retrieve_startup_digest(
                max_items=int(getattr(self, "_startup_memory_digest_max_items", 2)),
                max_chars=int(getattr(self, "_startup_memory_digest_max_chars", 280)),
                user_id=manager.get_active_user_id(),
            )
        except Exception as exc:  # pragma: no cover - defensive fail-open
            logger.warning("Startup memory digest retrieval failed: %s", exc)
            return None
        if digest is None or not digest.items:
            return None
        lines = ["Startup memory digest (stable user context):"]
        for index, item in enumerate(digest.items, start=1):
            lines.append(render_startup_memory_digest_item(index=index, item=item))
        if digest.truncated:
            lines.append("Additional pinned memories were omitted due to startup digest limits.")
        return "\n".join(lines)

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
        await websocket.send(json.dumps(note_event))

    def _find_stop_word(self, text: str) -> str | None:
        if not text or not self._stop_words:
            return None
        lowered = text.lower()
        for word in self._stop_words:
            if " " in word:
                if word in lowered:
                    return word
                continue
            if re.search(rf"\\b{re.escape(word)}\\b", lowered):
                return word
        return None

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
        options = "Approve / Deny / Dry-run" if dry_run_supported else "Approve / Deny"
        options_suffix = f"options: {options}."

        def _normalize_sentence(value: str | None) -> str:
            normalized = " ".join(str(value or "").split()).strip()
            if normalized.endswith("."):
                return normalized
            return f"{normalized}."

        normalized_summary_sentence = _normalize_sentence(action_summary)
        normalized_reason_sentence = _normalize_sentence(confirm_reason)

        def _is_contract_compliant(prompt_text: str) -> bool:
            parts = [part.strip() for part in prompt_text.split(" Reason: ", maxsplit=1)]
            if len(parts) != 2:
                return False
            if not parts[0].startswith("Action summary: "):
                return False
            reason_clause = parts[1]
            expected_suffix = f"; {options_suffix}"
            if not reason_clause.endswith(expected_suffix):
                return False
            action_body = parts[0][len("Action summary: ") :].strip()
            reason_body = reason_clause[: -len(expected_suffix)].strip()
            return bool(action_body.endswith(".") and reason_body.endswith("."))

        if confirm_prompt:
            prompt_override = " ".join(str(confirm_prompt).split()).strip()
            if _is_contract_compliant(prompt_override):
                logger.info(
                    "Approval prompt override accepted | tool=%s details=%s",
                    action.tool_name,
                    json.dumps({"confirm_prompt": prompt_override}, sort_keys=True),
                )
                return prompt_override
            logger.info(
                "Approval prompt override ignored due to contract violation | tool=%s details=%s",
                action.tool_name,
                json.dumps({"confirm_prompt": prompt_override}, sort_keys=True),
            )
        return (
            f"Action summary: {normalized_summary_sentence} "
            f"Reason: {normalized_reason_sentence}; {options_suffix}"
        )

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
        await websocket.send(json.dumps(function_call_output))
        if include_response_create:
            response_create_event = {"type": "response.create"}
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
        self._confirmation_timeout_markers[fingerprint] = time.monotonic()
        self._confirmation_timeout_causes[fingerprint] = cause

    def _record_recent_confirmation_outcome(self, idempotency_key: str | None, outcome: str) -> None:
        normalized_key = str(idempotency_key or "").strip()
        if not normalized_key:
            return
        self._recent_confirmation_outcomes[normalized_key] = {
            "outcome": str(outcome),
            "timestamp": time.monotonic(),
        }

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
        outcomes = getattr(self, "_recent_confirmation_outcomes", {})
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
        expired = [
            marker
            for marker, ts in self._confirmation_timeout_markers.items()
            if now - ts > self._confirmation_timeout_debounce_window_s
        ]
        for marker in expired:
            self._confirmation_timeout_markers.pop(marker, None)
            self._confirmation_timeout_causes.pop(marker, None)
        fingerprint = self._build_tool_call_fingerprint(tool_name, args)
        ts = self._confirmation_timeout_markers.get(fingerprint)
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
        normalized = {
            "query": self._normalize_research_query_text(query),
            "source": self._normalize_research_source_text(source),
        }
        payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

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
        now_ts = time.monotonic() if now is None else now
        outcomes = getattr(self, "_research_permission_outcomes", None)
        if not isinstance(outcomes, dict):
            self._research_permission_outcomes = {}
            self._research_suppressed_fingerprints = {}
            return
        ttl_s = float(getattr(self, "_research_permission_outcome_ttl_s", 20.0))
        expired = [
            fingerprint
            for fingerprint, payload in outcomes.items()
            if now_ts - float(payload.get("recorded_at", 0.0)) > ttl_s
        ]
        for fingerprint in expired:
            outcomes.pop(fingerprint, None)
            if isinstance(getattr(self, "_research_suppressed_fingerprints", None), dict):
                self._research_suppressed_fingerprints.pop(fingerprint, None)

    def _record_research_permission_outcome(self, fingerprint: str, *, approved: bool) -> None:
        self._prune_research_permission_outcomes()
        if not isinstance(getattr(self, "_research_permission_outcomes", None), dict):
            self._research_permission_outcomes = {}
        self._research_permission_outcomes[fingerprint] = {
            "approved": approved,
            "recorded_at": time.monotonic(),
        }
        if not isinstance(getattr(self, "_research_suppressed_fingerprints", None), dict):
            self._research_suppressed_fingerprints = {}
        if approved:
            self._research_suppressed_fingerprints.pop(fingerprint, None)

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
        if normalized_origin == "tool_output":
            return 3
        if normalized_origin == "assistant_message":
            return 2
        return 1

    def _sync_pending_response_create_queue(self) -> None:
        self._response_create_queue.clear()
        pending = self._pending_response_create
        if pending is None:
            self._queued_confirmation_reminder_keys.clear()
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
        }
        if pending.queued_reminder_key is not None:
            queued_item["queued_reminder_key"] = pending.queued_reminder_key
            self._queued_confirmation_reminder_keys = {pending.queued_reminder_key}
        else:
            self._queued_confirmation_reminder_keys.clear()
        self._response_create_queue.append(queued_item)

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
        turn_id = self._resolve_response_create_turn_id(origin=origin, response_create_event=response_create_event)
        current_input_event_key = self._ensure_response_create_correlation(
            response_create_event=response_create_event,
            origin=origin,
            turn_id=turn_id,
        )
        with self._utterance_context_scope(
            turn_id=turn_id,
            input_event_key=current_input_event_key,
        ) as resolved_context:
            turn_id = resolved_context.turn_id
            current_input_event_key = resolved_context.input_event_key
            canonical_key = resolved_context.canonical_key
        response_metadata = self._extract_response_create_metadata(response_create_event)
        normalized_origin = str(origin or "").strip().lower()
        consumes_canonical_slot = self._response_consumes_canonical_slot(response_metadata)
        explicit_multipart = self._response_is_explicit_multipart(response_metadata)
        if consumes_canonical_slot and self._canonical_first_audio_started(canonical_key) and not explicit_multipart:
            logger.info(
                "response_schedule_blocked run_id=%s turn_id=%s origin=%s mode=direct reason=canonical_audio_already_started canonical_key=%s",
                self._current_run_id() or "",
                turn_id,
                normalized_origin,
                canonical_key,
            )
            return False
        suppression_turns = getattr(self, "_preference_recall_suppressed_turns", set())
        created_keys = getattr(self, "_response_created_canonical_keys", set())
        if consumes_canonical_slot:
            single_flight_block_reason = self._single_flight_block_reason(
                turn_id=turn_id,
                input_event_key=current_input_event_key,
            )
            if single_flight_block_reason:
                self._log_response_create_blocked(
                    turn_id=turn_id,
                    origin=normalized_origin,
                    input_event_key=current_input_event_key,
                    canonical_key=canonical_key,
                    block_reason=single_flight_block_reason,
                )
                return False
        obligation_present = self._response_obligation_key(
            turn_id=turn_id,
            input_event_key=current_input_event_key,
        ) in getattr(self, "_response_obligations", {})
        pending_server_auto_len = len(getattr(self, "_pending_server_auto_input_event_keys", deque()) or ())
        logger.debug(
            "[RESPTRACE] response_create_request run_id=%s turn_id=%s origin=%s trigger_reason=%s input_event_key=%s "
            "canonical_key=%s sent_now=false scheduled=false replaced=false queue_len=%s pending_server_auto_len=%s "
            "suppression_active=%s suppression_reason=%s obligation_present=%s response_done_serial=%s",
            self._current_run_id() or "",
            turn_id,
            normalized_origin,
            reason,
            current_input_event_key or "unknown",
            canonical_key,
            len(getattr(self, "_response_create_queue", deque()) or ()),
            pending_server_auto_len,
            bool(turn_id in suppression_turns),
            self._resptrace_suppression_reason(turn_id=turn_id, input_event_key=current_input_event_key),
            obligation_present,
            getattr(self, "_response_done_serial", 0),
        )
        if self._is_response_already_delivered(turn_id=turn_id, input_event_key=current_input_event_key):
            logger.debug(
                "duplicate_response_prevented run_id=%s turn_id=%s input_event_key=%s reason=already_delivered origin=%s",
                self._current_run_id() or "",
                turn_id,
                current_input_event_key or "unknown",
                normalized_origin,
            )
            if reason == "audio_playback_busy":
                logger.debug(
                    "playback_busy_retry_dropped run_id=%s input_event_key=%s reason=already_delivered",
                    self._current_run_id() or "",
                    current_input_event_key or "unknown",
                )
            return False
        if self._is_preference_recall_lock_blocked(
            turn_id=turn_id,
            input_event_key=current_input_event_key,
            normalized_origin=normalized_origin,
            response_metadata=response_metadata,
        ):
            return False
        if (
            consumes_canonical_slot
            and self._canonical_first_audio_started(canonical_key)
            and not self._response_is_explicit_multipart(response_metadata)
        ):
            logger.info(
                "response_schedule_blocked run_id=%s turn_id=%s origin=%s mode=queued reason=canonical_audio_already_started canonical_key=%s",
                self._current_run_id() or "",
                turn_id,
                normalized_origin,
                canonical_key,
            )
            return False
        if (
            consumes_canonical_slot
            and canonical_key in created_keys
            and not self._response_has_safety_override(response_create_event)
        ):
            logger.info(
                "response_schedule_blocked run_id=%s turn_id=%s origin=%s mode=queued reason=canonical_response_already_created canonical_key=%s",
                self._current_run_id() or "",
                turn_id,
                normalized_origin,
                canonical_key,
            )
            return False
        suppression_active = turn_id in suppression_turns and not current_input_event_key
        if suppression_active and normalized_origin == "server_auto":
            self._drop_suppressed_scheduled_response_creates(turn_id=turn_id, origin=normalized_origin)
            self._mark_transcript_response_outcome(
                input_event_key=current_input_event_key,
                turn_id=turn_id,
                outcome="response_not_scheduled",
                reason="preference_recall_suppressed",
                details="queued response.create blocked",
            )
            logger.info(
                "response_schedule_blocked run_id=%s turn_id=%s origin=%s mode=queued reason=preference_recall_suppressed",
                self._current_run_id() or "",
                turn_id,
                normalized_origin,
            )
            return False
        schedule_logged_turn_ids = getattr(self, "_response_schedule_logged_turn_ids", None)
        if not isinstance(schedule_logged_turn_ids, set):
            schedule_logged_turn_ids = set()
            self._response_schedule_logged_turn_ids = schedule_logged_turn_ids
        if turn_id not in schedule_logged_turn_ids:
            schedule_logged_turn_ids.add(turn_id)
            turn_timestamps_store = getattr(self, "_turn_diagnostic_timestamps", None)
            if not isinstance(turn_timestamps_store, dict):
                turn_timestamps_store = {}
                self._turn_diagnostic_timestamps = turn_timestamps_store
            turn_timestamps = turn_timestamps_store.setdefault(turn_id, {})
            turn_timestamps["response_schedule"] = time.monotonic()
            logger.info(
                "response_schedule_marker run_id=%s turn_id=%s origin=%s input_event_key=%s canonical_key=%s suppression_active=%s mode=queued",
                self._current_run_id() or "",
                turn_id,
                origin,
                current_input_event_key or "unknown",
                canonical_key,
                suppression_active,
            )
            logger.debug(
                "turn_diagnostic_timestamps run_id=%s turn_id=%s transcript_final_ts=%s preference_recall_start_ts=%s preference_recall_end_ts=%s response_schedule_ts=%s",
                self._current_run_id() or "",
                turn_id,
                turn_timestamps.get("transcript_final"),
                turn_timestamps.get("preference_recall_start"),
                turn_timestamps.get("preference_recall_end"),
                turn_timestamps.get("response_schedule"),
            )
        reminder_key = self._extract_confirmation_reminder_dedupe_key(response_create_event)
        candidate = PendingResponseCreate(
            websocket=websocket,
            event=response_create_event,
            origin=origin,
            turn_id=turn_id,
            created_at=time.monotonic(),
            reason=reason,
            record_ai_call=record_ai_call,
            debug_context=debug_context,
            memory_brief_note=memory_brief_note,
            queued_reminder_key=reminder_key,
            enqueued_done_serial=self._response_done_serial,
        )
        previous = self._pending_response_create
        if previous is None:
            self._pending_response_create = candidate
            self._sync_pending_response_create_queue()
            if reason == "audio_playback_busy":
                logger.debug(
                    "playback_busy_retry_enqueued run_id=%s input_event_key=%s reason=audio_playback_busy",
                    self._current_run_id() or "",
                    current_input_event_key or "unknown",
                )
            logger.info(
                "response_create_scheduled turn_id=%s origin=%s reason=%s",
                candidate.turn_id,
                candidate.origin,
                reason,
            )
            self._log_response_site_debug(
                site="response_create_scheduled",
                turn_id=turn_id,
                input_event_key=current_input_event_key,
                canonical_key=canonical_key,
                origin=normalized_origin,
                trigger=reason,
            )
            logger.debug(
                "[RESPTRACE] response_create_request run_id=%s turn_id=%s origin=%s trigger_reason=%s input_event_key=%s "
                "canonical_key=%s sent_now=false scheduled=true replaced=false queue_len=%s pending_server_auto_len=%s "
                "suppression_active=%s suppression_reason=%s obligation_present=%s response_done_serial=%s",
                self._current_run_id() or "",
                turn_id,
                normalized_origin,
                reason,
                current_input_event_key or "unknown",
                canonical_key,
                len(getattr(self, "_response_create_queue", deque()) or ()),
                len(getattr(self, "_pending_server_auto_input_event_keys", deque()) or ()),
                bool(turn_id in suppression_turns),
                self._resptrace_suppression_reason(turn_id=turn_id, input_event_key=current_input_event_key),
                obligation_present,
                getattr(self, "_response_done_serial", 0),
            )
            return False

        should_replace = False
        replacement_reason = ""
        if candidate.turn_id != previous.turn_id:
            should_replace = True
            replacement_reason = "newer_turn"
        else:
            candidate_priority = self._response_create_priority(candidate.origin)
            previous_priority = self._response_create_priority(previous.origin)
            if candidate_priority > previous_priority:
                should_replace = True
                replacement_reason = "higher_priority"
            elif candidate_priority == previous_priority:
                should_replace = True
                replacement_reason = "latest_update"

        if should_replace:
            self._pending_response_create = candidate
            self._sync_pending_response_create_queue()
            if reason == "audio_playback_busy":
                logger.debug(
                    "playback_busy_retry_enqueued run_id=%s input_event_key=%s reason=audio_playback_busy",
                    self._current_run_id() or "",
                    current_input_event_key or "unknown",
                )
            logger.info(
                "response_create_replaced turn_id=%s old_origin=%s new_origin=%s reason=%s",
                candidate.turn_id,
                previous.origin,
                candidate.origin,
                replacement_reason,
            )
            self._log_response_site_debug(
                site="response_create_replaced",
                turn_id=turn_id,
                input_event_key=current_input_event_key,
                canonical_key=canonical_key,
                origin=normalized_origin,
                trigger=replacement_reason,
            )
            logger.debug(
                "[RESPTRACE] response_create_request run_id=%s turn_id=%s origin=%s trigger_reason=%s input_event_key=%s "
                "canonical_key=%s sent_now=false scheduled=true replaced=true queue_len=%s pending_server_auto_len=%s "
                "suppression_active=%s suppression_reason=%s obligation_present=%s response_done_serial=%s",
                self._current_run_id() or "",
                turn_id,
                normalized_origin,
                replacement_reason,
                current_input_event_key or "unknown",
                canonical_key,
                len(getattr(self, "_response_create_queue", deque()) or ()),
                len(getattr(self, "_pending_server_auto_input_event_keys", deque()) or ()),
                bool(turn_id in suppression_turns),
                self._resptrace_suppression_reason(turn_id=turn_id, input_event_key=current_input_event_key),
                obligation_present,
                getattr(self, "_response_done_serial", 0),
            )
        else:
            logger.info(
                "response_create_dropped turn_id=%s origin=%s reason=%s",
                candidate.turn_id,
                candidate.origin,
                "lower_priority_than_pending",
            )
            logger.debug(
                "[RESPTRACE] response_create_request run_id=%s turn_id=%s origin=%s trigger_reason=%s input_event_key=%s "
                "canonical_key=%s sent_now=false scheduled=false replaced=false queue_len=%s pending_server_auto_len=%s "
                "suppression_active=%s suppression_reason=%s obligation_present=%s response_done_serial=%s",
                self._current_run_id() or "",
                turn_id,
                normalized_origin,
                "lower_priority_than_pending",
                current_input_event_key or "unknown",
                canonical_key,
                len(getattr(self, "_response_create_queue", deque()) or ()),
                len(getattr(self, "_pending_server_auto_input_event_keys", deque()) or ()),
                bool(turn_id in suppression_turns),
                self._resptrace_suppression_reason(turn_id=turn_id, input_event_key=current_input_event_key),
                obligation_present,
                getattr(self, "_response_done_serial", 0),
            )
        return False

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
        now = time.monotonic()
        delta_ms = None
        if self._last_response_create_ts is not None:
            delta_ms = (now - self._last_response_create_ts) * 1000.0
        if self._response_create_debug_trace:
            ctx = debug_context or {}
            logger.info(
                "[Debug][response.create] active_response_id=%s origin=%s tool=%s call_id=%s research_id=%s delta_ms=%s",
                self._active_response_id,
                origin,
                ctx.get("tool_name"),
                ctx.get("call_id"),
                ctx.get("research_id"),
                f"{delta_ms:.1f}" if delta_ms is not None else "n/a",
            )

        # Migration (phase 1): delegated policy decisions keep signature stable.
        if memory_brief_note:
            try:
                await self._send_memory_brief_note(websocket, memory_brief_note)
            except Exception as exc:  # pragma: no cover - defensive fail-open
                logger.warning("Memory brief injection skipped due to error: %s", exc)
        context_hint = utterance_context or getattr(self, "_utterance_context", None)
        if context_hint is not None:
            metadata = self._extract_response_create_metadata(response_create_event)
            metadata.setdefault("turn_id", context_hint.turn_id)
            if context_hint.input_event_key:
                metadata.setdefault("input_event_key", context_hint.input_event_key)
        turn_id = self._resolve_response_create_turn_id(origin=origin, response_create_event=response_create_event)
        current_input_event_key = self._ensure_response_create_correlation(
            response_create_event=response_create_event,
            origin=origin,
            turn_id=turn_id,
        )
        with self._utterance_context_scope(turn_id=turn_id, input_event_key=current_input_event_key) as resolved_context:
            turn_id = resolved_context.turn_id
            current_input_event_key = resolved_context.input_event_key
            canonical_key = resolved_context.canonical_key

        response_metadata = self._extract_response_create_metadata(response_create_event)
        normalized_origin = str(origin or "").strip().lower()
        consumes_canonical_slot = self._response_consumes_canonical_slot(response_metadata)
        explicit_multipart = self._response_is_explicit_multipart(response_metadata)
        suppression_turns = getattr(self, "_preference_recall_suppressed_turns", set())
        created_keys = getattr(self, "_response_created_canonical_keys", set())
        single_flight_block_reason = self._single_flight_block_reason(
            turn_id=turn_id,
            input_event_key=current_input_event_key,
        ) if consumes_canonical_slot else ""
        suppression_active = turn_id in suppression_turns and not current_input_event_key
        preference_recall_lock_blocked = self._is_preference_recall_lock_blocked(
            turn_id=turn_id,
            input_event_key=current_input_event_key,
            normalized_origin=normalized_origin,
            response_metadata=response_metadata,
        )
        decision = self._lifecycle_policy().decide_response_create(
            response_in_flight=bool(self._response_in_flight),
            audio_playback_busy=bool(self._audio_playback_busy),
            consumes_canonical_slot=consumes_canonical_slot,
            canonical_audio_started=self._canonical_first_audio_started(canonical_key),
            explicit_multipart=explicit_multipart,
            single_flight_block_reason=single_flight_block_reason,
            already_delivered=self._is_response_already_delivered(turn_id=turn_id, input_event_key=current_input_event_key),
            preference_recall_lock_blocked=preference_recall_lock_blocked,
            canonical_key_already_created=canonical_key in created_keys,
            has_safety_override=self._response_has_safety_override(response_create_event),
            suppression_active=suppression_active,
            normalized_origin=normalized_origin,
        )

        if decision.action is ResponseCreateDecisionAction.SCHEDULE:
            return self._schedule_pending_response_create(
                websocket=websocket,
                response_create_event=response_create_event,
                origin=origin,
                reason=str(decision.queue_reason or decision.reason_code),
                record_ai_call=record_ai_call,
                debug_context=debug_context,
                memory_brief_note=memory_brief_note,
            )
        if decision.action is ResponseCreateDecisionAction.BLOCK:
            if decision.reason_code == "already_delivered":
                logger.debug(
                    "duplicate_response_prevented run_id=%s turn_id=%s input_event_key=%s reason=already_delivered origin=%s",
                    self._current_run_id() or "",
                    turn_id,
                    current_input_event_key or "unknown",
                    normalized_origin,
                )
                return False
            if decision.reason_code == "preference_recall_lock_blocked":
                return False
            if single_flight_block_reason and decision.reason_code == single_flight_block_reason:
                self._log_response_create_blocked(
                    turn_id=turn_id,
                    origin=normalized_origin,
                    input_event_key=current_input_event_key,
                    canonical_key=canonical_key,
                    block_reason=decision.reason_code,
                )
                return False
            if decision.reason_code == "preference_recall_suppressed":
                self._drop_suppressed_scheduled_response_creates(turn_id=turn_id, origin=normalized_origin)
                self._mark_transcript_response_outcome(
                    input_event_key=current_input_event_key,
                    turn_id=turn_id,
                    outcome="response_not_scheduled",
                    reason=decision.reason_code,
                    details="direct response.create blocked",
                )
            logger.info(
                "response_schedule_blocked run_id=%s turn_id=%s origin=%s mode=direct reason=%s canonical_key=%s",
                self._current_run_id() or "",
                turn_id,
                normalized_origin,
                decision.reason_code,
                canonical_key,
            )
            return False

        obligation_present = self._response_obligation_key(
            turn_id=turn_id,
            input_event_key=current_input_event_key,
        ) in getattr(self, "_response_obligations", {})
        pending_server_auto_len = len(getattr(self, "_pending_server_auto_input_event_keys", deque()) or ())
        logger.debug(
            "[RESPTRACE] response_create_request run_id=%s turn_id=%s origin=%s trigger_reason=%s input_event_key=%s "
            "canonical_key=%s sent_now=false scheduled=false replaced=false queue_len=%s pending_server_auto_len=%s "
            "suppression_active=%s suppression_reason=%s obligation_present=%s response_done_serial=%s",
            self._current_run_id() or "",
            turn_id,
            normalized_origin,
            decision.reason_code,
            current_input_event_key or "unknown",
            canonical_key,
            len(getattr(self, "_response_create_queue", deque()) or ()),
            pending_server_auto_len,
            bool(turn_id in suppression_turns),
            self._resptrace_suppression_reason(turn_id=turn_id, input_event_key=current_input_event_key),
            obligation_present,
            getattr(self, "_response_done_serial", 0),
        )
        schedule_logged_turn_ids = getattr(self, "_response_schedule_logged_turn_ids", None)
        if not isinstance(schedule_logged_turn_ids, set):
            schedule_logged_turn_ids = set()
            self._response_schedule_logged_turn_ids = schedule_logged_turn_ids
        if turn_id not in schedule_logged_turn_ids:
            schedule_logged_turn_ids.add(turn_id)
            turn_timestamps_store = getattr(self, "_turn_diagnostic_timestamps", None)
            if not isinstance(turn_timestamps_store, dict):
                turn_timestamps_store = {}
                self._turn_diagnostic_timestamps = turn_timestamps_store
            turn_timestamps = turn_timestamps_store.setdefault(turn_id, {})
            turn_timestamps["response_schedule"] = now
            logger.info(
                "response_schedule_marker run_id=%s turn_id=%s origin=%s input_event_key=%s canonical_key=%s suppression_active=%s mode=direct",
                self._current_run_id() or "",
                turn_id,
                origin,
                current_input_event_key or "unknown",
                canonical_key,
                suppression_active,
            )
            logger.debug(
                "turn_diagnostic_timestamps run_id=%s turn_id=%s transcript_final_ts=%s preference_recall_start_ts=%s preference_recall_end_ts=%s response_schedule_ts=%s",
                self._current_run_id() or "",
                turn_id,
                turn_timestamps.get("transcript_final"),
                turn_timestamps.get("preference_recall_start"),
                turn_timestamps.get("preference_recall_end"),
                turn_timestamps.get("response_schedule"),
            )
        log_ws_event("Outgoing", response_create_event)
        self._track_outgoing_event(response_create_event, origin=origin)
        await websocket.send(json.dumps(response_create_event))
        if consumes_canonical_slot:
            self._set_response_delivery_state(
                turn_id=turn_id,
                input_event_key=current_input_event_key,
                state="created",
            )
        self._mark_input_event_key_scheduled(input_event_key=current_input_event_key)
        self._last_response_create_ts = now
        self._response_in_flight = True
        if record_ai_call:
            self._record_ai_call()
        logger.debug(
            "[RESPTRACE] response_create_request run_id=%s turn_id=%s origin=%s trigger_reason=%s input_event_key=%s "
            "canonical_key=%s sent_now=true scheduled=false replaced=false queue_len=%s pending_server_auto_len=%s "
            "suppression_active=%s suppression_reason=%s obligation_present=%s response_done_serial=%s",
            self._current_run_id() or "",
            turn_id,
            normalized_origin,
            decision.reason_code,
            current_input_event_key or "unknown",
            canonical_key,
            len(getattr(self, "_response_create_queue", deque()) or ()),
            len(getattr(self, "_pending_server_auto_input_event_keys", deque()) or ()),
            bool(turn_id in suppression_turns),
            self._resptrace_suppression_reason(turn_id=turn_id, input_event_key=current_input_event_key),
            obligation_present,
            getattr(self, "_response_done_serial", 0),
        )
        return True

    async def _drain_response_create_queue(self, source_trigger: str | None = None) -> None:
        source_candidate = source_trigger
        if source_candidate is None:
            source_candidate = getattr(self, "_response_create_queue_drain_source", "explicit_caller")
        self._response_create_queue_drain_source = "explicit_caller"
        normalized_source_trigger = str(source_candidate or "").strip().lower()
        if normalized_source_trigger not in {"response_done", "playback_complete"}:
            normalized_source_trigger = "explicit_caller"

        selected_pending_origin = "none"
        selected_pending_turn_id = "none"
        selected_pending_input_event_key = "unknown"
        selected_pending_canonical_key = "unknown"
        selected_pending_trigger = "none"
        selected_pending_enqueued_done_serial = "none"
        selected_pending_serial_relation = "none"
        drain_result = "none"
        skipped_reason = "none"

        def _serial_relation(serial: int | None) -> str:
            if serial is None:
                return "none"
            if serial < self._response_done_serial:
                return "older"
            if serial > self._response_done_serial:
                return "newer"
            return "equal"

        def _emit_drain_trace(*, stage: str, queue_len_before_value: int, queue_len_after_value: int) -> None:
            logger.debug(
                "[RESPTRACE] queue_drain_%s source_trigger=%s run_id=%s turn_id=%s input_event_key=%s canonical_key=%s "
                "queue_len_before=%s queue_len_after=%s picked_origin=%s picked_turn_id=%s "
                "picked_input_event_key=%s picked_canonical_key=%s selected_pending_trigger=%s skipped_reason=%s "
                "enqueued_done_serial=%s enqueued_done_serial_relation=%s response_done_serial=%s drain_result=%s",
                stage,
                normalized_source_trigger,
                self._current_run_id() or "",
                trace_turn_id,
                trace_input_event_key or "unknown",
                trace_canonical_key,
                queue_len_before_value,
                queue_len_after_value,
                selected_pending_origin,
                selected_pending_turn_id,
                selected_pending_input_event_key,
                selected_pending_canonical_key,
                selected_pending_trigger,
                skipped_reason,
                selected_pending_enqueued_done_serial,
                selected_pending_serial_relation,
                getattr(self, "_response_done_serial", 0),
                drain_result,
            )

        current_state = getattr(self.state_manager, "state", InteractionState.IDLE)
        queue_len_before = len(getattr(self, "_response_create_queue", deque()) or ())
        trace_turn_id = self._current_turn_id_or_unknown()
        trace_input_event_key = str(getattr(self, "_current_input_event_key", "") or "").strip()
        trace_canonical_key = self._canonical_utterance_key(
            turn_id=trace_turn_id,
            input_event_key=trace_input_event_key,
        )
        _emit_drain_trace(
            stage="pre",
            queue_len_before_value=queue_len_before,
            queue_len_after_value=queue_len_before,
        )
        if (
            self._response_in_flight
            or self._audio_playback_busy
            or current_state == InteractionState.LISTENING
        ):
            drain_result = "state_or_flight_gate"
            _emit_drain_trace(
                stage="post",
                queue_len_before_value=queue_len_before,
                queue_len_after_value=queue_len_before,
            )
            return

        if self._pending_response_create is None and self._response_create_queue:
            self._dedupe_queued_confirmation_reminders()
            queue_len = len(self._response_create_queue)
            for _ in range(queue_len):
                queued = self._response_create_queue.popleft()
                metadata = self._extract_response_create_metadata(queued.get("event") or {})
                queued_trigger = self._extract_response_create_trigger(metadata)
                if not self._can_release_queued_response_create(queued_trigger, metadata):
                    self._response_create_queue.append(queued)
                    skipped_reason = "release_gate_blocked"
                    continue
                picked_origin = str(queued.get("origin") or "unknown")
                picked_turn_id = str(queued.get("turn_id") or "turn-unknown")
                picked_input_event_key = str(metadata.get("input_event_key") or "").strip() or "unknown"
                enqueued_done_serial_value = int(queued.get("enqueued_done_serial") or self._response_done_serial)
                selected_pending_origin = picked_origin
                selected_pending_turn_id = picked_turn_id
                selected_pending_input_event_key = picked_input_event_key
                selected_pending_canonical_key = self._canonical_utterance_key(
                    turn_id=picked_turn_id,
                    input_event_key=str(metadata.get("input_event_key") or "").strip(),
                )
                selected_pending_trigger = queued_trigger
                selected_pending_enqueued_done_serial = str(enqueued_done_serial_value)
                selected_pending_serial_relation = _serial_relation(enqueued_done_serial_value)
                self._pending_response_create = PendingResponseCreate(
                    websocket=queued["websocket"],
                    event=queued["event"],
                    origin=queued["origin"],
                    turn_id=str(queued.get("turn_id") or self._next_response_turn_id()),
                    created_at=time.monotonic(),
                    reason="legacy_queue_hydration",
                    record_ai_call=bool(queued.get("record_ai_call", False)),
                    debug_context=queued.get("debug_context"),
                    memory_brief_note=queued.get("memory_brief_note"),
                    queued_reminder_key=self._queued_response_reminder_key(queued),
                    enqueued_done_serial=enqueued_done_serial_value,
                )
                break
        if self._pending_response_create is None:
            drain_result = skipped_reason if skipped_reason != "none" else "no_pending"
            _emit_drain_trace(
                stage="post",
                queue_len_before_value=queue_len_before,
                queue_len_after_value=len(getattr(self, "_response_create_queue", deque()) or ()),
            )
            return

        pending = self._pending_response_create
        response_metadata = self._extract_response_create_metadata(pending.event)
        queued_trigger = self._extract_response_create_trigger(response_metadata)
        pending_input_event_key = str(response_metadata.get("input_event_key") or "").strip()
        selected_pending_origin = pending.origin
        selected_pending_turn_id = pending.turn_id
        selected_pending_input_event_key = pending_input_event_key or "unknown"
        selected_pending_canonical_key = self._canonical_utterance_key(
            turn_id=pending.turn_id,
            input_event_key=pending_input_event_key,
        )
        selected_pending_trigger = queued_trigger
        selected_pending_enqueued_done_serial = str(pending.enqueued_done_serial)
        selected_pending_serial_relation = _serial_relation(pending.enqueued_done_serial)
        if not self._can_release_queued_response_create(queued_trigger, response_metadata):
            if pending.reason != "legacy_queue_hydration":
                self._sync_pending_response_create_queue()
            logger.info(
                "Deferring queued response.create origin=%s trigger=%s while awaiting confirmation.",
                pending.origin,
                queued_trigger,
            )
            drain_result = "release_gate_blocked"
            _emit_drain_trace(
                stage="post",
                queue_len_before_value=queue_len_before,
                queue_len_after_value=len(getattr(self, "_response_create_queue", deque()) or ()),
            )
            return

        self._pending_response_create = None
        if pending.reason != "legacy_queue_hydration":
            self._sync_pending_response_create_queue()
        await self._send_response_create(
            pending.websocket,
            pending.event,
            origin=pending.origin,
            record_ai_call=pending.record_ai_call,
            debug_context=pending.debug_context,
            memory_brief_note=pending.memory_brief_note,
        )
        drain_result = "sent_to_send_response_create"
        _emit_drain_trace(
            stage="post",
            queue_len_before_value=queue_len_before,
            queue_len_after_value=len(getattr(self, "_response_create_queue", deque()) or ()),
        )


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

    def _is_awaiting_confirmation_phase(self) -> bool:
        if getattr(self, "_pending_confirmation_token", None) is not None:
            return True
        phase = getattr(self.orchestration_state, "phase", None)
        return phase == OrchestrationPhase.AWAITING_CONFIRMATION

    def _is_user_confirmation_trigger(self, trigger: str, metadata: dict[str, Any]) -> bool:
        normalized = str(trigger).strip().lower()
        if normalized == "text_message":
            return True
        source = str(metadata.get("source", "")).strip().lower()
        return source in {"user_audio", "user_text", "voice_confirmation", "text_confirmation"}

    def _can_release_queued_response_create(self, trigger: str, metadata: dict[str, Any]) -> bool:
        if not self._has_active_confirmation_token() and not self._is_awaiting_confirmation_phase():
            return True
        if self._is_user_confirmation_trigger(trigger, metadata):
            return True
        approval_flow = str(metadata.get("approval_flow", "")).strip().lower()
        return approval_flow in {"true", "1", "yes"}

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
        token = getattr(self, "_pending_confirmation_token", None)
        if token is None:
            return None
        if getattr(self, "_confirmation_state", ConfirmationState.IDLE) != ConfirmationState.AWAITING_DECISION:
            return None
        timeout_s = self._get_confirmation_timeout_s(token)
        if timeout_s <= 0.0:
            return None
        self._refresh_confirmation_pause()
        pause_reason = self._confirmation_pause_reason()
        remaining_s = self._confirmation_remaining_seconds()
        self._log_confirmation_timeout_check(token, remaining_s=remaining_s, pause_reason=pause_reason)
        if pause_reason is not None:
            return None
        if remaining_s > 0.0:
            return None
        self._set_confirmation_state(ConfirmationState.RESOLVING, reason="awaiting_decision_timeout")
        self._close_confirmation_token(outcome="awaiting_decision_timeout")
        self._awaiting_confirmation_completion = False
        self.orchestration_state.transition(
            OrchestrationPhase.IDLE,
            reason="confirmation timeout",
        )
        return token

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
        transition_lock = await self._confirmation_transition_guard()
        async with transition_lock:
            expired_token = self._expire_confirmation_awaiting_decision_timeout()
        if expired_token is None:
            return False
        logger.info(
            "CONFIRMATION_TIMEOUT token=%s kind=%s cause=awaiting_decision_timeout source_event=%s",
            expired_token.id,
            expired_token.kind,
            source_event,
        )
        action = expired_token.pending_action.action if expired_token.pending_action is not None else None
        self._log_structured_noop_event(
            outcome="timeout",
            reason="awaiting_decision_timeout",
            tool_name=getattr(action, "tool_name", None),
            action_id=getattr(action, "id", None),
            token_id=expired_token.id,
            idempotency_key=getattr(getattr(expired_token, "pending_action", None), "idempotency_key", None),
        )
        await self._emit_final_noop_user_text(
            websocket,
            outcome="timeout",
            reason="awaiting_decision_timeout",
        )
        return True

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
        pending_confirmation_active = self._has_active_confirmation_token() or self._is_awaiting_confirmation_phase()
        if not pending_confirmation_active or self._audio_playback_busy:
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
        await websocket.send(json.dumps(instruction_event))

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
        previous = getattr(self, "_confirmation_state", ConfirmationState.IDLE)
        token = getattr(self, "_pending_confirmation_token", None)
        if previous != state:
            self._log_confirmation_transition(previous, state, reason=reason, token=token)
        self._confirmation_state = state
        if state == ConfirmationState.AWAITING_DECISION and token is not None:
            metadata = token.metadata if isinstance(token.metadata, dict) else {}
            metadata.setdefault("awaiting_decision_since", time.monotonic())
            token.metadata = metadata

    def _get_confirmation_timeout_s(self, token: PendingConfirmationToken | None) -> float:
        if token is not None and token.kind == "research_permission":
            return max(0.0, float(getattr(self, "_research_permission_awaiting_decision_timeout_s", 60.0)))
        return max(0.0, float(getattr(self, "_confirmation_awaiting_decision_timeout_s", 20.0)))

    def _confirmation_pause_reason(self) -> str | None:
        if getattr(self, "_confirmation_speech_active", False):
            return "speech_active"
        if getattr(self, "_confirmation_asr_pending", False):
            return "asr_pending"
        return None

    def _is_confirmation_ttl_paused(self) -> bool:
        return self._confirmation_pause_reason() in {"speech_active", "asr_pending"}

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
        lock = getattr(self, "_confirmation_transition_lock", None)
        if lock is None:
            lock = asyncio.Lock()
            self._confirmation_transition_lock = lock
        return lock

    def _refresh_confirmation_pause(self) -> None:
        token = getattr(self, "_pending_confirmation_token", None)
        if token is None:
            self._confirmation_pause_started_at = None
            return
        now = time.monotonic()
        if self._confirmation_pause_reason() is None:
            started = getattr(self, "_confirmation_pause_started_at", None)
            if isinstance(started, (int, float)):
                self._confirmation_paused_accum_s += max(0.0, now - float(started))
            self._confirmation_pause_started_at = None
            return
        if self._confirmation_pause_started_at is None:
            self._confirmation_pause_started_at = now

    def _mark_confirmation_activity(self, *, reason: str, now: float | None = None) -> None:
        token = getattr(self, "_pending_confirmation_token", None)
        if token is None:
            return
        current = time.monotonic() if now is None else float(now)
        self._confirmation_last_activity_at = current
        self._refresh_confirmation_pause()
        remaining_s = self._confirmation_remaining_seconds(now=current)
        logger.info(
            "CONFIRMATION_ACTIVITY token=%s kind=%s reason=%s remaining_s=%.1f paused_reason=%s",
            token.id,
            token.kind,
            reason,
            remaining_s,
            self._confirmation_pause_reason() or "none",
        )

    def _confirmation_effective_elapsed_s(self, now: float | None = None) -> float:
        token = getattr(self, "_pending_confirmation_token", None)
        if token is None:
            return 0.0
        current = time.monotonic() if now is None else float(now)
        started_at = self._confirmation_token_created_at
        if not isinstance(started_at, (int, float)):
            started_at = token.created_at
        paused_s = float(getattr(self, "_confirmation_paused_accum_s", 0.0))
        pause_started = getattr(self, "_confirmation_pause_started_at", None)
        if isinstance(pause_started, (int, float)) and self._confirmation_pause_reason() is not None:
            paused_s += max(0.0, current - float(pause_started))
        return max(0.0, current - float(started_at) - paused_s)

    def _confirmation_remaining_seconds(self, *, now: float | None = None) -> float:
        token = getattr(self, "_pending_confirmation_token", None)
        if token is None:
            return 0.0
        timeout_s = self._get_confirmation_timeout_s(token)
        return max(0.0, timeout_s - self._confirmation_effective_elapsed_s(now=now))

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
        now = time.monotonic()
        token = PendingConfirmationToken(
            id=f"confirm_{uuid.uuid4().hex}",
            kind=kind,
            tool_name=tool_name,
            request=request,
            pending_action=pending_action,
            created_at=now,
            expiry_ts=expiry_ts,
            max_retries=max_retries,
            metadata=metadata or {},
        )
        self._pending_confirmation_token = token
        self._confirmation_token_created_at = now
        self._confirmation_last_activity_at = now
        self._confirmation_speech_active = False
        self._confirmation_asr_pending = False
        self._confirmation_pause_started_at = None
        self._confirmation_paused_accum_s = 0.0
        self._set_confirmation_state(ConfirmationState.PENDING_PROMPT, reason=f"token_created:{kind}")
        logger.info(
            "CONFIRMATION_TOKEN_CREATED token=%s kind=%s tool=%s",
            token.id,
            kind,
            tool_name or "none",
        )
        return token

    def _sync_confirmation_legacy_fields(self) -> None:
        token = getattr(self, "_pending_confirmation_token", None)
        if token is None:
            self._pending_action = None
            self._pending_research_request = None
            if getattr(self, "_confirmation_state", ConfirmationState.IDLE) != ConfirmationState.IDLE:
                self._set_confirmation_state(ConfirmationState.IDLE, reason="token_cleared")
            return
        self._pending_action = token.pending_action
        self._pending_research_request = token.request if token.kind in {"research_permission", "research_budget"} else None

    def _close_confirmation_token(self, *, outcome: str) -> None:
        token = getattr(self, "_pending_confirmation_token", None)
        if token is None:
            self._sync_confirmation_legacy_fields()
            return
        self._set_confirmation_state(ConfirmationState.COMPLETED, reason=f"outcome:{outcome}")
        logger.info(
            "CONFIRMATION_TOKEN_CLOSED token=%s kind=%s outcome=%s",
            token.id,
            token.kind,
            outcome,
        )
        if token.pending_action is not None:
            action = token.pending_action.action
            idempotency_key = token.pending_action.idempotency_key
            if outcome in {"accepted", "approved"}:
                self._record_intent_state(
                    action.tool_name,
                    action.tool_args,
                    "approved",
                    idempotency_key=idempotency_key,
                )
            elif outcome in {"rejected", "cancelled_by_stop_word", "cleared_pending_action"}:
                self._record_intent_state(
                    action.tool_name,
                    action.tool_args,
                    "denied",
                    idempotency_key=idempotency_key,
                )
            elif outcome in {"awaiting_decision_timeout", "expired", "retry_exhausted", "unclear_cancelled"}:
                self._record_intent_state(
                    action.tool_name,
                    action.tool_args,
                    "timeout",
                    idempotency_key=idempotency_key,
                )
            elif outcome == "replaced":
                self._record_intent_state(
                    action.tool_name,
                    action.tool_args,
                    "replaced",
                    idempotency_key=idempotency_key,
                )
            self._record_recent_confirmation_outcome(idempotency_key, outcome)
        pending_deferred_call = self._deferred_research_tool_call
        deferred_call_id = None
        deferred_action = "none"
        if (
            token.kind in {"research_permission", "research_budget"}
            and isinstance(pending_deferred_call, dict)
            and str(pending_deferred_call.get("token_id") or "") == token.id
        ):
            deferred_call_id = str(pending_deferred_call.get("call_id") or "") or "unknown"
            if outcome in {"accepted", "approved"}:
                deferred_action = "retain_for_dispatch"
            else:
                self._deferred_research_tool_call = None
                deferred_action = "cleared"
        logger.debug(
            "CONFIRMATION_TOKEN_CLOSE_PENDING_TOOL token=%s kind=%s outcome=%s call_id=%s action=%s",
            token.id,
            token.kind,
            outcome,
            deferred_call_id or "none",
            deferred_action,
        )
        self._research_pending_call_ids.clear()
        self._confirmation_last_closed_token = {
            "token": token,
            "kind": token.kind,
            "closed_at": time.monotonic(),
            "outcome": outcome,
        }
        if token.pending_action:
            self._presented_actions.discard(token.pending_action.action.id)
        self._confirmation_timeout_check_last_logged_at.pop(token.id, None)
        self._confirmation_timeout_check_last_pause_reason.pop(token.id, None)
        reminder_key = self._confirmation_reminder_key(token)
        reminder_tracker = getattr(self, "_confirmation_reminder_tracker", None)
        if not isinstance(reminder_tracker, dict):
            reminder_tracker = {}
            self._confirmation_reminder_tracker = reminder_tracker
        if reminder_key is not None:
            reminder_tracker.pop(reminder_key, None)
        if outcome in {"accepted", "rejected", "awaiting_decision_timeout", "replaced"}:
            latch_key = self._confirmation_prompt_latch_key(token)
            if latch_key is not None:
                prompt_latches = getattr(self, "_pending_confirmation_prompt_latches", None)
                if not isinstance(prompt_latches, set):
                    prompt_latches = set()
                    self._pending_confirmation_prompt_latches = prompt_latches
                prompt_latches.discard(latch_key)
        self._clear_queued_confirmation_reminder_markers(token)
        self._pending_confirmation_token = None
        self._confirmation_token_created_at = None
        self._confirmation_last_activity_at = None
        self._confirmation_speech_active = False
        self._confirmation_asr_pending = False
        self._confirmation_pause_started_at = None
        self._confirmation_paused_accum_s = 0.0
        self._sync_confirmation_legacy_fields()

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
        if getattr(self, "_pending_confirmation_token", None) is not None:
            return True
        legacy_pending = getattr(self, "_pending_action", None)
        if legacy_pending is None:
            return False
        phase = getattr(getattr(self, "orchestration_state", None), "phase", None)
        return phase == OrchestrationPhase.AWAITING_CONFIRMATION

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
        if key is None:
            logger.info(
                "CONFIRMATION_REMINDER_SUPPRESSED reason=%s suppress_reason=no_confirmation_key",
                reason,
            )
            return False, None, 0, None, "no_confirmation_key", None
        now = time.monotonic()
        max_reminders, schedule, min_interval_s = self._confirmation_reminder_schedule(token)
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
        reminder_tracker = getattr(self, "_confirmation_reminder_tracker", None)
        if not isinstance(reminder_tracker, dict):
            reminder_tracker = {}
            self._confirmation_reminder_tracker = reminder_tracker
        reminder_tracker[key] = {
            "count": sent_count,
            "last_sent_at": now,
            "last_reason": reason,
            "last_token_id": token.id if token is not None else None,
            "last_run_id": self._current_run_id(),
        }
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
        token_active = token is not None or self._pending_action is not None
        awaiting_phase = self._is_awaiting_confirmation_phase()
        if not token_active or not awaiting_phase:
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
        normalized = " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())
        if not normalized:
            return "unclear"
        yes_pattern = re.compile(
            r"^(yes|y|yeah|yep|sure|ok|okay|approve|proceed|go ahead|go ahead and do it|do it|please do|please go ahead)( please| thanks| thank you)?$"
        )
        no_pattern = re.compile(
            r"^(no|n|nope|nah|deny|do not|dont|don t|don t do that|don t do it|do not do that|do not do it)( please| thanks| thank you)?$"
        )
        cancel_pattern = re.compile(r"^(cancel|cancel that|never mind|nevermind|stop|ignore that|ignore it)( please)?$")
        if yes_pattern.fullmatch(normalized):
            return "yes"
        if no_pattern.fullmatch(normalized):
            return "no"
        if cancel_pattern.fullmatch(normalized):
            return "cancel"
        return "unclear"

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
        token = getattr(self, "_pending_confirmation_token", None)
        if token is not None:
            return False
        decision = self._parse_confirmation_decision(text)
        if decision not in {"yes", "no", "cancel"}:
            return False
        marker = getattr(self, "_confirmation_last_closed_token", None)
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
            "CONFIRMATION_GRACE_DECISION_APPLIED token=%s kind=%s decision=%s age_s=%.1f grace_s=%.1f",
            last_token.id,
            marker.get("kind"),
            decision,
            age_s,
            grace_s,
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
        logger.info(
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
        await websocket.send(json.dumps(cancel_event))

    async def _maybe_handle_approval_response(
        self, text: str, websocket: Any
    ) -> bool:
        transition_lock = await self._confirmation_transition_guard()
        async with transition_lock:
            token = getattr(self, "_pending_confirmation_token", None)
            if token is not None and (token.kind != "tool_governance" or token.pending_action is None):
                return False
            pending = token.pending_action if token is not None else self._pending_action
            if pending is None:
                return False
            if token is not None:
                self._mark_confirmation_activity(reason="approval_transcript")

            action = pending.action
            if await self._handle_stop_word(text, websocket, source="approval_response"):
                return True
            cooldown_remaining = self._tool_execution_cooldown_remaining()
            if cooldown_remaining > 0:
                if token is not None:
                    self._set_confirmation_state(ConfirmationState.RESOLVING, reason="stop_word_cooldown")
                await self._reject_tool_call(
                    action,
                    f"Tool execution paused for {cooldown_remaining:.0f}s due to stop word.",
                    websocket,
                    staging=pending.staging,
                    status="cancelled",
                )
                if token is not None:
                    self._close_confirmation_token(outcome="cancelled_by_stop_word")
                else:
                    self._clear_pending_action()
                self._awaiting_confirmation_completion = False
                self.orchestration_state.transition(
                    OrchestrationPhase.IDLE,
                    reason="confirmation cancelled by stop word",
                )
                return True

            normalized = text.strip()
            if not normalized:
                return False

            decision = self._parse_confirmation_decision(normalized)
            now = time.monotonic()
            if self._approval_expired_for_action(action, now=now, decision=decision):
                if token is not None:
                    self._set_confirmation_state(ConfirmationState.RESOLVING, reason="approval_expired")
                logger.info("CONFIRMATION_TIMEOUT tool=%s cause=expiry", action.tool_name)
                self._record_confirmation_timeout(action, cause="expiry")
                self._log_structured_noop_event(
                    outcome="timeout",
                    reason="approval_expired",
                    tool_name=action.tool_name,
                    action_id=action.id,
                    token_id=getattr(token, "id", None),
                    idempotency_key=pending.idempotency_key,
                )
                if token is not None:
                    self._close_confirmation_token(outcome="expired")
                else:
                    self._clear_pending_action()
                await self._emit_final_noop_user_text(
                    websocket,
                    outcome="timeout",
                    reason="approval_expired",
                )
                self._awaiting_confirmation_completion = False
                self.orchestration_state.transition(
                    OrchestrationPhase.IDLE,
                    reason="confirmation timeout",
                )
                return True

            logger.info(
                'CONFIRMATION_CANDIDATE transcript="%s" len=%d decision=%s',
                self._clip_text(normalized, limit=100),
                len(normalized),
                decision,
            )

            if decision == "yes":
                if token is not None:
                    self._set_confirmation_state(ConfirmationState.RESOLVING, reason="tool_confirmation_accepted")
                logger.info("CONFIRMATION_ACCEPTED tool=%s", action.tool_name)
                staging = pending.staging or self._stage_action(action)
                self._awaiting_confirmation_completion = True
                await self._execute_action(
                    action,
                    staging,
                    websocket,
                    idempotency_key=pending.idempotency_key,
                    force_no_tools_followup=True,
                    inject_no_tools_instruction=True,
                )
                if token is not None:
                    self._close_confirmation_token(outcome="accepted")
                else:
                    self._clear_pending_action()
                return True

            if decision in {"no", "cancel"}:
                if token is not None:
                    self._set_confirmation_state(ConfirmationState.RESOLVING, reason="tool_confirmation_rejected")
                logger.info("CONFIRMATION_REJECTED tool=%s", action.tool_name)
                await self._reject_tool_call(
                    action,
                    "User declined approval.",
                    websocket,
                    status="cancelled",
                    include_assistant_message=False,
                )
                if token is not None:
                    self._close_confirmation_token(outcome="rejected")
                else:
                    self._clear_pending_action()
                await self._emit_final_noop_user_text(
                    websocket,
                    outcome="cancelled",
                    reason="user_declined_approval",
                )
                self._awaiting_confirmation_completion = False
                self.orchestration_state.transition(
                    OrchestrationPhase.IDLE,
                    reason="confirmation rejected",
                )
                return True

            if decision == "unclear" and self._detect_alternate_intent_while_confirmation_pending(
                normalized,
                pending_action=pending,
            ):
                self._pending_alternate_intent_override = {
                    "detected_at": time.monotonic(),
                    "token_id": token.id if token is not None else "",
                    "pending_tool": action.tool_name,
                    "idempotency_key": pending.idempotency_key,
                }
                logger.info(
                    "CONFIRMATION_REMINDER_SUPPRESSED reason=%s suppress_reason=alternate_intent_detected idempotency_key=%s run_id=%s",
                    "non_decision_input",
                    pending.idempotency_key or "none",
                    self._current_run_id() or "none",
                )
                return False

            if token is not None:
                token.retry_count += 1
                retry_count = token.retry_count
                max_retries = token.max_retries
            else:
                pending.retry_count += 1
                retry_count = pending.retry_count
                max_retries = pending.max_retries

            if retry_count > max_retries:
                if token is not None:
                    self._set_confirmation_state(ConfirmationState.RESOLVING, reason="tool_confirmation_retry_exhausted")
                logger.info("CONFIRMATION_TIMEOUT tool=%s cause=retry_exhausted", action.tool_name)
                self._record_confirmation_timeout(action, cause="retry_exhausted")
                self._log_structured_noop_event(
                    outcome="timeout",
                    reason="retry_exhausted",
                    tool_name=action.tool_name,
                    action_id=action.id,
                    token_id=getattr(token, "id", None),
                    idempotency_key=pending.idempotency_key,
                )
                if token is not None:
                    self._close_confirmation_token(outcome="retry_exhausted")
                else:
                    self._clear_pending_action()
                await self._emit_final_noop_user_text(
                    websocket,
                    outcome="timeout",
                    reason="retry_exhausted",
                )
                self._awaiting_confirmation_completion = False
                self.orchestration_state.transition(
                    OrchestrationPhase.IDLE,
                    reason="confirmation timeout",
                )
                return True

            parsed_intent_category, parsed_intent_idempotency_key = self._classify_pending_confirmation_intent(
                normalized,
                pending_action=pending,
            )
            if token is not None:
                await self._maybe_emit_confirmation_reminder(
                    websocket,
                    reason="non_decision_input",
                    parsed_intent_category=parsed_intent_category,
                    parsed_intent_idempotency_key=parsed_intent_idempotency_key,
                )
            else:
                await self._maybe_emit_confirmation_reminder(
                    websocket,
                    reason="non_decision_input_legacy",
                    parsed_intent_category=parsed_intent_category,
                    parsed_intent_idempotency_key=parsed_intent_idempotency_key,
                )
            return True

    async def _maybe_handle_research_permission_response(self, text: str, websocket: Any) -> bool:
        token = getattr(self, "_pending_confirmation_token", None)
        if token is not None and token.kind != "research_permission":
            return False
        request = token.request if token is not None else self._pending_research_request
        if request is None:
            return False
        if token is not None:
            self._mark_confirmation_activity(reason="research_permission_transcript")

        token_metadata = token.metadata if token is not None and isinstance(token.metadata, dict) else {}
        if (
            self._is_domain_preview_request(text)
            and token is not None
            and not bool(token_metadata.get("domains_known"))
        ):
            domains = await self._discover_research_domains(request)
            domains_known = bool(domains)
            token_metadata["domains"] = domains
            token_metadata["domains_known"] = domains_known
            token.metadata = token_metadata
            if domains_known:
                await self.send_assistant_message(
                    f"Preview complete: likely sources include {', '.join(domains)}. Proceed with web lookup? (yes/no)",
                    websocket,
                    response_metadata={
                        "trigger": "confirmation_prompt",
                        "approval_flow": "true",
                        "confirmation_token": token.id,
                    },
                )
            else:
                await self.send_assistant_message(
                    "I still couldn't determine domains from a lightweight preview. Proceed with web lookup anyway? (yes/no)",
                    websocket,
                    response_metadata={
                        "trigger": "confirmation_prompt",
                        "approval_flow": "true",
                        "confirmation_token": token.id,
                    },
                )
            return True

        decision = self._parse_confirmation_decision(text)
        logger.info(
            "CONFIRMATION_DECISION token=%s kind=%s decision=%s",
            token.id if token is not None else "legacy",
            "research_permission",
            decision,
        )

        if decision == "yes":
            if token is not None:
                self._set_confirmation_state(ConfirmationState.RESOLVING, reason="research_permission_accepted")
            domain = self._extract_primary_research_domain(request.prompt)
            if domain and not self._is_research_domain_allowlisted(domain):
                normalized = self._trusted_domain_store.add_domain(domain, added_by="user")
                if normalized:
                    self._trusted_research_domains.add(normalized)
                    self._research_firecrawl_allowlist_domains = merge_allowlists(
                        self._research_firecrawl_allowlist_domains,
                        self._trusted_research_domains,
                    )
                    logger.info("[Allowlist] domain_added domain=%s", normalized)
            logger.info("[Research] Permission granted by user")
            self._mark_prior_research_permission_granted(request)
            token_id = token.id if token is not None else ""
            if token is not None:
                self._close_confirmation_token(outcome="accepted")
            else:
                self._pending_research_request = None
            if token_id and await self._dispatch_deferred_research_tool_call(websocket, token_id=token_id):
                return True
            await self._dispatch_research_request(request, websocket)
            return True

        if decision in {"no", "cancel"}:
            fingerprint = self._build_research_request_fingerprint(request)
            self._record_research_permission_outcome(fingerprint, approved=False)
            if token is not None:
                self._set_confirmation_state(ConfirmationState.RESOLVING, reason="research_permission_rejected")
            logger.info("[Research] Permission denied by user")
            if token is not None:
                self._close_confirmation_token(outcome="rejected")
            else:
                self._pending_research_request = None
            await self.send_assistant_message(
                "Understood — I won't perform web research right now.",
                websocket,
            )
            self.orchestration_state.transition(
                OrchestrationPhase.IDLE,
                reason="research permission rejected",
            )
            return True

        metadata = token.metadata if token is not None and isinstance(token.metadata, dict) else {}
        reprompt_count = int(metadata.get("unclear_reprompt_count", 0))
        max_reprompts = self._confirmation_unclear_max_reprompts
        if reprompt_count >= max_reprompts:
            if token is not None:
                self._set_confirmation_state(ConfirmationState.RESOLVING, reason="research_permission_timeout")
            logger.info("[Research] Permission cancelled after unclear decision")
            if token is not None:
                self._close_confirmation_token(outcome="unclear_cancelled")
            else:
                self._pending_research_request = None
            await self.send_assistant_message(
                "I couldn't confirm research permission, so I cancelled it. Ask again if you still want web lookup.",
                websocket,
            )
            self.orchestration_state.transition(
                OrchestrationPhase.IDLE,
                reason="research permission unclear cancel",
            )
            return True

        if token is not None:
            metadata["unclear_reprompt_count"] = reprompt_count + 1
            token.metadata = metadata

        domains = metadata.get("domains") if isinstance(metadata.get("domains"), list) else []
        domain_hint = f" for {', '.join(domains)}" if domains else ""
        await self.send_assistant_message(
            f"Please reply yes, no, or cancel{domain_hint}.",
            websocket,
            response_metadata={
                "trigger": "confirmation_reminder",
                "approval_flow": "true",
                "confirmation_token": token.id if token is not None else "",
            },
        )
        return True

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
        if not has_research_intent(text):
            return False

        token = getattr(self, "_pending_confirmation_token", None)
        if token is not None and token.kind in {"research_permission", "research_budget"}:
            logger.info("[Research] Duplicate intent ignored while confirmation prompt is pending")
            return True

        if token is not None and token.kind == "tool_governance" and token.tool_name == "perform_research":
            logger.info(
                "[Research] Duplicate intent ignored while governance confirmation is already pending"
            )
            return True

        if source == "input_audio_transcription":
            preview = self._clip_text(" ".join(text.split()), limit=80)
            logger.info(
                "[Research] Transcript intent detected (source=%s, preview=%s)",
                source,
                preview,
            )

        request = ResearchRequest(prompt=text, context={"source": source})
        request_fingerprint = self._build_research_request_fingerprint(request)
        outcome = self._get_research_permission_outcome(request_fingerprint)
        if outcome is False:
            logger.info("[Research] Suppressing re-prompt for recently denied request fingerprint")
            await self.send_assistant_message(
                "Understood — I won't perform web research for that request right now.",
                websocket,
            )
            return True
        logger.info("[Research] Requested from %s", source)

        if not self._research_can_run_now():
            existing_token = getattr(self, "_pending_confirmation_token", None)
            if existing_token is not None and existing_token.kind == "research_budget":
                logger.debug("[Research] Budget prompt already pending; duplicate intent ignored")
                return True
            budget_token = self._create_confirmation_token(
                kind="research_budget",
                tool_name="perform_research",
                request=request,
                max_retries=1,
                metadata={"approval_flow": True, "budget_remaining": self._research_budget_remaining()},
            )
            self._sync_confirmation_legacy_fields()
            await self.send_assistant_message(
                "Research budget is currently 0, so I can't run web research yet. Approve one over-budget research attempt? (yes/no)",
                websocket,
                response_metadata={
                    "trigger": "confirmation_prompt",
                    "approval_flow": "true",
                    "confirmation_token": budget_token.id,
                },
            )
            budget_token.prompt_sent = True
            self._set_confirmation_state(ConfirmationState.AWAITING_DECISION, reason="research_budget_prompt_sent")
            return True

        domains = await self._discover_research_domains(request)
        domains_known = bool(domains) and all(self._is_research_domain_allowlisted(domain) for domain in domains)
        permission_needed, reason = self.should_request_research_permission(
            request,
            source=source,
            domains_known=domains_known,
        )
        if permission_needed:
            provider = "firecrawl_fetch" if self._research_firecrawl_enabled else "openai_responses_web_search"
            if reason == "non_allowlisted_domain" and domains:
                logger.info("[Allowlist] permission_required domain=%s", domains[0])
            logger.info(
                "[Research] permission_required reason=%s provider=%s domains=%s domains_known=%s",
                reason,
                provider,
                domains,
                domains_known,
            )
            token = self._create_confirmation_token(
                kind="research_permission",
                tool_name="perform_research",
                request=request,
                max_retries=2,
                metadata={"approval_flow": True, "domains": domains, "domains_known": domains_known},
            )
            self._sync_confirmation_legacy_fields()
            logger.info(
                "[Research] Permission asked",
                extra={"event": "research_permission_prompt", "domains": domains, "domains_known": domains_known},
            )
            prompt_message = self._build_research_permission_prompt(domains=domains, domains_known=domains_known)
            await self.send_assistant_message(
                prompt_message,
                websocket,
                response_metadata={
                    "trigger": "confirmation_prompt",
                    "approval_flow": "true",
                    "confirmation_token": token.id,
                },
            )
            token.prompt_sent = True
            self._set_confirmation_state(ConfirmationState.AWAITING_DECISION, reason="research_prompt_sent")
            return True

        provider = "firecrawl_fetch" if self._research_firecrawl_enabled else "openai_responses_web_search"
        allowlisted = self._research_request_domains_allowlisted(request)
        logger.info(
            "[Research] permission_bypass reason=%s mode=%s allowlisted=%s provider=%s",
            reason,
            self._research_mode,
            allowlisted,
            provider,
        )
        await self._dispatch_research_request(request, websocket)
        return True

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
                asyncio.create_task(
                    self.websocket.send(json.dumps({"type": "input_audio_buffer.clear"}))
                )
            except Exception:
                logger.exception("Failed to send input_audio_buffer.clear")

        self.mic.start_recording()
        if self._response_create_queue:
            self._response_create_queue_drain_source = "playback_complete"
            asyncio.create_task(self._drain_response_create_queue())
        if self._pending_image_flush_after_playback and self._pending_image_stimulus:
            self._pending_image_flush_after_playback = False
            self._schedule_pending_image_flush("playback complete")

    async def run(self) -> None:
        websockets = _require_websockets()
        _, ConnectionClosedError = _resolve_websocket_exceptions(websockets)

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

        try:
            while True:
                try:
                    self._note_connection_attempt()
                    url = "wss://api.openai.com/v1/realtime?model=gpt-realtime"
                    _validate_outbound_endpoint(url)
                    headers = {"Authorization": f"Bearer {self.api_key}"}

                    async with websockets.connect(
                        url,
                        additional_headers=headers,
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
                        self._ws_close_started = False
                        self._ws_close_done = False

                        try:
                            self.loop.add_signal_handler(signal.SIGTERM, self.shutdown_handler)
                            self.loop.add_signal_handler(signal.SIGINT, self.shutdown_handler)
                        except (NotImplementedError, RuntimeError):
                            logger.debug("Signal handlers unavailable; relying on task cancellation.")

                        if self.prompts:
                            await self.send_initial_prompts(websocket)
                        else:
                            self.mic.start_recording()
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
        await websocket.send(json.dumps(session_update))
        startup_digest_note = self._build_startup_memory_digest_note()
        if startup_digest_note:
            await self._send_memory_brief_note(websocket, startup_digest_note)

    async def process_ws_messages(self, websocket: Any) -> None:
        websockets = _require_websockets()
        ConnectionClosed, _ = _resolve_websocket_exceptions(websockets)
        while True:
            try:
                message = await websocket.recv()
                event = json.loads(message)
                log_ws_event("Incoming", event)
                try:
                    await self.handle_event(event, websocket)
                except Exception:
                    event_type = str(event.get("type") or "unknown")
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
            except asyncio.CancelledError:
                log_info("WebSocket receive loop cancelled.")
                self._note_disconnect("websocket loop cancelled")
                break
            except ConnectionClosed:
                log_warning("⚠️ WebSocket connection lost.")
                self._note_disconnect("websocket connection closed")
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
        if policy_decision.action is ServerAutoCreatedDecisionAction.CANCEL_PRE_AUDIO:
            return ServerAutoArbitrationOutcome.CANCEL_PRE_AUDIO, policy_decision.reason_code
        if policy_decision.action is ServerAutoCreatedDecisionAction.DEFER:
            return ServerAutoArbitrationOutcome.DEFER, policy_decision.reason_code
        return ServerAutoArbitrationOutcome.ALLOW, policy_decision.reason_code

    async def handle_event(self, event: dict[str, Any], websocket: Any) -> None:
        event_type = event.get("type")
        await self._maybe_handle_confirmation_decision_timeout(
            websocket,
            source_event=str(event_type or "unknown"),
        )
        if event_type == "response.created":
            origin = self._consume_response_origin(event)
            log_info(f"response.created: origin={origin}")
            response = event.get("response") or {}
            response_id = response.get("id")
            self._active_response_id = str(response_id) if response_id else None
            self._active_response_origin = str(origin)
            self._active_response_input_event_key = None
            self._active_response_canonical_key = None
            pending_confirmation_active = self._has_active_confirmation_token() or self._is_awaiting_confirmation_phase()
            self._active_response_preference_guarded = False
            turn_id = self._current_turn_id_or_unknown()
            self._cancel_micro_ack(turn_id=turn_id, reason="response_created")
            if origin == "server_auto":
                expected_input_event_key = self._active_input_event_key_for_turn(turn_id)
                input_event_key = None
                pending_server_auto_keys = getattr(self, "_pending_server_auto_input_event_keys", None)
                if not isinstance(pending_server_auto_keys, deque):
                    pending_server_auto_keys = deque(maxlen=64)
                    self._pending_server_auto_input_event_keys = pending_server_auto_keys
                while pending_server_auto_keys:
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
                    await websocket.send(json.dumps(cancel_event))
                    return
                if not input_event_key:
                    input_event_key = self._next_synthetic_input_event_key("server_auto")
                self._active_server_auto_input_event_key = input_event_key
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
                current_input_event_key = str(getattr(self, "_current_input_event_key", "") or "").strip()
                if not current_input_event_key:
                    current_input_event_key = self._next_synthetic_input_event_key(origin)
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
            if origin != "server_auto":
                current_input_event_key = resolved_input_event_key
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
                    await websocket.send(json.dumps(cancel_event))
                return
            self._active_response_input_event_key = str(resolved_input_event_key or "").strip() or None
            self._active_response_canonical_key = lifecycle_canonical_key
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
                await websocket.send(json.dumps(cancel_event))
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
                await websocket.send(json.dumps(cancel_event))
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
            if self.audio_player:
                self.audio_player.start_response()
            self._audio_accum.clear()
            self._mic_receive_on_first_audio = True
            self.response_in_progress = True
            self._response_in_flight = True
            self._speaking_started = False
            self._assistant_reply_accum = ""
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
        elif event_type == "response.output_item.added":
            await self.handle_output_item_added(event)
        elif event_type == "response.function_call_arguments.delta":
            self.function_call_args += event.get("delta", "")
        elif event_type == "response.function_call_arguments.done":
            await self.handle_function_call(event, websocket)
        elif event_type == "response.text.delta":
            if self._is_active_response_guarded():
                return
            self._cancel_micro_ack(turn_id=self._current_turn_id_or_unknown(), reason="response_started")
            delta = event.get("delta", "")
            self.assistant_reply += delta
            self._assistant_reply_accum += delta
            self.state_manager.update_state(InteractionState.SPEAKING, "text output")
        elif event_type == "response.output_audio.delta":
            if self._is_active_response_guarded():
                return
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
                audio_log_level = logging.INFO
                if (
                    audio_decision.action is LifecycleDecisionAction.ALLOW
                    and audio_decision.reason == "state=audio_started"
                ):
                    audio_log_level = logging.DEBUG
                self._log_lifecycle_event(
                    turn_id=self._current_turn_id_or_unknown(),
                    input_event_key=getattr(self, "_active_response_input_event_key", None),
                    canonical_key=active_canonical_key,
                    origin=getattr(self, "_active_response_origin", "unknown"),
                    response_id=getattr(self, "_active_response_id", None),
                    decision=f"audio_delta_{audio_decision.action.value}:{audio_decision.reason}",
                    level=audio_log_level,
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
        elif event_type == "response.output_audio.done":
            await self.handle_audio_response_done()
            self.state_manager.update_state(InteractionState.IDLE, "audio output done")
        elif event_type == "response.output_audio_transcript.delta":
            if self._is_active_response_guarded():
                return
            delta = event.get("delta", "")
            self.assistant_reply += delta
            self._assistant_reply_accum += delta
        elif event_type == "response.output_audio_transcript.done":
            await self.handle_transcribe_response_done()
            self.state_manager.update_state(
                InteractionState.IDLE,
                "audio transcript done",
            )
        elif event_type == "response.done":
            await self.handle_response_done(event)
        elif event_type == "response.completed":
            await self.handle_response_completed(event)
        elif event_type == "error":
            await self.handle_error(event, websocket)
        elif event_type in {
            "conversation.item.input_audio_transcription.delta",
            "conversation.item.input_audio_transcription.partial",
        }:
            partial_text = event.get("delta")
            if not isinstance(partial_text, str) or not partial_text.strip():
                partial_text = self._extract_transcript(event) or ""
            transcript_input_event_key = self._resolve_input_event_key(event)
            with self._utterance_context_scope(
                turn_id=self._current_turn_id_or_unknown(),
                input_event_key=transcript_input_event_key,
            ):
                self._log_user_transcript(partial_text, final=False, event_type=event_type)
        elif event_type == "conversation.item.input_audio_transcription.completed":
            transcript = self._extract_transcript(event)
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
            )
            self._clear_stale_pending_server_auto_for_turn(
                turn_id=resolved_turn_id,
                active_input_event_key=input_event_key,
                reason="new_transcript_final",
            )
            memory_intent = self._is_memory_intent(transcript or "")
            logger.info(
                "memory_intent_classification run_id=%s source=input_audio_transcription input_event_key=%s memory_intent=%s",
                self._current_run_id() or "",
                input_event_key,
                str(memory_intent).lower(),
            )
            if memory_intent:
                self._set_response_obligation(
                    turn_id=self._current_turn_id_or_unknown(),
                    input_event_key=input_event_key,
                    source="input_audio_transcription",
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
            self._maybe_schedule_micro_ack(
                turn_id=resolved_turn_id,
                category=self._micro_ack_category_for_reason("transcript_finalized"),
                channel="voice",
                action=self._canonical_utterance_key(turn_id=resolved_turn_id, input_event_key=input_event_key),
                reason="transcript_finalized",
                expected_delay_ms=700,
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
            if transcript:
                if memory_intent:
                    decision_path = "upgraded_response" if str(getattr(self, "_active_response_origin", "")).strip().lower() == "server_auto" else "canonical_transcript"
                    logger.info(
                        "memory_intent_decision_path run_id=%s turn_id=%s input_event_key=%s decision_path=%s",
                        self._current_run_id() or "",
                        resolved_turn_id,
                        input_event_key,
                        decision_path,
                    )
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
                if await self._maybe_process_research_intent(
                    transcript,
                    websocket,
                    source="input_audio_transcription",
                ):
                    return
            elif confirmation_active and self._has_active_confirmation_token():
                token = self._pending_confirmation_token
                metadata = token.metadata if token is not None and isinstance(token.metadata, dict) else {}
                empty_reprompted = bool(metadata.get("empty_transcript_reprompted", False))
                if not empty_reprompted:
                    metadata["empty_transcript_reprompted"] = True
                    if token is not None:
                        token.metadata = metadata
                    await self.send_assistant_message(
                        "Sorry—was that a yes or no?",
                        websocket,
                        response_metadata={
                            "trigger": "confirmation_reminder",
                            "approval_flow": "true",
                            "confirmation_token": token.id if token is not None else "",
                        },
                    )
        elif event_type == "input_audio_buffer.speech_started":
            logger.info("Speech detected, listening...")
            manager = getattr(self, "_micro_ack_manager", None)
            talk_over_active = self.state_manager.state == InteractionState.SPEAKING or self._audio_playback_busy
            if manager is not None:
                manager.on_user_speech_started()
                if talk_over_active:
                    manager.mark_talk_over_incident()
                manager.cancel_all(reason="speech_active")
            if talk_over_active:
                self._clear_all_pending_response_creates(reason="talk_over_abort")
                turn_id = self._current_turn_id_or_unknown()
                input_event_key = str(getattr(self, "_current_input_event_key", "") or "").strip()
                self._clear_pending_response_contenders(
                    turn_id=turn_id,
                    input_event_key=input_event_key,
                    reason="talk_over_abort",
                )
                if bool(getattr(self, "_response_in_flight", False)):
                    cancel_event = {"type": "response.cancel"}
                    log_ws_event("Outgoing", cancel_event)
                    self._track_outgoing_event(cancel_event, origin="talk_over_abort")
                    try:
                        await websocket.send(json.dumps(cancel_event))
                    except Exception as exc:
                        logger.debug("talk_over_abort_cancel_failed turn_id=%s error=%s", turn_id, exc)
            self._utterance_counter += 1
            next_turn_id = self._next_response_turn_id()
            with self._utterance_context_scope(
                turn_id=next_turn_id,
                input_event_key="",
                utterance_seq=self._utterance_counter,
            ):
                pass
            self._active_utterance = {
                "utterance_id": self._utterance_counter,
                "t_start": time.monotonic(),
                "t_stop": None,
                "duration_ms": None,
                "rms_estimate": None,
                "peak_estimate": None,
                "transcript": "",
                "transcript_len": 0,
                "confirmation_candidate": False,
                "decision": "unclear",
                "suppressed": False,
            }
            self._log_utterance_envelope(event_type)
            if self._has_active_confirmation_token():
                self._confirmation_speech_active = True
                self._confirmation_asr_pending = True
                self._mark_confirmation_activity(reason="speech_started")
                logger.info(
                    "Speech started while awaiting confirmation; confirmation mode remains active."
                )
            else:
                self.orchestration_state.transition(
                    OrchestrationPhase.SENSE,
                    reason="speech started",
                )
            self.state_manager.update_state(InteractionState.LISTENING, "speech started")
        elif event_type == "input_audio_buffer.speech_stopped":
            manager = getattr(self, "_micro_ack_manager", None)
            if manager is not None:
                manager.on_user_speech_ended()
            if self._active_utterance is not None:
                self._active_utterance["t_stop"] = time.monotonic()
                self._active_utterance["duration_ms"] = (
                    self._active_utterance["t_stop"] - self._active_utterance["t_start"]
                ) * 1000.0
                self._refresh_utterance_audio_levels()
                self._log_utterance_envelope(event_type)
            if self._has_active_confirmation_token():
                self._confirmation_speech_active = False
                self._confirmation_asr_pending = True
                self._mark_confirmation_activity(reason="speech_stopped")
            await self.handle_speech_stopped(websocket)
            current_turn_id = self._current_turn_id_or_unknown()
            self._maybe_schedule_micro_ack(
                turn_id=current_turn_id,
                category=self._micro_ack_category_for_reason("speech_stopped"),
                channel="voice",
                action=self._canonical_utterance_key(
                    turn_id=current_turn_id,
                    input_event_key=self._active_input_event_key_for_turn(current_turn_id),
                ),
                reason="speech_stopped",
                expected_delay_ms=700,
            )
            self.state_manager.update_state(InteractionState.THINKING, "speech stopped")
        elif event_type == "input_audio_buffer.committed":
            self._refresh_utterance_audio_levels()
            self._log_utterance_envelope(event_type)
        elif event_type == "rate_limits.updated":
            expected_buckets = {"requests", "tokens"}
            rl = {
                str(bucket.get("name")): bucket
                for bucket in event.get("rate_limits", [])
                if isinstance(bucket, dict) and bucket.get("name")
            }
            self.rate_limits = rl

            present_buckets = expected_buckets.intersection(rl.keys())
            warned_missing_buckets = getattr(self, "_warned_missing_rate_limit_buckets", set())
            missing_buckets = expected_buckets.difference(present_buckets)
            previous_present_buckets = getattr(self, "_last_rate_limit_present_buckets", set())
            disappeared_buckets = previous_present_buckets.difference(present_buckets)
            should_warn_missing_buckets = (
                len(present_buckets) == 0
                or bool(expected_buckets.intersection(disappeared_buckets))
            )

            if should_warn_missing_buckets and missing_buckets != warned_missing_buckets:
                logger.warning(
                    "Realtime API rate_limits.updated missing expected bucket(s): %s",
                    ", ".join(sorted(missing_buckets)),
                )
                self._warned_missing_rate_limit_buckets = set(missing_buckets)
            elif not should_warn_missing_buckets and missing_buckets != warned_missing_buckets:
                self._warned_missing_rate_limit_buckets = set()

            self._last_rate_limit_present_buckets = set(present_buckets)

            has_requests_bucket = "requests" in rl
            if has_requests_bucket:
                self._missing_requests_bucket_consecutive_updates = 0
            else:
                self._missing_requests_bucket_consecutive_updates = (
                    getattr(self, "_missing_requests_bucket_consecutive_updates", 0) + 1
                )
                warning_interval = max(
                    1,
                    int(getattr(self, "_missing_requests_bucket_warning_interval", 3)),
                )
                if self._missing_requests_bucket_consecutive_updates % warning_interval == 0:
                    session_id = "unknown"
                    memory_manager = getattr(self, "_memory_manager", None)
                    if memory_manager is not None:
                        session_id_value = memory_manager.get_active_session_id()
                        if session_id_value:
                            session_id = str(session_id_value)
                    logger.warning(
                        "Realtime API requests bucket missing for %s consecutive rate_limits.updated events "
                        "(session_id=%s)",
                        self._missing_requests_bucket_consecutive_updates,
                        session_id,
                    )

            req = rl.get("requests", {})
            tok = rl.get("tokens", {})
            logger.info(
                "Rate limits: requests_bucket=%s | requests %s/%s reset=%s | tokens %s/%s reset=%s",
                "present" if has_requests_bucket else "missing",
                _format_rate_limit_field(req.get("remaining")),
                _format_rate_limit_field(req.get("limit")),
                _format_rate_limit_duration(req.get("reset_seconds")),
                _format_rate_limit_field(tok.get("remaining")),
                _format_rate_limit_field(tok.get("limit")),
                _format_rate_limit_duration(tok.get("reset_seconds")),
            )
        elif event_type == "session.updated":
            log_session_updated(event, full_payload=True)
            if not self.ready_event.is_set():
                logger.info("Realtime API ready to accept injections.")
                self.ready_event.set()

    async def handle_output_item_added(self, event: dict[str, Any]) -> None:
        item = event.get("item", {})
        if item.get("type") == "function_call":
            self.function_call = item
            self.function_call_args = ""

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
        await websocket.send(json.dumps(function_call_output))
        if inject_no_tools_instruction:
            await self._add_no_tools_follow_up_instruction(websocket)
        research_id = self._extract_research_id(result) if function_name == "perform_research" else None
        if self._mark_or_suppress_research_spoken_response(research_id):
            self.function_call = None
            self.function_call_args = ""
            return
        response_create_event: dict[str, Any] = {"type": "response.create"}
        if force_no_tools_followup:
            response_create_event["response"] = {"tool_choice": "none"}
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
        message = self._build_approval_prompt(
            action,
            action_summary=action_summary,
            confirm_prompt=confirm_prompt,
            confirm_reason=confirm_reason,
        )
        token = getattr(self, "_pending_confirmation_token", None)
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
        await websocket.send(json.dumps(function_call_output))
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
        await websocket.send(json.dumps(function_call_output))
        response_create_event = {"type": "response.create"}
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
        await websocket.send(json.dumps(function_call_output))
        response_create_event = {"type": "response.create"}
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
            include_response_create=True,
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
            await self.websocket.send(json.dumps(image_item))
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
        assistant_item = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": message}],
            },
        }
        log_ws_event("Outgoing", assistant_item)
        await websocket.send(json.dumps(assistant_item))
        if not speak:
            return

        metadata = {"origin": "assistant_message"}
        if response_metadata:
            metadata.update(response_metadata)
        context_hint = utterance_context or getattr(self, "_utterance_context", None) or self._build_utterance_context()
        metadata.setdefault("turn_id", context_hint.turn_id)
        if context_hint.input_event_key:
            metadata.setdefault("input_event_key", context_hint.input_event_key)
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
            {"type": "response.create", "response": {"metadata": metadata}},
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
        await websocket.send(json.dumps(error_item))

    async def handle_transcribe_response_done(self) -> None:
        if (
            self._active_response_confirmation_guarded
            and (self._has_active_confirmation_token() or self._pending_action is not None)
            and self._is_awaiting_confirmation_phase()
        ):
            self.assistant_reply = ""
            self._assistant_reply_accum = ""
            if self.websocket is not None:
                token = getattr(self, "_pending_confirmation_token", None)
                if token is not None:
                    if (
                        token is not None
                        and self._is_confirmation_prompt_latched(token)
                        and self._is_guarded_server_auto_reminder_allowed(reason="transcribe_response_done")
                    ):
                        await self._maybe_emit_confirmation_reminder(
                            self.websocket,
                            reason="transcribe_response_done",
                        )
                else:
                    if self._is_guarded_server_auto_reminder_allowed(
                        reason="transcribe_response_done_legacy"
                    ):
                        await self._maybe_emit_confirmation_reminder(
                            self.websocket,
                            reason="transcribe_response_done_legacy",
                        )
            self._active_response_confirmation_guarded = False
            self.response_in_progress = False
            self._recover_confirmation_guard_microphone("transcribe_response_done")
            self._maybe_enqueue_reflection("response transcript done")
            if self._pending_image_stimulus and not self._pending_image_flush_after_playback:
                await self._flush_pending_image_stimulus("response transcript done")
            logger.info("Finished handle_transcribe_response_done()")
            return

        if self.assistant_reply:
            log_info(f"Assistant Response: {self.assistant_reply}", style="bold blue")
            self.assistant_reply = ""

        self.response_in_progress = False
        self._maybe_enqueue_reflection("response transcript done")
        if self._pending_image_stimulus and not self._pending_image_flush_after_playback:
            await self._flush_pending_image_stimulus("response transcript done")
        logger.info("Finished handle_transcribe_response_done()")

    async def handle_audio_response_done(self) -> None:
        if self.response_start_time is not None:
            response_end_time = time.perf_counter()
            response_duration = response_end_time - self.response_start_time
            log_runtime("realtime_api_response", response_duration)
            self.response_start_time = None

        log_info("Assistant audio response complete.", style="bold blue")
        self.response_in_progress = False
        if self._audio_accum:
            if self.audio_player:
                self.audio_player.play_audio(bytes(self._audio_accum))
            self._audio_accum.clear()
        if self.audio_player:
            self.audio_player.close_response()
        manager = getattr(self, "_micro_ack_manager", None)
        if manager is not None:
            manager.mark_assistant_audio_ended()
        delivered_turn_id = self._current_turn_id_or_unknown()
        delivered_input_event_key = str(getattr(self, "_active_response_input_event_key", "") or "").strip()
        if delivered_input_event_key:
            self._set_response_delivery_state(
                turn_id=delivered_turn_id,
                input_event_key=delivered_input_event_key,
                state="delivered",
            )
            logger.debug(
                "response_delivery_marked run_id=%s input_event_key=%s",
                self._current_run_id() or "",
                delivered_input_event_key,
            )
        if self._pending_image_stimulus:
            if self.audio_player:
                self._pending_image_flush_after_playback = True
            else:
                await self._flush_pending_image_stimulus("audio response done")

    async def handle_response_done(self, event: dict[str, Any] | None = None) -> None:
        turn_id = self._current_turn_id_or_unknown()
        done_input_event_key = str(getattr(self, "_active_response_input_event_key", "") or "").strip()
        done_canonical_key = self._canonical_utterance_key(
            turn_id=turn_id,
            input_event_key=done_input_event_key,
        )
        active_response_id_before_clear = getattr(self, "_active_response_id", None)
        active_response_origin_before_clear = getattr(self, "_active_response_origin", "unknown")
        active_response_input_event_key_before_clear = getattr(self, "_active_response_input_event_key", None)
        active_response_canonical_key_before_clear = getattr(self, "_active_response_canonical_key", None)
        suppressed_turns = getattr(self, "_preference_recall_suppressed_turns", None)
        if not isinstance(suppressed_turns, set):
            suppressed_turns = set()
            self._preference_recall_suppressed_turns = suppressed_turns
        suppressed_input_event_keys = getattr(self, "_preference_recall_suppressed_input_event_keys", None)
        if not isinstance(suppressed_input_event_keys, set):
            suppressed_input_event_keys = set()
            self._preference_recall_suppressed_input_event_keys = suppressed_input_event_keys
        obligations = getattr(self, "_response_obligations", {})
        obligation_count_before_clear = len(obligations) if isinstance(obligations, dict) else 0
        obligation_key = self._response_obligation_key(turn_id=turn_id, input_event_key=done_input_event_key)
        obligation_present_before = isinstance(obligations, dict) and obligation_key in obligations
        pending_queue_len_before_clear = len(getattr(self, "_response_create_queue", deque()) or ())
        current_state_before_cleanup = getattr(self.state_manager, "state", InteractionState.IDLE)
        self._cancel_micro_ack(turn_id=turn_id, reason="response_done")
        self._response_done_serial += 1
        self.response_in_progress = False
        self._response_in_flight = False
        if done_input_event_key and bool(getattr(self, "_active_response_consumes_canonical_slot", True)):
            self._lifecycle_controller().on_response_done(done_canonical_key)
            self._log_lifecycle_event(
                turn_id=turn_id,
                input_event_key=done_input_event_key,
                canonical_key=done_canonical_key,
                origin=active_response_origin_before_clear,
                response_id=active_response_id_before_clear,
                decision="transition_done:response_done",
            )
            self._debug_dump_canonical_key_timeline(
                canonical_key=done_canonical_key,
                trigger="response_done",
            )
            existing_delivery_state = self._response_delivery_state(
                turn_id=turn_id,
                input_event_key=done_input_event_key,
            )
            if existing_delivery_state != "cancelled":
                self._set_response_delivery_state(
                    turn_id=turn_id,
                    input_event_key=done_input_event_key,
                    state="done",
                )
        suppressed_turn_id = self._current_turn_id_or_unknown()
        suppressed_turn_present_before = suppressed_turn_id in suppressed_turns
        logger.debug(
            "[RESPTRACE] response_done_cleanup_before run_id=%s active_response_id=%s "
            "active_response_origin=%s active_response_input_event_key=%s active_response_canonical_key=%s "
            "suppressed_turns_count=%s suppressed_keys_count=%s suppressed_turn_present_before=%s "
            "obligation_count_before=%s pending_queue_len=%s response_done_serial=%s state=%s",
            self._current_run_id() or "",
            active_response_id_before_clear,
            active_response_origin_before_clear,
            active_response_input_event_key_before_clear,
            active_response_canonical_key_before_clear,
            len(suppressed_turns),
            len(suppressed_input_event_keys),
            suppressed_turn_present_before,
            obligation_count_before_clear,
            pending_queue_len_before_clear,
            getattr(self, "_response_done_serial", 0),
            current_state_before_cleanup,
        )
        self._preference_recall_suppressed_turns.discard(suppressed_turn_id)
        active_input_event_key = str(getattr(self, "_active_server_auto_input_event_key", "") or "").strip()
        active_input_event_key_present_before = bool(active_input_event_key and active_input_event_key in suppressed_input_event_keys)
        if active_input_event_key:
            self._preference_recall_suppressed_input_event_keys.discard(active_input_event_key)
        was_confirmation_guarded = self._active_response_confirmation_guarded
        self._active_response_id = None
        self._active_response_confirmation_guarded = False
        self._active_response_preference_guarded = False
        self._active_response_origin = "unknown"
        self._active_response_input_event_key = None
        self._active_response_canonical_key = None
        self._active_server_auto_input_event_key = None
        obligations_after_cleanup = getattr(self, "_response_obligations", {})
        obligation_count_after_clear = len(obligations_after_cleanup) if isinstance(obligations_after_cleanup, dict) else 0
        obligation_present_after = (
            isinstance(obligations_after_cleanup, dict) and obligation_key in obligations_after_cleanup
        )
        resolved_input_event_key = done_input_event_key or "unknown"
        resolved_canonical_key = self._canonical_utterance_key(
            turn_id=turn_id,
            input_event_key=done_input_event_key,
        )
        logger.debug(
            "[RESPTRACE] response_done_cleanup_after run_id=%s removed_suppressed_turn=%s "
            "removed_suppressed_input_event_key=%s suppressed_turns_count=%s suppressed_keys_count=%s "
            "obligation_count_before=%s obligation_count_after=%s pending_queue_len=%s response_done_serial=%s state=%s "
            "turn_id=%s input_event_key=%s canonical_key=%s obligation_key=%s "
            "obligation_present_before=%s obligation_present_after=%s total_obligations=%s "
            "active_response_input_event_key=%s active_server_auto_input_event_key=%s",
            self._current_run_id() or "",
            suppressed_turn_present_before and suppressed_turn_id not in self._preference_recall_suppressed_turns,
            active_input_event_key_present_before
            and active_input_event_key not in self._preference_recall_suppressed_input_event_keys,
            len(suppressed_turns),
            len(suppressed_input_event_keys),
            obligation_count_before_clear,
            obligation_count_after_clear,
            len(getattr(self, "_response_create_queue", deque()) or ()),
            getattr(self, "_response_done_serial", 0),
            getattr(self.state_manager, "state", InteractionState.IDLE),
            turn_id,
            resolved_input_event_key,
            resolved_canonical_key,
            obligation_key,
            obligation_present_before,
            obligation_present_after,
            obligation_count_after_clear,
            str(getattr(self, "_active_response_input_event_key", "") or "").strip() or "none",
            str(getattr(self, "_active_server_auto_input_event_key", "") or "").strip() or "none",
        )
        current_state = getattr(self.state_manager, "state", InteractionState.IDLE)
        if current_state != InteractionState.LISTENING:
            self.state_manager.update_state(InteractionState.IDLE, "response done")
        else:
            logger.debug(
                "Skipping IDLE transition for response.done while still listening; deferring until speech stop."
            )
        logger.info("Received response.done event.")
        if event:
            self._last_response_metadata = {
                "event_type": event.get("type"),
                "response": event.get("response"),
                "rate_limits": self.rate_limits,
            }
        self._emit_preference_recall_skip_trace_if_needed(turn_id=self._current_response_turn_id)
        phase_is_awaiting_confirmation = (
            self.orchestration_state.phase == OrchestrationPhase.AWAITING_CONFIRMATION
        )
        has_active_confirmation_token = self._has_active_confirmation_token()
        is_awaiting_confirmation_phase = self._is_awaiting_confirmation_phase()
        confirmation_hold_active = (
            phase_is_awaiting_confirmation
            or has_active_confirmation_token
            or is_awaiting_confirmation_phase
        )
        if confirmation_hold_active:
            logger.info(
                "Confirmation state is holding phase progression; skipping REFLECT transition "
                "(phase=%s token_active=%s awaiting_phase=%s).",
                self.orchestration_state.phase,
                has_active_confirmation_token,
                is_awaiting_confirmation_phase,
            )
            if phase_is_awaiting_confirmation and has_active_confirmation_token:
                logger.info("Staying in AWAITING_CONFIRMATION until user accepts/rejects.")
            elif phase_is_awaiting_confirmation and self._awaiting_confirmation_completion:
                self._awaiting_confirmation_completion = False
                self.orchestration_state.transition(
                    OrchestrationPhase.IDLE,
                    reason="confirmation follow-up completed",
                )
        else:
            self.orchestration_state.transition(
                OrchestrationPhase.REFLECT,
                reason="response done",
            )
            self._enqueue_response_done_reflection("response done")
            self.orchestration_state.transition(
                OrchestrationPhase.IDLE,
                reason="response done reflection",
            )
        if was_confirmation_guarded:
            if self._has_active_confirmation_token():
                self._mark_confirmation_activity(reason="guarded_response_done")
            if (
                self.websocket is not None
                and self._has_active_confirmation_token()
                and self._is_awaiting_confirmation_phase()
                and self._is_confirmation_prompt_latched(self._pending_confirmation_token)
                and self._should_send_response_done_fallback_reminder()
                and self._is_guarded_server_auto_reminder_allowed(reason="response_done_fallback")
            ):
                await self._maybe_emit_confirmation_reminder(self.websocket, reason="response_done_fallback")
            self._recover_confirmation_guard_microphone("response_done")
        self._response_create_queue_drain_source = "response_done"
        await self._drain_response_create_queue()
        if self._pending_image_stimulus and not self._pending_image_flush_after_playback:
            await self._flush_pending_image_stimulus("response done")

    async def handle_response_completed(self, event: dict[str, Any] | None = None) -> None:
        turn_id = self._current_turn_id_or_unknown()
        done_input_event_key = str(getattr(self, "_active_response_input_event_key", "") or "").strip()
        self._cancel_micro_ack(turn_id=turn_id, reason="response_completed")
        self.response_in_progress = False
        self._response_in_flight = False
        if done_input_event_key and bool(getattr(self, "_active_response_consumes_canonical_slot", True)):
            done_canonical_key = self._canonical_utterance_key(
                turn_id=turn_id,
                input_event_key=done_input_event_key,
            )
            self._lifecycle_controller().on_response_done(done_canonical_key)
            self._log_lifecycle_event(
                turn_id=turn_id,
                input_event_key=done_input_event_key,
                canonical_key=done_canonical_key,
                origin=getattr(self, "_active_response_origin", "unknown"),
                response_id=getattr(self, "_active_response_id", None),
                decision="transition_done:response_completed",
            )
            self._debug_dump_canonical_key_timeline(
                canonical_key=done_canonical_key,
                trigger="response_completed",
            )
            existing_delivery_state = self._response_delivery_state(
                turn_id=turn_id,
                input_event_key=done_input_event_key,
            )
            if existing_delivery_state != "cancelled":
                self._set_response_delivery_state(
                    turn_id=turn_id,
                    input_event_key=done_input_event_key,
                    state="done",
                )
        self._preference_recall_suppressed_turns.discard(self._current_turn_id_or_unknown())
        active_input_event_key = str(getattr(self, "_active_server_auto_input_event_key", "") or "").strip()
        if active_input_event_key:
            self._preference_recall_suppressed_input_event_keys.discard(active_input_event_key)
        self._active_response_id = None
        self._active_response_confirmation_guarded = False
        self._active_response_preference_guarded = False
        self._active_response_origin = "unknown"
        self._active_server_auto_input_event_key = None
        current_state = getattr(self.state_manager, "state", InteractionState.IDLE)
        if current_state != InteractionState.LISTENING:
            self.state_manager.update_state(InteractionState.IDLE, "response completed")
        else:
            logger.debug(
                "Skipping IDLE transition for response.completed while still listening; deferring until speech stop."
            )
        logger.info("Received response.completed event.")
        if event:
            self._last_response_metadata = {
                "event_type": event.get("type"),
                "response": event.get("response"),
                "rate_limits": self.rate_limits,
            }
        self._emit_preference_recall_skip_trace_if_needed(turn_id=self._current_response_turn_id)
        phase_is_awaiting_confirmation = (
            self.orchestration_state.phase == OrchestrationPhase.AWAITING_CONFIRMATION
        )
        has_active_confirmation_token = self._has_active_confirmation_token()
        is_awaiting_confirmation_phase = self._is_awaiting_confirmation_phase()
        confirmation_hold_active = (
            phase_is_awaiting_confirmation
            or has_active_confirmation_token
            or is_awaiting_confirmation_phase
        )
        if confirmation_hold_active:
            logger.info(
                "Confirmation state is holding phase progression; skipping REFLECT transition "
                "(phase=%s token_active=%s awaiting_phase=%s).",
                self.orchestration_state.phase,
                has_active_confirmation_token,
                is_awaiting_confirmation_phase,
            )
            if phase_is_awaiting_confirmation and has_active_confirmation_token:
                logger.info("Staying in AWAITING_CONFIRMATION until user accepts/rejects.")
            elif phase_is_awaiting_confirmation and self._awaiting_confirmation_completion:
                self._awaiting_confirmation_completion = False
                self.orchestration_state.transition(
                    OrchestrationPhase.IDLE,
                    reason="confirmation follow-up completed",
                )
        else:
            self.orchestration_state.transition(
                OrchestrationPhase.REFLECT,
                reason="response completed",
            )
            self._maybe_enqueue_reflection("response completed")
            self.orchestration_state.transition(
                OrchestrationPhase.IDLE,
                reason="reflection enqueued",
            )
        self._response_create_queue_drain_source = "explicit_caller"
        await self._drain_response_create_queue()
        if self._pending_image_stimulus and not self._pending_image_flush_after_playback:
            await self._flush_pending_image_stimulus("response completed")

    async def handle_error(self, event: dict[str, Any], websocket: Any) -> None:
        error_message = event.get("error", {}).get("message", "")
        if "no active response found" in error_message.lower():
            logger.info(
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
        await websocket.send(json.dumps(event))

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
        await self.websocket.send(json.dumps(text_event))
        if request_response:
            await self._stimuli_coordinator.enqueue(
                trigger="text_message",
                metadata={
                    "text_length": len(text_message),
                    "bypass_limits": bypass_response_suppression,
                    "safety_override": safety_override,
                },
                priority=self._get_injection_priority("text_message"),
            )

    async def maybe_request_response(self, trigger: str, metadata: dict[str, Any]) -> None:
        if not self.websocket:
            log_warning("Skipping injected response (%s): websocket unavailable.", trigger)
            return

        if self._has_active_confirmation_token():
            if not self._is_user_confirmation_trigger(trigger, metadata):
                logger.info(
                    "Suppressing duplicate non-user response request trigger=%s phase=%s pending_action=%s",
                    trigger,
                    getattr(getattr(self.orchestration_state, "phase", None), "value", "unknown"),
                    self._pending_action is not None,
                )
                return

        trigger_config = self._injection_response_triggers.get(trigger, {})
        trigger_cooldown_s = float(
            trigger_config.get("cooldown_s", self._injection_response_cooldown_s)
        )
        trigger_max_per_minute = int(
            trigger_config.get("max_per_minute", self._max_injection_responses_per_minute)
        )
        trigger_priority = int(trigger_config.get("priority", 0))
        trigger_timestamps = self._injection_response_trigger_timestamps.setdefault(
            trigger, deque()
        )
        bypass_limits = bool(metadata.get("bypass_limits", False))

        if self.response_in_progress and not bypass_limits:
            if trigger == "image_message":
                self._queue_pending_image_stimulus(trigger, metadata)
                return
            logger.info("Skipping injected response (%s): response already in progress.", trigger)
            return

        if (
            self.state_manager.state not in (InteractionState.IDLE, InteractionState.LISTENING)
            and not bypass_limits
        ):
            logger.info(
                "Skipping injected response (%s): invalid state %s.",
                trigger,
                self.state_manager.state.value,
            )
            return

        if trigger_cooldown_s > 0.0 and not bypass_limits:
            now = time.monotonic()
            last_ts = trigger_timestamps[-1] if trigger_timestamps else None
            if last_ts is not None and now - last_ts < trigger_cooldown_s:
                logger.info(
                    "Skipping injected response (%s): trigger cooldown %.2fs remaining.",
                    trigger,
                    trigger_cooldown_s - (now - last_ts),
                )
                return

        if trigger_max_per_minute > 0 and not bypass_limits:
            now = time.monotonic()
            while trigger_timestamps and now - trigger_timestamps[0] > 60:
                trigger_timestamps.popleft()
            if len(trigger_timestamps) >= trigger_max_per_minute:
                logger.info(
                    "Skipping injected response (%s): trigger rate limit %s/minute reached.",
                    trigger,
                    trigger_max_per_minute,
                )
                return

        bypass_global_limits = trigger_priority > 0 or bypass_limits

        if self._injection_response_cooldown_s > 0.0 and not bypass_global_limits:
            now = time.monotonic()
            last_ts = (
                self._injection_response_timestamps[-1]
                if self._injection_response_timestamps
                else None
            )
            if last_ts is not None and now - last_ts < self._injection_response_cooldown_s:
                logger.info(
                    "Skipping injected response (%s): cooldown %.2fs remaining.",
                    trigger,
                    self._injection_response_cooldown_s - (now - last_ts),
                )
                return

        if self._max_injection_responses_per_minute > 0 and not bypass_global_limits:
            now = time.monotonic()
            while self._injection_response_timestamps and now - self._injection_response_timestamps[0] > 60:
                self._injection_response_timestamps.popleft()
            if len(self._injection_response_timestamps) >= self._max_injection_responses_per_minute:
                logger.info(
                    "Skipping injected response (%s): rate limit %s/minute reached.",
                    trigger,
                    self._max_injection_responses_per_minute,
                )
                return

        if self.rate_limits:
            for limit_name in ("requests", "responses"):
                rate_limit = self.rate_limits.get(limit_name)
                if not rate_limit:
                    continue
                remaining = rate_limit.get("remaining")
                if remaining is not None and remaining <= 0:
                    logger.info(
                        "Skipping injected response (%s): %s rate limit exhausted.",
                        trigger,
                        limit_name,
                    )
                    return

        bypass_budget = trigger == "text_message"
        if not self._allow_ai_call(f"injection:{trigger}", bypass=bypass_budget):
            logger.info("Skipping injected response (%s): AI call budget exhausted.", trigger)
            return
        response_metadata = {
            "trigger": trigger,
            "priority": str(trigger_priority),
            "stimulus": json.dumps(metadata, sort_keys=True),
            "origin": "injection",
        }
        if bool(metadata.get("safety_override", False)):
            response_metadata["safety_override"] = "true"
        if self._has_active_confirmation_token():
            response_metadata["approval_flow"] = "true"
        memory_brief_note = self._consume_pending_memory_brief_note()
        response_create_event = {
            "type": "response.create",
            "response": {"metadata": response_metadata},
        }
        log_info(
            f"Requesting injected response for {trigger} with metadata {response_metadata}."
        )
        sent_now = await self._send_response_create(
            self.websocket,
            response_create_event,
            origin="injection",
            record_ai_call=True,
            memory_brief_note=memory_brief_note,
        )
        if sent_now:
            now = time.monotonic()
            self._injection_response_timestamps.append(now)
            trigger_timestamps.append(now)

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
            self.loop.create_task(self._flush_pending_image_stimulus(reason))
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
                        await websocket.send(json.dumps(audio_event))
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
        with self._shutdown_lock:
            if self._shutdown_requested:
                logger.debug("Shutdown already in progress; ignoring duplicate shutdown signal.")
                return
            self._shutdown_requested = True

        logger.info("Received shutdown signal. Initiating graceful shutdown...")
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self._request_shutdown)
            for task in asyncio.all_tasks(self.loop):
                task.cancel()

    def _request_shutdown(self) -> None:
        self.exit_event.set()
        if self.websocket:
            self.loop.create_task(self._close_websocket("signal"))

    async def _close_websocket(
        self,
        reason: str,
        *,
        websocket: Any | None = None,
        timeout_s: float | None = None,
    ) -> None:
        ws = websocket or self.websocket
        if not ws:
            return
        close_timeout_s = (
            getattr(self, "_websocket_close_timeout_s", 5.0)
            if timeout_s is None
            else timeout_s
        )
        async with self._ws_close_lock:
            if self._ws_close_started or self._ws_close_done:
                return
            self._ws_close_started = True
        try:
            await asyncio.wait_for(ws.close(), timeout=close_timeout_s)
            logger.info("WebSocket closed (%s).", reason)
        except asyncio.TimeoutError:
            logger.warning("Timed out closing WebSocket (%s).", reason)
        except Exception as exc:
            logger.warning("Failed to close WebSocket (%s): %s", reason, exc)
        finally:
            async with self._ws_close_lock:
                self._ws_close_started = False
                self._ws_close_done = True
