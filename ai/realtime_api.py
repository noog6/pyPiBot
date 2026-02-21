"""Realtime API controller for OpenAI realtime connections."""

from __future__ import annotations

import asyncio
import audioop
import base64
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import hashlib
from io import BytesIO
import json
import os
import random
import re
import signal
import threading
import time
import uuid
from typing import Any
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
from ai.reflection import ReflectionCoordinator, ReflectionContext
from ai.stimuli_coordinator import StimuliCoordinator
from ai.governance import ActionPacket, GovernanceLayer, build_tool_specs
from ai.orchestration import OrchestrationPhase, OrchestrationState
from ai.event_bus import Event, EventBus
from ai.event_injector import EventInjector
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


@dataclass
class PendingAction:
    action: ActionPacket
    staging: dict[str, Any]
    original_intent: str
    created_at: float
    retry_count: int = 0
    max_retries: int = 2


class ConfirmationState(str, Enum):
    IDLE = "idle"
    PENDING_PROMPT = "pending_prompt"
    AWAITING_DECISION = "awaiting_decision"
    RESOLVING = "resolving"
    COMPLETED = "completed"


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
    required_fields = ("tier", "reversible", "cost_hint", "safety_tags")
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
        self._audio_input_device_name = input_cfg.get("device_name")
        self._audio_output_device_name = output_cfg.get("device_name")
        self.exit_event = asyncio.Event()
        self.mic = AsyncMicrophone(
            input_device_name=self._audio_input_device_name,
            debug_list_devices=False,
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
        self._pending_response_create_origins: deque[str] = deque(maxlen=64)
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
        tool_specs = build_tool_specs(governance_cfg.get("tool_specs") or {})
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
        self._confirmation_unclear_max_reprompts = max(
            0,
            int(realtime_cfg.get("confirmation_unclear_max_reprompts", 1)),
        )
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
        self._active_response_origin = "unknown"
        self._last_executed_tool_call: dict[str, Any] | None = None

        realtime_cfg = config.get("realtime") or {}
        self._debug_vad = bool(realtime_cfg.get("debug_vad", False))
        self._awaiting_confirmation_allowed_sources = (
            self._load_awaiting_confirmation_source_policy(realtime_cfg)
        )
        self._utterance_counter = 0
        self._active_utterance: dict[str, Any] | None = None
        self._recent_input_levels: deque[dict[str, float]] = deque(maxlen=256)
        self._minimum_non_confirmation_duration_ms = int(
            realtime_cfg.get("minimum_non_confirmation_duration_ms", 120)
        )
        self._vad_turn_detection = self._resolve_vad_turn_detection(config)

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
            self._queue_response_origin(origin)

    def _queue_response_origin(self, origin: str | None) -> None:
        normalized_origin = str(origin).strip() if origin else "unknown"
        self._pending_response_create_origins.append(normalized_origin)

    def _consume_response_origin(self) -> str:
        if self._pending_response_create_origins:
            return self._pending_response_create_origins.popleft()
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
        if event.source == "battery":
            request_response = self._should_request_battery_response(event, fallback=request_response)
            bypass_response_suppression = request_response and self._is_battery_query_context_active()
        elif event.request_response is not None:
            request_response = event.request_response
        self._log_injection_event(event, request_response)
        self._send_text_message(
            message,
            request_response=request_response,
            bypass_response_suppression=bypass_response_suppression,
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
    ) -> None:
        if not self.loop:
            logger.debug("Unable to send message; event loop unavailable.")
            return
        future = asyncio.run_coroutine_threadsafe(
            self.send_text_message_to_conversation(
                message,
                request_response=request_response,
                bypass_response_suppression=bypass_response_suppression,
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
            return event.content, True
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
        return f"{event.source} event: {event.metadata}", True


    def _should_request_battery_response(self, event: Event, *, fallback: bool = False) -> bool:
        metadata = event.metadata or {}
        severity = str(metadata.get("severity", "info"))
        event_type = str(metadata.get("event_type", "status"))
        transition = str(metadata.get("transition", "steady"))

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
        self._last_user_input_text = clean_text
        self._last_user_input_time = time.monotonic()
        self._last_user_input_source = source
        if self._is_battery_status_query(clean_text):
            self._last_user_battery_query_time = self._last_user_input_time
        self._prepare_turn_memory_brief(clean_text, source=source)

    def _should_skip_turn_memory_retrieval(self, user_text: str) -> bool:
        text = user_text.strip()
        if len(text) < int(getattr(self, "_memory_retrieval_min_user_chars", 12)):
            return True
        if len(text.split()) < int(getattr(self, "_memory_retrieval_min_user_tokens", 3)):
            return True
        noisy = {"ok", "okay", "thanks", "thank you", "yes", "no", "cool", "nice", "got it"}
        return text.lower() in noisy

    def _prepare_turn_memory_brief(self, user_text: str, *, source: str) -> None:
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
                scope=str(getattr(self, "_memory_retrieval_scope", MemoryScope.USER_GLOBAL.value)),
                session_id=manager.get_active_session_id(),
            )
            retrieval_debug = manager.get_last_turn_retrieval_debug_metadata()
            if retrieval_debug:
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
                    "Turn memory retrieval audit source=%s mode=%s lexical_candidates=%s semantic_candidates=%s semantic_scored=%s candidates_without_ready_embedding=%s candidates_below_influence_threshold=%s candidates_semantic_applied=%s selected=%s fallback_reason=%s latency_ms=%s truncated=%s truncation_count=%s dedupe_count=%s semantic_provider=%s semantic_model=%s semantic_query_timeout_ms=%s semantic_query_timeout_ms_used=%s semantic_query_duration_ms=%s semantic_query_embed_elapsed_ms=%s semantic_result_status=%s semantic_error_code=%s semantic_error_class=%s semantic_failure_class=%s semantic_scoring_skipped_reason=%s query_fingerprint_hash=%s query_fingerprint_length=%s%s",
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
                    retrieval_debug.get("semantic_result_status"),
                    retrieval_debug.get("semantic_error_code"),
                    retrieval_debug.get("semantic_error_class"),
                    retrieval_debug.get("semantic_failure_class"),
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

    def _build_approval_prompt(self, action: ActionPacket) -> str:
        packet = self._format_action_packet(action)
        if action.tier >= 3:
            return (
                "Approval required for this action.\n"
                f"{packet}\n"
                "Approve? Reply exactly: 'Yes, do it now' / no / modify."
            )
        return f"Approval required for this action.\n{packet}\nApprove? (yes / no / modify)"

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

    def _build_tool_call_fingerprint(self, tool_name: str, args: dict[str, Any]) -> str:
        payload = json.dumps({"tool": tool_name, "args": args}, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _record_confirmation_timeout(self, action: ActionPacket, cause: str) -> None:
        fingerprint = self._build_tool_call_fingerprint(action.tool_name, action.tool_args)
        self._confirmation_timeout_markers[fingerprint] = time.monotonic()
        self._confirmation_timeout_causes[fingerprint] = cause

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

    async def _send_response_create(
        self,
        websocket: Any,
        response_create_event: dict[str, Any],
        *,
        origin: str,
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
        if self._response_in_flight or self._audio_playback_busy:
            if self._response_in_flight:
                logger.info("Deferring response.create while another response is active (origin=%s).", origin)
            else:
                logger.info("Deferring response.create while audio playback is still active (origin=%s).", origin)
            self._response_create_queue.append(
                {
                    "websocket": websocket,
                    "event": response_create_event,
                    "origin": origin,
                    "record_ai_call": record_ai_call,
                    "debug_context": debug_context,
                    "memory_brief_note": memory_brief_note,
                    "enqueued_done_serial": self._response_done_serial,
                }
            )
            return False
        if memory_brief_note:
            try:
                await self._send_memory_brief_note(websocket, memory_brief_note)
            except Exception as exc:  # pragma: no cover - defensive fail-open
                logger.warning("Memory brief injection skipped due to error: %s", exc)
        log_ws_event("Outgoing", response_create_event)
        self._track_outgoing_event(response_create_event, origin=origin)
        await websocket.send(json.dumps(response_create_event))
        self._last_response_create_ts = now
        self._response_in_flight = True
        if record_ai_call:
            self._record_ai_call()
        return True

    async def _drain_response_create_queue(self) -> None:
        current_state = getattr(self.state_manager, "state", InteractionState.IDLE)
        if (
            self._response_in_flight
            or self._audio_playback_busy
            or current_state == InteractionState.LISTENING
            or not self._response_create_queue
        ):
            return
        queue_len = len(self._response_create_queue)
        for _ in range(queue_len):
            queued = self._response_create_queue.popleft()
            if self._is_stale_queued_response_create(queued):
                logger.info(
                    "Dropping stale queued response.create origin=%s after newer responses completed.",
                    queued.get("origin"),
                )
                continue
            response_metadata = self._extract_response_create_metadata(queued.get("event") or {})
            queued_trigger = self._extract_response_create_trigger(response_metadata)
            if not self._can_release_queued_response_create(queued_trigger, response_metadata):
                self._response_create_queue.append(queued)
                logger.info(
                    "Deferring queued response.create origin=%s trigger=%s while awaiting confirmation.",
                    queued.get("origin"),
                    queued_trigger,
                )
                continue
            await self._send_response_create(
                queued["websocket"],
                queued["event"],
                origin=queued["origin"],
                record_ai_call=bool(queued.get("record_ai_call", False)),
                debug_context=queued.get("debug_context"),
                memory_brief_note=queued.get("memory_brief_note"),
            )
            return

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
        expired_token = self._expire_confirmation_awaiting_decision_timeout()
        if expired_token is None:
            return False
        logger.info(
            "CONFIRMATION_TIMEOUT token=%s kind=%s cause=awaiting_decision_timeout source_event=%s",
            expired_token.id,
            expired_token.kind,
            source_event,
        )
        await self.send_assistant_message(
            "I didn't get a clear yes/no in time, so I cancelled that request.",
            websocket,
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
        return getattr(self, "_pending_confirmation_token", None) is not None

    def _is_token_approval_flow_metadata(self, metadata: dict[str, Any]) -> bool:
        approval_flow = str(metadata.get("approval_flow", "")).strip().lower()
        if approval_flow in {"true", "1", "yes"}:
            return True
        token = getattr(self, "_pending_confirmation_token", None)
        if token is None:
            return False
        token_id = str(metadata.get("confirmation_token", "")).strip()
        return token_id != "" and token_id == token.id

    async def _send_confirmation_reminder(self, websocket: Any, *, reason: str) -> None:
        token = getattr(self, "_pending_confirmation_token", None)
        if token is None or token.reminder_sent:
            return
        token.reminder_sent = True
        if self._confirmation_state == ConfirmationState.PENDING_PROMPT:
            self._set_confirmation_state(ConfirmationState.AWAITING_DECISION, reason="reminder_sent")
        logger.info(
            "CONFIRMATION_NON_DECISION_RESPONSE_SUPPRESSED origin=%s response_id=%s reason=%s",
            self._active_response_origin,
            self._active_response_id or "unknown",
            reason,
        )
        await self.send_assistant_message(
            "Please reply with: yes or no.",
            websocket,
            response_metadata={
                "trigger": "confirmation_reminder",
                "approval_flow": "true",
                "confirmation_token": token.id if token is not None else "",
            },
        )

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

        now = time.monotonic()
        if action.expiry_ts is not None and now > action.expiry_ts:
            if token is not None:
                self._set_confirmation_state(ConfirmationState.RESOLVING, reason="approval_expired")
            logger.info("CONFIRMATION_TIMEOUT tool=%s cause=expiry", action.tool_name)
            self._record_confirmation_timeout(action, cause="expiry")
            await self.send_assistant_message(
                "Approval window expired. Please ask again if you still want this.",
                websocket,
            )
            if token is not None:
                self._close_confirmation_token(outcome="expired")
            else:
                self._clear_pending_action()
            self._awaiting_confirmation_completion = False
            self.orchestration_state.transition(
                OrchestrationPhase.IDLE,
                reason="confirmation timeout",
            )
            return True

        normalized = text.strip()
        if not normalized:
            return False

        decision = self._parse_confirmation_decision(normalized)
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
            )
            if token is not None:
                self._close_confirmation_token(outcome="rejected")
            else:
                self._clear_pending_action()
            self._awaiting_confirmation_completion = False
            self.orchestration_state.transition(
                OrchestrationPhase.IDLE,
                reason="confirmation rejected",
            )
            return True

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
            await self.send_assistant_message(
                "I couldn't confirm this action. Please ask again if you still want it.",
                websocket,
            )
            if token is not None:
                self._close_confirmation_token(outcome="retry_exhausted")
            else:
                self._clear_pending_action()
            self._awaiting_confirmation_completion = False
            self.orchestration_state.transition(
                OrchestrationPhase.IDLE,
                reason="confirmation timeout",
            )
            return True

        if token is not None:
            await self._send_confirmation_reminder(websocket, reason="non_decision_input")
        else:
            await self.send_assistant_message("Please reply with: yes or no.", websocket)
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
            request.context = {**request.context, "over_budget_approved": True}
            if await self._dispatch_deferred_research_tool_call(websocket, token_id=token_id):
                return True
            await self._dispatch_research_request(request, websocket)
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
                await self.handle_event(event, websocket)
            except asyncio.CancelledError:
                log_info("WebSocket receive loop cancelled.")
                self._note_disconnect("websocket loop cancelled")
                break
            except ConnectionClosed:
                log_warning("⚠️ WebSocket connection lost.")
                self._note_disconnect("websocket connection closed")
                break

    async def handle_event(self, event: dict[str, Any], websocket: Any) -> None:
        event_type = event.get("type")
        await self._maybe_handle_confirmation_decision_timeout(
            websocket,
            source_event=str(event_type or "unknown"),
        )
        if event_type == "response.created":
            origin = self._consume_response_origin()
            log_info(f"response.created: origin={origin}")
            response = event.get("response") or {}
            response_id = response.get("id")
            self._active_response_id = str(response_id) if response_id else None
            self._active_response_origin = str(origin)
            pending_confirmation_active = self._has_active_confirmation_token() or self._is_awaiting_confirmation_phase()
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
            if self._active_response_confirmation_guarded:
                return
            delta = event.get("delta", "")
            self.assistant_reply += delta
            self._assistant_reply_accum += delta
            self.state_manager.update_state(InteractionState.SPEAKING, "text output")
        elif event_type == "response.output_audio.delta":
            if self._active_response_confirmation_guarded:
                return
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
            if self._active_response_confirmation_guarded:
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
            self._log_user_transcript(partial_text, final=False, event_type=event_type)
        elif event_type == "conversation.item.input_audio_transcription.completed":
            transcript = self._extract_transcript(event)
            self._log_user_transcript(transcript or "", final=True, event_type=event_type)
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
            self._utterance_counter += 1
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
            if self._has_active_confirmation_token() or self._is_awaiting_confirmation_phase():
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
            if self._active_utterance is not None:
                self._active_utterance["t_stop"] = time.monotonic()
                self._active_utterance["duration_ms"] = (
                    self._active_utterance["t_stop"] - self._active_utterance["t_start"]
                ) * 1000.0
                self._refresh_utterance_audio_levels()
                self._log_utterance_envelope(event_type)
            if self._has_active_confirmation_token() or self._is_awaiting_confirmation_phase():
                self._confirmation_speech_active = False
                self._confirmation_asr_pending = True
                self._mark_confirmation_activity(reason="speech_stopped")
            await self.handle_speech_stopped(websocket)
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
                    function_call_output = {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": json.dumps(
                                {
                                    "status": "blocked_by_research_permission",
                                    "message": "Research request was denied for this query; do not retry until user re-approves.",
                                    "tool": function_name,
                                }
                            ),
                        },
                    }
                    log_ws_event("Outgoing", function_call_output)
                    self._track_outgoing_event(function_call_output)
                    await websocket.send(json.dumps(function_call_output))
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
                function_call_output = {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps(
                            {
                                "status": "waiting_for_permission",
                                "token": token.id if token is not None else "",
                                "message": "Permission required; ask user first.",
                                "tool": function_name,
                            }
                        ),
                    },
                }
                log_ws_event("Outgoing", function_call_output)
                self._track_outgoing_event(function_call_output)
                await websocket.send(json.dumps(function_call_output))
                self.function_call = None
                self.function_call_args = ""
                return
            pending_action = token.pending_action if token is not None else self._pending_action
            if pending_action is not None:
                logger.info(
                    "Function call outcome: suppressed pending confirmation | incoming=%s pending=%s",
                    function_name,
                    pending_action.action.tool_name,
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
                function_call_output = {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps(
                            {
                                "status": "redundant",
                                "message": "Duplicate tool call ignored; use previous tool results.",
                                "tool": function_name,
                            }
                        ),
                    },
                }
                log_ws_event("Outgoing", function_call_output)
                self._track_outgoing_event(function_call_output)
                await websocket.send(json.dumps(function_call_output))
                self.function_call = None
                self.function_call_args = ""
                return
            if self._is_suppressed_after_confirmation_timeout(function_name, args):
                logger.info(
                    "Function call outcome: suppressed confirmation-timeout debounce | tool=%s call_id=%s",
                    function_name,
                    call_id,
                )
                function_call_output = {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps(
                            {
                                "status": "suppressed_after_confirmation_timeout",
                                "message": "Tool call suppressed after confirmation timeout; retry shortly.",
                                "tool": function_name,
                            }
                        ),
                    },
                }
                log_ws_event("Outgoing", function_call_output)
                self._track_outgoing_event(function_call_output)
                await websocket.send(json.dumps(function_call_output))
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
            cooldown_remaining = self._tool_execution_cooldown_remaining()
            if cooldown_remaining > 0:
                await self._reject_tool_call(
                    action,
                    f"Tool execution paused for {cooldown_remaining:.0f}s due to stop word.",
                    websocket,
                    status="cancelled",
                )
                return
            staging = self._stage_action(action)
            if dry_run_requested:
                await self._send_dry_run_output(action, staging, websocket)
                return
            if not staging["valid"]:
                await self._reject_tool_call(
                    action,
                    "Argument validation failed.",
                    websocket,
                    staging=staging,
                    status="invalid_arguments",
                )
                return
            decision = self._governance.review(action)
            approved_via_prior_permission = (
                decision.needs_confirmation
                and self._consume_prior_research_permission_marker_if_fresh(action)
            )
            should_execute = decision.approved or approved_via_prior_permission
            if should_execute:
                decision_reason = (
                    "approved_via_prior_research_permission"
                    if approved_via_prior_permission
                    else decision.reason
                )
                final_execution_decision = "execute"
                if self._debug_governance_decisions:
                    log_info(
                        f"🛡️ Governance decision: {decision.status} ({decision.reason}) {action.summary()}"
                    )
                if approved_via_prior_permission and self._debug_governance_decisions:
                    log_info(
                        "🛡️ Governance decision: approved "
                        f"(approved_via_prior_research_permission) {action.summary()}"
                    )
                if self._debug_governance_decisions:
                    logger.info(
                        "Function call outcome: executing tool | tool=%s call_id=%s decision_reason=%s",
                        function_name,
                        call_id,
                        decision_reason,
                    )
                self.orchestration_state.transition(
                    OrchestrationPhase.ACT,
                    reason=f"function_call {function_name}",
                )
                await self._execute_action(action, staging, websocket)
            elif decision.needs_confirmation:
                final_execution_decision = "request_confirmation"
                if self._debug_governance_decisions:
                    log_info(
                        f"🛡️ Governance decision: {decision.status} ({decision.reason}) {action.summary()}"
                    )
                action.requires_confirmation = True
                action.expiry_ts = time.monotonic() + self._approval_timeout_s
                pending_action = PendingAction(
                    action=action,
                    staging=staging,
                    original_intent=self._last_user_input_text or "",
                    created_at=time.monotonic(),
                )
                token = self._create_confirmation_token(
                    kind="tool_governance",
                    tool_name=action.tool_name,
                    pending_action=pending_action,
                    expiry_ts=action.expiry_ts,
                    max_retries=pending_action.max_retries,
                    metadata={"approval_flow": True},
                )
                self._pending_action = pending_action
                self._sync_confirmation_legacy_fields()
                await self._request_tool_confirmation(action, decision.reason, websocket, staging)
                token.prompt_sent = True
                self._set_confirmation_state(ConfirmationState.AWAITING_DECISION, reason="tool_prompt_sent")
            else:
                final_execution_decision = "reject"
                if self._debug_governance_decisions:
                    log_info(
                        f"🛡️ Governance decision: {decision.status} ({decision.reason}) {action.summary()}"
                    )
                await self._reject_tool_call(
                    action,
                    decision.reason,
                    websocket,
                    staging=staging,
                    status="denied",
                )
            logger.info(
                "Governance review summary | call_id=%s tool=%s initial_status=%s "
                "initial_reason=%s prior_permission_override=%s "
                "final_execution_decision=%s",
                call_id,
                function_name,
                decision.status,
                decision.reason,
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
                await self.send_error_message_to_assistant(error_message, websocket)
        else:
            error_message = (
                f"Function '{function_name}' not found. Add to function_map in tools.py."
            )
            log_error(error_message)
            result = {"error": error_message}
            await self.send_error_message_to_assistant(error_message, websocket)

        self._tool_call_records.append(
            {
                "name": function_name,
                "call_id": call_id,
                "args": args,
                "result": result,
                "action_packet": action.to_payload() if action else None,
                "staging": staging,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
        self._last_tool_call_results = list(self._tool_call_records)

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
        self._governance.record_execution(action)
        self._presented_actions.discard(action.id)

    async def _request_tool_confirmation(
        self,
        action: ActionPacket,
        reason: str,
        websocket: Any,
        staging: dict[str, Any],
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
        message = self._build_approval_prompt(action)
        token = getattr(self, "_pending_confirmation_token", None)
        response_metadata = {"trigger": "confirmation_prompt", "approval_flow": "true"}
        if token is not None:
            response_metadata["confirmation_token"] = token.id
        await self.send_assistant_message(
            message,
            websocket,
            response_metadata=response_metadata,
        )
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
        payload = {
            "status": "awaiting_confirmation",
            "tool": action.tool_name,
            "message": "Awaiting explicit yes/no confirmation for pending action.",
            "action_packet": action.to_payload(),
        }
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
    ) -> None:
        packet = self._format_action_packet(action)
        message = f"Tool execution not run.\n{packet}\nReason: {reason}."
        await self.send_assistant_message(message, websocket)
        function_call_output = {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": action.id,
                "output": json.dumps(
                    {
                        "status": status,
                        "message": message,
                        "tool": action.tool_name,
                        "tool_metadata": self._governance.describe_tool(action.tool_name),
                        "action_packet": action.to_payload(),
                        "staging": staging,
                    }
                ),
            },
        }
        log_ws_event("Outgoing", function_call_output)
        self._track_outgoing_event(function_call_output)
        await websocket.send(json.dumps(function_call_output))
        response_create_event = {"type": "response.create"}
        await self._send_response_create(websocket, response_create_event, origin="tool_output")
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
        token = getattr(self, "_pending_confirmation_token", None)
        if token is not None or self._is_awaiting_confirmation_phase():
            metadata.setdefault("approval_flow", "true")
            if token is not None:
                metadata.setdefault("confirmation_token", token.id)
        await self._send_response_create(
            websocket,
            {"type": "response.create", "response": {"metadata": metadata}},
            origin="assistant_message",
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
                if self._has_active_confirmation_token():
                    await self._send_confirmation_reminder(
                        self.websocket,
                        reason="transcribe_response_done",
                    )
                else:
                    await self.send_assistant_message(
                        "Please reply with: yes or no.",
                        self.websocket,
                        response_metadata={"trigger": "confirmation_reminder", "approval_flow": "true"},
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
        if self._pending_image_stimulus:
            if self.audio_player:
                self._pending_image_flush_after_playback = True
            else:
                await self._flush_pending_image_stimulus("audio response done")

    async def handle_response_done(self, event: dict[str, Any] | None = None) -> None:
        self._response_done_serial += 1
        self.response_in_progress = False
        self._response_in_flight = False
        was_confirmation_guarded = self._active_response_confirmation_guarded
        self._active_response_id = None
        self._active_response_confirmation_guarded = False
        self._active_response_origin = "unknown"
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
            if self.websocket is not None and self._has_active_confirmation_token() and self._is_awaiting_confirmation_phase():
                await self._send_confirmation_reminder(self.websocket, reason="response_done_fallback")
            self._recover_confirmation_guard_microphone("response_done")
        await self._drain_response_create_queue()
        if self._pending_image_stimulus and not self._pending_image_flush_after_playback:
            await self._flush_pending_image_stimulus("response done")

    async def handle_response_completed(self, event: dict[str, Any] | None = None) -> None:
        self.response_in_progress = False
        self._response_in_flight = False
        self._active_response_id = None
        self._active_response_confirmation_guarded = False
        self._active_response_origin = "unknown"
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
        await self._drain_response_create_queue()
        if self._pending_image_stimulus and not self._pending_image_flush_after_playback:
            await self._flush_pending_image_stimulus("response completed")

    async def handle_error(self, event: dict[str, Any], websocket: Any) -> None:
        error_message = event.get("error", {}).get("message", "")
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
                },
                priority=self._get_injection_priority("text_message"),
            )

    async def maybe_request_response(self, trigger: str, metadata: dict[str, Any]) -> None:
        if not self.websocket:
            log_warning("Skipping injected response (%s): websocket unavailable.", trigger)
            return

        if self._has_active_confirmation_token() or self._is_awaiting_confirmation_phase():
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
        if self._has_active_confirmation_token() or self._is_awaiting_confirmation_phase():
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
        if self._has_active_confirmation_token() or self._is_awaiting_confirmation_phase():
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
