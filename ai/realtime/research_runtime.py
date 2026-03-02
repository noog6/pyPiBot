"""Runtime service for research permission and intent flows."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
import hashlib
import json
import time
from typing import Any, Protocol

from ai.realtime.confirmation import ConfirmationState
from ai.orchestration import OrchestrationPhase
from core.logging import logger
from services.research import ResearchRequest, has_research_intent
from storage.trusted_domains import merge_allowlists


class ResearchRuntimeAPI(Protocol):
    ...


@dataclass
class ResearchRuntime:
    api: ResearchRuntimeAPI

    def build_research_fingerprint(self, *, query: str, source: str | None) -> str:
        api = self.api
        normalized = {
            "query": api._normalize_research_query_text(query),
            "source": api._normalize_research_source_text(source),
        }
        payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def prune_research_permission_outcomes(self, now: float | None = None) -> None:
        api = self.api
        now_ts = time.monotonic() if now is None else now
        outcomes = getattr(api, "_research_permission_outcomes", None)
        if not isinstance(outcomes, dict):
            api._research_permission_outcomes = {}
            api._research_suppressed_fingerprints = {}
            return
        ttl_s = float(getattr(api, "_research_permission_outcome_ttl_s", 20.0))
        expired = [
            fingerprint
            for fingerprint, payload in outcomes.items()
            if now_ts - float(payload.get("recorded_at", 0.0)) > ttl_s
        ]
        for fingerprint in expired:
            outcomes.pop(fingerprint, None)
            if isinstance(getattr(api, "_research_suppressed_fingerprints", None), dict):
                api._research_suppressed_fingerprints.pop(fingerprint, None)

    def record_research_permission_outcome(self, fingerprint: str, *, approved: bool) -> None:
        api = self.api
        self.prune_research_permission_outcomes()
        if not isinstance(getattr(api, "_research_permission_outcomes", None), dict):
            api._research_permission_outcomes = {}
        api._research_permission_outcomes[fingerprint] = {
            "approved": approved,
            "recorded_at": time.monotonic(),
        }
        if not isinstance(getattr(api, "_research_suppressed_fingerprints", None), dict):
            api._research_suppressed_fingerprints = {}
        if approved:
            api._research_suppressed_fingerprints.pop(fingerprint, None)

    async def maybe_handle_research_permission_response(self, text: str, websocket: Any) -> bool:
        api = self.api
        token = getattr(api, "_pending_confirmation_token", None)
        if token is not None and token.kind != "research_permission":
            return False
        request = token.request if token is not None else api._pending_research_request
        if request is None:
            return False
        if token is not None:
            api._mark_confirmation_activity(reason="research_permission_transcript")
    
        token_metadata = token.metadata if token is not None and isinstance(token.metadata, dict) else {}
        if (
            api._is_domain_preview_request(text)
            and token is not None
            and not bool(token_metadata.get("domains_known"))
        ):
            domains = await api._discover_research_domains(request)
            domains_known = bool(domains)
            token_metadata["domains"] = domains
            token_metadata["domains_known"] = domains_known
            token.metadata = token_metadata
            if domains_known:
                await api.send_assistant_message(
                    f"Preview complete: likely sources include {', '.join(domains)}. Proceed with web lookup? (yes/no)",
                    websocket,
                    response_metadata={
                        "trigger": "confirmation_prompt",
                        "approval_flow": "true",
                        "confirmation_token": token.id,
                    },
                )
            else:
                await api.send_assistant_message(
                    "I still couldn't determine domains from a lightweight preview. Proceed with web lookup anyway? (yes/no)",
                    websocket,
                    response_metadata={
                        "trigger": "confirmation_prompt",
                        "approval_flow": "true",
                        "confirmation_token": token.id,
                    },
                )
            return True
    
        decision = api._parse_confirmation_decision(text)
        logger.info(
            "CONFIRMATION_DECISION token=%s kind=%s decision=%s",
            token.id if token is not None else "legacy",
            "research_permission",
            decision,
        )
    
        if decision == "yes":
            if token is not None:
                api._set_confirmation_state(ConfirmationState.RESOLVING, reason="research_permission_accepted")
            domain = api._extract_primary_research_domain(request.prompt)
            if domain and not api._is_research_domain_allowlisted(domain):
                normalized = api._trusted_domain_store.add_domain(domain, added_by="user")
                if normalized:
                    api._trusted_research_domains.add(normalized)
                    api._research_firecrawl_allowlist_domains = merge_allowlists(
                        api._research_firecrawl_allowlist_domains,
                        api._trusted_research_domains,
                    )
                    logger.info("[Allowlist] domain_added domain=%s", normalized)
            logger.info("[Research] Permission granted by user")
            api._mark_prior_research_permission_granted(request)
            token_id = token.id if token is not None else ""
            if token is not None:
                api._close_confirmation_token(outcome="accepted")
            else:
                api._pending_research_request = None
            if token_id and await api._dispatch_deferred_research_tool_call(websocket, token_id=token_id):
                return True
            await api._dispatch_research_request(request, websocket)
            return True
    
        if decision in {"no", "cancel"}:
            fingerprint = api._build_research_request_fingerprint(request)
            api._record_research_permission_outcome(fingerprint, approved=False)
            if token is not None:
                api._set_confirmation_state(ConfirmationState.RESOLVING, reason="research_permission_rejected")
            logger.info("[Research] Permission denied by user")
            if token is not None:
                api._close_confirmation_token(outcome="rejected")
            else:
                api._pending_research_request = None
            await api.send_assistant_message(
                "Understood — I won't perform web research right now.",
                websocket,
            )
            api.orchestration_state.transition(
                OrchestrationPhase.IDLE,
                reason="research permission rejected",
            )
            return True
    
        metadata = token.metadata if token is not None and isinstance(token.metadata, dict) else {}
        reprompt_count = int(metadata.get("unclear_reprompt_count", 0))
        max_reprompts = api._confirmation_unclear_max_reprompts
        if reprompt_count >= max_reprompts:
            if token is not None:
                api._set_confirmation_state(ConfirmationState.RESOLVING, reason="research_permission_timeout")
            logger.info("[Research] Permission cancelled after unclear decision")
            if token is not None:
                api._close_confirmation_token(outcome="unclear_cancelled")
            else:
                api._pending_research_request = None
            await api.send_assistant_message(
                "I couldn't confirm research permission, so I cancelled it. Ask again if you still want web lookup.",
                websocket,
            )
            api.orchestration_state.transition(
                OrchestrationPhase.IDLE,
                reason="research permission unclear cancel",
            )
            return True
    
        if token is not None:
            metadata["unclear_reprompt_count"] = reprompt_count + 1
            token.metadata = metadata
    
        domains = metadata.get("domains") if isinstance(metadata.get("domains"), list) else []
        domain_hint = f" for {', '.join(domains)}" if domains else ""
        await api.send_assistant_message(
            f"Please reply yes, no, or cancel{domain_hint}.",
            websocket,
            response_metadata={
                "trigger": "confirmation_reminder",
                "approval_flow": "true",
                "confirmation_token": token.id if token is not None else "",
            },
        )
        return True
    

    async def maybe_process_research_intent(self, text: str, websocket: Any, *, source: str) -> bool:
        api = self.api
        if not has_research_intent(text):
            return False
    
        token = getattr(api, "_pending_confirmation_token", None)
        if token is not None and token.kind in {"research_permission", "research_budget"}:
            logger.info("[Research] Duplicate intent ignored while confirmation prompt is pending")
            return True
    
        if token is not None and token.kind == "tool_governance" and token.tool_name == "perform_research":
            logger.info(
                "[Research] Duplicate intent ignored while governance confirmation is already pending"
            )
            return True
    
        if source == "input_audio_transcription":
            preview = api._clip_text(" ".join(text.split()), limit=80)
            logger.info(
                "[Research] Transcript intent detected (source=%s, preview=%s)",
                source,
                preview,
            )
    
        request = ResearchRequest(prompt=text, context={"source": source})
        request_fingerprint = api._build_research_request_fingerprint(request)
        outcome = api._get_research_permission_outcome(request_fingerprint)
        if outcome is False:
            logger.info("[Research] Suppressing re-prompt for recently denied request fingerprint")
            await api.send_assistant_message(
                "Understood — I won't perform web research for that request right now.",
                websocket,
            )
            return True
        logger.info("[Research] Requested from %s", source)
    
        if not api._research_can_run_now():
            existing_token = getattr(api, "_pending_confirmation_token", None)
            if existing_token is not None and existing_token.kind == "research_budget":
                logger.debug("[Research] Budget prompt already pending; duplicate intent ignored")
                return True
            budget_token = api._create_confirmation_token(
                kind="research_budget",
                tool_name="perform_research",
                request=request,
                max_retries=1,
                metadata={"approval_flow": True, "budget_remaining": api._research_budget_remaining()},
            )
            api._sync_confirmation_legacy_fields()
            await api.send_assistant_message(
                "Research budget is currently 0, so I can't run web research yet. Approve one over-budget research attempt? (yes/no)",
                websocket,
                response_metadata={
                    "trigger": "confirmation_prompt",
                    "approval_flow": "true",
                    "confirmation_token": budget_token.id,
                },
            )
            budget_token.prompt_sent = True
            api._set_confirmation_state(ConfirmationState.AWAITING_DECISION, reason="research_budget_prompt_sent")
            return True
    
        domains = await api._discover_research_domains(request)
        domains_known = bool(domains) and all(api._is_research_domain_allowlisted(domain) for domain in domains)
        permission_needed, reason = api.should_request_research_permission(
            request,
            source=source,
            domains_known=domains_known,
        )
        if permission_needed:
            provider = "firecrawl_fetch" if api._research_firecrawl_enabled else "openai_responses_web_search"
            if reason == "non_allowlisted_domain" and domains:
                logger.info("[Allowlist] permission_required domain=%s", domains[0])
            logger.info(
                "[Research] permission_required reason=%s provider=%s domains=%s domains_known=%s",
                reason,
                provider,
                domains,
                domains_known,
            )
            token = api._create_confirmation_token(
                kind="research_permission",
                tool_name="perform_research",
                request=request,
                max_retries=2,
                metadata={"approval_flow": True, "domains": domains, "domains_known": domains_known},
            )
            api._sync_confirmation_legacy_fields()
            logger.info(
                "[Research] Permission asked",
                extra={"event": "research_permission_prompt", "domains": domains, "domains_known": domains_known},
            )
            prompt_message = api._build_research_permission_prompt(domains=domains, domains_known=domains_known)
            await api.send_assistant_message(
                prompt_message,
                websocket,
                response_metadata={
                    "trigger": "confirmation_prompt",
                    "approval_flow": "true",
                    "confirmation_token": token.id,
                },
            )
            token.prompt_sent = True
            api._set_confirmation_state(ConfirmationState.AWAITING_DECISION, reason="research_prompt_sent")
            return True
    
        provider = "firecrawl_fetch" if api._research_firecrawl_enabled else "openai_responses_web_search"
        allowlisted = api._research_request_domains_allowlisted(request)
        logger.info(
            "[Research] permission_bypass reason=%s mode=%s allowlisted=%s provider=%s",
            reason,
            api._research_mode,
            allowlisted,
            provider,
        )
        await api._dispatch_research_request(request, websocket)
        return True
    
