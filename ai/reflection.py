"""Reflection coordinator for generating lightweight post-response analysis."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
import json
import time
from typing import Any
from urllib import request

from core.logging import logger
from storage import StorageController


REFLECTION_PROMPT_TEMPLATE = """You are an assistant reflecting on a recent interaction.
Return ONLY valid JSON with keys: summary, mistakes, improvements, follow_up.
Rules:
- summary: 1-2 short sentences.
- mistakes: short list of issues (empty list if none).
- improvements: short list of concrete improvements.
- follow_up: short suggestion for next step or follow-up question.

Context:
User input: {user_input}
Assistant reply: {assistant_reply}
Tool calls: {tool_calls}
Response metadata: {response_metadata}
"""


@dataclass(frozen=True)
class ReflectionContext:
    """Captured context for a reflection."""

    last_user_input: str | None
    assistant_reply: str | None
    tool_calls: list[dict[str, Any]]
    response_metadata: dict[str, Any]


class ReflectionCoordinator:
    """Schedule lightweight reflections without blocking the realtime loop."""

    def __init__(
        self,
        *,
        api_key: str,
        enabled: bool,
        min_interval_s: float,
        storage: StorageController | None = None,
        model: str = "gpt-4o-mini",
    ) -> None:
        self._api_key = api_key
        self._enabled = enabled
        self._min_interval_s = max(0.0, min_interval_s)
        self._storage = storage
        self._model = model
        self._last_reflection_ts = 0.0
        self._pending_task: asyncio.Task[None] | None = None

    def enqueue_reflection(self, context: ReflectionContext) -> None:
        """Enqueue a reflection generation task if allowed."""

        if not self._enabled:
            logger.debug("Reflection enqueue skipped: disabled.")
            return

        now = time.monotonic()
        if self._min_interval_s > 0.0 and (now - self._last_reflection_ts) < self._min_interval_s:
            remaining = self._min_interval_s - (now - self._last_reflection_ts)
            logger.debug(
                "Reflection enqueue skipped: min interval not elapsed (%.2fs remaining).",
                remaining,
            )
            return

        if self._pending_task is not None and not self._pending_task.done():
            logger.debug("Reflection enqueue skipped: previous task still running.")
            return

        trigger = context.response_metadata.get("trigger", "unknown")
        logger.debug("Reflection task enqueued (trigger=%s).", trigger)
        self._pending_task = asyncio.create_task(self._run_reflection(context))

    async def _run_reflection(self, context: ReflectionContext) -> None:
        start_time = time.monotonic()
        try:
            trigger = context.response_metadata.get("trigger", "unknown")
            logger.debug("Reflection task started (trigger=%s).", trigger)
            reflection = await self.generate_reflection(context)
            if reflection is None:
                logger.debug("Reflection task ended with no payload.")
                return
            payload = {
                "timestamp": datetime.utcnow().isoformat(),
                "context": {
                    "last_user_input": context.last_user_input,
                    "assistant_reply": context.assistant_reply,
                    "tool_calls": context.tool_calls,
                    "response_metadata": context.response_metadata,
                },
                "reflection": reflection,
            }
            if self._storage is not None:
                self._storage.add_data("reflection", json.dumps(payload))
                logger.debug("Reflection payload stored.")
        except Exception as exc:  # noqa: BLE001 - guard background task
            logger.warning("Reflection task failed: %s", exc)
        finally:
            duration = time.monotonic() - start_time
            self._last_reflection_ts = time.monotonic()
            logger.debug("Reflection task finished in %.2fs.", duration)

    async def generate_reflection(
        self, context: ReflectionContext
    ) -> dict[str, Any] | None:
        """Generate a reflection JSON payload using a lightweight prompt."""

        if not self._api_key:
            logger.debug("Skipping reflection: missing API key.")
            return None

        prompt = self._build_prompt(context)
        try:
            logger.debug("Requesting reflection from model=%s.", self._model)
            raw_response = await asyncio.to_thread(self._call_openai, prompt)
        except Exception as exc:  # noqa: BLE001 - surface errors in response payload
            logger.warning("Reflection request failed: %s", exc)
            return {
                "summary": "Reflection unavailable due to an error.",
                "mistakes": [str(exc)],
                "improvements": [],
                "follow_up": "Retry reflection when connectivity is restored.",
            }

        try:
            reflection = json.loads(raw_response)
            if isinstance(reflection, dict):
                logger.debug("Reflection JSON parsed successfully.")
                return reflection
        except json.JSONDecodeError:
            logger.debug("Reflection response was not valid JSON.")

        return {
            "summary": "Reflection response was not valid JSON.",
            "mistakes": [],
            "improvements": [],
            "follow_up": "Review the raw response and retry.",
            "raw": raw_response,
        }

    def _build_prompt(self, context: ReflectionContext) -> str:
        user_input = self._clip_text(context.last_user_input)
        assistant_reply = self._clip_text(context.assistant_reply)
        tool_calls = self._clip_text(json.dumps(context.tool_calls, ensure_ascii=False))
        response_metadata = self._clip_text(
            json.dumps(context.response_metadata, ensure_ascii=False)
        )
        return REFLECTION_PROMPT_TEMPLATE.format(
            user_input=user_input,
            assistant_reply=assistant_reply,
            tool_calls=tool_calls,
            response_metadata=response_metadata,
        )

    def _clip_text(self, text: str | None, limit: int = 1200) -> str:
        if not text:
            return "(none)"
        return text if len(text) <= limit else f"{text[:limit]}…"

    def _call_openai(self, prompt: str) -> str:
        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a concise reflection generator.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 220,
        }
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=30) as response:
            body = response.read().decode("utf-8")
        response_payload = json.loads(body)
        return response_payload["choices"][0]["message"]["content"].strip()
