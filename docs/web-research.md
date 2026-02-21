# Web Research Subsystem

This document describes the web-research capability and how to configure it.

## What It Does

The runtime can now route web-search requests through a dedicated research subsystem.

At a high level:

1. User utterances are checked for research intent (for example: "search the web", "look this up online").
2. If enabled, the runtime can require explicit user confirmation before running research.
3. The `perform_research` tool executes through the research service.
4. Results are normalized into a structured `research_packet_v1` payload.
5. A research transcript is written to the current run directory for later auditing.

## Default Behavior

In `config/default.yaml`, research is enabled with the OpenAI provider, while live Firecrawl web scraping remains disabled by default:

- `research.enabled: true`
- `research.permission_required: false`
- `research.provider: openai`
- `research.firecrawl.enabled: false`
- `research.firecrawl.allowlist_domains` includes `files.waveshare.com` by default for manufacturer docs

This means OpenAI-backed research can run in production when enabled, and optional Firecrawl scraping is opt-in.

## Configuration Reference

All keys below are under the top-level `research:` block.

### Core toggles

- `enabled` — Global research on/off switch.
- `provider` — Active provider (`openai` by default).
- `packet_schema` — Expected response schema (default `research_packet_v1`).
- `permission_required` — If `true`, Theo asks before dispatching research; if `false`, research runs without that confirmation gate.

### Realtime dedupe/debug

- `tool_call_dedupe_ttl_s` — Dedupe window for repeated tool-call dispatches.
- `spoken_response_dedupe_ttl_s` — Dedupe window for repeated spoken research summaries.
- `debug_response_create_trace` — Enables additional response.create trace logging.

### OpenAI provider

- `openai.enabled`
- `openai.model`
- `openai.timeout_s`
- `openai.max_output_chars`
- `openai.max_facts`
- `openai.max_sources`

### Firecrawl (optional web scraping)

- `firecrawl.enabled`
- `firecrawl.max_pages`
- `firecrawl.max_markdown_chars`
- `firecrawl.cache_dir`
- `firecrawl.cache_ttl_hours`
- `firecrawl.allowlist_mode` (`off | public | explicit`)
- `firecrawl.allowlist_domains`

### Budgeting and cache

- `budget.daily_limit` (default: `50`)
- `budget.state_file` (legacy migration input only)
- `cache.dir`
- `cache.ttl_hours`

### Escalation scaffold

- `escalation.enabled`
- `escalation.max_rounds`

## Secrets / Environment Variables

- `OPENAI_API_KEY` is required for the OpenAI provider path.
- `FIRECRAWL_API_KEY` is required only when `research.firecrawl.enabled: true`.
- Set/update Firecrawl credentials with:
  ```bash
  ./scripts/update-firecrawl-key.sh
  ```

## Operator Notes

- Research transcripts are written per run and include request text, packet metadata, facts, and sources.
- If research is disabled (`research.enabled: false`), the tool path returns a disabled packet and Theo responds without web lookup.
- If permission is required and denied, no research request is dispatched.

## Quick Enablement Example

```yaml
research:
  enabled: true
  permission_required: true
  openai:
    enabled: true
    model: "gpt-4.1-mini"
  firecrawl:
    enabled: true
    allowlist_mode: "public"
```

With this configuration, Theo still asks for permission first, then performs live web retrieval when allowed.


## Budget-gating behavior

- When `research.budget.daily_limit` is exhausted (remaining=0), Theo now gates research before provider dispatch.
- Theo asks once for explicit over-budget approval via confirmation (`research_budget`).
- Without approval, no provider dispatch, no content fetch attempt, and no Firecrawl escalation occurs.
- To avoid prompts entirely, increase `research.budget.daily_limit` in config.

## Research budget storage model (SQLite source of truth)

Research budget tracking is now database-backed. The SQLite database is the source of truth for both remaining budget and spend audit history. Legacy JSON budget files are only read once for migration compatibility.

### Tables

- `research_budget_state`
  - One row per logical budget key.
  - Stores the current UTC date window, remaining units, configured limit, and last update timestamp.
- `research_budget_usage`
  - Append-only audit log of successful spend events.
  - One inserted row per successful spend attempt.
  - Includes `date_utc`, timestamp, units, and optional request context fields (`request_fingerprint`, `research_id`, `source`, `provider`).
  - Override-authorized executions include `metadata_json` with `over_budget_approved=true` and `over_budget_decision_source` so operators can distinguish explicit approvals from in-budget spends.

### Auditing today's budget consumers

To inspect what consumed budget for the current UTC day:

```sql
SELECT
  spent_at_ts,
  units,
  request_fingerprint,
  research_id,
  source,
  provider,
  prompt_preview,
  json_extract(metadata_json, '$.over_budget_approved') AS over_budget_approved,
  json_extract(metadata_json, '$.over_budget_decision_source') AS over_budget_decision_source
FROM research_budget_usage
WHERE date_utc = strftime('%Y-%m-%d', 'now')
ORDER BY spent_at_ts ASC, id ASC;
```
