# Systemd Setup

This guide describes how to run the Theo runtime as a systemd service.

## Prerequisites

- Repository checked out on the target Raspberry Pi.
- Python virtual environment created under `.venv`.
- Log directory created at `log/`.
- `OPENAI_API_KEY` set in `/etc/environment` (or another file referenced by `EnvironmentFile` in the unit).
- `FIRECRAWL_API_KEY` set as well if you enable `research.firecrawl.enabled: true` (use `./scripts/update-firecrawl-key.sh`).

## Install the Service (preferred: helper script)

Use the helper script for reproducible install/update flow:

```bash
./scripts/install-systemd-service.sh --enable --restart
```

Useful variants:

- Install/update unit without changing enable/start state:
  ```bash
  ./scripts/install-systemd-service.sh
  ```
- Install/update and restart now, but do not enable at boot:
  ```bash
  ./scripts/install-systemd-service.sh --restart
  ```

The script validates repo/service paths, copies `systemd/pyPiBot.service`, reloads systemd, and prints the status command.

### Manual fallback

If you prefer manual commands:

1. Copy the service unit file:
   ```bash
   sudo cp systemd/pyPiBot.service /lib/systemd/system/
   ```
2. Reload systemd:
   ```bash
   sudo systemctl daemon-reload
   ```
3. Enable the service at boot:
   ```bash
   sudo systemctl enable pyPiBot.service
   ```
4. Start or restart the service:
   ```bash
   sudo service pyPiBot start
   # or, for updates:
   sudo service pyPiBot restart
   ```

## Service Management

- Start:
  ```bash
  sudo service pyPiBot start
  ```
- Stop:
  ```bash
  sudo service pyPiBot stop
  ```
- Restart:
  ```bash
  sudo service pyPiBot restart
  ```

If you update `OPENAI_API_KEY` or `FIRECRAWL_API_KEY` in environment files, restart the service so the new values are loaded:

```bash
sudo systemctl daemon-reload
sudo service pyPiBot restart
```

`daemon-reload` is only required if you changed the unit file or `EnvironmentFile` wiring; for key-value updates in `/etc/environment`, a service restart is the required step.

## Git sync logging

Service startup uses a best-effort git sync wrapper (`scripts/systemd-git-sync.sh`) before launching the app.

- If `fetch/switch/pull` succeed, the service starts normally.
- If one or more sync steps fail, failures are logged, `overall_status=warning` is recorded, and the service still starts (non-fatal pre-start behavior).

Useful troubleshooting commands:

```bash
tail -f /home/pi/workarea/pyPiBot/log/git-sync.log
journalctl -u pyPiBot.service -f
journalctl -t pyPiBot-git-sync --since today
```

## Application run logs

In addition to systemd-managed stdout/stderr logs, the application creates per-run log files under:

- `log/<run_id>/run_<run_id>.log`

Run IDs are incrementing integers persisted in `var/current_run` to avoid ambiguity across restarts.

Example:

- `log/314/run_314.log`

This allows operators to inspect an individual run in isolation, while still keeping aggregated service-level logs.

## Log rotation

A logrotate template is provided at `ops/logrotate/pypibot` to rotate:

- `/home/pi/workarea/pyPiBot/log/pyPiBot.log`
- `/home/pi/workarea/pyPiBot/log/pyPiBot-error.log`
- `/home/pi/workarea/pyPiBot/log/git-sync.log`

Install it on the target host with:

```bash
sudo cp ops/logrotate/pypibot /etc/logrotate.d/pypibot
sudo logrotate -f /etc/logrotate.d/pypibot
```

## Log types: systemd vs app-managed

- **Systemd stdout/stderr logs** (`pyPiBot.log`, `pyPiBot-error.log`) capture process-level output configured by `StandardOutput`/`StandardError` in the unit file.
- **App-managed per-run logs** (`log/<run_id>/run_<run_id>.log`, for example `log/314/run_314.log`) are created by the application to preserve run-scoped execution details.

Use systemd logs for service health and lifecycle troubleshooting; use per-run logs for investigating behavior within a specific run.

Per-run logs are intentionally managed directly by Theo and are not included in `ops/logrotate/pypibot`.

## Operator verification checklist

After deploying the updated logrotate config:

1. Verify per-run logs are discoverable:
   ```bash
   find log -name 'run_*.log'
   ```
   Expect one or more matched files when runs have executed.

2. Dry-run logrotate and verify systemd-level logs are included:
   ```bash
   sudo logrotate -d /etc/logrotate.d/pypibot
   ```
   Expect debug output showing consideration/rotation checks for `pyPiBot.log`, `pyPiBot-error.log`, and `git-sync.log` only (not `run_*.log`).

## Operator checklist: Raspberry Pi Zero 2 W

Use this checklist when running on Pi Zero 2 W class hardware where CPU headroom is limited.

1. **Set conservative CPU budgets** in `config/default.yaml` (or your deployed override):
   - `ops.budgets.ai_calls_per_minute`
   - `ops.budgets.sensor_reads_per_minute`
   - `ops.budgets.logs_per_minute`
   - `ops.budgets.micro_presence_per_hour`

   Start with lower values than desktop/dev defaults, then increase gradually while observing stability.

2. **Tune semantic retrieval batch pressure** to reduce per-turn work:
   - If semantic retrieval is still enabled, lower `memory_semantic.max_candidates_for_semantic` so fewer candidates are reranked each turn.
   - If semantic retrieval is disabled for the rollout profile, set:

   ```yaml
   memory_semantic:
     enabled: false
     rerank_enabled: false
   ```

3. **Keep logs informative but lightweight**:
   - Use `logging_level: "INFO"` for normal operation.
   - Temporarily switch to `DEBUG` only during active troubleshooting windows.
   - If log volume impacts responsiveness, reduce `ops.budgets.logs_per_minute` and re-check service behavior with `journalctl -u pyPiBot.service -f`.

## Customization

Update the following fields in `systemd/pyPiBot.service` as needed:

- `ExecStart` path (virtualenv + entry point).
- `WorkingDirectory` path.
- Log file paths in `StandardOutput` and `StandardError`.
- `User` to the appropriate runtime user.
- `EnvironmentFile` if you store API keys (`OPENAI_API_KEY`, optional `FIRECRAWL_API_KEY`) somewhere else.
- Optional semantic startup override: `PYPIBOT_SEMANTIC_CANARY_BYPASS=1` (recommended only for explicit offline/testing scenarios).

## Auto-sync security considerations

If you enable `ExecStartPre` git sync commands (for example `fetch/switch/pull`) so service startup auto-updates code, be aware of the supply-chain and operational risks:

- **Unreviewed code deployment:** a restart can deploy new commits immediately, even if they were not manually validated on the device.
- **Compromised remote risk:** if `origin` is hijacked or credentials are leaked, malicious code can be pulled at service start.
- **Restart-loop amplification:** repeated crashes/restarts can repeatedly pull and redeploy changing upstream state.

Recommended guardrails:

- Use `git pull --ff-only` only (avoid implicit merge commits during startup automation).
- Pin `origin` to SSH and use a restricted deploy key with least privilege.
- Protect `main` with required reviews and passing CI before merge.
- Optionally require commit-signature verification in your release/deployment workflow before service restarts are allowed.

### Rollback procedure

If a bad update is pulled, roll back to a known-good commit:

```bash
git -C /home/pi/workarea/pyPiBot switch main
git -C /home/pi/workarea/pyPiBot reset --hard <known-good-sha>
sudo service pyPiBot restart
```

Replace `<known-good-sha>` with the commit hash you want to restore.
