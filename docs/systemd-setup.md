# Systemd Setup

This guide describes how to run the Theo runtime as a systemd service.

## Prerequisites

- Repository checked out on the target Raspberry Pi.
- Python virtual environment created under `.venv`.
- Log directory created at `log/`.
- `OPENAI_API_KEY` set in `/etc/environment` (or another file referenced by `EnvironmentFile` in the unit).
- `FIRECRAWL_API_KEY` set as well if you enable `research.firecrawl.enabled: true`.

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

## Customization

Update the following fields in `systemd/pyPiBot.service` as needed:

- `ExecStart` path (virtualenv + entry point).
- `WorkingDirectory` path.
- Log file paths in `StandardOutput` and `StandardError`.
- `User` to the appropriate runtime user.
- `EnvironmentFile` if you store API keys (`OPENAI_API_KEY`, optional `FIRECRAWL_API_KEY`) somewhere else.

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
