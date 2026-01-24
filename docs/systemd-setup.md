# Systemd Setup

This guide describes how to run the Theo runtime as a systemd service.

## Prerequisites

- Repository checked out on the target Raspberry Pi.
- Python virtual environment created under `.venv`.
- Log directory created at `log/`.
- `OPENAI_API_KEY` set in `/etc/environment` (or another file referenced by `EnvironmentFile` in the unit).

## Install the Service

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
4. Start the service:
   ```bash
   sudo service pyPiBot start
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

## Customization

Update the following fields in `systemd/pyPiBot.service` as needed:

- `ExecStart` path (virtualenv + entry point).
- `WorkingDirectory` path.
- Log file paths in `StandardOutput` and `StandardError`.
- `User` to the appropriate runtime user.
- `EnvironmentFile` if you store `OPENAI_API_KEY` somewhere else.
