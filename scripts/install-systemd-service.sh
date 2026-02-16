#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: install-systemd-service.sh [--enable] [--restart] [--help]

Install/update pyPiBot systemd unit from this repository.

Options:
  --enable   Enable pyPiBot.service at boot.
  --restart  Restart pyPiBot.service after installing/updating the unit.
  --help     Show this help message.
USAGE
}

ENABLE=false
RESTART=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --enable)
      ENABLE=true
      ;;
    --restart)
      RESTART=true
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
  shift
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SERVICE_TEMPLATE="${REPO_ROOT}/systemd/pyPiBot.service"
SERVICE_TARGET="/lib/systemd/system/pyPiBot.service"

if [[ ! -d "${REPO_ROOT}/.git" ]]; then
  echo "Error: ${REPO_ROOT} does not look like the pyPiBot repository root (.git missing)." >&2
  exit 1
fi

if [[ ! -f "${SERVICE_TEMPLATE}" ]]; then
  echo "Error: service template not found at ${SERVICE_TEMPLATE}." >&2
  exit 1
fi

echo "Installing systemd unit from ${SERVICE_TEMPLATE}"
sudo cp "${SERVICE_TEMPLATE}" "${SERVICE_TARGET}"

echo "Reloading systemd daemon"
sudo systemctl daemon-reload

if [[ "${ENABLE}" == true ]]; then
  echo "Enabling pyPiBot.service at boot"
  sudo systemctl enable pyPiBot.service
fi

if [[ "${RESTART}" == true ]]; then
  echo "Restarting pyPiBot.service"
  sudo systemctl restart pyPiBot.service
fi

echo "Done. Check service status with:"
echo "  sudo systemctl status pyPiBot.service"
