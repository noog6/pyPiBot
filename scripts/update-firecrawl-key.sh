#!/usr/bin/env bash
set -euo pipefail

FIRECRAWL_API_KEY_INPUT="${1:-}"

if [[ -z "${FIRECRAWL_API_KEY_INPUT}" ]]; then
  echo "A Firecrawl API key is required."
  while [[ -z "${FIRECRAWL_API_KEY_INPUT}" ]]; do
    read -r -s -p "Enter your Firecrawl API key: " FIRECRAWL_API_KEY_INPUT
    echo ""
  done
fi

export FIRECRAWL_API_KEY="${FIRECRAWL_API_KEY_INPUT}"

if [[ -f "${HOME}/.bashrc" ]]; then
  sed -i '/^export FIRECRAWL_API_KEY=/d' "${HOME}/.bashrc"
fi
printf '\nexport FIRECRAWL_API_KEY=%q\n' "${FIRECRAWL_API_KEY}" >> "${HOME}/.bashrc"

sudo sed -i '/^FIRECRAWL_API_KEY=/d' /etc/environment
printf 'FIRECRAWL_API_KEY=%s\n' "${FIRECRAWL_API_KEY}" | sudo tee -a /etc/environment >/dev/null

echo "FIRECRAWL_API_KEY updated in /etc/environment, ~/.bashrc, and current session."
