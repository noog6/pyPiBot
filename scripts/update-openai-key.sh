#!/usr/bin/env bash
set -euo pipefail

OPENAI_API_KEY_INPUT="${1:-}"

if [[ -z "${OPENAI_API_KEY_INPUT}" ]]; then
  echo "An OpenAI API key is required."
  while [[ -z "${OPENAI_API_KEY_INPUT}" ]]; do
    read -r -s -p "Enter your OpenAI API key: " OPENAI_API_KEY_INPUT
    echo ""
  done
fi

export OPENAI_API_KEY="${OPENAI_API_KEY_INPUT}"

if [[ -f "${HOME}/.bashrc" ]]; then
  sed -i '/^export OPENAI_API_KEY=/d' "${HOME}/.bashrc"
fi
printf '\nexport OPENAI_API_KEY=%q\n' "${OPENAI_API_KEY}" >> "${HOME}/.bashrc"

sudo sed -i '/^OPENAI_API_KEY=/d' /etc/environment
printf 'OPENAI_API_KEY=%s\n' "${OPENAI_API_KEY}" | sudo tee -a /etc/environment >/dev/null

echo "OPENAI_API_KEY updated in /etc/environment, ~/.bashrc, and current session."
