#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

sudo apt-get update
sudo apt-get upgrade -y

sudo apt-get install -y \
  alsa-utils \
  ffmpeg \
  git \
  i2c-tools \
  joystick \
  libasound-dev \
  libsndfile1 \
  portaudio19-dev \
  pulseaudio \
  python3 \
  python3-numpy \
  python3-picamera2 \
  python3-pip \
  python3-pyaudio \
  python3-smbus \
  python3-venv \
  python3-yaml

cp "${SCRIPT_DIR}/../config/asoundrc" "${HOME}/.asoundrc"

OPENAI_API_KEY="${OPENAI_API_KEY:-}"
if [[ -z "${OPENAI_API_KEY}" ]]; then
  echo "An OpenAI API key is required for this project."
  while [[ -z "${OPENAI_API_KEY}" ]]; do
    read -r -s -p "Enter your OpenAI API key: " OPENAI_API_KEY
    echo ""
  done
fi

export OPENAI_API_KEY

if [[ -f "${HOME}/.bashrc" ]]; then
  sed -i '/^export OPENAI_API_KEY=/d' "${HOME}/.bashrc"
fi
printf '\nexport OPENAI_API_KEY=%q\n' "${OPENAI_API_KEY}" >> "${HOME}/.bashrc"

if [[ -w "/etc/environment" ]]; then
  sudo sed -i '/^OPENAI_API_KEY=/d' /etc/environment
  echo "OPENAI_API_KEY=${OPENAI_API_KEY}" | sudo tee -a /etc/environment >/dev/null
else
  echo "OPENAI_API_KEY=${OPENAI_API_KEY}" | sudo tee -a /etc/environment >/dev/null
fi
