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
