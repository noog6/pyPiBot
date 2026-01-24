#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"

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
  python3-websockets \
  python3-yaml

PYTHON_VERSION="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 13) else 1)'; then
  if [[ ! -d "${VENV_DIR}" ]]; then
    echo "Creating virtual environment at ${VENV_DIR}."
    python3 -m venv --system-site-packages "${VENV_DIR}"
    if [[ -f "${HOME}/.bashrc" ]] && ! grep -q "^source ${VENV_DIR}/bin/activate$" "${HOME}/.bashrc"; then
      printf '\nsource %s/bin/activate\n' "${VENV_DIR}" >> "${HOME}/.bashrc"
    fi
  else
    echo "Using existing virtual environment at ${VENV_DIR}."
  fi
  "${VENV_DIR}/bin/python" -m pip install --upgrade pip
  echo "Installing audioop-lts for Python ${PYTHON_VERSION}."
  "${VENV_DIR}/bin/python" -m pip install --upgrade audioop-lts
else
  echo "Python ${PYTHON_VERSION} includes audioop; skipping audioop-lts install."
fi

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

cat <<'EOF'

Setup completed successfully!
Example:
  python main.py --prompts="Say Hello!"
EOF
