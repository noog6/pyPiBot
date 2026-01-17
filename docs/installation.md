# Installation

This project targets Raspberry Pi OS and uses system packages wherever possible.

## System Packages (apt)

```bash
sudo apt-get update
sudo apt-get install -y \
  python3 \
  python3-venv \
  python3-pip \
  python3-yaml \
  python3-numpy \
  python3-pyaudio \
  python3-smbus \
  i2c-tools
```

Notes:
- `python3-yaml` provides PyYAML for configuration loading.
- `python3-pyaudio` and `python3-numpy` are needed for audio I/O.
- `python3-smbus` + `i2c-tools` are needed for I2C devices like the PCA9685 and
  ADS1015.

## Optional Python Packages (pip)

If you prefer a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pyyaml numpy pyaudio
```

## Enable I2C on Raspberry Pi

Make sure I2C is enabled in `raspi-config`, then reboot:

```bash
sudo raspi-config
# Interface Options -> I2C -> Enable
sudo reboot
```
