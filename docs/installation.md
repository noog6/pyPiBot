# Installation

This project targets Raspberry Pi OS (Trixie 64-bit) and uses system packages
wherever possible. For hardware assembly notes, see
`docs/hardware-setup.md`.

## 1) Prepare Raspberry Pi OS

- Flash Raspberry Pi OS (Trixie 64-bit) to your microSD card.
- Use the imager advanced options to set hostname, enable SSH, and create a
  user account.
- Boot the Raspberry Pi and connect via SSH.

## 2) Install Git and Clone the Repo

```bash
sudo apt-get update
sudo apt-get install -y git

mkdir -p ~/workarea
cd ~/workarea
# Replace the URL below with your pyPiBot repository.
git clone https://github.com/<your-org>/pyPiBot.git
cd pyPiBot
```

## 3) Run the Environment Setup Script

The script installs system dependencies for audio, I2C, camera, and Python.
It also installs the project's ALSA configuration to `~/.asoundrc`.

```bash
./scripts/setup-environment.sh
```

## 4) Enable I2C (for PCA9685, Sense HAT B, etc.)

```bash
sudo raspi-config
# Interface Options -> I2C -> Enable
sudo reboot
```
