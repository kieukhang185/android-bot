#!/bin/bash

set -e

apt_update() {
    echo "Updating package lists..."
    sudo apt update -y
}

apt_install() {
    apt_update
    echo "Installing required packages..."
    sudo apt install -y ca-certificates curl git unzip jq build-essential pkg-config xz-utils liblzma-dev \
      libjpeg-dev libpng-dev libgl1 libglib2.0-0 python3 python3-venv python3-pip
}

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    return 1
  fi
}

install_adb() {
  if need_cmd adb; then
    echo "ADB tools are already installed. Skipping installation."
    adb version
    return
  fi

  echo "Installing ADB tools..."
  sudo apt install -y android-tools-adb
  adb version
}

install_nodejs() {
  if need_cmd node; then
    echo "Nodejs is already installed. Skipping installation."
    return
  fi

  echo "Installing Nodejs..."
  curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
  sudo apt install -y nodejs
  node -v
  npm -v
}

create_python_env() {
  echo "Setting up Python virtual environment and installing dependencies..."
  python3 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip wheel
  pip install opencv-python numpy pillow scipy
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
}

setup_all() {
  apt_install
  install_adb
  install_nodejs
  create_python_env
}
