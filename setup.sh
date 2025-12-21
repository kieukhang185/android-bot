sudo apt update
sudo apt install -y ca-certificates curl git unzip jq build-essential pkg-config xz-utils liblzma-dev \
  libjpeg-dev libpng-dev libgl1 libglib2.0-0 python3 python3-venv python3-pip

sudo apt install -y android-tools-adb
adb version

curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt install -y nodejs

mkdir -p ~/project && cd ~/project
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install opencv-python numpy pillow
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
