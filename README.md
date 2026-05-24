# Android Puzzle Bot

A small local dashboard for controlling Android devices through ADB and running an OpenCV-based auto-click / swipe puzzle bot. The web UI lists connected devices, starts or stops automation per device, and streams runtime logs from the Python bot.

> Use this only on devices and apps you are allowed to automate.

## Repo structure

```text
android-bot-main/
├── README.md
├── setup.sh                  # Ubuntu setup helper for system tools
├── start.sh                  # Start server and web dashboard
├── scripts/                  # Helper scripts for ROI/template testing
│   ├── bench_roi.py
│   ├── bot_match_and_swipe_gray.py
│   └── get_roi_from_template.py
├── server/
│   ├── index.js              # Express API + static web server
│   ├── adb.js                # ADB device helpers
│   ├── proc_manager.js       # Starts/stops Python bot processes
│   ├── auto_puzzle.py        # Main automation loop
│   ├── image_check.py        # OpenCV matching and swipe planning
│   ├── utils.py              # Shared ADB/config helpers
│   ├── config.json           # Screen profiles, ROIs, matching config
│   ├── package.json          # Node.js dependencies/scripts
│   └── requirements.txt      # Python dependencies
└── web/
    ├── index.html            # Dashboard UI
    ├── app.js                # UI actions + SSE logs
    └── styles.css            # Dashboard styles
```

## Requirements

### Tools

- Linux/macOS/WSL or Ubuntu VM
- Python 3.10+
- Node.js LTS + npm
- Android Debug Bridge (`adb`)
- Android device or emulator with USB debugging / wireless debugging enabled (e.g. BlueStasks)

### Tech stack

- Node.js + Express
- Server-Sent Events (SSE) for live logs
- Vanilla HTML/CSS/JavaScript dashboard
- Python + OpenCV image matching
- ADB shell input for tap/swipe actions

## Setup

### 1. Install system tools

Ubuntu example:

```bash
sudo apt update
sudo apt install -y ca-certificates curl git unzip jq build-essential pkg-config \
  xz-utils liblzma-dev libjpeg-dev libpng-dev libgl1 libglib2.0-0 \
  python3 python3-venv python3-pip android-tools-adb

curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt install -y nodejs
```

### 2. Install project dependencies

```bash
cd server
npm install

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt
```

Or run the helper script:

```bash
source setup.sh
setup_all
```

### 3. Connect a device

Find your Windows Host IP inside Vagrant/WSL
```bash
ip route show default | awk '{print $3}'
```

Connect to a device

```bash
adb connect <device-ip>:<device-port>
adb devices
```

Make sure the device appears with status `device`.

### 4. Prepare templates

The bot expects template images in `server/templates/`. Template names depend on the detected screen size, for example:

```text
btn_throw_960x540.png
btn_close_960x540.png
btn_done_960x540.png
congrats_960x540.png
empty_960x540.png
waiting_960x540.png
```

Supported screen profiles are configured in `server/config.json`. Current profiles include:

- `960x540`
- `1600x900`

If your device uses another resolution, add a matching profile and template images.

## Start

Start the dashboard and API server from the `server` folder:

```bash
cd server
source .venv/bin/activate
npm start

# Or run
./start.sh
```

Open:

```text
http://127.0.0.1:3000
```

Then:

1. Confirm your device appears in the device list.
2. For wireless ADB, enter `<device-ip>:<device-port>` and click **Add** if needed.
3. Click **Start** beside a device to run automation.
4. Click **Stop** to stop automation.
5. Use the log panel to watch bot events.

## Useful commands

```bash
# List connected devices
adb devices

# Start server
cd server && source .venv/bin/activate && npm start

# Run bot manually for one device
cd server && source .venv/bin/activate && python3 auto_puzzle.py <device-id> --workdir templates
```

## Configuration

Main config file:

```text
server/config.json
```

Important settings:

- `debug`: enable/disable debug output and debug images
- `throw_roi`, `done_roi`, `close_roi`, `congrats_roi`: button detection regions
- `profiles`: screen-size-specific puzzle ROIs and matching settings
- `matching.threshold`: template matching threshold
- `matching.verify_enabled`: enable second-pass verification
- `matching.verify_threshold`: verification threshold

## Notes

- Keep the phone screen awake while the bot is running.
- Run the server locally because it can control connected devices through ADB.
- If matching is unstable, check templates, screen resolution, ROI values, and `matching.threshold` in `server/config.json`.
