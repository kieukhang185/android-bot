#!/usr/bin/env python3

import subprocess, json
from typing import Optional, List, Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def pick_profile(cfg: Dict[str, Any], screen_w: int, screen_h: int) -> Dict[str, Any]:
    for p in cfg.get("profiles", []):
        if p.get("screen", {}).get("w") == screen_w and p.get("screen", {}).get("h") == screen_h:
            return p
    avail = [f'{p.get("screen",{}).get("w")}x{p.get("screen",{}).get("h")}' for p in cfg.get("profiles", [])]
    raise RuntimeError(f"No profile for {screen_w}x{screen_h}. Available: {avail}")

def run_adb(args: List[str], device_id: Optional[str] = None, timeout: int = 20) -> str:
    cmd = ["adb"]
    if device_id:
        cmd += ["-s", device_id]
    cmd += args
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip() or p.stdout.strip() or f"adb failed: {cmd}")
    return p.stdout

def tap(device_id: str, x: int, y: int):
    run_adb([
        "shell", "input", "tap",
        str(x), str(y)],
        device_id=device_id
    )

def swipe(device_id: str, sx: int, sy: int, tx: int, ty: int, duration_ms: int = 320):
    run_adb([
        "shell", "input", "swipe",
        str(sx), str(sy), str(tx), str(ty), 
        str(duration_ms)], 
        device_id=device_id
    )
    # print(f"s: {sx}:{sx}, y: {tx}:{ty}")
