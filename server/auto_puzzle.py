#!/usr/bin/env python3
import os
import sys
import json
import time
import random
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List, Union

import cv2
import numpy as np

from utils import run_adb, tap, load_config, pick_profile
from image_check import android_bot, swipe_pairs


# -------------------------
# Helpers
# -------------------------
def emit(obj: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def random_sleep(a: float, b: float) -> None:
    time.sleep(random.uniform(a, b))


def jitter(val: float, err: int) -> int:
    return int(round(val + random.randint(-err, err)))


# -------------------------
# Screenshot (fast path: in-memory)
# -------------------------
def screencap_bytes(device_id: str, timeout: int = 30) -> bytes:
    """
    Capture a PNG screenshot from device via adb exec-out and return raw bytes.
    Faster than writing to disk + re-reading.
    """
    cmd = ["adb", "-s", device_id, "exec-out", "screencap", "-p"]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode("utf-8", "ignore").strip() or "screencap failed")
    return p.stdout


def decode_png(png_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Failed to decode PNG bytes")
    return img


def screencap_to_file(device_id: str, out_path: str) -> str:
    """
    Compatibility mode: write screenshot to a file on disk (original behavior).
    """
    time.sleep(0.2)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cmd = ["adb", "-s", device_id, "exec-out", "screencap", "-p"]
    with open(out_path, "wb") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, timeout=30)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode("utf-8", "ignore").strip() or "screencap failed")
    time.sleep(0.2)
    return out_path


# -------------------------
# Template Matching
# -------------------------
@dataclass(frozen=True)
class Match:
    score: float
    top_left: Tuple[int, int]
    bottom_right: Tuple[int, int]

    @property
    def center(self) -> Tuple[int, int]:
        cx = (self.top_left[0] + self.bottom_right[0]) // 2
        cy = (self.top_left[1] + self.bottom_right[1]) // 2
        return cx, cy


class TemplateCache:
    def __init__(self) -> None:
        self._cache: Dict[str, np.ndarray] = {}

    def load(self, path: str) -> np.ndarray:
        if path in self._cache:
            return self._cache[path]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Cannot read template: {path}")
        self._cache[path] = img
        return img


def clamp_roi(roi: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = roi
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(1, min(w, x2))
    y2 = max(1, min(h, y2))
    return x1, y1, x2, y2


def match_template(
    haystack_bgr: np.ndarray,
    needle_bgr: np.ndarray,
    roi: Optional[Tuple[int, int, int, int]] = None,
    method: int = cv2.TM_CCOEFF_NORMED,
) -> Match:
    if haystack_bgr is None or needle_bgr is None:
        raise ValueError("haystack_bgr/needle_bgr is None")

    hs = haystack_bgr
    offx, offy = 0, 0

    if roi is not None:
        hH, wH = haystack_bgr.shape[:2]
        x1, y1, x2, y2 = clamp_roi(roi, wH, hH)
        hs = haystack_bgr[y1:y2, x1:x2]
        offx, offy = x1, y1

    th, tw = needle_bgr.shape[:2]
    hh, hw = hs.shape[:2]
    if th > hh or tw > hw:
        return Match(0.0, (offx, offy), (offx, offy))

    res = cv2.matchTemplate(hs, needle_bgr, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    if method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
        score = float(1.0 - min_val)
        loc = min_loc
    else:
        score = float(max_val)
        loc = max_loc

    top_left = (loc[0] + offx, loc[1] + offy)
    bottom_right = (top_left[0] + tw, top_left[1] + th)
    return Match(score=score, top_left=top_left, bottom_right=bottom_right)


def find_template(
    screen_bgr: np.ndarray,
    tpl_bgr: np.ndarray,
    threshold: float = 0.85,
    roi: Optional[Tuple[int, int, int, int]] = None,
) -> Optional[Match]:
    m = match_template(screen_bgr, tpl_bgr, roi=roi)
    return m if m.score >= threshold else None


def adb_tap(device_id: str, x: int, y: int) -> None:
    run_adb(["shell", "input", "tap", str(int(x)), str(int(y))], device_id=device_id)


# -------------------------
# Main bot logic
# -------------------------
@dataclass
class Templates:
    btn_throw: str
    btn_close: str
    btn_done: str
    congrats: str
    empty: str
    waiting: str


class AutoPuzzleBot:
    def __init__(self, device_id: str, workdir: str, threshold: float = 0.85) -> None:
        self.device_id = device_id
        self.workdir = workdir
        self.threshold = threshold
        self.cache = TemplateCache()

        os.makedirs(workdir, exist_ok=True)
        self.tmp_dir = os.path.join(workdir, "tmp")
        os.makedirs(self.tmp_dir, exist_ok=True)

        self.screen_path = os.path.join(self.tmp_dir, f"screen_{device_id.replace(':','_')}.png")

        self.fish_count = 1
        self.throw_count = 1
        self.loop_idx = 0

        # will be set after first screenshot
        self.sw = 0
        self.sh = 0
        self.tpl: Optional[Templates] = None

        self.throw_roi = tuple[0, 0, 0, 0]
        self.done_roi = tuple[0, 0, 0, 0]
        self.close_roi = tuple[0, 0, 0, 0]
        self.congrats_roi = tuple[0, 0, 0, 0]

    def capture_screen(self, save_to_disk: bool = True) -> np.ndarray:
        if save_to_disk:
            screencap_to_file(self.device_id, self.screen_path)
            img = cv2.imread(self.screen_path, cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError("Failed to read captured screen image from disk")
            return img
        else:
            png = screencap_bytes(self.device_id)
            img = decode_png(png)
            return img

    def parse_btn_config(self, cfg: Dict[str, Any], key: str):
            key_value = cfg[key]
            return (
                int(round(key_value["x"] * self.sw)),
                int(round(key_value["y"] * self.sh)),
                int(round((key_value["x"] + key_value["w"]) * self.sw)),
                int(round((key_value["y"] + key_value["h"]) * self.sh))
            )

    def init_templates(self, screen_bgr: np.ndarray) -> None:
        self.sh, self.sw = screen_bgr.shape[:2]

        cfg = load_config("config.json")
        self.throw_roi = self.parse_btn_config(cfg, "throw_roi")
        self.done_roi = self.parse_btn_config(cfg, "done_roi")
        self.close_roi = self.parse_btn_config(cfg, "close_roi")
        self.congrats_roi = self.parse_btn_config(cfg, "congrats_roi")

        self.tpl = Templates(
            btn_throw=os.path.join(self.workdir, f"btn_throw_{self.sw}x{self.sh}.png"),
            btn_close=os.path.join(self.workdir, f"btn_close_{self.sw}x{self.sh}.png"),
            btn_done=os.path.join(self.workdir, f"btn_done_{self.sw}x{self.sh}.png"),
            congrats=os.path.join(self.workdir, f"congrats_{self.sw}x{self.sh}.png"),
            empty=os.path.join(self.workdir, f"empty_{self.sw}x{self.sh}.png"),
            waiting=os.path.join(self.workdir, f"waiting_{self.sw}x{self.sh}.png"),
        )

        # Preload templates to fail fast if missing
        for p in self.tpl.__dict__.values():
            if not os.path.exists(p):
                emit({"type": "error", "msg": f"missing_template: {p}"})
                raise SystemExit(2)
            self.cache.load(p)

    def click_if_present(
            self, screen: np.ndarray,
            tpl_path: str,
            label: str,
            roi: Optional[Tuple[int, int, int, int]] = None,
        ) -> bool:
        if not os.path.isfile(tpl_path):
            emit({"type": "error", "msg": f"missing_template: {tpl_path}"})
        tpl = self.cache.load(tpl_path)
        match = find_template(screen, tpl, threshold=self.threshold, roi=roi)
        if not match:
            return False
        cx, cy = match.center
        adb_tap(self.device_id, cx, cy)
        emit({"type": "info", "msg": f"clicked_{label}", "score": round(match.score, 4), "center": [cx, cy]})
        return True

    def is_present(
            self, screen: np.ndarray,
            tpl_path: str,
            roi: Optional[Tuple[int, int, int, int]] = None
        ) -> bool:
        if not os.path.isfile(tpl_path):
            emit({"type": "error", "msg": f"missing_template: {tpl_path}"})
        tpl = self.cache.load(tpl_path)
        return find_template(screen, tpl, threshold=self.threshold, roi=roi) is not None

    def run_once(self) -> None:
        assert self.tpl is not None, "Templates not initialized"

        # capture
        screen = self.capture_screen(save_to_disk=True)
        emit({"type": "info", "msg": "screencap_ok", "i": self.loop_idx, "screen": self.screen_path})
        random_sleep(0.2, 0.3)

        # Must be at fishing position: either throw exists OR done exists
        throw_present = self.is_present(screen, self.tpl.btn_throw, self.throw_roi)
        done_present = self.is_present(screen, self.tpl.btn_done, self.done_roi)
        waiting_present = self.is_present(screen, self.tpl.waiting)

        if throw_present:
            # click throw
            _ = self.click_if_present(screen, self.tpl.btn_throw, "throw", self.throw_roi)
            # refresh quickly
            screen = self.capture_screen(save_to_disk=True)

            # out of bait check
            if self.is_present(screen, self.tpl.empty):
                emit({"type": "error", "msg": "out_of_bait", "i": self.loop_idx})
                raise SystemExit(0)

            emit({"type": "info", "msg": f"started_throw: {self.throw_count}", "i": self.loop_idx})
            self.throw_count += 1

        elif done_present:
            self.click_if_present(screen, self.tpl.btn_done, "done", self.close_roi)
            random_sleep(0.3, 0.5)
            return
        elif waiting_present:
            random_sleep(0.3, 0.5)
            return
        else:
            emit({"type": "error", "i": self.loop_idx, "msg": "Not in fishing position!"})
            raise SystemExit(1)

        # wait for puzzle
        random_sleep(8, 10)

        run_vision = False
        max_try = 10
        waite_second = 1.0

        for try_idx in range(max_try):
            screen = self.capture_screen(save_to_disk=True)
            random_sleep(0.2, 0.4)

            close_absent = not self.is_present(screen, self.tpl.btn_close, self.close_roi)
            if close_absent:
                # if close absent, maybe we can finish directly
                if self.is_present(screen, self.tpl.btn_done, self.done_roi):
                    self.click_if_present(screen, self.tpl.btn_done, "done", self.done_roi)
                    run_vision = False
                    random_sleep(0.2, 0.4)
                    break
                else:
                    run_vision = True
                    break

            time.sleep(waite_second)

            if try_idx == max_try - 1:
                emit({"type": "error", "i": self.loop_idx, "msg": "Error when fishing!"})
                # attempt to close if stuck
                self.click_if_present(screen, self.tpl.btn_close, "close", self.close_roi)
                self.click_if_present(screen, self.tpl.btn_done, "done", self.done_roi)

        if not run_vision:
            return

        # vision solve
        pairs, _vision_out = android_bot(self.device_id, self.screen_path)
        swipe_pairs(self.device_id, pairs, duration_ms=320, jitter=2)

        random_sleep(0.8, 1.0)
        screen = self.capture_screen(save_to_disk=True)
        random_sleep(0.4, 0.5)

        # post-vision checks
        if self.is_present(screen, self.tpl.btn_done, self.done_roi):
            self.click_if_present(screen, self.tpl.btn_done, "done", self.done_roi)
            emit({"type": "info", "i": self.loop_idx, "msg": "fishing_failed"})
            time.sleep(0.4)
            return

        if self.is_present(screen, self.tpl.congrats, self.congrats_roi):
            emit({"type": "info", "i": self.loop_idx, "msg": f"fishing_success: {self.fish_count}"})
            # tap to dismiss congrats (keep your logic)
            w_click = self.sw - (self.sw / 8)
            tap(self.device_id, jitter(w_click, 3), jitter(self.sh / 2, 20))
            random_sleep(0.4, 0.5)

            # wait for done, then click
            for lag_idx in range(3):
                screen = self.capture_screen(save_to_disk=True)
                if not self.is_present(screen, self.tpl.btn_done, self.done_roi):
                    emit({"type": "warn", "i": self.loop_idx, "msg": "lagging_wait"})
                    random_sleep(0.4, 0.5)
                    continue
                self.click_if_present(screen, self.tpl.btn_done, "done", self.done_roi)
                emit({"type": "info", "i": self.loop_idx, "msg": "done_game_loop"})
                time.sleep(0.4)
                self.fish_count += 1
                break
            return

        # fallback
        for fallback_idx in range(3):
            emit({"type": "warn", "i": self.loop_idx, "msg": "swipe_failed_wait_5s"})
            screen = self.capture_screen(save_to_disk=True)
            if not self.is_present(screen, self.tpl.btn_done, self.done_roi):
                time.sleep(5)
                continue
            self.click_if_present(screen, self.tpl.btn_done, "done", self.done_roi)
            random_sleep(0.2, 0.4)
            break

    def run_forever(self) -> None:
        # init on first screen
        first = self.capture_screen(save_to_disk=True)
        self.init_templates(first)

        while True:
            self.run_once()
            self.loop_idx += 1


def parse_args(argv: List[str]) -> Dict[str, Any]:
    if len(argv) < 2:
        raise SystemExit("Usage: auto_puzzle.py <device-id> [--workdir PATH] [--threshold FLOAT]")
    device_id = argv[1]
    workdir = "templates"
    threshold = 0.85

    if "--workdir" in argv:
        workdir = argv[argv.index("--workdir") + 1]
    if "--threshold" in argv:
        threshold = float(argv[argv.index("--threshold") + 1])

    return {"device_id": device_id, "workdir": workdir, "threshold": threshold}


def main() -> None:
    args = parse_args(sys.argv)
    bot = AutoPuzzleBot(
        device_id=args["device_id"],
        workdir=args["workdir"],
        threshold=args["threshold"],
    )
    bot.run_forever()


if __name__ == "__main__":
    main()
