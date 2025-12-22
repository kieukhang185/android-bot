#!/usr/bin/env python3

import os
import sys
import json
import time
import subprocess
from typing import Optional, List, Tuple, Dict, Any
import shutil
import cv2
import random
import numpy as np

from utils import run_adb, tap
from image_check import android_bot, swipe_pairs
import warnings


warnings.filterwarnings("ignore", message="xFormers is not available")


def random_sleep(a: float, b: float):
    """Sleep for a random amount of time between a and b seconds."""
    time.sleep(random.uniform(a, b))


def screencap(device_id: str, out_path: str) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cmd = ["adb", "-s", device_id, "exec-out", "screencap", "-p"]
    with open(out_path, "wb") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, timeout=30)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode("utf-8", "ignore").strip() or "screencap failed")
    return out_path

# -------------------------
# Template matching utilities (optional)
# -------------------------
def read_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return img


def match_template(
    haystack_bgr: np.ndarray,
    needle_bgr: np.ndarray,
    roi: Optional[Tuple[int, int, int, int]] = None,  # (x1,y1,x2,y2)
    method: int = cv2.TM_CCOEFF_NORMED,
) -> Tuple[float, Tuple[int, int], Tuple[int, int]]:
    """
    Returns:
      score: float
      top_left: (x, y) in full-screen coords
      bottom_right: (x, y) in full-screen coords
    """
    if haystack_bgr is None or needle_bgr is None:
        raise ValueError("haystack_bgr/needle_bgr is None")

    hs = haystack_bgr
    offx, offy = 0, 0

    if roi is not None:
        x1, y1, x2, y2 = roi
        hH, wH = haystack_bgr.shape[:2]
        x1 = max(0, min(wH - 1, x1))
        y1 = max(0, min(hH - 1, y1))
        x2 = max(1, min(wH, x2))
        y2 = max(1, min(hH, y2))

        hs = haystack_bgr[y1:y2, x1:x2]
        offx, offy = x1, y1

    th, tw = needle_bgr.shape[:2]
    hh, hw = hs.shape[:2]

    # If template larger than search area => cannot match
    if th > hh or tw > hw:
        return 0.0, (offx, offy), (offx, offy)

    res = cv2.matchTemplate(hs, needle_bgr, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # For SQDIFF methods: lower is better, convert to "score where higher is better"
    if method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
        score = float(1.0 - min_val)
        loc = min_loc
    else:
        score = float(max_val)
        loc = max_loc

    top_left = (loc[0] + offx, loc[1] + offy)
    bottom_right = (top_left[0] + tw, top_left[1] + th)
    return score, top_left, bottom_right


def click_template(device, screen_path, template_path, threshold=0.85):
    if not os.path.exists(screen_path):
        return False
    if not os.path.exists(template_path):
        return False

    img = cv2.imread(screen_path)
    tmpl = cv2.imread(template_path)
    if img is None:
        return False

    # 2. Template matching
    res = cv2.matchTemplate(img, tmpl, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    # Not found
    if max_val < threshold:
        return False

    h, w = tmpl.shape[:2]
    cx = max_loc[0] + w // 2
    cy = max_loc[1] + h // 2

    run_adb(["shell", "input", "tap", str(cx), str(cy)], device_id=device)
    return True


def is_template_present(
    screen_bgr: np.ndarray,
    template_path: str,
    threshold: float = 0.85,
    roi: Optional[Tuple[int,int,int,int]] = None,
) -> Tuple[bool, float]:
    """
    Returns:
      (present, score)

    present = True  -> template FOUND
    present = False -> template NOT FOUND
    """
    screen = cv2.imread(screen_bgr)
    tpl = cv2.imread(template_path)

    score, _, _ = match_template(
        haystack_bgr=screen,
        needle_bgr=tpl,
        roi=roi,
    )

    return (score >= threshold), score


def is_btn_absent(
    screen_bgr: np.ndarray,
    close_tpl: str,
    roi: Optional[Tuple[int,int,int,int]] = None,
) -> bool:
    """
    True  -> btn_close NOT on screen
    False -> btn_close IS on screen
    """
    present, score = is_template_present(
        screen_bgr,
        close_tpl,
        threshold=0.85,
        roi=roi,
    )
    return not present


# -------------------------
# Central loop (generic)
# Emits JSONL (one JSON object per line) for Node to forward via SSE.
# -------------------------
def emit(obj: Dict[str, Any]):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def parse_args(argv: List[str]) -> Dict[str, Any]:
    if len(argv) < 2:
        raise SystemExit("Usage: auto_puzzle.py <device-id> [--interval-ms N] [--workdir PATH]")
    device_id = argv[1]
    interval_ms = 1200
    workdir = "templates"

    if "--interval-ms" in argv:
        interval_ms = int(argv[argv.index("--interval-ms")+1])
    if "--workdir" in argv:
        workdir = argv[argv.index("--workdir")+1]
    return {"device_id": device_id, "interval_ms": interval_ms, "workdir": workdir}


def reset_tmp_dir(tmp_dir: str):
    """
    Remove and recreate tmp directory.
    Safe to call at game start.
    """
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)


def main():
    args = parse_args(sys.argv)
    device_id = args["device_id"]
    interval_ms = args["interval_ms"]
    workdir = args["workdir"]
    tmp = f"{workdir}/tmp"
    btn_throw = f"{workdir}/btn_throw.png"
    btn_close = f"{workdir}/btn_close.png"
    btn_done = f"{workdir}/btn_done.png"
    congtats = f"{workdir}/congrats.png"
    empty = f"{workdir}/empty.png"
    REAL_THRESHOLD = 0.85
    os.makedirs(workdir, exist_ok=True)
    reset_tmp_dir(tmp)

    i = 0
    while True:
        t0 = 0
        # screen_path = os.path.join(tmp, f"screen_{device_id.replace(':','_')}_{i:06d}.png")
        screen_path = os.path.join(tmp, f"screen_{device_id.replace(':','_')}.png")
        # try:
        run_vision = False
        screencap(device_id, screen_path)
        # emit({"type": "step", "msg": "screencap_ok", "i": i, "screen": screen_path})

        # Check and click on 'throw btn' 798:841
        if not is_btn_absent(screen_path, btn_throw):
            click_template(device_id, screen_path, btn_throw, REAL_THRESHOLD)
            screencap(device_id, screen_path)
            if not is_btn_absent(screen_path, empty):
                emit({"type": "error", "msg": "out_of_bait", "i": i})
            # emit({"type": "step", "msg": "started_throw", "i": i})
        else:
            # 1464:845
            if is_btn_absent(screen_path, btn_close):
                emit({"type": "error", "i": i, "msg": "Not in fishing position!"})
            click_template(device_id, screen_path, btn_close, REAL_THRESHOLD)
            break

        random_sleep(10, 12)
        max_try = 10
        waite_second = 1
        for i in range(max_try):
            if (i >=max_try):
                emit({"type": "error", "i": i, "msg": "Error when fishing!"})
            screencap(device_id, screen_path)
            random_sleep(0.2, 0.6)
            if is_btn_absent(screen_path, btn_close):
                # emit({"type": "decision","msg": "check_fishing"})
                screencap(device_id, screen_path)
                random_sleep(0.2, 0.4)
                if not is_btn_absent(screen_path, btn_done):
                    click_template(device_id, screen_path, btn_done, REAL_THRESHOLD)
                    # emit({"type": "step", "msg": "fishing_failed", "i": i})
                    run_vision = False
                    random_sleep(0.2, 0.4)
                    break
                else:
                    # emit({"type": "step", "msg": "fishing_success", "i": i})
                    run_vision = True
                    random_sleep(0.2, 0.4)
                    break
            time.sleep(waite_second)

        if run_vision:
            pairs, vision_out = android_bot(screen_path)
            swipe_pairs(device_id, pairs, duration_ms=320, jitter=8)
            # emit({"type": "step", "i": i, "msg": f"swiped {len(pairs)} pairs"})
            screencap(device_id, screen_path)
            random_sleep(0.2, 0.5)

            # Check cau ca thanh cong khong se close popup
            if not is_btn_absent(screen_path, btn_done):
                click_template(device_id, screen_path, btn_done, REAL_THRESHOLD)
                emit({"type": "step", "i": i, "msg": "fishing_failed"})

            elif not is_btn_absent(screen_path, congtats):
                # random tap outof congrats popup
                # print("Found CONGRATS")
                emit({"type": "step", "i": i, "msg": "fishing_success"})
                tap(device_id, 1198, 743)
                time.sleep(0.1)
                screencap(device_id, screen_path)
                click_template(device_id, screen_path, btn_done, REAL_THRESHOLD)
                time.sleep(0.4)
            else:
                time.sleep(5)

        # except Exception as e:
        #     emit({"type": "error", "i": i, "msg": str(e)})

        i += 1
        dt = time.time() - t0
        sleep_s = max(0.0, interval_ms/1000.0 - dt)
        time.sleep(sleep_s)


if __name__ == "__main__":
    main()
