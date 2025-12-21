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

from vision_layout_dinov2 import match_ghost_to_real, LayoutCfg, ROI, HeuristicCfg
import warnings

warnings.filterwarnings("ignore", message="xFormers is not available")
warnings.filterwarnings("ignore", message="Using cache found in /home/vagrant/.cache/torch/hub/facebookresearch_dinov2_main")

def center_of_box(box_xywh):
    x, y, w, h = box_xywh
    return (x + w // 2, y + h // 2)

def jitter_point(pt, jitter=8):
    return (pt[0] + random.randint(-jitter, jitter),
            pt[1] + random.randint(-jitter, jitter))

def swipe_pairs(device_id: str, pairs, duration_ms=320, jitter=8):
    """
    pairs: list of (from_xy, to_xy) where each is (x,y) in screen pixels.
    """
    for (src, dst) in pairs:
        sx, sy = jitter_point(src, jitter)
        tx, ty = jitter_point(dst, jitter)
        swipe(device_id, sx, sy, tx, ty, duration_ms)
        time.sleep(0.15)

# -------------------------
# Generic ADB helpers
# -------------------------
def run_adb(args: List[str], device_id: Optional[str] = None, timeout: int = 20) -> str:
    cmd = ["adb"]
    if device_id:
        cmd += ["-s", device_id]
    cmd += args
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip() or p.stdout.strip() or f"adb failed: {cmd}")
    return p.stdout

def screencap(device_id: str, out_path: str) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cmd = ["adb", "-s", device_id, "exec-out", "screencap", "-p"]
    with open(out_path, "wb") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, timeout=30)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode("utf-8", "ignore").strip() or "screencap failed")
    return out_path

# Generic UI-test actions (not app/game specific)
def tap(device_id: str, x: int, y: int):
    run_adb(["shell", "input", "tap", str(x), str(y)], device_id=device_id)

def swipe(device_id: str, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300):
    run_adb(["shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(duration_ms)], device_id=device_id)

# -------------------------
# Template matching utilities (optional)
# -------------------------
def read_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return img

# def match_template(haystack_bgr: np.ndarray, needle_bgr: np.ndarray, roi: Optional[Tuple[int,int,int,int]] = None) -> Tuple[float, Tuple[int,int], Tuple[int,int]]:
#     hs = haystack_bgr
#     offx, offy = 0, 0
#     if roi:
#         x1,y1,x2,y2 = roi
#         hs = haystack_bgr[y1:y2, x1:x2]
#         offx, offy = x1, y1
#     res = cv2.matchTemplate(hs, needle_bgr, cv2.TM_CCOEFF_NORMED)
#     _, max_val, _, max_loc = cv2.minMaxLoc(res)
#     h, w = needle_bgr.shape[:2]
#     tl = (max_loc[0] + offx, max_loc[1] + offy)
#     br = (tl[0] + w, tl[1] + h)
#     return float(max_val), tl, br

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


def check_exists(screen_path, template_path, threshold=0.85):
    img = cv2.imread(screen_path)
    tmpl = cv2.imread(template_path)

    if img is None:
        return False

    # 2. Template matching
    result = cv2.matchTemplate(img, tmpl, cv2.TM_CCOEFF_NORMED)
    locations = (result >= threshold).any()
    if locations:
        print(f"Location: {locations}")
        return True
    else:
        return False

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
    print(f"Clicked on: {cx}:{cy}")
    return True

def serializable_data(data):
    serializable_data = [int(x) for x in data]

    # Serialize to JSON
    json_output = json.dumps(serializable_data)
    return json_output  # Output: [798, 839]

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

    emit({
        "type": "check",
        "target": "btn_close",
        "present": present,
        "score": round(score, 3),
    })

    return not present


# -------------------------
# DINOv2 vision config
# -------------------------
VISION_CFG = LayoutCfg(
    right_roi=ROI(0.87, 0.08, 0.99, 0.92),
    middle_roi=ROI(0.12, 0.55, 0.88, 0.73),
    right_heur=HeuristicCfg(min_area_ratio=0.001, max_area_ratio=0.12, min_side_px=28),
    middle_heur=HeuristicCfg(min_area_ratio=0.001, max_area_ratio=0.12, min_side_px=28),
    sort_right="top_to_bottom",
    sort_middle="left_to_right",
)

def build_swipe_plan_sorted(vision_out):
    matches = vision_out.get("matches", [])
    matches = sorted(matches, key=lambda m: m["ghost_index"])  # hoặc sort theo box x/y
    return [(center_of_box(m["real_box"]), center_of_box(m["ghost_box"])) for m in matches]

def run_dinov2(screen_path: str, debug_path: str) -> Dict[str, Any]:
    return match_ghost_to_real(
        screenshot_path=screen_path,
        cfg=VISION_CFG,
        debug_out_path=debug_path,
        model_name="dinov2_vits14",
    )

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
    real_path = os.path.realpath(f"./{tmp}/*")
    btn_throw = f"{workdir}/btn_throw.png"
    btn_close = f"{workdir}/btn_close.png"
    btn_done = f"{workdir}/btn_done.png"
    congtats = f"{workdir}/congrats.png"
    REAL_THRESHOLD = 0.85
    os.makedirs(workdir, exist_ok=True)
    reset_tmp_dir(tmp)

    emit({"type": "status", "msg": f"python loop started (interval_ms={interval_ms})"})

    i = 0
    while True:
        screen_path = os.path.join(tmp, f"screen_{device_id.replace(':','_')}_{i:06d}.png")
        debug_path  = os.path.join(tmp, f"debug_{device_id.replace(':','_')}_{i:06d}.png")

        try:
            run_vision = False
            screencap(device_id, screen_path)
            emit({"type": "step", "msg": "screencap_ok", "i": i, "screen": screen_path})
            time.sleep(1)

            # Check and click on 'throw btn'
            if check_exists(screen_path, btn_throw, REAL_THRESHOLD):
                click_template(device_id, screen_path, btn_throw, REAL_THRESHOLD)
                emit({"type": "step", "msg": "started_throw", "i": i, "screen": btn_throw})
            else:
                if check_exists(screen_path, btn_close, REAL_THRESHOLD):
                    click_template(device_id, screen_path, btn_close, REAL_THRESHOLD)
                else:
                    emit({"type": "error", "i": i, "msg": "Not in fishing position!"})
                    sys.exit(1)

            emit({"type": "step", "msg": "Waite 10 seconds", "i": i})
            time.sleep(10)
            max_try = 10
            waite_second = 2
            for i in range(max_try):
                screencap(device_id, screen_path)
                time.sleep(1)
                if is_btn_absent(screen_path, btn_close):
                    emit({"type": "decision","msg": "btn_close found → continue main flow"})
                    screencap(device_id, screen_path)
                    time.sleep(1)
                    if not is_btn_absent(screen_path, btn_done):
                        click_template(device_id, screen_path, btn_done, REAL_THRESHOLD)
                        emit({"type": "step", "msg": "fishing_failed", "i": i})
                        run_vision = False
                        break
                    else:
                        emit({"type": "step", "msg": "fishing_success", "i": i})
                        run_vision = True
                        break
                time.sleep(waite_second)
                emit({"type": "step", "msg": f"Waite {waite_second} seconds", "i": i})

            if run_vision:
                out = run_dinov2(screen_path, debug_path)
                emit({"type": "vision", "i": i, "matchesCount": len(out.get("matches", [])), "debug": debug_path})
                swipe_plan = build_swipe_plan_sorted(out)

                # IMPORTANT: For QA/testing YOUR app: execute planned swipes
                swipe_pairs(device_id, swipe_plan, duration_ms=320, jitter=10)

                emit({"type": "step", "i": i, "msg": f"swiped {len(swipe_plan)} pairs"})

                time.sleep(2)
                screencap(device_id, screen_path)
                time.sleep(1)

                # Check cau ca thanh cong khong se close popup
                # pts = match_template(screen_path, congtats, REAL_THRESHOLD)

                if not is_btn_absent(screen_path, congtats):
                    # random tap outof congrats popup
                    print("Found CONGRATS")
                    tap(device_id, 100, 80)

        except Exception as e:
            emit({"type": "error", "i": i, "msg": str(e)})

        # i += 1
        # dt = time.time() - t0
        # sleep_s = max(0.0, interval_ms/1000.0 - dt)
        # time.sleep(sleep_s)

if __name__ == "__main__":
    main()
