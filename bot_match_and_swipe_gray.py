import json
import os
import time
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np


# =========================
# YOUR SWIPE UTILITIES
# =========================
def center_of_box(box_xywh):
    x, y, w, h = box_xywh
    return (x + w // 2, y + h // 2)

def jitter_point(pt, jitter=8):
    return (pt[0] + random.randint(-jitter, jitter),
            pt[1] + random.randint(-jitter, jitter))

def swipe(device_id: str, sx: int, sy: int, tx: int, ty: int, duration_ms: int):
    """
    TODO: Replace with your adb swipe implementation.
    Example:
      os.system(f'adb -s {device_id} shell input swipe {sx} {sy} {tx} {ty} {duration_ms}')
    """
    print(f"[SWIPE] {device_id}: ({sx},{sy}) -> ({tx},{ty}) dur={duration_ms}ms")

def swipe_pairs(device_id: str, pairs, duration_ms=320, jitter=8):
    """
    pairs: list of (from_xy, to_xy) where each is (x,y) in screen pixels.
    """
    for (src, dst) in pairs:
        sx, sy = jitter_point(src, jitter)
        tx, ty = jitter_point(dst, jitter)
        swipe(device_id, sx, sy, tx, ty, duration_ms)
        time.sleep(0.15)

def build_swipe_plan_sorted(vision_out):
    matches = vision_out.get("matches", [])
    matches = sorted(matches, key=lambda m: m["ghost_index"])  # left->right
    return [(center_of_box(m["real_box"]), center_of_box(m["ghost_box"])) for m in matches]


# =========================
# DATA STRUCT
# =========================
@dataclass
class Hit:
    template_id: int
    score: float
    ghost_box_xywh: Tuple[int, int, int, int]  # FULL screen coords


# =========================
# CONFIG
# =========================
def load_profile(config_path: str, screen_w: int, screen_h: int) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    for p in cfg.get("profiles", []):
        if p.get("screen", {}).get("w") == screen_w and p.get("screen", {}).get("h") == screen_h:
            return p
    available = [f'{p.get("screen", {}).get("w")}x{p.get("screen", {}).get("h")}' for p in cfg.get("profiles", [])]
    raise RuntimeError(f"No profile for {screen_w}x{screen_h}. Available: {available}")

def parse_profile(profile: Dict[str, Any]):
    right = profile["right_icons"]["rois"]
    if len(right) != 6:
        raise ValueError("right_icons.rois must have exactly 6 items")
    right_rois = [(int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])) for r in right]

    m = profile["mid_roi"]
    mid_roi = (int(m["x"]), int(m["y"]), int(m["w"]), int(m["h"]))

    matching = profile.get("matching", {})
    threshold = float(matching.get("threshold", 0.50))        # lowered default
    cluster_dx = int(matching.get("cluster_delta_x", 50))     # usually ~= half icon width
    max_hits_per_template = int(matching.get("max_hits", 20))
    blur_ksize = int(matching.get("blur_ksize", 7))           # 5/7/9 are typical

    debug = bool(profile.get("debug", True))
    return right_rois, mid_roi, threshold, cluster_dx, max_hits_per_template, blur_ksize, debug


# =========================
# PREPROCESS (GRAY + BLUR)
# =========================
def preprocess_gray_blur(img_bgr: np.ndarray, blur_ksize: int = 7) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    blur = cv2.GaussianBlur(gray, (k, k), 0)
    return blur


# =========================
# MULTI-HIT SEARCH (LOCAL ROI)
# =========================
def find_all_hits_in_mid(
    mid_bgr: np.ndarray,
    tpl_bgr: np.ndarray,
    threshold: float,
    max_hits: int,
    blur_ksize: int,
) -> List[Tuple[int, int, float, int, int]]:
    """
    Returns list of (x, y, score, tw, th) in MID-LOCAL coords.
    """
    mid = preprocess_gray_blur(mid_bgr, blur_ksize)
    tpl = preprocess_gray_blur(tpl_bgr, blur_ksize)

    th, tw = tpl.shape[:2]
    mh, mw = mid.shape[:2]
    if th > mh or tw > mw:
        return []

    work = mid.copy()
    hits = []

    for _ in range(max_hits):
        res = cv2.matchTemplate(work, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if float(max_val) < threshold:
            break

        x, y = max_loc  # local
        hits.append((x, y, float(max_val), tw, th))

        # mask matched region to find next
        pad = 6
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(mw, x + tw + pad)
        y2 = min(mh, y + th + pad)
        work[y1:y2, x1:x2] = 0

    return hits


def cluster_hits_by_x(hits: List[Hit], delta_x: int) -> List[Hit]:
    """
    Cluster across ALL templates by x-center (keep best score per cluster).
    """
    if not hits:
        return []

    def x_center(h: Hit) -> float:
        x, _, w, _ = h.ghost_box_xywh
        return x + w / 2.0

    hits_sorted = sorted(hits, key=x_center)
    clusters: List[List[Hit]] = [[hits_sorted[0]]]

    for h in hits_sorted[1:]:
        if abs(x_center(h) - x_center(clusters[-1][0])) < delta_x:
            clusters[-1].append(h)
        else:
            clusters.append([h])

    return [max(c, key=lambda h: h.score) for c in clusters]


# =========================
# VISION OUT (FOR YOUR build_swipe_plan_sorted)
# =========================
def make_vision_out(right_rois: List[Tuple[int,int,int,int]], hits_sorted_lr: List[Hit]) -> Dict[str, Any]:
    matches = []
    for idx, h in enumerate(hits_sorted_lr):
        matches.append({
            "ghost_index": idx,
            "template_id": int(h.template_id),
            "score": float(h.score),
            "real_box": tuple(map(int, right_rois[h.template_id])),
            "ghost_box": tuple(map(int, h.ghost_box_xywh)),
        })
    return {"matches": matches}


# =========================
# MAIN
# =========================
def main():
    device_id = "YOUR_DEVICE_ID"  # TODO

    screen = cv2.imread("server/templates/tmp/screen_10.0.2.2_5555_000000.png")
    if screen is None:
        raise RuntimeError("Cannot read screen.png")

    sh, sw = screen.shape[:2]
    profile = load_profile("server/config.json", sw, sh)
    right_rois, mid_roi, threshold, cluster_dx, max_hits, blur_ksize, debug = parse_profile(profile)

    out_root = "out"
    os.makedirs(out_root, exist_ok=True)

    mx, my, mw, mh = mid_roi
    mid_bgr = screen[my:my+mh, mx:mx+mw]
    if mid_bgr.size == 0:
        raise RuntimeError("MID_ROI is empty. Check x,y,w,h.")

    if debug:
        cv2.imwrite(os.path.join(out_root, "mid_roi.png"), mid_bgr)

    # Save right templates (optional debug)
    if debug:
        os.makedirs(os.path.join(out_root, "templates"), exist_ok=True)
        for i, (x,y,w,h) in enumerate(right_rois):
            crop = screen[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(out_root, "templates", f"icon_{i:02d}.png"), crop)

    # 1) Find hits for each template in MID (local), convert to FULL coords
    all_hits: List[Hit] = []
    for tid, (rx, ry, rw, rh) in enumerate(right_rois):
        tpl_bgr = screen[ry:ry+rh, rx:rx+rw]
        if tpl_bgr.size == 0:
            raise RuntimeError(f"Right ROI #{tid} empty. Check x,y,w,h.")

        local_hits = find_all_hits_in_mid(
            mid_bgr=mid_bgr,
            tpl_bgr=tpl_bgr,
            threshold=threshold,
            max_hits=max_hits,
            blur_ksize=blur_ksize,
        )

        for (lx, ly, sc, tw, th) in local_hits:
            # convert local box -> full screen xywh
            gx = mx + lx
            gy = my + ly
            all_hits.append(Hit(
                template_id=tid,
                score=sc,
                ghost_box_xywh=(int(gx), int(gy), int(tw), int(th))
            ))

    if not all_hits:
        print("❌ No hits found.")
        print("Try: lower threshold (0.45-0.55), increase blur_ksize (7/9), or fix MID_ROI.")
        return

    # 2) Cluster duplicates across templates by X, keep best score
    hits_clean = cluster_hits_by_x(all_hits, cluster_dx)

    # 3) Sort left -> right
    hits_sorted = sorted(hits_clean, key=lambda h: h.ghost_box_xywh[0] + h.ghost_box_xywh[2] / 2.0)

    # 4) Build vision_out + pairs
    vision_out = make_vision_out(right_rois, hits_sorted)
    pairs = build_swipe_plan_sorted(vision_out)

    # Debug print
    print(f"\n✅ Profile: {profile.get('name')} ({sw}x{sh})")
    print(f"threshold={threshold}, blur_ksize={blur_ksize}, cluster_dx={cluster_dx}, max_hits={max_hits}")
    print("\n✅ Matches (left->right):")
    for m in vision_out["matches"]:
        print(f"idx={m['ghost_index']:02d} tid={m['template_id']} score={m['score']:.2f} "
              f"real_box={m['real_box']} ghost_box={m['ghost_box']}")

    print("\n🧭 Swipe plan (src -> dst):")
    for i, (src, dst) in enumerate(pairs, start=1):
        print(f"{i:02d}. {src} -> {dst}")

    # Optional overlay
    if debug:
        dbg = screen.copy()
        for m in vision_out["matches"]:
            x, y, w, h = m["ghost_box"]
            cv2.rectangle(dbg, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(dbg, f"id={m['template_id']} {m['score']:.2f}", (x, y-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(out_root, "debug_hits.png"), dbg)
        print(f"\n🖼️ Saved overlay: {out_root}/debug_hits.png")

    # 5) Execute swipes
    swipe_pairs(device_id, pairs, duration_ms=320, jitter=8)


if __name__ == "__main__":
    main()
