#!/usr/bin/env python3
import argparse
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


# -------------------------
# ROI helpers
# -------------------------
def clamp_roi_xyxy(roi: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = roi
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(1, min(w, x2))
    y2 = max(1, min(h, y2))
    return x1, y1, x2, y2


def xyxy_to_xywh(roi: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = roi
    return x1, y1, (x2 - x1), (y2 - y1)


def xywh_to_xyxy(roi: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    x, y, w, h = roi
    return x, y, x + w, y + h


# -------------------------
# Template matching
# -------------------------
@dataclass(frozen=True)
class Match:
    score: float
    top_left: Tuple[int, int]
    bottom_right: Tuple[int, int]

    @property
    def center(self) -> Tuple[int, int]:
        return (
            (self.top_left[0] + self.bottom_right[0]) // 2,
            (self.top_left[1] + self.bottom_right[1]) // 2,
        )


def match_template(
    screen_bgr: np.ndarray,
    tpl_bgr: np.ndarray,
    roi_xyxy: Optional[Tuple[int, int, int, int]] = None,
    method: int = cv2.TM_CCOEFF_NORMED,
) -> Match:
    sh, sw = screen_bgr.shape[:2]
    offx, offy = 0, 0
    hay = screen_bgr

    if roi_xyxy is not None:
        x1, y1, x2, y2 = clamp_roi_xyxy(roi_xyxy, sw, sh)
        hay = screen_bgr[y1:y2, x1:x2]
        offx, offy = x1, y1

    th, tw = tpl_bgr.shape[:2]
    hh, hw = hay.shape[:2]
    if th > hh or tw > hw:
        return Match(0.0, (offx, offy), (offx, offy))

    res = cv2.matchTemplate(hay, tpl_bgr, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    if method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
        score = float(1.0 - min_val)
        loc = min_loc
    else:
        score = float(max_val)
        loc = max_loc

    top_left = (loc[0] + offx, loc[1] + offy)
    bottom_right = (top_left[0] + tw, top_left[1] + th)
    return Match(score, top_left, bottom_right)


def find_template(
    screen_bgr: np.ndarray,
    tpl_bgr: np.ndarray,
    threshold: float,
    roi_xyxy: Optional[Tuple[int, int, int, int]] = None,
) -> Optional[Match]:
    m = match_template(screen_bgr, tpl_bgr, roi_xyxy=roi_xyxy)
    return m if m.score >= threshold else None


# -------------------------
# ROI from match
# -------------------------
def roi_from_match(m: Match, sw: int, sh: int, padding: int) -> Tuple[int, int, int, int]:
    x1, y1 = m.top_left
    x2, y2 = m.bottom_right
    roi = (x1 - padding, y1 - padding, x2 + padding, y2 + padding)
    return clamp_roi_xyxy(roi, sw, sh)


# -------------------------
# Debug output
# -------------------------
def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def put_label(img: np.ndarray, text: str, xy: Tuple[int, int]) -> None:
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def save_debug(screen: np.ndarray, out_dir: str, name: str,
               roi_xyxy: Tuple[int, int, int, int],
               match: Optional[Match],
               threshold: float) -> Tuple[str, str]:
    ensure_dir(out_dir)
    tag = stamp()

    sh, sw = screen.shape[:2]
    rx1, ry1, rx2, ry2 = clamp_roi_xyxy(roi_xyxy, sw, sh)

    crop_path = os.path.join(out_dir, f"{tag}_{name}_roi.png")
    cv2.imwrite(crop_path, screen[ry1:ry2, rx1:rx2])

    annotated = screen.copy()
    cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
    put_label(annotated, f"ROI {name}", (rx1, max(0, ry1 - 8)))

    if match:
        x1, y1 = match.top_left
        x2, y2 = match.bottom_right
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cx, cy = match.center
        cv2.circle(annotated, (cx, cy), 6, (0, 0, 255), -1)
        put_label(annotated, f"FOUND score={match.score:.3f} thr={threshold:.2f}", (10, 30))
    else:
        put_label(annotated, f"NOT FOUND thr={threshold:.2f}", (10, 30))

    ann_path = os.path.join(out_dir, f"{tag}_{name}_annotated.png")
    cv2.imwrite(ann_path, annotated)

    return crop_path, ann_path


# -------------------------
# Main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", required=True, help="screenshot path")
    ap.add_argument("--template", required=True, help="template path")
    ap.add_argument("--threshold", type=float, default=0.85, help="match threshold")
    ap.add_argument("--padding", type=int, default=40, help="padding in pixels for ROI")
    ap.add_argument("--out", default="debug", help="debug output folder")
    ap.add_argument("--name", default="btn", help="label/name for outputs")
    args = ap.parse_args()

    screen = cv2.imread(args.screen, cv2.IMREAD_COLOR)
    if screen is None:
        raise SystemExit(f"Cannot read screen: {args.screen}")

    tpl = cv2.imread(args.template, cv2.IMREAD_COLOR)
    if tpl is None:
        raise SystemExit(f"Cannot read template: {args.template}")

    sh, sw = screen.shape[:2]
    print(f"Screen: {sw}x{sh}")
    print(f"Template: {tpl.shape[1]}x{tpl.shape[0]}")

    # 1) Full-screen find
    full = find_template(screen, tpl, threshold=args.threshold, roi_xyxy=None)
    if not full:
        # still dump debug with a broad ROI = full screen
        full_roi = (0, 0, sw, sh)
        save_debug(screen, args.out, f"{args.name}_full", full_roi, None, args.threshold)
        raise SystemExit(f"NOT FOUND on full screen (thr={args.threshold}). Try lower thr (0.80) or update template.")

    print(f"[FULL] FOUND score={full.score:.4f} top_left={full.top_left} bottom_right={full.bottom_right} center={full.center}")

    # 2) ROI from match + padding
    roi_xyxy = roi_from_match(full, sw, sh, padding=args.padding)
    x, y, w, h = xyxy_to_xywh(roi_xyxy)

    # Percent (xywh)
    xp, yp, wp, hp = x / sw, y / sh, w / sw, h / sh

    print("\nROI (pixel xywh) for config.json:")
    print(f'"{args.name}_roi": {{ "x": {x}, "y": {y}, "w": {w}, "h": {h} }}')

    print("\nROI (pixel xyxy) for matcher:")
    print(f"roi_xyxy = ({roi_xyxy[0]}, {roi_xyxy[1]}, {roi_xyxy[2]}, {roi_xyxy[3]})")

    print("\nROI (percent xywh) portable:")
    print(f'"{args.name}_roi": {{ "x": {xp:.4f}, "y": {yp:.4f}, "w": {wp:.4f}, "h": {hp:.4f} }}')

    # 3) Re-check within ROI (verification)
    again = find_template(screen, tpl, threshold=args.threshold, roi_xyxy=roi_xyxy)
    if again:
        print(f"\n[ROI] FOUND score={again.score:.4f} center={again.center}")
    else:
        print(f"\n[ROI] NOT FOUND (thr={args.threshold}) -> ROI too tight or padding too small.")

    # 4) Save debug images (ROI + match)
    crop_path, ann_path = save_debug(screen, args.out, args.name, roi_xyxy, again, args.threshold)
    print(f"\nSaved ROI crop: {crop_path}")
    print(f"Saved annotated: {ann_path}")


if __name__ == "__main__":
    main()
