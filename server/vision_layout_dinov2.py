# server/vision_layout_dinov2.py
# Python 3.11+ (works fine on 3.10+)
# DINOv2 + heuristic icon proposal + 1-1 matching (Hungarian)

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from scipy.optimize import linear_sum_assignment


# =========================
# DATA STRUCTURES
# =========================

@dataclass
class ROI:
    x1: float  # 0..1
    y1: float
    x2: float
    y2: float


@dataclass
class HeuristicCfg:
    # Filters for contour boxes
    min_area_ratio: float = 0.0004
    max_area_ratio: float = 0.12
    min_aspect: float = 0.6
    max_aspect: float = 1.6
    min_side_px: int = 18

    # Edge/morph
    morph_kernel: int = 3
    canny1: int = 60
    canny2: int = 140


@dataclass
class LayoutCfg:
    right_roi: ROI
    middle_roi: ROI
    right_heur: HeuristicCfg
    middle_heur: HeuristicCfg
    sort_right: str = "top_to_bottom"
    sort_middle: str = "left_to_right"


# =========================
# ROI + BOX UTILS
# =========================

def crop_roi(img: np.ndarray, roi: ROI) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    h, w = img.shape[:2]
    x1 = int(roi.x1 * w)
    y1 = int(roi.y1 * h)
    x2 = int(roi.x2 * w)
    y2 = int(roi.y2 * h)

    x1 = max(0, min(w - 1, x1))
    x2 = max(1, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(1, min(h, y2))

    return img[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


def bbox_to_global(
    box_xywh: Tuple[int, int, int, int],
    offset_xyxy: Tuple[int, int, int, int],
) -> Tuple[int, int, int, int]:
    ox1, oy1, _, _ = offset_xyxy
    x, y, w, h = box_xywh
    return (x + ox1, y + oy1, w, h)


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)

    inter = iw * ih
    union = aw * ah + bw * bh - inter + 1e-6
    return inter / union


def _dedup_boxes(
    boxes: List[Tuple[int, int, int, int]],
    iou_thr: float = 0.4,
) -> List[Tuple[int, int, int, int]]:
    # Keep bigger boxes first; drop near-duplicates
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept: List[Tuple[int, int, int, int]] = []
    for b in boxes:
        if all(_iou(b, k) < iou_thr for k in kept):
            kept.append(b)
    return kept


def sort_boxes(
    boxes: List[Tuple[int, int, int, int]],
    mode: str,
) -> List[Tuple[int, int, int, int]]:
    if mode == "left_to_right":
        return sorted(boxes, key=lambda b: b[0] + b[2] / 2.0)
    if mode == "top_to_bottom":
        return sorted(boxes, key=lambda b: b[1] + b[3] / 2.0)
    if mode == "row_major":
        return sorted(boxes, key=lambda b: (b[1], b[0]))
    return boxes


# =========================
# HEURISTIC ICON PROPOSAL
# =========================

def propose_icons(roi_img: np.ndarray, cfg: HeuristicCfg) -> List[Tuple[int, int, int, int]]:
    """
    Returns list of bboxes in ROI coords: (x, y, w, h)
    Heuristic: edges -> contours -> bounding rect -> filter by size/aspect/area.
    """
    h, w = roi_img.shape[:2]
    roi_area = float(h * w)

    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    edges = cv2.Canny(gray, cfg.canny1, cfg.canny2)

    k = max(1, int(cfg.morph_kernel))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes: List[Tuple[int, int, int, int]] = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = bw * bh

        if bw < cfg.min_side_px or bh < cfg.min_side_px:
            continue

        ar = bw / float(bh + 1e-6)
        if ar < cfg.min_aspect or ar > cfg.max_aspect:
            continue

        area_ratio = area / roi_area
        if area_ratio < cfg.min_area_ratio or area_ratio > cfg.max_area_ratio:
            continue

        # light padding trim to avoid thick outlines
        pad = int(min(bw, bh) * 0.06)
        x2 = max(0, x + pad)
        y2 = max(0, y + pad)
        bw2 = max(1, bw - 2 * pad)
        bh2 = max(1, bh - 2 * pad)

        bboxes.append((x2, y2, bw2, bh2))

    return _dedup_boxes(bboxes, iou_thr=0.4)


# =========================
# DINOv2 EMBEDDING
# =========================

class DinoV2Embedder:
    def __init__(self, model_name: str = "dinov2_vits14", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("facebookresearch/dinov2", model_name).to(self.device)
        self.model.eval()

        self.pre = transforms.Compose(
            [
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    @torch.inference_mode()
    def embed_bgr(self, crop_bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        x = self.pre(img).unsqueeze(0).to(self.device)
        feat = self.model(x)  # (1, D)
        feat = F.normalize(feat, dim=-1)
        return feat.squeeze(0).detach().cpu()


# =========================
# MATCHING (1-1 ASSIGNMENT)
# =========================

def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def match_ghost_to_real(
    screenshot_path: str,
    cfg: LayoutCfg,
    debug_out_path: Optional[str] = None,
    model_name: str = "dinov2_vits14",
    min_score: float = 0.0,
) -> Dict[str, Any]:
    """
    - Detect 'real' icons in right panel ROI
    - Detect 'ghost' icons in middle row ROI
    - Embed crops by DINOv2
    - Build similarity matrix (G x R)
    - Solve 1-1 assignment by Hungarian algorithm
    - Return matches sorted by ghost_index (left->right slots)

    min_score: filter low-confidence assigned matches (after Hungarian)
    """
    img = cv2.imread(screenshot_path)
    if img is None:
        raise RuntimeError(f"Cannot read screenshot: {screenshot_path}")

    right_img, right_off = crop_roi(img, cfg.right_roi)
    mid_img, mid_off = crop_roi(img, cfg.middle_roi)

    # propose boxes in ROI coords
    right_boxes = propose_icons(right_img, cfg.right_heur)
    mid_boxes = propose_icons(mid_img, cfg.middle_heur)

    # sort for stable indexing
    right_boxes = sort_boxes(right_boxes, cfg.sort_right)
    mid_boxes = sort_boxes(mid_boxes, cfg.sort_middle)

    # global boxes (screen coords)
    right_global = [bbox_to_global(b, right_off) for b in right_boxes]
    mid_global = [bbox_to_global(b, mid_off) for b in mid_boxes]

    # crops (screen coords)
    right_crops = [img[y : y + h, x : x + w] for (x, y, w, h) in right_global]
    mid_crops = [img[y : y + h, x : x + w] for (x, y, w, h) in mid_global]

    if len(right_crops) == 0 or len(mid_crops) == 0:
        return {
            "right_boxes": right_global,
            "middle_boxes": mid_global,
            "matches": [],
            "note": "No boxes detected in one ROI. Tune ROI/heuristics.",
        }

    embedder = DinoV2Embedder(model_name=model_name)

    # Embed all reals: (R, D)
    real_feats = torch.stack([embedder.embed_bgr(c) for c in right_crops], dim=0)

    # Embed all ghosts: (G, D)
    ghost_feats = torch.stack([embedder.embed_bgr(c) for c in mid_crops], dim=0)

    # Similarity matrix (G, R)
    sim = ghost_feats @ real_feats.T  # cosine-like (normalized feats)

    # Hungarian: minimize cost
    cost = (-sim).numpy()  # (G, R)
    gi, ri = linear_sum_assignment(cost)  # matches min(G,R) pairs

    matches: List[Dict[str, Any]] = []
    for g_idx, r_idx in zip(gi.tolist(), ri.tolist()):
        score = float(sim[g_idx, r_idx].item())
        if score < float(min_score):
            continue
        matches.append(
            {
                "ghost_index": int(g_idx),
                "real_index": int(r_idx),
                "score": score,
                "ghost_box": mid_global[g_idx],
                "real_box": right_global[r_idx],
            }
        )

    # Keep match order by ghost slots (left->right)
    matches.sort(key=lambda m: m["ghost_index"])

    # Debug overlay
    if debug_out_path:
        _ensure_parent_dir(debug_out_path)
        dbg = img.copy()

        # draw reals (green)
        for idx, (x, y, w, h) in enumerate(right_global):
            cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(dbg, f"R{idx}", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # draw ghosts (blue)
        for idx, (x, y, w, h) in enumerate(mid_global):
            cv2.rectangle(dbg, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(dbg, f"G{idx}", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # link matches (yellow)
        for m in matches:
            gx, gy, gw, gh = m["ghost_box"]
            rx, ry, rw, rh = m["real_box"]
            gc = (gx + gw // 2, gy + gh // 2)
            rc = (rx + rw // 2, ry + rh // 2)
            cv2.line(dbg, rc, gc, (0, 255, 255), 2)
            cv2.putText(
                dbg,
                f"{m['score']:.2f}",
                (gc[0] + 6, gc[1] + 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
            )

        ok = cv2.imwrite(debug_out_path, dbg)
        if not ok:
            raise RuntimeError(f"cv2.imwrite failed: {debug_out_path}")

    return {
        "right_boxes": right_global,
        "middle_boxes": mid_global,
        "matches": matches,
    }
