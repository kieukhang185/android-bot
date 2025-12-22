import json, os, cv2, random, time
from typing import Dict, Any, Tuple,List
from dataclasses import dataclass
import numpy as np
from utils import swipe
import argparse

@dataclass
class Hit:
    template_id: int
    score: float
    ghost_box_xywh: Tuple[int, int, int, int]  # full coords
    verify_score: float = 0.0

# ---------- your swipe utils ----------
def center_of_box(box_xywh):
    x, y, w, h = box_xywh
    return (x + w // 2, y + h // 2)

# +-jitter
def jitter_point(pt, jitter=8):
    return (pt[0] + random.randint(-jitter, jitter),
            pt[1] + random.randint(-jitter, jitter))


def swipe_pairs(device_id: str, pairs, duration_ms=320, jitter=8):
    for (src, dst) in pairs:
        sx, sy = jitter_point(src, jitter)
        tx, ty = jitter_point(dst, jitter)
        swipe(device_id, sx, sy, tx, ty, duration_ms)
        time.sleep(0.25)

def build_swipe_plan_sorted(vision_out):
    matches = vision_out.get("matches", [])
    matches = sorted(matches, key=lambda m: m["ghost_index"])
    return [(center_of_box(m["real_box"]), center_of_box(m["ghost_box"])) for m in matches]


# ---------- config helpers ----------
def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def pick_profile(cfg: Dict[str, Any], screen_w: int, screen_h: int) -> Dict[str, Any]:
    for p in cfg.get("profiles", []):
        if p.get("screen", {}).get("w") == screen_w and p.get("screen", {}).get("h") == screen_h:
            return p
    avail = [f'{p.get("screen",{}).get("w")}x{p.get("screen",{}).get("h")}' for p in cfg.get("profiles", [])]
    raise RuntimeError(f"No profile for {screen_w}x{screen_h}. Available: {avail}")


def parse_profile(profile: Dict[str, Any]):
    right = profile["right_icons"]["rois"]
    if len(right) != 6:
        raise ValueError("right_icons.rois must have exactly 6 items")
    right_rois = [(int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])) for r in right]

    m = profile["mid_roi"]
    mid_roi = (int(m["x"]), int(m["y"]), int(m["w"]), int(m["h"]))

    matching = profile.get("matching", {})
    threshold = float(matching.get("threshold", 0.50))
    cluster_dx = int(matching.get("cluster_delta_x", 55))
    max_hits = int(matching.get("max_hits", 20))
    blur_ksize = int(matching.get("blur_ksize", 7))
    verify_enabled = str(matching.get("verify_enabled", "true"))
    verify_threshold = int(matching.get("verify_threshold", 0.55))
    verify_method = str(matching.get("verify_method", "TM_CCOEFF_NORMED"))
    verify_inset = int(matching.get("verify_threshold", 6))


    return right_rois, mid_roi, threshold, cluster_dx, max_hits, blur_ksize, verify_enabled, verify_threshold, verify_method, verify_inset


# ---------- vision core (GRAY+BLUR template match) ----------
def preprocess_gray_blur(img_bgr, blur_ksize: int):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    return cv2.GaussianBlur(gray, (k, k), 0)


def find_all_hits_in_mid(mid_bgr, tpl_bgr, threshold: float, max_hits: int, blur_ksize: int):
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
        x, y = max_loc
        hits.append((x, y, float(max_val), tw, th))
        pad = 6
        x1, y1 = max(0, x-pad), max(0, y-pad)
        x2, y2 = min(mw, x+tw+pad), min(mh, y+th+pad)
        work[y1:y2, x1:x2] = 0
    return hits


def verify_hit_with_template(screen_bgr: np.ndarray, ghost_box_xywh: Tuple[int,int,int,int],
                             template_bgr: np.ndarray, method: int,
                             threshold: float, inset: int):
    x, y, w, h = ghost_box_xywh
    ghost = screen_bgr[y:y+h, x:x+w]
    if ghost is None or ghost.size == 0:
        return False, 0.0

    if inset > 0 and w > 2*inset and h > 2*inset:
        ghost = ghost[inset:-inset, inset:-inset]
        if template_bgr.shape[0] > 2*inset and template_bgr.shape[1] > 2*inset:
            template_bgr = template_bgr[inset:-inset, inset:-inset]

    gh, gw = ghost.shape[:2]
    tpl = cv2.resize(template_bgr, (gw, gh), interpolation=cv2.INTER_AREA)

    if isinstance(method, str):
        method = getattr(cv2, method, cv2.TM_CCOEFF_NORMED)
    method = int(method)
    res = cv2.matchTemplate(ghost, tpl, method)
    min_val, max_val, _, _ = cv2.minMaxLoc(res)

    if method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
        score = float(1.0 - min_val)
    else:
        score = float(max_val)

    return (score >= float(threshold)), float(score)

def cluster_hits_by_x(all_hits, delta_x: int):
    if not all_hits:
        return []
    # all_hits items: (template_id, score, gx, gy, gw, gh)
    def x_center(h):  # h tuple
        x, y, w, hgt = h.ghost_box_xywh
        return x + w / 2.0

    all_hits = sorted(all_hits, key=x_center)
    clusters = [[all_hits[0]]]
    for h in all_hits[1:]:
        if abs(x_center(h) - x_center(clusters[-1][0])) < delta_x:
            clusters[-1].append(h)
        else:
            clusters.append([h])
    def best_key(h: Hit):
        return (h.verify_score, h.score)
    # keep best score in each cluster
    return [max(c, key=best_key) for c in clusters]


def make_vision_out(right_rois, hits_sorted_lr):
    matches = []
    for idx, h in enumerate(hits_sorted_lr):
        x, y, w, hgt = h.ghost_box_xywh
        matches.append({
            "ghost_index": idx,
            "template_id": int(h.template_id),
            "score": float(h.score),
            "verify_score": float(getattr(h, "verify_score", 0.0)),
            "real_box": tuple(map(int, right_rois[h.template_id])),
            "ghost_box": (int(x), int(y), int(w), int(hgt)),
        })
    return {"matches": matches}


def hsv_feat(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = hsv[:, :, 1] > 50   # bỏ vùng trắng, giữ vùng màu
    if mask.sum() < 30:
        return hsv.mean(axis=(0,1))
    return hsv[mask].mean(axis=0)

def hsv_distance(a, b):
    return abs(a[0] - b[0]) + 0.6 * abs(a[1] - b[1])

def combine_score(shape_score, color_dist,
                  color_scale=120.0,
                  w_shape=0.75,
                  w_color=0.25):
    color_score = max(0.0, 1.0 - (color_dist / color_scale))
    return w_shape * shape_score + w_color * color_score

# ---------- android_bot entrypoint ----------
def android_bot(screen_path: str, config_path: str = "config.json"):
    if not isinstance(screen_path, (str, bytes, os.PathLike)):
        raise TypeError(f"screen_path must be a path string, got: {type(screen_path)}")

    cfg = load_config(config_path)

    # ROOT output/debug (IMPORTANT: extract strings, not dict)
    output_cfg = cfg.get("output", {}) or {}
    out_root = output_cfg.get("out_root", "out")
    templates_dir = output_cfg.get("templates_dir", "templates")
    debug = bool(cfg.get("debug", True))

    # Ensure dirs are strings
    if not isinstance(out_root, str) or not isinstance(templates_dir, str):
        raise TypeError("config.output.out_root and templates_dir must be strings")

    screen = cv2.imread(str(screen_path))
    if screen is None:
        raise RuntimeError(f"Cannot read image at: {screen_path}")

    sh, sw = screen.shape[:2]
    profile = pick_profile(cfg, sw, sh)
    right_rois, mid_roi, threshold, cluster_dx, max_hits, blur_ksize, verify_enabled, verify_threshold, verify_method, verify_inset = parse_profile(profile)

    # prepare output dirs
    if debug:
        os.makedirs(out_root, exist_ok=True)
        tpl_out_dir = os.path.join(out_root, templates_dir)
        os.makedirs(tpl_out_dir, exist_ok=True)
    else:
        tpl_out_dir = ""

    mx, my, mw, mh = mid_roi
    mid_bgr = screen[my:my+mh, mx:mx+mw]
    if mid_bgr.size == 0:
        raise RuntimeError("MID_ROI empty, check config mid_roi")

    if debug:
        cv2.imwrite(os.path.join(out_root, "mid_roi.png"), mid_bgr)

    # Cache template crops from right ROIs + save
    templates_bgr: Dict[int, np.ndarray] = {}
    for tid, (rx, ry, rw, rh) in enumerate(right_rois):
        crop = screen[ry:ry+rh, rx:rx+rw]
        templates_bgr[int(tid)] = crop
        if debug:
            cv2.imwrite(os.path.join(tpl_out_dir, f"icon_{tid:02d}.png"), crop)

    # Phase1: find hits; Phase2: verify hits
    all_hits: List[Hit] = []
    for tid, tpl_bgr in templates_bgr.items():
        if tpl_bgr is None or tpl_bgr.size == 0:
            continue

        local_hits = find_all_hits_in_mid(mid_bgr, tpl_bgr, threshold, max_hits, blur_ksize)

        for (lx, ly, sc, tw, th) in local_hits:
            gx, gy = mx + int(lx), my + int(ly)
            ghost_box = (int(gx), int(gy), int(tw), int(th))

            v_ok, v_score = (True, 0.0)
            if verify_enabled:
                v_ok, v_score = verify_hit_with_template(
                    screen_bgr=screen,
                    ghost_box_xywh=ghost_box,
                    template_bgr=tpl_bgr,
                    method=verify_method,
                    threshold=verify_threshold,
                    inset=verify_inset,
                )

            if (not verify_enabled) or v_ok:
                all_hits.append(Hit(template_id=int(tid), score=float(sc), ghost_box_xywh=ghost_box, verify_score=float(v_score)))

    if not all_hits:
        return [], {"matches": []}

    # dedup + sort left->right
    hits_clean = cluster_hits_by_x(all_hits, cluster_dx)
    # Sort left->right
    hits_sorted = sorted(hits_clean, key=lambda h: h.ghost_box_xywh[0] + h.ghost_box_xywh[2]/2.0)

    vision_out = make_vision_out(right_rois, hits_sorted)
    pairs = build_swipe_plan_sorted(vision_out)

    if debug:
        dbg = screen.copy()
        for m in vision_out["matches"]:
            x, y, w, h = m["ghost_box"]
            cv2.rectangle(dbg, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # cv2.putText(
            #     dbg,
            #     f"id={m['template_id']} s={m['score']:.2f} v={m['verify_score']:.2f}",
            #     (x, max(0, y-6)),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.45,
            #     (0, 255, 0),
            #     1,
            #     cv2.LINE_AA,
            # )
            cv2.putText(
                dbg,
                f"id={m['template_id']} v={m['verify_score']:.2f}",
                (x, max(0, y-6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        cv2.imwrite(os.path.join(out_root, "debug_hits.png"), dbg)

        with open(os.path.join(out_root, "vision_out.json"), "w", encoding="utf-8") as f:
            json.dump(vision_out, f, ensure_ascii=False, indent=2)

    return pairs, vision_out


# ---------- example usage ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", default="screen.png")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--device", default="")
    ap.add_argument("--do-swipe", action="store_true")
    args = ap.parse_args()

    pairs, vision_out = android_bot(args.screen, args.config)
    print("pairs example:", pairs[0], type(pairs[0]))

    if not pairs:
        print("❌ No hits found. Lower threshold / fix ROIs / tweak preprocess/verify.")
        return

    print("\n✅ Matches (left->right):")
    for m in vision_out["matches"]:
        print(f"idx={m['ghost_index']:02d} tid={m['template_id']} shape={m['score']:.2f} verify={m['verify_score']:.2f} ghost={m['ghost_box']}")

    print("\n🧭 Swipe plan:")
    for i, (src, dst) in enumerate(pairs, start=1):
        print(f"{i:02d}. {src} -> {dst}")

    if args.do_swipe:
        if not args.device:
            raise RuntimeError("--device is required when --do-swipe is set")
        swipe_pairs(args.device, pairs, duration_ms=320, jitter=5)


if __name__ == "__main__":
    main()
