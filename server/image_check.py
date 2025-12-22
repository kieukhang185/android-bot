import json, os, cv2, random, time, subprocess
from typing import Dict, Any, Tuple, List, Optional


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
    run_adb(["shell", "input", "tap", str(x), str(y)], device_id=device_id)

def swipe(device_id: str, sx: int, sy: int, tx: int, ty: int, duration_ms: int = 320):
    run_adb(["shell", "input", "swipe", str(sx), str(sy), str(tx), str(ty), str(duration_ms)], device_id=device_id)

# ---------- your swipe utils ----------
def center_of_box(box_xywh):
    x, y, w, h = box_xywh
    return (x + w // 2, y + h // 2)

def jitter_point(pt, jitter=8):
    return (pt[0] + random.randint(-jitter, jitter),
            pt[1] + random.randint(-jitter, jitter))

def swipe_pairs(device_id: str, pairs, duration_ms=320, jitter=8):
    for (src, dst) in pairs:
        sx, sy = jitter_point(src, jitter)
        tx, ty = jitter_point(dst, jitter)
        swipe(device_id, sx, sy, tx, ty, duration_ms)
        time.sleep(0.15)

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

    return right_rois, mid_roi, threshold, cluster_dx, max_hits, blur_ksize


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

def cluster_hits_by_x(all_hits, delta_x: int):
    if not all_hits:
        return []
    # all_hits items: (template_id, score, gx, gy, gw, gh)
    def x_center(h):  # h tuple
        _, _, x, _, w, _ = h
        return x + w / 2.0

    all_hits = sorted(all_hits, key=x_center)
    clusters = [[all_hits[0]]]
    for h in all_hits[1:]:
        if abs(x_center(h) - x_center(clusters[-1][0])) < delta_x:
            clusters[-1].append(h)
        else:
            clusters.append([h])
    # keep best score in each cluster
    return [max(c, key=lambda h: h[1]) for c in clusters]

def make_vision_out(right_rois, hits_sorted_lr):
    matches = []
    for idx, h in enumerate(hits_sorted_lr):
        template_id, score, gx, gy, gw, gh = h
        matches.append({
            "ghost_index": idx,
            "template_id": int(template_id),
            "score": float(score),
            "real_box": tuple(map(int, right_rois[template_id])),
            "ghost_box": (int(gx), int(gy), int(gw), int(gh)),
        })
    return {"matches": matches}


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
    right_rois, mid_roi, threshold, cluster_dx, max_hits, blur_ksize = parse_profile(profile)

    # prepare output dirs
    if debug:
        os.makedirs(out_root, exist_ok=True)
        tpl_out_dir = os.path.join(out_root, templates_dir)
        os.makedirs(tpl_out_dir, exist_ok=True)

    mx, my, mw, mh = mid_roi
    mid_bgr = screen[my:my+mh, mx:mx+mw]
    if mid_bgr.size == 0:
        raise RuntimeError("MID_ROI empty, check config mid_roi")

    if debug:
        cv2.imwrite(os.path.join(out_root, "mid_roi.png"), mid_bgr)

    # find hits
    all_hits = []  # (template_id, score, gx, gy, gw, gh)
    for tid, (rx, ry, rw, rh) in enumerate(right_rois):
        tpl_bgr = screen[ry:ry+rh, rx:rx+rw]
        if tpl_bgr.size == 0:
            raise RuntimeError(f"Right ROI #{tid} empty, check config")

        if debug:
            cv2.imwrite(os.path.join(tpl_out_dir, f"icon_{tid:02d}.png"), tpl_bgr)

        local_hits = find_all_hits_in_mid(mid_bgr, tpl_bgr, threshold, max_hits, blur_ksize)
        for (lx, ly, sc, tw, th) in local_hits:
            gx, gy = mx + lx, my + ly
            all_hits.append((tid, sc, gx, gy, tw, th))

    if not all_hits:
        return []  # no pairs

    # dedup + sort left->right
    hits_clean = cluster_hits_by_x(all_hits, cluster_dx)
    hits_sorted = sorted(hits_clean, key=lambda h: (h[2] + h[4] / 2.0))  # gx + gw/2

    vision_out = make_vision_out(right_rois, hits_sorted)
    pairs = build_swipe_plan_sorted(vision_out)

    if debug:
        # quick overlay
        dbg = screen.copy()
        for m in vision_out["matches"]:
            x, y, w, h = m["ghost_box"]
            cv2.rectangle(dbg, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(dbg, f"id={m['template_id']} {m['score']:.2f}", (x, y-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(out_root, "debug_hits.png"), dbg)

    return pairs


# ---------- example usage ----------
if __name__ == "__main__":
    screen_path = "templates/tmp/screen_10.0.2.2_5555_000000.png"
    pairs = android_bot(screen_path, "config.json")
    print("pairs:", pairs)

    # If you want to swipe:
    # device_id = "10.0.2.2:5555"
    # swipe_pairs(device_id, pairs, duration_ms=320, jitter=8)
