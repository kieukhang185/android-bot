import time
import statistics
from pathlib import Path
import cv2

# ---- paste your clamp_roi here ----
from typing import Tuple

def clamp_roi(roi: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = roi
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(1, min(w, x2))
    y2 = max(1, min(h, y2))
    return x1, y1, x2, y2


def match_time_ms(screen, tpl, roi=None, method=cv2.TM_CCOEFF_NORMED):
    h, w = screen.shape[:2]
    if roi is not None:
        x1, y1, x2, y2 = clamp_roi(roi, w, h)
        hs = screen[y1:y2, x1:x2]
    else:
        hs = screen

    t0 = time.perf_counter()
    _ = cv2.matchTemplate(hs, tpl, method)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0


def summarize(samples):
    samples = sorted(samples)
    avg = statistics.mean(samples)
    p95 = samples[int(0.95 * (len(samples) - 1))]
    mn = samples[0]
    mx = samples[-1]
    return avg, p95, mn, mx


def bench_one(screen, tpl, roi, loops=50, warmup=20):
    # Warm-up (important: first calls can be slower)
    print("bench_one")
    for _ in range(warmup):
        match_time_ms(screen, tpl, roi=None)
        match_time_ms(screen, tpl, roi=roi)

    full = [match_time_ms(screen, tpl, roi=None) for _ in range(loops)]
    roii = [match_time_ms(screen, tpl, roi=roi) for _ in range(loops)]

    f_avg, f_p95, f_min, f_max = summarize(full)
    r_avg, r_p95, r_min, r_max = summarize(roii)
    speedup = f_avg / r_avg if r_avg > 0 else float("inf")

    return {
        "full_avg_ms": f_avg, "full_p95_ms": f_p95, "full_min_ms": f_min, "full_max_ms": f_max,
        "roi_avg_ms": r_avg,  "roi_p95_ms": r_p95,  "roi_min_ms": r_min,  "roi_max_ms": r_max,
        "speedup_x": speedup,
    }


def main():
    # 1) Provide a real screenshot path
    SCREEN_PATH = "debug_screen.png"  # <-- set this
    screen = cv2.imread(SCREEN_PATH, cv2.IMREAD_COLOR)
    if screen is None:
        raise SystemExit(f"Cannot read screenshot: {SCREEN_PATH}")

    sh, sw = screen.shape[:2]
    print(f"Screen: {sw}x{sh}")

    # 2) Provide templates + ROIs
    # ROI examples below are percentages; adjust to your UI.
    # ROI = (x1, y1, x2, y2)

    rois = {
        "btn_throw": (int(sw*0.60), int(sh*0.65), int(sw*0.98), int(sh*0.98)),  # bottom-right-ish
        "btn_done":  (int(sw*0.60), int(sh*0.65), int(sw*0.98), int(sh*0.98)),
        "btn_close": (int(sw*0.75), int(sh*0.00), int(sw*0.99), int(sh*0.25)),  # top-right-ish
    }

    templates = {
        "btn_throw": f"../server/templates/btn_throw_{sw}x{sh}.png",
        "btn_done":  f"../server/templates/btn_done_{sw}x{sh}.png",
        "btn_close": f"../server/templates/btn_close_{sw}x{sh}.png",
    }

    print("Testing here")
    for name, tpath in templates.items():
        print("In for")
        tpl = cv2.imread(tpath, cv2.IMREAD_COLOR)
        if tpl is None:
            print(f"[SKIP] missing template: {tpath}")
            continue

        stats = bench_one(screen, tpl, rois[name], loops=200, warmup=20)
        print(f"\n=== {name} ===")
        print(f"Full: avg {stats['full_avg_ms']:.3f} ms | p95 {stats['full_p95_ms']:.3f} ms")
        print(f" ROI: avg {stats['roi_avg_ms']:.3f} ms | p95 {stats['roi_p95_ms']:.3f} ms")
        print(f"Speedup: {stats['speedup_x']:.2f}x")


if __name__ == "__main__":
    main()
