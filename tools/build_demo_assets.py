#!/usr/bin/env python3
"""Build the Clean Sheet webapp demo assets from a real pipeline run.

Renders one sample page, exports each pipeline *stage* as a web-sized image, and
dumps the Coons source->flat UV grid (as JSON) that drives the WebGL un-warp
animation. Re-run to regenerate: `python tools/build_demo_assets.py`.
"""
import json, os, sys
from pathlib import Path
import numpy as np
import cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import flat_scan as fs
import flatscan_dewarp as fd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE = os.path.join(ROOT, "samples", "russlan_raw_phone_scan.pdf")
OUT = os.path.join(ROOT, "docs", "assets")
STAGES = os.path.join(OUT, "stages")
PAGE = 0                      # 0-based page index of the demo
GRID_W, GRID_H = 40, 54       # mesh resolution for the reprojection morph
os.makedirs(STAGES, exist_ok=True)


def web_write(name, img, max_h=920, q=84):
    """Downscale to web height and save as JPEG."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    if h > max_h:
        s = max_h / h
        img = cv2.resize(img, (round(w * s), max_h), interpolation=cv2.INTER_AREA)
    path = os.path.join(STAGES, name + ".jpg")
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return os.path.getsize(path)


def main():
    inp = fs.InputPages(Path(SAMPLE), render_scale=1.0)
    img_bgr = inp.render_page_bgr(PAGE)
    H, W = img_bgr.shape[:2]

    # --- run the real pipeline steps, capturing each stage ---
    mask, seg = fs.segment_page(img_bgr)
    rect, edges = fs.rectify_boundary_coons(img_bgr, mask, 
        out_w=round(8.9375 * 220), out_h=round(12 * 220), smooth=0.045)
    cleaned, ink, thr = fs.clean_ink(rect, mode="soft-gray")
    straightened, _ = fd.straighten_staves(cleaned)
    straightened, _ = fd.deskew_barlines(straightened)
    straightened, _ = fd.align_system_margins(straightened)
    straightened, _ = fd.align_right_margin(straightened)
    straightened, _ = fd.center_content(straightened)

    stages = []
    def add(key, title, blurb, img):
        web_write(key, img)
        stages.append({"key": key, "title": title, "blurb": blurb})

    add("input",     "The capture",       "A phone photo on a table — angled, unevenly lit, curling at the edges.", img_bgr)
    add("lighting",  "Even the lighting", "We estimate the illumination field and divide it out, so paper reads uniformly bright regardless of shadow or glare.", seg["02_lightness_normalized.png"])
    add("segment",   "Find the paper",    "A threshold on the normalized image isolates the sheet as one clean shape — robust to wood grain and shadow.", seg["03_segmentation_seed.png"])
    add("mask",      "The page mask",     "Bridges are severed, holes filled, and boundary notches repaired to leave a single crisp page silhouette.", mask)
    add("uv",        "Map the edges",     "The four page edges drive a Coons patch — a UV grid that knows exactly how the paper is warped in space.", cv2.imread(os.path.join('/tmp/demo_dbg/page_0001','08_source_boundary_uv_grid.png')))
    add("rectified", "Lift it flat",      "The patch un-warps perspective and curl, reprojecting the page onto a true rectangle.", rect)
    add("ink",       "Clean the ink",     "Uneven lighting is removed and the ink is rendered as fine, anti-aliased soft grayscale.", cleaned)
    add("straighten","Straighten staves", "Using the staff lines themselves, residual waviness and skew are ironed flat and the margins squared.", straightened)

    # --- reprojection morph grid: source (photo) positions per mesh vertex ---
    us = np.linspace(0, 1, GRID_W, dtype=np.float32)
    vs = np.linspace(0, 1, GRID_H, dtype=np.float32)
    mx, my = fs.coons_maps(edges, GRID_W, GRID_H)   # (GH, GW) source pixel coords
    # normalize to 0..1 in photo space (also used as texture coords)
    src = np.stack([mx / (W - 1), my / (H - 1)], axis=-1)  # (GH, GW, 2)

    demo = {
        "photo": {"w": int(W), "h": int(H), "image": "stages/input.jpg"},
        "grid": {"w": GRID_W, "h": GRID_H,
                 "src": src.reshape(-1, 2).round(5).tolist()},
        "stages": stages,
        "output_aspect": round((8.9375) / 12.0, 5),
    }
    with open(os.path.join(OUT, "demo.json"), "w") as f:
        json.dump(demo, f)
    print(f"wrote {len(stages)} stages + demo.json (grid {GRID_W}x{GRID_H}) to {OUT}")
    for s in stages:
        print("  -", s["key"])


if __name__ == "__main__":
    main()
