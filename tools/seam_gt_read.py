#!/usr/bin/env python3
"""Read hand-painted seam marks back into ground-truth control points.

Point this at the FOLDER of marked PNGs (default: the RAW export folder). It
detects bright pure-color pixels on each page, clusters them into control
points (centroids), and writes them to tools/seam_labels.json. It also renders
an overlay PDF connecting your points so you can confirm the read.

A page with NO colored marks is recorded as having no seam (useful negative).
Keep the PNGs lossless - re-exporting as JPEG crushes the mark color.

Usage: python3 tools/seam_gt_read.py ["~/Downloads/Rach 2 - seam ground truth RAW"]
"""
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seam_gt_prep import MANIFEST, OUT_DIR, png_name  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = os.path.join(HERE, "seam_labels.json")
OVERLAY = os.path.expanduser("~/Downloads/Rach 2 - seam gt (READBACK).pdf")

MIN_BLOB_AREA = 40      # px; ignore stray specks / chroma noise
MIN_SAT = 200           # HSV S: painted pure color (natural scans top out ~200)
MIN_VAL = 150           # HSV V threshold for "bright"


def detect_curve(rgb):
    """Trace a painted seam line: for each row it covers, take the mean x.

    Returns (ys_frac, xs_frac) sampled as control points along the drawn line,
    or (None, None) if no mark is present. Handles thick lines (mean across
    thickness) and dotted marks alike.
    """
    H, W = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = (hsv[:, :, 1] >= MIN_SAT) & (hsv[:, :, 2] >= MIN_VAL)
    if mask.sum() < MIN_BLOB_AREA:
        return None, None
    rows = np.where(mask.any(axis=1))[0]
    y0, y1 = rows.min(), rows.max()
    # per-row mean x over the drawn extent
    ys, xs = [], []
    xs_idx = np.arange(W)
    for y in range(y0, y1 + 1):
        m = mask[y]
        if m.any():
            ys.append(y)
            xs.append(xs_idx[m].mean())
    ys = np.array(ys, float)
    xs = np.array(xs, float)
    # downsample to ~9 control points along the covered span
    n = min(9, len(ys))
    picks = np.linspace(0, len(ys) - 1, n).round().astype(int)
    return ys[picks] / H, xs[picks] / W


def main():
    src_dir = os.path.expanduser(sys.argv[1]) if len(sys.argv) > 1 else OUT_DIR
    if not os.path.isdir(src_dir):
        print(f"Not a folder: {src_dir}\n")
        print(__doc__)
        sys.exit(1)

    cases, overlays = [], []
    for i, (part, file, page, note) in enumerate(MANIFEST, 1):
        path = os.path.join(src_dir, png_name(i, part, page))
        if not os.path.exists(path):
            print(f"  P{i:02d} MISSING: {os.path.basename(path)}")
            continue
        rgb = np.array(Image.open(path).convert("RGB"))
        ys, xs = detect_curve(rgb)
        pts = [] if ys is None else list(zip(xs, ys))
        side = "right"
        if ys is not None and float(np.mean(xs)) < 0.5:
            side = "left"
        rec = {"part": part, "file": file, "page": page, "side": side,
               "points": [[round(float(y), 4), round(float(x), 4)]
                          for (x, y) in pts]}
        cases.append(rec)
        if ys is None:
            status = "NO SEAM (no marks)"
        else:
            status = (f"{len(pts)} pts, x {xs.min()*100:.1f}-{xs.max()*100:.1f}%,"
                      f" y {ys.min()*100:.0f}-{ys.max()*100:.0f}%")
        print(f"  P{i:02d} {part} p{page} [{note}]: {status}")

        ov = rgb.copy()
        H, W = ov.shape[:2]
        if ys is not None and len(pts) >= 2:
            px = [int(x * W) for x, _ in pts]
            py = [int(y * H) for _, y in pts]
            yf = np.arange(py[0], py[-1] + 1)
            xf = np.interp(yf, py, px).astype(int)
            for k, yy in enumerate(yf):
                xx = xf[k]
                if 0 <= xx < W:
                    ov[yy, max(0, xx-1):xx+2] = (0, 200, 0)
            for x, y in pts:
                cv2.circle(ov, (int(x*W), int(y*H)), 10, (255, 0, 255), 3)
        overlays.append(Image.fromarray(ov))

    with open(LABELS, "w") as f:
        json.dump({
            "_doc": "Ground-truth binding seams from hand-painted marks. "
                    "points are [y_frac, x_frac] top->bottom in raw-page "
                    "normalized coords. Empty points = no seam on that page.",
            "cases": cases,
        }, f, indent=2)
    print(f"\nWrote {LABELS}")

    if overlays:
        overlays[0].save(OVERLAY, save_all=True, append_images=overlays[1:],
                         resolution=200.0, quality=80)
        print(f"Wrote overlay {OVERLAY}")


if __name__ == "__main__":
    main()
