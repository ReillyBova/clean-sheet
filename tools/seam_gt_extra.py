#!/usr/bin/env python3
"""Prep/read extra ground-truth seam pages and merge them into seam_labels.json.

Add pages to EXTRA below. `prep` writes clean lossless raw PNGs to mark (pure red
line along the true fold, save as PNG). `read` detects the red line and upserts
each page into tools/seam_labels.json (matched by part+page), so the existing
validator/detector picks them up.

Usage:
  python3 tools/seam_gt_extra.py prep
  python3 tools/seam_gt_extra.py read
"""
import json
import os
import re
import sys

import cv2
import numpy as np
import pypdfium2 as pdfium
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.expanduser("~/Downloads/Rach 2")
OUT_DIR = os.path.expanduser("~/Downloads/Rach 2 - seam GT extra")
LABELS = os.path.join(HERE, "seam_labels.json")

# (part, file, pdf_page)
EXTRA = [
    ("Clarinet I", "Rach 2 Clarinet I.pdf", 21),
    ("Clarinet I", "Rach 2 Clarinet I.pdf", 27),
    ("Horn II",    "Rach 2 Horn II.pdf",    3),
    ("Violin I",   "Rach 2 Violin I_l.pdf", 3),
    ("Trombone I", "Rach 2 Trombone I_l.pdf", 3),
]

MIN_SAT, MIN_VAL, MIN_AREA = 200, 150, 40


def _slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def png_name(part, page):
    return f"{_slug(part)}_p{page}.png"


def prep():
    os.makedirs(OUT_DIR, exist_ok=True)
    for part, file, page in EXTRA:
        pil = pdfium.PdfDocument(os.path.join(RAW_DIR, file))[page - 1] \
            .render(scale=1.0).to_pil().convert("RGB")
        out = os.path.join(OUT_DIR, png_name(part, page))
        pil.save(out)
        print(f"  {part} p{page} -> {os.path.basename(out)}")
    print(f"\nWrote {len(EXTRA)} raw PNGs to:\n  {OUT_DIR}")


def detect_curve(rgb):
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = (hsv[:, :, 1] >= MIN_SAT) & (hsv[:, :, 2] >= MIN_VAL)
    if mask.sum() < MIN_AREA:
        return None, None
    rows = np.where(mask.any(axis=1))[0]
    xs_idx = np.arange(w)
    ys, xs = [], []
    for y in range(rows.min(), rows.max() + 1):
        m = mask[y]
        if m.any():
            ys.append(y)
            xs.append(xs_idx[m].mean())
    ys, xs = np.array(ys, float), np.array(xs, float)
    n = min(9, len(ys))
    picks = np.linspace(0, len(ys) - 1, n).round().astype(int)
    return ys[picks] / h, xs[picks] / w


def read():
    data = json.load(open(LABELS))
    by_key = {(c["part"], c["page"]): c for c in data["cases"]}
    for part, file, page in EXTRA:
        path = os.path.join(OUT_DIR, png_name(part, page))
        if not os.path.exists(path):
            print(f"  {part} p{page}: MISSING PNG")
            continue
        rgb = np.array(Image.open(path).convert("RGB"))
        ys, xs = detect_curve(rgb)
        if ys is None:
            print(f"  {part} p{page}: NO MARK")
            continue
        side = "left" if float(np.mean(xs)) < 0.5 else "right"
        rec = {"part": part, "file": file, "page": page, "side": side,
               "points": [[round(float(y), 4), round(float(x), 4)]
                          for y, x in zip(ys, xs)]}
        by_key[(part, page)] = rec
        print(f"  {part} p{page}: {len(ys)} pts, {side}, "
              f"x {xs.min()*100:.1f}-{xs.max()*100:.1f}%")
    data["cases"] = list(by_key.values())
    with open(LABELS, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nMerged into {LABELS}  ({len(data['cases'])} cases)")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("prep", "read"):
        print(__doc__)
        sys.exit(1)
    (prep if sys.argv[1] == "prep" else read)()
