#!/usr/bin/env python3
"""Validate flatscan_seam.detect_seam against ground truth and render overlays.

Runs the production detector on every labeled case, reports per-row x-error vs
the hand-labeled seam, and writes an overlay PDF: GREEN = ground truth, RED =
detector. Side is auto-decided from the mask's binding side (falling back to the
label's side only if ambiguous).

Usage: python3 tools/seam_validate.py
Output: ~/Downloads/Rach 2 - seam DETECTOR vs GT.pdf
"""
import json
import os
import sys

import cv2
import numpy as np
import pypdfium2 as pdfium
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import flat_scan as fs  # noqa: E402
import flatscan_seam as seam  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.expanduser("~/Downloads/Rach 2")
LABELS = os.path.join(HERE, "seam_labels.json")
OUT = os.path.expanduser("~/Downloads/Rach 2 - seam DETECTOR vs GT.pdf")


def gt_x(points, yf):
    ys = [p[0] for p in points]
    xs = [p[1] for p in points]
    return float(np.interp(yf, ys, xs))


def main():
    data = json.load(open(LABELS))
    pages, errs = [], []
    print(f"{'case':22s} {'side':5s} {'cue':>13s} {'conf':>6s} {'mean%':>6s} {'max%':>6s} {'top%':>6s} {'bot%':>6s}")
    for c in data["cases"]:
        if not c["points"]:
            continue
        doc = pdfium.PdfDocument(os.path.join(RAW_DIR, c["file"]))
        rgb = np.array(doc[c["page"] - 1].render(scale=1.0).to_pil().convert("RGB"))
        bgr = rgb[:, :, ::-1].copy()
        h, w = bgr.shape[:2]
        mask, _ = fs.segment_page(bgr)

        sides = seam.binding_sides(mask)
        side = c["side"] if c["side"] in sides or not sides else sides[0]
        if len(sides) == 1:
            side = sides[0]
        res = seam.detect_seam(mask, side, img_bgr=bgr)
        if res is None:
            print(f"{c['part']+' p'+str(c['page']):22s} {side:5s}  DETECT FAILED")
            pages.append(Image.fromarray(rgb))
            continue

        # endpoint-inclusive metric: sample the full fold incl. the top/bottom
        # ends where extrapolation is weakest (the old [0.1,0.9] hid endpoint error)
        def ex(yf):
            return abs(res["curve"][int(yf*(h-1))]/w - gt_x(c["points"], yf)) * 100
        rerr = [ex(yf) for yf in np.linspace(0.03, 0.97, 33)]
        te = max(ex(0.03), ex(0.05))
        be = max(ex(0.95), ex(0.97))
        errs.append(max(rerr))
        print(f"{c['part']+' p'+str(c['page']):22s} {side:5s} {res['cue']:>13s} "
              f"{res['conf']:6.3f} {np.mean(rerr):6.2f} {max(rerr):6.2f} "
              f"{te:6.2f} {be:6.2f}")

        ov = rgb.copy()
        # ground truth (green)
        for yf in np.linspace(0.0, 1.0, h):
            pass
        ys = np.arange(h)
        gtx = np.array([gt_x(c["points"], y / (h - 1)) * w for y in ys])
        for y in range(0, h, 2):
            for line, col in ((gtx, (0, 200, 0)), (res["curve"], (255, 0, 0))):
                x = int(line[y])
                if 0 <= x < w:
                    ov[y, max(0, x-2):x+3] = col
        # apex markers
        for (ax, ay) in (res["top_apex"], res["bot_apex"]):
            cv2.circle(ov, (int(ax), int(ay)), 14, (0, 90, 255), 4)
        pages.append(Image.fromarray(ov))

    if errs:
        print(f"\nDETECTOR vs GT   mean-of-maxrow={np.mean(errs):.2f}%  "
              f"worst-row={np.max(errs):.2f}%   (n={len(errs)})")
    pages[0].save(OUT, save_all=True, append_images=pages[1:],
                  resolution=200.0, quality=80)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
