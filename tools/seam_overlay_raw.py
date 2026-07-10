#!/usr/bin/env python3
"""Overlay the detected seam on raw pages so its placement can be judged.

Runs detect_seam_for_page on each flagged page and draws the seam curve (red)
plus the top/bottom fold apexes (orange) on the raw capture, captioned with the
chosen side and cue. Purely diagnostic -- no clipping, no processing.

Usage: python3 tools/seam_overlay_raw.py
Output: ~/Downloads/Rach 2 - SEAM CHECK (raw).pdf
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flat_scan as fs  # noqa: E402
from regen_fixes import FIXES, resolve_raw  # noqa: E402

OUT = os.path.expanduser("~/Downloads/Rach 2 - SEAM CHECK (raw).pdf")


def _font(sz):
    for p in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf",
              "/System/Library/Fonts/Helvetica.ttc"):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def render(part, page, klass, status, note):
    from pathlib import Path
    raw = resolve_raw(part)
    inp = fs.InputPages(Path(raw), render_scale=1.0)
    bgr = inp.render_page_bgr(page - 1)
    h, w = bgr.shape[:2]
    mask, _ = fs.segment_page(bgr)
    seam = fs.detect_seam_for_page(bgr, mask)
    vis = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    label = f"{part} p{page}  [{klass}]"
    if seam is None:
        label += "  -- NO SEAM (floating / no binding side)"
    else:
        xs = np.round(seam["curve"]).astype(int)
        for y in range(0, h, 2):
            x = xs[y]
            if 0 <= x < w:
                vis[y, max(0, x - 3):x + 4] = (255, 0, 0)
        for (ax, ay) in (seam["top_apex"], seam["bot_apex"]):
            cv2.circle(vis, (int(ax), int(ay)), 16, (255, 140, 0), 6)
        label += f"  side:{seam['side']} cue:{seam['cue']} conf:{seam['conf']:.3f}"
    # downscale + caption
    s = 900 / w
    small = cv2.resize(vis, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    H, W = small.shape[:2]
    band = 70
    canvas = np.full((H + band, W, 3), 255, np.uint8)
    canvas[band:] = small
    pil = Image.fromarray(canvas)
    d = ImageDraw.Draw(pil)
    d.text((10, 6), label, fill=(0, 0, 0), font=_font(22))
    d.text((10, 38), note[:120], fill=(90, 90, 90), font=_font(18))
    return pil


def main():
    pages = []
    for i, (part, page, klass, status, note) in enumerate(FIXES, 1):
        print(f"  [{i}/{len(FIXES)}] {part} p{page}")
        try:
            pages.append(render(part, page, klass, status, note))
        except Exception as e:  # noqa: BLE001
            print(f"    ERR {e}")
    pages[0].save(OUT, save_all=True, append_images=pages[1:],
                  resolution=150.0, quality=80)
    print(f"\nWrote {OUT}  ({len(pages)} pages)")


if __name__ == "__main__":
    main()
