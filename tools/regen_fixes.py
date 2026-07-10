#!/usr/bin/env python3
"""Regenerate the triaged fix pages through the pipeline into one review PDF.

Runs each flagged (part, pdf_page) through the exact production path
(InputPages.render_page_bgr + process_page with --straighten --booklet) and
assembles a single, captioned review PDF so the fixes can be checked in one place
before splicing into the combined books. Nothing is written back to the books.

Usage: python3 tools/regen_fixes.py [--jobs N]
Output: ~/Downloads/Rach 2 - FIX PROOFS.pdf
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import flat_scan as fs  # noqa: E402

RAW_DIR = os.path.expanduser("~/Downloads/Rach 2")
OUT = os.path.expanduser("~/Downloads/Rach 2 - FIX PROOFS.pdf")
OUT_W, OUT_H = 3575, 4800  # 8.9375 x 12 in @ 400 dpi
MODE, BSMOOTH, TBIAS = "soft-gray", 0.045, 16

# (part, pdf_page, klass, status, note)
FIXES = [
    ("Bass Clarinet", 9,  "binding-side-clip", "fix-validated", "the 'in' of 'in A' before indented system partially clipped"),
    ("Bass Clarinet", 18, "blank-verso-bleed", "open", "blankish but grabbing left of page 19"),
    ("Bass Clarinet", 20, "system-end-slope", "open", "top system skews way down at the end"),
    ("Bassoon I", 2,  "blank-verso-bleed", "fix-validated", "page2 blank verso broken"),
    ("Cello", 2,  "blank-verso-bleed", "fix-validated", "blank verso grabbed page3 clef strip on right"),
    ("Cello", 3,  "blank-verso-bleed/top-right-cutoff", "fix-validated", "lost sliver on left; cutoff upper-right p3"),
    ("Clarinet I", 2,  "system-end-slope", "open", "bending at bottom right"),
    ("Clarinet I", 6,  "system-end-slope", "open", "bendy time signature ending 3rd staff from bottom"),
    ("Clarinet I", 13, "binding-side-clip/system-end-slope", "fix-validated", "left margin cutting content (BLOCKER); bottom system skewed"),
    ("Clarinet I", 21, "binding-side-clip", "fix-validated", "cutting off content on the left"),
    ("Clarinet I", 27, "neighbor-bleed-recto", "open", "grabbing some of page 26 (non-blank recto)"),
    ("Clarinet II", 19, "binding-side-clip", "fix-validated", "cutting off some content on the left"),
    ("Flute I", 2,  "blank-verso-bleed", "fix-validated", "page2 blank verso broken"),
    ("Flute I", 3,  "left-top-clip", "open", "page3 clipping a bit near top on the left"),
    ("Flute II", 2,  "system-end-slope", "open", "bottom 5 systems slope at the ends"),
    ("Flute III", 2,  "blank-verso-bleed", "fix-validated", "page2 empty but broken like others"),
    ("Horn I", 2,  "blank-verso-bleed", "open", "blank page issue"),
    ("Horn I", 3,  "binding-side-clip", "open", "cutoff on page3 eg instrument label"),
    ("Horn II", 2,  "blank-verso-bleed", "open", "page2 problem"),
    ("Horn II", 3,  "binding-side-clip", "fix-validated", "page3 left cutoff"),
    ("Horn III", 2,  "blank-verso-bleed", "open", "page2 broken, page3 ok"),
    ("Oboe II", 2,  "system-end-slope", "open", "bending at bottom right, staffs not straight"),
    ("Oboe III", 2,  "blank-verso-bleed", "fix-validated", "blank page problem"),
    ("Trombone I", 2,  "blank-verso-bleed", "fix-validated", "blank page problem"),
    ("Trombone I", 3,  "top-right-cutoff", "open", "page3 top right content cutoff"),
    ("Trombone II", 2,  "blank-verso-bleed", "fix-validated", "page2 regression"),
    ("Trumpet I", 2,  "blank-verso-bleed", "fix-validated", "page2 similarly bad"),
    ("Trumpet I", 12, "blank-verso-bleed", "open", "page 12 broken [publisher BLANK PAGE]"),
    ("Trumpet II", 2,  "blank-verso-bleed", "fix-validated", "page2 should be blank"),
    ("Trumpet II", 12, "blank-verso-bleed", "fix-validated", "blank page regression"),
    ("Viola", 2,  "blank-verso-bleed", "fix-validated", "page2 issue"),
    ("Violin I", 2,  "blank-verso-bleed", "fix-validated", "page2 grabs page3"),
    ("Violin I", 3,  "blank-verso-bleed", "fix-validated", "minor left cutoff"),
    ("Violin II", 2,  "blank-verso-bleed", "fix-validated", "page2 issue"),
    ("Violin II", 22, "system-end-wedge", "open", "each staff ends like < instead of |"),
]


def resolve_raw(part):
    for name in (f"Rach 2 {part}.pdf", f"Rach 2 {part}_l.pdf"):
        p = os.path.join(RAW_DIR, name)
        if os.path.exists(p):
            return p
    return None


def worker(item):
    part, page, klass, status, note = item
    raw = resolve_raw(part)
    if raw is None:
        return (item, None, f"raw not found for {part}")
    try:
        inp = fs.InputPages(Path(raw), render_scale=1.0)
        img = inp.render_page_bgr(page - 1)
        cleaned = fs.process_page(img, page, OUT_W, OUT_H, MODE, BSMOOTH, TBIAS,
                                  None, straighten=True, booklet=True)
        # downscale for review
        scale = 1000 / cleaned.shape[1]
        small = cv2.resize(cleaned, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        return (item, rgb, None)
    except Exception as e:  # noqa: BLE001
        return (item, None, f"{type(e).__name__}: {e}")


def _font(sz):
    for p in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf",
              "/System/Library/Fonts/Helvetica.ttc"):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def captioned(item, rgb):
    part, page, klass, status, note = item
    H, W = rgb.shape[:2]
    band = 88
    canvas = np.full((H + band, W, 3), 255, np.uint8)
    canvas[band:] = rgb
    pil = Image.fromarray(canvas)
    d = ImageDraw.Draw(pil)
    title = f"{part}  -  pdf p{page}   [{klass}]  ({status})"
    d.text((10, 6), title, fill=(0, 0, 0), font=_font(26))
    d.text((10, 44), note[:110], fill=(90, 90, 90), font=_font(20))
    d.rectangle([0, 0, W - 1, H + band - 1], outline=(0, 0, 0), width=2)
    d.line([(0, band - 1), (W, band - 1)], fill=(180, 180, 180), width=1)
    return pil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=5)
    args = ap.parse_args()

    results = {}
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(worker, it): it for it in FIXES}
        done = 0
        for fu in as_completed(futs):
            item, rgb, err = fu.result()
            done += 1
            tag = f"{item[0]} p{item[1]}"
            print(f"  [{done}/{len(FIXES)}] {tag}: "
                  f"{'OK' if err is None else 'ERR ' + err}")
            results[(item[0], item[1])] = (item, rgb, err)

    pages = []
    for it in FIXES:  # keep sorted order
        item, rgb, err = results[(it[0], it[1])]
        if rgb is None:
            ph = np.full((300, 1000, 3), 255, np.uint8)
            cv2.putText(ph, f"{it[0]} p{it[1]} FAILED: {err}", (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)
            rgb = cv2.cvtColor(ph, cv2.COLOR_BGR2RGB)
        pages.append(captioned(item, rgb))

    pages[0].save(OUT, save_all=True, append_images=pages[1:],
                  resolution=150.0, quality=80)
    print(f"\nWrote {OUT}  ({len(pages)} pages)")


if __name__ == "__main__":
    main()
