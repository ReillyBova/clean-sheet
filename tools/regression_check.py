#!/usr/bin/env python3
"""Lightweight regression harness for FlatScan.

Processes a fixed manifest of representative pages and compares the output to
stored golden thumbnails, so an iterative change to the pipeline can be checked
for regressions on pages that already look good before it ships.

    python tools/regression_check.py capture    # (re)generate goldens from current code
    python tools/regression_check.py check      # compare current output to goldens

Goldens are downscaled grayscale PNGs (small enough to commit) -- enough to catch
structural regressions (clipping, misplaced seams, lost content) without storing
full-resolution rasters. ``check`` prints a per-page mean abs diff and flags any
page whose diff exceeds --threshold for a human look.

The manifest deliberately over-samples binding/booklet geometries (run-off pages,
real neighbour bleed, title pages, blank versos) because that is what the crease /
seam detector affects most.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pypdfium2 as pdfium

REPO = Path(__file__).resolve().parent.parent
GOLD = REPO / "tools" / "regression" / "golden"
RACH = Path.home() / "Downloads" / "Rach 2"
THUMB_W = 900

# id -> (source pdf, [1-based pages], why-it-is-here)
MANIFEST: dict[str, tuple[Path, list[int], str]] = {
    # russlan reference pair (whole doc); the committed clean output is the anchor.
    "russlan": (REPO / "samples" / "russlan_raw_phone_scan.pdf", [1, 2, 3, 4], "committed reference scan"),
    # Oboe I -- the canonical binding cases.
    "oboe1_p10": (RACH / "Rach 2 Oboe I.pdf", [10], "even run-off right, closing barlines"),
    "oboe1_p11": (RACH / "Rach 2 Oboe I.pdf", [11], "odd run-off left, leading clefs"),
    "oboe1_p16": (RACH / "Rach 2 Oboe I.pdf", [16], "even with REAL neighbour bleed -- must stay excluded"),
    "oboe3_p4": (RACH / "Rach 2 Oboe III.pdf", [4], "bottom-system closing barline outlier"),
    "tuba_p4": (RACH / "Rach 2 Tuba_l.pdf", [4], "binding curl"),
    # Title / label pages fixed this round -- must stay whole.
    "cello_p3": (RACH / "Rach 2 Cello_l.pdf", [3], "Violoncello title box near binding"),
    "clar1_p13": (RACH / "Rach 2 Clarinet I.pdf", [13], "left margin labels/clefs restored"),
    "clar1_p21": (RACH / "Rach 2 Clarinet I.pdf", [21], "'Clarinet I in Bb' label restored"),
    "bclar_p9": (RACH / "Rach 2 Bass Clarinet.pdf", [9], "'in A' indented-system label"),
    # Neighbour bleed on a non-blank recto -- must stay excluded.
    "clar1_p27": (RACH / "Rach 2 Clarinet I.pdf", [27], "recto neighbour bleed"),
    # Plain interior pages -- should be untouched.
    "cello_p10": (RACH / "Rach 2 Cello_l.pdf", [10], "plain even"),
    "clar1_p8": (RACH / "Rach 2 Clarinet I.pdf", [8], "plain odd"),
    "violin1_p5": (RACH / "Rach 2 Violin I_l.pdf", [5], "plain"),
    "horn2_p14": (RACH / "Rach 2 Horn II.pdf", [14], "plain"),
}

BASE_ARGS = ["--width-in", "8.9375", "--height-in", "12", "--dpi", "400",
             "--straighten", "--booklet", "--no-pad-even"]


def _thumb(arr_bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    return cv2.resize(g, (THUMB_W, int(h * THUMB_W / w)), interpolation=cv2.INTER_AREA)


def _process(src: Path, pages: list[int]) -> list[np.ndarray]:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "out.pdf"
        cmd = ["python3", str(REPO / "flat_scan.py"), str(src), str(out),
               *BASE_ARGS, "--pages", ",".join(str(p) for p in pages), "--jobs", "2"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not out.exists():
            raise RuntimeError(f"processing failed for {src.name} {pages}:\n{r.stderr[-2000:]}")
        pdf = pdfium.PdfDocument(str(out))
        return [np.array(pdf[i].render(scale=1.0).to_pil().convert("RGB"))[:, :, ::-1] for i in range(len(pdf))]


def run(mode: str, only: list[str] | None, threshold: float) -> int:
    GOLD.mkdir(parents=True, exist_ok=True)
    ids = only or list(MANIFEST)
    worst = 0.0
    flagged: list[tuple[str, float]] = []
    for id_ in ids:
        src, pages, why = MANIFEST[id_]
        if not src.exists():
            print(f"  SKIP {id_:14s} (missing {src})")
            continue
        try:
            outs = _process(src, pages)
        except RuntimeError as e:
            print(f"  FAIL {id_:14s} {e}")
            flagged.append((id_, float("inf")))
            continue
        for pi, arr in enumerate(outs):
            key = f"{id_}_{pages[pi]}" if len(pages) > 1 else id_
            gp = GOLD / f"{key}.png"
            th = _thumb(arr)
            if mode == "capture":
                cv2.imwrite(str(gp), th)
                print(f"  saved {key}")
                continue
            if not gp.exists():
                print(f"  ---- {key:16s} no golden (run capture)")
                continue
            gold = cv2.imread(str(gp), cv2.IMREAD_GRAYSCALE)
            if gold.shape != th.shape:
                th = cv2.resize(th, (gold.shape[1], gold.shape[0]))
            d = float(np.abs(gold.astype(int) - th.astype(int)).mean())
            worst = max(worst, d)
            flag = d > threshold
            if flag:
                flagged.append((key, d))
            print(f"  {'DIFF' if flag else 'ok  '} {key:16s} meandiff={d:6.3f}   [{why}]")
    if mode == "check":
        print(f"\nworst meandiff={worst:.3f}; {len(flagged)} page(s) over threshold {threshold}")
        if flagged:
            print("review:", ", ".join(f"{k}({d:.2f})" for k, d in flagged))
            return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["capture", "check"])
    ap.add_argument("--only", nargs="*", help="subset of manifest ids")
    ap.add_argument("--threshold", type=float, default=1.5, help="meandiff to flag for review")
    args = ap.parse_args()
    return run(args.mode, args.only, args.threshold)


if __name__ == "__main__":
    sys.exit(main())
