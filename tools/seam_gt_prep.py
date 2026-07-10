#!/usr/bin/env python3
"""Export raw scan pages as lossless PNGs for binding-seam ground-truth marking.

Dumps one clean, unmodified PNG per curated page (no captions, no guides) into a
folder. Paint the true seam directly on each PNG using a BRIGHT PURE color (pure
red 255,0,0 is ideal): a few dots top -> bottom along the fold. Save as PNG
(lossless - do NOT re-export as JPEG, it crushes the color). Then run
`seam_gt_read.py` on the folder to recover the control points.

Usage: python3 tools/seam_gt_prep.py
Output: ~/Downloads/Rach 2 - seam ground truth RAW/NN_Part_pN.png
"""
import os
import re
import pypdfium2 as pdfium

RAW_DIR = os.path.expanduser("~/Downloads/Rach 2")
OUT_DIR = os.path.expanduser("~/Downloads/Rach 2 - seam ground truth RAW")

# (part label, file, 1-based pdf page, category note)
MANIFEST = [
    ("Horn I",        "Rach 2 Horn I.pdf",        2,  "blank verso"),
    ("Horn II",       "Rach 2 Horn II.pdf",       2,  "blank verso"),
    ("Horn III",      "Rach 2 Horn III.pdf",      2,  "blank verso"),
    ("Cello",         "Rach 2 Cello_l.pdf",       2,  "blank verso"),
    ("Trombone I",    "Rach 2 Trombone I_l.pdf",  2,  "blank verso"),
    ("Bass Clarinet", "Rach 2 Bass Clarinet.pdf", 18, "blank verso"),
    ("Cello",         "Rach 2 Cello_l.pdf",       3,  "music binding"),
    ("Oboe I",        "Rach 2 Oboe I.pdf",        10, "music binding"),
    ("Oboe I",        "Rach 2 Oboe I.pdf",        16, "music binding"),
    ("Clarinet I",    "Rach 2 Clarinet I.pdf",    13, "music binding"),
    ("Viola",         "Rach 2 Viola_l.pdf",       2,  "blank verso"),
    ("Flute II",      "Rach 2 Flute II.pdf",      2,  "blank verso"),
    ("Oboe II",       "Rach 2 Oboe II.pdf",       2,  "blank verso"),
    ("Bass Clarinet", "Rach 2 Bass Clarinet.pdf", 20, "music binding"),
]

SCALE = 1.0


def _slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def png_name(idx, part, page):
    return f"{idx:02d}_{_slug(part)}_p{page}.png"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for i, (part, file, page, note) in enumerate(MANIFEST, 1):
        pil = pdfium.PdfDocument(os.path.join(RAW_DIR, file))[page - 1] \
            .render(scale=SCALE).to_pil().convert("RGB")
        out = os.path.join(OUT_DIR, png_name(i, part, page))
        pil.save(out)  # lossless PNG
        print(f"  P{i:02d} {part} p{page} [{note}] -> {os.path.basename(out)}")
    print(f"\nWrote {len(MANIFEST)} raw PNGs to:\n  {OUT_DIR}")


if __name__ == "__main__":
    main()
