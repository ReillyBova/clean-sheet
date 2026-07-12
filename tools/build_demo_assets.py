#!/usr/bin/env python3
"""Build the Clean Sheet webapp demo assets from real pipeline runs.

Renders several sample pages through the exact production pipeline (booklet-aware),
and for each one exports the raw capture and the final cleaned page as web-sized
JPEGs plus the Coons source->flat UV grid (JSON) that drives the WebGL un-warp
animation. The webapp picks one example at random on each load, so the showcase
cycles through a variety of pages -- including hard binding-fold booklet captures.

Re-run to regenerate: `python tools/build_demo_assets.py`.
Sources that are missing on disk are skipped with a warning (the in-repo
`samples/` page always builds).
"""
import json, os, sys
from pathlib import Path
import numpy as np
import cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import flat_scan as fs
import flatscan_dewarp as fd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "assets")
STAGES = os.path.join(OUT, "stages")
GRID_W, GRID_H = 40, 54       # mesh resolution for the reprojection morph
DPI = 220                     # web render resolution for the flat plate

_RACH = os.path.expanduser("~/Downloads/Rachmaninoff Symphony No. 2 — Originals")

# id, label, source pdf, 1-based page, booklet, (width_in, height_in)
EXAMPLES = [
    ("russlan", "Glinka — Ruslan Overture",
     os.path.join(ROOT, "samples", "russlan_raw_phone_scan.pdf"), 1, False, (8.9375, 12)),
    ("horn4", "Rachmaninoff 2 — Horn IV (serpentine curl)",
     os.path.join(_RACH, "Rach 2 Horn IV.pdf"), 6, True, (8.9375, 12)),
    ("horn1", "Rachmaninoff 2 — Horn I (binding-tail curl)",
     os.path.join(_RACH, "Rach 2 Horn I.pdf"), 4, True, (8.9375, 12)),
    ("flute2", "Rachmaninoff 2 — Flute II (sloped ends)",
     os.path.join(_RACH, "Rach 2 Flute II.pdf"), 2, True, (8.9375, 12)),
    ("clar1", "Rachmaninoff 2 — Clarinet I",
     os.path.join(_RACH, "Rach 2 Clarinet I.pdf"), 2, True, (8.9375, 12)),
    ("violin2", "Rachmaninoff 2 — Violin II",
     os.path.join(_RACH, "Rach 2 Violin II_l.pdf"), 5, True, (8.9375, 12)),
]


def web_write(path, img, max_h=920, q=84):
    """Downscale to web height and save as JPEG. Returns bytes written."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    if h > max_h:
        s = max_h / h
        img = cv2.resize(img, (round(w * s), max_h), interpolation=cv2.INTER_AREA)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return os.path.getsize(path)


def build_example(eid, label, pdf, page1, booklet, dims):
    """Run the pipeline for one page; write input.jpg + ink.jpg; return metadata."""
    w_in, h_in = dims
    out_w, out_h = round(w_in * DPI), round(h_in * DPI)
    inp = fs.InputPages(Path(pdf), render_scale=1.0)
    img_bgr = inp.render_page_bgr(page1 - 1)
    H, W = img_bgr.shape[:2]

    # exact production pipeline (mirrors process_page), booklet-aware
    mask, _seg = fs.segment_page(img_bgr)
    seam = fs.detect_seam_for_page(img_bgr, mask) if booklet else None
    if seam is not None:
        mask = fs.clip_mask_at_crease(mask, seam)
    rect, edges = fs.rectify_boundary_coons(img_bgr, mask, out_w, out_h, smooth=0.045)
    cleaned, _ink, _thr = fs.clean_ink(rect, mode="soft-gray")
    # rect = the dewarped page BEFORE ink cleaning (still lit, staves still bent).
    # The webapp irons THIS flat (staves + page), then develops the clean ink last
    # -- a reordered, best-looking story (not the engine's true order). `cleaned`
    # (soft-gray, pre-straighten) is only used to trace the staff geometry, which
    # has stronger contrast than the lit rect.
    gwv = cleaned if cleaned.ndim == 2 else cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
    hh, ww = gwv.shape

    # The webapp's "iron flat" is exactly the guided staff-straighten warp, so the
    # demo's final ink must be that SAME warp applied to the cleaned page -- not
    # the full production straighten (which also deskews, aligns margins and
    # centres). Matching them makes the iron->clean crossfade a pure tone change
    # (no ghosting), which the extra production steps would otherwise introduce.
    sdisp, _sinfo = fd._staff_guided_displacement(gwv)
    if sdisp is not None:
        smx, smy = sdisp
        final_clean = cv2.remap(cleaned, smx.astype(np.float32), smy.astype(np.float32),
                                cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    else:
        final_clean, _ = fd.straighten_staves(cleaned)

    web_write(os.path.join(STAGES, eid, "input.jpg"), img_bgr)
    web_write(os.path.join(STAGES, eid, "rect.jpg"), rect)
    web_write(os.path.join(STAGES, eid, "ink.jpg"), final_clean)

    # reprojection morph grid: source (photo) positions per mesh vertex, 0..1.
    # Uses rectify's (inset) edges so the page mesh matches rect.jpg/ink.jpg
    # exactly -- if the mesh traced a wider boundary than the rasters, the
    # lift->rect crossfade would show a ~0.5% content shift (a blur). The "find
    # the page" outline is nudged back out to the true paper edge in the webapp
    # (cinematic.js), where it is display-only and needn't match the rasters.
    mx, my = fs.coons_maps(edges, GRID_W, GRID_H)
    src = np.stack([mx / (W - 1), my / (H - 1)], axis=-1)

    # staff geometry (for the "find the staves" overlay and its ironing) + the
    # straighten UV-morph grid (irons the rect texture flat as a real warp). Reuse
    # the guided displacement computed above so the morph exactly matches ink.jpg.
    straighten = None
    staves = []
    if sdisp is not None:
        _smx, smy = sdisp
        gi = np.linspace(0, ww - 1, GRID_W).round().astype(int)
        gj = np.linspace(0, hh - 1, GRID_H).round().astype(int)
        su = np.tile((gi / (ww - 1))[None, :], (GRID_H, 1))
        sv = smy[np.ix_(gj, gi)] / (hh - 1)
        ssrc = np.stack([su, sv], axis=-1)
        straighten = {"w": GRID_W, "h": GRID_H, "src": ssrc.reshape(-1, 2).round(5).tolist()}

    xcs, sys_lines, _space = fd._trace_staff_lines(gwv)
    xcs = np.asarray(xcs, float)
    for (_yt, _yb, L) in sys_lines:
        good = ~np.isnan(L).any(axis=0)
        if good.sum() < 8:
            continue
        gx = xcs[good]
        idx = np.linspace(0, len(gx) - 1, min(48, len(gx))).round().astype(int)
        xs = [round(float(gx[i] / (ww - 1)), 4) for i in idx]
        lines, flat = [], []
        for k in range(L.shape[0]):
            yk = L[k][good]
            lines.append([round(float(yk[i] / (hh - 1)), 4) for i in idx])
            flat.append(round(float(np.nanmean(L[k]) / (hh - 1)), 4))
        staves.append({"xs": xs, "lines": lines, "flat": flat})

    return {
        "id": eid,
        "label": label,
        "photo": {"w": int(W), "h": int(H), "image": f"stages/{eid}/input.jpg"},
        "rect": f"stages/{eid}/rect.jpg",
        "ink": f"stages/{eid}/ink.jpg",
        "grid": {"w": GRID_W, "h": GRID_H, "src": src.reshape(-1, 2).round(5).tolist()},
        "straighten": straighten,
        "staves": staves,
        "output_aspect": round(w_in / h_in, 5),
    }


def main():
    examples = []
    for eid, label, pdf, page1, booklet, dims in EXAMPLES:
        if not os.path.exists(pdf):
            print(f"  skip {eid}: source not found ({pdf})", flush=True)
            continue
        try:
            examples.append(build_example(eid, label, pdf, page1, booklet, dims))
            print(f"  built {eid}: {label}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED {eid}: {type(e).__name__}: {e}", flush=True)
    if not examples:
        raise SystemExit("no examples built")
    with open(os.path.join(OUT, "demo.json"), "w") as f:
        json.dump({"examples": examples}, f)
    print(f"wrote demo.json with {len(examples)} example(s) to {OUT}")


if __name__ == "__main__":
    main()
