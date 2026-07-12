#!/usr/bin/env python3
"""
Phone-scan cleanup for loose orchestra/music parts.

Pipeline:
  PDF page/image -> page segmentation mask -> boundary Coons UV rectification ->
  illumination normalization -> clean ink rendering -> PDF at requested physical size/DPI.

Default output mode is "soft-gray": white paper with anti-aliased grayscale ink.
This is usually better than harsh binary for music parts because it preserves thin
slurs/articulations and avoids turning paper texture into speckle.

Examples:
  python flat_scan.py input.pdf output.pdf --page-size 9x12 --dpi 400
  python flat_scan.py input.pdf output.pdf --width-in 9 --height-in 12 --dpi 400 --debug --debug-pages 1
  python flat_scan.py input.pdf output.pdf --page-size 8.5x11 --mode binary
  python flat_scan.py input.pdf output.pdf --page-size 9x12 --starts-on-even
  python flat_scan.py input.pdf output.pdf --page-size 9x12 --resume --work-dir output_work

Dependencies:
  pip install opencv-python numpy pypdfium2 img2pdf
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import gc
import fnmatch
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pypdfium2 as pdfium
import img2pdf

import flatscan_dewarp
import flatscan_seam


# ----------------------------- utilities -----------------------------

@dataclass(frozen=True)
class PageSize:
    width_in: float
    height_in: float

    def pixels_for(self, dpi: int) -> tuple[int, int]:
        return int(round(self.width_in * dpi)), int(round(self.height_in * dpi))

    def points(self) -> tuple[float, float]:
        return self.width_in * 72.0, self.height_in * 72.0


def parse_page_size(s: str) -> PageSize:
    val = s.strip().lower().replace(" ", "")
    presets = {
        "letter": PageSize(8.5, 11.0),
        "usletter": PageSize(8.5, 11.0),
        "legal": PageSize(8.5, 14.0),
        "concert": PageSize(9.0, 12.0),
        "9x12": PageSize(9.0, 12.0),
        "a4": PageSize(8.2677165, 11.6929134),
    }
    if val in presets:
        return presets[val]
    m = re.match(r"^([0-9]*\.?[0-9]+)x([0-9]*\.?[0-9]+)(in|inch|inches)?$", val)
    if not m:
        raise argparse.ArgumentTypeError(
            "page size must be like '9x12', '8.5x11', 'letter', 'legal', or 'a4'"
        )
    return PageSize(float(m.group(1)), float(m.group(2)))


def parse_pages(spec: str | None, total: int) -> list[int]:
    """Parse 1-based page/range spec into zero-based indices."""
    if not spec or spec.strip().lower() in {"all", "*"}:
        return list(range(total))
    out: set[int] = set()
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            start = int(a) if a else 1
            end = int(b) if b else total
            if start > end:
                start, end = end, start
            for p in range(start, end + 1):
                if 1 <= p <= total:
                    out.add(p - 1)
        else:
            p = int(part)
            if 1 <= p <= total:
                out.add(p - 1)
    return sorted(out)


def ensure_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def imwrite(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), img)
    if not ok:
        raise RuntimeError(f"Could not write image: {path}")


def save_preview(src: Path, dst: Path, max_w: int = 1400) -> None:
    img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if img is None:
        return
    h, w = img.shape[:2]
    scale = min(1.0, max_w / max(w, 1))
    if scale < 1.0:
        img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    params = [int(cv2.IMWRITE_JPEG_QUALITY), 92] if dst.suffix.lower() in {'.jpg', '.jpeg'} else []
    cv2.imwrite(str(dst), img, params)


# ----------------------------- input rendering -----------------------------

class InputPages:
    def __init__(self, input_path: Path, render_scale: float = 1.0):
        self.input_path = input_path
        self.render_scale = render_scale
        self.suffix = input_path.suffix.lower()
        self.doc = None
        if self.suffix == '.pdf':
            self.doc = pdfium.PdfDocument(str(input_path))
            self.count = len(self.doc)
        elif self.suffix in {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.webp'}:
            self.count = 1
        else:
            raise ValueError(f"Unsupported input type: {input_path.suffix}")

    def _native_image_scale(self, page) -> float:
        """Render scale (pixels-per-point) that reproduces the native resolution
        of the largest raster image embedded on the page.

        This is fully agnostic to any expected page size. It simply measures the
        actual pixel dimensions of the embedded scan and compares them to the
        declared page box:

          * A well-formed export declares a page box that matches its embedded
            image, so this returns ~1.0 and rendering is unchanged.
          * A misconfigured export declares a small page box (e.g. US-Letter at
            72 dpi) around a full-resolution photo. Rendering at scale=1.0 would
            throw away most pixels; this returns the larger scale needed to
            recover the real detail.

        Returns 0.0 when the page has no measurable raster image (e.g. pure
        vector content), so the caller can fall back to the base scale.
        """
        page_w_pt, page_h_pt = page.get_size()
        if page_w_pt <= 0 or page_h_pt <= 0:
            return 0.0
        best = 0.0
        try:
            objects = page.get_objects()
        except Exception:
            return 0.0
        for obj in objects:
            if getattr(obj, "type", None) != 3:  # 3 == image object
                continue
            try:
                img_w, img_h = obj.get_px_size()
            except Exception:
                continue
            best = max(best, img_w / page_w_pt, img_h / page_h_pt)
        return best

    def render_page_bgr(self, index: int) -> np.ndarray:
        if self.doc is not None:
            page = self.doc[index]
            native = self._native_image_scale(page)
            base = native if native > 0 else 1.0
            scale = base * self.render_scale
            if native > 1.05:
                print(
                    f"    Note: page box under-samples embedded scan by {native:.2f}x; "
                    f"rendering at native resolution (scale={scale:.2f})",
                    file=sys.stderr,
                )
            bitmap = page.render(scale=scale, rotation=0)
            pil = bitmap.to_pil().convert('RGB')
            rgb = np.array(pil)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        img = cv2.imread(str(self.input_path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Could not read image: {self.input_path}")
        if index != 0:
            raise IndexError(index)
        if self.render_scale != 1.0:
            img = cv2.resize(img, None, fx=self.render_scale, fy=self.render_scale, interpolation=cv2.INTER_CUBIC)
        return ensure_bgr(img)


# ----------------------------- segmentation -----------------------------

def fill_holes(binary: np.ndarray) -> np.ndarray:
    """Fill background pockets fully enclosed by foreground.

    Background components that reach the image border are "outside" the page;
    any background component that does *not* touch the border is an enclosed
    hole and gets filled. This is deliberately border-agnostic rather than a
    single flood from a fixed corner: if the foreground happens to touch that
    corner (e.g. a bright table bridge reaching the frame edge), a corner flood
    would fail and fill the *entire* frame. Labelling the background and keeping
    only border-connected components as "outside" can never balloon the mask to
    the whole image.
    """
    bg = cv2.bitwise_not(binary)
    num, labels = cv2.connectedComponents(bg, connectivity=4)
    if num <= 1:
        return binary
    border = np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
    outside = np.unique(border)
    outside_mask = np.isin(labels, outside)
    out = binary.copy()
    out[(bg > 0) & ~outside_mask] = 255
    return out


def largest_component(binary: np.ndarray, min_area_frac: float = 0.05) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num <= 1:
        return np.zeros_like(binary)
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    area = stats[idx, cv2.CC_STAT_AREA]
    if area < min_area_frac * binary.size:
        return np.zeros_like(binary)
    out = np.zeros_like(binary)
    out[labels == idx] = 255
    return out


def repair_boundary_defects(mask: np.ndarray, min_dev_frac: float = 0.006,
                         max_width_frac: float = 0.15) -> np.ndarray:
    """Repair narrow, localized segmentation defects on the page boundary.

    A photographed page is essentially a convex quad whose edges are straight or
    bow gently (perspective/barrel). Segmentation defects are *sharp, localized*
    departures from that smooth edge, and they come in both directions:
      * inward notches — a finger, cast shadow, or torn corner makes the mask
        cave in toward the page centre;
      * outward bulges — glare, an adjacent sheet, or a bright table edge gets
        absorbed into the mask so the boundary balloons out past the paper.
    Either way the Coons patch then propagates the defect into the page interior
    (as waviness/pinch) or samples background into the result.

    We compare the boundary to a heavily-smoothed copy of itself. Smoothing is a
    low-pass filter: it keeps the gentle, low-frequency bow of a real edge but
    erases sharp, high-frequency defects. Wherever the actual boundary departs
    from that smooth reference by more than a small fraction of the page — in
    either direction — over a short, localized run, we snap it back to the
    reference. The reference follows the real edge on both shoulders of the
    defect, so the repaired boundary stays on paper: inward notches are filled
    out to the true edge and outward bulges are trimmed back to it. Broad edge
    bow is left untouched (the smooth copy matches it, so nothing is flagged).
    No absolute page dimensions are assumed — thresholds scale with the contour.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return mask
    c = max(contours, key=cv2.contourArea)[:, 0, :].astype(np.float32)
    if len(c) < 16:
        return mask

    m = 1500
    poly = resample_polyline_by_arclength(np.vstack([c, c[:1]]), m + 1)[:m]

    # Roll so the seam (index 0) sits at a real corner. Corners are protected
    # from repair, so no defect run can ever straddle the seam — which keeps the
    # circular run-growing logic below simple and correct.
    corners = find_corners_from_contour(poly)
    tl_i = int(np.argmin(np.sum((poly - corners['tl']) ** 2, axis=1)))
    poly = np.roll(poly, -tl_i, axis=0)

    win = max(9, int(m * 0.05) | 1)
    pad = win // 2
    kernel = np.ones(win, np.float32) / win

    diag = float(np.hypot(*mask.shape[:2]))
    min_dev = min_dev_frac * diag
    max_run = int(max_width_frac * m)

    # The page's four real corners are high-frequency features that the
    # smoothing rounds off, so they read as large deviations. Exclude a
    # neighbourhood around each corner: genuine notches/bulges live along the
    # edges, and snapping a true corner would blunt it. Corners don't move under
    # our edge-only repairs, so this is computed once.
    guard = pad + max(9, m // 40)
    protected = np.zeros(m, bool)
    for cp in corners.values():
        ci = int(np.argmin(np.sum((poly - cp) ** 2, axis=1)))
        protected[[(ci + off) % m for off in range(-guard, guard + 1)]] = True

    def smoothed(p: np.ndarray) -> np.ndarray:
        padded = np.vstack([p[-pad:], p, p[:pad]])
        return np.column_stack([np.convolve(padded[:, 0], kernel, mode='valid'),
                                np.convolve(padded[:, 1], kernel, mode='valid')])

    # Iterate: a low-pass of the boundary partially *follows* a defect, so one
    # pass only shaves its tip. Bridging that tip and recomputing the reference
    # exposes more of the defect; a few passes converge to full removal. Each
    # accepted run is bridged by a straight segment between its two clean
    # shoulders (points just outside the run); those shoulders lie on the real
    # edge, so the bridge stays on paper — inward notches are filled out and
    # outward bulges trimmed in, both without ever sampling background.
    changed = False
    for _ in range(8):
        dev = np.linalg.norm(poly - smoothed(poly), axis=1)
        defect = (dev > min_dev) & ~protected
        if not defect.any() or defect.all():
            break
        idx = np.where(defect)[0]
        starts = idx[np.where(np.diff(np.r_[idx[-1] - m, idx]) != 1)[0]]
        pass_changed = False
        new_poly = poly.copy()
        for s0 in starts:
            run = []
            i = s0
            while defect[i % m]:
                run.append(i % m)
                i += 1
                if len(run) > m:
                    break
            if len(run) > max_run:
                continue
            # Grow the run outward past the defect's shallow flanks (where the
            # deviation dips below threshold but the boundary is still displaced)
            # so the bridge anchors on stable edge. Never cross into a corner.
            grow = max(win, len(run))
            lo, hi = run[0], run[-1]
            for _g in range(grow):
                if not protected[(lo - 1) % m]:
                    lo -= 1
                if not protected[(hi + 1) % m]:
                    hi += 1
            span = [(k) % m for k in range(lo, hi + 1)]
            if len(span) > max_run + 2 * grow:
                continue
            a = poly[(lo - 1) % m]
            b = poly[(hi + 1) % m]
            for step, j in enumerate(span, start=1):
                t = step / (len(span) + 1)
                new_poly[j] = (1.0 - t) * a + t * b
            pass_changed = True
        if not pass_changed:
            break
        poly = new_poly
        changed = True
    if not changed:
        return mask

    out = np.zeros_like(mask)
    cv2.fillPoly(out, [np.round(poly).astype(np.int32).reshape(-1, 1, 2)], 255)
    return fill_holes(out)


def segment_page(img_bgr: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Segment the paper as the dominant bright connected component."""
    h, w = img_bgr.shape[:2]
    blur_bgr = cv2.medianBlur(img_bgr, 5)
    lab = cv2.cvtColor(blur_bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]

    # Illumination estimate: use a large kernel (~1/6 of the short side, similar
    # to clean_ink's) so this captures the *lighting* field — including soft
    # edge/curl shadows — while preserving the paper-vs-background reflectance
    # step. A smaller kernel tracks that reflectance step itself, which is why an
    # earlier min/12 estimate forced a brittle absolute-brightness fallback that
    # dropped shadowed paper and carved notches into the mask.
    k = max(101, (min(h, w) // 6) | 1)
    bg = cv2.GaussianBlur(L, (k, k), 0)
    norm = cv2.divide(L, bg, scale=180)
    norm = cv2.normalize(norm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Segment on the illumination-normalized image. Paper (high reflectance)
    # separates cleanly from the darker surface regardless of its colour (wood,
    # mat, etc.) and shadowed paper stays on the paper side of the threshold, so
    # curl/edge shadows no longer read as background.
    _, seed = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    close_k = max(17, (min(h, w) // 160) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    # Remove small bright specks, then isolate the page as the largest bright
    # component *before* any heavy closing. Doing this first is important: a
    # strong close can bridge the page across small gaps into unrelated bright
    # regions that touch the image border (photo/lens edges, an adjacent sheet),
    # fusing them into one blob and destroying the page's corner geometry.
    #
    # The opening kernel is deliberately larger than the closing kernel. The
    # illumination-normalized image inflates the local contrast of dark surfaces
    # (wood grain, mat texture) into thin bright filaments; a filament that
    # bridges the page to a border speck fuses them into one component that
    # wraps around the background, and the mask then balloons to the whole frame.
    # A wider opening severs these thin bridges while leaving the solid page body
    # (and its corners) intact — the page is far thicker than any grain filament.
    open_k = max(31, (min(h, w) // 80) | 1)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
    seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, open_kernel, iterations=1)
    mask = largest_component(seed)

    # Now that the page is isolated, consolidate its interior safely.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = fill_holes(mask)

    smooth_k = max(11, (min(h, w) // 220) | 1)
    smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (smooth_k, smooth_k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, smooth, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, smooth, iterations=1)
    mask = fill_holes(mask)
    # Safety net: smoothing can occasionally reconnect a stray region; keep only
    # the dominant page component for a clean single-contour boundary.
    mask = largest_component(mask)
    # Repair localized concave defects (finger/shadow notches) so they are not
    # propagated into the page interior by the Coons patch.
    mask = repair_boundary_defects(mask)

    debug = {
        "01_lightness_raw.png": L,
        "02_lightness_normalized.png": norm,
        "03_segmentation_seed.png": seed,
    }
    return mask, debug


def detect_crease(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    min_score: float = 6.0,
    touch_frac: float = 0.02,
) -> dict | None:
    """Locate the binding-side clip line on a booklet half-spread scan.

    In "booklet" captures each photo shows one page plus the fold into the spine,
    and a sliver of the facing page frequently bleeds in past the fold. That bleed
    corrupts the page boundary (the Coons patch reaches into the neighbour and the
    binding-side edge buckles). Clipping the mask along this page's own outer edge
    excludes the neighbour and hands the patch a clean binding edge.

    Which side is the binding is decided by geometry, not guesswork: a booklet
    half-spread always runs off into the spine on one L/R side, so the paper mask
    reaches the image border there with no table margin. A full flat sheet (e.g. a
    title page) is instead "floating" -- table background surrounds it on all four
    sides -- and is never clipped, which protects full sheets whose content
    straddles the fold.

    Given the binding side, we clip along the outer edge of *this page's music*
    rather than hunting for the fold shadow. This is far more robust: on a page
    that simply runs off the frame (no facing page in view) the old fold-valley
    filter would latch onto this page's own content -- the stacked closing
    barlines, or the shadow where dense notes end -- and amputate the binding-side
    ends of systems. Instead we isolate this page's music with a black-tophat
    (which suppresses the broad fold/spine shadow that would bridge the two pages)
    and keep the wide, page-centred ink components; a bled neighbour lives across
    the blank gutter as its own component and is dropped. The crease is a smooth
    low-order curve fit just *outside* that music edge, so this page's notes,
    barlines and clefs are always preserved while the fold shadow and any facing
    page are removed.

    Returns ``None`` for a floating sheet or when no binding-side music is found.
    Otherwise returns ``{side, curve, score}`` where ``side`` is
    ``"left"``/``"right"``, ``curve`` is a per-row array of crease x-positions and
    ``score`` is the width of the strip being clipped (fold + any neighbour).
    """
    h, w = img_bgr.shape[:2]
    m = mask > 0
    if not m.any():
        return None

    ys, xs = np.where(m)
    x0, x1 = int(xs.min()), int(xs.max())
    if x1 - x0 <= 0:
        return None

    # Binding side = the L/R side where the paper runs off the frame (no table
    # margin). Floating pages (margin on both sides) are full sheets -> no clip.
    touch_px = max(2, int(round(touch_frac * w)))
    touch_l = x0 <= touch_px
    touch_r = x1 >= (w - 1 - touch_px)
    if not touch_l and not touch_r:
        return None

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    candidates = []
    if touch_r:
        c = _binding_edge_crease(gray, m, "right")
        if c is not None:
            candidates.append(c)
    if touch_l:
        c = _binding_edge_crease(gray, m, "left")
        if c is not None:
            candidates.append(c)
    if not candidates:
        return None
    # When both sides run off the frame, the binding is the side whose music ends
    # furthest from that frame edge (a real fold/neighbour strip to remove). A
    # clip that would only shave a hair off the frame edge is not worth doing.
    best = max(candidates, key=lambda c: c["score"])
    if best["score"] < 0.012 * w:
        return None
    return best


def _binding_edge_crease(gray: np.ndarray, page: np.ndarray, side: str) -> dict | None:
    """Smooth crease hugging the outer edge of this page's music on ``side``."""
    h, w = gray.shape
    # Black-tophat isolates thin dark strokes (music) while suppressing the broad
    # fold/spine shadow that would otherwise bridge this page to its neighbour.
    k = max(9, (min(h, w) // 80) | 1)
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    ink = ((bh > 25) & page).astype(np.uint8)
    # Close note-to-note and staff-dash gaps, but with a kernel far narrower than
    # the blank inter-page gutter so this page and any bled neighbour stay apart.
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(ink, 8)
    cen = w // 2
    keep = np.zeros((h, w), dtype=bool)
    for i in range(1, n):
        wide = stats[i, cv2.CC_STAT_WIDTH] >= 0.15 * w
        big = stats[i, cv2.CC_STAT_AREA] >= 0.002 * h * w
        centred = abs(cents[i][0] - cen) < 0.30 * w
        if wide and big and centred:
            keep |= lab == i
    e = np.full(h, np.nan)
    for y in range(h):
        col = np.where(keep[y])[0]
        if col.size:
            e[y] = float(col[-1] if side == "right" else col[0])
    have = ~np.isnan(e)
    if int(have.sum()) < 20:
        return None

    # Robust smooth boundary: per-band outer percentile of the music edge, drop
    # outlier bands (stray marks), then a low-order fit that tracks the gentle
    # fold bend without buckling. A near-vertical crease is fine here.
    yy = np.arange(h)
    q = 92 if side == "right" else 8
    nb = 24
    bounds = np.linspace(0, h, nb + 1).astype(int)
    by, bx = [], []
    for i in range(nb):
        s = e[bounds[i]:bounds[i + 1]]
        s = s[~np.isnan(s)]
        if s.size >= 8:
            by.append(0.5 * (bounds[i] + bounds[i + 1]))
            bx.append(float(np.percentile(s, q)))
    if len(by) < 5:
        return None
    by = np.asarray(by)
    bx = np.asarray(bx)
    lin = np.polyfit(by, bx, 1)
    resid = bx - np.polyval(lin, by)
    mad = float(np.median(np.abs(resid - np.median(resid)))) + 1e-6
    good = np.abs(resid - np.median(resid)) < 3.0 * mad
    if int(good.sum()) >= 6:
        by, bx = by[good], bx[good]
    deg = 2 if len(by) >= 8 else 1
    base = np.polyval(np.polyfit(by, bx, deg), yy)
    margin = int(0.006 * w)
    sgn = 1.0 if side == "right" else -1.0
    curve = base + sgn * margin

    # Never clip this page's own music. The low-order fit tracks the gentle fold
    # bend but can *undershoot* an outlier system whose binding-side end juts past
    # the smooth trend -- e.g. a lone closing barline on the bottom system -- which
    # would amputate real content. ``e`` is the per-row outer edge of THIS page's
    # kept components only (a bled neighbour lives across the blank gutter as a
    # separate component that was dropped above), so clamping the crease to sit
    # just outside it preserves every note/barline while never reaching into the
    # neighbour. Gaps between systems are interpolated and the edge is widened a
    # touch vertically (running max/min) so the protection is smooth, not a notch.
    idx = np.where(have)[0]
    if idx.size:
        env = np.interp(yy, idx, e[idx]).astype(np.float32).reshape(-1, 1)
        ksz = max(3, int(0.01 * h) | 1)
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ksz))
        if side == "right":
            env = cv2.dilate(env, kern)[:, 0]
            curve = np.maximum(curve, env + margin)
        else:
            env = cv2.erode(env, kern)[:, 0]
            curve = np.minimum(curve, env - margin)

    curve = np.clip(curve, 0.0, w - 1.0)
    score = float(w - np.median(curve)) if side == "right" else float(np.median(curve))
    return {"side": side, "curve": curve, "score": score}


def clip_mask_at_crease(mask: np.ndarray, crease: dict) -> np.ndarray:
    """Drop everything past the crease on its side, so the fold becomes the edge.

    This removes the bled-in facing page and hands the Coons patch a clean, smooth
    binding edge to map onto the straight output border. Content on the page's own
    side of the fold is untouched.
    """
    out = mask.copy()
    h = mask.shape[0]
    side = crease["side"]
    xs = np.round(crease["curve"]).astype(np.int32)
    for y in range(h):
        cx = int(xs[y])
        if side == "right":
            out[y, cx:] = 0
        else:
            out[y, : cx + 1] = 0
    # A clean cut can leave a disconnected neighbour speck on the far side; keep
    # only the dominant page component so the boundary is a single clean contour.
    return largest_component(out)


def crease_overlay(img_bgr: np.ndarray, crease: dict) -> np.ndarray:
    """Draw the detected crease curve for debug inspection."""
    vis = img_bgr.copy()
    h = vis.shape[0]
    xs = np.round(crease["curve"]).astype(np.int32)
    for y in range(0, h, 2):
        x = int(xs[y])
        if 0 <= x < vis.shape[1]:
            cv2.circle(vis, (x, y), 2, (255, 0, 0), -1)
    return vis


def mask_overlay(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    tint = np.zeros_like(img_bgr)
    tint[:, :, 1] = 255
    alpha = (mask.astype(np.float32) / 255.0) * 0.35
    return (img_bgr.astype(np.float32) * (1 - alpha[..., None]) + tint.astype(np.float32) * alpha[..., None]).astype(np.uint8)


# ----------------------------- boundary Coons rectification -----------------------------

def largest_external_contour(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError("No page contour found. Is the page clearly visible against the background?")
    c = max(contours, key=cv2.contourArea)[:, 0, :].astype(np.float32)
    if cv2.contourArea(c.astype(np.float32)) < 0.05 * mask.size:
        raise RuntimeError("Detected page contour is too small; segmentation probably failed.")
    return c


def find_corners_from_contour(c: np.ndarray) -> dict[str, np.ndarray]:
    s = c[:, 0] + c[:, 1]
    d = c[:, 0] - c[:, 1]
    return {
        'tl': c[np.argmin(s)],
        'br': c[np.argmax(s)],
        'tr': c[np.argmax(d)],
        'bl': c[np.argmin(d)],
    }


def contour_index_nearest(c: np.ndarray, p: np.ndarray) -> int:
    return int(np.argmin(np.sum((c - p) ** 2, axis=1)))


def chain_between(c: np.ndarray, i0: int, i1: int) -> np.ndarray:
    if i0 <= i1:
        return c[i0:i1 + 1]
    return np.vstack([c[i0:], c[:i1 + 1]])


def split_into_edges(c: np.ndarray, corners: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    idx = {k: contour_index_nearest(c, v) for k, v in corners.items()}

    def choose(a: str, b: str, criterion):
        c1 = chain_between(c, idx[a], idx[b])
        c2 = chain_between(c, idx[b], idx[a])
        return c1 if criterion(c1) < criterion(c2) else c2

    return {
        'top': choose('tl', 'tr', lambda pts: np.mean(pts[:, 1])),
        'bottom': choose('bl', 'br', lambda pts: -np.mean(pts[:, 1])),
        'left': choose('tl', 'bl', lambda pts: np.mean(pts[:, 0])),
        'right': choose('tr', 'br', lambda pts: -np.mean(pts[:, 0])),
    }


def resample_polyline_by_arclength(points: np.ndarray, n: int) -> np.ndarray:
    pts = np.asarray(points, np.float32)
    d = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] <= 1e-6:
        return np.repeat(pts[:1], n, axis=0)
    targets = np.linspace(0, s[-1], n)
    x = np.interp(targets, s, pts[:, 0])
    y = np.interp(targets, s, pts[:, 1])
    return np.column_stack([x, y]).astype(np.float32)


def smooth_curve(points: np.ndarray, n: int, window_frac: float) -> np.ndarray:
    q = resample_polyline_by_arclength(points, n)
    win = max(5, int(n * window_frac) | 1)
    pad = win // 2
    padded = np.pad(q, ((pad, pad), (0, 0)), mode='edge')
    kernel = np.ones(win, np.float32) / win
    xs = np.convolve(padded[:, 0], kernel, mode='valid')
    ys = np.convolve(padded[:, 1], kernel, mode='valid')
    return np.column_stack([xs, ys]).astype(np.float32)





def orient_edges(raw_edges: dict[str, np.ndarray], corners: dict[str, np.ndarray], n: int = 1200, smooth: float = 0.045) -> dict[str, np.ndarray]:
    specs = {
        'top': ('tl', 'tr'),
        'bottom': ('bl', 'br'),
        'left': ('tl', 'bl'),
        'right': ('tr', 'br'),
    }
    out = {}
    for name, (start, _) in specs.items():
        curve = smooth_curve(raw_edges[name], n, smooth)
        if np.linalg.norm(curve[0] - corners[start]) > np.linalg.norm(curve[-1] - corners[start]):
            curve = curve[::-1].copy()
        out[name] = curve
    return out


def interp_curve_array(curve: np.ndarray, t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0, 1).astype(np.float32)
    pos = t * (len(curve) - 1)
    i = np.floor(pos).astype(np.int32)
    j = np.minimum(i + 1, len(curve) - 1)
    a = (pos - i).astype(np.float32)[..., None]
    return (1 - a) * curve[i] + a * curve[j]


def coons_maps(edges: dict[str, np.ndarray], out_w: int, out_h: int) -> tuple[np.ndarray, np.ndarray]:
    """Full Coons map builder, kept for small/debug uses."""
    u = np.linspace(0, 1, out_w, dtype=np.float32)
    v = np.linspace(0, 1, out_h, dtype=np.float32)
    U, V = np.meshgrid(u, v)

    T = interp_curve_array(edges['top'], U)
    B = interp_curve_array(edges['bottom'], U)
    L = interp_curve_array(edges['left'], V)
    R = interp_curve_array(edges['right'], V)

    TL = edges['top'][0]
    TR = edges['top'][-1]
    BL = edges['bottom'][0]
    BR = edges['bottom'][-1]
    bilinear = (
        ((1 - U) * (1 - V))[..., None] * TL +
        (U * (1 - V))[..., None] * TR +
        (((1 - U) * V))[..., None] * BL +
        (U * V)[..., None] * BR
    )
    P = ((1 - V)[..., None] * T + V[..., None] * B + (1 - U)[..., None] * L + U[..., None] * R - bilinear)
    return P[..., 0].astype(np.float32), P[..., 1].astype(np.float32)


def remap_coons_chunked(img_bgr: np.ndarray, edges: dict[str, np.ndarray], out_w: int, out_h: int, chunk_rows: int = 256) -> np.ndarray:
    """Remap using a Coons patch without allocating several full-size 3D arrays."""
    u = np.linspace(0, 1, out_w, dtype=np.float32)
    U_row = u[None, :]
    T_row = interp_curve_array(edges['top'], U_row)[0]      # (W, 2)
    B_row = interp_curve_array(edges['bottom'], U_row)[0]   # (W, 2)

    TL = edges['top'][0].astype(np.float32)
    TR = edges['top'][-1].astype(np.float32)
    BL = edges['bottom'][0].astype(np.float32)
    BR = edges['bottom'][-1].astype(np.float32)

    out = np.empty((out_h, out_w, 3), dtype=np.uint8)
    one_minus_u = (1.0 - U_row).astype(np.float32)

    for y0 in range(0, out_h, chunk_rows):
        y1 = min(out_h, y0 + chunk_rows)
        v = np.linspace(y0 / max(out_h - 1, 1), (y1 - 1) / max(out_h - 1, 1), y1 - y0, dtype=np.float32)
        V = v[:, None]
        one_minus_v = 1.0 - V

        L = interp_curve_array(edges['left'], V)[:, 0, :]   # (Hc, 2)
        R = interp_curve_array(edges['right'], V)[:, 0, :]  # (Hc, 2)

        # Terms are maintained as separate x/y maps to avoid huge (..., 2) temporaries.
        bil_x = (one_minus_u * one_minus_v) * TL[0] + (U_row * one_minus_v) * TR[0] + (one_minus_u * V) * BL[0] + (U_row * V) * BR[0]
        bil_y = (one_minus_u * one_minus_v) * TL[1] + (U_row * one_minus_v) * TR[1] + (one_minus_u * V) * BL[1] + (U_row * V) * BR[1]

        map_x = (one_minus_v * T_row[None, :, 0] + V * B_row[None, :, 0] + one_minus_u * L[:, None, 0] + U_row * R[:, None, 0] - bil_x).astype(np.float32)
        map_y = (one_minus_v * T_row[None, :, 1] + V * B_row[None, :, 1] + one_minus_u * L[:, None, 1] + U_row * R[:, None, 1] - bil_y).astype(np.float32)

        out[y0:y1] = cv2.remap(img_bgr, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return out


def draw_source_uv_grid(img_bgr: np.ndarray, edges: dict[str, np.ndarray], samples: int = 9) -> np.ndarray:
    overlay = img_bgr.copy()

    def point(u: float, v: float) -> np.ndarray:
        T = interp_curve_array(edges['top'], np.array([[u]], np.float32))[0, 0]
        B = interp_curve_array(edges['bottom'], np.array([[u]], np.float32))[0, 0]
        L = interp_curve_array(edges['left'], np.array([[v]], np.float32))[0, 0]
        R = interp_curve_array(edges['right'], np.array([[v]], np.float32))[0, 0]
        TL, TR, BL, BR = edges['top'][0], edges['top'][-1], edges['bottom'][0], edges['bottom'][-1]
        return (1 - v) * T + v * B + (1 - u) * L + u * R - (
            (1 - u) * (1 - v) * TL + u * (1 - v) * TR + (1 - u) * v * BL + u * v * BR
        )

    thickness = max(2, round(max(img_bgr.shape[:2]) / 1200))
    for uu in np.linspace(0.1, 0.9, samples):
        pts = np.array([point(float(uu), float(v)) for v in np.linspace(0, 1, 300)], dtype=np.float32)
        cv2.polylines(overlay, [np.round(pts).astype(np.int32)], False, (255, 220, 0), thickness, cv2.LINE_AA)
    for vv in np.linspace(0.1, 0.9, samples):
        pts = np.array([point(float(u), float(vv)) for u in np.linspace(0, 1, 300)], dtype=np.float32)
        cv2.polylines(overlay, [np.round(pts).astype(np.int32)], False, (0, 210, 255), thickness, cv2.LINE_AA)
    for edge in edges.values():
        cv2.polylines(overlay, [np.round(edge).astype(np.int32)], False, (0, 0, 255), thickness + 1, cv2.LINE_AA)
    return overlay


def inset_edges(edges: dict[str, np.ndarray], frac: float) -> dict[str, np.ndarray]:
    """Shrink the boundary quad slightly toward its centroid.

    Segmentation places the page boundary in the soft reflectance ramp at the
    paper's physical edge, so the rectifier samples a 1-3px sliver of the dark
    surround/edge-shadow at the very border. After ink cleanup that sliver
    becomes a hard near-black line down the outer edge, which reads as a skewed,
    lopsided page. The outermost fraction of a music page is always margin (staff
    content never reaches the physical sheet edge), so nudging every edge inward
    a hair excludes the dark border for all four sides without touching content.
    Because every point is scaled about the shared centroid by the same factor,
    corners stay consistent and the Coons patch remains valid.
    """
    allpts = np.concatenate([edges[k] for k in ('top', 'bottom', 'left', 'right')], axis=0)
    c = allpts.mean(axis=0)
    s = 1.0 - frac
    return {k: (c + s * (v - c)).astype(np.float32) for k, v in edges.items()}


def rectify_boundary_coons(img_bgr: np.ndarray, mask: np.ndarray, out_w: int, out_h: int, smooth: float, inset_frac: float = 0.005) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    contour = largest_external_contour(mask)
    corners = find_corners_from_contour(contour)
    raw_edges = split_into_edges(contour, corners)
    edges = orient_edges(raw_edges, corners, n=1200, smooth=smooth)
    if inset_frac > 0:
        edges = inset_edges(edges, inset_frac)
    rect = remap_coons_chunked(img_bgr, edges, out_w, out_h)
    return rect, edges


# ----------------------------- ink cleanup -----------------------------

def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / (edge1 - edge0), 0, 1)
    return t * t * (3 - 2 * t)


def remove_small_components(mask: np.ndarray, min_area: int = 10) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = np.zeros_like(mask, dtype=np.uint8)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        # Preserve long/thin musical lines even when their area is small.
        if area >= min_area or w >= 25 or h >= 25:
            out[labels == i] = 255
    return out


def despeckle_isolated(mask: np.ndarray, big_area: int = 150, protect_radius: int = 15,
                       border_frac: float = 0.045) -> np.ndarray:
    """Drop isolated speckle in the page's outer border, away from real content.

    Illumination normalization is unreliable in the perimeter band of the page
    (edge shadow / vignetting), so the ink threshold sprinkles small isolated
    specks there. We remove only speckle that lies entirely within the outer
    ``border_frac`` frame and is not adjacent to substantial ink -- the region
    where genuine musical content never lives. Interior marks (staccato dots,
    dashed/dotted lines, tempo text) are always kept, even when small and
    isolated, so this cleans the edge frame without ever eating real content.
    """
    m = (mask > 0).astype(np.uint8)
    h, w = m.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return mask
    areas = stats[:, cv2.CC_STAT_AREA]
    ws = stats[:, cv2.CC_STAT_WIDTH]
    hs = stats[:, cv2.CC_STAT_HEIGHT]
    xs = stats[:, cv2.CC_STAT_LEFT]
    ys = stats[:, cv2.CC_STAT_TOP]
    is_large = (areas >= big_area) | (ws >= 25) | (hs >= 25)
    is_large[0] = False  # background

    bx = border_frac * w
    by = border_frac * h
    # A component is in the border frame only if its whole bbox sits in the outer
    # band on some side (never straddling into the interior content region).
    in_border = ((xs + ws) <= bx) | (xs >= (w - bx)) | ((ys + hs) <= by) | (ys >= (h - by))

    large = is_large[labels]
    k = 2 * protect_radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    prot = cv2.dilate(large.astype(np.uint8), kernel)
    touch = np.zeros(n, dtype=bool)
    touch[np.unique(labels[prot > 0])] = True

    # Discard only: small, isolated, AND in the border frame. Everything else stays.
    drop = (~is_large) & (~touch) & in_border
    drop[0] = False
    keep = ~drop
    keep[0] = False
    return np.where(keep[labels], np.uint8(255), np.uint8(0))


def clean_ink(rect_bgr: np.ndarray, mode: str = "soft-gray", threshold_bias: int = 16) -> tuple[np.ndarray, dict[str, np.ndarray], float]:
    gray = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray8 = np.clip(gray, 0, 255).astype(np.uint8)

    # Gentle paper-texture denoise before background estimation.
    den = cv2.bilateralFilter(gray8, d=5, sigmaColor=18, sigmaSpace=9).astype(np.float32)

    # Estimate and remove uneven illumination.
    bg = cv2.GaussianBlur(den, (0, 0), sigmaX=45, sigmaY=45)
    norm = den / np.maximum(bg, 1) * 238.0
    norm = np.clip(norm, 0, 255).astype(np.uint8)

    otsu_t, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = float(np.clip(otsu_t + threshold_bias, 145, 205))

    adap = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 10)
    glob = (norm < thr).astype(np.uint8) * 255
    mask = cv2.bitwise_or(glob, adap)

    # Speckle removal only. We intentionally skip a morphological open here: an
    # open erodes ~1px off every stroke, which noticeably thins staff lines and
    # fine articulations. remove_small_components already drops isolated noise
    # while preserving long/thin musical lines.
    mask = remove_small_components(mask, min_area=10)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8), iterations=1)
    # Clean isolated illumination speckle from the outer page frame only, so the
    # margins stay clean without ever removing interior content.
    mask = despeckle_isolated(mask, big_area=150, protect_radius=15)

    support = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    # Reach full ink over a wider, brighter band so entire stroke bodies render
    # solid instead of only their darkest cores (improves ink retention). The
    # support mask still confines ink to detected strokes, keeping paper clean.
    darkness = 1.0 - smoothstep(thr - 22, thr + 10, norm.astype(np.float32))
    alpha = np.where(support, np.clip(darkness, 0, 1), 0)
    alpha = cv2.GaussianBlur(alpha.astype(np.float32), (0, 0), sigmaX=0.45)
    alpha = np.clip(alpha ** 0.8, 0, 1)

    paper = np.full_like(gray, 255.0)
    soft_black = np.clip(paper * (1 - alpha) + 8.0 * alpha, 0, 255).astype(np.uint8)
    # Soft-gray ink tone: the original fill was `norm * 0.5`, which specular flash
    # glare lifted in the page centre so ink there washed toward light gray while
    # the edges stayed dark. Clamp that tone to a tight soft-gray band: the floor
    # keeps ink from going crude/black and the ceiling caps the glare wash, so ink
    # darkness is normalized across the page while the original fine, soft-gray
    # character (and the anti-aliased edges from `alpha`) is preserved.
    ink_tone = np.clip(norm.astype(np.float32) * 0.5, 24, 40)
    soft_gray = np.clip(paper * (1 - alpha) + ink_tone * alpha, 0, 255).astype(np.uint8)
    binary = np.full_like(norm, 255)
    binary[mask > 0] = 0

    # Normalized-gray option keeps paper shading normalized rather than removing paper completely.
    normalized_gray = norm

    overlay = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
    red = overlay.copy()
    red[mask > 0] = (0, 0, 255)
    overlay = cv2.addWeighted(overlay, 0.65, red, 0.35, 0)

    outputs = {
        "normalized-gray": normalized_gray,
        "soft-black": soft_black,
        "soft-gray": soft_gray,
        "binary": binary,
        "mask": mask,
        "ink-overlay": overlay,
    }
    return outputs[mode], outputs, thr


# ----------------------------- output PDF -----------------------------

def write_pdf_from_images(image_paths: list[Path], output_pdf: Path, page_size: PageSize, dpi: int) -> None:
    page_w_pt, page_h_pt = page_size.points()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    # img2pdf embeds each PNG's compressed stream verbatim (lossless, no
    # re-encoding), so the pages stay bit-identical while the container avoids the
    # ~40% bloat ReportLab added by decoding and re-deflating at a weak level. A
    # fixed page size stretches every image to the exact physical dimensions,
    # matching the previous preserveAspectRatio=False behaviour and preserving DPI.
    layout = img2pdf.get_layout_fun((page_w_pt, page_h_pt))
    with open(output_pdf, "wb") as f:
        f.write(img2pdf.convert([str(p) for p in image_paths], layout_fun=layout))


def save_output_image(img: np.ndarray, out_path: Path) -> None:
    # PNG grayscale is ideal for exact pixel output; img2pdf embeds the compressed
    # stream verbatim, so max PNG compression here directly shrinks the final PDF.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
    if not ok:
        raise RuntimeError(f"Could not write image: {out_path}")


def make_blank_page(out_w: int, out_h: int, mode: str) -> np.ndarray:
    # For all supported modes, a blank page should be pure white.
    return np.full((out_h, out_w), 255, dtype=np.uint8)


# ----------------------------- per-page / main -----------------------------
# ----------------------------- per-page / document / batch -----------------------------

SUPPORTED_INPUT_EXTENSIONS = {'.pdf', '.png', '.jpg', '.jpeg', '.tif', '.tiff', '.webp'}


def detect_seam_for_page(img_bgr: np.ndarray, mask: np.ndarray) -> dict | None:
    """Pick the binding side and locate its fold for a booklet page.

    Uses the silhouette+gutter seam detector (flatscan_seam) as the single clip
    line for every booklet page. A fully-floating sheet (table margin on both L
    and R) has no binding side and is never clipped, protecting full sheets whose
    content straddles the fold. When both sides run off the frame, the fold with
    the stronger prominence wins.
    """
    sides = flatscan_seam.binding_sides(mask)
    if not sides:
        return None
    cands = [flatscan_seam.detect_seam(mask, s, img_bgr=img_bgr) for s in sides]
    cands = [c for c in cands if c is not None]
    if not cands:
        return None
    return max(cands, key=lambda c: c["conf"])


def process_page(
    img_bgr: np.ndarray,
    page_num_1based: int,
    out_w: int,
    out_h: int,
    mode: str,
    boundary_smooth: float,
    threshold_bias: int,
    debug_dir: Path | None,
    straighten: bool = False,
    booklet: bool = False,
) -> np.ndarray:
    mask, seg_debug = segment_page(img_bgr)
    seam = detect_seam_for_page(img_bgr, mask) if booklet else None
    if seam is not None:
        mask = clip_mask_at_crease(mask, seam)
    rect, edges = rectify_boundary_coons(img_bgr, mask, out_w, out_h, boundary_smooth)
    cleaned, ink_debug, thr = clean_ink(rect, mode=mode, threshold_bias=threshold_bias)

    straighten_info = None
    cleaned_pre_straighten = cleaned
    align_info = None
    shear_info = None
    rightsq_info = None
    center_info = None
    cleaned_pre_align = cleaned
    if straighten:
        cleaned_pre_straighten = cleaned.copy()
        cleaned, straighten_info = flatscan_dewarp.straighten_staves(cleaned)
        cleaned, shear_info = flatscan_dewarp.deskew_barlines(cleaned)
        cleaned_pre_align = cleaned.copy()
        cleaned, align_info = flatscan_dewarp.align_system_margins(cleaned)
        cleaned, rightsq_info = flatscan_dewarp.align_right_margin(cleaned)
        cleaned, center_info = flatscan_dewarp.center_content(cleaned)

    if debug_dir is not None:
        d = debug_dir / f"page_{page_num_1based:04d}"
        d.mkdir(parents=True, exist_ok=True)
        imwrite(d / "00_input_render.png", img_bgr)
        imwrite(d / "06_page_mask.png", mask)
        imwrite(d / "07_page_mask_overlay.png", mask_overlay(img_bgr, mask))
        if seam is not None:
            imwrite(d / "05_seam_overlay.png", crease_overlay(img_bgr, seam))
        for name, arr in seg_debug.items():
            imwrite(d / name, arr)
        uv = draw_source_uv_grid(img_bgr, edges)
        imwrite(d / "08_source_boundary_uv_grid.png", uv)
        imwrite(d / "09_rectified_color.png", rect)
        imwrite(d / "10_clean_output.png", cleaned)
        # Retain alternate color/ink outputs for tuning.
        imwrite(d / "11_alt_soft_gray.png", ink_debug["soft-gray"])
        imwrite(d / "12_alt_soft_black.png", ink_debug["soft-black"])
        imwrite(d / "13_alt_binary.png", ink_debug["binary"])
        imwrite(d / "14_alt_normalized_gray.png", ink_debug["normalized-gray"])
        imwrite(d / "15_ink_mask.png", ink_debug["mask"])
        imwrite(d / "16_ink_mask_overlay.png", ink_debug["ink-overlay"])
        if straighten_info is not None:
            imwrite(d / "17_pre_straighten.png", cleaned_pre_straighten)
            imwrite(d / "18_straightened.png", cleaned_pre_align)
        if align_info is not None:
            imwrite(d / "19_aligned.png", cleaned)
        for png in d.glob("*.png"):
            if png.name in {"07_page_mask_overlay.png", "08_source_boundary_uv_grid.png", "09_rectified_color.png", "10_clean_output.png", "11_alt_soft_gray.png", "12_alt_soft_black.png", "13_alt_binary.png"}:
                save_preview(png, d / (png.stem + "_preview.jpg"))
        with open(d / "params.txt", "w", encoding="utf-8") as f:
            f.write(f"mode={mode}\n")
            f.write(f"boundary_smooth={boundary_smooth}\n")
            f.write(f"threshold_bias={threshold_bias}\n")
            f.write(f"ink_threshold={thr}\n")
            f.write(f"output_pixels={out_w}x{out_h}\n")
            f.write(f"straighten={straighten_info}\n")
            f.write(f"shear={shear_info}\n")
            f.write(f"align={align_info}\n")
            f.write(f"rightsq={rightsq_info}\n")
            f.write(f"center={center_info}\n")
            if seam is not None:
                f.write(f"seam=side:{seam['side']},cue:{seam['cue']},"
                        f"conf:{seam['conf']:.3f}\n")
            else:
                f.write(f"seam={'none' if booklet else 'off'}\n")
    return cleaned


def default_jobs() -> int:
    """Conservative default worker count.

    Each page worker renders a full-resolution scan and several large intermediate
    arrays, so it is both CPU- and memory-hungry. Use half the logical cores,
    capped at 4, to keep memory pressure and CPU contention reasonable on typical
    laptops while still giving a solid speedup.
    """
    cpu = os.cpu_count() or 2
    return max(1, min(4, cpu // 2))


def _render_clean_save_page(task: dict) -> int:
    """Worker entry point: render one input page, clean it, and save the PNG.

    Runs in a separate process, so it re-opens the input itself (pdfium documents
    are not shareable across processes) and takes only picklable arguments.
    """
    inp = InputPages(task["input_path"], render_scale=task["render_scale"])
    img = inp.render_page_bgr(task["index"])
    cleaned = process_page(
        img,
        page_num_1based=task["index"] + 1,
        out_w=task["out_w"],
        out_h=task["out_h"],
        mode=task["mode"],
        boundary_smooth=task["boundary_smooth"],
        threshold_bias=task["threshold_bias"],
        debug_dir=task["debug_dir"],
        straighten=task.get("straighten", False),
        booklet=task.get("booklet", False),
    )
    save_output_image(cleaned, task["page_png"])
    del img, cleaned
    gc.collect()
    return task["index"]


def make_blank_artifacts(
    pages_dir: Path,
    name: str,
    out_w: int,
    out_h: int,
    mode: str,
    resume: bool,
) -> Path:
    blank_png = pages_dir / f"{name}.png"
    if not (resume and blank_png.exists()):
        blank = make_blank_page(out_w, out_h, mode)
        save_output_image(blank, blank_png)
        del blank
    return blank_png


def process_document(
    input_path: Path,
    output_path: Path,
    args: argparse.Namespace,
    page_size: PageSize,
    starts_on_even: bool = False,
) -> dict[str, object]:
    """Process one PDF/image into one normalized PDF. Returns a small stats dict."""
    inp = InputPages(input_path, render_scale=args.render_scale)
    pages = parse_pages(args.pages, inp.count)
    if not pages:
        raise ValueError(f"No pages selected for {input_path}")

    debug_pages: set[int] = set()
    if args.debug:
        debug_spec = args.debug_pages or args.pages
        debug_pages = set(parse_pages(debug_spec, inp.count))

    out_w, out_h = page_size.pixels_for(args.dpi)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    work_dir = args.work_dir or (output_path.with_suffix("").parent / (output_path.stem + "_work"))
    debug_root = args.debug_dir or output_path.with_suffix("").parent / (output_path.stem + "_debug")
    pages_dir = work_dir / "pages_png"
    pages_dir.mkdir(parents=True, exist_ok=True)

    if args.debug:
        debug_root.mkdir(parents=True, exist_ok=True)

    # Unless resuming, remove stale rendered pages while preserving debug output.
    if not args.resume:
        for old in pages_dir.glob("*"):
            if old.is_file():
                old.unlink()

    final_page_paths: list[Path] = []
    kept_dir = output_path.with_suffix("").parent / (output_path.stem + "_page_pngs")
    if args.keep_page_pngs:
        kept_dir.mkdir(parents=True, exist_ok=True)

    if starts_on_even:
        print("  Creating leading blank page...", file=sys.stderr)
        blank_png = make_blank_artifacts(
            pages_dir, "page_0000_blank_leading", out_w, out_h,
            args.mode, args.resume,
        )
        final_page_paths.append(blank_png)
        if args.keep_page_pngs:
            shutil.copy2(blank_png, kept_dir / blank_png.name)

    jobs = max(1, int(getattr(args, "jobs", 1) or 1))

    # Determine which pages still need processing (resume reuses existing PNGs).
    page_pngs: dict[int, Path] = {idx: pages_dir / f"page_{idx + 1:04d}.png" for idx in pages}
    todo: list[int] = []
    for idx in pages:
        if args.resume and page_pngs[idx].exists():
            print(f"  Reusing page {idx + 1}/{inp.count}: {page_pngs[idx]}", file=sys.stderr)
        else:
            todo.append(idx)

    def make_task(idx: int) -> dict:
        return {
            "input_path": input_path,
            "render_scale": args.render_scale,
            "index": idx,
            "page_png": page_pngs[idx],
            "out_w": out_w,
            "out_h": out_h,
            "mode": args.mode,
            "boundary_smooth": args.boundary_smooth,
            "threshold_bias": args.threshold_bias,
            "debug_dir": debug_root if (args.debug and idx in debug_pages) else None,
            "straighten": getattr(args, "straighten", False),
            "booklet": getattr(args, "booklet", False),
        }

    workers = min(jobs, len(todo)) if todo else 0
    if workers <= 1:
        for idx in todo:
            print(f"  Processing page {idx + 1}/{inp.count}...", file=sys.stderr)
            _render_clean_save_page(make_task(idx))
    else:
        print(f"  Processing {len(todo)} page(s) with {workers} workers...", file=sys.stderr)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_render_clean_save_page, make_task(idx)): idx for idx in todo}
            for fut in as_completed(futures):
                idx = futures[fut]
                fut.result()  # re-raise any worker exception
                print(f"  Finished page {idx + 1}/{inp.count}", file=sys.stderr)

    for idx in pages:
        page_png = page_pngs[idx]
        final_page_paths.append(page_png)
        if args.keep_page_pngs:
            shutil.copy2(page_png, kept_dir / page_png.name)

    appended_trailing_blank = False
    if args.pad_even and (len(final_page_paths) % 2 == 1):
        print("  Creating trailing blank page to make page count even...", file=sys.stderr)
        blank_png = make_blank_artifacts(
            pages_dir, "page_9999_blank_trailing", out_w, out_h,
            args.mode, args.resume,
        )
        final_page_paths.append(blank_png)
        appended_trailing_blank = True
        if args.keep_page_pngs:
            shutil.copy2(blank_png, kept_dir / blank_png.name)

    write_pdf_from_images(final_page_paths, output_path, page_size, args.dpi)

    print(f"  Wrote {output_path}", file=sys.stderr)
    print(f"  Physical page size: {page_size.width_in:g} x {page_size.height_in:g} in", file=sys.stderr)
    print(f"  Output raster size: {out_w} x {out_h} px ({args.dpi} dpi)", file=sys.stderr)
    print(f"  Working pages: {work_dir}", file=sys.stderr)
    if starts_on_even:
        print("  Inserted leading blank page for even-page scan start", file=sys.stderr)
    if appended_trailing_blank:
        print("  Appended trailing blank page for duplex-even page count", file=sys.stderr)
    if args.debug:
        print(f"  Debug outputs: {debug_root}", file=sys.stderr)
    if args.keep_page_pngs:
        print(f"  Per-page PNGs: {kept_dir}", file=sys.stderr)
    if args.clean_work:
        shutil.rmtree(work_dir, ignore_errors=True)
        print(f"  Removed working directory: {work_dir}", file=sys.stderr)

    return {
        "input": input_path,
        "output": output_path,
        "source_pages_processed": len(pages),
        "final_pages": len(final_page_paths),
        "leading_blank": starts_on_even,
        "trailing_blank": appended_trailing_blank,
    }


def discover_input_files(input_dir: Path, recursive: bool = False) -> list[Path]:
    pattern_iter: Iterable[Path] = input_dir.rglob("*") if recursive else input_dir.iterdir()
    files = [p for p in pattern_iter if p.is_file() and p.suffix.lower() in SUPPORTED_INPUT_EXTENSIONS]
    return sorted(files, key=lambda p: str(p.relative_to(input_dir)).lower())


def read_starts_on_even_patterns(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []
    patterns: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if not item or item.startswith("#"):
            continue
        patterns.append(item)
    return patterns


def parse_comma_patterns(value: str | None) -> list[str]:
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def pattern_matches_file(pattern: str, file_path: Path, input_root: Path) -> bool:
    """Match against filename, stem, and relative path. Case-insensitive; globs allowed."""
    pat = pattern.strip().lower().replace("\\", "/")
    rel = str(file_path.relative_to(input_root)).replace("\\", "/")
    candidates = [
        file_path.name,
        file_path.stem,
        rel,
        str(Path(rel).with_suffix("")),
    ]
    candidates = [c.lower().replace("\\", "/") for c in candidates]

    # Exact-ish match first.
    if pat in candidates:
        return True

    # Glob match. If no wildcard is supplied, also allow substring-on-stem/name.
    has_glob = any(ch in pat for ch in "*?[]")
    if any(fnmatch.fnmatchcase(c, pat) for c in candidates):
        return True
    if not has_glob and any(pat in c for c in candidates):
        return True
    return False


def file_starts_on_even(file_path: Path, input_root: Path, args: argparse.Namespace, patterns: list[str]) -> bool:
    if args.starts_on_even_all:
        return True
    if bool(args.starts_on_even or args.leading_blank) and not input_root.is_dir():
        return True
    return any(pattern_matches_file(pat, file_path, input_root) for pat in patterns)


def default_batch_output_dir(input_dir: Path) -> Path:
    return input_dir.with_name(input_dir.name + "-Processed")


def output_path_for_batch(input_file: Path, input_root: Path, output_root: Path) -> Path:
    rel = input_file.relative_to(input_root)
    return (output_root / rel).with_suffix(".pdf")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Clean phone-scanned orchestra parts into normalized page-size PDFs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("input", type=Path, help="Input PDF/image, or a directory of PDFs/images for batch mode")
    ap.add_argument("output", type=Path, nargs="?", help="Output PDF, or output directory in batch mode. If omitted for a directory input, uses <input>-Processed")
    size = ap.add_mutually_exclusive_group()
    size.add_argument("--page-size", type=parse_page_size, default=PageSize(9.0, 12.0), help="Physical output page size, e.g. 9x12, 8.5x11, letter, a4")
    size.add_argument("--size-in", type=parse_page_size, help="Alias for --page-size, e.g. 9x12")
    ap.add_argument("--width-in", type=float, default=None, help="Output page width in inches; overrides --page-size when paired with --height-in")
    ap.add_argument("--height-in", type=float, default=None, help="Output page height in inches; overrides --page-size when paired with --width-in")
    ap.add_argument("--dpi", type=int, default=400, help="Output raster DPI")
    ap.add_argument("--jobs", "-j", type=int, default=default_jobs(), help="Number of pages to process in parallel (separate processes). Each worker is CPU- and memory-heavy; the default is half your cores capped at 4. Use 1 to disable parallelism")
    ap.add_argument("--pages", default="all", help="1-based page list/ranges to process, e.g. '1', '1-4', '1,3,5-7', or 'all'")
    ap.add_argument("--render-scale", type=float, default=1.0, help="Multiplier on the input rasterization resolution. Pages are auto-rendered at their embedded image's native resolution; this scales that up/down (e.g. 0.5 for faster/lower-res)")
    ap.add_argument("--mode", choices=["soft-gray", "soft-black", "binary", "normalized-gray"], default="soft-gray", help="Final output style")
    ap.add_argument("--boundary-smooth", type=float, default=0.045, help="Boundary curve smoothing fraction for Coons warp")
    ap.add_argument("--straighten", action="store_true", help="Opt-in: after rectification, straighten wavy/skewed staff lines using detected staves (best for sheet music). Safely no-ops on pages without clear staves")
    ap.add_argument("--booklet", action="store_true", help="Opt-in: half-spread/booklet captures where one L/R edge folds into the spine and the facing page bleeds in. Detects the fold crease and clips the page there so the neighbour is excluded and the binding edge comes out clean. Full flat sheets in view (e.g. title pages) are left whole")
    ap.add_argument("--threshold-bias", type=int, default=16, help="Ink threshold adjustment; lower keeps less ink/noise, higher keeps more faint ink")
    ap.add_argument("--debug", action="store_true", help="Write intermediate masks/UV grids/alternate outputs")
    ap.add_argument("--debug-dir", type=Path, default=None, help="Directory for debug outputs; defaults to <output_stem>_debug")
    ap.add_argument("--debug-pages", default=None, help="Only write debug outputs for these 1-based pages/ranges; defaults to processed pages when --debug is set")
    ap.add_argument("--starts-on-even", action="store_true", help="Single-file mode: insert a blank page at the beginning because the first scanned page is an even-numbered page")
    ap.add_argument("--leading-blank", action="store_true", help="Alias for --starts-on-even")
    ap.add_argument("--starts-on-even-all", action="store_true", help="Batch mode: insert a leading blank page for every input file")
    ap.add_argument("--starts-on-even-files", default=None, help="Batch mode: comma-separated filename/stem/relative-path patterns that need a leading blank page, e.g. 'violin I,violin 2.pdf,*cello*'")
    ap.add_argument("--starts-on-even-list", type=Path, default=None, help="Batch mode: text file with one filename/stem/glob pattern per line for files that start on even pages. Defaults to starts_on_even.txt inside the input directory if present")
    ap.add_argument("--no-pad-even", dest="pad_even", action="store_false", help="Do not append a trailing blank page when the final page count is odd")
    ap.set_defaults(pad_even=True)
    ap.add_argument("--recursive", action="store_true", help="Batch mode: scan input directory recursively and preserve subdirectories in output")
    ap.add_argument("--work-dir", type=Path, default=None, help="Persistent page-by-page working directory. In batch mode, each output file gets its own default work dir; avoid setting this unless processing one file")
    ap.add_argument("--resume", action="store_true", help="Reuse already-rendered per-page outputs in the work directory instead of reprocessing them")
    ap.add_argument("--clean-work", action="store_true", help="Delete each working directory after a successful final PDF assembly")
    ap.add_argument("--keep-page-pngs", action="store_true", help="Also copy final per-page PNGs next to the output PDF")
    return ap


def resolve_page_size(args: argparse.Namespace, ap: argparse.ArgumentParser) -> PageSize:
    if args.width_in is not None or args.height_in is not None:
        if args.width_in is None or args.height_in is None:
            ap.error("--width-in and --height-in must be supplied together")
        return PageSize(args.width_in, args.height_in)
    return args.size_in or args.page_size


def validate_args(args: argparse.Namespace, ap: argparse.ArgumentParser) -> None:
    if args.dpi < 72 or args.dpi > 1200:
        ap.error("--dpi should be between 72 and 1200")
    if args.render_scale <= 0:
        ap.error("--render-scale must be positive")
    if args.jobs < 1:
        ap.error("--jobs must be at least 1")
    if args.input.is_dir() and args.work_dir is not None:
        ap.error("--work-dir is only supported for single-file mode; batch mode creates one work dir per output file")


def run_single(args: argparse.Namespace, page_size: PageSize) -> int:
    if args.output is None:
        raise SystemExit("Single-file mode requires an output PDF path")
    starts = bool(args.starts_on_even or args.leading_blank)
    process_document(args.input, args.output, args, page_size, starts_on_even=starts)
    return 0


def run_batch(args: argparse.Namespace, page_size: PageSize) -> int:
    input_dir = args.input
    output_dir = args.output or default_batch_output_dir(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = discover_input_files(input_dir, recursive=args.recursive)
    if not files:
        raise SystemExit(f"No supported input files found in {input_dir}")

    list_path = args.starts_on_even_list
    if list_path is None:
        auto = input_dir / "starts_on_even.txt"
        list_path = auto if auto.exists() else None
    patterns = parse_comma_patterns(args.starts_on_even_files) + read_starts_on_even_patterns(list_path)

    print(f"Batch input: {input_dir}", file=sys.stderr)
    print(f"Batch output: {output_dir}", file=sys.stderr)
    print(f"Found {len(files)} file(s)", file=sys.stderr)
    if patterns:
        print(f"Leading-blank patterns: {patterns}", file=sys.stderr)
    if args.pad_even:
        print("Trailing blank padding is ON: odd final page counts will be made even", file=sys.stderr)

    completed = 0
    failures: list[tuple[Path, Exception]] = []
    for i, input_file in enumerate(files, start=1):
        output_file = output_path_for_batch(input_file, input_dir, output_dir)
        starts = file_starts_on_even(input_file, input_dir, args, patterns)
        print(f"\n[{i}/{len(files)}] {input_file.relative_to(input_dir)}", file=sys.stderr)
        if starts:
            print("  Marked as starting on an even printed page; adding leading blank", file=sys.stderr)
        try:
            # Ensure batch mode never reuses one explicitly supplied work/debug dir across files.
            single_args = argparse.Namespace(**vars(args))
            single_args.work_dir = None
            single_args.debug_dir = None if args.debug_dir is None else (args.debug_dir / input_file.relative_to(input_dir).with_suffix(""))
            process_document(input_file, output_file, single_args, page_size, starts_on_even=starts)
            completed += 1
        except Exception as exc:  # Keep batch processing the remaining parts.
            failures.append((input_file, exc))
            print(f"  FAILED: {exc}", file=sys.stderr)

    print(f"\nBatch complete: {completed}/{len(files)} succeeded", file=sys.stderr)
    if failures:
        print("Failures:", file=sys.stderr)
        for p, exc in failures:
            print(f"  - {p.relative_to(input_dir)}: {exc}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)
    page_size = resolve_page_size(args, ap)
    validate_args(args, ap)

    if args.input.is_dir():
        return run_batch(args, page_size)
    return run_single(args, page_size)


if __name__ == "__main__":
    raise SystemExit(main())
