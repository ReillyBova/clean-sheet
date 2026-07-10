"""Binding-fold ("seam") detection.

The seam is the physical fold of a bound book in a half-spread phone capture.
A wrong seam corrupts the page mask and the Coons rectification (which shows up
as skew), so this must be highly reliable.

The detector is deterministic, illumination-independent, and makes no assumption
about where content sits on the page. It reads the fold straight off the
crease's own visual signature: the fold is a thin, sharp brightness *couplet* (a
bright ridge beside a dark valley where the paper bends) that runs continuously
down the whole page. Content edges (notes, barlines, labels) are the same kind
of local feature but appear on only *some* rows, so the per-column median of the
couplet response cancels them out while the fold -- present on every row --
stands out (``_crease_couplet``). That anchor is then refined per row onto the
local crease line to recover tilt/bow (``_crease_snap``).

Validated against hand-labeled ground truth (tools/seam_labels.json,
tools/seam_validate.py): mean per-row error ~0.6%, worst row <=1.4% of page
width across 19 diverse pages.

Public API:
    binding_sides(mask, touch_frac)          -> ["left"|"right", ...]
    detect_seam(mask, side, img_bgr)         -> dict | None
"""
from __future__ import annotations

import cv2
import numpy as np


def binding_sides(mask: np.ndarray, touch_frac: float = 0.02) -> list[str]:
    """Sides where the paper runs off the frame (candidate binding sides).

    A booklet half-spread always runs into the spine on one L/R side, so the mask
    reaches the image border there. A fully-floating sheet (margin all around) is
    not a half-spread and returns no sides.
    """
    m = mask > 0
    if not m.any():
        return []
    h, w = mask.shape
    xs = np.where(m.any(axis=0))[0]
    x0, x1 = int(xs.min()), int(xs.max())
    tp = max(2, int(round(touch_frac * w)))
    out = []
    if x0 <= tp:
        out.append("left")
    if x1 >= (w - 1 - tp):
        out.append("right")
    return out


def _crease_snap(img_bgr, mask, curve, w, h, win_frac=0.025):
    """Snap a near-fold seam onto the physical crease.

    The crease is a sharp, *local* feature (a thin brightness line where the
    paper bends), distinct from the broad, low-frequency gutter shadow. We remove
    the shadow with a high-pass (subtract a wide horizontal blur), then in a small
    window around the incoming seam find, per row band, the strongest local dark
    line, and fit a robust line through them. The window is tight so the snap is a
    local correction that cannot wander onto content.
    """
    L = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    k = (int(0.04 * w) | 1)
    resid = L - cv2.GaussianBlur(L, (k, 1), 0)  # shadow removed; crease = min
    win = int(win_frac * w)
    dy = int(0.02 * h)
    yfs = np.linspace(0.1, 0.9, 9)
    xs = []
    for yf in yfs:
        y = int(yf * (h - 1))
        a = int(curve[y])
        lo, hi = max(0, a - win), min(w, a + win)
        if hi - lo < 3:
            xs.append(a)
            continue
        band = resid[max(0, y - dy):y + dy, lo:hi].mean(0)
        m = mask[y, lo:hi] > 0
        if m.sum() < 3:
            xs.append(a)
            continue
        band = band.copy()
        band[~m] = 1e9
        xs.append(lo + int(np.argmin(band)))
    xs = np.array(xs, float)
    yy = yfs * (h - 1)
    for _ in range(2):  # robust line fit, reject outlier bands
        p = np.polyfit(yy, xs, 1)
        r = xs - np.polyval(p, yy)
        keep = np.abs(r) < 1.5 * np.std(r) + 1
        if keep.sum() < 4:
            break
        yy, xs = yy[keep], xs[keep]
    p = np.polyfit(yy, xs, 1)
    return np.clip(np.polyval(p, np.arange(h)), 0, w - 1)


def _crease_couplet(img_bgr, mask, side, w, h):
    """Fold x from the crease's own visual signature -- content-agnostic.

    The binding fold is a thin, sharp brightness *couplet* (a bright ridge beside
    a dark valley where the paper bends), running continuously down the whole
    page. Content edges (notes, barlines, labels) are the same kind of local
    feature but only appear on *some* rows, so the per-column median of the
    couplet response cancels them out while the fold -- present on every row --
    stands out. We take the strongest consistent step of either sign (its
    polarity flips with the lighting/which page is raised), within the binding
    third. Returns (x, score) or (None, 0).
    """
    L = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    k = (int(0.045 * w) | 1)
    resid = L - cv2.GaussianBlur(L, (k, 1), 0)      # broad shadow removed
    kk = max(3, int(0.006 * w))
    coup = np.roll(resid, kk, axis=1) - np.roll(resid, -kk, axis=1)
    coup[~(mask > 0)] = np.nan
    score = np.abs(np.nan_to_num(np.nanmedian(coup, axis=0), nan=0.0))
    score = cv2.GaussianBlur(score.reshape(1, -1), (9, 1), 0).ravel()
    cols = np.where((mask > 0).any(axis=0))[0]
    if len(cols) < 10:
        return None, 0.0
    c0, c1 = int(cols.min()), int(cols.max())
    if side == "right":
        lo, hi = int(c0 + 0.58 * (c1 - c0)), c1 - 3
    else:
        lo, hi = c0 + 3, int(c0 + 0.42 * (c1 - c0))
    if hi - lo < 5:
        return None, 0.0
    x = lo + int(np.argmax(score[lo:hi + 1]))
    return x, float(score[x])


def detect_seam(
    mask: np.ndarray,
    side: str,
    img_bgr: np.ndarray | None = None,
) -> dict | None:
    """Locate the binding fold on the given side of a half-spread page.

    The fold is found directly from its crease signature (``_crease_couplet``) --
    a continuous ridge/valley line, distinguished from content by being present
    on every row -- then refined per row onto the local crease (``_crease_snap``)
    to recover tilt. Deterministic, illumination-independent, and makes no
    assumption about where content sits. Validated to <=1.4% of page width
    (mean ~0.6%) against hand-labeled ground truth over 19 diverse pages.

    Returns {curve, points, side, cue, top_apex, bot_apex, conf} or None.
    """
    h, w = mask.shape
    if img_bgr is None:
        return None
    ax, score = _crease_couplet(img_bgr, mask, side, w, h)
    if ax is None:
        return None
    curve = _crease_snap(img_bgr, mask, np.full(h, float(ax)), w, h)
    pts = [[round(float(yf), 4), round(float(curve[int(yf * (h - 1))] / w), 4)]
           for yf in np.linspace(0.05, 0.95, 9)]
    return {
        "curve": curve,
        "points": pts,
        "side": side,
        "cue": "crease",
        "top_apex": (int(curve[0]), 0),
        "bot_apex": (int(curve[-1]), h - 1),
        "conf": score,
    }
