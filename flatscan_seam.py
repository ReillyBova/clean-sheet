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
stands out (``_crease_couplet``), giving the offset anchor. A robust Theil-Sen
line through the crease body recovers the tilt (``_crease_body_line``).

The endpoints -- where slope extrapolation is weakest -- are pinned by the fold's
own edge geometry: at the paper's top/bottom edge the two facing pages meet at
the binding in a V-notch (inverted at the bottom) whose vertex is exactly on the
fold (``_vnotch``). A notch is trusted only when two independent estimates (its
apex and the change-point of the edge profile) agree, so content never fakes one.

Validated against hand-labeled ground truth (tools/seam_labels.json,
tools/seam_validate.py): mean per-row error ~0.6%, worst row <=1.7% of page
width across 41 diverse pages (endpoint-inclusive metric).

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


def _foldmap(img_bgr, w):
    """2D 'foldness' response: adjacent bright ridge * dark valley (either polarity).

    A physical 3D crease is a bright highlight ridge right beside a dark shadow
    valley. Content edges are dark marks on flat paper (a valley with no ridge),
    so demanding BOTH suppresses content while the crease stands out. The broad
    gutter shadow is removed first with a wide horizontal high-pass.
    """
    L = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    k = (int(0.045 * w) | 1)
    resid = L - cv2.GaussianBlur(L, (k, 1), 0)
    kk = max(3, int(0.006 * w))
    lft = np.roll(resid, kk, axis=1)
    rgt = np.roll(resid, -kk, axis=1)
    return np.maximum(np.clip(lft, 0, None) * np.clip(-rgt, 0, None),
                      np.clip(-lft, 0, None) * np.clip(rgt, 0, None))


def _theilsen(y, x):
    """Robust (median-of-slopes) line fit x = sl*y + inter."""
    n = len(y)
    if n < 3:
        return 0.0, (float(np.median(x)) if n else 0.0)
    i, j = np.triu_indices(n, 1)
    dy = y[j] - y[i]
    ok = dy != 0
    if not ok.any():
        return 0.0, float(np.median(x))
    sl = float(np.median((x[j] - x[i])[ok] / dy[ok]))
    inter = float(np.median(x - sl * y))
    return sl, inter


def _crease_body_line(fmap, mask, anc, w, h, win_frac=0.025):
    """Straight fold line from the crease body (robust slope, anchored offset).

    The fold is a straight tilted line, so we sample its brightest crease column
    per row band inside a tight window around the anchor and fit a robust
    (Theil-Sen) line. Its offset is re-pinned to the reliable ``anc`` at mid-page;
    only the slope (tilt) comes from the body. This is the stable fallback that
    carries any endpoint the V-notch can't confirm.
    """
    fk = cv2.GaussianBlur(fmap, (1, (int(0.01 * h) | 1)), 0)
    win = int(win_frac * w)
    a, b = max(0, anc - win), min(w - 1, anc + win)
    py, px, pw = [], [], []
    for y in range(int(0.03 * h), int(0.97 * h), 6):
        seg = fk[y, a:b + 1]
        if seg.max() <= 0 or not (mask[y, a:b + 1] > 0).any():
            continue
        px.append(a + int(np.argmax(seg)))
        py.append(float(y))
        pw.append(float(seg.max()))
    if len(px) < 10:
        return np.full(h, float(anc))
    px, py, pw = np.array(px, float), np.array(py, float), np.array(pw, float)
    keep = pw >= np.percentile(pw, 40)  # drop content-corrupted weak rows
    if keep.sum() >= 10:
        px, py = px[keep], py[keep]
    sl, inter = _theilsen(py, px)
    slmax = 0.06 * w / h
    sl = float(np.clip(sl, -slmax, slmax))
    line = inter + sl * np.arange(h)
    line = line - (line[h // 2] - anc)  # re-pin offset to the reliable anchor
    return np.clip(line, 0, w - 1)


def _sharp_silhouette(img_bgr):
    """Crisp paper/background split (Otsu) that preserves the edge V-notch.

    Unlike the pipeline's morphologically-smoothed page mask (which erases the
    small notch), this keeps the sharp paper outline so the fold's fingerprint at
    the top/bottom edges survives. Median-filtered to drop isolated speckle
    without moving edges.
    """
    L = cv2.GaussianBlur(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, m = cv2.threshold(L, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.medianBlur((m > 0).astype(np.uint8), 3)


def _changepoint(xf, ef):
    """Fold x = the discontinuity in an edge profile (best 2-segment line split).

    Handles a facing-page 'step' (one page top lower than the other) as well as a
    symmetric V: both share a kink at the fold. Returns (x, confidence) where
    confidence is the fraction of residual explained by splitting vs one line.
    """
    n = len(xf)
    if n < 8:
        return None, 0.0

    def sse(x, y):
        if len(x) < 2:
            return 0.0
        p = np.polyfit(x, y, 1)
        r = y - np.polyval(p, x)
        return float((r * r).sum())

    single = sse(xf, ef)
    best, best_sse = None, 1e18
    for k in range(3, n - 3):
        s = sse(xf[:k], ef[:k]) + sse(xf[k:], ef[k:])
        if s < best_sse:
            best_sse, best = s, k
    if best is None:
        return None, 0.0
    return float(xf[best]), (single - best_sse) / (single + 1e-6)


def _vnotch(img_bgr, sil, anc, w, h, which, win_frac=0.05):
    """Two independent estimates of the fold x at the top/bottom paper edge.

    Where the two facing pages meet at the binding, their edges form a V-notch
    (an inverted V at the bottom), often with a tiny tooth-gap; its vertex sits
    exactly on the fold at the paper edge -- the precise, illumination-independent
    anchor for the hard-to-extrapolate endpoints. We read the paper's edge profile
    in a window around the anchor and locate the notch two ways: its apex (extreme
    edge point) and the change-point of the edge profile. When they agree the
    notch is real. Returns (apex_x, changepoint_x) or None.
    """
    win = int(win_frac * w)
    a, b = max(0, anc - win), min(w - 1, anc + win)
    xs = np.arange(a, b + 1)
    ed = np.full(len(xs), np.nan)
    for i, x in enumerate(xs):
        col = np.where(sil[:, x] > 0)[0]
        if len(col):
            ed[i] = col[0] if which == "top" else col[-1]
    ok = ~np.isnan(ed)
    if ok.sum() < 8:
        return None
    xf, ef = xs[ok], ed[ok].astype(np.float32)
    efs = cv2.GaussianBlur(ef.reshape(1, -1), (0, 0), 2).ravel()
    ai = int(np.argmax(efs)) if which == "top" else int(np.argmin(efs))
    cpx, _ = _changepoint(xf, ef)
    if cpx is None:
        return None
    return float(xf[ai]), cpx


def _crease_couplet(fmap, mask, side, w, h):
    """Fold x (offset anchor) from the crease's own visual signature.

    The crease runs continuously down the page, so the per-column median of the
    foldness response cancels sporadic content and leaves the fold standing out.
    Search is confined to the binding third. Returns (x, score) or (None, 0).
    """
    fold = fmap.copy()
    fold[~(mask > 0)] = np.nan
    score = np.nan_to_num(np.nanmedian(fold, axis=0), nan=0.0)
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

    Three independent, content-agnostic cues are fused into one straight fold line
    (the fold is straight to <1% bow across the labeled set):

    1. Offset anchor -- the fold column from the per-row-median crease signature
       (``_crease_couplet``); reliable for *where* the fold sits at mid-page.
    2. Slope -- a robust (Theil-Sen) line through the crease body
       (``_crease_body_line``); reliable for the *tilt*, anchored at the offset.
    3. Endpoints -- the V-notch vertex where the two page edges meet the binding
       at the top/bottom paper edge (``_vnotch``); the precise anchor exactly
       where slope extrapolation is weakest. Used only when its two independent
       estimates (apex + change-point) agree, so a spurious notch never wins.

    When both endpoints have a corroborated notch and their mutual tilt is
    plausible, they pin the line directly (the offset anchor can drift toward the
    paper rim on near-edge folds, so it is bypassed). A single confirmed endpoint
    is gated against the anchor and blended in; otherwise the crease-body line
    carries that end. Deterministic and illumination-independent. Validated to
    <=1.7% of page width (mean ~0.6%) against hand-labeled ground truth over 41
    diverse pages.

    Returns {curve, points, side, cue, top_apex, bot_apex, conf} or None.
    """
    h, w = mask.shape
    if img_bgr is None:
        return None
    fmap = _foldmap(img_bgr, w)
    ax, score = _crease_couplet(fmap, mask, side, w, h)
    if ax is None:
        return None

    line = _crease_body_line(fmap, mask, ax, w, h)
    sil = _sharp_silhouette(img_bgr)
    slmax = 0.06 * w / h
    gate = 0.035 * w
    agree_tol = 0.012 * w
    ends = {"top": int(0.04 * h), "bot": int(0.96 * h)}

    cons = {}  # end -> (y, consensus_x, alpha) where apex & change-point corroborate
    for end, yi in ends.items():
        v = _vnotch(img_bgr, sil, ax, w, h, end)
        if v is None:
            continue
        apx, cpx = v
        agree = abs(apx - cpx)
        if agree > agree_tol:
            continue
        alpha = float(np.clip(1.0 - agree / agree_tol, 0.0, 1.0)) * 0.85
        cons[end] = (yi, 0.5 * (apx + cpx), alpha)

    tgt = {e: (yi, float(line[yi])) for e, yi in ends.items()}
    if "top" in cons and "bot" in cons:
        (yt, xt, at), (yb, xb, ab) = cons["top"], cons["bot"]
        if abs((xb - xt) / (yb - yt)) <= slmax:  # notches corroborate each other
            tgt["top"] = (yt, float(line[yt]) + at * (xt - float(line[yt])))
            tgt["bot"] = (yb, float(line[yb]) + ab * (xb - float(line[yb])))
    else:  # single notch: gate against the offset anchor before trusting it
        for end, (yi, cx, alpha) in cons.items():
            if abs(cx - ax) <= gate:
                tgt[end] = (yi, float(line[yi]) + alpha * (cx - float(line[yi])))

    (y0, x0), (y1, x1) = tgt["top"], tgt["bot"]
    if y1 > y0:
        sl = float(np.clip((x1 - x0) / (y1 - y0), -slmax, slmax))
        line = np.clip(x0 + sl * (np.arange(h) - y0), 0, w - 1)

    cue = "crease+vnotch" if cons else "crease"
    pts = [[round(float(yf), 4), round(float(line[int(yf * (h - 1))] / w), 4)]
           for yf in np.linspace(0.05, 0.95, 9)]
    return {
        "curve": line,
        "points": pts,
        "side": side,
        "cue": cue,
        "top_apex": (int(line[0]), 0),
        "bot_apex": (int(line[-1]), h - 1),
        "conf": score,
    }
