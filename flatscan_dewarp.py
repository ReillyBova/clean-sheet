"""Optional staff-line-guided dewarping for FlatScan.

This is an *opt-in* refinement applied after the main Coons rectification /
ink-cleanup pipeline. The Coons warp corrects gross page geometry from the four
boundary curves, but it has no information about the page interior, so residual
"waviness" (staff lines dipping/rising) and a gentle skew can remain. Sheet
music gives us an excellent interior signal for fixing this: the staff lines,
which should be perfectly straight, horizontal, and parallel.

Approach (grounded in standard document-dewarping / OMR techniques):

  1. Estimate staff line thickness and staff space from vertical run-length
     histograms (Dalitz/Fujinaga-style).
  2. Emphasize thin horizontal ink to isolate staff-line structure and group it
     into staff *systems* (the 5-line "combs").
  3. For each system, measure how its whole comb shifts vertically as we scan
     across columns, using INCREMENTAL neighbour-to-neighbour correlation. The
     comb is periodic, so a fixed reference can lock onto the wrong line
     (+/- one staff space); incremental tracking with a small max-shift follows
     the gentle, real wave instead.
  4. Flatten each system by holding its correction CONSTANT across its own
     vertical band (rigid translation -> internal staff-line spacing is
     preserved; this avoids a "pinch"), transitioning only between systems in
     the gaps. Build a monotonic vertical displacement field (which cannot fold
     and smear) and remap.

Safety first: if too few systems are found (title pages, near-blank pages) or
the required displacement is implausibly large, the input is returned unchanged.
This module never makes a page worse than the reliable Coons output.

Public entry point: ``straighten_staves(gray) -> (out_gray, info)``.
"""
from __future__ import annotations

import cv2
import numpy as np


def _staff_metrics(bw: np.ndarray) -> tuple[int, int]:
    """Estimate (staff_line_thickness, staff_space) via vertical run-lengths."""
    h, w = bw.shape
    ink = bw > 0
    white: list[int] = []
    black: list[int] = []
    for x in range(0, w, 6):
        col = ink[:, x]
        idx = np.where(np.diff(col.astype(np.int8)) != 0)[0] + 1
        for s in np.split(col, idx):
            (black if s[0] else white).append(len(s))

    def mode(vals: list[int], lo: int, hi: int) -> int | None:
        vals = [v for v in vals if lo <= v <= hi]
        return int(np.argmax(np.bincount(vals))) if vals else None

    return (mode(black, 1, 20) or 3), (mode(white, 4, 90) or 20)


def _emphasize(bw: np.ndarray, thickness: int) -> np.ndarray:
    """Keep horizontally-extended ink (staff-line-like structure)."""
    hlen = max(15, thickness * 12)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (hlen, 1))
    return cv2.morphologyEx(bw, cv2.MORPH_OPEN, hk)


def _find_systems(horiz: np.ndarray, space: int) -> list[tuple[int, int]]:
    """Return (ytop, ybot) row windows for each detected staff system."""
    proj = (horiz > 0).sum(1).astype(np.float32)
    if proj.max() <= 0:
        return []
    rows = proj > proj.max() * 0.22
    centers: list[int] = []
    y = 0
    h = len(rows)
    while y < h:
        if rows[y]:
            y0 = y
            while y < h and rows[y]:
                y += 1
            centers.append((y0 + y - 1) // 2)
        else:
            y += 1
    if not centers:
        return []
    # Group nearby line-centers into systems (a system spans ~5 lines).
    systems: list[list[int]] = []
    grp = [centers[0]]
    for c in centers[1:]:
        if c - grp[-1] <= space * 2.2:
            grp.append(c)
        else:
            systems.append(grp)
            grp = [c]
    systems.append(grp)
    out: list[tuple[int, int]] = []
    for g in systems:
        if len(g) < 3:  # require a real comb, not a stray line
            continue
        top = g[0] - int(space * 1.5)
        bot = g[-1] + int(space * 1.5)
        out.append((max(0, top), min(horiz.shape[0], bot)))
    return out


def _profile(col_block: np.ndarray, ytop: int, ybot: int) -> np.ndarray:
    return col_block[ytop:ybot].sum(1).astype(np.float32)


def _shift_by_corr(prof: np.ndarray, ref: np.ndarray, maxshift: int) -> tuple[float, float]:
    """Best sub-pixel vertical shift aligning prof to ref within +/-maxshift."""
    n = len(ref)
    best = 0
    bestval = -1e18
    scores: dict[int, float] = {}
    for s in range(-maxshift, maxshift + 1):
        if s >= 0:
            p = prof[s:]
            r = ref[:n - s]
        else:
            p = prof[:n + s]
            r = ref[-s:]
        if len(p) < n // 2:
            continue
        val = float(np.dot(p, r) / (np.linalg.norm(p) * np.linalg.norm(r) + 1e-9))
        scores[s] = val
        if val > bestval:
            bestval = val
            best = s
    if best - 1 in scores and best + 1 in scores:  # parabolic sub-pixel refine
        a, b, c = scores[best - 1], scores[best], scores[best + 1]
        denom = a - 2 * b + c
        if abs(denom) > 1e-9:
            best = best + 0.5 * (a - c) / denom
    return best, bestval


def compute_displacement(gray: np.ndarray):
    """Return ((map_x, map_y), info) for the dewarp, or (None, info) if N/A."""
    h, w = gray.shape
    bw = (gray < 150).astype(np.uint8) * 255
    thickness, space = _staff_metrics(bw)
    horiz = _emphasize(bw, thickness)
    systems = _find_systems(horiz, space)
    if len(systems) < 2:
        return None, dict(reason="too few systems", nsys=len(systems))

    colink = (horiz > 0).sum(0)
    xs = np.where(colink > colink.max() * 0.05)[0]
    if len(xs) < 10:
        return None, dict(reason="no content span", nsys=len(systems))
    xL, xR = int(xs.min()), int(xs.max())

    block = max(80, (xR - xL) // 24)
    step = block // 2
    xcs = list(range(xL + block // 2, xR - block // 2, step))
    if len(xcs) < 4:
        return None, dict(reason="content too narrow", nsys=len(systems))
    # Also sample the true content ends. The interior scan insets by half a
    # block so its correlation window stays fully on content, which leaves the
    # outermost ~half-block of every staff unmeasured -- and that is exactly
    # where staves visibly bend up/down (bottom-right corners especially),
    # because the resample below freezes those columns flat via constant
    # extrapolation. Adding explicit end samples (with clamped half-windows in
    # _profile) measures the end bend so it gets corrected; if an end has too
    # little ink to correlate reliably, the val<=0.5 gate simply drops it and we
    # degrade to the previous constant-hold behaviour -- never worse.
    if xcs[0] - xL > step // 2:
        xcs = [xL] + xcs
    if xR - xcs[-1] > step // 2:
        xcs = xcs + [xR]

    step_shift = max(3, int(space * 0.6))
    cmid = len(xcs) // 2
    sys_curves: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for (ytop, ybot) in systems:
        profs = [_profile(horiz[:, max(0, xc - block // 2): xc + block // 2], ytop, ybot) for xc in xcs]
        offs = np.zeros(len(xcs), np.float32)
        acc = 0.0
        for i in range(cmid + 1, len(xcs)):
            s, val = _shift_by_corr(profs[i], profs[i - 1], step_shift)
            acc += s if val > 0.5 else 0.0
            offs[i] = acc
        acc = 0.0
        for i in range(cmid - 1, -1, -1):
            s, val = _shift_by_corr(profs[i], profs[i + 1], step_shift)
            acc += s if val > 0.5 else 0.0
            offs[i] = acc
        k = 5  # light moving-average: kill jitter, keep the wave shape
        pad = np.pad(offs, k // 2, mode="edge")
        offs_s = np.convolve(pad, np.ones(k) / k, mode="valid")
        disp = offs_s - offs_s.mean()  # flatten to the system's mean height
        sys_curves.append((ytop, ybot, np.array(xcs, np.float32), disp.astype(np.float32)))

    if len(sys_curves) < 2:
        return None, dict(reason="no reliable systems", nsys=len(sys_curves))

    # Resample each system's displacement to full width and sort top-to-bottom.
    disp_full = np.array([
        np.interp(np.arange(w), xa, disp, left=disp[0], right=disp[-1])
        for (_, _, xa, disp) in sys_curves
    ])
    tops = np.array([c[0] for c in sys_curves], np.float32)
    bots = np.array([c[1] for c in sys_curves], np.float32)
    order = np.argsort(tops)
    tops, bots, disp_o = tops[order], bots[order], disp_full[order]
    S = len(tops)

    # Displacement is held CONSTANT across each system's band (rigid translation,
    # preserving staff spacing) via control points at the band edges; linear
    # interpolation fills the inter-system gaps. Edge y's are forced strictly
    # increasing so overlapping padded bands degrade to a clean boundary rather
    # than a fold. This avoids the "pinch" that arises from interpolating between
    # system centres.
    yk = np.empty(2 * S, np.float32)
    yk[0::2] = tops
    yk[1::2] = bots
    yk = np.maximum.accumulate(yk) + np.arange(2 * S) * 1e-3
    dk = np.empty((2 * S, w), np.float32)
    dk[0::2] = disp_o
    dk[1::2] = disp_o

    yout = np.arange(h, dtype=np.float32)
    map_y = np.empty((h, w), np.float32)
    for x in range(w):
        map_y[:, x] = yout + np.interp(yout, yk, dk[:, x])
    np.maximum.accumulate(map_y, axis=0, out=map_y)  # forbid folds

    map_x = np.repeat(np.arange(w).reshape(1, -1), h, axis=0).astype(np.float32)
    maxdisp = float(np.abs(map_y - yout[:, None]).max())
    info = dict(reason="ok", nsys=len(sys_curves), space=space,
                thickness=thickness, maxdisp=maxdisp)
    return (map_x, map_y), info


def straighten_staves(gray: np.ndarray, max_disp_factor: float = 6.0,
                      passes: int = 5, refine_min_px: float = 1.5):
    """Flatten wavy/skewed staff lines. Returns (output_gray, info).

    Falls back to returning ``gray`` unchanged when no reliable staff structure
    is found or the required warp is implausibly large.

    The first pass removes the bulk of the waviness. Because the correction is a
    per-system rigid translation blended across gaps, a little residual bow can
    survive the first pass (the measured shift lags a sharp local wave). Running
    the *same* estimator again on the once-straightened page measures only that
    small residual and squeezes it out -- e.g. peak-to-peak staff deviation
    ~5.5px -> ~3px on a typical part page -- for the price of a tiny (~2px)
    extra warp. We iterate until the refinement becomes negligible
    (``refine_min_px``) or ``passes`` is reached, so the cost is bounded and a
    page that is already flat stops after one measuring pass.
    """
    res, info = compute_displacement(gray)
    if res is None:
        info["applied"] = False
        return gray, info
    if info["maxdisp"] > info["space"] * max_disp_factor:
        info["applied"] = False
        info["reason"] = f"maxdisp {info['maxdisp']:.0f} exceeds safety limit"
        return gray, info

    map_x, map_y = res
    out = cv2.remap(gray, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    info["applied"] = True
    info["passes"] = 1

    prev_refine = float("inf")
    for _ in range(passes - 1):
        res_r, info_r = compute_displacement(out)
        if res_r is None:
            break
        md = float(info_r["maxdisp"])
        # A refinement pass must be small: large -> the estimator locked onto a
        # different line (unreliable), so keep the previous, safe result.
        if md > info["space"] * max_disp_factor:
            break
        if md < refine_min_px:
            break  # converged; nothing meaningful left to correct
        if md >= prev_refine:
            break  # not shrinking -> at the measurement noise floor; stop
        out = cv2.remap(out, res_r[0], res_r[1], cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REPLICATE)
        info["passes"] += 1
        info["maxdisp_refine"] = md
        prev_refine = md

    return out, info


def _staff_left_margins(gray: np.ndarray, min_lines: int = 3):
    """Robust per-system staff left-edge x, and (ytop, ybot) bands.

    The reference is the leftmost column that contains at least ``min_lines`` of
    the system's staff lines. This deliberately ignores rehearsal-mark boxes
    (only 2 horizontal edges), measure numbers and other marginalia sitting left
    of / above the staff, which otherwise corrupt a naive "leftmost ink" margin
    and trigger spurious corrections on already-aligned pages.
    """
    bw = (gray < 150).astype(np.uint8) * 255
    thickness, space = _staff_metrics(bw)
    horiz = _emphasize(bw, thickness)
    systems = _find_systems(horiz, space)
    cys: list[float] = []
    lefts: list[float] = []
    bands: list[tuple[int, int]] = []
    for (ytop, ybot) in systems:
        band = horiz[ytop:ybot, :]
        rowproj = (band > 0).sum(1).astype(np.float32)
        if rowproj.max() <= 0:
            continue
        line_rows = np.where(rowproj > rowproj.max() * 0.4)[0]
        if len(line_rows) < min_lines:
            continue
        col_line_count = (band[line_rows, :] > 0).sum(0)
        xs = np.where(col_line_count >= min_lines)[0]
        if len(xs) == 0:
            continue
        cys.append((ytop + ybot) / 2.0)
        lefts.append(float(xs.min()))
        bands.append((ytop, ybot))
    return np.array(cys, np.float32), np.array(lefts, np.float32), bands, space


def _theil_sen(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    slopes = []
    n = len(x)
    for i in range(n):
        for j in range(i + 1, n):
            if x[j] != x[i]:
                slopes.append((y[j] - y[i]) / (x[j] - x[i]))
    if not slopes:
        return 0.0, float(np.median(y))
    m = float(np.median(slopes))
    b = float(np.median(y - m * x))
    return m, b


def _robust_drift_slope(cys: np.ndarray, lefts: np.ndarray) -> float:
    """Slope of the smooth left-margin drift (distortion), robust to indent and
    top-of-page outliers. Theil-Sen for a robust first estimate, then a
    least-squares refit on inliers (|resid| <= 3*MAD) to sharpen it."""
    m, b = _theil_sen(cys, lefts)
    resid = lefts - (m * cys + b)
    mad = np.median(np.abs(resid - np.median(resid))) + 1e-6
    keep = np.abs(resid - np.median(resid)) < 3.0 * mad
    if keep.sum() >= 3:
        A = np.vstack([cys[keep], np.ones(keep.sum())]).T
        m = float(np.linalg.lstsq(A, lefts[keep], rcond=None)[0][0])
    return m


def align_system_margins(gray: np.ndarray, max_shift_factor: float = 10.0,
                         min_shift_px: float = 10.0):
    """De-drift staff-system left margins so the page reads as a clean vertical
    column, without leaning bar lines. Returns (output_gray, info).

    Only the *smooth linear drift* of the per-system margin (capture distortion)
    is removed; each system is then translated horizontally as a rigid block,
    held constant across its own staff band and blended linearly through the
    gaps -- the horizontal mirror of ``straighten_staves``. Because the shift is
    constant within a band, bar lines and stems inside a system are never
    sheared; the gentle transition lives only in the (near-empty) gaps between
    systems. Intentional indents live in the *residual* about the drift line, so
    they are not part of the removed slope and are preserved. Robust fitting
    ignores top-of-page and indent outliers, so already-aligned pages get a
    slope near zero and this is a no-op.
    """
    h, w = gray.shape
    cys, lefts, bands, space = _staff_left_margins(gray)
    info: dict = dict(reason="ok", nsys=int(len(cys)), space=int(space))
    if len(cys) < 4:
        info.update(applied=False, reason="too few systems")
        return gray, info

    slope = _robust_drift_slope(cys, lefts)
    y_ref = float(cys.mean())
    shifts = slope * (y_ref - cys)  # per-system rigid horizontal shift (de-drift)
    max_shift = float(np.abs(shifts).max())
    info["drift_px_over_page"] = float(slope * h)
    info["max_shift"] = max_shift
    if max_shift < min_shift_px:
        info.update(applied=False, reason="drift below threshold")
        return gray, info
    if max_shift > max_shift_factor * space or max_shift > 0.10 * w:
        info.update(applied=False, reason=f"shift {max_shift:.0f}px exceeds safety limit")
        return gray, info

    # Build a per-row horizontal shift: constant across each system band, linear
    # in the gaps. Control points at band edges (strictly increasing y) mirror
    # the vertical dewarp so a system translates rigidly (no internal shear).
    order = np.argsort([b[0] for b in bands])
    tops = np.array([bands[i][0] for i in order], np.float32)
    bots = np.array([bands[i][1] for i in order], np.float32)
    sh = shifts[order]
    yk = np.empty(2 * len(tops), np.float32)
    yk[0::2] = tops
    yk[1::2] = bots
    yk = np.maximum.accumulate(yk) + np.arange(len(yk)) * 1e-3
    dk = np.empty(2 * len(tops), np.float32)
    dk[0::2] = sh
    dk[1::2] = sh
    ys = np.arange(h, dtype=np.float32)
    row_shift = np.interp(ys, yk, dk, left=dk[0], right=dk[-1]).astype(np.float32)
    map_x = (np.arange(w, dtype=np.float32)[None, :] - row_shift[:, None]).astype(np.float32)
    map_y = np.repeat(ys[:, None], w, axis=1).astype(np.float32)
    out = cv2.remap(gray, map_x, map_y, cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    info.update(applied=True)
    return out, info
