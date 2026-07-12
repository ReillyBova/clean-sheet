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
  3. Trace each system as a *coupled comb* (``_trace_staff_lines``): all five
     lines share one centre curve and one slowly-varying spacing, fit per column,
     so a line that jumps onto a tie, note or bar line is rejected as a spacing
     outlier instead of dragging the trace off the staff. The comb is followed in
     a single continuous pass across the page so there is no seam at the middle.
  4. Build two candidate warps and keep whichever leaves the staff lines flatter
     (``_staff_flatness``), so a page is never made worse:
       - **guided** (``_staff_guided_displacement``) irons each traced comb flat
         with a per-column RIGID vertical shift -- internal staff-line spacing is
         preserved (no "pinch"/smear), only the whole comb translates -- blended
         through the inter-system gaps into a monotonic displacement field. This
         is the primary path and wins on essentially every real page.
       - **rigid** (``_straighten_staves_rigid``) is the robust fallback: it
         measures each system's shift by incremental neighbour-to-neighbour comb
         correlation and translates it rigidly. Used when the trace is unclear.

Safety first: if too few systems are found (title pages, near-blank pages) or
the required displacement is implausibly large, the input is returned unchanged.
This module never makes a page worse than the reliable Coons output.

Public entry point: ``straighten_staves(gray) -> (out_gray, info)``.
"""
from __future__ import annotations

import cv2
import numpy as np

# A guided warp whose residual staff waviness is at or below this (px) is
# accepted outright, skipping the slower rigid comparison pass.
_GUIDED_ACCEPT_PX = 3.0


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


def _smooth_disp_across_systems(disp_o: np.ndarray, scale: float = 4.0,
                                iters: int = 80) -> np.ndarray:
    """Regularize per-system displacement curves for vertical continuity.

    ``disp_o`` is ``(S, w)`` -- one per-column displacement curve per system,
    ordered top-to-bottom. The page's warp field is continuous in y, so a
    system's displacement at a given column should vary smoothly from the staves
    above to those below. A system whose value disagrees sharply with that trend
    (typically a binding-side end where the correlation/trace went blind and
    latched onto a wrong match) is an outlier; left alone it both under-corrects
    that end and, differing from its neighbours, collapses the inter-system gap
    into a vertical "scrunch".

    Each iteration relaxes every entry toward the mean of its vertical
    neighbours, weighted so entries already consistent with the trend are kept
    (a linear trend is a fixed point -- smooth curl is preserved) while strong
    outliers are pulled onto it. Converges to identity when all systems agree, so
    clean pages are untouched. Returns the regularized ``(S, w)`` array.
    """
    S = disp_o.shape[0]
    if S < 3:
        return disp_o.copy()
    d = disp_o.copy()
    for _ in range(iters):
        # Linear (not flat) extrapolation past the stack ends: the virtual
        # neighbour above the top / below the bottom continues the local trend
        # (2*edge - inner). This makes a linear curl trend a fixed point at the
        # boundary too -- otherwise edge replication treats the outermost systems
        # (which carry the *largest* binding curl) as outliers and damps their
        # correction, leaving the top/bottom tails bent.
        up = np.vstack([2 * d[:1] - d[1:2], d[:-1]])
        dn = np.vstack([d[1:], 2 * d[-1:] - d[-2:-1]])
        sm = 0.5 * (up + dn)
        resid = d - sm
        wkeep = 1.0 / (1.0 + (resid / scale) ** 2)   # ~1 consistent, ->0 outlier
        d = wkeep * d + (1.0 - wkeep) * sm
    return d


def _traced_centers(gray: np.ndarray, w: int):
    """Ridge-traced comb centre per system, interpolated to full width.

    Returns ``[(ytop, ybot, center_full), ...]``. Unlike the block correlation,
    the ridge trace keeps following each staff line into the binding tail (via
    its raw-ink fallback) where the lines foreshorten, so it recovers the steep
    end curl the correlation goes blind to. Used to extend the displacement into
    that unreliable tail (see compute_displacement)."""
    xcs_tr, sys_lines, _ = _trace_staff_lines(gray)
    out = []
    xs_all = np.arange(w)
    for (ytop, ybot, L) in sys_lines:
        center = np.nanmedian(L, axis=0)
        good = ~np.isnan(center)
        if good.sum() < 4:
            continue
        cf = np.interp(xs_all, xcs_tr[good], center[good],
                       left=center[good][0], right=center[good][-1]).astype(np.float32)
        out.append((float(ytop), float(ybot), cf))
    return out


def _match_center(centers, ytop, ybot):
    """Pick the ridge-traced centre whose band best overlaps [ytop, ybot]."""
    best, best_ov = None, 0.0
    for (t, b, cf) in centers:
        ov = max(0.0, min(ybot, b) - max(ytop, t))
        if ov > best_ov:
            best_ov, best = ov, cf
    return best


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
    xcs_a = np.array(xcs, np.float32)
    centers = _traced_centers(gray, w)          # ridge-traced curl for tail extension
    tail_lim = space * 4.0                       # cap on how far the tail may be lifted
    sys_curves: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for (ytop, ybot) in systems:
        profs = [_profile(horiz[:, max(0, xc - block // 2): xc + block // 2], ytop, ybot) for xc in xcs]
        offs = np.zeros(len(xcs), np.float32)
        vals = np.zeros(len(xcs), np.float32)
        vals[cmid] = 1.0
        acc = 0.0
        for i in range(cmid + 1, len(xcs)):
            s, val = _shift_by_corr(profs[i], profs[i - 1], step_shift)
            acc += s if val > 0.5 else 0.0
            offs[i] = acc; vals[i] = val
        acc = 0.0
        for i in range(cmid - 1, -1, -1):
            s, val = _shift_by_corr(profs[i], profs[i + 1], step_shift)
            acc += s if val > 0.5 else 0.0
            offs[i] = acc; vals[i] = val
        # Extend into the binding tail: where the correlation loses confidence
        # (val<=0.5) it freezes the displacement flat and under-corrects the steep
        # end curl. The ridge trace still follows that curl, so past the last
        # confident sample on each side we continue the displacement along the
        # traced comb centre (bounded), recovering the end bend.
        rb = cmid
        while rb + 1 < len(xcs) and vals[rb + 1] > 0.5:
            rb += 1
        lb = cmid
        while lb - 1 >= 0 and vals[lb - 1] > 0.5:
            lb -= 1
        cen = _match_center(centers, ytop, ybot)
        if cen is not None:
            if rb < len(xcs) - 1:
                base = offs[rb]; cb = cen[int(xcs_a[rb])]
                for i in range(rb + 1, len(xcs)):
                    d = float(cen[int(xcs_a[i])] - cb)
                    offs[i] = base + float(np.clip(d, -tail_lim, tail_lim))
            if lb > 0:
                base = offs[lb]; cb = cen[int(xcs_a[lb])]
                for i in range(lb - 1, -1, -1):
                    d = float(cen[int(xcs_a[i])] - cb)
                    offs[i] = base + float(np.clip(d, -tail_lim, tail_lim))
        k = 5  # light moving-average: kill jitter, keep the wave shape
        pad = np.pad(offs, k // 2, mode="edge")
        offs_s = np.convolve(pad, np.ones(k) / k, mode="valid")
        # flatten to the mean over the confidently-measured span (so extending the
        # tail cannot shift the reliable body's correction)
        ref = offs_s[lb:rb + 1].mean() if rb >= lb else offs_s.mean()
        disp = offs_s - ref
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

    # Enforce vertical continuity of the warp field: a system whose end
    # displacement disagrees sharply with the staves above/below is almost always
    # a bad binding-side measurement (the correlation latches onto a plausible
    # but wrong match where the lines foreshorten). Pull such outliers onto the
    # neighbours' smooth trend, then re-flatten each system to its own mean.
    disp_o = _smooth_disp_across_systems(disp_o)
    disp_o = disp_o - disp_o.mean(axis=1, keepdims=True)

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


def _seed_line_rows(prof: np.ndarray, space: int) -> list[int]:
    """Row indices of the staff-line peaks in a vertical ink profile."""
    peaks: list[int] = []
    thr = prof.max() * 0.30
    for i in range(1, len(prof) - 1):
        if prof[i] > thr and prof[i] >= prof[i - 1] and prof[i] >= prof[i + 1]:
            if peaks and (i - peaks[-1]) < space * 0.6:
                if prof[i] > prof[peaks[-1]]:
                    peaks[-1] = i
                continue
            peaks.append(i)
    return peaks


def _snap_comb(seeds: np.ndarray, strengths: np.ndarray | None = None) -> np.ndarray:
    """Trim seed peaks down to a uniformly-spaced staff comb.

    Staff lines are equally spaced and, sampled at the page centre, uniformly
    dark, so a peak that sits at an anomalous gap from the comb *or* is markedly
    fainter than its siblings (e.g. a slur, tie, ledger line or text baseline
    caught just above or below the staff) is not a real line -- it would
    otherwise make the coupled tracer maintain a phantom "6-line staff". We
    estimate the true line pitch from the *interior* gaps (robust to a bad
    endpoint) and peel off any end line whose gap to its neighbour departs from
    that pitch or whose ink is far weaker than the comb's, since the spurious
    catch is essentially always at the top or bottom edge of the comb. Interior
    gaps are never touched, so a genuinely missed faint line is preserved rather
    than mistaken for a merge."""
    seeds = np.asarray(seeds, np.float32)
    strengths = None if strengths is None else np.asarray(strengths, np.float32)
    while len(seeds) > 3:
        gaps = np.diff(seeds)
        interior = gaps[1:-1] if len(gaps) > 2 else gaps
        s = float(np.median(interior))
        if s <= 0:
            break
        r0, r1 = gaps[0] / s, gaps[-1] / s
        lo, hi = 0.72, 1.35   # a real comb step is ~1.0; outside => not a staff line
        c0 = r0 < lo or r0 > hi
        c1 = r1 < lo or r1 > hi
        # An endpoint far fainter than the comb body is a phantom even when its
        # spacing looks plausible (a light tie/ledger sitting near the pitch).
        w0 = w1 = 0.0
        if strengths is not None and len(strengths) == len(seeds):
            body = float(np.median(strengths[1:-1])) if len(strengths) > 2 else float(np.median(strengths))
            if body > 0:
                if strengths[0] < 0.55 * body:
                    c0 = True; w0 = 1.0 - strengths[0] / body
                if strengths[-1] < 0.55 * body:
                    c1 = True; w1 = 1.0 - strengths[-1] / body
        if not (c0 or c1):
            break
        sev0 = abs(np.log(r0)) + w0 if c0 else -1.0
        sev1 = abs(np.log(r1)) + w1 if c1 else -1.0
        if sev0 >= sev1:
            seeds = seeds[1:]
            if strengths is not None and len(strengths) == len(seeds) + 1:
                strengths = strengths[1:]
        else:
            seeds = seeds[:-1]
            if strengths is not None and len(strengths) == len(seeds) + 1:
                strengths = strengths[:-1]
    return seeds


def _fit_uniform_comb(prof: np.ndarray, space: int, thickness: int) -> list[int] | None:
    """Lock a rigid five-line staff comb onto a vertical ink profile.

    An orchestral staff has exactly five equally-spaced lines, so counting raw
    profile peaks is the wrong tool: a faint line is missed (four "lines") or a
    slur/tie/ledger is caught (six), and either wrong count corrupts the comb's
    relative geometry and kinks the warp. The staff pitch is already known
    robustly (``space + thickness``), so we simply slide a rigid five-tooth comb
    of that pitch over the profile in phase and keep the placement that lands the
    most ink on all five teeth, requiring at least four of five to sit on real
    ink. The fifth (possibly faint) line is then placed by the comb's own
    uniformity rather than trusted to peak detection. Returns five row indices,
    or ``None`` when no confident five-line fit exists so genuine one/two-line
    percussion staves fall back to peak seeding."""
    prof = np.asarray(prof, np.float32)
    m = float(prof.max())
    n = len(prof)
    pitch = float(space + thickness)
    span = 4.0 * pitch
    if m <= 0 or span >= n:
        return None
    thr = 0.30 * m
    win = max(1, thickness // 2 + 1)
    best = None
    y0 = 0.0
    while y0 <= n - 1 - span:
        pos = y0 + pitch * np.arange(5)
        yi = np.round(pos).astype(int)
        v = np.array([float(prof[max(0, yy - win):min(n, yy + win + 1)].max())
                      for yy in yi], np.float32)
        if int((v > thr).sum()) >= 4:
            tot = float(v.sum())
            if best is None or tot > best[0]:
                best = (tot, pos.copy())
        y0 += 1.0
    if best is None:
        return None
    return [int(round(y)) for y in best[1]]


def _staff_hextent(horiz: np.ndarray, raw: np.ndarray, xcs: np.ndarray,
                   L: np.ndarray, ytop: int, ybot: int):
    """Sampled-column indices ``(jL, jR)`` bounding the real staff horizontally.

    A traced comb is only meaningful between where the staff actually starts
    (first inked columns, after the left margin) and where it ends (the final
    barline on the binding side). Past those the trace merely coasts on its slope
    into the blank margin and fans out. We bound it so the correction can be held
    flat outside the staff rather than bending the margin.

    Left edge: the first sampled column where nearly the whole comb carries
    emphasized (staff-line) ink. Right edge: the rightmost near-full-height
    vertical in the band -- the final barline -- which anchors the true end even
    where the binding fade has killed the horizontal emphasis; we fall back to the
    last inked column when no clear barline is present."""
    n = L.shape[0]
    H = horiz.shape[0]
    cov = np.zeros(len(xcs))
    for j, xc in enumerate(xcs):
        cnt = 0
        for ki in range(n):
            y = int(L[ki, j])
            if 0 <= y < H and horiz[max(0, y - 2):y + 3, xc].any():
                cnt += 1
        cov[j] = cnt
    inked = np.where(cov >= n - 1)[0]
    if len(inked) == 0:
        return 0, len(xcs) - 1
    jL, jR = int(inked[0]), int(inked[-1])
    band_h = ybot - ytop
    yb0, yb1 = max(0, ytop), min(raw.shape[0], ybot)
    colsum = raw[yb0:yb1].sum(0)                      # inked rows per column
    bar = np.where(colsum > 0.55 * band_h)[0]         # full-height verticals = barlines
    if len(bar):
        jbar = int(np.argmin(np.abs(xcs - bar[-1])))
        jR = max(jR, jbar)                            # extend to the final barline
    return jL, jR


def _trace_staff_lines(gray: np.ndarray):
    """Trace every staff line of every system across the full page width.

    Returns ``(xcs, systems_lines, space)`` where ``systems_lines`` is a list of
    ``(ytop, ybot, L)`` and ``L`` is an ``(nlines, len(xcs))`` array of the y
    position of each staff line at each sampled column.

    Each line is followed as a ridge: seeded from the staff-line peaks at the
    (reliable) page centre, then tracked column by column within a half-space
    window. Ridge-following stays locked on the real line through notes and the
    crease fade far more robustly than global comb correlation, which is what
    lets us iron the binding-side line *ends* flat rather than just translating a
    rigid comb. Per-column sort forbids traced lines from crossing.
    """
    h, w = gray.shape
    bw = (gray < 150).astype(np.uint8) * 255
    thickness, space = _staff_metrics(bw)
    emph = _emphasize(bw, thickness)
    horiz = (emph > 0).astype(np.uint8)
    systems = _find_systems(emph, space)
    xcs = np.arange(0, w, 6)
    # Raw threshold as a fallback signal. Near the binding the staff lines curl
    # and foreshorten, and the horizontal-opening emphasis (which needs a long
    # straight run) stops catching them -- so a comb-following trace would go
    # blind exactly where the bend is worst and hold its last value flat, which
    # then makes the straightener over/under-correct the very edge. Where the
    # emphasized signal is missing we fall back to the raw ink nearest the line
    # so the trace keeps following the real (sloped) line to the page edge.
    raw = (bw > 0).astype(np.uint8)
    win = max(3, int(round(space * 0.45)))
    slope_cap = 0.5  # px of vertical move per px of x -- bounds runaway extrapolation

    def _measure(pred: np.ndarray, xc: int):
        """Locate ink near each line's predicted y at column ``xc``.

        Returns ``(ys, found)`` where ``ys[k]`` is the observed y of line ``k``
        (NaN where blind). Emphasized (horizontally-extended) ink is preferred;
        raw ink nearest the prediction is the fallback where the binding fade
        kills the emphasis. Note that both signals happily latch onto a tie,
        slur or the terminal barline that happens to pass through the search
        window -- rejecting those is the job of the coupled comb fit below, not
        this per-line probe."""
        ys = np.full(pred.shape[0], np.nan, np.float32)
        for k in range(pred.shape[0]):
            pk = float(pred[k])
            lo = max(0, int(pk - win)); hi = int(pk + win)
            seg = np.where(horiz[lo:hi, xc] > 0)[0]
            if len(seg):
                ys[k] = lo + float(np.median(seg))
            else:
                seg = np.where(raw[lo:hi, xc] > 0)[0]
                if len(seg):
                    yy = lo + seg
                    ys[k] = float(yy[int(np.argmin(np.abs(yy - pk)))])
        return ys, ~np.isnan(ys)

    out = []
    extents = []
    for (ytop, ybot) in systems:
        cx0, cx1 = int(w * 0.40), int(w * 0.60)
        prof = horiz[ytop:ybot, cx0:cx1].sum(1).astype(np.float32)
        if prof.max() <= 0:
            continue
        rows = sorted(_seed_line_rows(prof, space))
        comb5 = _fit_uniform_comb(prof, space, thickness)
        if comb5 is not None:
            # A confident five-line lock overrides raw peak counts (which miss a
            # faint line or catch a slur/ledger and kink the warp); otherwise keep
            # the peak seeding so genuine short percussion staves still trace.
            seeds = np.array([ytop + r for r in comb5], np.float32)
        else:
            seeds = np.array([ytop + r for r in rows], np.float32)
            if len(seeds) < 3:
                continue
            seeds = _snap_comb(seeds, np.array([prof[r] for r in rows], np.float32))
        n = len(seeds)
        # Fixed relative geometry of the staff: the lines translate and slowly
        # scale together, they never re-order or change their *relative* spacing
        # abruptly. offset encodes the seed layout (so a grand staff's wide inner
        # gap is preserved), scale is the per-column spacing multiplier.
        offset = seeds - seeds.mean()
        oo = float((offset * offset).sum())
        lam = 4.0 * oo          # stiff ridge: spacing may drift only slowly
        tol = 0.4 * space       # residual beyond this => line jumped off the comb
        seedx = (cx0 + cx1) // 2
        ci = int(np.argmin(np.abs(xcs - seedx)))
        L = np.full((n, len(xcs)), np.nan, np.float32)
        L[:, ci] = seeds

        def _comb_fit(ys, found, c0, s0):
            """Robustly fit (center, scale) to the found line observations under
            ``y_k ~= center + scale*offset_k``, with a stiff ridge holding scale
            near ``s0``. Two IRLS passes reject the tie/note/barline outliers
            (large residual => tiny weight). Returns (center, scale)."""
            c, s = c0, s0
            for _ in range(2):
                r = ys - (c + s * offset)
                wg = np.where(found, 1.0 / (1.0 + (r / tol) ** 2), 0.0)
                if wg.sum() < 1e-6:
                    return c, s
                Sw = wg.sum()
                Swo = float((wg * offset).sum())
                Swy = float((wg * np.nan_to_num(ys)).sum())
                Swoo = float((wg * offset * offset).sum())
                Swoy = float((wg * offset * np.nan_to_num(ys)).sum())
                den = (Swoo - Swo * Swo / Sw) + lam
                s = (Swoy - Swo * Swy / Sw + lam * s0) / den if abs(den) > 1e-9 else s0
                s = float(np.clip(s, 0.6, 1.6))
                c = (Swy - s * Swo) / Sw
            return c, s

        def _sweep(idxs, center, scale, cslope, start_j):
            blind = 0
            prev_x = xcs[start_j]
            for j in idxs:
                dx = float(xcs[j] - prev_x); prev_x = xcs[j]
                pred_c = center + cslope * dx
                pred = pred_c + scale * offset
                ys, found = _measure(pred, int(xcs[j]))
                if found.sum() >= 2:
                    c, s = _comb_fit(ys, found, pred_c, scale)
                    cslope = float(np.clip(0.5 * cslope + 0.5 * (c - center) / max(abs(dx), 1e-6),
                                           -slope_cap, slope_cap))
                    center = c
                    scale = 0.85 * scale + 0.15 * s   # spacing eases, never jumps
                    blind = 0
                else:
                    center = pred_c                    # blind: coast the comb
                    blind += 1
                    if blind > 6:
                        cslope *= 0.7
                L[:, j] = center + scale * offset

        # A rough left sweep from the (robust, page-centre) seed establishes the
        # comb at the far-left column, then a single continuous left-to-right pass
        # re-traces the whole width. Tracing in one direction leaves the only seam
        # at column 0 (the blank margin, later truncated) instead of at page
        # centre, where two independent half-sweeps used to meet with a small
        # slope mismatch and leave a visible kink through the music.
        _sweep(range(ci - 1, -1, -1), float(seeds.mean()), 1.0, 0.0, ci)
        c0 = float(np.nanmean(L[:, 0]))
        denom = float((offset * offset).sum())
        s0 = float(np.clip(((L[:, 0] - c0) * offset).sum() / denom, 0.6, 1.6)) if denom > 0 else 1.0
        _sweep(range(1, len(xcs)), c0, s0, 0.0, 0)
        L[:, 0] = c0 + s0 * offset

        k = 7  # light along-x smoothing kills jitter, keeps the real bend
        for ki in range(L.shape[0]):
            L[ki] = np.convolve(np.pad(L[ki], k // 2, mode="edge"), np.ones(k) / k, mode="valid")
        jL, jR = _staff_hextent(horiz, raw, xcs, L, ytop, ybot)
        out.append((ytop, ybot, L))
        extents.append((jL, jR))

    # Truncate at the staff's real horizontal extent so the trace cannot fan into
    # the blank margin past the first/last barline and drag the warp with it. The
    # page block is one continuous surface: every system starts and ends at the
    # same binding x, so we share a single robust extent across systems rather
    # than each system's own. This rescues the outermost systems, whose own
    # final-barline is itself curled/short and under-detects -- clamping them
    # individually would cut away the worst of the curl and leave it uncorrected.
    #
    # Past the extent we do NOT freeze the comb flat (that leaves the binding
    # gutter's curl uncorrected -- a visible hook at the very edge). The staff
    # rules on into the gutter but curls too steeply for the trace to follow
    # reliably, so instead we *extrapolate* the comb along the curl's own slope,
    # measured from a reliable window just inside the extent. The gutter is
    # effectively blank past the final barline, so continuing the established
    # smooth curvature straightens those ruled ends without trusting a fanning
    # trace. Slope is bounded and the lines keep their relative spacing (a single
    # comb slope), so extrapolation can never fan or fold.
    if extents:
        jL = int(np.median([e[0] for e in extents]))
        jR = int(np.median([e[1] for e in extents]))
        win = max(2, int(round(space * 8 / 6)))   # ~8 staff-spaces of columns
        for (_yt, _yb, L) in out:
            if jR > jL:
                center = L.mean(axis=0)
                # right slope from [jR-win, jR]; least-squares over the window
                a = max(jL, jR - win)
                if jR > a:
                    xr = xcs[a:jR + 1].astype(np.float64)
                    sl = np.polyfit(xr, center[a:jR + 1], 1)[0]
                    sl = float(np.clip(sl, -0.5, 0.5))
                    for j in range(jR + 1, len(xcs)):
                        L[:, j] = L[:, jR] + sl * (xcs[j] - xcs[jR])
                # left slope from [jL, jL+win]
                b = min(jR, jL + win)
                if b > jL:
                    xl = xcs[jL:b + 1].astype(np.float64)
                    sl = np.polyfit(xl, center[jL:b + 1], 1)[0]
                    sl = float(np.clip(sl, -0.5, 0.5))
                    for j in range(0, jL):
                        L[:, j] = L[:, jL] + sl * (xcs[j] - xcs[jL])
    return xcs, out, space


def _staff_guided_displacement(gray: np.ndarray, max_disp_factor: float = 8.0):
    """Vertical dewarp that irons each staff *comb* flat, per column.

    Returns ``((map_x, map_y), info)`` or ``(None, info)``. For every system we
    trace its staff lines and take their per-column median as the comb centre,
    then flatten that centre curve to its mean height. The correction is applied
    as a per-column RIGID vertical shift of the whole comb -- every row in a
    system's band at a given column moves by the same offset -- held constant
    across the band and blended linearly through the inter-system gaps.

    Because the shift never varies *within* a band, staff-line spacing is exactly
    preserved: the comb can bend or curl (crease ends), and we straighten that
    bend, but lines can never be squeezed together or stretched apart. That is
    what a per-line re-spacing warp got wrong -- where lines merged in the source
    near the binding it smeared them into a solid black bar; a rigid comb shift
    cannot. The offset is bounded and smooth, so map_y stays monotonic.
    """
    h, w = gray.shape
    xcs, sys_lines, space = _trace_staff_lines(gray)
    info = dict(reason="ok", nsys=len(sys_lines), space=int(space), method="guided")
    if len(sys_lines) < 2:
        return None, dict(info, reason="too few traced systems")

    xs_all = np.arange(w)
    tops: list[float] = []
    bots: list[float] = []
    offsets: list[np.ndarray] = []  # per-system per-column comb-centre offset
    for (ytop, ybot, L) in sys_lines:
        # Comb centre per column = median across the traced lines (robust to a
        # single mistraced line); interpolate over columns that were traced.
        center = np.nanmedian(L, axis=0)
        good = ~np.isnan(center)
        if good.sum() < 4:
            continue
        cf = np.interp(xs_all, xcs[good], center[good],
                       left=center[good][0], right=center[good][-1]).astype(np.float32)
        # light smoothing to avoid injecting per-column jitter
        k = 31
        cf = np.convolve(np.pad(cf, k // 2, mode="edge"), np.ones(k) / k, mode="valid").astype(np.float32)
        tops.append(float(ytop)); bots.append(float(ybot))
        offsets.append(cf - float(np.mean(cf)))  # flatten centre to its mean
    if len(offsets) < 2:
        return None, dict(info, reason="too few traced combs")

    tops = np.array(tops); bots = np.array(bots)
    order = np.argsort(tops)
    tops = tops[order]; bots = bots[order]
    offsets = [offsets[i] for i in order]
    S = len(tops)

    # No cross-system smoothing: with the coupled-comb trace (constant spacing)
    # and a shared page extent, each system now flattens to ~1-2px on its own,
    # and the shared extent already keeps the systems mutually consistent at the
    # binding. Relaxing offsets toward neighbours here only fought that reliable
    # per-system fit -- it *raised* the median residual several-fold -- so the
    # continuity is now enforced by geometry (shared extent) rather than by
    # blurring the warp field across systems.

    # Control points at each band's edges (strictly increasing y); the per-column
    # offset is constant across a band and blends linearly through the gaps.
    yk = np.empty(2 * S, np.float64)
    yk[0::2] = tops; yk[1::2] = bots
    yk = np.maximum.accumulate(yk) + np.arange(2 * S) * 1e-3
    dk = np.empty((2 * S, w), np.float32)
    dk[0::2] = offsets
    dk[1::2] = offsets

    yout = np.arange(h, dtype=np.float64)
    ki = np.clip(np.searchsorted(yk, yout) - 1, 0, 2 * S - 2)
    yk0 = yk[ki]; yk1 = yk[ki + 1]
    frac = ((yout - yk0) / np.maximum(yk1 - yk0, 1e-6))[:, None]
    off = dk[ki] * (1 - frac) + dk[ki + 1] * frac        # (h, w) offset field
    above = yout < yk[0]; below = yout > yk[-1]
    if above.any():
        off[above] = dk[0][None, :]                      # rigid hold above first band
    if below.any():
        off[below] = dk[-1][None, :]                     # rigid hold below last band
    map_y = (yout[:, None] + off).astype(np.float32)
    # Offsets are bounded and smooth, but guard monotonicity defensively so a
    # steep gap transition can never fold.
    np.maximum.accumulate(map_y, axis=0, out=map_y)

    maxdisp = float(np.abs(map_y - yout[:, None]).max())
    info["maxdisp"] = maxdisp
    if maxdisp > space * max_disp_factor:
        return None, dict(info, reason=f"maxdisp {maxdisp:.0f} exceeds safety limit")
    map_x = np.tile(np.arange(w, dtype=np.float32)[None, :], (h, 1))
    return (map_x, map_y), info


def _staff_flatness(gray: np.ndarray) -> float:
    """Median per-system comb-centre waviness (peak-to-peak px) across the staff.

    Lower is flatter. Used to pick the better of the guided and rigid warps so a
    page can never be made worse: whichever remaps to flatter staff lines wins.

    Measured off the coupled-comb trace (the same reliable estimator the guided
    warp uses), not an independent re-seed of a single line: the old metric
    tracked only the emphasized *first* line, which is exactly the signal that
    breaks down at the binding, so it under-credited a warp that had genuinely
    ironed the tail flat. The trace is horizontally truncated to the real staff
    extent (held flat in the margins), so the peak-to-peak reflects only the
    staff itself and includes the binding endpoints honestly. A robust 2..98
    percentile span rejects a single mistraced column."""
    xcs, sys_lines, _ = _trace_staff_lines(gray)
    devs = []
    for (_ytop, _ybot, L) in sys_lines:
        center = np.nanmean(L, axis=0)
        good = center[~np.isnan(center)]
        if len(good) < 8:
            continue
        lo, hi = np.percentile(good, [2, 98])
        devs.append(float(hi - lo))
    return float(np.median(devs)) if devs else float("inf")


def straighten_staves(gray: np.ndarray, max_disp_factor: float = 6.0,
                      passes: int = 5, refine_min_px: float = 1.5):
    """Flatten wavy/skewed staff lines. Returns (output_gray, info).

    Builds two candidates -- a staff-line-guided warp that irons each comb flat
    per column (excellent on curled binding-side ends) and the rigid per-system
    correlation warp (robust on clean flat scans) -- and keeps whichever leaves
    the staff lines flatter. Guaranteed never to make a page worse than the
    rigid result: if the guided trace misbehaves (unusual editions, dense pages)
    its output measures wavier and is discarded.
    """
    # Try the (fast, vectorized) guided warp first. If it lands the staff lines
    # convincingly flat, accept it without paying for the slower rigid pass; only
    # when it is marginal or unavailable do we compute rigid and keep the flatter.
    guided, ginfo = _staff_guided_displacement(gray)
    guided_out = None
    if guided is not None:
        guided_out = cv2.remap(gray, guided[0], guided[1], cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)
        # Refinement pass: the first warp flattens the bulk, but a little residual
        # survives wherever the trace lagged the steepest binding curl. Re-tracing
        # the now near-flat page measures only that small leftover error (on an
        # easy, mostly-straight line) and irons it out. Kept only if it actually
        # helps, so a already-perfect page is never disturbed.
        gflat = _staff_flatness(guided_out)
        r_guided, rginfo = _staff_guided_displacement(guided_out)
        if r_guided is not None:
            refined = cv2.remap(guided_out, r_guided[0], r_guided[1],
                                cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            rflat = _staff_flatness(refined)
            if rflat + 0.05 < gflat:
                guided_out, gflat = refined, rflat
                ginfo["passes"] = 2
        ginfo["flatness"] = gflat
        ginfo["method"] = "guided"
        if gflat <= _GUIDED_ACCEPT_PX:
            ginfo["applied"] = True
            return guided_out, ginfo

    rigid_out, rinfo = _straighten_staves_rigid(gray, max_disp_factor, passes, refine_min_px)
    rigid_flat = _staff_flatness(rigid_out)
    rinfo["method"] = "rigid"

    # Keep whichever leaves the staff lines flatter; the guided warp must clear
    # the rigid result by a margin so we never trade robustness for a noisy tie.
    if guided_out is not None and ginfo["flatness"] + 0.5 < rigid_flat:
        ginfo["applied"] = True
        ginfo["rigid_flatness"] = rigid_flat
        return guided_out, ginfo
    rinfo["applied"] = True
    rinfo["flatness"] = rigid_flat
    if guided_out is not None:
        rinfo["guided_flatness"] = ginfo["flatness"]
    return rigid_out, rinfo


def _straighten_staves_rigid(gray: np.ndarray, max_disp_factor: float = 6.0,
                             passes: int = 5, refine_min_px: float = 1.5):
    """Rigid per-system straightener (legacy fallback). Returns (output, info).

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


def _vertical_strokes(gray: np.ndarray, space: int, thickness: int):
    """Return (y_centers, leans, heights) for tall, thin vertical strokes.

    Bar lines and note stems are drawn perpendicular to the staff, so in a truly
    rectified page they are vertical. Their measured lean (dx/dy) is therefore a
    direct read-out of any residual horizontal shear -- a signal far more robust
    than the page margin (which rehearsal marks and indents corrupt).

    A tall 1-D opening *locates* vertical ink, but its slope must not be measured
    on the opened result: a vertical structuring element clips the leaning stroke
    back toward vertical and roughly halves the apparent lean. So we use the
    opening only to find each stroke's bounding box, then fit its slope on the
    *raw* ink via per-row centroids (skipping rows where a neighbouring blob
    bleeds into the window), which recovers the true lean.
    """
    ink = (gray < 100).astype(np.uint8)
    vlen = max(21, int(round(space * 2.5)))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vlen))
    vert = cv2.morphologyEx(ink, cv2.MORPH_OPEN, vk)
    n, lab, st, _ = cv2.connectedComponentsWithStats(vert, 8)
    min_h = max(vlen, int(round(space * 3)))
    max_w = max(6, thickness * 3)
    pad = 3
    ys_c: list[float] = []
    leans: list[float] = []
    heights: list[float] = []
    for i in range(1, n):
        x, y, ww, hh, area = st[i]
        if hh < min_h or ww > max_w or hh < ww * 3:
            continue
        x0 = max(0, x - pad)
        sub = ink[y:y + hh, x0:x + ww + pad]
        rys: list[float] = []
        rxs: list[float] = []
        wlim = ww + 2 * pad
        for r in range(hh):
            cx = np.where(sub[r] > 0)[0]
            if len(cx) == 0 or len(cx) > wlim:
                continue  # empty row, or a neighbouring blob bled into the window
            rys.append(float(r))
            rxs.append(float(cx.mean()) + x0)
        if len(rys) < min_h * 0.7:
            continue
        m = np.polyfit(np.asarray(rys), np.asarray(rxs), 1)[0]
        ys_c.append(float(y + hh / 2.0))
        leans.append(float(m))
        heights.append(float(hh))
    return np.array(ys_c), np.array(leans), np.array(heights)


def deskew_barlines(gray: np.ndarray, max_shift_factor: float = 8.0,
                    min_lean_deg: float = 0.12):
    """Remove a residual horizontal shear so bar lines/stems read as vertical.

    Returns ``(output_gray, info)``. The straightener flattens staff lines
    (horizontals) but a page can still carry a horizontal shear -- vertical
    strokes leaning, and that lean drifting down the page -- which the eye reads
    as a skew even when the staves are level. We model it as a shift field
    ``g(y)`` that is the same at every x (so horizontal staff lines only slide,
    staying horizontal) and whose derivative is the local vertical-stroke lean.

    We measure the lean of many bar lines and stems, robustly fit ``lean(y)`` as
    a line in y (constant + gradient, covering both a uniform shear and one that
    rotates through the page), integrate to ``g(y)``, and remap. Bounded and
    guarded: needs enough well-spread strokes, ignores a negligible lean, and
    caps the applied shift so a mis-measurement cannot warp the page.
    """
    h, w = gray.shape
    bw = (gray < 150).astype(np.uint8) * 255
    thickness, space = _staff_metrics(bw)
    ys, leans, heights = _vertical_strokes(gray, space, thickness)
    info: dict = dict(reason="ok", nstrokes=int(len(ys)), space=int(space))
    if len(ys) < 20:
        info.update(applied=False, reason="too few vertical strokes")
        return gray, info

    # Robust inlier set: drop italic/outlier strokes far from the median lean.
    med = float(np.median(leans))
    mad = float(np.median(np.abs(leans - med))) + 1e-6
    keep = np.abs(leans - med) < 4.0 * mad
    if keep.sum() < 15 or (ys[keep].max() - ys[keep].min()) < 0.3 * h:
        info.update(applied=False, reason="strokes not well spread")
        return gray, info

    yk = ys[keep]
    lk = leans[keep]
    wk = heights[keep]
    yc = float(np.average(yk, weights=wk))
    # Weighted least-squares fit lean(y) = a + b*(y - yc).
    dy = yk - yc
    W = wk
    S0 = W.sum(); S1 = (W * dy).sum(); S2 = (W * dy * dy).sum()
    T0 = (W * lk).sum(); T1 = (W * dy * lk).sum()
    det = S0 * S2 - S1 * S1
    if abs(det) < 1e-9:
        a = T0 / S0; b = 0.0
    else:
        a = (T0 * S2 - T1 * S1) / det
        b = (S0 * T1 - S1 * T0) / det

    info["lean_center_deg"] = float(np.degrees(np.arctan(a)))
    info["lean_gradient_deg_per_page"] = float(np.degrees(np.arctan(b * h)))

    ys_all = np.arange(h, dtype=np.float64)
    d = ys_all - yc
    # g(y) = integral of lean(y) dy = a*(y-yc) + b*(y-yc)^2/2, centered at yc.
    g = a * d + 0.5 * b * d * d
    max_shift = float(np.abs(g).max())
    info["max_shift"] = max_shift

    # A lean this small is within engraving/measurement noise -> leave it.
    corner_lean_deg = abs(np.degrees(np.arctan(a))) + abs(np.degrees(np.arctan(b * h / 2.0)))
    if corner_lean_deg < min_lean_deg:
        info.update(applied=False, reason="lean below threshold")
        return gray, info
    if max_shift > max_shift_factor * space or max_shift > 0.05 * w:
        info.update(applied=False, reason=f"shift {max_shift:.0f}px exceeds safety limit")
        return gray, info

    map_x = (np.arange(w, dtype=np.float32)[None, :] + g[:, None].astype(np.float32))
    map_y = np.repeat(ys_all.astype(np.float32)[:, None], w, axis=1)
    out = cv2.remap(gray, map_x, map_y, cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    info.update(applied=True)
    return out, info


def _staff_hextents(gray: np.ndarray, min_lines: int = 3):
    """Per-system staff-line left AND right x extents, plus (ytop, ybot) bands.

    Like ``_staff_left_margins`` but also returns the rightmost column carrying
    at least ``min_lines`` staff lines. Staff lines are drawn edge-to-edge across
    a system regardless of the notes in it, so these extents trace the true music
    block width -- robust to rehearsal marks, measure numbers and marginalia that
    corrupt a naive ink bounding box.
    """
    bw = (gray < 150).astype(np.uint8) * 255
    thickness, space = _staff_metrics(bw)
    horiz = _emphasize(bw, thickness)
    systems = _find_systems(horiz, space)
    lefts: list[float] = []
    rights: list[float] = []
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
        lefts.append(float(xs.min()))
        rights.append(float(xs.max()))
        bands.append((ytop, ybot))
    return np.array(lefts, np.float32), np.array(rights, np.float32), bands, space


def _system_edges(gray: np.ndarray, bands, space: int):
    """Robust per-system (left, right) content edges, immune to binding seams.

    A half-spread scan often leaves a faint vertical seam (the neighbour page /
    fold shadow) near the outer edge, separated from the real music by a wide
    band of whitespace. Taken as the rightmost inked column it masquerades as the
    system's right edge, corrupting margin alignment and centring. We instead
    split each system's inked columns into segments at large horizontal gaps and
    peel off any *thin* trailing/leading segment that sits beyond such a gap --
    that is the seam. A real final barline hugs its content (small gap) and is
    kept; a resumed passage after a multi-measure rest is wide and is kept.
    """
    ink = (gray < 100)
    gap_thr = int(3 * space)
    thin = int(1.5 * space)
    lefts = []
    rights = []
    for (ytop, ybot) in bands:
        cols = np.where(ink[int(ytop):int(ybot)].sum(0) > 1)[0]
        if len(cols) == 0:
            lefts.append(np.nan); rights.append(np.nan); continue
        brk = np.where(np.diff(cols) > gap_thr)[0]
        starts = [cols[0]] + [cols[b + 1] for b in brk]
        ends = [cols[b] for b in brk] + [cols[-1]]
        # peel thin isolated seam segments off the right, then the left
        while len(ends) > 1 and (ends[-1] - starts[-1]) < thin:
            ends.pop(); starts.pop()
        while len(ends) > 1 and (ends[0] - starts[0]) < thin:
            ends.pop(0); starts.pop(0)
        lefts.append(float(starts[0]))
        rights.append(float(ends[-1]))
    return np.array(lefts, np.float64), np.array(rights, np.float64)


def center_content(gray: np.ndarray, min_imbalance_px: float = 24.0,
                   max_shift_frac: float = 0.06):
    """Balance the music block's left/right margins on the page. Returns
    ``(output_gray, info)``.

    Rectification maps the detected page edges to the output rectangle, so any
    asymmetry in where those edges landed -- most often a booklet crease clipped
    a hair inside one margin -- leaves the music sitting off-centre. We measure
    the staff-line block (median per-system left/right extents, robust to
    indents and marginalia) and translate the whole page horizontally so the two
    margins match. Bounded and guarded: skips a already-balanced page, caps the
    shift, and never pushes content past an edge.
    """
    h, w = gray.shape
    lefts, rights, bands, space = _staff_hextents(gray)
    info: dict = dict(reason="ok", nsys=int(len(lefts)))
    if len(lefts) < 3:
        info.update(applied=False, reason="too few systems")
        return gray, info

    # Use seam-immune content edges: the staff-line block right is under-read
    # where the line fades at the binding, and the raw ink extent is inflated by
    # the neighbour-page seam near the outer edge -- either skews the centring.
    c_lefts, c_rights = _system_edges(gray, bands, int(space))
    block_left = float(np.nanmedian(c_lefts))
    block_right = float(np.nanmedian(c_rights))
    left_margin = block_left
    right_margin = (w - 1) - block_right
    info["left_margin"] = left_margin
    info["right_margin"] = right_margin
    shift = (right_margin - left_margin) / 2.0  # >0 moves content right
    info["imbalance_px"] = float(left_margin - right_margin)

    if abs(left_margin - right_margin) < min_imbalance_px:
        info.update(applied=False, reason="already centered")
        return gray, info
    cap = max_shift_frac * w
    # shift>0 moves right (shrinks right margin); shift<0 moves left (shrinks
    # left margin). Keep at least a 2px margin on the shrinking side, and cap.
    lo = max(-(left_margin - 2.0), -cap)
    hi = min(right_margin - 2.0, cap)
    shift = float(np.clip(shift, lo, hi))
    if abs(shift) < 1.0:
        info.update(applied=False, reason="shift negligible")
        return gray, info
    info["shift"] = shift

    map_x = (np.arange(w, dtype=np.float32)[None, :] - np.float32(shift))
    map_x = np.repeat(map_x, h, axis=0)
    map_y = np.repeat(np.arange(h, dtype=np.float32)[:, None], w, axis=1)
    out = cv2.remap(gray, map_x, map_y, cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    info.update(applied=True)
    return out, info


def align_right_margin(gray: np.ndarray, min_dev_px: float = 10.0,
                       max_stretch: float = 0.08):
    """Square up the crease-side (right) staff-line margin. Returns
    ``(output_gray, info)``.

    Photographing a bound book compresses the page as it curves into the spine,
    so on a verso capture each system's right (binding-side) end lands at a
    slightly different x -- the right edge "dances" system to system even after
    the staves are level. Staff lines are drawn to a common right margin, so we
    take that as ground truth: for each system we horizontally scale about its
    own (already de-drifted) left edge so its right staff extent reaches a common
    target R. Scale is constant within a staff band (bar lines stay vertical,
    staves stay horizontal) and blended through the gaps.

    Left edges -- and any intentional indents living in them -- are the scaling
    anchor and are never moved. On a page whose right margin is already
    consistent (recto/flat scans) the per-system deviation is tiny and this
    no-ops. Bounded: needs enough systems and caps the per-system stretch.
    """
    h, w = gray.shape
    lefts, rights, bands, space = _staff_hextents(gray)
    info: dict = dict(reason="ok", nsys=int(len(lefts)))
    if len(lefts) < 4:
        info.update(applied=False, reason="too few systems")
        return gray, info

    # Re-measure each system's right edge from *content*, seam-immune. The staff
    # line fades where it curves into the binding (under-reads the crease-most
    # systems), while the raw ink extent is inflated by the neighbour-page seam
    # near the outer edge -- both corrupt the alignment. `_system_edges` trims the
    # thin seam beyond a wide whitespace gap and returns the true content edge.
    c_lefts, c_rights = _system_edges(gray, bands, int(space))
    keep = ~np.isnan(c_rights)
    if keep.sum() < 4:
        info.update(applied=False, reason="too few measurable systems")
        return gray, info
    lefts = lefts.astype(np.float64)
    lefts[keep] = c_lefts[keep]
    rights = np.where(keep, c_rights, rights.astype(np.float64))

    # Target a low percentile of the right edges and only ever pull wide systems
    # *inward*. Stretching a system outward would push content past the page
    # margin (top systems overflowing). Compression-only alignment squares the
    # margin without ever driving content off the page.
    target_r = float(np.percentile(rights, 30))
    dev = float(np.std(rights))
    info["right_std_before"] = dev
    info["target_r"] = target_r
    if float(np.max(rights) - target_r) < min_dev_px:
        info.update(applied=False, reason="right margin already square")
        return gray, info

    # Per-system horizontal affine map_x = alpha + beta * x, anchored at left_i so
    # the right extent right_i maps to target_r:
    #   beta_i  = (right_i - left_i) / (target_r - left_i)   (input span / output span)
    #   alpha_i = left_i * (1 - beta_i)
    order = np.argsort([b[0] for b in bands])
    tops = np.array([bands[i][0] for i in order], np.float32)
    bots = np.array([bands[i][1] for i in order], np.float32)
    li = lefts[order]
    ri = rights[order]
    denom = np.maximum(target_r - li, 1.0)
    beta = (ri - li) / denom
    # Only compress (beta >= 1): a system wider than target is pulled in; a
    # narrower one is left exactly as-is rather than stretched outward.
    beta = np.clip(beta, 1.0, 1.0 + max_stretch)
    alpha = li * (1.0 - beta)


    # Constant within each band, linearly blended across gaps (mirror of the
    # vertical straightener / left de-drift control-point scheme).
    yk = np.empty(2 * len(tops), np.float32)
    yk[0::2] = tops
    yk[1::2] = bots
    yk = np.maximum.accumulate(yk) + np.arange(len(yk)) * 1e-3
    ak = np.empty(2 * len(tops), np.float32); ak[0::2] = alpha; ak[1::2] = alpha
    bk = np.empty(2 * len(tops), np.float32); bk[0::2] = beta;  bk[1::2] = beta
    ys = np.arange(h, dtype=np.float32)
    a_row = np.interp(ys, yk, ak, left=ak[0], right=ak[-1]).astype(np.float32)
    b_row = np.interp(ys, yk, bk, left=bk[0], right=bk[-1]).astype(np.float32)

    xs = np.arange(w, dtype=np.float32)[None, :]
    map_x = (a_row[:, None] + b_row[:, None] * xs).astype(np.float32)
    map_y = np.repeat(ys[:, None], w, axis=1)
    out = cv2.remap(gray, map_x, map_y, cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    info.update(applied=True, max_stretch=float(np.abs(beta - 1.0).max()))
    return out, info
