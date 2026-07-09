# Converter design (`flat_scan.py`, `flatscan_dewarp.py`)

The engine turns a casual phone photo of a printed part — angled, unevenly lit,
curling, possibly a half-spread of a bound book with the facing page bleeding in
— into a crisp, flat, correctly-sized PDF.

## Guiding principles

1. **Never lose the musician's content.** Dropping a note, barline, clef, or
   dynamic is a hard failure; a little cosmetic bleed or a soft shadow is not.
   Every aggressive step (crease clip, ink threshold, straightening warp) is
   gated so it cannot amputate real content. This is the invariant everything
   else defers to.
2. **Staff lines must read ruler-straight to the ends.** Residual bends/slopes at
   system ends (especially the binding side) are confusing to play from and are
   treated as defects, not polish.
3. **Stay assumption-light and edition-agnostic.** Avoid hard-coding "what the
   page should look like" (dimensions, margins, number of systems). Prefer
   signals derived from the page itself so the pipeline generalizes across parts
   and publishers.
4. **Fail safe, not clever.** When a detector is unsure, degrade to leaving the
   page whole rather than risk a destructive edit.

## Pipeline overview

Per page (intermediate artifacts dumped with `--debug` as `NN_*` PNGs):

```
render → segment → [booklet: detect_crease → clip_mask_at_crease]
       → rectify_boundary_coons → clean_ink
       → [straighten: straighten_staves → deskew_barlines
                    → align_system_margins → align_right_margin → center_content]
       → assemble (size, parity pad, write PDF)
```

## Segmentation

Illumination-normalized threshold to separate paper from background under uneven
light (wood grain, shadows, adjacent surfaces), then: sever thin bridges to
neighbouring surfaces → keep the largest component → fill holes → repair boundary
notches (a finger or shadow can bite a notch out of the paper edge that would
otherwise propagate into the rectified interior).

## Rectification — Coons boundary patch

We rectify from the **four page-edge curves**, not an interior grid. The corners
give four boundary edges; a Coons patch maps the warped quad to a flat rectangle.
Rationale:

- The page boundary is the most reliable thing we can find — high contrast
  against the background, and it fully determines perspective + paper curl for a
  developable surface.
- No dependence on interior content, so it works on sparse or blank pages.

Supporting decisions:

- **`inset_edges`** nudges every boundary edge inward a hair. Segmentation lands
  the boundary in the soft reflectance ramp at the physical paper edge, so the
  rectifier would otherwise sample a 1–3px sliver of dark surround that ink-clean
  turns into a hard black border. Insetting excludes it on all four sides without
  touching content (music never reaches the sheet edge).
- **Robust boundary smoothing** (`orient_edges` → `_robust_poly_curve`, degree-4
  with outlier rejection, endpoints pinned) so a segmentation notch/spike doesn't
  ripple into the interior as waviness.

### Known limitation: binding foreshortening

Near the spine the page is a cylinder turning away from the camera, so the staff
comb **compresses** horizontally (e.g. a 189px span → 125px). The Coons boundary
patch does not model this interior cylindrical distortion, and the rigid
straightener deliberately preserves staff spacing (see below), so it is not fully
corrected. It is currently accepted; a proper fix would model the interior
developable surface, not just the boundary. See open problems.

## Ink cleanup

Denoise → remove residual uneven lighting → threshold → perimeter despeckle →
composite to **soft grayscale** (anti-aliased alpha), gently normalizing ink
darkness (including flash glare) while preserving thin lines. Soft-gray rather
than hard binary keeps the output readable and print-faithful.

Notably, illumination normalization is what makes the final output shadow-free —
which is also why the fold shadow is *gone* by the clean stage. The seam detector
therefore works on the **pre-normalization** lightness (see below).

## Booklet / seam handling — the hard problem

Half-spread captures of a bound book show one page plus the fold into the spine,
and the facing page frequently bleeds across the fold. We must clip at the fold
so the neighbour is excluded and the binding edge is clean — **without** clipping
this page's own binding-side content (barlines on even pages; clefs, key
signatures, part-name boxes, system brackets on odd pages).

### Approaches, in order (this is the road map of what we learned)

1. **Fold-shadow "valley" filter (original, rejected).** Find the darkest
   vertical valley in the binding region. *Failed* because on a page that runs
   off-frame with no neighbour in view, the deepest valley is often this page's
   own stacked closing barlines or the shadow where dense notes end — so it
   clipped *inside* the music. Also had a dead-zone bug near the frame edge where
   run-off folds actually live.

2. **Content-edge hugging (current default).** Instead of finding the fold, find
   *this page's music edge* and clip just outside it:
   - **Black-tophat** isolates thin music strokes and suppresses the broad
     fold/spine shadow that would otherwise bridge this page to its neighbour.
   - Keep **wide, page-centred** connected components (the staves) → per-row music
     edge → low-order (deg ≤ 2) smooth curve just outside it.
   - **Content-protection clamp:** the crease may never cross the binding-side
     edge of this page's own ink (`own` mask = every substantial component that
     is *not* facing-page bleed). This preserves outliers the smooth fit would
     otherwise drop — a lone bottom-system barline, or an off-centre part-name box
     / bracket that juts toward the binding.
   - **Blank-page fallback (`_neighbour_bleed_crease`):** a blank verso has no
     music of its own, so the music-edge finds nothing. If a substantial,
     off-centre ink cluster sits toward the binding, it is facing-page bleed;
     clip at its page-ward edge so the blank page comes out clean.

   *Strength:* robust, never amputates content. *Weakness (important):* it
   assumes the neighbour is a **separable blob** — largely out of view or across
   a blank gutter. If the facing page is substantially in view, "this page's
   content edge" is not the fold and the heuristic strains. Each new page shape
   needed another sub-rule (blank verso vs title box vs recto bleed), which is a
   symptom of not modelling the fold itself.

3. **Hybrid seam detector (in progress — the intended root fix).** Actually find
   the physical fold, drop the "neighbour out of view" assumption, and keep the
   content edge only as a safety floor. Cues, fused:
   - **Shadow map = `normalized − raw` lightness.** The fold shadow is exactly
     what illumination-normalization removes, so this residual isolates it as a
     broad, full-height dark band with content mostly gone. *Prototype confirmed
     the signal is present.*
   - **Interior-valley test:** the seam must be a shadow ridge flanked by
     *brighter paper on both sides* — this rejects frame-edge vignetting (paper on
     one side, dark surround on the other), which a naïve argmax otherwise picks.
   - **Full-height continuity:** the fold shadow is present in the blank rows
     *between* systems; barlines are not — cleanly rejects content.
   - **Geometry cue:** the paper's top/bottom boundary curves kink at the spine,
     so the seam-x shows up as a curvature spike / notch independent of shadow and
     content.
   - **Content-edge floor (invariant):** clamp so the clip can never cross this
     page's own music, even if every cue misfires.

   Status: prototyped; the shadow cue localizes the fold on a title page and the
   failure mode (naïve argmax grabbing the mask boundary) is understood. Being
   built behind the regression harness.

### Geometry decides the binding side

Which L/R side is the binding is decided by **geometry, not guesswork**: a
half-spread always runs off the frame into the spine on one side (paper touches
the frame with no table margin). A full flat sheet in view (e.g. a title page) is
"floating" — table background on all four sides — and is **never** clipped, which
protects sheets whose content straddles a visible fold.

## Staff-line straightening (`--straighten`, `flatscan_dewarp.py`)

Coons fixes gross geometry from the boundary but knows nothing about the interior,
so gentle waviness/skew remains. For music we exploit that staff lines *should* be
straight, horizontal, parallel:

- Estimate staff thickness/spacing from vertical run-length histograms; detect
  **systems** (5-line combs) by horizontal-ink projection.
- Two candidate warps, keep whichever leaves staves flatter (`_staff_flatness`):
  - **Guided** (`_staff_guided_displacement`): trace each system's lines and iron
    the comb flat per column. Great on curled binding-side ends.
  - **Rigid** (`_straighten_staves_rigid`): per-system rigid vertical translation,
    blended across gaps, iterated a few passes. Robust on clean scans. **Staff
    spacing is preserved deliberately** — the shift is uniform per column so
    merged/close lines never smear; this is also why it can't undo binding
    *compression*.
- **Raw-threshold fallback in the tracer** (`_trace_staff_lines`): `_emphasize`
  (horizontal opening) drops sloped/curling lines near the binding, so the tracer
  would hold its last value flat and the correction would over/under-shoot. The
  fallback follows the real curling line into the binding using nearest raw ink to
  the prediction.
- Then `deskew_barlines`, `align_system_margins`, `align_right_margin` (square the
  crease-side margin), `center_content`.

### Known limitation: localized end-of-system bows

The method chooses **one** warp for the whole page by *average* flatness. A single
system that bows sharply near the bottom edge can survive because the page average
still looks acceptable. Fixing this well (per-system method choice / stronger
local flattening) is an open dewarp problem — tracked because ruler-straight ends
are a hard requirement.

## Assembly, batch, parity

- Size to the physical page (`--width-in/--height-in/--dpi`); optional trailing
  blank so the final count is even for double-sided printing (`--no-pad-even` to
  disable).
- `--starts-on-even` / `--starts-on-even-files` / `starts_on_even.txt` insert a
  leading blank for parts whose first scanned page is even, so page parity lines
  up across a set.
- **Batch mode**: a directory in → `<input>-Processed` out. Per-part page-number
  offsets (some parts have 2 cover pages, so printed page 1 = PDF page 3) are
  **not uniform even within an instrument family** and must be checked per part.
- **Combine** into named books is a separate concatenation step (grouping:
  Flute I/II/III, Oboe I/II/III, Clarinet I/II + Bass Cl, Bassoon I/II, Horn
  I–IV, Trumpet I/II/III, Low Brass = Trombones/Bass Tbn/Tuba, Timpani
  Percussion, then strings). Fixing a page splices into its book by index rather
  than regenerating the whole set.

## Regression safety

Iterative pipeline changes are validated with `tools/regression_check.py`, a
lightweight golden-thumbnail harness over a manifest that over-samples binding
geometries (run-off pages, real neighbour bleed, title pages, blank versos) plus
the committed `samples/russlan` reference. `capture` stores goldens from
known-good code; `check` re-runs and flags any page whose output drifted, so a
crease/seam/dewarp tweak can be shown to fix the target pages without regressing
the good ones.

## Open problems / long-term quality

- **Interior developable-surface model** to correct binding *compression*
  (foreshortening) that the boundary Coons patch and rigid straightener leave.
- **Per-system straightening decisions** so a single bowing end-of-system is
  ironed flat without the page average hiding it.
- **Finish the hybrid seam detector** and retire the content-edge heuristics to a
  pure safety floor.
- **Grow the "perfect" bucket** over time — the standing goal is fewer "ok" pages,
  more indistinguishable-from-a-real-scan pages.
