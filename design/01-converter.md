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

3. **Fold-shadow seam — for blank pages (shipped).** We *do* find the physical
   fold, but only where it is both findable and safe to clip at: a blank / near-
   blank page (a blank verso, a publisher "BLANK PAGE") that has no music of its
   own. There the content-edge has nothing to hug, and — critically — there is no
   content to distort, so clipping at the true fold is a clean win even when the
   facing page is substantially in view (dropping the "neighbour out of view"
   assumption for exactly the case that needed it). How the fold is found
   (`_fold_seam`):
   - **Shadow map = `normalized − raw` lightness.** The fold shadow is exactly
     what illumination-normalization removes, so this residual isolates it as a
     broad, full-height ridge with content gone. Printed ink is dark in *both*
     images and cancels, so barlines/clefs leave no ridge — the old valley
     detector's failure (latching onto this page's barlines) cannot recur.
   - **Innermost strong ridge, flanked on both sides.** The seam is the *first*
     shadow ridge crossing from the page into the gutter (a local maximum with
     lower shadow on both flanks). Taking the innermost — not the global maximum —
     is essential: when the neighbour is in view the mask runs to the frame, and
     the frame-edge vignette is often the *taller* ridge; the fold is the one
     nearer the content.
   - Per-band ridge → robust low-order curve fit → **prominence confidence gate**;
     below it we fall back to `_neighbour_bleed_crease` (clip the off-centre bleed
     cluster at its page-ward edge).

   > **Why not use the seam on music pages too?** We tried; the regression harness
   > caught it. On a page *with* music, clipping at the true fold drags the
   > maximally-foreshortened near-spine band (which the boundary Coons patch cannot
   > flatten) into the plate, warping the binding-side systems. Clipping at the
   > **content edge** instead excludes that band, and since the neighbour lives
   > *beyond* this page's music it is removed all the same. So: **content-edge for
   > music pages, fold-seam for blank pages.** Music-page output is byte-identical
   > with or without the seam code.

Remaining honest caveat: on a music page whose facing page is *so* far in view
that its content crosses this page's music edge, the content-edge rule would not
fully exclude it. Not yet observed in the corpus; if it arises, the fix is to
extend the seam (with the same near-spine caveat) to bound the music-page clip.

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
    the comb flat per column. Great on curled binding-side ends, and now the
    default winner on essentially every real page (accepted immediately when
    flatness ≤ `_GUIDED_ACCEPT_PX`, skipping the slower rigid candidate).
  - **Rigid** (`_straighten_staves_rigid`): per-system rigid vertical translation,
    blended across gaps, iterated a few passes. Robust fallback on scans where the
    trace is unclear. **Staff spacing is preserved deliberately** — the shift is
    uniform per column so merged/close lines never smear; this is also why it
    can't undo binding *compression*.

**The tracer is a coupled comb** (`_trace_staff_lines`). Instead of following each
of the five lines independently, all five share **one centre curve** and **one
slowly-varying spacing**: `y_k(x) = centre(x) + scale(x)·offset_k`, where
`offset_k` is the fixed relative geometry of the comb and `scale`/`centre` are
fit per column with stiff ridge regularization and IRLS outlier rejection. This is
what makes the trace robust: a line that jumps onto a tie, note head, slur, or
bar line becomes a *spacing outlier* and is overridden by the group rather than
dragging the warp off the staff. Supporting pieces:

- **`_snap_comb`** trims phantom endpoint lines (a slur or ledger line mistaken
  for a 6th/0th staff line) by spacing uniformity **and** ink strength; interior
  gaps are never touched, so genuinely faint real lines survive.
- **`_staff_hextent` + shared page extent**: the correction is truncated to the
  staves' real horizontal extent (first inked column → final bar line), taken as
  the **median across systems** since the whole block shares one binding. This
  rescues bottom systems whose own bar line under-detects, and stops the warp from
  fanning out in the steep foreshortened gutter.
- **Slope-extrapolated tails**: past the extent the comb is continued along the
  measured curl slope (single slope → the lines keep their spacing and can never
  fold), instead of being frozen flat — which used to leave a visible hook where
  the correction stopped.
- **Honest flatness metric**: `_staff_flatness` measures the *re-traced* comb
  centre of the remapped ink (robust 2..98 percentile peak-to-peak), not the
  warped guide lines (which are circular and always look flat). An optional
  **second guided refinement pass** re-traces the near-flat output and is kept
  only when it is measurably flatter, driving residual trace error toward ~1px.

- Then `deskew_barlines`, `align_system_margins`, `align_right_margin` (square the
  crease-side margin), `center_content` — all run **after** straightening, so any
  slight page-level skew comes from these margin steps, not the tracer.

### Formerly: localized end-of-system bows

The old tracer chose one warp for the whole page by *average* flatness, so a
single system bowing sharply at the binding tail could survive because the page
average still looked acceptable. The coupled-comb tracer with shared extent and
slope-extrapolated tails resolved this: the five flagged binding-tail pages
(Horn IV p6's serpentine ink included) now land sub-1.5px through the full
pipeline, with no regression on good pages or on grand-staff (Percussion) /
bass-clef (Timpani) structures. True binding *compression* (horizontal
foreshortening) is still not undone — see open problems.

## Assembly, batch, parity

- Size to the physical page (`--width-in/--height-in/--dpi`); optional trailing
  blank so the final count is even for double-sided printing (`--no-pad-even` to
  disable).
- **Lossless PDF encoding** (`write_pdf_from_images`): each page is saved as a
  **maximally-compressed grayscale PNG** (`save_output_image`, zlib level 9) and
  embedded **verbatim** via `img2pdf` — the PNG's compressed stream is copied into
  the PDF with no decode/re-encode, so rasters stay bit-identical while a fixed
  page size stretches every image to the exact physical dimensions (DPI
  preserved). This replaced ReportLab's `drawImage`, which silently re-deflated
  every page at a weak level and inflated files ~40% (≈2641 → 1887 KB/page here).
  Soft-gray music at 400 dpi is ~1.8 MB/page of genuine anti-aliased content
  (~85% pure white, ~15% mid-gray ink at full 256 levels); that is the lossless
  floor — JPEG q95 is actually *larger*, and stronger deflate (zopfli/Pillow)
  saves only ~3%.
- `--starts-on-even` / `--starts-on-even-files` / `starts_on_even.txt` insert a
  leading blank for parts whose first scanned page is even, so page parity lines
  up across a set.
- **Batch mode**: a directory in → `<input>-Processed` out. It is **resume-aware
  and skips outputs whose names match the sources**, so before a full regen you
  must clear/move the `-Processed` folder (renamed score-order outputs are *not*
  recognized). `--jobs N` parallelizes pages; `--clean-work` deletes per-page work
  dirs. Per-part page-number offsets (some parts have 2 cover pages, so printed
  page 1 = PDF page 3) are **not uniform even within an instrument family** and
  must be checked per part.
- **Combine** into named books is a separate concatenation step
  (`pypdfium2.import_pages`), producing `NN. <Work> • <Part>.pdf` in score order.
  Fixing a page splices into its book by index rather than regenerating the set.

### Rach 2 run recipe (verified)

The reference batch. Rach 2 needs **no** `--starts-on-even`:

```bash
python3 flat_scan.py "$HOME/Downloads/Rach 2" \
  --width-in 8.9375 --height-in 12 --dpi 400 \
  --straighten --booklet --jobs 4 --clean-work
```

- Raw `~/Downloads/Rach 2/` (29 parts) → `~/Downloads/Rach 2-Processed/` (29) →
  combine → `~/Downloads/Rach 2-Combined/` (13 books,
  `NN. Rach Symphony No. 2 • <Part>.pdf`).
- **13-book grouping** (verified, page counts checked): Flute I/II/III · Oboe
  I/II/III · Clarinet I/II + Bass Clarinet (folded in as "Clarinet III") ·
  Bassoon I/II · Horn I–IV · Trumpet I/II/III · Low Brass = Trombone I/II + Bass
  Trombone + Tuba · Timpani + Percussion · Violin I · Violin II · Viola · Cello ·
  Bass.
- Other works differ: Glinka `8.9375×12` **with**
  `--starts-on-even-files "Glinka Violin I.pdf"`; Stravinsky `9.5×12.5` (no
  even-start).

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
- **Finish generalizing the seam** if a music page ever shows a facing page in
  view *past* its own music edge (see the caveat above) — extend the fold seam to
  bound the music-page clip while still excluding the near-spine warp band.
- **Grow the "perfect" bucket** over time — the standing goal is fewer "ok" pages,
  more indistinguishable-from-a-real-scan pages.
