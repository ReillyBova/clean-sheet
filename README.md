# Clean Sheet

**Turn phone photos of sheet music into clean, flat, correctly-sized PDFs.**

Clean Sheet takes a casual phone/tablet capture of a printed part — shot at an
angle, unevenly lit, curling off the table, maybe folded into a book — and
returns a crisp, deskewed, page-sized PDF that reads like a proper scan. The
engine is a single CLI, `flat_scan.py`.

> **▶ [See it in action → reillybova.github.io/clean-sheet](https://reillybova.github.io/clean-sheet/)** — an
> interactive walkthrough that animates the real pipeline on a sample page:
> finding the sheet, mapping the warped surface with a UV grid, lifting it flat,
> and developing the clean soft-gray ink.

| Raw capture | Clean Sheet output |
| --- | --- |
| angled, glare, curl, wavy staves, neighbour-page bleed | flat, evenly lit, soft-gray ink, straight staves, correct page size |

See `samples/russlan_raw_phone_scan.pdf` → `samples/russlan_clean_evenstart_8.9375x12_400dpi.pdf`.

## What it does

- **Segments** the sheet from the background under uneven lighting (illumination-
  normalized, robust to wood grain, shadows, and adjacent surfaces).
- **Rectifies** page geometry with a Coons boundary patch — the four page edges
  drive a UV map that un-warps perspective and paper curl into a true rectangle.
- **Cleans the ink** to a fine, anti-aliased **soft grayscale**, removing uneven
  lighting and gently normalizing ink darkness (including flash glare) without
  going crude or losing thin lines.
- **Booklet / crease mode** (`--booklet`): for half-spread captures of bound
  books, detects the binding fold and clips the page there so the neighbouring
  page never bleeds in.
- **Staff-line straightening** (`--straighten`): uses the staff lines themselves
  to flatten residual waviness and skew, square the binding-side margin, and
  center the music — while keeping bar lines vertical and staff spacing intact.
- Outputs a **correctly-sized PDF** at your chosen physical dimensions and DPI,
  with optional blank-page padding for clean double-sided printing.

## Pipeline

Each page flows through (intermediate artifacts are written with `--debug`):

1. **Render** — rasterize the page at the scan's native resolution.
2. **Segment** — illumination-normalized threshold → sever bridges → largest
   component → fill holes → repair boundary notches. *(Booklet mode: detect the
   crease and clip the mask at the fold.)*
3. **Rectify** — corners → boundary edges → smooth + slight inset → Coons remap
   to a flat rectilinear plate.
4. **Clean ink** — denoise → remove uneven lighting → threshold → perimeter
   despeckle → soft-gray alpha compositing.
5. **Straighten** *(opt-in)* — staff-guided vertical flattening → bar-line
   de-shear → per-system left-margin de-drift → crease-side right-margin squaring
   → page centering.
6. **Assemble** — size to the physical page, pad for parity, embed each page as a
   maximally-compressed grayscale PNG **losslessly** into the PDF (via `img2pdf`,
   no re-encoding — bit-identical rasters, exact dimensions/DPI, ~30% smaller than
   a re-deflated embed).

## Setup

```bash
cd ~/Git-Repositories/clean-sheet
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Basic usage

```bash
python flat_scan.py input.pdf output.pdf \
  --width-in 8.9375 --height-in 12 --dpi 400
```

Input resolution is detected automatically: each page is rasterized at its
embedded scan's native resolution, so exports that declare a small page box
around a full-resolution photo are handled correctly with no manual scaling.

## Batch mode

Pass a directory to process every PDF/image inside it. If the output is omitted,
results go to `<input>-Processed`.

```bash
python flat_scan.py "path/to/parts folder" \
  --width-in 8.9375 --height-in 12 --dpi 400
```

Add `--recursive` to scan subdirectories and preserve their structure.

## Booklet / crease mode (opt-in)

For half-spread photos of a bound book — where one left/right edge folds into the
spine and a sliver of the facing page bleeds in — `--booklet` detects the fold
crease and clips the page there. The binding side is chosen automatically (the
side where the paper runs off the frame), and full flat sheets in view (e.g.
title pages) are left whole.

```bash
python flat_scan.py "path/to/booklet scans" \
  --width-in 8.9375 --height-in 12 --dpi 400 \
  --booklet --straighten
```

## Straightening wavy staff lines (opt-in)

The Coons rectification corrects gross page geometry from the boundary, but has
no information about the page interior, so gentle waviness and skew can remain.
For sheet music, `--straighten` uses the staff lines — which should be straight,
horizontal, and parallel — to flatten this.

```bash
python flat_scan.py "path/to/parts folder" \
  --width-in 9.5 --height-in 12.5 --dpi 400 \
  --straighten
```

How it works (in `flatscan_dewarp.py`, applied after ink cleanup):

1. Estimate staff-line thickness and spacing from vertical run-length histograms.
2. Detect staff *systems* (the 5-line combs) via horizontal-ink projection.
3. **Flatten** each system: trace its staff lines as a single *coupled comb*
   — all five lines share one centre curve and one slowly-varying spacing, so a
   line that jumps onto a tie, note, or bar line becomes a spacing outlier and is
   overridden by the group instead of dragging the warp. Iron the whole comb flat
   with a per-column rigid shift (staff spacing preserved — lines can never be
   thickened or merged). The correction is truncated to the staves' shared
   horizontal extent and its slope is extrapolated across the binding gutter, so
   the curl is ironed all the way to the edge without a frozen hook. A robust
   rigid fallback runs when the trace is unclear, and whichever leaves the staves
   flatter — measured off the same comb trace, with an optional second refinement
   pass — is kept, so a page is never made worse.
4. **De-shear** so bar lines/stems read vertical; **de-drift** each system's left
   margin; **square** the crease-side right margin; **center** the music block —
   all margin-seam-aware so a binding shadow is never mistaken for content.

It is **opt-in and self-contained** — without the flag the pipeline is unchanged
— and **safely no-ops** on pages without clear staves. Compare `17_pre_straighten`
vs `18_straightened` vs `19_aligned` with `--debug`.

## Double-sided printing & parity

By default an odd final page count is padded with a trailing blank so parts stay
duplex-friendly (`--no-pad-even` to disable). For a part whose first scanned page
is an even (left-hand) page, insert a leading blank:

```bash
python flat_scan.py input.pdf output.pdf \
  --width-in 8.9375 --height-in 12 --dpi 400 \
  --starts-on-even
```

In batch mode, mark which files need a leading blank with
`--starts-on-even-files "Violin I.pdf,*cello*"`, `--starts-on-even-all`, or a
`starts_on_even.txt` file (one pattern per line) in the input directory.

## Parallel processing

Pages are processed across separate worker processes. Each worker renders a
full-resolution scan and is CPU- and memory-heavy, so the default is conservative
(half your logical cores, capped at 4). Tune with `--jobs`/`-j`; use `--jobs 1` to
disable parallelism. In batch mode the worker count applies per file.

## Resume & working files

Page-by-page working files are written to `<output_stem>_work`. Use `--resume` to
skip already-completed pages after an interruption, and `--clean-work` to delete
the working directory after a successful assembly.

## Debugging

```bash
python flat_scan.py input.pdf output.pdf \
  --width-in 8.9375 --height-in 12 --dpi 400 \
  --debug --debug-pages 1
```

Writes intermediate stages (segmentation seed, page mask, boundary UV grid,
rectified plate, ink mask, straightening steps) into the debug directory.

## Output modes

Default is `soft-gray`, the recommended anti-aliased grayscale ink.

| Mode | Description |
| --- | --- |
| `soft-gray` | Fine anti-aliased grayscale ink (default, recommended) |
| `soft-black` | Anti-aliased near-black ink |
| `binary` | Hard black/white |
| `normalized-gray` | Lighting-normalized grayscale, paper retained |

## Included samples

- `samples/russlan_raw_phone_scan.pdf` — the original phone-scan sample.
- `samples/russlan_clean_evenstart_8.9375x12_400dpi.pdf` — cleaned output with a
  leading blank page.

## Live demo (webapp)

The interactive walkthrough at
**[reillybova.github.io/clean-sheet](https://reillybova.github.io/clean-sheet/)**
is a static site served from `docs/` via GitHub Pages. It's a single continuous
WebGL shot (three.js) that plays the real pipeline stages on one page — the mesh,
UV grid, and un-warp are driven by actual Coons source coordinates exported from
a live run.

- `docs/` — the site (`index.html`, `css/`, `js/cinematic.js` scene + `js/app.js`
  controller, `assets/`).
- `tools/build_demo_assets.py` — regenerates the demo images and
  `docs/assets/demo.json` (the per-vertex source→flat UV grid) from a real
  pipeline run:

  ```bash
  python tools/build_demo_assets.py
  ```

No build step is required — three.js loads from a CDN, and a `.nojekyll` file lets
Pages serve the assets as-is.

## Design docs

The *why* behind the code — problems hit, approaches tried and rejected, and the
invariants we hold — lives in [`design/`](design/README.md): the
[converter pipeline](design/01-converter.md), the [demo website](design/02-website.md),
and the planned [live integration](design/03-live-integration.md).
