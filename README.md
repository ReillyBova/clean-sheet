# FlatScan

Phone-scan cleanup workflow for orchestra parts.

Pipeline:

1. PDF/image input
2. Paper segmentation
3. Boundary-curve UV / Coons-patch reprojection
4. Lighting normalization
5. Soft grayscale ink cleanup
6. Correct-size PDF output

Optional: staff-line straightening (`--straighten`) to flatten residual waviness/skew in sheet music.

## Setup

```bash
cd "/Users/rebova/Documents/Coding Projects/FlatScan"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Basic usage

```bash
python flat_scan.py input.pdf output.pdf \
  --width-in 8.9375 \
  --height-in 12 \
  --dpi 400
```

Input resolution is detected automatically: each page is rasterized at its embedded scan's native resolution, so exports that declare a small page box around a full-resolution photo are handled correctly without any manual scaling.

## Batch mode

Pass a directory as the input to process every PDF/image inside it. If the output is omitted, results go to `<input>-Processed`.

```bash
python flat_scan.py "path/to/parts folder" \
  --width-in 8.9375 \
  --height-in 12 \
  --dpi 400
```

Use `--starts-on-even-files` to mark which files begin on an even printed page (comma-separated filename/stem/glob patterns), or `--starts-on-even-all` for every file:

```bash
python flat_scan.py "path/to/parts folder" \
  --width-in 8.9375 \
  --height-in 12 \
  --dpi 400 \
  --starts-on-even-files "Violin I.pdf,*cello*"
```

You can also list one pattern per line in a `starts_on_even.txt` file inside the input directory (picked up automatically), or point at any file with `--starts-on-even-list`. Add `--recursive` to scan subdirectories and preserve their structure in the output.

## Parallel processing

Pages are processed in parallel across separate processes. Each worker renders a full-resolution scan and is CPU- and memory-heavy, so the default is conservative (half your logical cores, capped at 4). Tune it with `--jobs`/`-j`:

```bash
python flat_scan.py "path/to/parts folder" \
  --width-in 8.9375 --height-in 12 --dpi 400 \
  --jobs 6
```

Use `--jobs 1` to disable parallelism (e.g. on a low-memory machine or for clean, ordered debug logs). In batch mode the worker count is applied per file, so total concurrency stays capped at `--jobs`.

## Straightening wavy staff lines (opt-in)

The Coons rectification corrects gross page geometry from the page boundary, but has no information about the page interior, so gentle "waviness" (staff lines dipping/rising) and a slight skew can remain. For sheet music, `--straighten` uses the staff lines themselves — which should be straight, horizontal, and parallel — to flatten this residual distortion.

```bash
python flat_scan.py "path/to/parts folder" \
  --width-in 9.5 --height-in 12.5 --dpi 400 \
  --straighten
```

How it works (implemented in `flatscan_dewarp.py`, applied after ink cleanup):

1. Estimate staff line thickness and spacing from vertical run-length histograms.
2. Detect staff *systems* (the 5-line combs) via horizontal-ink projection.
3. Measure each system's vertical wave across the page by incremental comb correlation.
4. Flatten each system, build a smooth monotonic vertical displacement field (which cannot fold/smear), and remap.

It is **opt-in and self-contained**: without the flag the pipeline is byte-for-byte unchanged. It also **safely no-ops** on pages without clear staves (title pages, near-blank pages) and refuses implausibly large warps, so it never degrades a page below the standard rectified output. Use `--debug` to compare `17_pre_straighten.png` vs `18_straightened.png`.


This inserts a blank page at the start so double-sided printing preserves page turns.

```bash
python flat_scan.py input.pdf output.pdf \
  --width-in 8.9375 \
  --height-in 12 \
  --dpi 400 \
  --starts-on-even
```

## Resume after failure or timeout

The script writes page-by-page working files to `<output_stem>_work`. Use `--resume` to skip already completed pages.

```bash
python flat_scan.py input.pdf output.pdf \
  --width-in 8.9375 \
  --height-in 12 \
  --dpi 400 \
  --starts-on-even \
  --resume
```

## Debugging

```bash
python flat_scan.py input.pdf output.pdf \
  --width-in 8.9375 \
  --height-in 12 \
  --dpi 400 \
  --debug \
  --debug-pages 1
```

Debug outputs show intermediate steps such as segmentation, UV grid, reprojection, and cleanup.

## Alternate output modes

Default is `soft-gray`, the recommended anti-aliased grayscale-ink output.

```bash
--mode soft-gray
--mode soft-black
--mode binary
--mode normalized-gray
--mode color
```

## Included samples

- `samples/russlan_raw_phone_scan.pdf` — the original 4-page sample scan
- `samples/russlan_clean_evenstart_8.9375x12_400dpi.pdf` — cleaned output with a leading blank page
