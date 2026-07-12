# Live integration — technology research & roadmap

> **Status: options research (July 2026).** This document answers the open
> questions from the original placeholder — where the compute runs, which
> technologies make it possible, and what the porting path looks like.
> Once a direction is chosen, a separate spec will cover the implementation
> and UX in detail.

## Problem statement

Today the [website](02-website.md) is a canned cinematic demo and the
[converter](01-converter.md) is a local CLI. The gap: a visitor cannot actually
convert *their own* page. This project closes that gap — a user drops in a photo
at [reillybova.github.io/clean-sheet](https://reillybova.github.io/clean-sheet/)
and gets a clean PDF back, with all processing running on their device.

### Constraints

- **Static hosting only** — GitHub Pages, no server, no API.
- **Privacy first** — photos never leave the browser.
- **No build service** — must work as a static site (HTML/JS/WASM assets).
- **Quality parity** — the browser output should match the CLI.

---

## The landscape (July 2026)

### What changed since we wrote the placeholder

The landscape shifted dramatically. **Pyodide** — CPython compiled to
WebAssembly — hit production maturity as version **314.0.2** (tracking
CPython 3.14.2). The milestone that matters most for us:

- **`opencv-python 4.11.0.86`** is now a built-in Pyodide package. Every
  `cv2` function our pipeline uses — `remap`, `morphologyEx`, `threshold`,
  `findContours`, `connectedComponentsWithStats`, `GaussianBlur`,
  `adaptiveThreshold`, `bilateralFilter` — works in WASM.
- **`numpy 2.4.3`**, **`Pillow 12.2.0`** — built-in, full support.
- **`PyMuPDF 1.27.2.2`** — built-in. Renders PDF pages to raster at any
  DPI, creates PDFs. This replaces `pypdfium2`, which is **blocked** in WASM
  (ctypes + prebuilt native binaries; no path forward).
- **`img2pdf`** — pure Python, installable via `micropip.install('img2pdf')`.
- **PEP 783** lets package maintainers publish WASM wheels to PyPI directly,
  so the ecosystem is growing fast (~28 packages and counting).

The upshot: our entire dependency set is available in the browser. We can
run the *actual Python pipeline* — not a rewrite — in WebAssembly.

---

## Options evaluated

### Option A — Pyodide (run the existing Python in WASM) ✦ Recommended

Run the real `flat_scan.py` / `flatscan_dewarp.py` / `flatscan_seam.py` code
inside a Pyodide Web Worker. The user's photo enters the virtual filesystem,
the pipeline processes it in Python+OpenCV+NumPy exactly as on the CLI, and the
result comes back as a downloadable PDF.

**What needs to change in the Python code:**

| Area | Change needed |
| --- | --- |
| PDF input (pypdfium2) | Replace with PyMuPDF (`fitz`). ~30 lines in `flat_scan.py`. PyMuPDF has an equivalent `page.get_pixmap(dpi=…)` API. |
| PDF output (img2pdf) | Keep as-is — `micropip.install('img2pdf')` works. |
| File I/O | Swap `pathlib` / `os` paths for Emscripten virtual FS paths. Thin shim. |
| CLI / argparse | Replace with a function entry point that takes a config dict. The JS UI calls into Python with structured args. |
| Multiprocessing | Remove `ProcessPoolExecutor` — WASM is single-threaded. Process pages sequentially (one Web Worker). Per-page time is ~1–3 s on desktop, so single-threaded is fine for typical 1–8 page parts. |
| cv2.imwrite / imread for debug | Gate behind a `--debug` flag (already done); no-op in browser mode. |

**Nothing else changes.** The ~3200 lines of pipeline logic — segmentation,
Coons rectification, ink cleanup, seam detection, staff straightening — run
unmodified.

**Performance on a 4000×6000 image (desktop browser):**

| Metric | Estimate |
| --- | --- |
| OpenCV / NumPy ops (C compiled to WASM) | ~85–95% of native speed |
| Pure Python glue | ~20–33% of native (3–5× slower) |
| Full page pipeline end-to-end | ~3–8 seconds |
| Peak memory | ~400–600 MB (within 2–4 GB WASM heap on desktop) |

**Bundle sizes (compressed, cached after first load):**

| Component | Size |
| --- | --- |
| Pyodide runtime + stdlib | ~6.4 MB |
| numpy | ~3–4 MB |
| opencv-python | ~8–12 MB |
| Pillow | ~1–2 MB |
| PyMuPDF | ~3–5 MB |
| img2pdf (pure Python) | ~0.1 MB |
| **Total** | **~22–30 MB** |

First load: ~8–15 s on broadband (WASM download + compilation). All repeat
visits load from service-worker cache in milliseconds. Lazy-load OpenCV and
PyMuPDF only when the user actually uploads a file.

**Architecture:**

```
┌─────────────────────────────────────────────┐
│  Main thread (JS)                           │
│  ┌─────────┐  ┌──────────┐  ┌────────────┐ │
│  │ UI /    │  │ cinematic│  │ progress   │ │
│  │ upload  │  │ demo     │  │ & download │ │
│  └────┬────┘  └──────────┘  └─────▲──────┘ │
│       │ postMessage                │        │
│  ┌────▼────────────────────────────┤        │
│  │  Web Worker                     │        │
│  │  ┌──────────────────────────┐   │        │
│  │  │  Pyodide 314             │   │        │
│  │  │  ┌─────┐ ┌────┐ ┌─────┐ │   │        │
│  │  │  │cv2  │ │np  │ │fitz │ │   │        │
│  │  │  └─────┘ └────┘ └─────┘ │   │        │
│  │  │  flat_scan.py (real)     │   │        │
│  │  └──────────────────────────┘   │        │
│  └─────────────────────────────────┘        │
└─────────────────────────────────────────────┘
  Static files on GitHub Pages — no server.
```

**Pros:**
- Reuses the exact production pipeline — same quality, same edge-case handling.
- 3200 lines of battle-tested Python stay intact; no rewrite risk.
- Future improvements to the CLI automatically apply to the web version.
- Privacy: all processing stays on the user's device.
- Cost: zero (static hosting).
- Offline-capable once cached.

**Cons:**
- ~22–30 MB first-load download (mitigated by caching + lazy loading).
- 8–15 s cold start (mitigated by loading Pyodide during the cinematic demo).
- Mobile: 4000×6000 images may push memory limits on low-RAM phones — need
  an auto-downscale path for mobile (<3 GB RAM devices).
- Single-threaded: pages process sequentially (acceptable for typical parts).

---

### Option B — OpenCV.js + TypeScript (full rewrite)

Rewrite the entire pipeline in TypeScript, using OpenCV.js for image
processing and `pdf-lib` for PDF creation.

**Pros:**
- Smaller bundle: ~8–10 MB (OpenCV.js + pdf-lib).
- Faster cold start: ~3–5 s.
- No Python runtime overhead.
- TypeScript tooling / ecosystem.

**Cons:**
- **Massive rewrite**: 3200 lines of carefully tuned Python → TypeScript.
  The Coons boundary patch, coupled staff-comb tracing, seam detection, and
  illumination normalization are algorithmically complex. Porting them
  faithfully is months of work with significant regression risk.
- OpenCV.js API is lower-level (manual `Mat` lifecycle, no NumPy slicing).
  Many of our operations use NumPy array tricks that have no direct JS
  equivalent.
- Dual maintenance: CLI stays Python, web becomes TS — divergence is
  inevitable.
- PDF rendering input: would need PDF.js or MuPDF.js (~1–3 MB extra) to
  replace pypdfium2 / PyMuPDF on the input side.

**Verdict: not recommended** unless bundle size is the overriding constraint.

---

### Option C — Hosted API (server-side processing)

Keep the Python CLI as-is, deploy behind a lightweight API (e.g. a Cloud Run
container), and have the static site POST photos to it.

**Pros:**
- Zero porting work on the pipeline.
- Works on any device (even old phones).
- No bundle-size concerns.

**Cons:**
- **Not compatible with our constraints**: GitHub Pages is static-only; we'd
  need a separate hosting provider and ongoing server cost.
- Photos leave the user's device — privacy story is gone.
- Latency: upload + process + download for multi-MB images.
- Operational burden: monitoring, scaling, abuse prevention.

**Verdict: rejected.** Violates the static-hosting and privacy constraints.

---

## Recommendation: Option A (Pyodide)

Pyodide is the clear winner. The technology is mature, our full dependency
stack is available, and the porting work is minimal — a shim layer around
the existing pipeline, not a rewrite. The ~25 MB download is acceptable for
a tool people use repeatedly (cached after first visit), and the cold start
can be hidden behind the existing cinematic demo animation.

---

## Answered: the original open questions

> **Where does the compute run?**

In the browser, via Pyodide (CPython 3.14 compiled to WebAssembly). All
processing is client-side. Photos never leave the device.

> **How much of the pipeline ports cleanly?**

All of it. Every OpenCV function we use (`remap`, `morphologyEx`,
`threshold`, `findContours`, `GaussianBlur`, `adaptiveThreshold`,
`bilateralFilter`, `connectedComponentsWithStats`) is available in
Pyodide's opencv-python 4.11 WASM build. NumPy, Pillow — built-in.
The only dependency swap is pypdfium2 → PyMuPDF (both available; PyMuPDF
is built-in, pypdfium2 is blocked).

> **Interface & UX.**

(Deferred to the implementation spec.) High-level: upload/camera capture →
progress bar with per-stage labels → download PDF. The cinematic demo
can optionally replay on the user's *real* page.

> **Parameters.**

(Deferred to the implementation spec.) Default to standard US-letter-ish
music sizes, auto-detect orientation, offer a "booklet mode" toggle and
a "straighten staves" toggle. Advanced users can set exact dimensions/DPI.

> **Limits & privacy.**

Client-side only — explicit "your photos never leave this device" banner.
EXIF orientation handled by Pillow / browser canvas. No file-size cap
beyond available browser memory (~2–4 GB heap on desktop).

---

## Porting roadmap (high level)

Phase 0 below is the scope of the next PR. Phases 1–3 are future work.

### Phase 0 — pypdfium2 → PyMuPDF migration (CLI)

Swap the PDF rendering backend in `flat_scan.py` from pypdfium2 to PyMuPDF
(`fitz`). This is prerequisite infrastructure — PyMuPDF is the only PDF
library available in Pyodide, so the CLI must use it too to keep one
codebase. This is a self-contained change (~30–50 lines) with no pipeline
logic changes. Validate against the existing samples.

### Phase 1 — Pyodide harness + single-page proof-of-concept

- Create a Web Worker that loads Pyodide + opencv-python + PyMuPDF.
- Bundle the pipeline `.py` files into the worker's virtual FS.
- Wire up: JS uploads an image → Python processes it → returns a PNG.
- Validate output matches the CLI for a sample page.
- Measure: cold start time, processing time, memory peak.

### Phase 2 — Full pipeline + PDF output

- Wire up multi-page PDF input (PyMuPDF renders each page to raster).
- Wire up PDF output via img2pdf (or pdf-lib on the JS side).
- Add progress reporting (worker → main thread messages).
- Handle booklet mode, straightening, page-size / DPI configuration.
- Mobile: auto-downscale input images when available memory is low.

### Phase 3 — UX integration with the existing site

- Integrate the converter into the existing `docs/` site alongside the
  cinematic demo.
- Upload/capture UI, parameter controls, download flow.
- Service worker for offline support + package caching.
- Lazy-load Pyodide packages (start loading during the demo).
- Test on real devices: desktop (Chrome/Firefox/Safari/Edge), mobile
  (iOS Safari, Android Chrome).

---

## Risks and mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Mobile memory pressure (4000×6000 images) | OOM crash on <3 GB RAM phones | Auto-detect device memory via `navigator.deviceMemory`; downscale to 2000×3000 on constrained devices. |
| Cold start perceived as slow (~8–15 s) | User abandonment | Pre-load Pyodide while the cinematic demo plays; show a progress bar with "Loading processing engine…" |
| img2pdf incompatibility in Pyodide | No PDF output | Fallback: use `pdf-lib` (JavaScript) for PDF assembly. |
| OpenCV WASM build missing a function we use | Pipeline stage fails | All functions verified against the Pyodide 314 built-in package list. If a gap appears, the function is basic enough to reimplement in NumPy. |
| pypdfium2 → PyMuPDF API differences | Rendering regressions | Validate against the full regression suite before merging Phase 0. |

## Why this is a separate doc

It is a distinct project with its own runtime, deployment, and failure modes;
the converter and the demo site each stay focused. Cross-link the three docs
as the integration takes shape.
