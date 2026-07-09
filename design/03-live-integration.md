# Live integration (planned)

> **Status: placeholder.** This document reserves the design space for the
> separate project of wiring the live site to the converter, so anyone can drop
> in a photo and get a clean PDF back. Fill it in when that work begins.

## Problem statement (to refine)

Today the [website](02-website.md) is a canned cinematic demo and the
[converter](01-converter.md) is a local CLI. The gap: a visitor cannot actually
convert *their own* page. This project closes that gap.

## Open questions to resolve here

- **Where does the compute run?** In-browser (WASM/JS port of the pipeline —
  OpenCV.js, pyodide?) vs. a hosted service/API the site calls. Trade-offs:
  privacy (photos never leave the device) and cost (no server) for in-browser,
  vs. fidelity and simpler maintenance for a service.
- **How much of the pipeline ports cleanly?** Which stages are cheap in the
  browser and which (Coons remap, straightening passes) need native speed.
- **Interface & UX.** Upload/capture, progress, the same cinematic staging shown
  on the user's real page, download the sized PDF.
- **Parameters.** How to expose (or auto-infer) page size / DPI / booklet /
  straighten without overwhelming a casual user.
- **Limits & privacy.** File-size caps, EXIF/orientation handling, and an
  explicit stance on not retaining uploads.

## Why it is a separate doc

It is a distinct project with its own runtime, deployment, and failure modes; the
converter and the demo site each stay focused. Cross-link the three docs as the
integration takes shape.
