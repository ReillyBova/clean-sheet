# Design docs

Rationale and design decisions behind the Clean Sheet / FlatScan work. The
`README.md` at the repo root is the *user-facing* guide (what it does, how to run
it); these docs are the *why* — the problems we hit, the approaches we tried and
rejected, and the invariants we hold. They are meant to save future-us from
re-deriving hard-won decisions.

| Doc | Scope |
| --- | --- |
| [`01-converter.md`](01-converter.md) | The `flat_scan.py` / `flatscan_dewarp.py` pipeline — segmentation, Coons rectification, ink cleanup, booklet/seam handling, staff straightening, batch/assembly. |
| [`02-website.md`](02-website.md) | The GitHub Pages cinematic demo (`docs/`) — the single-shot Three.js walkthrough of the real pipeline. |
| [`03-live-integration.md`](03-live-integration.md) | *(planned)* Wiring the live site to the converter so anyone can upload a photo and get a clean PDF in the browser / via a service. |

## Conventions for these docs

- **Record the road not taken.** When we reject an approach, write down *why* —
  most of our worst time-sinks were re-litigating a dead end.
- **State invariants explicitly.** e.g. "never clip this page's own content."
  These are the guardrails that keep iterative changes safe.
- Keep them honest about **known limitations** and **open problems** — the docs
  double as the backlog for long-term quality work.
