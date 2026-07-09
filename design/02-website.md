# Website design (`docs/` — GitHub Pages demo)

An interactive, cinematic walkthrough that animates the **real** pipeline on a
sample page, served from `docs/` on `master` via GitHub Pages
(`reillybova.github.io/clean-sheet`). It is a *demo/marketing* surface, not the
converter — it replays the stages visually so a viewer immediately understands
what the tool does.

## Goals

- Show, don't tell: the actual stages (find the sheet → map the warped surface
  with a UV grid → lift it flat → develop the clean soft-gray ink) on a genuine
  sample, not a mockup.
- Feel premium and "3Blue1Brown-slick" — smooth, legible, deliberate motion.
- Zero build step / no framework lock-in; static files that GitHub Pages can
  serve directly (`.nojekyll` present).

## Structure

```
docs/
  index.html        entry point
  css/              styling
  js/               the cinematic (Three.js) walkthrough + stage logic
  assets/           sample imagery
  .nojekyll         serve as-is, no Jekyll processing
```

## Key design decisions (and the road taken)

- **Single continuous shot.** Reworked from discrete steps to one continuous
  cinematic so the transformation reads as a single fluid transformation rather
  than a slideshow.
- **Animate on the real plate.** The 3D plane is textured with the actual sample
  page and the overlay effects are driven from the true geometry, so what you see
  is faithful to the pipeline.
- **Overlay/photo alignment is the thing to get right.** The recurring bug was
  the 3D model drifting out of registration with the base image. Fixes: dropped
  the Ken-Burns drift, corrected a doubled "sheet on top of sheet" at rest, fixed
  a rotated/flipped background quad, and contain-fit the plate so its mapping
  doesn't overrun the frame (title/footer stay visible).
- **UV grid drawn from the corner, both axes together.** Rather than toggling
  individual u/v lines, sweep both axes smoothly from one corner across the page
  — reads as a slick coordinate-map reveal.
- **Restrained, natural motion curve.** The cinematic "lift" uses a constrained
  non-linear ease — an earlier curve overshot and lifted the page outside the
  frame before settling; the constrained version stays in-frame and settles
  naturally.
- **Fit flush at rest.** The final resting frame sits flush with the preview
  edges (page corners may fall outside due to rounding, but the edges are flush)
  so it looks like the finished page fills the viewport.
- **Tuned overlay legibility.** Highlight and UV-grid opacity/saturation were
  raised from near-invisible to clearly readable without looking heavy.

## Hosting

GitHub Pages is configured to serve the `docs/` folder on the `master` branch.
The repo README links the live demo. Because it is static, updating the site is
just committing to `docs/` and pushing.

## Possible future work

- A "bring your own page" mode that runs a real conversion in-browser — which is
  the bridge to the [live-integration](03-live-integration.md) project.
