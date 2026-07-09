// Clean Sheet — showcase controller.
// One continuous progress clock g in [0,1] loops the cinematic shot. Captions and
// the scrub bar reflect the phase; the whole thing is pausable and seekable.

import { Cinematic } from "./cinematic.js";

const $ = (s) => document.querySelector(s);

const PHASES = [
  { key: "capture",  title: "The capture",     blurb: "A phone photo, angled and unevenly lit, curling at the edges — shot on whatever dark surface was handy.", at: 0.0 },
  { key: "find",     title: "Find the page",   blurb: "The sheet is isolated from its background and its outline traced — no assumption about the surface behind it.", at: 0.13 },
  { key: "map",      title: "Map the surface", blurb: "A UV grid is fitted to the page, capturing exactly how the paper bends and curls in space.", at: 0.30 },
  { key: "flatten",  title: "Lift it flat",    blurb: "The page lifts off the background and un-warps — grid, border and all — onto a true, flat rectangle.", at: 0.46 },
  { key: "clean",    title: "Clean the ink",   blurb: "Uneven lighting is removed and the ink develops into fine, even soft-grayscale — a clean sheet.", at: 0.72 },
];

const LOOP_MS = 15000;
const HOLD_MS = 1400;

const els = {
  showcase: $("#showcase"),
  viewport: $("#viewport"),
  gl: $("#gl"),
  title: $("#stageTitle"),
  blurb: $("#stageBlurb"),
  readout: $(".readout"),
  track: $("#track"),
  fill: $("#trackFill"),
  ticks: $("#ticks"),
  playBtn: $("#playBtn"),
  playLabel: $("#playLabel"),
  replayBtn: $("#replayBtn"),
};

const state = { g: 0, playing: true, phase: -1, cine: null, last: 0, holding: 0 };

const loadImage = (src) => new Promise((res, rej) => {
  const im = new Image(); im.onload = () => res(im); im.onerror = rej; im.src = src;
});

async function boot() {
  const demo = await fetch("assets/demo.json").then((r) => r.json());
  const [photo, ink] = await Promise.all([
    loadImage("assets/" + demo.photo.image),
    loadImage("assets/stages/ink.jpg"),
  ]);

  state.cine = new Cinematic(els.gl);
  try { await state.cine.init(demo, photo, ink); } catch (e) { console.warn(e); }
  els.gl.classList.add("show");

  buildTicks();
  setPhase(0, true);

  window.addEventListener("resize", () => state.cine && state.cine.resize());
  els.playBtn.addEventListener("click", () => setPlaying(!state.playing));
  els.replayBtn.addEventListener("click", () => { seek(0); setPlaying(true); });
  els.track.addEventListener("click", (e) => {
    const r = els.track.getBoundingClientRect();
    seek((e.clientX - r.left) / r.width); setPlaying(true);
  });

  const q = new URLSearchParams(location.search);
  if (q.has("g")) { seek(+q.get("g")); setPlaying(q.get("play") === "1"); }

  state.last = performance.now();
  requestAnimationFrame(tick);
}

function buildTicks() {
  els.ticks.innerHTML = "";
  PHASES.forEach((p) => {
    const b = document.createElement("button");
    b.className = "tick"; b.title = p.title;
    b.style.left = p.at * 100 + "%";
    b.addEventListener("click", (e) => { e.stopPropagation(); seek(p.at + 0.001); setPlaying(true); });
    els.ticks.appendChild(b);
    p._tick = b;
  });
}

function phaseFor(g) {
  let idx = 0;
  for (let i = 0; i < PHASES.length; i++) if (g >= PHASES[i].at) idx = i;
  return idx;
}

function setPhase(i, immediate = false) {
  if (i === state.phase) return;
  state.phase = i;
  const p = PHASES[i];
  els.readout.classList.add("swap");
  const apply = () => {
    els.title.textContent = p.title;
    els.blurb.textContent = p.blurb;
    els.readout.classList.remove("swap");
  };
  if (immediate) apply(); else setTimeout(apply, 160);
  PHASES.forEach((ph, idx) => ph._tick.classList.toggle("passed", idx <= i));
}

function render() {
  state.cine && state.cine.render(state.g);
  els.fill.style.width = (state.g * 100).toFixed(2) + "%";
  setPhase(phaseFor(state.g));
}

function tick(ts) {
  const dt = Math.min(50, ts - state.last);
  state.last = ts;
  if (state.playing) {
    if (state.g >= 1 && state.holding < HOLD_MS) {
      state.holding += dt;
    } else {
      if (state.g >= 1) { state.g = 0; state.holding = 0; }
      state.g = Math.min(1, state.g + dt / LOOP_MS);
      render();
    }
  }
  requestAnimationFrame(tick);
}

function setPlaying(v) {
  state.playing = v;
  els.showcase.classList.toggle("paused", !v);
  els.playLabel.textContent = v ? "Pause" : "Play";
}
function seek(g) { state.g = Math.max(0, Math.min(1, g)); state.holding = 0; render(); }

boot().catch((e) => {
  console.error(e);
  els.title.textContent = "Could not load the demo";
  els.blurb.textContent = "Please refresh, or view the project on GitHub.";
});
