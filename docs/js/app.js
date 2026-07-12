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
  loading: $("#loading"),
  title: $("#stageTitle"),
  blurb: $("#stageBlurb"),
  readout: $(".readout"),
  track: $("#track"),
  fill: $("#trackFill"),
  ticks: $("#ticks"),
  reel: $("#reel"),
  playBtn: $("#playBtn"),
  playLabel: $("#playLabel"),
  replayBtn: $("#replayBtn"),
};

const state = { g: 0, playing: true, phase: -1, cine: null, last: 0, holding: 0,
                playlist: [], idx: 0, switching: false, ready: false };

const loadImage = (src) => new Promise((res, rej) => {
  const im = new Image(); im.onload = () => res(im); im.onerror = rej; im.src = src;
});

const shuffle = (a) => {
  const r = a.slice();
  for (let i = r.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [r[i], r[j]] = [r[j], r[i]];
  }
  return r;
};

async function boot() {
  const demo = await fetch("assets/demo.json").then((r) => r.json());
  // Support the legacy single-example shape as well as the {examples:[...]} list.
  const examples = demo.examples || [demo];
  state.playlist = shuffle(examples);
  state.idx = 0;
  // Optional ?ex=<id> pins a specific example first (sharing / testing).
  const q0 = new URLSearchParams(location.search);
  if (q0.has("ex")) {
    const i = state.playlist.findIndex((e) => e.id === q0.get("ex"));
    if (i > 0) { const [e] = state.playlist.splice(i, 1); state.playlist.unshift(e); }
  }

  state.cine = new Cinematic(els.gl);
  buildTicks();
  buildReel(examples);
  await loadExample(state.playlist[0]);
  els.gl.classList.add("show");
  setPhase(0, true);
  // Reveal only once the first frame is actually painted, so the clock never
  // advances over a blank/loading viewport.
  requestAnimationFrame(() => {
    els.loading.classList.add("hide");
    state.ready = true;
    state.last = performance.now();
  });

  window.addEventListener("resize", () => state.cine && state.cine.resize());
  els.playBtn.addEventListener("click", () => setPlaying(!state.playing));
  els.replayBtn.addEventListener("click", () => { seek(0); setPlaying(true); });
  els.track.addEventListener("click", (e) => {
    const r = els.track.getBoundingClientRect();
    seek((e.clientX - r.left) / r.width); setPlaying(true);
  });

  const q = new URLSearchParams(location.search);
  if (q.has("g")) { seek(+q.get("g")); setPlaying(q.get("play") === "1"); }

  requestAnimationFrame(tick);
}

// Load one example (its photo + ink) into the shared cinematic scene.
async function loadExample(ex) {
  state.switching = true;
  try {
    const [photo, ink] = await Promise.all([
      loadImage("assets/" + ex.photo.image),
      loadImage("assets/" + ex.ink),
    ]);
    await state.cine.init(ex, photo, ink);
  } catch (e) { console.warn(e); }
  state.g = 0; state.holding = 0; state.phase = -1;
  setActiveReel(ex.id);
  render();
  state.switching = false;
}

// Advance to the next example, reshuffling after a full pass so every example
// is shown once before any repeats.
function nextExample() {
  state.idx += 1;
  if (state.idx >= state.playlist.length) {
    state.playlist = shuffle(state.playlist);
    state.idx = 0;
  }
  loadExample(state.playlist[state.idx]);
}

// Jump straight to a chosen example (from the reel); auto-advance then continues
// through the shuffled playlist from that point.
function jumpTo(id) {
  if (state.switching) return;
  const i = state.playlist.findIndex((e) => e.id === id);
  if (i < 0 || state.playlist[i].id === state.playlist[state.idx].id) return;
  state.idx = i;
  loadExample(state.playlist[i]);
  setPlaying(true);
}

const shortLabel = (l) =>
  (l && l.includes("—") ? l.split("—").pop() : l || "").replace(/\(.*\)/, "").trim();

function buildReel(examples) {
  els.reel.innerHTML = "";
  state.reelItems = {};
  examples.forEach((ex) => {
    const b = document.createElement("button");
    b.className = "reel-item"; b.type = "button"; b.role = "tab";
    b.title = ex.label || ex.id;
    b.innerHTML =
      `<img class="reel-thumb" loading="lazy" alt="${ex.label || ex.id}" src="assets/${ex.photo.image}">` +
      `<span class="reel-label">${shortLabel(ex.label)}</span>`;
    b.addEventListener("click", () => jumpTo(ex.id));
    els.reel.appendChild(b);
    state.reelItems[ex.id] = b;
  });
}

function setActiveReel(id) {
  const items = state.reelItems || {};
  Object.keys(items).forEach((k) => items[k].classList.toggle("active", k === id));
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
  if (!p) return;
  els.readout.classList.add("swap");
  const apply = () => {
    els.title.textContent = p.title;
    els.blurb.textContent = p.blurb;
    els.readout.classList.remove("swap");
  };
  if (immediate) apply(); else setTimeout(apply, 160);
  PHASES.forEach((ph, idx) => ph._tick && ph._tick.classList.toggle("passed", idx <= i));
}

function render() {
  state.cine && state.cine.render(state.g);
  els.fill.style.width = (state.g * 100).toFixed(2) + "%";
  setPhase(phaseFor(state.g));
}

function tick(ts) {
  const dt = Math.min(50, ts - state.last);
  state.last = ts;
  if (state.ready && state.playing && !state.switching) {
    if (state.g >= 1 && state.holding < HOLD_MS) {
      state.holding += dt;
    } else if (state.g >= 1) {
      // Finished this example: move to the next one (reshuffles after a pass)
      // rather than replaying the same page.
      nextExample();
    } else {
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
