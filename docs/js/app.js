// Clean Sheet — showcase orchestration.
// A single progress clock drives stage cross-fades, captions, the timeline, and
// the WebGL reprojection morph — so everything is seekable and pausable.

import { Reprojector } from "./reproject.js";

const $ = (s) => document.querySelector(s);
const smooth = (a, b, x) => {
  const t = Math.max(0, Math.min(1, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

const els = {
  showcase: $("#showcase"),
  viewport: $("#viewport"),
  gl: $("#gl"),
  frameA: $("#frameA"),
  frameB: $("#frameB"),
  title: $("#stageTitle"),
  blurb: $("#stageBlurb"),
  readout: $(".readout"),
  index: $("#stageIndex"),
  total: $(".stage-total"),
  timeline: $("#timeline"),
  playBtn: $("#playBtn"),
  playLabel: $("#playLabel"),
  replayBtn: $("#replayBtn"),
};

// Per-stage durations (ms). The reprojection stage is longer for the morph.
const DUR = {
  input: 2600, lighting: 2400, segment: 2400, mask: 2200,
  uv: 2600, rectified: 3600, ink: 2600, straighten: 3200,
};

const state = {
  stages: [],
  i: 0,
  t: 0,            // elapsed ms in current stage
  playing: true,
  frontIsB: false, // which frame img currently shows
  repro: null,
  images: {},      // key -> HTMLImageElement
  lastTs: 0,
};

async function loadImage(src) {
  return new Promise((res, rej) => {
    const im = new Image();
    im.onload = () => res(im);
    im.onerror = rej;
    im.src = src;
  });
}

async function boot() {
  const demo = await fetch("assets/demo.json").then((r) => r.json());
  state.stages = demo.stages.map((s) => ({ ...s, dur: DUR[s.key] || 2400 }));
  els.total.textContent = "/ " + state.stages.length;

  // preload stage images + the photo for the mesh
  const photo = await loadImage("assets/" + demo.photo.image);
  await Promise.all(
    state.stages.map(async (s) => {
      state.images[s.key] = await loadImage(`assets/stages/${s.key}.jpg`);
    })
  );

  // set up the reprojector (hero morph); tolerate WebGL being unavailable
  state.repro = new Reprojector(els.gl);
  try { await state.repro.init(demo, photo); } catch (e) { console.warn("WebGL morph unavailable:", e); }
  state.morphOK = !!(state.repro && state.repro.ready);

  buildTimeline();
  // first frame
  els.frameA.src = state.images.input.src;
  els.frameA.classList.add("show");
  enterStage(0, true);

  window.addEventListener("resize", () => state.repro && state.repro.resize());
  els.playBtn.addEventListener("click", togglePlay);
  els.replayBtn.addEventListener("click", () => { jumpTo(0); setPlaying(true); });

  // debug: ?stage=N&p=0.5&play=0 to inspect a specific moment
  const q = new URLSearchParams(location.search);
  if (q.has("stage")) {
    const si = Math.max(0, Math.min(state.stages.length - 1, +q.get("stage")));
    jumpTo(si);
    setPlaying(q.get("play") === "1");
    if (!state.playing) {
      const pr = q.has("p") ? Math.max(0, Math.min(1, +q.get("p"))) : 0;
      state.t = pr * state.stages[si].dur;
      state.stages[si]._fill.style.width = pr * 100 + "%";
      if (state.stages[si].key === "rectified" && state.morphOK) updateMorph(pr);
    }
  }

  state.lastTs = performance.now();
  requestAnimationFrame(tick);
}

function buildTimeline() {
  els.timeline.innerHTML = "";
  state.stages.forEach((s, idx) => {
    const b = document.createElement("button");
    b.className = "tl-dot";
    b.setAttribute("role", "tab");
    b.title = s.title;
    b.innerHTML = '<span class="fill"></span>';
    b.addEventListener("click", () => { jumpTo(idx); setPlaying(true); });
    els.timeline.appendChild(b);
    s._dot = b;
    s._fill = b.querySelector(".fill");
  });
}

function setFrame(key) {
  // cross-fade to the image for `key` on the back frame
  const back = state.frontIsB ? els.frameA : els.frameB;
  const front = state.frontIsB ? els.frameB : els.frameA;
  back.src = state.images[key].src;
  // force reflow so the opacity transition runs
  void back.offsetWidth;
  back.classList.add("show");
  front.classList.remove("show");
  state.frontIsB = !state.frontIsB;
}

function currentFrontEl() { return state.frontIsB ? els.frameB : els.frameA; }
function currentBackEl() { return state.frontIsB ? els.frameA : els.frameB; }

function enterStage(i, immediate = false) {
  const s = state.stages[i];
  state.i = i;
  state.t = 0;

  // caption swap
  els.readout.classList.add("swap");
  setTimeout(() => {
    els.title.textContent = s.title;
    els.blurb.textContent = s.blurb;
    els.index.textContent = String(i + 1);
    els.readout.classList.remove("swap");
  }, immediate ? 0 : 180);

  // timeline dot states
  state.stages.forEach((st, idx) => {
    st._dot.classList.toggle("done", idx < i);
    if (idx > i) { st._dot.classList.remove("done"); st._fill.style.width = "0%"; }
  });

  if (s.key === "rectified" && state.morphOK) {
    // prepare morph: show plain photo underneath, reset mesh, reveal gl
    els.viewport.classList.add("morphing");
    setFrame("input");
    state.repro.setMorph(0);
    els.gl.classList.add("show");
  } else {
    els.viewport.classList.remove("morphing");
    els.gl.classList.remove("show");
    if (!immediate) setFrame(s.key);
    else { currentFrontEl().src = state.images[s.key].src; }
  }
}

function updateMorph(p) {
  // p: progress through the rectified stage [0,1]
  const meshIn = smooth(0.0, 0.12, p);
  const morphT = smooth(0.12, 0.82, p);
  const woodFade = 1 - smooth(0.18, 0.55, p);
  const reveal = smooth(0.82, 1.0, p);

  els.gl.style.opacity = String(Math.max(meshIn, 1 - reveal));
  currentFrontEl().style.opacity = String(woodFade); // the input photo (wood) fades
  state.repro.setMorph(morphT);

  // reveal the true rectified image on the back frame near the end
  const back = currentBackEl();
  if (reveal > 0 && back.dataset.key !== "rectified") {
    back.src = state.images.rectified.src;
    back.dataset.key = "rectified";
  }
  back.style.opacity = String(reveal);
}

function finishMorph() {
  // settle: rectified image becomes the front frame, gl hidden
  const back = currentBackEl();
  back.classList.add("show");
  back.style.opacity = "";
  currentFrontEl().classList.remove("show");
  currentFrontEl().style.opacity = "";
  state.frontIsB = !state.frontIsB;
  els.gl.classList.remove("show");
  els.gl.style.opacity = "";
  delete currentBackEl().dataset.key;
  els.viewport.classList.remove("morphing");
}

function tick(ts) {
  const dt = Math.min(64, ts - state.lastTs);
  state.lastTs = ts;
  const s = state.stages[state.i];

  if (state.playing) {
    state.t += dt;
    const p = Math.min(1, state.t / s.dur);
    s._fill.style.width = (p * 100) + "%";

    if (s.key === "rectified" && state.morphOK) updateMorph(p);

    if (p >= 1) {
      if (s.key === "rectified" && state.morphOK) finishMorph();
      s._dot.classList.add("done");
      const next = (state.i + 1) % state.stages.length;
      if (next === 0) {
        // loop: brief pause on the finished result, then restart
        enterStage(0);
      } else {
        enterStage(next);
      }
    }
  }
  requestAnimationFrame(tick);
}

function setPlaying(v) {
  state.playing = v;
  els.showcase.classList.toggle("paused", !v);
  els.playLabel.textContent = v ? "Pause" : "Play";
}
function togglePlay() { setPlaying(!state.playing); }

function jumpTo(i) {
  // clean up any morph visuals
  els.viewport.classList.remove("morphing");
  els.gl.classList.remove("show");
  els.gl.style.opacity = "";
  els.frameA.style.opacity = "";
  els.frameB.style.opacity = "";
  enterStage(i);
}

boot().catch((e) => {
  console.error(e);
  els.title.textContent = "Could not load the demo";
  els.blurb.textContent = "Please refresh, or view the project on GitHub.";
});
