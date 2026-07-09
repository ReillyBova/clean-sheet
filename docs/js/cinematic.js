// Clean Sheet — cinematic scene.
// One continuous WebGL shot over the original photo: a faint highlight fills the
// page, its outline traces on, a UV grid draws across it, the whole thing lifts
// and un-warps in 3D (grid + border morphing with it), then the flat page
// develops from raw capture into clean soft-gray ink. Driven by a single
// normalized progress `g` in [0,1] so it is fully seekable.

import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js";

const sstep = (a, b, x) => {
  const t = Math.max(0, Math.min(1, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};
const lerp = (a, b, t) => a + (b - a) * t;

const GOLD = 0xe0a94a, GOLD_SOFT = 0xf0cd8a;

export class Cinematic {
  constructor(canvas) { this.canvas = canvas; this.ready = false; }

  async init(demo, photoImg, inkImg) {
    const GW = demo.grid.w, GH = demo.grid.h;
    const src = demo.grid.src;
    const W = demo.photo.w, H = demo.photo.h;
    const aspect = H / W;
    this.aspect = aspect;

    // cover-fit flat plate (matches the ink still)
    const outWH = demo.output_aspect;
    const plateH = 1.0 / outWH, plateW = 1.0;
    const px0 = 0, px1 = plateW;
    const py0 = (aspect - plateH) / 2, py1 = py0 + plateH;

    const N = GW * GH;
    this.GW = GW; this.GH = GH; this.N = N;
    this.src2d = new Float32Array(N * 2);
    this.dst2d = new Float32Array(N * 2);
    for (let j = 0; j < GH; j++) {
      for (let i = 0; i < GW; i++) {
        const k = j * GW + i;
        const [su, sv] = src[k];
        this.src2d[k * 2] = su; this.src2d[k * 2 + 1] = sv * aspect;
        this.dst2d[k * 2] = lerp(px0, px1, i / (GW - 1));
        this.dst2d[k * 2 + 1] = lerp(py0, py1, j / (GH - 1));
      }
    }
    this.cur = new Float32Array(N * 3); // shared current positions (z=0)

    // --- scene / camera / renderer (y-down ortho) ---
    this.scene = new THREE.Scene();
    this.camera = new THREE.OrthographicCamera(0, 1, 0, aspect, -10, 10);
    try {
      this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, alpha: true, antialias: true });
    } catch (e) { this.failed = true; return; }
    this.renderer.setClearColor(0x000000, 0);

    const mkTex = (img) => {
      const t = new THREE.Texture(img); t.needsUpdate = true;
      t.colorSpace = THREE.SRGBColorSpace;
      t.minFilter = THREE.LinearFilter; t.magFilter = THREE.LinearFilter;
      return t;
    };
    this.photoTex = mkTex(photoImg);
    this.inkTex = inkImg ? mkTex(inkImg) : null;

    // background: full photo quad (page + surround), fixed
    {
      const g = new THREE.PlaneGeometry(1, aspect);
      const pos = g.attributes.position;
      // PlaneGeometry is centered; move to fill [0,1]x[0,aspect] in y-down space
      for (let i = 0; i < pos.count; i++) {
        pos.setXYZ(i, pos.getX(i) + 0.5, (0.5 - pos.getY(i)) * aspect, -0.2);
      }
      // uv already 0..1 but y flipped for y-down: flip v
      const uv = g.attributes.uv;
      for (let i = 0; i < uv.count; i++) uv.setY(i, 1 - uv.getY(i));
      this.bgMat = new THREE.MeshBasicMaterial({ map: this.photoTex, transparent: true });
      this.bg = new THREE.Mesh(g, this.bgMat);
      this.scene.add(this.bg);
    }

    // shared grid geometry helpers
    const indices = [];
    for (let j = 0; j < GH - 1; j++)
      for (let i = 0; i < GW - 1; i++) {
        const a = j * GW + i, b = a + 1, c = a + GW, d = c + 1;
        indices.push(a, c, b, b, c, d);
      }

    const uvSrc = new Float32Array(N * 2), uvFlat = new Float32Array(N * 2);
    for (let j = 0; j < GH; j++)
      for (let i = 0; i < GW; i++) {
        const k = j * GW + i;
        uvSrc[k * 2] = src[k][0]; uvSrc[k * 2 + 1] = 1 - src[k][1];
        uvFlat[k * 2] = i / (GW - 1); uvFlat[k * 2 + 1] = 1 - j / (GH - 1);
      }

    // page mesh (photo-textured, morphs)
    this.pageGeo = new THREE.BufferGeometry();
    this.pageGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(N * 3), 3));
    this.pageGeo.setAttribute("uv", new THREE.BufferAttribute(uvSrc, 2));
    this.pageGeo.setIndex(indices);
    this.pageMat = new THREE.MeshBasicMaterial({ map: this.photoTex, transparent: true });
    this.page = new THREE.Mesh(this.pageGeo, this.pageMat);
    this.page.position.z = 0.0;
    this.scene.add(this.page);

    // ink mesh (cleaned, flat, fades in for the develop)
    if (this.inkTex) {
      this.inkGeo = new THREE.BufferGeometry();
      const inkPos = new Float32Array(N * 3);
      for (let k = 0; k < N; k++) { inkPos[k*3]=this.dst2d[k*2]; inkPos[k*3+1]=this.dst2d[k*2+1]; inkPos[k*3+2]=0.02; }
      this.inkGeo.setAttribute("position", new THREE.BufferAttribute(inkPos, 3));
      this.inkGeo.setAttribute("uv", new THREE.BufferAttribute(uvFlat, 2));
      this.inkGeo.setIndex(indices);
      this.inkMat = new THREE.MeshBasicMaterial({ map: this.inkTex, transparent: true, opacity: 0 });
      this.ink = new THREE.Mesh(this.inkGeo, this.inkMat);
      this.scene.add(this.ink);
    }

    // highlight fill (faint gold, morphs)
    this.hlGeo = new THREE.BufferGeometry();
    this.hlGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(N * 3), 3));
    this.hlGeo.setIndex(indices);
    this.hlMat = new THREE.MeshBasicMaterial({ color: GOLD_SOFT, transparent: true, opacity: 0, depthWrite: false });
    this.hl = new THREE.Mesh(this.hlGeo, this.hlMat);
    this.hl.position.z = 0.01;
    this.scene.add(this.hl);

    // outline (perimeter order, draws on, morphs)
    this.perim = [];
    for (let i = 0; i < GW; i++) this.perim.push(i);                         // top L→R
    for (let j = 1; j < GH; j++) this.perim.push(j * GW + (GW - 1));         // right
    for (let i = GW - 2; i >= 0; i--) this.perim.push((GH - 1) * GW + i);    // bottom
    for (let j = GH - 2; j >= 0; j--) this.perim.push(j * GW);              // left
    this.outGeo = new THREE.BufferGeometry();
    this.outGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(this.perim.length * 3), 3));
    this.outMat = new THREE.LineBasicMaterial({ color: GOLD_SOFT, transparent: true, opacity: 0 });
    this.outline = new THREE.Line(this.outGeo, this.outMat);
    this.outline.position.z = 0.03;
    this.scene.add(this.outline);

    // interior UV grid (vertical then horizontal lines, draws on, morphs)
    this.gridPairs = [];
    const stepI = 4, stepJ = 5;
    for (let i = stepI; i < GW - 1; i += stepI)
      for (let j = 0; j < GH - 1; j++) this.gridPairs.push(j * GW + i, (j + 1) * GW + i);
    for (let j = stepJ; j < GH - 1; j += stepJ)
      for (let i = 0; i < GW - 1; i++) this.gridPairs.push(j * GW + i, j * GW + i + 1);
    this.gridGeo = new THREE.BufferGeometry();
    this.gridGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(this.gridPairs.length * 3), 3));
    this.gridMat = new THREE.LineBasicMaterial({ color: GOLD, transparent: true, opacity: 0 });
    this.grid = new THREE.LineSegments(this.gridGeo, this.gridMat);
    this.grid.position.z = 0.025;
    this.scene.add(this.grid);

    // group we can scale/lift for the "rise"
    this.rig = new THREE.Group();
    this.scene.remove(this.page, this.hl, this.outline, this.grid);
    if (this.ink) this.scene.remove(this.ink);
    this.rig.add(this.page, this.hl, this.outline, this.grid);
    if (this.ink) this.rig.add(this.ink);
    this.scene.add(this.rig);

    this._resize();
    this.ready = true;
    this.render(0);
  }

  _resize() {
    const r = this.canvas.getBoundingClientRect();
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setSize(r.width, r.height, false);
  }
  resize() { if (this.ready) { this._resize(); this.render(this._g || 0); } }

  _writePositions(morphT, lift) {
    const s = this.src2d, d = this.dst2d, c = this.cur;
    for (let k = 0; k < this.N; k++) {
      const sx = s[k*2], sy = s[k*2+1], dx = d[k*2], dy = d[k*2+1];
      c[k*3]   = lerp(sx, dx, morphT);
      c[k*3+1] = lerp(sy, dy, morphT) - lift;
      c[k*3+2] = 0;
    }
    // page + highlight share grid positions
    const pp = this.pageGeo.attributes.position.array;
    const hp = this.hlGeo.attributes.position.array;
    pp.set(c); hp.set(c);
    this.pageGeo.attributes.position.needsUpdate = true;
    this.hlGeo.attributes.position.needsUpdate = true;
    // outline
    const op = this.outGeo.attributes.position.array;
    for (let n = 0; n < this.perim.length; n++) {
      const k = this.perim[n];
      op[n*3] = c[k*3]; op[n*3+1] = c[k*3+1]; op[n*3+2] = 0;
    }
    this.outGeo.attributes.position.needsUpdate = true;
    // grid
    const gp = this.gridGeo.attributes.position.array;
    for (let n = 0; n < this.gridPairs.length; n++) {
      const k = this.gridPairs[n];
      gp[n*3] = c[k*3]; gp[n*3+1] = c[k*3+1]; gp[n*3+2] = 0;
    }
    this.gridGeo.attributes.position.needsUpdate = true;
  }

  // g in [0,1]
  render(g) {
    if (!this.ready) return;
    this._g = g;

    // --- phase envelopes (overlapping = continuous) ---
    const morphT   = sstep(0.44, 0.70, g);
    const lift     = Math.sin(sstep(0.44, 0.70, g) * Math.PI) * 0.05;
    const hlOp     = (sstep(0.10, 0.18, g) - sstep(0.40, 0.48, g)) * 0.20;
    const outDraw  = sstep(0.14, 0.28, g);
    const gridDraw = sstep(0.28, 0.44, g);
    const linesOut = 1 - sstep(0.60, 0.70, g);         // grid+outline fade as it flattens
    const outOp    = (sstep(0.14, 0.20, g)) * linesOut;
    const gridOp   = (sstep(0.28, 0.34, g)) * linesOut * 0.9;
    const bgFade   = 1 - sstep(0.44, 0.66, g) * 0.92;   // table darkens as page lifts
    const inkOp    = sstep(0.72, 0.90, g);
    const rise     = 1 + sstep(0.46, 0.66, g) * 0.035;  // subtle scale up
    const push     = 1 + sstep(0.0, 0.42, g) * 0.03;    // early ken-burns on the whole scene

    this._writePositions(morphT, lift);

    this.hlMat.opacity = Math.max(0, hlOp);
    this.outMat.opacity = Math.max(0, outOp);
    this.gridMat.opacity = Math.max(0, gridOp);
    this.bgMat.opacity = bgFade;
    if (this.inkMat) this.inkMat.opacity = inkOp;
    // page mesh only appears as it lifts off the background — before that the
    // background photo carries the image, so there is no doubled "sheet on top".
    this.pageMat.opacity = sstep(0.42, 0.47, g);

    // draw-on
    this.outGeo.setDrawRange(0, Math.max(2, Math.floor(this.perim.length * outDraw)));
    this.gridGeo.setDrawRange(0, Math.max(0, Math.floor((this.gridPairs.length / 2) * gridDraw) * 2));

    // rise (scale the rig about the plate center)
    const cx = 0.5, cy = this.aspect / 2;
    const sc = rise * push;
    this.rig.position.set(cx - cx * sc, cy - cy * sc, 0);
    this.rig.scale.set(sc, sc, 1);
    // early push on background too
    this.bg.scale.set(push, push, 1);
    this.bg.position.set(cx - cx * push, cy - cy * push, -0.2);

    this.renderer.render(this.scene, this.camera);
  }
}
