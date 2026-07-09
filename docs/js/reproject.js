// Clean Sheet — reprojection hero.
// Textures the input photo onto a deforming mesh whose vertices morph from their
// curved on-the-table positions (the Coons source grid) to a flat rectangle,
// so the page appears to lift off the table and un-warp into a clean plate.

import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js";

export class Reprojector {
  constructor(canvas) {
    this.canvas = canvas;
    this.ready = false;
  }

  // demo: parsed demo.json ; photoImg: loaded HTMLImageElement of the photo
  async init(demo, photoImg) {
    const GW = demo.grid.w, GH = demo.grid.h;
    const src = demo.grid.src; // [ [u,v], ... ] normalized photo coords (0..1)
    const W = demo.photo.w, H = demo.photo.h;
    const aspect = H / W; // viewport is portrait; coord space is [0,1] x [0,aspect]
    this.aspect = aspect;

    // Flat target plate: full-bleed to match the frame images' object-fit:cover,
    // so the morph hands off seamlessly to the rectified/ink stills. Cover-fit to
    // width (page is taller than the viewport), centering vertically.
    const outWH = demo.output_aspect; // width / height
    const plateW = 1.0;
    const plateH = 1.0 / outWH;
    const x0 = 0, x1 = 1;
    const y0 = (aspect - plateH) / 2, y1 = y0 + plateH;

    const n = GW * GH;
    const positions = new Float32Array(n * 3);
    const uvs = new Float32Array(n * 2);
    this.srcPos = new Float32Array(n * 2);
    this.dstPos = new Float32Array(n * 2);

    for (let j = 0; j < GH; j++) {
      for (let i = 0; i < GW; i++) {
        const k = j * GW + i;
        const [su, sv] = src[k];
        // source: photo-normalized -> coord space (x in [0,1], y in [0,aspect])
        const sx = su, sy = sv * aspect;
        // target: flat plate
        const tx = x0 + (i / (GW - 1)) * (x1 - x0);
        const ty = y0 + (j / (GH - 1)) * (y1 - y0);
        this.srcPos[k * 2] = sx; this.srcPos[k * 2 + 1] = sy;
        this.dstPos[k * 2] = tx; this.dstPos[k * 2 + 1] = ty;
        positions[k * 3] = sx; positions[k * 3 + 1] = sy; positions[k * 3 + 2] = 0;
        // texture coord: sample the photo where this vertex sits (v flipped for GL)
        uvs[k * 2] = su; uvs[k * 2 + 1] = 1 - sv;
      }
    }

    const indices = [];
    for (let j = 0; j < GH - 1; j++) {
      for (let i = 0; i < GW - 1; i++) {
        const a = j * GW + i, b = a + 1, c = a + GW, d = c + 1;
        indices.push(a, c, b, b, c, d);
      }
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geo.setAttribute("uv", new THREE.BufferAttribute(uvs, 2));
    geo.setIndex(indices);
    this.geo = geo;
    this.posAttr = geo.getAttribute("position");

    const tex = new THREE.Texture(photoImg);
    tex.needsUpdate = true;
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;
    const mat = new THREE.MeshBasicMaterial({ map: tex, transparent: true });
    this.mesh = new THREE.Mesh(geo, mat);

    this.scene = new THREE.Scene();
    this.scene.add(this.mesh);
    // y-down ortho: top=0 maps to +1 NDC, bottom=aspect maps to -1
    this.camera = new THREE.OrthographicCamera(0, 1, 0, aspect, -1, 1);

    try {
      this.renderer = new THREE.WebGLRenderer({
        canvas: this.canvas, alpha: true, antialias: true, premultipliedAlpha: false,
      });
    } catch (e) {
      // No WebGL (old browser / disabled): the showcase falls back to a plain
      // cross-fade for the reprojection stage.
      this.failed = true;
      return;
    }
    this.renderer.setClearColor(0x000000, 0);
    this._resize();
    this.ready = true;
    this.setMorph(0);
  }

  _resize() {
    const r = this.canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.renderer.setPixelRatio(dpr);
    this.renderer.setSize(r.width, r.height, false);
  }

  // easeInOutCubic
  static ease(t) { return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; }

  // t in [0,1]: 0 = curled on table, 1 = flat plate
  setMorph(t) {
    if (!this.ready) return;
    const e = Reprojector.ease(Math.max(0, Math.min(1, t)));
    const p = this.posAttr.array, s = this.srcPos, d = this.dstPos;
    const n = s.length / 2;
    // a gentle "breathing" lift: vertices rise slightly at mid-morph
    const lift = Math.sin(e * Math.PI) * 0.012;
    for (let k = 0; k < n; k++) {
      const sx = s[k * 2], sy = s[k * 2 + 1];
      const dx = d[k * 2], dy = d[k * 2 + 1];
      p[k * 3] = sx + (dx - sx) * e;
      p[k * 3 + 1] = sy + (dy - sy) * e - lift;
    }
    this.posAttr.needsUpdate = true;
    this.renderer.render(this.scene, this.camera);
  }

  resize() { if (this.ready) { this._resize(); this.renderer.render(this.scene, this.camera); } }
}
