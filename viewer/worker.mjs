// Fetches, decodes and packs a .splatc scene, then depth-sorts it per view.
import { decodeContainer } from './decode_container.mjs';
import { reconstruct } from './reconstruct.mjs';

// Three RGBA32UI texels per Gaussian, 1024 Gaussians per texture row:
// [x, y, z, sh label] [4 * covariance as three half2, unused] [r, g, b, opacity].
const PER_ROW = 1024;
const TEXELS = 3;
const CODEBOOK_WIDTH = 12; // 45 coefficients padded to 48 floats

const floatView = new Float32Array(1);
const int32View = new Int32Array(floatView.buffer);

function floatToHalf(float) {
  floatView[0] = float;
  const f = int32View[0];
  const sign = (f >> 31) & 0x0001;
  const exp = (f >> 23) & 0x00ff;
  let frac = f & 0x007fffff;
  let newExp;
  if (exp == 0) {
    newExp = 0;
  } else if (exp < 113) {
    newExp = 0;
    frac |= 0x00800000;
    frac = frac >> (113 - exp);
    if (frac & 0x01000000) {
      newExp = 1;
      frac = 0;
    }
  } else if (exp < 142) {
    newExp = exp - 112;
  } else {
    newExp = 31;
    frac = 0;
  }
  return (sign << 15) | (newExp << 10) | (frac >> 13);
}

function packHalf2x16(x, y) {
  return (floatToHalf(x) | (floatToHalf(y) << 16)) >>> 0;
}

function packGaussians({ count, means, scales, quats, opacities, colors, labels }) {
  const texwidth = PER_ROW * TEXELS;
  const texheight = Math.ceil(count / PER_ROW);
  const texdata = new Uint32Array(texwidth * texheight * 4);
  const texdata_f = new Float32Array(texdata.buffer);
  for (let i = 0; i < count; i += 1) {
    const o = 12 * i;
    texdata_f[o + 0] = means[3 * i + 0];
    texdata_f[o + 1] = means[3 * i + 1];
    texdata_f[o + 2] = means[3 * i + 2];
    texdata[o + 3] = labels ? labels[i] : 0;

    const s = [scales[3 * i], scales[3 * i + 1], scales[3 * i + 2]];
    const r = [quats[4 * i], quats[4 * i + 1], quats[4 * i + 2], quats[4 * i + 3]];
    const M = [
      1.0 - 2.0 * (r[2] * r[2] + r[3] * r[3]),
      2.0 * (r[1] * r[2] + r[0] * r[3]),
      2.0 * (r[1] * r[3] - r[0] * r[2]),
      2.0 * (r[1] * r[2] - r[0] * r[3]),
      1.0 - 2.0 * (r[1] * r[1] + r[3] * r[3]),
      2.0 * (r[2] * r[3] + r[0] * r[1]),
      2.0 * (r[1] * r[3] + r[0] * r[2]),
      2.0 * (r[2] * r[3] - r[0] * r[1]),
      1.0 - 2.0 * (r[1] * r[1] + r[2] * r[2]),
    ].map((k, j) => k * s[Math.floor(j / 3)]);
    const sigma = [
      M[0] * M[0] + M[3] * M[3] + M[6] * M[6],
      M[0] * M[1] + M[3] * M[4] + M[6] * M[7],
      M[0] * M[2] + M[3] * M[5] + M[6] * M[8],
      M[1] * M[1] + M[4] * M[4] + M[7] * M[7],
      M[1] * M[2] + M[4] * M[5] + M[7] * M[8],
      M[2] * M[2] + M[5] * M[5] + M[8] * M[8],
    ];
    // Upstream's 4x lifts small covariances out of half-precision subnormals.
    texdata[o + 4] = packHalf2x16(4 * sigma[0], 4 * sigma[1]);
    texdata[o + 5] = packHalf2x16(4 * sigma[2], 4 * sigma[3]);
    texdata[o + 6] = packHalf2x16(4 * sigma[4], 4 * sigma[5]);

    texdata_f[o + 8] = colors[3 * i + 0];
    texdata_f[o + 9] = colors[3 * i + 1];
    texdata_f[o + 10] = colors[3 * i + 2];
    texdata_f[o + 11] = opacities[i];
  }
  return { texdata, texwidth, texheight };
}

function packCodebook({ codebook, codebookRows }) {
  const height = Math.max(codebookRows, 1);
  const data = new Float32Array(height * CODEBOOK_WIDTH * 4);
  for (let row = 0; row < codebookRows; row += 1) {
    data.set(codebook.subarray(45 * row, 45 * row + 45), 48 * row);
  }
  return { data, width: CODEBOOK_WIDTH, height };
}

let positions = null;
let count = 0;
let viewProj = null;
let lastProj = null;
let sortRunning = false;

function runSort(view) {
  if (!positions || !view) return;
  if (lastProj) {
    const dot = lastProj[2] * view[2] + lastProj[6] * view[6] + lastProj[10] * view[10];
    if (Math.abs(dot - 1) < 0.01) return;
  }
  const start = performance.now();
  let maxDepth = -Infinity;
  let minDepth = Infinity;
  const sizeList = new Int32Array(count);
  for (let i = 0; i < count; i += 1) {
    const depth =
      ((view[2] * positions[3 * i] + view[6] * positions[3 * i + 1] + view[10] * positions[3 * i + 2]) * 4096) | 0;
    sizeList[i] = depth;
    if (depth > maxDepth) maxDepth = depth;
    if (depth < minDepth) minDepth = depth;
  }

  // 16-bit single-pass counting sort, nearest first.
  const depthInv = (256 * 256 - 1) / (maxDepth - minDepth);
  const counts0 = new Uint32Array(256 * 256);
  for (let i = 0; i < count; i += 1) {
    sizeList[i] = ((sizeList[i] - minDepth) * depthInv) | 0;
    counts0[sizeList[i]] += 1;
  }
  const starts0 = new Uint32Array(256 * 256);
  for (let i = 1; i < 256 * 256; i += 1) starts0[i] = starts0[i - 1] + counts0[i - 1];
  const depthIndex = new Uint32Array(count);
  for (let i = 0; i < count; i += 1) depthIndex[starts0[sizeList[i]]++] = i;

  lastProj = view;
  self.postMessage({ depthIndex, viewProj: view, count, sortMs: performance.now() - start }, [depthIndex.buffer]);
}

function throttledSort() {
  if (sortRunning) return;
  sortRunning = true;
  const lastView = viewProj;
  runSort(lastView);
  setTimeout(() => {
    sortRunning = false;
    if (lastView !== viewProj) throttledSort();
  }, 0);
}

async function load(url) {
  const t0 = performance.now();
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${response.status} unable to load ${url}`);
  const buffer = await response.arrayBuffer();
  const t1 = performance.now();
  const decoded = await decodeContainer(buffer);
  const t2 = performance.now();
  const scene = reconstruct(decoded);
  const t3 = performance.now();
  const gaussians = packGaussians(scene);
  const codebook = packCodebook(scene);
  const t4 = performance.now();

  self.postMessage(
    {
      ...gaussians,
      codebook,
      count: scene.count,
      phases: { fetch: t1 - t0, decode: t2 - t1, reconstruct: t3 - t2, pack: t4 - t3 },
    },
    [gaussians.texdata.buffer, codebook.data.buffer],
  );
  positions = scene.means;
  count = scene.count;
  throttledSort();
}

self.onmessage = (e) => {
  if (e.data.url) {
    load(e.data.url).catch((error) => self.postMessage({ error: String(error && error.message ? error.message : error) }));
  } else if (e.data.view) {
    viewProj = e.data.view;
    throttledSort();
  }
};
