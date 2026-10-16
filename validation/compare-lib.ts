// Measurements for comparing two renders of the same scene (validation/compare.ts).
// Fate codes: 0 escaped, 1 captured, 2 disk, 3 unfinished.

export const FATE_NAMES = ['escaped', 'captured', 'disk', 'unfinished'] as const;
export const CAPTURED = 1;

/**
 * Pixels that sit on a fate boundary: some pixel in their 3x3 neighbourhood
 * has a different fate, in either map. The gate excludes these, because a
 * boundary that moves by a fraction of a pixel flips them.
 */
export function boundaryMask(a: Uint8Array, b: Uint8Array, width: number, height: number): Uint8Array {
  const mask = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      outer: for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx, ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
          const j = ny * width + nx;
          if (a[j] !== a[i] || b[j] !== b[i]) {
            mask[i] = 1;
            break outer;
          }
        }
      }
    }
  }
  return mask;
}

export interface FateAgreement {
  /** Fraction of all pixels with the same fate. */
  all: number;
  /** Fraction of pixels away from any fate boundary (the gate metric). */
  interior: number;
  interiorPixels: number;
  boundaryPixels: number;
  /** confusion[a][b]: pixels with fate a in the first map and b in the second. */
  confusion: number[][];
}

export function fateAgreement(a: Uint8Array, b: Uint8Array, width: number, height: number): FateAgreement {
  const mask = boundaryMask(a, b, width, height);
  const confusion = FATE_NAMES.map(() => FATE_NAMES.map(() => 0));
  let same = 0, interior = 0, interiorSame = 0;
  for (let i = 0; i < a.length; i++) {
    confusion[Math.min(a[i], 3)][Math.min(b[i], 3)]++;
    const eq = a[i] === b[i];
    if (eq) same++;
    if (!mask[i]) {
      interior++;
      if (eq) interiorSame++;
    }
  }
  return {
    all: same / a.length,
    interior: interior ? interiorSame / interior : 1,
    interiorPixels: interior,
    boundaryPixels: a.length - interior,
    confusion,
  };
}

/** Squared Euclidean distance transform of one line (Felzenszwalb & Huttenlocher 2012). */
function edt1d(f: Float64Array, n: number, d: Float64Array, v: Int32Array, z: Float64Array) {
  let k = 0;
  v[0] = 0;
  z[0] = -Infinity;
  z[1] = Infinity;
  for (let q = 1; q < n; q++) {
    if (f[q] === Infinity) continue;
    if (f[v[k]] === Infinity) {
      v[k] = q;
      continue;
    }
    let s = (f[q] + q * q - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
    while (s <= z[k]) {
      k--;
      s = (f[q] + q * q - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
    }
    k++;
    v[k] = q;
    z[k] = s;
    z[k + 1] = Infinity;
  }
  if (f[v[0]] === Infinity) {
    d.fill(Infinity, 0, n);
    return;
  }
  k = 0;
  for (let q = 0; q < n; q++) {
    while (z[k + 1] < q) k++;
    d[q] = (q - v[k]) * (q - v[k]) + f[v[k]];
  }
}

/** Distance (pixels) from every pixel to the nearest pixel where feature is set. */
export function distanceTransform(feature: Uint8Array, width: number, height: number): Float64Array {
  const n = Math.max(width, height);
  const f = new Float64Array(n), d = new Float64Array(n), z = new Float64Array(n + 1);
  const v = new Int32Array(n);
  const grid = new Float64Array(width * height);
  for (let i = 0; i < grid.length; i++) grid[i] = feature[i] ? 0 : Infinity;
  for (let x = 0; x < width; x++) {
    for (let y = 0; y < height; y++) f[y] = grid[y * width + x];
    edt1d(f, height, d, v, z);
    for (let y = 0; y < height; y++) grid[y * width + x] = d[y];
  }
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) f[x] = grid[y * width + x];
    edt1d(f, width, d, v, z);
    for (let x = 0; x < width; x++) grid[y * width + x] = Math.sqrt(d[x]);
  }
  return grid;
}

/** Captured pixels with a non-captured 4-neighbour: the edge of the black-hole shadow (and of anything covering it). */
export function capturedEdge(fate: Uint8Array, width: number, height: number): Uint8Array {
  const edge = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      if (fate[i] !== CAPTURED) continue;
      if (
        (x > 0 && fate[i - 1] !== CAPTURED) ||
        (x < width - 1 && fate[i + 1] !== CAPTURED) ||
        (y > 0 && fate[i - width] !== CAPTURED) ||
        (y < height - 1 && fate[i + width] !== CAPTURED)
      ) {
        edge[i] = 1;
      }
    }
  }
  return edge;
}

export interface EdgeOffset {
  /** Mean over edge pixels of both maps of the distance to the other map's edge. */
  meanPx: number;
  p99Px: number;
  maxPx: number;
  edgePixels: number;
}

/** Symmetric distance between the captured-region edges of two fate maps, or null without a shadow. */
export function shadowEdgeOffset(a: Uint8Array, b: Uint8Array, width: number, height: number): EdgeOffset | null {
  const ea = capturedEdge(a, width, height), eb = capturedEdge(b, width, height);
  const da = distanceTransform(ea, width, height), db = distanceTransform(eb, width, height);
  const dists: number[] = [];
  for (let i = 0; i < ea.length; i++) {
    if (ea[i]) dists.push(db[i]);
    if (eb[i]) dists.push(da[i]);
  }
  if (!dists.length || !dists.every(Number.isFinite)) return null;
  dists.sort((x, y) => x - y);
  return {
    meanPx: dists.reduce((s, x) => s + x, 0) / dists.length,
    p99Px: dists[Math.floor(0.99 * (dists.length - 1))],
    maxPx: dists[dists.length - 1],
    edgePixels: dists.length,
  };
}

/** Display transform for comparing HDR images: Reinhard x / (1 + x) and gamma 2.2, to 8 bits. */
export function toneMap(hdr: Float32Array): Uint8Array {
  const out = new Uint8Array(hdr.length);
  for (let i = 0; i < hdr.length; i++) {
    const x = Math.max(0, hdr[i]);
    out[i] = Math.round(255 * Math.pow(x / (1 + x), 1 / 2.2));
  }
  return out;
}

export interface ImageDifference {
  meanAbsLevels: number;
  pctOver8: number;
  pctOver32: number;
}

/** Differences between two 8-bit RGB images (max over channels per pixel). */
export function imageDifference(a: Uint8Array, b: Uint8Array): ImageDifference {
  const n = a.length / 3;
  let sum = 0, over8 = 0, over32 = 0;
  for (let i = 0; i < n; i++) {
    let m = 0;
    for (let c = 0; c < 3; c++) {
      const d = Math.abs(a[3 * i + c] - b[3 * i + c]);
      sum += d;
      if (d > m) m = d;
    }
    if (m > 8) over8++;
    if (m > 32) over32++;
  }
  return { meanAbsLevels: sum / (3 * n), pctOver8: (100 * over8) / n, pctOver32: (100 * over32) / n };
}
