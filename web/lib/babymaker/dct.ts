/**
 * Orthonormal DCT-II, DCT-III (its inverse) and DCT-IV, all via the FFT,
 * matching scipy.fft.dct(..., norm="ortho") — the babymaker's reference.
 *
 *   DCT-II : X[k] = f_k Σ x[n] cos(πk(2n+1)/2N),   f_0 = √(1/N), f_k = √(2/N)
 *   DCT-IV : X[k] = √(2/N) Σ x[n] cos(π(2n+1)(2k+1)/4N)
 *
 * DCT-II uses Makhoul's even/odd reorder and an N-point FFT; DCT-IV pairs
 * x[2m] with x[N-1-2m] into an N/2-point FFT with twiddles
 * e^{-iπm/N} before and e^{-iπ(4j+1)/4N} after (derivation in the tests,
 * which check every one of these against the O(N²) definition).
 */
import { fft } from "./fft.ts";

interface Scratch { re: Float64Array; im: Float64Array; preC: Float64Array; preS: Float64Array; postC: Float64Array; postS: Float64Array }
const scratch4 = new Map<number, Scratch>();
const scratch2 = new Map<number, { re: Float64Array; im: Float64Array; twC: Float64Array; twS: Float64Array }>();

function s4(n: number): Scratch {
  let s = scratch4.get(n);
  if (s) return s;
  const m = n >> 1;
  const preC = new Float64Array(m), preS = new Float64Array(m), postC = new Float64Array(m), postS = new Float64Array(m);
  for (let i = 0; i < m; i++) {
    const a = (-Math.PI * i) / n;               // e^{-iπm/N}
    preC[i] = Math.cos(a); preS[i] = Math.sin(a);
    const b = (-Math.PI * (4 * i + 1)) / (4 * n); // e^{-iπ(4j+1)/4N}
    postC[i] = Math.cos(b); postS[i] = Math.sin(b);
  }
  s = { re: new Float64Array(m), im: new Float64Array(m), preC, preS, postC, postS };
  scratch4.set(n, s);
  return s;
}

function s2(n: number) {
  let s = scratch2.get(n);
  if (s) return s;
  const twC = new Float64Array(n), twS = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    const a = (-Math.PI * k) / (2 * n);          // e^{-iπk/2N}
    twC[k] = Math.cos(a); twS[k] = Math.sin(a);
  }
  s = { re: new Float64Array(n), im: new Float64Array(n), twC, twS };
  scratch2.set(n, s);
  return s;
}

/** DCT-IV, orthonormal (self-inverse). `x` and `out` may be the same array. */
export function dct4(x: Float64Array, out: Float64Array, off = 0, outOff = 0, n = x.length - off): void {
  const m = n >> 1;
  const { re, im, preC, preS, postC, postS } = s4(n);
  for (let i = 0; i < m; i++) {
    const a = x[off + 2 * i];
    const b = x[off + n - 1 - 2 * i];
    // (a + ib) · e^{-iπi/N}
    re[i] = a * preC[i] - b * preS[i];
    im[i] = a * preS[i] + b * preC[i];
  }
  fft(m).transform(re, im);
  const scale = Math.sqrt(2 / n);
  for (let j = 0; j < m; j++) {
    const zr = re[j], zi = im[j];
    const yr = zr * postC[j] - zi * postS[j];
    const yi = zr * postS[j] + zi * postC[j];
    out[outOff + 2 * j] = yr * scale;
    out[outOff + n - 1 - 2 * j] = -yi * scale;
  }
}

/** DCT-II, orthonormal. */
export function dct2(x: Float64Array, out: Float64Array, off = 0, outOff = 0, n = x.length - off): void {
  const { re, im, twC, twS } = s2(n);
  const h = n >> 1;
  for (let i = 0; i < h; i++) {
    re[i] = x[off + 2 * i];              // evens forward
    re[n - 1 - i] = x[off + 2 * i + 1];  // odds reversed
  }
  im.fill(0);
  fft(n).transform(re, im);
  const f0 = Math.sqrt(1 / n), fk = Math.sqrt(2 / n);
  for (let k = 0; k < n; k++) {
    // Re(e^{-iπk/2N} V[k])
    out[outOff + k] = (re[k] * twC[k] - im[k] * twS[k]) * (k === 0 ? f0 : fk);
  }
}

/** DCT-III, orthonormal — the inverse of dct2. Reads `q` coefficients (rest zero). */
export function idct2(c: Float64Array, out: Float64Array, off = 0, outOff = 0, n = out.length - outOff, q = n): void {
  const { re, im, twC, twS } = s2(n);
  const f0 = Math.sqrt(1 / n), fk = Math.sqrt(2 / n);
  // un-normalise, then V[k] = e^{+iπk/2N} (c[k] - i c[N-k]), c[N] = 0
  const ck = (k: number) => (k < q ? c[off + k] / (k === 0 ? f0 : fk) : 0);
  for (let k = 0; k < n; k++) {
    const a = ck(k);
    const b = k === 0 ? 0 : -ck(n - k);       // imaginary part of W
    // multiply (a + ib) by e^{+iπk/2N} = (twC, -twS)
    re[k] = a * twC[k] + b * twS[k];
    im[k] = b * twC[k] - a * twS[k];
  }
  fft(n).transform(re, im, true);
  const h = n >> 1;
  for (let i = 0; i < h; i++) {
    out[outOff + 2 * i] = re[i];
    out[outOff + 2 * i + 1] = re[n - 1 - i];
  }
}

// ---- O(N²) references, for the tests only -------------------------------

export function naiveDct2(x: Float64Array): Float64Array {
  const n = x.length, out = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += x[i] * Math.cos((Math.PI * k * (2 * i + 1)) / (2 * n));
    out[k] = s * (k === 0 ? Math.sqrt(1 / n) : Math.sqrt(2 / n));
  }
  return out;
}

export function naiveDct4(x: Float64Array): Float64Array {
  const n = x.length, out = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += x[i] * Math.cos((Math.PI * (2 * i + 1) * (2 * k + 1)) / (4 * n));
    out[k] = s * Math.sqrt(2 / n);
  }
  return out;
}
