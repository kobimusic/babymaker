/**
 * Lapped DCT transforms — a port of the babymaker's lapped.py.
 *
 * MDCT frame of a 2M-sample block (sine window w):
 *   X[k] = √(2/M) Σ w[n] x[n] cos(π/M (n + ½ + M/2)(k + ½))
 * The MDST uses sin(); X + jY is the MCLT, whose magnitude and phase behave
 * like an STFT's. With the sine window, IMDCT + overlap-add is exact
 * (TDAC), so a morph built in the MCLT domain is written back through the
 * real (DCT) part alone.
 *
 * Both are computed as a DCT-IV of the TDAC-folded block:
 *   MDCT(a, b, c, d) = DCT-IV(-c_r - d, a - b_r)      (quarters of the block)
 *   MDST(x)[k]       = -(-1)^k MDCT(reverse(x))[k]
 * and the IMDCT unfolds DCT-IV(X) = (v1, v2) to (v2, -v2_r, -v1_r, -v1).
 *
 * hop = M / os: the os interleaved hop-M lattices each reconstruct on their
 * own, so overlap-add / os is still exact, and modified coefficients are
 * averaged over os lattices.
 */
import { dct4 } from "./dct.ts";

export interface Spectra { X: Float64Array; Y: Float64Array; T: number }

export class MDCT {
  readonly M: number;
  readonly hop: number;
  readonly os: number;
  readonly pad: number;
  readonly window: Float64Array;
  readonly binOmega: Float64Array;
  private readonly alt: Float64Array;
  private readonly frame: Float64Array;
  private readonly folded: Float64Array;

  constructor(M = 1024, hop = M / 4) {
    if (M % hop !== 0) throw new Error("hop must divide M");
    this.M = M;
    this.hop = hop;
    this.os = M / hop;
    this.pad = 2 * M;
    this.window = new Float64Array(2 * M);
    for (let n = 0; n < 2 * M; n++) this.window[n] = Math.sin((Math.PI * (n + 0.5)) / (2 * M));
    this.binOmega = new Float64Array(M);
    this.alt = new Float64Array(M);
    for (let k = 0; k < M; k++) {
      this.binOmega[k] = (Math.PI * (k + 0.5)) / M;
      this.alt[k] = k % 2 === 0 ? -1 : 1;
    }
    this.frame = new Float64Array(2 * M);
    this.folded = new Float64Array(M);
  }

  binHz(sr: number, k: number): number {
    return (this.binOmega[k] * sr) / (2 * Math.PI);
  }

  nFrames(nSamples: number): number {
    return Math.ceil((this.pad + nSamples + this.M) / this.hop) + 1;
  }

  /** Windowed frame t of x into `dst`. */
  private loadFrame(x: Float64Array, t: number, dst: Float64Array): void {
    const { M, hop, pad, window } = this;
    const start = t * hop - pad;
    for (let n = 0; n < 2 * M; n++) {
      const i = start + n;
      dst[n] = i >= 0 && i < x.length ? x[i] * window[n] : 0;
    }
  }

  private fold(fr: Float64Array, reverse: boolean): Float64Array {
    const { M, folded } = this;
    const h = M >> 1;
    const at = reverse ? (i: number) => fr[2 * M - 1 - i] : (i: number) => fr[i];
    for (let i = 0; i < h; i++) {
      // a = [0,h) b = [h,2h) c = [2h,3h) d = [3h,4h)
      folded[i] = -at(3 * h - 1 - i) - at(3 * h + i);   // -c_r - d
      folded[h + i] = at(i) - at(2 * h - 1 - i);         // a - b_r
    }
    return folded;
  }

  /** MDCT and MDST of every frame, each (T × M) row-major. */
  analyze(x: Float64Array): Spectra {
    const { M, frame, alt } = this;
    const T = this.nFrames(x.length);
    const X = new Float64Array(T * M), Y = new Float64Array(T * M);
    for (let t = 0; t < T; t++) {
      this.loadFrame(x, t, frame);
      dct4(this.fold(frame, false), X, 0, t * M, M);
      dct4(this.fold(frame, true), Y, 0, t * M, M);
      for (let k = 0; k < M; k++) Y[t * M + k] *= alt[k];
    }
    return { X, Y, T };
  }

  /** IMDCT + overlap-add of (T × M) MDCT coefficients. */
  synthesize(X: Float64Array, T: number, nSamples: number): Float64Array {
    const { M, hop, pad, window, os } = this;
    const h = M >> 1;
    const v = new Float64Array(M);
    const y = new Float64Array((T - 1) * hop + 2 * M);
    for (let t = 0; t < T; t++) {
      dct4(X, v, t * M, 0, M);
      const base = t * hop;
      // block = (v2, -v2_r, -v1_r, -v1) · window
      for (let i = 0; i < h; i++) {
        y[base + i]         += v[h + i]         * window[i];
        y[base + h + i]     += -v[M - 1 - i]    * window[h + i];
        y[base + 2 * h + i] += -v[h - 1 - i]    * window[2 * h + i];
        y[base + 3 * h + i] += -v[i]            * window[3 * h + i];
      }
    }
    const out = new Float64Array(nSamples);
    const s = 1 / os;
    for (let i = 0; i < nSamples; i++) {
      const j = pad + i;
      out[i] = j < y.length ? y[j] * s : 0;
    }
    return out;
  }
}
