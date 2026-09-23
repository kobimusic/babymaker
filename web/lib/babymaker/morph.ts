/**
 * N-way morphing of prepared sources in the MCLT (lapped DCT) domain — a
 * port of the babymaker's morph.py, in the configuration the browser runs:
 * `synth="mix"`, `env_mode="log"`.
 *
 * Per source, every frame is split into
 *   level          – frame RMS in dB                      (temporal envelope)
 *   envelope       – DCT-cepstral true envelope of the frame's spectral shape
 *   fine structure – shape / envelope: the harmonic comb and noise detail
 * Sources are aligned in time by DTW on (level, low-order cepstrum) against
 * source 0. For a weight vector w the morph is
 *   time:     the output timeline is the w-average of the aligned timelines,
 *             so attack / decay / release lengths interpolate;
 *   level:    w-average in dB;
 *   envelope: w-average in dB (the Python default is an optimal-transport
 *             barycentre; `env_mode="log"` is its documented alternative);
 *   fine:     w-average in the loudness-like domain |a|^γ, γ = 0.6;
 *   phase:    from a real signal — every source WSOLA-warped onto the
 *             output timeline and summed with the weights; the mix's MCLT
 *             phases are kept while its magnitudes are replaced by the
 *             model's, then Griffin-Lim iterations make the two agree.
 */
import { dct2, idct2 } from "./dct.ts";
import { MDCT } from "./mdct.ts";
import { wsolaWarp } from "./wsola.ts";

export interface MorphConfig {
  sr: number;
  frame: number;          // MDCT M; 0 = auto from pitch (1024 / 2048)
  oversample: number;     // hop = M / oversample
  cepFrac: number;        // cepstral cutoff q = cepFrac · sr / f0
  cepMin: number;
  cepMax: number;
  envIters: number;
  floorDb: number;        // magnitude floor below the source's peak bin
  fineGamma: number;      // fine-structure domain exponent; 0 = log
  dtwLevelW: number;
  dtwCepW: number;
  dtwNcep: number;
  dtwOffdiagPenalty: number;
  dtwSmoothS: number;
  levelFloorDb: number;
  maxStretch: number;     // slope limit of each source's time warp
  attackS: number;        // sources play at natural speed this long after onset
  wsolaFrame: number;
  mixGainMaxDb: number;   // cap on |target| / |mix| when taking the mix's phase
  glIters: number;
  normalizeWeights: boolean;
}

/** The Python defaults, except where the browser config differs (marked). */
export const DEFAULT_CONFIG: MorphConfig = {
  sr: 44100,
  frame: 0,
  oversample: 2,          // python default 4
  cepFrac: 0.3,
  cepMin: 12,
  cepMax: 160,
  envIters: 30,
  floorDb: -100,
  fineGamma: 0.6,
  dtwLevelW: 0.12,
  dtwCepW: 1.0,
  dtwNcep: 12,
  dtwOffdiagPenalty: 0.15,
  dtwSmoothS: 0.08,
  levelFloorDb: -80,
  maxStretch: 2.5,
  attackS: 0.1,
  wsolaFrame: 1024,
  mixGainMaxDb: 20,
  glIters: 2,             // python default 4
  normalizeWeights: true,
};

export interface SourceInput {
  name: string;
  /** prepared mono audio at cfg.sr: pitch-corrected, onset-trimmed, peak short-time RMS = 1 */
  audio: Float64Array;
  /** the level the source had before normalisation — restored, dB-interpolated, on output */
  peakRmsIn: number;
}

export interface Analysis {
  T: number;
  levelDb: Float64Array;   // (T)
  env: Float64Array;       // (T×M) log envelope of the unit-RMS shape
  finePow: Float64Array;   // (T×M) exp(γ · fine)
  feat: Float64Array;      // (T×d) DTW features
  d: number;
  nSamples: number;
}

export const midiToHz = (note: number) => 440 * Math.pow(2, (note - 69) / 12);

// ---------------------------------------------------------------- helpers

/** np.interp: linear, clamped at both ends; xp strictly increasing. */
function interp1(x: number, xp: Float64Array, fp: Float64Array): number {
  const n = xp.length;
  if (x <= xp[0]) return fp[0];
  if (x >= xp[n - 1]) return fp[n - 1];
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (xp[mid] <= x) lo = mid; else hi = mid; }
  const f = (x - xp[lo]) / (xp[hi] - xp[lo]);
  return fp[lo] + f * (fp[hi] - fp[lo]);
}

/** Row t0·(1-f) + row t1·f of a (T×M) matrix, accumulated into `out` with weight w. */
function addInterpRow(out: Float64Array, outOff: number, mat: Float64Array, T: number, M: number, pos: number, w: number): void {
  const i0 = Math.min(Math.max(Math.floor(pos), 0), T - 1);
  const i1 = Math.min(i0 + 1, T - 1);
  const f = pos - i0;
  const a = w * (1 - f), b = w * f;
  const o0 = i0 * M, o1 = i1 * M;
  for (let k = 0; k < M; k++) out[outOff + k] += a * mat[o0 + k] + b * mat[o1 + k];
}

function interpRow1(vec: Float64Array, pos: number): number {
  const T = vec.length;
  const i0 = Math.min(Math.max(Math.floor(pos), 0), T - 1);
  const i1 = Math.min(i0 + 1, T - 1);
  const f = pos - i0;
  return vec[i0] * (1 - f) + vec[i1] * f;
}

/**
 * Roebel & Rodet's true envelope on every row of a (T×M) log-magnitude
 * matrix: cepstral smoothing iterated so the envelope rides over the
 * harmonic peaks instead of through them. The convergence test is global
 * over all frames, as in the reference.
 */
function trueEnvelope(logMag: Float64Array, T: number, M: number, q: number, iters: number, tolDb = 0.3): Float64Array {
  const env = new Float64Array(T * M);
  const next = new Float64Array(T * M);
  const cep = new Float64Array(M);
  const row = new Float64Array(M);
  const tol = (tolDb * Math.LN10) / 20;

  const smooth = (src: Float64Array, dst: Float64Array, clampTo: Float64Array | null) => {
    for (let t = 0; t < T; t++) {
      const o = t * M;
      if (clampTo) for (let k = 0; k < M; k++) row[k] = Math.max(src[o + k], clampTo[o + k]);
      else row.set(src.subarray(o, o + M));
      dct2(row, cep);
      idct2(cep, dst, 0, o, M, q);
    }
  };

  smooth(logMag, env, null);
  for (let it = 0; it < iters; it++) {
    smooth(logMag, next, env);
    let maxDiff = 0;
    for (let i = 0; i < env.length; i++) {
      const d = Math.abs(next[i] - env[i]);
      if (d > maxDiff) maxDiff = d;
    }
    env.set(next);
    if (maxDiff < tol) break;
  }
  return env;
}

// ---------------------------------------------------------------- DTW

function dtwMap(ref: Float64Array, other: Float64Array, d: number, nRef: number, nOther: number, pen: number): Float64Array {
  // cost matrix, then the accumulated-cost recursion with a penalty on the
  // off-diagonal (time-stretching) steps
  const D = new Float64Array((nRef + 1) * (nOther + 1)).fill(Infinity);
  const step = new Int8Array(nRef * nOther);
  const W = nOther + 1;
  D[0] = 0;
  for (let i = 1; i <= nRef; i++) {
    for (let j = 1; j <= nOther; j++) {
      let c = 0;
      for (let k = 0; k < d; k++) { const x = ref[(i - 1) * d + k] - other[(j - 1) * d + k]; c += x * x; }
      c = Math.sqrt(c);
      const a = D[(i - 1) * W + (j - 1)], b = D[(i - 1) * W + j] + pen, e = D[i * W + (j - 1)] + pen;
      if (a <= b && a <= e) { D[i * W + j] = a + c; step[(i - 1) * nOther + (j - 1)] = 0; }
      else if (b <= e)      { D[i * W + j] = b + c; step[(i - 1) * nOther + (j - 1)] = 1; }
      else                  { D[i * W + j] = e + c; step[(i - 1) * nOther + (j - 1)] = 2; }
    }
  }
  // backtrack, then average the path's j over each i (path_to_map)
  const sums = new Float64Array(nRef), cnt = new Float64Array(nRef);
  let i = nRef - 1, j = nOther - 1;
  sums[i] += j; cnt[i]++;
  while (i > 0 || j > 0) {
    const s = step[i * nOther + j];
    if (s === 0 && i > 0 && j > 0) { i--; j--; }
    else if (s === 1 && i > 0) i--;
    else if (j > 0) j--;
    else i--;
    sums[i] += j; cnt[i]++;
  }
  const g = new Float64Array(nRef);
  for (let t = 0; t < nRef; t++) g[t] = sums[t] / Math.max(cnt[t], 1);
  g[0] = 0;                               // onsets coincide by construction
  for (let t = 1; t < nRef; t++) g[t] = Math.max(g[t], g[t - 1]);
  for (let t = 0; t < nRef; t++) g[t] = Math.min(Math.max(g[t], 0), nOther - 1);
  return g;
}

/** Gaussian-smooth a monotone time map, keeping both end points and monotonicity. */
function smoothMap(g: Float64Array, sigma: number): Float64Array {
  const n = g.length;
  if (sigma < 0.5 || n < 3) return g;
  const r = Math.floor(3 * sigma);
  const k = new Float64Array(2 * r + 1);
  let ks = 0;
  for (let i = -r; i <= r; i++) { k[i + r] = Math.exp(-0.5 * (i / sigma) ** 2); ks += k[i + r]; }
  for (let i = 0; i < k.length; i++) k[i] /= ks;
  // odd reflection at both ends
  const padded = new Float64Array(n + 2 * r);
  for (let i = 0; i < r; i++) padded[i] = 2 * g[0] - g[r - i];
  padded.set(g, r);
  for (let i = 0; i < r; i++) padded[n + r + i] = 2 * g[n - 1] - g[n - 2 - i];
  const gs = new Float64Array(n);
  for (let t = 0; t < n; t++) {
    let s = 0;
    for (let i = 0; i < k.length; i++) s += padded[t + i] * k[k.length - 1 - i]; // convolution
    gs[t] = s;
  }
  const a = g[0] - gs[0], b = g[n - 1] - gs[n - 1];
  let gmin = Infinity, gmax = -Infinity;
  for (let t = 0; t < n; t++) { gmin = Math.min(gmin, g[t]); gmax = Math.max(gmax, g[t]); }
  for (let t = 0; t < n; t++) gs[t] += a + ((b - a) * t) / (n - 1);   // pin the ends
  for (let t = 1; t < n; t++) gs[t] = Math.max(gs[t], gs[t - 1]);
  for (let t = 0; t < n; t++) gs[t] = Math.min(Math.max(gs[t], gmin), gmax);
  return gs;
}

// ---------------------------------------------------------------- analysis

/**
 * The per-key analysis context: transform, cepstral order, pitch. Analyse
 * every source once; a Morpher then aligns any subset of them.
 */
export class Analyser {
  readonly cfg: MorphConfig;
  readonly note: number;
  readonly f0: number;
  readonly mdct: MDCT;
  readonly q: number;

  constructor(note: number, cfg: Partial<MorphConfig> = {}) {
    this.cfg = { ...DEFAULT_CONFIG, ...cfg };
    this.note = note;
    this.f0 = midiToHz(note);
    const M = this.cfg.frame || (this.f0 < 110 ? 2048 : 1024);
    this.mdct = new MDCT(M, M / this.cfg.oversample);
    this.q = Math.floor(Math.min(Math.max((this.cfg.cepFrac * this.cfg.sr) / this.f0, this.cfg.cepMin), this.cfg.cepMax));
  }

  analyse(src: SourceInput): Analysis {
    const { cfg, mdct } = this;
    const M = mdct.M;
    const { X, Y, T } = mdct.analyze(src.audio);

    const levelDb = new Float64Array(T);
    const logShape = new Float64Array(T * M);
    let peakDb = -Infinity, shapeMax = 0;
    for (let t = 0; t < T; t++) {
      let e = 0;
      for (let k = 0; k < M; k++) { const m = Math.hypot(X[t * M + k], Y[t * M + k]); logShape[t * M + k] = m; e += m * m; }
      const rms = Math.sqrt(e / M);
      levelDb[t] = 20 * Math.log10(Math.max(rms, 1e-12));
      peakDb = Math.max(peakDb, levelDb[t]);
      const inv = 1 / Math.max(rms, 1e-12);
      for (let k = 0; k < M; k++) { const s = logShape[t * M + k] * inv; logShape[t * M + k] = s; if (s > shapeMax) shapeMax = s; }
    }
    for (let t = 0; t < T; t++) levelDb[t] = Math.max(levelDb[t], peakDb + cfg.levelFloorDb);
    const floor = shapeMax * Math.pow(10, cfg.floorDb / 20);
    for (let i = 0; i < logShape.length; i++) logShape[i] = Math.log(Math.max(logShape[i], floor));

    const env = trueEnvelope(logShape, T, M, this.q, cfg.envIters);
    const finePow = new Float64Array(T * M);
    const g = cfg.fineGamma;
    for (let i = 0; i < finePow.length; i++) {
      const fine = logShape[i] - env[i];
      finePow[i] = g > 0 ? Math.exp(g * fine) : fine;
    }

    // DTW features: level + low-order cepstrum of the envelope
    const d = 1 + cfg.dtwNcep;
    const feat = new Float64Array(T * d);
    const cep = new Float64Array(M);
    for (let t = 0; t < T; t++) {
      dct2(env, cep, t * M, 0, M);
      feat[t * d] = (levelDb[t] - peakDb) * cfg.dtwLevelW;
      for (let c = 1; c <= cfg.dtwNcep; c++) feat[t * d + c] = cep[c] * cfg.dtwCepW;
    }
    return { T, levelDb, env, finePow, feat, d, nSamples: src.audio.length };
  }
}

// ---------------------------------------------------------------- the morpher

/** Aligns a set of analysed sources; `render(weights)` synthesises any point between them. */
export class Morpher {
  readonly cfg: MorphConfig;
  readonly sources: SourceInput[];
  readonly note: number;
  readonly f0: number;
  readonly mdct: MDCT;
  readonly q: number;
  private readonly analyses: Analysis[];
  private readonly maps: Float64Array[];   // per source: ref frame -> source frame

  constructor(analyser: Analyser, sources: SourceInput[], analyses: Analysis[]) {
    if (sources.length < 1) throw new Error("need at least one source");
    if (sources.length !== analyses.length) throw new Error("one analysis per source");
    this.cfg = analyser.cfg;
    this.note = analyser.note;
    this.f0 = analyser.f0;
    this.mdct = analyser.mdct;
    this.q = analyser.q;
    this.sources = sources;
    this.analyses = analyses;
    this.maps = this.align();
  }

  /** Analyse and align in one go. */
  static from(sources: SourceInput[], note: number, cfg: Partial<MorphConfig> = {}): Morpher {
    const an = new Analyser(note, cfg);
    return new Morpher(an, sources, sources.map((s) => an.analyse(s)));
  }

  get n(): number { return this.sources.length; }

  // -- time alignment against source 0
  private align(): Float64Array[] {
    const ref = this.analyses[0];
    const sigma = (this.cfg.dtwSmoothS * this.cfg.sr) / this.mdct.hop;
    return this.analyses.map((a) => {
      if (a === ref) return Float64Array.from({ length: ref.T }, (_, t) => t);
      const g = dtwMap(ref.feat, a.feat, ref.d, ref.T, a.T, this.cfg.dtwOffdiagPenalty);
      return smoothMap(g, sigma);
    });
  }

  private weights(w: ArrayLike<number>): Float64Array {
    if (w.length !== this.n) throw new Error(`expected ${this.n} weights, got ${w.length}`);
    const out = Float64Array.from(w);
    if (this.cfg.normalizeWeights) {
      let s = 0;
      for (const v of out) s += v;
      if (Math.abs(s) < 1e-9) throw new Error("weights sum to zero");
      for (let i = 0; i < out.length; i++) out[i] /= s;
    }
    return out;
  }

  /** For weights w: (J output frames, positions[i][j] = source frame of output frame j). */
  private timeline(w: Float64Array): { J: number; pos: Float64Array[] } {
    const nRef = this.maps[0].length;
    const tau = new Float64Array(nRef);
    for (let t = 0; t < nRef; t++) {
      let s = 0;
      for (let i = 0; i < this.n; i++) s += w[i] * this.maps[i][t];
      tau[t] = s;
    }
    for (let t = 1; t < nRef; t++) tau[t] = Math.max(tau[t], tau[t - 1]);
    for (let t = 0; t < nRef; t++) tau[t] += t * 1e-6;                  // strictly increasing
    const J = Math.floor(tau[nRef - 1]) + 1;
    const refIdx = Float64Array.from({ length: nRef }, (_, t) => t);
    const tRef = new Float64Array(J);
    for (let j = 0; j < J; j++) tRef[j] = interp1(j, tau, refIdx);
    const pos = this.maps.map((g) => {
      const p = new Float64Array(J);
      for (let j = 0; j < J; j++) p[j] = interp1(tRef[j], refIdx, g);
      return p;
    });
    return { J, pos };
  }

  /** Clamp each source's warp slope to [1/maxStretch, maxStretch], and to exactly 1 during the attack. */
  private limitWarp(pos: Float64Array[], J: number): Float64Array[] {
    const { cfg, mdct } = this;
    const nAtt = Math.round((cfg.attackS * cfg.sr) / mdct.hop);
    const smax = Math.max(1, cfg.maxStretch);
    return pos.map((p, i) => {
      const last = this.analyses[i].T - 1;
      const out = new Float64Array(J);
      out[0] = 0;
      for (let j = 1; j < J; j++) {
        const s = j <= nAtt ? 1 : smax;
        const v = j < p.length ? p[j] : p[p.length - 1];
        out[j] = Math.min(Math.max(v, out[j - 1] + 1 / s), out[j - 1] + s, last);
      }
      return out;
    });
  }

  /** Weighted sum of the sources, each warped in the time domain onto the output timeline. */
  private coherentMix(w: Float64Array, pos: Float64Array[], J: number, nOut: number): Float64Array {
    const { mdct, cfg } = this;
    const period = cfg.sr / this.f0;
    const mix = new Float64Array(nOut);
    const outCenters = Float64Array.from({ length: J }, (_, j) => j * mdct.hop - mdct.M);
    const srcOfOut = new Float64Array(nOut);
    for (let i = 0; i < this.n; i++) {
      if (w[i] === 0) continue;
      const audio = this.sources[i].audio;
      const centers = Float64Array.from(pos[i], (p) => p * mdct.hop - mdct.M);
      for (let s = 0; s < nOut; s++) srcOfOut[s] = Math.min(Math.max(interp1(s, outCenters, centers), 0), audio.length - 1);
      const y = wsolaWarp(audio, srcOfOut, nOut, period, cfg.wsolaFrame);
      for (let s = 0; s < nOut; s++) mix[s] += w[i] * y[s];
    }
    return mix;
  }

  /** Synthesise the morph for `weights`. Mono float64 at cfg.sr. */
  render(weights: ArrayLike<number>): Float64Array {
    const { cfg, mdct, analyses: A } = this;
    const M = mdct.M;
    const w = this.weights(weights);
    let { J, pos } = this.timeline(w);

    let nOut = 0;
    for (let i = 0; i < this.n; i++) nOut += w[i] * A[i].nSamples;
    nOut = Math.max(mdct.hop, Math.round(nOut));

    J = mdct.nFrames(nOut);                    // the analysis grid of an nOut signal
    pos = this.limitWarp(pos, J);

    // gather time-warped features: only gathers and weighted sums here
    const level = new Float64Array(J);
    const env = new Float64Array(J * M);
    const fp = new Float64Array(J * M);
    for (let i = 0; i < this.n; i++) {
      for (let j = 0; j < J; j++) {
        level[j] += w[i] * interpRow1(A[i].levelDb, pos[i][j]);
        addInterpRow(env, j * M, A[i].env, A[i].T, M, pos[i][j], w[i]);
        addInterpRow(fp, j * M, A[i].finePow, A[i].T, M, pos[i][j], w[i]);
      }
    }
    const g = cfg.fineGamma;
    const mag = new Float64Array(J * M);
    for (let j = 0; j < J; j++) {
      const lin = Math.pow(10, level[j] / 20);
      for (let k = 0; k < M; k++) {
        const i = j * M + k;
        const fine = g > 0 ? Math.log(Math.max(fp[i], 1e-12)) / g : fp[i];
        mag[i] = Math.exp(env[i] + fine) * lin;
      }
    }

    // the magnitude model says what; the waveform comes from a coherent mix
    const mix = this.coherentMix(w, pos, J, nOut);
    let { X, Y } = mdct.analyze(mix);
    const gmax = Math.pow(10, cfg.mixGainMaxDb / 20);
    for (let i = 0; i < X.length; i++) {
      const r = Math.hypot(X[i], Y[i]);
      X[i] *= Math.min(mag[i] / Math.max(r, 1e-12), gmax);
    }
    for (let it = 0; it < cfg.glIters; it++) {          // make magnitude and phase agree
      const s = mdct.analyze(mdct.synthesize(X, J, nOut));
      X = s.X; Y = s.Y;
      for (let i = 0; i < X.length; i++) X[i] *= mag[i] / Math.max(Math.hypot(X[i], Y[i]), 1e-12);
    }
    const y = mdct.synthesize(X, J, nOut);
    return this.finish(y, w);
  }

  /** Restore the dB-interpolated original level and tidy the tail. */
  private finish(y: Float64Array, w: Float64Array): Float64Array {
    const { cfg } = this;
    const win = Math.max(1, Math.floor(0.02 * cfg.sr));
    let prms: number;
    if (y.length > win) {
      let e = 0, best = 0;
      for (let i = 0; i < y.length; i++) {
        e += y[i] * y[i];
        if (i >= win) e -= y[i - win] * y[i - win];
        if (i >= win - 1 && e > best) best = e;
      }
      prms = Math.sqrt(best / win) + 1e-12;
    } else {
      let e = 0;
      for (const v of y) e += v * v;
      prms = Math.sqrt(e / Math.max(y.length, 1)) + 1e-12;
    }
    let logT = 0;
    for (let i = 0; i < this.n; i++) logT += w[i] * Math.log(Math.max(this.sources[i].peakRmsIn, 1e-6));
    const gain = Math.exp(logT) / prms;
    for (let i = 0; i < y.length; i++) y[i] *= gain;
    const nf = Math.min(y.length, Math.floor(0.01 * cfg.sr));
    if (nf > 1) for (let i = 0; i < nf; i++) y[y.length - nf + i] *= 1 - i / (nf - 1);
    return y;
  }

  describe() {
    return {
      note: this.note, f0: this.f0, frame: this.mdct.M, hop: this.mdct.hop, q: this.q,
      sources: this.sources.map((s, i) => ({ name: s.name, frames: this.analyses[i].T, seconds: s.audio.length / this.cfg.sr })),
    };
  }
}
