/**
 * Iterative radix-2 complex FFT on split re/im Float64Arrays, in place.
 * Sizes are powers of two; one instance per size is cached, with its
 * bit-reversal table and twiddles built once.
 */
export class FFT {
  readonly n: number;
  private readonly rev: Uint32Array;
  private readonly cos: Float64Array;
  private readonly sin: Float64Array;

  constructor(n: number) {
    if (n < 2 || (n & (n - 1)) !== 0) throw new Error(`FFT size must be a power of two, got ${n}`);
    this.n = n;
    const bits = Math.log2(n);
    this.rev = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
      let r = 0;
      for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
      this.rev[i] = r;
    }
    const half = n >> 1;
    this.cos = new Float64Array(half);
    this.sin = new Float64Array(half);
    for (let k = 0; k < half; k++) {
      const a = (2 * Math.PI * k) / n;
      this.cos[k] = Math.cos(a);
      this.sin[k] = Math.sin(a);
    }
  }

  /** Forward: X[k] = Σ x[n] e^{-2πikn/N}. Inverse divides by N. */
  transform(re: Float64Array, im: Float64Array, inverse = false): void {
    const { n, rev, cos, sin } = this;
    for (let i = 0; i < n; i++) {
      const j = rev[i];
      if (j > i) {
        let t = re[i]; re[i] = re[j]; re[j] = t;
        t = im[i]; im[i] = im[j]; im[j] = t;
      }
    }
    const sgn = inverse ? 1 : -1;
    for (let size = 2; size <= n; size <<= 1) {
      const half = size >> 1;
      const step = n / size;
      for (let start = 0; start < n; start += size) {
        for (let k = 0, w = 0; k < half; k++, w += step) {
          const wr = cos[w];
          const wi = sgn * sin[w];
          const a = start + k;
          const b = a + half;
          const tr = re[b] * wr - im[b] * wi;
          const ti = re[b] * wi + im[b] * wr;
          re[b] = re[a] - tr; im[b] = im[a] - ti;
          re[a] += tr;        im[a] += ti;
        }
      }
    }
    if (inverse) {
      const s = 1 / n;
      for (let i = 0; i < n; i++) { re[i] *= s; im[i] *= s; }
    }
  }
}

const cache = new Map<number, FFT>();

export function fft(n: number): FFT {
  let f = cache.get(n);
  if (!f) { f = new FFT(n); cache.set(n, f); }
  return f;
}
