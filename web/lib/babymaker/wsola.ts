/**
 * Time-domain warp (WSOLA) — a port of the babymaker's wsola_warp.
 *
 * Output sample s is taken from around source sample srcOfOut[s]. Each
 * Hann block is picked within ±1.5 periods of its nominal position so that
 * it continues the previous block's waveform (max normalised cross-
 * correlation); 50 % overlap-add. Ratio 1 reproduces x exactly; moderate
 * ratios keep the waveform — and hence the phases — intact.
 */
export function wsolaWarp(
  x: Float64Array,
  srcOfOut: Float64Array,
  nOut: number,
  period: number,
  frame = 1024,
): Float64Array {
  const N = frame;
  const H = N >> 1;
  const win = new Float64Array(N);
  for (let n = 0; n < N; n++) win[n] = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / N); // periodic Hann
  const dmax = Math.floor(Math.min(Math.max(1.5 * period, 32), H));

  const xp = new Float64Array(x.length + N + 2 * dmax + H);
  xp.set(x);
  const csum = new Float64Array(xp.length + 1);
  for (let i = 0; i < xp.length; i++) csum[i + 1] = csum[i] + xp[i] * xp[i];

  const nx = srcOfOut.length;
  const last = x.length - 1;
  const y = new Float64Array(nOut + N);
  let prev = -1;

  const score = (d: number, nat: number): number => {
    let dot = 0;
    for (let n = 0; n < N; n++) dot += xp[d + n] * xp[nat + n];
    return dot / (Math.sqrt(csum[d + N] - csum[d]) + 1e-9);
  };

  for (let s0 = 0; s0 < nOut; s0 += H) {
    const si = s0 < nx ? s0 : nx - 1;
    const nom = Math.round(Math.min(Math.max(srcOfOut[si], 0), last));
    let p: number;
    if (prev < 0) {
      p = nom;
    } else {
      const nat = prev + H;                       // natural continuation
      const lo = Math.max(nom - dmax, 0);
      const hi = nom + dmax;
      // coarse search every 4th lag on a grid that contains the natural
      // continuation (so ratio 1 is always exact), then refine around the best
      let best = -1e300;
      p = nat;
      const first = nat - Math.floor((nat - lo) / 4) * 4;
      for (let d = first; d <= hi; d += 4) {
        const v = score(d, nat);
        if (v > best) { best = v; p = d; }
      }
      const c0 = p;
      for (let d = Math.max(lo, c0 - 3); d <= Math.min(hi, c0 + 3); d++) {
        if (d === c0) continue;
        const v = score(d, nat);
        if (v > best) { best = v; p = d; }
      }
    }
    for (let n = 0; n < N; n++) y[s0 + n] += win[n] * xp[p + n];
    prev = p;
  }
  return y.subarray(0, nOut);
}
