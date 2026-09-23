import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dct2, dct4, idct2, naiveDct2, naiveDct4 } from "../dct.ts";
import { MDCT } from "../mdct.ts";
import { parseMidi } from "../../midi/smf.ts";

function rnd(n: number, seed = 1): Float64Array {
  let s = seed >>> 0;
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    out[i] = s / 4294967296 - 0.5;
  }
  return out;
}

const maxAbsDiff = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
};

for (const n of [8, 64, 1024]) {
  test(`dct4 matches the O(N²) definition, N=${n}`, () => {
    const x = rnd(n);
    const fast = new Float64Array(n);
    dct4(x, fast);
    assert.ok(maxAbsDiff(fast, naiveDct4(x)) < 1e-9);
  });

  test(`dct4 is its own inverse, N=${n}`, () => {
    const x = rnd(n, 7);
    const a = new Float64Array(n), b = new Float64Array(n);
    dct4(x, a); dct4(a, b);
    assert.ok(maxAbsDiff(b, x) < 1e-9);
  });

  test(`dct2 matches the O(N²) definition, N=${n}`, () => {
    const x = rnd(n, 3);
    const fast = new Float64Array(n);
    dct2(x, fast);
    assert.ok(maxAbsDiff(fast, naiveDct2(x)) < 1e-9);
  });

  test(`idct2 inverts dct2, N=${n}`, () => {
    const x = rnd(n, 5);
    const c = new Float64Array(n), y = new Float64Array(n);
    dct2(x, c); idct2(c, y);
    assert.ok(maxAbsDiff(y, x) < 1e-9);
  });
}

test("idct2 with q coefficients equals zero-padding the rest", () => {
  const n = 64, q = 10;
  const x = rnd(n, 9);
  const c = new Float64Array(n);
  dct2(x, c);
  const lifted = new Float64Array(n);
  idct2(c, lifted, 0, 0, n, q);
  const padded = new Float64Array(n); padded.set(c.subarray(0, q));
  const ref = new Float64Array(n);
  idct2(padded, ref);
  assert.ok(maxAbsDiff(lifted, ref) < 1e-12);
});

for (const os of [1, 2, 4]) {
  test(`MDCT reconstructs exactly (TDAC), hop = M/${os}`, () => {
    const M = 64;
    const m = new MDCT(M, M / os);
    const x = rnd(1000, 11);
    const { X, T } = m.analyze(x);
    const y = m.synthesize(X, T, x.length);
    assert.ok(maxAbsDiff(y, x) < 1e-9, `max err ${maxAbsDiff(y, x)}`);
  });
}

test("MCLT magnitude of a pure tone peaks in the right bin", () => {
  const M = 256, sr = 8000;
  const m = new MDCT(M, M / 2);
  const f = 1000;
  const x = new Float64Array(4000);
  for (let i = 0; i < x.length; i++) x[i] = Math.sin((2 * Math.PI * f * i) / sr);
  const { X, Y, T } = m.analyze(x);
  const t = Math.floor(T / 2);
  let best = 0, bestK = 0;
  for (let k = 0; k < M; k++) {
    const mag = Math.hypot(X[t * M + k], Y[t * M + k]);
    if (mag > best) { best = mag; bestK = k; }
  }
  assert.ok(Math.abs(m.binHz(sr, bestK) - f) < sr / (2 * M));
});

test("parses the Alla Turca loop", () => {
  const buf = readFileSync(new URL("../../../public/media/alla-turca.mid", import.meta.url));
  const midi = parseMidi(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
  assert.equal(midi.ticksPerBeat, 256);
  assert.equal(midi.notes.length, 260);
  const pitches = midi.notes.map((n) => n.note);
  assert.equal(Math.min(...pitches), 67);
  assert.equal(Math.max(...pitches), 84);
  assert.ok(midi.notes.every((n) => n.end > n.start), "every note has a positive length");
  console.log(`  alla turca: ${midi.notes.length} notes, last ends at ${midi.duration.toFixed(2)}s`);
});
