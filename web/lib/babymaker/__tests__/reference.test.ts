/**
 * The port against the Python babymaker: the same six Sonatina sources
 * (tools/prep-babies.py), same config (env_mode=log, oversample=2,
 * gl_iters=2), ten Dirichlet-random six-way weight vectors. Compares
 * waveforms by normalised correlation at zero lag — the algorithm is
 * deterministic, so a faithful port should land very close.
 *
 * Skips itself if the reference renders are missing.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { decodeWav } from "../wav.ts";
import { Morpher, type SourceInput } from "../morph.ts";

// the Python renders in /reference, written by tools/prep-babies.py
const REF = fileURLToPath(new URL("../../../../reference", import.meta.url));
const NOTE = 76;

/** prepare() step 4: peak 20 ms RMS -> 1. The dumped WAVs are a scalar multiple of that. */
function normalise(x: Float64Array, sr: number): Float64Array {
  const win = Math.max(1, Math.floor(0.02 * sr));
  let e = 0, best = 0;
  for (let i = 0; i < x.length; i++) {
    e += x[i] * x[i];
    if (i >= win) e -= x[i - win] * x[i - win];
    if (e > best) best = e;
  }
  const prms = Math.sqrt(best / win);
  return x.map((v) => v / (prms + 1e-12));
}

const load = (p: string) => {
  const b = readFileSync(p);
  return decodeWav(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
};

function corr(a: Float64Array, b: Float64Array): number {
  const n = Math.min(a.length, b.length);
  let ab = 0, aa = 0, bb = 0;
  for (let i = 0; i < n; i++) { ab += a[i] * b[i]; aa += a[i] * a[i]; bb += b[i] * b[i]; }
  return ab / Math.sqrt(aa * bb);
}

test("JS morph matches the Python reference renders", { skip: !existsSync(REF) && "no reference renders here" }, () => {
  const meta = JSON.parse(readFileSync(join(REF, "key076_morph.json"), "utf8")).morph;
  const sources: SourceInput[] = meta.sources.map((s: { name: string; peak_rms_in: number }) => {
    const wav = load(join(REF, `key076_src_${s.name.replace(/:/g, "-")}.wav`));
    return { name: s.name, audio: normalise(wav.samples, wav.sampleRate), peakRmsIn: s.peak_rms_in };
  });

  const t0 = performance.now();
  const mp = Morpher.from(sources, NOTE, { oversample: 2, glIters: 2 });
  const analyseMs = performance.now() - t0;
  const info = mp.describe();
  console.log(`  analysis ${analyseMs.toFixed(0)} ms; q=${info.q} hop=${info.hop} frames=${info.sources.map((s) => s.frames)}`);
  assert.equal(info.q, meta.cepstral_q);
  assert.equal(info.hop, meta.hop);
  assert.deepEqual(info.sources.map((s) => s.frames), meta.sources.map((s: { frames: number }) => s.frames));

  const renders: { file: string; weights: number[] }[] = JSON.parse(readFileSync(join(REF, "key076_morph.json"), "utf8")).renders;
  assert.ok(renders.length >= 6);
  let worst = 1, sumMs = 0;
  for (const { file: f, weights: w } of renders) {
    const t1 = performance.now();
    const y = mp.render(w);
    const ms = performance.now() - t1;
    sumMs += ms;
    const ref = load(join(REF, f)).samples;
    assert.equal(y.length, ref.length, `${f}: length ${y.length} vs ${ref.length}`);
    const c = corr(y, ref);
    worst = Math.min(worst, c);
    console.log(`  ${f.padEnd(44)} corr ${c.toFixed(4)}  ${ms.toFixed(0)} ms`);
    assert.ok(c > 0.99, `${f}: correlation ${c}`);
  }
  console.log(`  worst ${worst.toFixed(4)}; mean render ${(sumMs / renders.length).toFixed(0)} ms`);
});
