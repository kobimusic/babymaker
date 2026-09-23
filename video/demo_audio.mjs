/**
 * Renders the demo video's music track.
 *
 * Reads a timeline from video/frames.py (the weights and the trio at every
 * frame), morphs the real instruments at each step with the same code the
 * browser runs, and sequences Alla Turca through them, looping as the demo
 * does, so the audio in the video is the algorithm working, not a re-recording.
 *
 *   node video/demo_audio.mjs <timeline.json> <out.wav>
 */
import { readFileSync, writeFileSync } from "node:fs";
import { decodeWav } from "../web/lib/babymaker/wav.ts";
import { Analyser, Morpher } from "../web/lib/babymaker/morph.ts";
import { parseMidi } from "../web/lib/midi/smf.ts";

const [, , timelinePath, outPath] = process.argv;
const tl = JSON.parse(readFileSync(timelinePath, "utf8"));
const ROOT = new URL("..", import.meta.url).pathname;
const SR = 44100;

const read = (p) => {
  const b = readFileSync(p);
  return decodeWav(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
};

/** prepare() step 4, as the browser does on load. */
function normalise(x) {
  const win = Math.floor(0.02 * SR);
  let e = 0, best = 0;
  for (let i = 0; i < x.length; i++) {
    e += x[i] * x[i];
    if (i >= win) e -= x[i - win] * x[i - win];
    if (e > best) best = e;
  }
  return x.map((v) => v / (Math.sqrt(best / win) + 1e-12));
}

const MEDIA = `${ROOT}web/public/media/`;
const man = JSON.parse(readFileSync(`${MEDIA}manifest.json`, "utf8"));
const sources = man.instruments.map((inst) => {
  const w = read(`${MEDIA}${inst.file}`);
  return { name: inst.label, audio: normalise(w.samples), peakRmsIn: inst.peakRmsIn };
});

const midiBuf = readFileSync(`${MEDIA}${man.midi}`);
const midi = parseMidi(midiBuf.buffer.slice(midiBuf.byteOffset, midiBuf.byteOffset + midiBuf.byteLength));

const analyser = new Analyser(man.note, { oversample: 2, glIters: 2 });
const analyses = sources.map((s) => analyser.analyse(s));
process.stderr.write(`analysed ${sources.length} instruments\n`);

/** Morphers are per trio; weight vectors are quantised so renders are reused. */
const morphers = new Map();
const cache = new Map();
function morph(ids, weights) {
  const key = ids.join(",");
  let mp = morphers.get(key);
  if (!mp) {
    mp = new Morpher(analyser, ids.map((i) => sources[i]), ids.map((i) => analyses[i]));
    morphers.set(key, mp);
  }
  const q = weights.map((v) => Math.round(v * 40) / 40);
  const ck = `${key}|${q.join(",")}`;
  let buf = cache.get(ck);
  if (!buf) {
    buf = mp.render(q);
    cache.set(ck, buf);
    if (cache.size % 20 === 0) process.stderr.write(`  ${cache.size} morphs rendered\n`);
  }
  return buf;
}

/** The timeline's state at time t (frames are evenly spaced at 1/fps). */
function stateAt(t) {
  const i = Math.min(tl.frames.length - 1, Math.max(0, Math.round(t * tl.fps)));
  return tl.frames[i];
}

const total = Math.ceil((tl.duration + 2) * SR);
const mix = new Float64Array(total);

const LOOP = midi.duration + 0.6;
const passes = [];
for (let base = 0; base < tl.musicEnd; base += LOOP) {
  for (const note of midi.notes) passes.push({ ...note, start: note.start + base, end: note.end + base });
}
let n = 0;
for (const note of passes) {
  if (note.start > tl.musicEnd) break;
  const t = note.start + tl.musicStart;
  if (t > tl.duration) break;
  const st = stateAt(t);
  const src = morph(st.selection, st.weights);

  // playbackRate, as the browser's AudioBufferSourceNode does it
  const rate = Math.pow(2, (note.note - man.note) / 12);
  const vel = Math.pow(note.velocity / 127, 1.4) * 0.28;
  const start = Math.round(t * SR);
  const holdN = Math.round((note.end - note.start) * SR);
  const releaseN = Math.round(0.07 * SR);
  const attackN = Math.round(0.006 * SR);
  const lenN = Math.min(Math.round(src.length / rate), holdN + releaseN);

  for (let k = 0; k < lenN; k++) {
    const j = start + k;
    if (j >= total) break;
    const sp = k * rate;
    const i0 = Math.floor(sp);
    if (i0 + 1 >= src.length) break;
    const f = sp - i0;
    const s = src[i0] * (1 - f) + src[i0 + 1] * f;
    let g = 1;
    if (k < attackN) g = k / attackN;
    else if (k > holdN) g = Math.max(0, 1 - (k - holdN) / releaseN);
    mix[j] += s * vel * g;
  }
  n++;
}
process.stderr.write(`sequenced ${n} notes, ${cache.size} distinct morphs\n`);

// 16-bit WAV
let peak = 0;
for (const v of mix) peak = Math.max(peak, Math.abs(v));
const gain = peak > 0.95 ? 0.95 / peak : 1;
const pcm = Buffer.alloc(44 + total * 2);
pcm.write("RIFF", 0); pcm.writeUInt32LE(36 + total * 2, 4); pcm.write("WAVE", 8);
pcm.write("fmt ", 12); pcm.writeUInt32LE(16, 16); pcm.writeUInt16LE(1, 20); pcm.writeUInt16LE(1, 22);
pcm.writeUInt32LE(SR, 24); pcm.writeUInt32LE(SR * 2, 28); pcm.writeUInt16LE(2, 32); pcm.writeUInt16LE(16, 34);
pcm.write("data", 36); pcm.writeUInt32LE(total * 2, 40);
for (let i = 0; i < total; i++) {
  pcm.writeInt16LE(Math.max(-32768, Math.min(32767, Math.round(mix[i] * gain * 32767))), 44 + i * 2);
}
writeFileSync(outPath, pcm);
process.stderr.write(`wrote ${outPath} (${(total / SR).toFixed(1)}s)\n`);
