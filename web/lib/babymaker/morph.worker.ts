/// <reference lib="webworker" />
/**
 * The morph runs off the main thread. Every instrument is analysed once at
 * load (the slow part); a render names which three are on the dots and the
 * weights between them — the trio is aligned the first time it is asked
 * for and cached until the dots change instrument. Requests carry an id so
 * a stale render that lands after a newer one can be dropped.
 */
import { Analyser, Morpher, type Analysis, type MorphConfig, type SourceInput } from "./morph.ts";

export type WorkerIn =
  | { type: "init"; sources: SourceInput[]; note: number; cfg?: Partial<MorphConfig> }
  | { type: "render"; id: number; ids: number[]; weights: number[] };

export type WorkerOut =
  | { type: "progress"; done: number; total: number }
  | { type: "ready"; frames: number[]; ms: number }
  | { type: "rendered"; id: number; audio: Float32Array<ArrayBuffer>; ms: number }
  | { type: "error"; message: string };

declare const self: DedicatedWorkerGlobalScope;

let analyser: Analyser | null = null;
let sources: SourceInput[] = [];
let analyses: Analysis[] = [];
let trio: { key: string; morpher: Morpher } | null = null;

self.onmessage = (e: MessageEvent<WorkerIn>) => {
  const msg = e.data;
  try {
    if (msg.type === "init") {
      const t0 = performance.now();
      analyser = new Analyser(msg.note, msg.cfg);
      sources = msg.sources;
      analyses = [];
      for (const src of sources) {
        analyses.push(analyser.analyse(src));
        const p: WorkerOut = { type: "progress", done: analyses.length, total: sources.length };
        self.postMessage(p);
      }
      trio = null;
      const out: WorkerOut = { type: "ready", frames: analyses.map((a) => a.T), ms: performance.now() - t0 };
      self.postMessage(out);
    } else if (msg.type === "render") {
      if (!analyser) throw new Error("render before init");
      const key = msg.ids.join(",");
      if (!trio || trio.key !== key) {
        trio = {
          key,
          morpher: new Morpher(analyser, msg.ids.map((i) => sources[i]), msg.ids.map((i) => analyses[i])),
        };
      }
      const t0 = performance.now();
      const audio = Float32Array.from(trio.morpher.render(msg.weights));
      const out: WorkerOut = { type: "rendered", id: msg.id, audio, ms: performance.now() - t0 };
      self.postMessage(out, [audio.buffer]);
    }
  } catch (err) {
    const out: WorkerOut = { type: "error", message: err instanceof Error ? err.message : String(err) };
    self.postMessage(out);
  }
};
