import { useCallback, useEffect, useRef, useState } from "react";
import { Progress } from "./Progress.tsx";
import type { WorkerIn, WorkerOut } from "../lib/babymaker/morph.worker.ts";
import type { SourceInput } from "../lib/babymaker/morph.ts";
import { decodeWav } from "../lib/babymaker/wav.ts";
import { parseMidi, type MidiNote } from "../lib/midi/smf.ts";
import { cx } from "./cx.ts";
import styles from "./Babymaker.module.css";

/* -------------------------------------------------------------------- data */

interface Instrument { id: string; label: string; file: string; peakRmsIn: number }

interface Manifest {
  note: number;
  sr: number;
  default: number[];
  credit: { name: string; author: string; license: string; url: string };
  midi: string;
  instruments: Instrument[];
}

type Phase = "idle" | "loading" | "analysing" | "rendering" | "error" | "ready";

interface Pt { x: number; y: number }

/** Starting layout: the three dots on a triangle, the playhead in the middle. */
const START_DOTS: Pt[] = [{ x: 22, y: 26 }, { x: 78, y: 26 }, { x: 50, y: 72 }];
const START_HEAD: Pt = { x: 50, y: 42 };

/** prepare() step 4: peak 20 ms RMS -> 1, so the morph sees what the Python one saw. */
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

/**
 * Inverse-distance weights of the playhead against the dots. On a dot ->
 * that instrument alone; anywhere else, nearer dots pull harder. Unlike
 * barycentric coordinates it works wherever the dots are dragged, even if
 * they end up in a line.
 */
function weightsFor(head: Pt, dots: Pt[]): number[] {
  const raw = dots.map((d) => 1 / ((head.x - d.x) ** 2 + (head.y - d.y) ** 2 + 0.5));
  const sum = raw.reduce((a, b) => a + b, 0);
  return raw.map((r) => r / sum);
}

const RENDER_THROTTLE_MS = 90;
const LOOKAHEAD_S = 0.15;
const TICK_MS = 30;
const LOOP_GAP_S = 0.6;
const CLICK_PX = 4;        // a press that moves less than this is a click, not a drag

/* How the loading bar is divided. Analysis dominates the wall clock, so it
   gets most of the ring; the rest keeps the number moving from the start. */
const P_MANIFEST = 0.06;
const P_FETCHED = 0.34;    // manifest + every sample + the MIDI downloaded
const P_ANALYSED = 0.96;   // all instruments analysed

const LOAD_LABEL: Record<string, string> = {
  loading: "loading samples",
  analysing: "analysing",
  rendering: "first morph",
};

/* --------------------------------------------------------------- component */

export function Babymaker({ manifest = "media/manifest.json" }: { manifest?: string }) {
  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState<string | null>(null);
  const [instruments, setInstruments] = useState<Instrument[]>([]);
  const [credit, setCredit] = useState<Manifest["credit"] | null>(null);
  const [selection, setSelection] = useState<number[]>([0, 1, 2]);
  const [playing, setPlaying] = useState(false);
  const [renderMs, setRenderMs] = useState<number | null>(null);
  const [analyseMs, setAnalyseMs] = useState<number | null>(null);
  const [dots, setDots] = useState<Pt[]>(START_DOTS);
  const [head, setHead] = useState<Pt>(START_HEAD);
  const [menu, setMenu] = useState<number | null>(null);   // which dot's picker is open
  const [progress, setRawProgress] = useState(0);          // 0..1 while loading
  // Parallel fetches finish out of order and the analyser reports per
  // source, so a raw reading can come back lower than the last one. A ring
  // that unwinds reads as a fault, so it only ever moves forwards.
  const setProgress = useCallback((v: number) => setRawProgress((p) => Math.max(p, v)), []);

  const ctxRef = useRef<AudioContext | null>(null);
  const masterRef = useRef<GainNode | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const workerRef = useRef<Worker | null>(null);
  const bufferRef = useRef<AudioBuffer | null>(null);
  const noteRef = useRef(76);
  const midiRef = useRef<MidiNote[]>([]);
  const loopLenRef = useRef(0);
  const cursorRef = useRef({ idx: 0, base: 0 });
  const liveRef = useRef<Set<AudioBufferSourceNode>>(new Set());
  const autoStarted = useRef(false);
  const timerRef = useRef<number | null>(null);
  const rafRef = useRef<number | null>(null);
  const headElRef = useRef<SVGCircleElement>(null);
  const ringElRef = useRef<SVGCircleElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const reqRef = useRef({ next: 1, latest: 0, pending: false, queued: null as { ids: number[]; w: number[] } | null, last: 0 });
  const drag = useRef<{ what: "head" | number; x0: number; y0: number; moved: boolean } | null>(null);
  const manRef = useRef<Manifest | null>(null);

  const weights = weightsFor(head, dots);

  /* ---- the manifest is fetched on mount, so the dots carry real names and
     the picker works before any audio is loaded; audio waits for a gesture */
  useEffect(() => {
    let live = true;
    fetch(manifest)
      .then((r) => r.json())
      .then((man: Manifest) => {
        if (!live) return;
        manRef.current = man;
        noteRef.current = man.note;
        setInstruments(man.instruments);
        setCredit(man.credit);
        setSelection(man.default);
      })
      .catch(() => { /* the load button reports errors */ });
    return () => { live = false; };
  }, [manifest]);

  /* ---- ask the worker for the morph at the current trio + weights, throttled */
  const requestRender = useCallback((ids: number[], w: number[]) => {
    const r = reqRef.current;
    const worker = workerRef.current;
    if (!worker) return;
    const now = performance.now();
    if (r.pending || now - r.last < RENDER_THROTTLE_MS) {
      r.queued = { ids, w };                     // newest wins
      return;
    }
    r.pending = true;
    r.last = now;
    const id = r.next++;
    r.latest = id;
    const msg: WorkerIn = { type: "render", id, ids, weights: w };
    worker.postMessage(msg);
  }, []);

  useEffect(() => {
    if (phase === "ready" || phase === "rendering") requestRender(selection, weights);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [head, dots, selection, phase]);

  /* ---- load everything on the first press: needs a gesture for audio anyway */
  const start = useCallback(async () => {
    if (phase !== "idle") return;
    setPhase("loading");
    try {
      const ctx = new AudioContext({ sampleRate: 44100 });
      const master = ctx.createGain();
      master.gain.value = 0.8;
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 1024;
      master.connect(analyser).connect(ctx.destination);
      ctxRef.current = ctx; masterRef.current = master; analyserRef.current = analyser;

      // a pick made before loading must survive the load
      const fresh = manRef.current === null;
      const man: Manifest = manRef.current ?? (await (await fetch(manifest)).json());
      setProgress(P_MANIFEST);
      manRef.current = man;
      noteRef.current = man.note;
      setInstruments(man.instruments);
      setCredit(man.credit);
      if (fresh) setSelection(man.default);

      // the manifest names its files relative to itself
      const base = new URL(manifest, document.baseURI);
      const rel = (f: string) => new URL(f, base).href;
      const nFiles = man.instruments.length + 1;
      let fetched = 0;
      const step = () => {
        fetched += 1;
        setProgress(P_MANIFEST + ((P_FETCHED - P_MANIFEST) * fetched) / nFiles);
      };
      const [wavs, midiBuf] = await Promise.all([
        Promise.all(man.instruments.map(async (s) => {
          const w = decodeWav(await (await fetch(rel(s.file))).arrayBuffer());
          step();
          return w;
        })),
        (await fetch(rel(man.midi))).arrayBuffer().then((b) => { step(); return b; }),
      ]);
      const midi = parseMidi(midiBuf);
      midiRef.current = midi.notes;
      loopLenRef.current = midi.duration + LOOP_GAP_S;

      const sources: SourceInput[] = man.instruments.map((s, i) => ({
        name: s.label,
        audio: normalise(wavs[i].samples, wavs[i].sampleRate),
        peakRmsIn: s.peakRmsIn,
      }));

      setPhase("analysing");
      const worker = new Worker(new URL("../lib/babymaker/morph.worker.ts", import.meta.url), { type: "module" });
      workerRef.current = worker;
      worker.onmessage = (e: MessageEvent<WorkerOut>) => {
        const m = e.data;
        if (m.type === "progress") {
          setProgress(P_FETCHED + ((P_ANALYSED - P_FETCHED) * m.done) / m.total);
        } else if (m.type === "ready") {
          setAnalyseMs(m.ms);
          setProgress(P_ANALYSED);
          setPhase("rendering");
        } else if (m.type === "rendered") {
          const r = reqRef.current;
          r.pending = false;
          if (m.id === r.latest) {
            const buf = ctx.createBuffer(1, m.audio.length, ctx.sampleRate);
            buf.copyToChannel(m.audio, 0);
            bufferRef.current = buf;
            setRenderMs(m.ms);
            setProgress(1);
            setPhase("ready");
          }
          if (r.queued) { const q = r.queued; r.queued = null; requestRender(q.ids, q.w); }
        } else if (m.type === "error") {
          setError(m.message); setPhase("error");
        }
      };
      worker.onerror = (ev) => { setError(ev.message); setPhase("error"); };
      const init: WorkerIn = { type: "init", sources, note: man.note, cfg: { oversample: 2, glIters: 2 } };
      worker.postMessage(init, sources.map((s) => s.audio.buffer));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setPhase("error");
    }
  }, [manifest, phase, requestRender, setProgress]);

  /* ---- the sequencer: schedule a little ahead of the clock, loop forever */
  const scheduleNote = useCallback((n: MidiNote, base: number) => {
    const ctx = ctxRef.current, master = masterRef.current, buf = bufferRef.current;
    if (!ctx || !master || !buf) return;
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.playbackRate.value = Math.pow(2, (n.note - noteRef.current) / 12);
    const g = ctx.createGain();
    const vel = Math.pow(n.velocity / 127, 1.4);
    const t0 = base + n.start;
    const t1 = base + n.end;
    g.gain.setValueAtTime(0, t0);
    g.gain.linearRampToValueAtTime(vel, t0 + 0.006);
    g.gain.setValueAtTime(vel, Math.max(t0 + 0.006, t1));
    g.gain.linearRampToValueAtTime(0, t1 + 0.07);
    src.connect(g).connect(master);
    src.start(t0);
    src.stop(t1 + 0.08);
    // kept so stop() can cut voices that are already scheduled
    liveRef.current.add(src);
    src.onended = () => liveRef.current.delete(src);
  }, []);

  const tick = useCallback(() => {
    const ctx = ctxRef.current;
    const notes = midiRef.current;
    if (!ctx || !notes.length) return;
    const c = cursorRef.current;
    const horizon = ctx.currentTime + LOOKAHEAD_S;
    while (c.base + notes[c.idx].start < horizon) {
      scheduleNote(notes[c.idx], c.base);
      c.idx++;
      if (c.idx >= notes.length) { c.idx = 0; c.base += loopLenRef.current; }
    }
  }, [scheduleNote]);

  const stop = useCallback(() => {
    if (timerRef.current !== null) window.clearInterval(timerRef.current);
    timerRef.current = null;
    for (const src of liveRef.current) {
      try { src.stop(); } catch { /* already ended */ }
    }
    liveRef.current.clear();
    setPlaying(false);
  }, []);

  const play = useCallback(async () => {
    const ctx = ctxRef.current;
    if (!ctx) return;
    await ctx.resume();
    cursorRef.current = { idx: 0, base: ctx.currentTime + 0.05 };
    tick();
    timerRef.current = window.setInterval(tick, TICK_MS);
    setPlaying(true);
  }, [tick]);

  /* ---- it plays as soon as there is something to play. The load was
     started by a click on the stage, so the audio context is unlocked. */
  useEffect(() => {
    if (phase === "ready" && !autoStarted.current) {
      autoStarted.current = true;
      void play();
    }
  }, [phase, play]);

  /* ---- the playhead pulses with what's coming out of the speakers */
  useEffect(() => {
    const data = new Float32Array(1024);
    const loop = () => {
      const an = analyserRef.current, el = headElRef.current, ring = ringElRef.current;
      if (an && el && ring) {
        an.getFloatTimeDomainData(data);
        let e = 0;
        for (let i = 0; i < data.length; i++) e += data[i] * data[i];
        const rms = Math.sqrt(e / data.length);
        const r = 2.2 + Math.min(4, rms * 22);
        el.setAttribute("r", r.toFixed(2));
        ring.setAttribute("r", (r + 1.6 + rms * 30).toFixed(2));
      }
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);
    return () => { if (rafRef.current !== null) cancelAnimationFrame(rafRef.current); };
  }, []);

  useEffect(() => () => {
    if (timerRef.current !== null) window.clearInterval(timerRef.current);
    for (const src of liveRef.current) { try { src.stop(); } catch { /* already ended */ } }
    workerRef.current?.terminate();
    ctxRef.current?.close();
  }, []);

  /* ---- the picker closes on a press outside it, or Escape */
  useEffect(() => {
    if (menu === null) return;
    const onDown = (e: PointerEvent) => {
      if (!menuRef.current?.contains(e.target as Node)) setMenu(null);
    };
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setMenu(null); };
    document.addEventListener("pointerdown", onDown, true);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("pointerdown", onDown, true);
      document.removeEventListener("keydown", onKey);
    };
  }, [menu]);

  /* ---- dragging, in the SVG's own 0..100 coordinates. The dot being
     dragged is read off a data attribute, so no handler is built during
     render. A press that doesn't move is a click, which opens the picker. */
  const onDown = useCallback((e: React.PointerEvent<SVGCircleElement>) => {
    const tag = e.currentTarget.dataset.dot ?? "head";
    drag.current = { what: tag === "head" ? "head" : Number(tag), x0: e.clientX, y0: e.clientY, moved: false };
    e.currentTarget.setPointerCapture(e.pointerId);
  }, []);
  const onMove = useCallback((e: React.PointerEvent<SVGSVGElement>) => {
    const d = drag.current, svg = svgRef.current;
    if (!d || !svg) return;
    if (!d.moved && Math.hypot(e.clientX - d.x0, e.clientY - d.y0) < CLICK_PX) return;
    d.moved = true;
    const r = svg.getBoundingClientRect();
    const clamp = (v: number) => Math.min(100, Math.max(0, v));
    const p = { x: clamp(((e.clientX - r.left) / r.width) * 100), y: clamp(((e.clientY - r.top) / r.height) * 100) };
    if (d.what === "head") setHead(p);
    else { const i = d.what; setDots((ds) => ds.map((q, k) => (k === i ? p : q))); }
  }, []);
  const onUp = useCallback(() => {
    const d = drag.current;
    drag.current = null;
    if (d && !d.moved && d.what !== "head") setMenu(d.what);
  }, []);

  const pick = useCallback((dot: number, instrument: number) => {
    setSelection((sel) => sel.map((v, k) => (k === dot ? instrument : v)));
    setMenu(null);
  }, []);

  const ready = phase === "ready";
  const nameOf = (dot: number) => instruments[selection[dot]]?.label ?? `instrument ${dot + 1}`;

  return (
    <figure className={styles.wrap}>
      <div className={styles.stageWrap}>
        <svg ref={svgRef} className={cx(styles.stage, phase !== "ready" && styles.dim)} viewBox="0 0 100 75" onPointerMove={onMove} onPointerUp={onUp} onPointerCancel={onUp}>
          {dots.map((d, i) => (
            // drawn from the dot to the playhead, so the dashes flow the way
            // the sound does; the heavier the weight, the faster
            <line key={`l${i}`} className={styles.line} x1={d.x} y1={d.y * 0.75} x2={head.x} y2={head.y * 0.75}
              style={{
                opacity: 0.08 + 0.92 * weights[i],
                animationDuration: `${(1.9 - 1.3 * weights[i]).toFixed(2)}s`,
                animationPlayState: playing ? "running" : "paused",
              }} />
          ))}
          {dots.map((d, i) => {
            // label away from the playhead, so it never lands on the weight line
            const dx = d.x - head.x, dy = (d.y - head.y) * 0.75;
            const len = Math.hypot(dx, dy) || 1;
            const lx = d.x + (dx / len) * 7.5, ly = d.y * 0.75 + (dy / len) * 7.5;
            return (
              <g key={`d${i}`}>
                <circle className={cx(styles.sample, menu === i && styles.sampleOpen)} cx={d.x} cy={d.y * 0.75} r={2.6}
                  data-dot={i} onPointerDown={onDown} />
                <text className={styles.label} x={lx} y={ly - 0.6} textAnchor="middle">{nameOf(i)}</text>
                <text className={styles.pct} x={lx} y={ly + 2.4} textAnchor="middle">{Math.round(weights[i] * 100)}%</text>
              </g>
            );
          })}
          <circle ref={ringElRef} className={styles.headRing} cx={head.x} cy={head.y * 0.75} r={3.8} />
          <circle ref={headElRef} className={styles.head} cx={head.x} cy={head.y * 0.75} r={2.2} data-dot="head" onPointerDown={onDown} />
        </svg>

        {!ready && <div className={styles.preFade} aria-hidden="true" />}

        {phase === "idle" ? (
          <button type="button" className={styles.startBtn} onClick={start}>
            <span className={styles.startHint}>click to load the instruments</span>
          </button>
        ) : null}

        {phase === "loading" || phase === "analysing" || phase === "rendering" ? (
          <div className={styles.overlay}>
            <Progress value={progress} label={LOAD_LABEL[phase]} immediate />
          </div>
        ) : null}

        {phase === "ready" ? (
          <button type="button" className={styles.transport} aria-label={playing ? "stop" : "play"}
            onClick={playing ? stop : () => void play()}>
            <svg viewBox="0 0 16 16" aria-hidden="true">
              {playing ? <rect x="3.5" y="3.5" width="9" height="9" rx="0.8" /> : <path d="M4.5 2.8v10.4l8.6-5.2z" />}
            </svg>
          </button>
        ) : null}

        {menu !== null && instruments.length > 0 ? (
          <div ref={menuRef} className={styles.menu} role="menu" aria-label={`instrument for dot ${menu + 1}`}
            style={{ left: `${dots[menu].x}%`, top: `${dots[menu].y}%` }}>
            <span className={cx(styles.label2, styles.menuTitle)}>this dot plays</span>
            {instruments.map((inst, k) => {
              const current = selection[menu] === k;
              const taken = !current && selection.includes(k);
              return (
                <button key={inst.id} type="button" role="menuitemradio" aria-checked={current} disabled={taken}
                  className={cx(styles.menuItem, current && styles.menuCurrent)} onClick={() => pick(menu, k)}>
                  {inst.label}
                  {taken ? <span className={styles.menuNote}>on another dot</span> : null}
                </button>
              );
            })}
          </div>
        ) : null}
      </div>

      <figcaption className={styles.bar}>
        <span className={styles.status}>
          {ready && (
            <>
              click a dot to change its instrument · <span className={styles.label2}>analysis</span>{" "}
              {analyseMs?.toFixed(0)} ms · <span className={styles.label2}>morph</span>{" "}
              {renderMs === null ? "…" : `${renderMs.toFixed(0)} ms`}
            </>
          )}
          {phase === "error" && <span className={styles.err}>{error}</span>}
        </span>
        {credit ? (
          <span className={styles.credit}>
            samples: <a href={credit.url} target="_blank" rel="noopener noreferrer">{credit.name}</a> by {credit.author}, {credit.license}
          </span>
        ) : null}
      </figcaption>
    </figure>
  );
}
