// Voiceover recorder: record takes over the video, see them on the timeline, hear them back
// over the picture, export an mp4. Takes are saved on the server the moment they stop.

const $ = (id) => document.getElementById(id);
const video = $("video");
const state = {
  cuts: [], cut: null, lines: [], takes: [], buffers: new Map(),
  sel: 0, recording: null, sources: [],
};

let ctx = null, node = null, stream = null, voiceGain = null, micId = "";
const fmt = (t) => `${Math.floor(t / 60)}:${(t % 60).toFixed(1).padStart(4, "0")}`;
const offset = () => Number($("offset").value) / 1000;
const preroll = () => Number($("preroll").value);
const status = (s) => { $("status").textContent = s; };

// ------------------------------------------------------------------ audio

function audio() {
  if (!ctx) {
    ctx = new AudioContext({ latencyHint: "interactive" });
    voiceGain = ctx.createGain();
    voiceGain.gain.value = Number($("ovol").value);
    voiceGain.connect(ctx.destination);
  }
  return ctx;
}

async function ensureMic() {
  audio();
  await ctx.resume();
  if (!node) {
    await ctx.audioWorklet.addModule("/capture.js");
    node = new AudioWorkletNode(ctx, "capture", { numberOfInputs: 1, numberOfOutputs: 1 });
    const mute = ctx.createGain();
    mute.gain.value = 0;
    node.connect(mute).connect(ctx.destination);     // keeps the worklet running, silently
    node.port.onmessage = onCapture;
  }
  if (!stream || stream._id !== micId) {
    if (stream) stream.getTracks().forEach((t) => t.stop());
    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        deviceId: micId ? { exact: micId } : undefined,
        echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1,
      },
    });
    stream._id = micId;
    ctx.createMediaStreamSource(stream).connect(node);
    listMics();
  }
}

async function listMics() {
  const sel = $("mic");
  const devs = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "audioinput");
  const cur = sel.value;
  sel.innerHTML = '<option value="">default</option>' +
    devs.map((d) => `<option value="${d.deviceId}">${d.label || "microphone"}</option>`).join("");
  sel.value = cur;
}

let rec = null;   // the capture in progress
function onCapture(e) {
  const m = e.data;
  if (m.type === "level") {
    $("meter").style.width = `${Math.min(100, Math.sqrt(m.peak) * 100)}%`;
    if (m.peak > 0.98) { $("clip").hidden = false; setTimeout(() => ($("clip").hidden = true), 1500); }
  } else if (rec && m.type === "first") {
    rec.firstFrame = m.frame;
  } else if (rec && m.type === "data") {
    rec.chunks.push(m.buf);
  } else if (rec && m.type === "stopped") {
    rec.resolveStop();
  }
}

// ------------------------------------------------------------------ takes: which parts are heard

/** The parts of each take that are heard: a newer take wins wherever two overlap. */
function pieces() {
  const out = [], covered = [];
  const takes = [...state.takes].sort((a, b) => b.created - a.created);
  for (const tk of takes) {
    const t0 = tk.t0 + offset();
    let parts = [[Math.max(tk.keepA, t0), Math.min(tk.keepB, t0 + tk.dur)]];
    for (const [ca, cb] of covered) {
      parts = parts.flatMap(([a, b]) => {
        if (cb <= a || ca >= b) return [[a, b]];
        const r = [];
        if (ca > a) r.push([a, ca]);
        if (cb < b) r.push([cb, b]);
        return r;
      });
    }
    for (const [a, b] of parts) if (b - a > 0.02) out.push({ tk, a, b });
    covered.push([Math.max(tk.keepA, t0), Math.min(tk.keepB, t0 + tk.dur)]);
  }
  return out;
}

function stopVoice() {
  for (const s of state.sources) { try { s.stop(); } catch { /* already done */ } }
  state.sources = [];
}

/** Hear the takes over the picture: schedule each heard piece against the video's clock. */
function scheduleVoice() {
  stopVoice();
  if (!ctx || video.paused || state.recording) return;
  const v = video.currentTime, now = ctx.currentTime + 0.03;
  for (const p of pieces()) {
    if (p.b <= v) continue;
    const buf = state.buffers.get(p.tk.id);
    if (!buf) continue;
    const start = Math.max(p.a, v);
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(voiceGain);
    src.start(now + (start - v), start - (p.tk.t0 + offset()), p.b - start);
    state.sources.push(src);
  }
}

async function loadTakeAudio(tk) {
  if (state.buffers.has(tk.id)) return;
  const r = await fetch(`/take/${state.cut.id}/${tk.file}`);
  state.buffers.set(tk.id, await audio().decodeAudioData(await r.arrayBuffer()));
}

// ------------------------------------------------------------------ recording

/** 16-bit PCM WAV: every browser decodes it when the takes are loaded again. */
function wav(samples, sr) {
  const b = new ArrayBuffer(44 + samples.length * 2);
  const d = new DataView(b);
  const s = (o, t) => [...t].forEach((c, i) => d.setUint8(o + i, c.charCodeAt(0)));
  s(0, "RIFF"); d.setUint32(4, 36 + samples.length * 2, true); s(8, "WAVE");
  s(12, "fmt "); d.setUint32(16, 16, true); d.setUint16(20, 1, true); d.setUint16(22, 1, true);
  d.setUint32(24, sr, true); d.setUint32(28, sr * 2, true); d.setUint16(32, 2, true); d.setUint16(34, 16, true);
  s(36, "data"); d.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const v = Math.max(-1, Math.min(1, samples[i]));
    d.setInt16(44 + i * 2, v < 0 ? v * 32768 : v * 32767, true);
  }
  return b;
}

const seek = (t) => new Promise((res) => {
  if (Math.abs(video.currentTime - t) < 0.01) return res();
  video.addEventListener("seeked", res, { once: true });
  video.currentTime = t;
});

async function record(mode) {
  if (state.recording) return;
  try { await ensureMic(); } catch (e) { status(`no microphone: ${e.message}`); return; }
  stopVoice();
  video.pause();
  const line = state.lines[state.sel];
  let from = video.currentTime, keepA = from, stopAt = Infinity;
  if (mode === "line" && line) {
    from = Math.max(0, line.t - preroll());
    keepA = Math.max(0, line.t - 0.35);
    stopAt = line.t + line.dur + 1.2;
  }
  await seek(from);
  rec = { chunks: [], firstFrame: null, map: null, mode, line: mode === "line" ? line?.id : null, keepA, stopAt };
  rec.stopped = new Promise((r) => (rec.resolveStop = r));
  state.recording = rec;
  node.port.postMessage("start");
  video.muted = !$("hear").checked;
  // tie the video's clock to the audio clock at the first frame shown
  const mark = (meta) => {
    const perfNow = performance.now(), ctxNow = ctx.currentTime;
    const disp = meta ? meta.expectedDisplayTime : perfNow;
    rec.map = { media: meta ? meta.mediaTime : video.currentTime, ctx: ctxNow + (disp - perfNow) / 1000 };
  };
  if (video.requestVideoFrameCallback) video.requestVideoFrameCallback((_, meta) => mark(meta));
  else video.addEventListener("playing", () => mark(null), { once: true });
  await video.play();
  ui();
}

async function stop() {
  const r = state.recording;
  if (!r) { video.pause(); return; }
  if (r.stopping) return;          // the frame loop and a key press can both ask
  r.stopping = true;
  const keepB = video.currentTime;
  node.port.postMessage("stop");
  await r.stopped;
  video.pause();
  video.muted = false;
  state.recording = null;
  ui();
  const n = r.chunks.reduce((a, c) => a + c.length, 0);
  if (!r.map || r.firstFrame === null || n < ctx.sampleRate * 0.3) { status("take too short, not kept"); return; }
  const samples = new Float32Array(n);
  let o = 0;
  for (const c of r.chunks) { samples.set(c, o); o += c.length; }
  // the video time of the take's first sample
  const t0 = r.map.media - (r.map.ctx - r.firstFrame / ctx.sampleRate);
  const q = new URLSearchParams({ cut: state.cut.id, t0: t0.toFixed(4), keepA: r.keepA.toFixed(3),
                                  keepB: keepB.toFixed(3), line: r.line || "" });
  status("saving take…");
  const res = await fetch(`/api/takes?${q}`, { method: "POST", body: wav(samples, ctx.sampleRate),
                                              headers: { "Content-Type": "audio/wav" } });
  const tk = await res.json();
  const buf = ctx.createBuffer(1, samples.length, ctx.sampleRate);
  buf.copyToChannel(samples, 0);
  state.buffers.set(tk.id, buf);
  state.takes.push(tk);
  status(`take saved (${tk.dur.toFixed(1)} s)`);
  renderTakes();
  renderLines();
}

// ------------------------------------------------------------------ the page

function ui() {
  const r = !!state.recording;
  $("rec").hidden = !r;
  $("stop").disabled = !r && video.paused;
  $("recHere").disabled = r;
  $("recLine").disabled = r;
  $("play").textContent = video.paused ? "play" : "pause";
  $("play").disabled = r;
  document.querySelectorAll("#cuts button").forEach((b) => (b.disabled = r));
}

function lineCoverage(l) {
  const ps = pieces();
  let got = 0;
  for (const p of ps) got += Math.max(0, Math.min(p.b, l.t + l.dur) - Math.max(p.a, l.t));
  return got / l.dur;
}

function renderLines() {
  const ol = $("lines");
  ol.innerHTML = "";
  state.lines.forEach((l, i) => {
    const li = document.createElement("li");
    const cov = lineCoverage(l);
    li.innerHTML = `<span class="t mono">${fmt(l.t)}</span>
      <span>${l.cap}<span class="seg">${l.segment}</span></span>
      <span class="ok">${cov > 0.8 ? "✓ recorded" : cov > 0.1 ? "part" : ""}</span>`;
    li.onclick = () => select(i, true);
    li.ondblclick = () => { select(i, false); record("line"); };
    ol.appendChild(li);
  });
  highlight();
}

function select(i, jump) {
  state.sel = Math.max(0, Math.min(state.lines.length - 1, i));
  if (jump && !state.recording) {
    video.currentTime = Math.max(0, state.lines[state.sel].t - Math.min(1.0, preroll()));
  }
  highlight();
  const li = $("lines").children[state.sel];
  if (li) li.scrollIntoView({ block: "nearest" });
}

function highlight() {
  const t = video.currentTime;
  const cur = state.lines.findLastIndex((l) => l.t <= t + 0.05);
  [...$("lines").children].forEach((li, i) => {
    li.classList.toggle("sel", i === state.sel);
    li.classList.toggle("cur", i === cur);
  });
}

function renderTakes() {
  const ol = $("takes");
  ol.innerHTML = "";
  const heard = new Map();
  for (const p of pieces()) heard.set(p.tk.id, (heard.get(p.tk.id) || 0) + (p.b - p.a));
  $("takeCount").textContent = state.takes.length ? `(${state.takes.length})` : "";
  for (const tk of [...state.takes].sort((a, b) => a.keepA - b.keepA)) {
    const li = document.createElement("li");
    const h = heard.get(tk.id) || 0;
    const when = new Date(tk.created * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    li.innerHTML = `<span class="mono">${fmt(tk.keepA)}–${fmt(tk.keepB)}</span>
      <span class="what">${tk.line ? tk.line : "pass"} <span class="faint">· ${when}${h < 0.05 ? " · replaced by a newer take" : ""}</span></span>`;
    const play = document.createElement("button");
    play.className = "small";
    play.textContent = "hear";
    play.onclick = async () => { video.currentTime = Math.max(0, tk.keepA - 0.5); await audio().resume(); video.play(); };
    const del = document.createElement("button");
    del.className = "small";
    del.textContent = "delete";
    del.onclick = async () => {
      if (!confirm("Delete this take? (it moves to voiceover/…/deleted)")) return;
      await fetch(`/api/takes?cut=${state.cut.id}&id=${tk.id}`, { method: "DELETE" });
      state.takes = state.takes.filter((x) => x.id !== tk.id);
      renderTakes(); renderLines(); scheduleVoice();
    };
    li.append(play, del);
    ol.appendChild(li);
  }
  drawTimeline();
}

function drawTimeline() {
  const c = $("timeline");
  const dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth, h = 64;
  if (c.width !== w * dpr) { c.width = w * dpr; c.height = h * dpr; }
  const g = c.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  const css = getComputedStyle(document.documentElement);
  const col = (n) => css.getPropertyValue(n).trim();
  const D = state.cut ? state.cut.duration : 1;
  const X = (t) => (t / D) * w;
  g.clearRect(0, 0, w, h);
  g.fillStyle = col("--wash");
  g.fillRect(0, 0, w, h);
  // the script's lines, as faint bars at their estimated length
  g.fillStyle = col("--rule");
  for (const l of state.lines) g.fillRect(X(l.t), 8, Math.max(2, X(l.t + l.dur) - X(l.t)), 14);
  // takes: what is heard, and (faint) what a newer take replaced
  g.fillStyle = col("--faint");
  for (const tk of state.takes) g.fillRect(X(tk.keepA), 30, Math.max(1, X(tk.keepB) - X(tk.keepA)), 3);
  g.fillStyle = col("--accent");
  for (const p of pieces()) g.fillRect(X(p.a), 27, Math.max(2, X(p.b) - X(p.a)), 16);
  const t = video.currentTime;
  g.fillStyle = state.recording ? col("--rec") : col("--ink");
  g.fillRect(X(t) - 1, 0, 2, h);
  g.fillStyle = col("--faint");
  g.font = `11px ${col("--mono")}`;
  g.fillText(fmt(D), w - 44, h - 6);
}

function prompter() {
  const t = video.currentTime;
  const ls = state.lines;
  let i = ls.findLastIndex((l) => l.t <= t);
  let live = i >= 0 && t <= ls[i].t + ls[i].dur + 0.3;
  let show = live ? i : i + 1;
  const l = ls[show];
  $("pNow").textContent = l ? l.cap : "";
  $("pAfter").textContent = ls[show + 1] ? ls[show + 1].cap : "";
  const head = $("pState").parentElement;
  if (!l) {
    $("pState").textContent = "end";
    $("pCount").textContent = "";
    $("pBar").style.width = "0";
  } else if (live) {
    $("pState").textContent = "now";
    head.classList.add("live");
    $("pNow").classList.remove("soon");
    $("pCount").textContent = `${Math.max(0, l.t + l.dur - t).toFixed(1)} s`;
    $("pBar").style.width = `${Math.min(100, ((t - l.t) / l.dur) * 100)}%`;
  } else {
    head.classList.remove("live");
    $("pNow").classList.add("soon");
    const wait = l.t - t;
    $("pState").textContent = "up next";
    $("pCount").textContent = `in ${wait.toFixed(1)} s`;
    $("pBar").style.width = `${Math.max(0, 100 - (wait / 3) * 100)}%`;
  }
}

function tick() {
  const t = video.currentTime;
  $("clock").textContent = fmt(t);
  const r = state.recording;
  if (r) {
    $("recTime").textContent = `REC ${fmt(t)}`;
    if (t >= r.stopAt || video.ended) stop();
  }
  prompter();
  highlight();
  drawTimeline();
  requestAnimationFrame(tick);
}

async function loadCut(id) {
  stopVoice();
  const cut = state.cuts.find((c) => c.id === id);
  state.cut = cut;
  state.lines = cut.lines;
  state.sel = 0;
  video.src = cut.video;
  video.volume = Number($("vvol").value);
  document.querySelectorAll("#cuts button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.id === id)));
  state.takes = await (await fetch(`/api/takes?cut=${id}`)).json();
  renderLines();
  renderTakes();
  status(`${cut.title}: ${cut.lines.length} lines, ${state.takes.length} takes`);
  await Promise.all(state.takes.map(loadTakeAudio));
  $("exportOut").innerHTML = cut.exported ? `<a href="/out/${cut.video.split("/").pop().replace(".mp4", "-vo.mp4")}" target="_blank">last export</a>` : "";
  try { localStorage.setItem("vo-cut", id); } catch { /* private window */ }
}

async function main() {
  state.cuts = await (await fetch("/api/cuts")).json();
  const nav = $("cuts");
  for (const c of state.cuts) {
    const b = document.createElement("button");
    b.textContent = c.id;
    b.dataset.id = c.id;
    b.onclick = () => loadCut(c.id);
    nav.appendChild(b);
  }
  let first = state.cuts[0].id;
  try { first = localStorage.getItem("vo-cut") || first; } catch { /* private window */ }
  await loadCut(state.cuts.some((c) => c.id === first) ? first : state.cuts[0].id);

  $("play").onclick = async () => { await audio().resume(); video.paused ? video.play() : video.pause(); };
  $("recHere").onclick = () => record("here");
  $("recLine").onclick = () => record("line");
  $("stop").onclick = () => stop();
  $("timeline").onclick = (e) => {
    if (state.recording) return;
    const r = e.currentTarget.getBoundingClientRect();
    video.currentTime = ((e.clientX - r.left) / r.width) * state.cut.duration;
  };
  $("mic").onchange = async (e) => { micId = e.target.value; if (stream) await ensureMic(); };
  $("preroll").oninput = (e) => ($("prerollV").textContent = `${Number(e.target.value).toFixed(1)} s`);
  $("offset").oninput = (e) => { $("offsetV").textContent = `${e.target.value} ms`; scheduleVoice(); drawTimeline(); };
  $("vvol").oninput = (e) => (video.volume = Number(e.target.value));
  $("ovol").oninput = (e) => { if (voiceGain) voiceGain.gain.value = Number(e.target.value); };
  $("export").onclick = async () => {
    $("export").disabled = true;
    $("exportOut").textContent = "mixing and encoding…";
    const r = await (await fetch(`/api/export?cut=${state.cut.id}&offset_ms=${$("offset").value}`, { method: "POST" })).json();
    $("export").disabled = false;
    $("exportOut").innerHTML = r.error ? `<span style="color:var(--rec)">${r.error}</span>` :
      `<a href="${r.mp4}?v=${Date.now()}" target="_blank">${r.mp4.split("/").pop()}</a> · <a href="${r.voice}" download>voice only .wav</a> · <span class="mono faint">${r.path}</span>`;
  };

  video.addEventListener("playing", () => { scheduleVoice(); ui(); });
  video.addEventListener("pause", () => { stopVoice(); ui(); });
  video.addEventListener("seeking", stopVoice);
  video.addEventListener("seeked", () => { if (!video.paused) scheduleVoice(); });
  video.addEventListener("ended", () => { if (state.recording) stop(); ui(); });
  window.addEventListener("resize", drawTimeline);
  window.addEventListener("keydown", (e) => {
    if (e.target.matches("input, select, textarea")) return;
    if (e.key === " ") { e.preventDefault(); state.recording ? stop() : $("play").click(); }
    else if (e.key === "Escape") stop();
    else if (e.key === "r" || e.key === "R") record("here");
    else if (e.key === "l" || e.key === "L") record("line");
    else if (e.key === "ArrowDown") { e.preventDefault(); select(state.sel + 1, true); }
    else if (e.key === "ArrowUp") { e.preventDefault(); select(state.sel - 1, true); }
  });
  ui();
  requestAnimationFrame(tick);
}

main().catch((e) => status(`could not load: ${e.message}`));
