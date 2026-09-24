#!/usr/bin/env python3
"""The numbers behind the video's "why the DCT" and "how fast" chapters.

Three 4-second notes (a flute, a violin and a harp at E4, rendered from FluidR3 through the
bundled TinySoundFont renderer) are morphed on the CPU (numpy) and on the GPU (torch),
and by the TypeScript port under node. It also measures the DCT facts the video quotes:
how exactly the MDCT round-trips, how long one DCT takes, and how much of a frame's
log spectrum the first few cepstral coefficients carry.

    CUDA_VISIBLE_DEVICES=1 python3 video/bench.py      -> video/work/bench.json

The neural numbers the video compares against are quoted from their papers, not measured
here (see NEURAL below).
"""
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from babymaker import MorphConfig, Morpher, load_source  # noqa: E402
from babymaker.lapped import MDCT  # noqa: E402
from babymaker.sources import render_sf2_notes  # noqa: E402

SR = 44100
KEY = 64
FONT = os.path.expanduser(os.environ.get("BENCH_SF2", "~/babymaker-video/fonts/FluidR3_GM.sf2"))
PROGRAMS = {"flute": 73, "violin": 40, "harp": 46}
W = [0.45, 0.35, 0.20]

# Published figures, quoted with their sources (the video says where each one comes from).
NEURAL = [
    {"name": "NSynth WaveNet autoencoder", "year": 2017, "seconds": 1077.53, "note_s": 4.0,
     "hw": "TitanX GPU", "train": "trained on the NSynth dataset",
     "src": "Engel et al. 2019, GANSynth (ICLR), section 6: 1077.53 s for one 4 s sample"},
    {"name": "GANSynth", "year": 2019, "seconds": 0.020, "note_s": 4.0, "hw": "TitanX GPU",
     "train": "trained on NSynth; no encoder, so it cannot take your own samples",
     "src": "Engel et al. 2019, GANSynth (ICLR), section 6: 20 ms for one 4 s sample"},
    {"name": "RAVE", "year": 2021, "realtime_x": 20, "hw": "laptop CPU",
     "train": "needs at least 3 hours of recordings and a training run per sound",
     "src": "Caillon & Esling 2021 (abstract): 20x real time on a laptop CPU; RAVE README: >= 3 h of audio"},
]


def timed(fn, n):
    fn()                                   # warm-up (numba compile, cuda init)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    raw = {name: render_sf2_notes(FONT, 0, prog, [KEY], 100, 3.6, 0.6, SR)[KEY] for name, prog in PROGRAMS.items()}
    srcs = [load_source(f"{FONT}:0:{PROGRAMS[n]}", SR, KEY, 100, 3.6, 0.6, raw=raw[n]) for n in PROGRAMS]
    for s in srcs:                          # every source exactly 4.0 s, so the note lengths compare
        n = int(4.0 * SR)
        s.audio = np.pad(s.audio, (0, max(0, n - len(s.audio))))[:n]
    out = {"note_s": 4.0, "sr": SR, "key": KEY, "neural": NEURAL}

    for dev in ("cpu", "cuda"):
        try:
            cfg = MorphConfig(sr=SR, device=dev)
            a = timed(lambda: Morpher(srcs, cfg), 5)
            mp = Morpher(srcs, cfg)
            r = timed(lambda: mp.render(W), 20)
            y = mp.render(W)
            out[dev] = {"analysis_s": a, "render_s": r, "out_s": len(y) / SR,
                        "realtime_x": (len(y) / SR) / r}
            print(f"{dev:5s} analysis {a * 1000:7.1f} ms   render {r * 1000:7.1f} ms   "
                  f"{(len(y) / SR) / r:6.0f}x real time")
        except Exception as e:  # no GPU here
            print(f"{dev}: {e}")

    # the TypeScript port, as the browser runs it (env_mode=log, oversample 2, 2 Griffin-Lim passes)
    with tempfile.TemporaryDirectory() as td:
        for s in srcs:
            sf.write(os.path.join(td, f"{s.name.split(':')[-1]}.wav"), s.audio / np.abs(s.audio).max() * 0.9, SR)
        js = f"""
import {{ readFileSync, readdirSync }} from "node:fs";
import {{ decodeWav }} from "{ROOT}/web/lib/babymaker/wav.ts";
import {{ Morpher }} from "{ROOT}/web/lib/babymaker/morph.ts";
const files = readdirSync("{td}").filter(f => f.endsWith(".wav")).sort();
const srcs = files.map(f => {{ const b = readFileSync("{td}/" + f);
  const w = decodeWav(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
  return {{ name: f, audio: w.samples, peakRmsIn: 1 }}; }});
let t0 = performance.now();
const mp = Morpher.from(srcs, {KEY}, {{ oversample: 2, glIters: 2 }});
const analyse = performance.now() - t0;
mp.render([{W[0]}, {W[1]}, {W[2]}]);
const ts = [];
for (let i = 0; i < 10; i++) {{ t0 = performance.now(); mp.render([{W[0]}, {W[1]}, {W[2]}]); ts.push(performance.now() - t0); }}
ts.sort((a, b) => a - b);
console.log(JSON.stringify({{ analysis_s: analyse / 1000, render_s: ts[5] / 1000 }}));
"""
        p = os.path.join(ROOT, "video", "work", "_bench.mjs")
        open(p, "w").write(js)
        res = subprocess.run(["node", p], capture_output=True, text=True, check=True)
        out["js"] = json.loads(res.stdout.strip().splitlines()[-1])
        out["js"]["realtime_x"] = 4.0 / out["js"]["render_s"]
        os.remove(p)
        print(f"js    analysis {out['js']['analysis_s'] * 1000:7.1f} ms   render {out['js']['render_s'] * 1000:7.1f} ms")

    # ---- the DCT facts, on the flute the explainer chapters show (the demo's prepared sample),
    # under the explainer's config
    fl, fsr = sf.read(os.path.join(ROOT, "web", "public", "media", "flute.wav"), dtype="float64")
    m = MDCT(1024, 512)
    X, _ = m.analyze(fl)
    rec = m.synthesize(X, len(fl))
    out["mdct_roundtrip_max_err"] = float(np.abs(rec - fl).max())
    out["mdct_signal_peak"] = float(np.abs(fl).max())
    from scipy.fft import dct, idct
    v = np.random.default_rng(0).standard_normal(1024)
    out["dct1024_us"] = timed(lambda: dct(v, type=2, norm="ortho"), 2000) * 1e6
    from babymaker import Source
    man = json.load(open(os.path.join(ROOT, "web", "public", "media", "manifest.json")))
    fsrc = Source(name="flute", audio=fl, sr=fsr, note=float(man["note"]), meta={"peak_rms_in": 1.0})
    mp = Morpher([fsrc, fsrc], MorphConfig(sr=fsr, oversample=2, gl_iters=2, env_mode="log", device="cpu"))
    A = mp.analyses[0]
    ti = int(round(0.30 / (mp.mdct.hop / fsr)))                # the explainer's close-up frame
    ls = np.asarray(A.env, np.float64)[ti] + np.asarray(A.fine, np.float64)[ti]
    c = dct(ls - ls.mean(), type=2, norm="ortho")
    e = np.cumsum(c ** 2) / np.sum(c ** 2)
    q = int(mp.q)
    keep = np.zeros_like(c)
    keep[:q] = c[:q]
    smooth = idct(keep, type=2, norm="ortho") + ls.mean()
    hz = m.bin_hz(fsr)
    kmax = int(np.searchsorted(hz, 5200))
    out["cepstrum"] = {"q": q, "bins": int(len(ls)), "energy_first_q": float(e[q - 1]),
                       "abs": np.round(np.abs(c[:120]), 4).tolist(),
                       "hz": np.round(hz[:kmax], 1).tolist(),
                       "logspec": np.round(ls[:kmax], 3).tolist(), "smooth": np.round(smooth[:kmax], 3).tolist()}
    out["frames_per_note"] = int(round(4.0 * SR / 256))       # hop M/4 at 44.1 kHz
    out["flute_wave"] = fl[::40].round(4).tolist()
    out["flute_rec"] = rec[::40].round(4).tolist()
    json.dump(out, open(os.path.join(HERE, "work", "bench.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "neural"}, indent=1))


if __name__ == "__main__":
    main()
