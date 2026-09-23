#!/usr/bin/env python3
"""Dump the babymaker's intermediates for the explainer video.

Everything the animation draws is a real number out of the real algorithm:
the MCLT spectra, the true envelope, the fine structure, the DTW cost
matrix and its path, the barycentre of the weighted average, and the
morphed audio.  It runs the Python babymaker on the same three
prepared instruments the browser demo ships with, under the same config.

    python3 video/explainer_data.py            -> video/work/explainer.json (+ .wav)
"""
import json, os, sys

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from babymaker.lapped import cepstrum, lifter, true_envelope  # noqa: E402
from babymaker.morph import MorphConfig, Morpher, dtw_path, path_to_map  # noqa: E402
from babymaker.sources import Source  # noqa: E402

SRC = os.path.join(ROOT, "web", "public", "media")        # the demo's prepared instruments
DST = os.path.join(ROOT, "video", "work")                 # the video's inputs
OUT = os.path.join(DST, "explainer.json")

PICK = ["flute", "violin", "harp"]      # the demo's default trio, in its order
WEIGHTS = [0.45, 0.35, 0.20]            # the point in the triangle the video morphs to
FRAME_T = 0.30                          # seconds into the note that the frame close-ups use


def r(a, n=4):
    """Round for JSON: the video only needs plotting precision."""
    return np.round(np.asarray(a, np.float64), n).tolist()


def decimate(x, n):
    """Decimate to ~n buckets for a waveform plot, as interleaved min/max so
    the drawn polyline still fills the signal's outline rather than sampling
    a random point inside it."""
    k = max(1, len(x) // n)
    m = (len(x) // k) * k
    b = np.asarray(x[:m], np.float64).reshape(-1, k)
    return np.stack([b.min(1), b.max(1)], axis=1).ravel()


def main():
    os.makedirs(DST, exist_ok=True)
    man = json.load(open(os.path.join(SRC, "manifest.json")))
    by_id = {i["id"]: i for i in man["instruments"]}
    sr = man["sr"]

    sources = []
    for iid in PICK:
        x, s = sf.read(os.path.join(SRC, os.path.basename(by_id[iid]["file"])), dtype="float64")
        assert s == sr
        sources.append(Source(name=iid, audio=x, sr=sr, note=float(man["note"]),
                              meta={"peak_rms_in": by_id[iid]["peakRmsIn"]}))

    cfg = MorphConfig(sr=sr, oversample=2, gl_iters=2, env_mode="log", device="cpu")
    mp = Morpher(sources, cfg)
    m, A = mp.mdct, mp.analyses
    hop_s = m.hop / sr
    hz = m.bin_hz(sr)
    w = np.array(WEIGHTS, np.float64)

    ti = int(round(FRAME_T / hop_s))
    ITER_SHOW = [0, 1, 2, 4, 8, 16]         # true-envelope iterations worth drawing
    # only the first ~5 kHz is worth drawing; above it the plot is a hairline
    kmax = int(np.searchsorted(hz, 5200))

    data = {
        "sr": sr, "note": man["note"], "names": PICK, "weights": r(w, 3),
        "M": int(m.M), "hop": int(m.hop), "hopS": round(hop_s, 6),
        "q": int(mp.q), "f0": round(float(mp.f0), 2),
        "frameT": round(ti * hop_s, 4),
        "hz": r(hz[:kmax], 1),
        "sources": [],
    }

    # -- per source: waveform, level, the frame's log spectrum split three ways
    for a, s in zip(A, sources):
        env, fine = np.asarray(a.env, np.float64), np.asarray(a.fine, np.float64)
        data["sources"].append({
            "name": a.name,
            "wave": r(decimate(s.audio, 900), 4),
            "durS": round(len(s.audio) / sr, 4),
            "levelDb": r(a.level_db, 2),
            "logShape": r((env + fine)[ti, :kmax], 3),
            "env": r(env[ti, :kmax], 3),
            "fine": r(fine[ti, :kmax], 3),
            "envAll": r(env[:, :kmax:4], 2),          # (T, k/4) for the spectrogram-ish panel
        })

    # -- the true envelope's iteration, on the first source's close-up frame
    ls = (np.asarray(A[0].env, np.float64) + np.asarray(A[0].fine, np.float64))[ti]
    e = lifter(cepstrum(ls), mp.q)                    # plain cepstral smoothing: cuts through peaks
    iters = [e]
    for _ in range(max(ITER_SHOW)):
        e = lifter(cepstrum(np.maximum(ls, e)), mp.q)
        iters.append(e)
    data["envIters"] = [r(iters[i][:kmax], 3) for i in ITER_SHOW]
    data["envIterN"] = ITER_SHOW

    # -- DTW: the real cost matrix and path, source 0 (flute) against source 1 (violin)
    fa, fb = A[0].dtw_feat, A[1].dtw_feat
    diff = fa[:, None, :] - fb[None, :, :]
    cost = np.sqrt((diff * diff).sum(-1))
    path = dtw_path(fa, fb, cfg.dtw_offdiag_penalty)
    data["dtw"] = {
        "a": PICK[0], "b": PICK[1],
        "cost": r(cost, 3),
        "path": path.tolist(),
        "maps": r(mp.maps, 3),
        "map1": r(path_to_map(path, fa.shape[0], fb.shape[0]), 3),
    }

    # -- the weighted average, in each of the three domains
    J, pos = mp.timeline(w)
    lev = np.stack([np.interp(pos[i], np.arange(len(a.level_db)), a.level_db)
                    for i, a in enumerate(A)])
    envs = np.stack([Morpher._interp_rows(np.asarray(a.env, np.float64), pos[i]) for i, a in enumerate(A)])
    fines = np.stack([Morpher._interp_rows(np.asarray(a.fine, np.float64), pos[i]) for i, a in enumerate(A)])
    fp = np.exp(cfg.fine_gamma * fines)
    fine_m = np.log(np.maximum(np.einsum("i,ijk->jk", w, fp), 1e-30)) / cfg.fine_gamma
    data["morph"] = {
        "J": int(J), "durS": round(J * hop_s, 4),
        "posEnds": r([p[-1] for p in pos], 2),
        "levelSrc": r(lev, 2),
        "levelMix": r(np.einsum("i,ij->j", w, lev), 2),
        "envSrc": r(envs[:, ti, :kmax], 3),
        "envMix": r(np.einsum("i,ik->k", w, envs[:, ti, :kmax]), 3),
        "fineSrc": r(fines[:, ti, :kmax], 3),
        "fineMix": r(fine_m[ti, :kmax], 3),
        # what a plain cross-fade of the same three would give, for the contrast
        "crossfadeEnv": r(np.log(np.maximum(
            np.einsum("i,ik->k", w, np.exp(envs[:, ti, :kmax])), 1e-30)), 3),
    }

    # -- the render itself, plus the sources, as audio for the video
    y = mp.render(w)
    sf.write(os.path.join(DST, "explainer-morph.wav"), y / (np.abs(y).max() + 1e-9) * 0.9, sr)
    for a, s in zip(A, sources):
        sf.write(os.path.join(DST, f"explainer-{a.name}.wav"),
                 s.audio / (np.abs(s.audio).max() + 1e-9) * 0.9, sr)
    data["morph"]["wave"] = r(decimate(y, 900), 4)

    json.dump(data, open(OUT, "w"), separators=(",", ":"))
    print(f"wrote {OUT} ({os.path.getsize(OUT)/1e6:.2f} MB)")
    print(f"  M={m.M} hop={m.hop} q={mp.q} f0={mp.f0:.1f}Hz  frames: "
          + ", ".join(f"{a.name} {len(a.level_db)}" for a in A) + f"  morph {J}")


if __name__ == "__main__":
    main()
