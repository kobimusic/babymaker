#!/usr/bin/env python3
"""Build the demo's instrument set from the Sonatina Symphonic Orchestra.

For each instrument: take the SSO sample nearest E5, cut it to CUT_S with a
fade, hand the cuts to babymaker (`random --sources-too`) so the browser gets
the same prepared sources the Python morph analyses, plus reference renders
for the JS-vs-Python test (reference/), then export 16-bit WAVs + a manifest
to web/public/media/.

    SSO=/path/to/sso/Samples python3 tools/prep-babies.py
"""
import json, os, shutil, subprocess, sys, tempfile
import numpy as np
import soundfile as sf

# the "Samples" folder of https://github.com/peastman/sso
SSO = os.path.expanduser(os.environ.get("SSO", "~/sfz/sso/Sonatina Symphonic Orchestra/Samples"))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "reference")
DST = os.path.join(ROOT, "web", "public", "media")

KEY = 76                 # E5: the loop's notes run G4..C6, median E5
CUT_S, FADE_S = 1.7, 0.06

# id, label, sample, the MIDI note that sample is at
INSTRUMENTS = [
    ("harp",    "harp",    "Harp/harp-d#5.wav",           75),
    ("piano",   "piano",   "Grand Piano/Mf E5.flac",      76),
    ("flute",   "flute",   "Flute/flute-d#5.wav",         75),
    ("violin",  "violin",  "Violin/violin-e5.wav",        76),
    ("viola",   "viola",   "Viola/viola-sus-d#5.flac",    75),
    ("oboe",    "oboe",    "Oboe/oboe-e5.wav",            76),
]
DEFAULT = [2, 3, 0]      # flute, violin, harp: two sustains and a decay

CREDIT = {
    "name": "Sonatina Symphonic Orchestra",
    "author": "Mattias Westlund",
    "license": "CC Sampling Plus 1.0",
    "url": "https://github.com/peastman/sso",
}


def cut(path):
    x, sr = sf.read(path, dtype="float64", always_2d=True)
    x = x.mean(axis=1)
    thr = np.abs(x).max() * 10 ** (-60 / 20)
    on = int(np.argmax(np.abs(x) > thr))
    x = x[on:on + int(CUT_S * sr)]
    nf = min(len(x), int(FADE_S * sr))
    x[-nf:] *= np.linspace(1, 0, nf)
    return x, sr


def main():
    tmp = tempfile.mkdtemp(prefix="babies-")
    srcs = []
    for iid, _, rel, note in INSTRUMENTS:
        x, sr = cut(os.path.join(SSO, rel))
        p = os.path.join(tmp, f"{iid}.wav")
        sf.write(p, x, sr, subtype="FLOAT")
        srcs += ["--src", f"{p}:{note}"]
        print(f"  {iid:9s} {rel:28s} {len(x)/sr:.2f}s")

    shutil.rmtree(OUT, ignore_errors=True)
    cmd = [sys.executable, "-m", "babymaker", "random", *srcs, "--key", str(KEY),
           "--n", "10", "--seed", "0", "--alpha", "0.7", "--sources-too", "--jobs", "1",
           "--device", "cpu", "--set", "env_mode=log", "gl_iters=2", "oversample=2",
           "--out", OUT, "-v"]
    subprocess.run(cmd, cwd=ROOT, check=True)

    morph = json.load(open(os.path.join(OUT, f"key{KEY:03d}_morph.json")))["morph"]
    prms = {s["name"]: s["peak_rms_in"] for s in morph["sources"]}

    for f in os.listdir(DST):
        if f.endswith(".wav"):
            os.remove(os.path.join(DST, f))
    manifest = {"note": KEY, "sr": 44100, "default": DEFAULT, "credit": CREDIT,
                "midi": "alla-turca.mid", "instruments": []}
    for iid, label, _, _ in INSTRUMENTS:
        x, sr = sf.read(os.path.join(OUT, f"key{KEY:03d}_src_{iid}.wav"))
        assert sr == 44100 and np.abs(x).max() <= 1.0
        sf.write(os.path.join(DST, f"{iid}.wav"), x, sr, subtype="PCM_16")   # re-normalised in the browser
        manifest["instruments"].append({"id": iid, "label": label, "file": f"{iid}.wav",
                                        "peakRmsIn": prms[iid]})
    json.dump(manifest, open(os.path.join(DST, "manifest.json"), "w"), indent=1)
    shutil.rmtree(tmp)
    print("wrote", os.path.relpath(DST, ROOT), "and", os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    main()
