#!/usr/bin/env python3
"""Write the four arrange plans the videos use.

The playhead mostly wanders through mixes (Dirichlet-random points, a new one every few
seconds, moving the whole time) and only touches a single source briefly, where the
narration names it. The long Satie ends on all eight pianos at once.

    python3 video/plans/make_plans.py        # then: python3 -m babymaker.arrange video/plans/<name>.json
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
F = "../fonts/"

PIANOS = {
    "a320":      {"path": F + "A320U.sf2", "preset": 0, "label": "A320U grand", "short": "A320U"},
    "fluid":     {"path": F + "FluidR3_GM.sf2", "preset": 0, "label": "FluidR3 Yamaha grand", "short": "FluidR3"},
    "tbo":       {"path": F + "TBOSf.sf2", "preset": 0, "label": "TBO grand", "short": "TBO"},
    "msgs":      {"path": F + "MS_Basic.sf2", "preset": 0, "label": "Microsoft GS Wavetable", "short": "MS GS"},
    "fairlight": {"path": F + "piano.sf2", "preset": 0, "label": "Fairlight piano", "short": "Fairlight"},
    "mkds":      {"path": F + "Mario Kart DS Soundfont.sf2", "preset": 0, "label": "Mario Kart DS piano",
                  "short": "Mario Kart DS"},
    "fb01":      {"path": F + "FB01.sf2", "preset": 1, "label": "Yamaha FB-01 (FM)", "short": "FB-01"},
    "opl4":      {"path": F + "OPL4_XI_GM_set__WIP_.sf2", "preset": 0, "label": "OPL4 (FM)", "short": "OPL4"},
}
BAND = {"fluid": {"path": F + "FluidR3_GM.sf2", "label": "FluidR3 GM"},
        "msgs": {"path": F + "MS_Basic.sf2", "label": "Microsoft GS Wavetable"},
        "fb01": {"path": F + "FB01.sf2", "label": "Yamaha FB-01 (FM)"}}


def wander(fonts, t0, t1, step, seed, alpha=1.3, cap=0.8, touch=(), start=None):
    """Keyframes [t, {font: w}] from t0 to t1: a new random mix every `step` seconds (no
    source above `cap`), plus `touch` = [(t, font)] single-source points with no hold."""
    rng = np.random.default_rng(seed)
    keys = []
    t = t0
    while t < t1 - 0.5 * step:
        while True:
            w = rng.dirichlet([alpha] * len(fonts))
            if w.max() <= cap:
                break
        keys.append([round(t, 3), {f: round(float(x), 3) for f, x in zip(fonts, w) if x > 0.005}])
        t += step * rng.uniform(0.8, 1.2)
    if start:
        keys[0] = [t0, start]
    keys.append([round(t1 - 0.01, 3), keys[-1][1]])
    for tt, f in touch:
        keys = [k for k in keys if abs(k[0] - tt) > 0.9 * step / 2] + [[tt, {f: 1.0}]]
    return sorted(keys, key=lambda k: k[0])


def main():
    sets = [("normal grands", ["a320", "fluid", "tbo"], 0.0, 27.0),
            ("nostalgia", ["msgs", "fairlight", "mkds"], 27.0, 53.4),
            ("all eight pianos", list(PIANOS), 53.4, 96.7)]
    path = []
    path += wander(sets[0][1], 0.0, 27.0, 2.6, 1)
    # the nostalgia line names each piano: touch them as it does, then mix
    path += wander(sets[1][1], 27.0, 53.4, 2.6, 2, touch=[(30.2, "msgs"), (34.4, "fairlight"), (37.4, "mkds")])
    path += wander(sets[2][1], 53.4, 96.7, 2.8, 3, alpha=0.7, cap=0.7)
    common = {"midi": "../midi/gymnopedie1.mid", "fonts": PIANOS, "reverb": 0.24, "reverb_s": 2.6,
              "key_span": 2, "hold_max": 7.0, "tail": 3.0}
    json.dump(dict(common, out="../renders/pianos_long.wav", path=path, start=0.0, end=96.7,
                   trios=[{"name": n, "fonts": f, "t0": a, "t1": b} for n, f, a, b in sets]),
              open(os.path.join(HERE, "pianos_long.json"), "w"), indent=1)

    trio = ["fluid", "msgs", "fairlight"]
    json.dump(dict(common, out="../renders/pianos_short.wav", path=wander(trio, 9.0, 17.9, 1.7, 4),
                   start=9.0, end=17.9, tail=1.6,
                   trios=[{"name": "", "fonts": trio, "t0": 9.0, "t1": 17.9}]),
              open(os.path.join(HERE, "pianos_short.json"), "w"), indent=1)

    bcommon = {"midi": "../midi/dotabata.mid", "fonts": BAND, "reverb": 0.14, "reverb_s": 1.8,
               "key_span": 2, "hold_max": 4.0, "tail": 1.2, "bend_range": 2.0}
    band = list(BAND)
    json.dump(dict(bcommon, out="../renders/band_long.wav", path=wander(band, 3.4, 57.5, 3.0, 5, alpha=1.5),
                   start=3.4, end=57.5, trios=[{"name": "", "fonts": band, "t0": 3.4, "t1": 57.5}]),
              open(os.path.join(HERE, "band_long.json"), "w"), indent=1)
    json.dump(dict(bcommon, out="../renders/band_short.wav", path=wander(band, 8.8, 14.3, 1.4, 6, alpha=1.5),
                   start=8.8, end=14.3, tail=0.8, trios=[{"name": "", "fonts": band, "t0": 8.8, "t1": 14.3}]),
              open(os.path.join(HERE, "band_short.json"), "w"), indent=1)
    for n in ("pianos_long", "pianos_short", "band_long", "band_short"):
        p = json.load(open(os.path.join(HERE, f"{n}.json")))["path"]
        corner = sum(1 for _, w in p if max(w.values()) > 0.99)
        print(f"{n:13s} {len(p):3d} keyframes, {corner} single-source touches")


if __name__ == "__main__":
    main()
