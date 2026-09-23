"""Command line for babymaker.

  python3 -m babymaker render  --src A.sf2:0:0 --src B.sf2:0:0 --key 60 --weights 0.5,0.5 --out baby.wav
  python3 -m babymaker grid    --src A.sf2:0:0 --src B.sf2:0:0 --src C.wav:C4 --key 60 --steps 4 --out out/morphs
  python3 -m babymaker random  --src ... --n 20 --seed 0 --out out/morphs
  python3 -m babymaker sources --src ... --out out/sources        (dump the prepared sources)
  python3 -m babymaker list    A.sf2                              (bank preset name)

Source spec: `font.sf2:bank:preset`, `font.sf2:preset` (bank 0), `sound.wav`, `sound.wav:note`
(the note the recording is at, e.g. 60 or C4; default = --key).  Several --key values
build one morph per key.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time

import numpy as np
import soundfile as sf

from .morph import MorphConfig, Morpher, random_weights, simplex_grid
from .sources import ensure_renderer, load_source, load_wav, parse_source_spec, render_sf2_notes


def _cfg_from_args(a) -> MorphConfig:
    cfg = MorphConfig(sr=a.sr)
    if getattr(a, "device", None):
        cfg.device = a.device
    for k, v in (a.set or {}).items():
        if not hasattr(cfg, k):
            raise SystemExit(f"unknown config key {k}; keys: {', '.join(cfg.to_dict())}")
        cur = getattr(cfg, k)
        setattr(cfg, k, type(cur)(v) if not isinstance(cur, bool) else str(v).lower() in ("1", "true", "yes"))
    return cfg


def _parse_set(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k] = v
    return out


def _weights_name(w) -> str:
    return "w" + "_".join(f"{x:.3f}".rstrip("0").rstrip(".") if abs(x) < 10 else f"{x:.2f}" for x in w)


def _prerender(a, cfg) -> dict:
    """{(spec, key): raw audio} - one renderer call (one soundfont load) per sf2 source."""
    raw = {}
    for spec in a.src:
        d = parse_source_spec(spec)
        if d["kind"] == "sf2":
            notes = render_sf2_notes(d["path"], d["bank"], d["preset"], a.key, a.vel, a.hold, a.tail, cfg.sr)
            for key, x in notes.items():
                raw[(spec, key)] = x
        else:
            x = load_wav(d["path"], cfg.sr)
            for key in a.key:
                raw[(spec, key)] = x
    return raw


def _build(a, key: int, raw: dict | None = None) -> Morpher:
    cfg = _cfg_from_args(a)
    t0 = time.time()
    raw = raw if raw is not None else _prerender(a, cfg)
    sources = [load_source(s, cfg.sr, key, a.vel, a.hold, a.tail, correct_pitch=not a.no_pitch_correct,
                           raw=raw[(s, key)]) for s in a.src]
    t1 = time.time()
    mp = Morpher(sources, cfg)
    if a.verbose:
        print(f"key {key}: loaded {len(sources)} sources in {t1 - t0:.1f}s, analysed in {time.time() - t1:.1f}s "
              f"(M={mp.mdct.M}, hop={mp.mdct.hop}, q={mp.q}, device={mp.describe()['device']})")
        for s in sources:
            off = s.meta.get("measured_cents_off")
            print(f"  {s.name}: {s.duration:.2f}s, pitch error {off:+.1f} cents" if off is not None
                  else f"  {s.name}: {s.duration:.2f}s, no stable pitch found (left as is)")
    return mp


def _write_source(s, path: str, sr: int):
    """The prepared source at its original level, clip-guarded like the morphs."""
    y = s.audio * s.meta.get("peak_rms_in", 1.0)
    if np.max(np.abs(y)) > 0.99:
        y = y / np.max(np.abs(y)) * 0.99
    sf.write(path, y.astype(np.float32), sr, subtype="FLOAT")


def _write(mp: Morpher, w, path: str, a, extra: dict | None = None):
    y, info = mp.render(w, return_info=True)
    if a.peak_norm:
        y = y / (np.max(np.abs(y)) + 1e-12) * 10 ** (-1 / 20)
    elif np.max(np.abs(y)) > 0.99:
        y = y / np.max(np.abs(y)) * 0.99
        info["clipped_guard"] = True
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sf.write(path, y.astype(np.float32), mp.cfg.sr, subtype="FLOAT")
    meta = {"file": os.path.basename(path), "key": mp.note, "f0_hz": mp.f0, **info, **(extra or {})}
    return meta


def cmd_render(a):
    metas = []
    raw = _prerender(a, _cfg_from_args(a))
    for key in a.key:
        mp = _build(a, key, raw)
        w = [float(x) for x in a.weights.split(",")]
        out = a.out if len(a.key) == 1 else os.path.join(a.out, f"key{key:03d}_{_weights_name(w)}.wav")
        metas.append(_write(mp, w, out, a, {"sources": mp.describe()["sources"]}))
        print(f"wrote {out} ({metas[-1]['duration_s']:.2f}s)")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(metas if len(metas) > 1 else metas[0], fh, indent=1)


_WORKER_MORPHER = None


def _render_job(args):
    w, path, a = args
    return _write(_WORKER_MORPHER, w, path, a)


def _batch(a, weight_sets_fn):
    global _WORKER_MORPHER
    os.makedirs(a.out, exist_ok=True)
    index = []
    raw = _prerender(a, _cfg_from_args(a))
    for key in a.key:
        morpher = _build(a, key, raw)
        W = weight_sets_fn(morpher)
        jobs = [(w, os.path.join(a.out, f"key{key:03d}_{_weights_name(w)}.wav"), a) for w in W]
        on_cuda = morpher.dev is not None and morpher.dev.type == "cuda"
        if a.jobs > 1 and len(jobs) > 1 and not on_cuda:
            # fork: the analysed Morpher is inherited by the workers, nothing is pickled
            _WORKER_MORPHER = morpher
            with multiprocessing.get_context("fork").Pool(min(a.jobs, len(jobs))) as pool:
                metas = pool.map(_render_job, jobs)
        else:
            _WORKER_MORPHER = morpher
            metas = [_render_job(j) for j in jobs]
        for (w, path, _), meta in zip(jobs, metas):
            index.append(meta)
            if a.verbose:
                print(f"  {os.path.basename(path)} ({meta['duration_s']:.2f}s)")
        mp_ = morpher
        if a.sources_too:
            for s in mp_.sources:
                _write_source(s, os.path.join(a.out, f"key{key:03d}_src_{s.name.replace(':', '-')}.wav"), mp_.cfg.sr)
        with open(os.path.join(a.out, f"key{key:03d}_morph.json"), "w") as fh:
            json.dump({"morph": mp_.describe(), "renders": [m for m in index if m["key"] == key]}, fh, indent=1)
        print(f"key {key}: {len(W)} morphs -> {a.out}")
    with open(os.path.join(a.out, "index.json"), "w") as fh:
        json.dump(index, fh, indent=1)


def cmd_grid(a):
    _batch(a, lambda mp: simplex_grid(mp.n, a.steps))


def cmd_random(a):
    rng = np.random.default_rng(a.seed)
    _batch(a, lambda mp: random_weights(mp.n, a.n, rng, a.alpha))


def cmd_sources(a):
    os.makedirs(a.out, exist_ok=True)
    raw = _prerender(a, _cfg_from_args(a))
    for key in a.key:
        morpher = _build(a, key, raw)
        for s in morpher.sources:
            p = os.path.join(a.out, f"key{key:03d}_{s.name.replace(':', '-')}.wav")
            _write_source(s, p, morpher.cfg.sr)
            print(f"wrote {p} ({s.duration:.2f}s)")
        with open(os.path.join(a.out, f"key{key:03d}_sources.json"), "w") as fh:
            json.dump(morpher.describe(), fh, indent=1)


def cmd_list(a):
    import subprocess
    exe = ensure_renderer()
    for path in a.sf2:
        out = subprocess.run([exe, "--list", os.path.expanduser(path)], capture_output=True, text=True)
        print(f"# {path}")
        print(out.stdout.rstrip())


def _add_common(p):
    p.add_argument("--src", action="append", required=True, help="source spec (repeat)")
    p.add_argument("--key", type=int, nargs="+", default=[60], help="MIDI key(s) to morph at")
    p.add_argument("--vel", type=int, default=100)
    p.add_argument("--hold", type=float, default=2.0, help="seconds the sf2 note is held")
    p.add_argument("--tail", type=float, default=1.5, help="release tail rendered after note-off")
    p.add_argument("--sr", type=int, default=44100)
    p.add_argument("--set", nargs="*", default=[], help="MorphConfig overrides, e.g. --set env_mode=log fine_gamma=0")
    p.add_argument("--no-pitch-correct", action="store_true")
    p.add_argument("--peak-norm", action="store_true", help="normalise each output to -1 dBFS peak")
    p.add_argument("--jobs", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)),
                   help="parallel renders per key (grid / random; ignored on CUDA)")
    p.add_argument("--device", default=None, help="auto (default) | cuda | cpu (numpy) | torch (torch on cpu)")
    p.add_argument("-v", "--verbose", action="store_true")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="babymaker", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("render", help="one morph for an explicit weight vector")
    _add_common(p)
    p.add_argument("--weights", required=True, help="comma-separated, one per source")
    p.add_argument("--out", required=True, help="output wav (a directory when several --key)")
    p.add_argument("--json", default=None, help="write render metadata here")
    p.set_defaults(fn=cmd_render)

    p = sub.add_parser("grid", help="every point of a simplex grid")
    _add_common(p)
    p.add_argument("--steps", type=int, default=4, help="subdivisions per edge")
    p.add_argument("--out", required=True)
    p.add_argument("--sources-too", action="store_true", help="also write the prepared sources")
    p.set_defaults(fn=cmd_grid)

    p = sub.add_parser("random", help="Dirichlet-random weight vectors")
    _add_common(p)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=1.0, help="Dirichlet concentration (<1: near corners)")
    p.add_argument("--out", required=True)
    p.add_argument("--sources-too", action="store_true")
    p.set_defaults(fn=cmd_random)

    p = sub.add_parser("sources", help="dump the prepared (aligned) sources")
    _add_common(p)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_sources)

    p = sub.add_parser("list", help="bank/preset inventory of soundfonts")
    p.add_argument("sf2", nargs="+")
    p.set_defaults(fn=cmd_list)

    a = ap.parse_args(argv)
    if hasattr(a, "set"):
        a.set = _parse_set(a.set)
    a.fn(a)


if __name__ == "__main__":
    main()
