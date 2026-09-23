"""Play a MIDI file through morphed soundfonts, with the morph weights moving over time.

Every note is played by the baby of the fonts at the weights in effect at its onset, the
same way the browser demo works: notes already sounding keep their sample, new notes get
the new one.  Per track, each font plays the track's own program (or a fixed preset from
the plan); pitched tracks are morphed every 2 * key_span + 1 semitones and resampled in
between, drums key by key.  The sampler honours velocity, CC7 / CC11 volume, CC10 pan, the
sustain pedal and the pitch wheel, and adds a plain convolution reverb.

    python3 -m babymaker.arrange plan.json            # writes plan["out"] (.wav) + .json

plan.json:
    {"midi": "song.mid", "out": "out/song.wav",
     "fonts": {"fluid": {"path": "FluidR3_GM.sf2"},
               "fm":    {"path": "FB01.sf2", "preset": 4},        # force a preset
               "nena":  {"path": "Nena.SF2", "drums_only": true}}, # only plays drum tracks
     "path": [[0, {"fluid": 1}], [20, {"fluid": 0.5, "fm": 0.5}], [40, {"fm": 1}]],
     "start": 0, "end": null, "key_span": 2, "hold_max": 6.0, "reverb": 0.18,
     "level_match": true, "quant": 24}

`path` is a list of [seconds, {font: weight}] keyframes, linearly interpolated.
The .json written next to the audio lists every note with the weights it was played at.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import soundfile as sf

from .morph import MorphConfig, Morpher
from .sources import load_source, render_sf2_notes


# ----------------------------------------------------------------------------- weights

class WeightPath:
    """Keyframes [(t, {font: w})], linearly interpolated, clamped at both ends."""

    def __init__(self, keys, quant: int = 24):
        self.keys = sorted([(float(t), dict(w)) for t, w in keys], key=lambda k: k[0])
        self.fonts = sorted({f for _, w in self.keys for f in w})
        self.quant = quant

    def at(self, t: float) -> dict:
        ks = self.keys
        if t <= ks[0][0]:
            w = ks[0][1]
        elif t >= ks[-1][0]:
            w = ks[-1][1]
        else:
            i = max(j for j in range(len(ks)) if ks[j][0] <= t)
            (t0, a), (t1, b) = ks[i], ks[i + 1]
            u = (t - t0) / max(1e-9, t1 - t0)
            w = {f: (1 - u) * a.get(f, 0.0) + u * b.get(f, 0.0) for f in set(a) | set(b)}
        return self.quantise(w)

    def quantise(self, w: dict) -> dict:
        """Round to multiples of 1/quant on the simplex, so renders are shared between notes."""
        tot = sum(max(0.0, v) for v in w.values())
        items = sorted(((f, max(0.0, v) / tot) for f, v in w.items() if v > 0), key=lambda kv: -kv[1])
        q = self.quant
        units = {f: int(np.floor(v * q)) for f, v in items}
        rest = q - sum(units.values())
        for f, v in sorted(items, key=lambda kv: -(kv[1] * q - np.floor(kv[1] * q)))[:rest]:
            units[f] += 1
        return {f: u / q for f, u in units.items() if u > 0}


# ----------------------------------------------------------------------------- one track

def plan_centers(pitches, span: int) -> dict:
    """{pitch: center} such that every pitch is within +-span of its center (greedy groups)."""
    ps = sorted(set(int(p) for p in pitches))
    out, i = {}, 0
    while i < len(ps):
        group = [p for p in ps[i:] if p <= ps[i] + 2 * span]
        c = int(round((group[0] + group[-1]) / 2.0))
        for p in group:
            out[p] = c
        i += len(group)
    return out


def _onset(y: np.ndarray, db: float = -40.0) -> int:
    m = np.abs(y)
    if m.max() <= 0:
        return 0
    return max(0, int(np.argmax(m > m.max() * 10 ** (db / 20))) - 44)


class Track:
    """One MIDI track, played by a morph of every font it can be found in."""

    def __init__(self, name, program, is_drum, notes, fonts: dict, sr, key_span, hold_max, tail_s,
                 level_match, cfg: MorphConfig, log=print):
        self.name, self.program, self.is_drum, self.sr = name, program, is_drum, sr
        self.level_match, self.cfg, self.log = level_match, cfg, log
        pitches = [n.pitch for n in notes]
        self.center_of = plan_centers(pitches, 0 if is_drum else key_span)
        centers = sorted(set(self.center_of.values()))
        longest = max(n.end - n.start for n in notes)
        self.hold = float(min(hold_max, max(0.4, longest)))
        self.srcs = {}          # (font, center) -> prepared Source
        self.fonts_ok = set()
        for fname, f in fonts.items():
            if f.get("drums_only") and not is_drum:
                continue
            if f.get("melodic_only") and is_drum:
                continue
            bank = 128 if is_drum else f.get("bank", 0)
            preset = f.get("drum_preset", program) if is_drum else f.get("preset", program)
            try:
                raw = render_sf2_notes(f["path"], bank, preset, centers, 100, self.hold, tail_s, sr)
            except RuntimeError as e:
                log(f"    {fname}: {e}")
                continue
            got = 0
            for c in centers:
                try:
                    self.srcs[(fname, c)] = load_source(f"{f['path']}:{bank}:{preset}", sr, c, 100, self.hold,
                                                        tail_s, correct_pitch=not is_drum, raw=raw[c])
                    got += 1
                except ValueError:      # silent: this font has no zone for this key
                    pass
            if got:
                self.fonts_ok.add(fname)
        self.morphers = {}
        self.cache = {}

    def sample(self, center: int, w: dict):
        """The baby for this key at these weights (fonts missing the key are left out)."""
        w = {f: v for f, v in w.items() if (f, center) in self.srcs}
        if not w:
            return None
        tot = sum(w.values())
        w = {f: v / tot for f, v in w.items()}
        key = (center, tuple(sorted(w.items())))
        if key in self.cache:
            return self.cache[key]
        fonts = tuple(sorted(w))
        mk = (center, fonts)
        if mk not in self.morphers:
            self.morphers[mk] = Morpher([self.srcs[(f, center)] for f in fonts], self.cfg)
        y = self.morphers[mk].render(np.array([w[f] for f in fonts]))
        if self.level_match:
            # peak short-time RMS -> the geometric mean of every font's level for this key, so a
            # quiet font does not make its corner of the triangle quiet: timbre moves, level stays
            lv = [self.srcs[(f, center)].meta["peak_rms_in"] for f in self.fonts_ok if (f, center) in self.srcs]
            win = int(0.02 * self.sr)
            e = np.convolve(y ** 2, np.ones(win) / win, mode="same")
            y = y / (np.sqrt(e.max()) + 1e-12) * float(np.exp(np.mean(np.log(lv))))
        y = np.asarray(y[_onset(y):], np.float32)
        self.cache[key] = y
        return y


# ----------------------------------------------------------------------------- sampler

def _cc_at(ccs, number, t, default):
    v = default
    for c in ccs:
        if c.number != number:
            continue
        if c.time > t + 1e-6:
            break
        v = c.value
    return v


def _pedal_spans(ccs, total):
    spans, down = [], None
    for c in ccs:
        if c.number != 64:
            continue
        if c.value >= 64 and down is None:
            down = c.time
        elif c.value < 64 and down is not None:
            spans.append((down, c.time))
            down = None
    if down is not None:
        spans.append((down, total))
    return spans


def reverb_ir(sr, seconds=2.2, seed=0):
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    ir = rng.standard_normal((2, n)) * np.exp(-6.9 * t / seconds)      # -60 dB at `seconds`
    # darker as it decays: a one-pole lowpass whose cutoff falls with time
    from scipy.signal import lfilter
    for ch in range(2):
        ir[ch] = lfilter([0.35], [1, -0.65], ir[ch])
    ir[:, : int(0.012 * sr)] = 0                                          # pre-delay
    return ir / np.sqrt((ir ** 2).sum(axis=1, keepdims=True))


def render(plan: dict, log=print) -> dict:
    import pretty_midi
    t_start = time.time()
    sr = int(plan.get("sr", 44100))
    pm = pretty_midi.PrettyMIDI(plan["midi"])
    start = float(plan.get("start", 0.0))
    end = plan.get("end") or pm.get_end_time()
    end = float(end)
    path = WeightPath(plan["path"], plan.get("quant", 24))
    fonts = {k: dict(v, path=os.path.expanduser(v["path"])) for k, v in plan["fonts"].items()}
    fonts = {k: v for k, v in fonts.items() if k in path.fonts}
    cfg = MorphConfig(sr=sr, **plan.get("morph", {}))
    tail = float(plan.get("tail", 1.2))
    total = end - start + tail + 2.5
    n = int(total * sr)
    dry = np.zeros((2, n))
    wet = np.zeros((2, n))
    notes_out = []
    skip = set(plan.get("skip_tracks", []))

    for ti, inst in enumerate(pm.instruments):
        notes = [x for x in inst.notes if start <= x.start < end]
        if not notes or ti in skip:
            continue
        name = inst.name.strip() or ("drums" if inst.is_drum else pretty_midi.program_to_instrument_name(inst.program))
        t0 = time.time()
        ccs = sorted(inst.control_changes, key=lambda c: c.time)
        tr = Track(name, inst.program, inst.is_drum, notes, fonts, sr, int(plan.get("key_span", 2)),
                   float(plan.get("hold_max", 6.0)), tail, bool(plan.get("level_match", True)), cfg, log)
        if not tr.fonts_ok:
            log(f"  track {ti} {name}: no font plays it, skipped")
            continue
        bends = sorted(inst.pitch_bends, key=lambda b: b.time)
        rng = float(plan.get("bend_range", 2.0))
        if bends and not inst.is_drum:
            bt = np.array([0.0] + [b.time - start for b in bends])
            bv = np.array([0.0] + [b.pitch / 8192.0 * rng for b in bends])
        pedal = _pedal_spans(ccs, pm.get_end_time())
        rel_n = int(float(plan.get("release", 0.18)) * sr)
        send = float(plan.get("reverb", 0.18))
        for x in notes:
            c = tr.center_of[int(x.pitch)]
            w = path.at(x.start)
            smp = tr.sample(c, w)
            if smp is None:
                continue
            off = x.end
            for a, b in pedal:
                if a <= off < b:
                    off = b
                    break
            i0 = int((x.start - start) * sr)
            i_off = int((min(off, end + tail) - start) * sr)
            i1 = min(n, i0 + len(smp)) if inst.is_drum else min(n, i_off + rel_n, i0 + len(smp))
            if i1 <= i0:
                continue
            L = i1 - i0
            if inst.is_drum:
                seg = smp[:L].astype(np.float64)
            else:
                semis = float(x.pitch - c)
                if bends:
                    semis = semis + np.interp((i0 + np.arange(L)) / sr, bt, bv)
                ratio = np.broadcast_to(2.0 ** (np.asarray(semis) / 12.0), (L,))
                pos = np.cumsum(ratio) - ratio[0]
                ok = pos < len(smp) - 1
                p = pos[ok]
                k = p.astype(int)
                fr = p - k
                seg = np.zeros(L)
                seg[ok] = smp[k] * (1 - fr) + smp[k + 1] * fr
                if i_off < i1:
                    j = np.arange(max(i_off, i0), i1) - i_off
                    seg[max(i_off, i0) - i0:] *= np.exp(-5.0 * j / max(rel_n, 1))
            vol = (_cc_at(ccs, 7, x.start, 100) / 127.0) * (_cc_at(ccs, 11, x.start, 127) / 127.0)
            pan = (_cc_at(ccs, 10, x.start, 64) - 64) / 64.0
            gain = (np.clip(x.velocity, 1, 127) / 127.0) ** 1.6 * vol ** 1.5
            lr = np.array([np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)]) * np.sqrt(2)
            s = seg * gain
            dry[:, i0:i1] += lr[:, None] * s
            rv = send * (_cc_at(ccs, 91, x.start, 40) / 40.0 if plan.get("cc91", True) else 1.0)
            wet[:, i0:i1] += lr[:, None] * s * rv
            notes_out.append({"track": ti, "t": round(x.start - start, 4), "end": round(off - start, 4),
                              "pitch": int(x.pitch), "vel": int(x.velocity), "w": w})
        log(f"  track {ti:2d} {name[:22]:22s} {len(notes):5d} notes  fonts {sorted(tr.fonts_ok)}  "
            f"{len(tr.cache)} babies  {time.time() - t0:5.1f}s")
        del tr

    if np.any(wet):
        from scipy.signal import fftconvolve
        ir = reverb_ir(sr, float(plan.get("reverb_s", 2.2)))
        rev = np.stack([fftconvolve(wet[ch], ir[ch])[:n] for ch in range(2)])
        dry = dry + rev
    peak = np.abs(dry).max()
    dry *= float(plan.get("peak", 0.89)) / (peak + 1e-9)
    last = int((end - start + tail + 0.5) * sr)
    dry = dry[:, :min(n, last)]
    fade = int(0.4 * sr)
    dry[:, -fade:] *= np.linspace(1, 0, fade)
    out = plan["out"]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    sf.write(out, dry.T.astype(np.float32), sr)
    meta = {"midi": plan["midi"], "sr": sr, "start": start, "end": end, "duration": dry.shape[1] / sr,
            "fonts": list(fonts), "path": plan["path"],
            "tracks": {ti: {"name": inst.name.strip() or ("drums" if inst.is_drum else
                                                          pretty_midi.program_to_instrument_name(inst.program)),
                            "program": int(inst.program), "drum": bool(inst.is_drum)}
                       for ti, inst in enumerate(pm.instruments)},
            "notes": notes_out, "render_s": round(time.time() - t_start, 1)}
    json.dump(meta, open(os.path.splitext(out)[0] + ".json", "w"))
    log(f"wrote {out}: {meta['duration']:.1f}s, {len(notes_out)} notes, {meta['render_s']}s")
    return meta


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        sys.exit(__doc__)
    for p in argv:
        plan = json.load(open(p))
        base = os.path.dirname(os.path.abspath(p))
        for k in ("midi", "out"):
            plan[k] = os.path.join(base, os.path.expanduser(plan[k]))
        for f in plan["fonts"].values():
            f["path"] = os.path.join(base, os.path.expanduser(f["path"]))
        render(plan)


if __name__ == "__main__":
    main()
