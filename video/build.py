#!/usr/bin/env python3
"""Build the two babymaker videos: a 30-second short and a longer, more technical cut.

Needs, in order (see video/README.md):
  video/work/vo_short, vo_long      narration, from video/tts.py
  video/work/explainer.json (+wavs) from video/explainer_data.py
  renders/*.wav + .json             the MIDI renders, from `python3 -m babymaker.arrange <plan>`

    python3 video/build.py short|long [--renders DIR] [--jobs N]

Each segment is rendered on its own (manim for the explainer chapters, cairo for the demo
scenes), the narration is laid over them at the times computed here, the music is ducked
under the voice, and the whole thing is encoded with captions (burned into the short,
as a sidecar .srt for both).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WORK = os.path.join(HERE, "work")
sys.path.insert(0, HERE)
import script  # noqa: E402

SR = 44100
FPS = 30
GAP = 0.45                 # breath between two narration lines
VOICE_PEAK = 0.72
MUSIC_PEAK = 0.62
SFX_PEAK = 0.30
DUCK = 0.60                # how far the music dips under the voice (only with --voice)
READ_WPS = 2.8             # Nathan's reading pace (he talks at ~3.0 words/s), for timing without a voice
OUTRO = os.path.expanduser("~/ribbon_logo/kobimusic/outro/kobimusic_outro.mp4")
OUTRO_CUT, OUTRO_FADE = 5.0, 0.8
LEAD = {"why": 3.6}        # picture before a chapter's first line (the title card)
LEAD_DEFAULT = 0.6

# the browser demo's layout and default trio (flute, violin, harp)
DOTS = [(22.0, 26.0), (78.0, 26.0), (50.0, 72.0)]
CENTRE = (50.0, 42.0)
TRIO = {"labels": ["flute", "violin", "harp"], "selection": [2, 3, 0]}


def run(cmd, **kw):
    subprocess.run(cmd, check=True, **kw)


def load(path, sr=SR):
    x, s = sf.read(path, dtype="float64", always_2d=True)
    x = x.mean(1) if x.shape[1] == 1 else x
    if s != sr:
        import soxr
        x = soxr.resample(x, s, sr, quality="VHQ")
    return x


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nw=1:nk=1", path], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


class Voice:
    """The narration of one video: every line's clip, levelled, and its caption."""

    def __init__(self, which, use_clips=False):
        sc = script.SHORT if which == "short" else script.LONG
        self.lines = script.lines(sc)
        self.clip, self.dur = {}, {}
        if not use_clips:
            # no narration in the film: time the picture for a live read of the script
            for k, v in self.lines.items():
                self.dur[k] = len(v["say"].split()) / READ_WPS + 0.3
            return
        d = os.path.join(WORK, f"vo_{which}")
        idx = json.load(open(os.path.join(d, "index.json")))
        for k in self.lines:
            x = load(os.path.join(d, f"{k}.wav"))
            x = x / (np.sqrt(np.mean(x ** 2)) + 1e-9) * 0.1          # every line at the same loudness
            self.clip[k], self.dur[k] = x, len(x) / SR
            if idx[k]["score"] < 0.9:
                print(f"  warning: {k} matched its text at {idx[k]['score']}: heard {idx[k]['heard']!r}")

    def ids(self, section):
        return [k for k in self.lines if k.split(".")[0] == section]


class Seq:
    """Lines placed one after another from a start time, each no earlier than asked."""

    def __init__(self, vo):
        self.vo, self.t, self.at = vo, 0.0, []

    def say(self, k, not_before=0.0):
        s = max(self.t + (GAP if self.at else 0.0), not_before)
        self.at.append((round(s, 3), k))
        self.t = s + self.vo.dur[k]
        return s, self.t


# ----------------------------------------------------------------------------- segments

def seg_manim(cls, vo, media, quality, ids=None):
    """One explainer chapter; its narration timing is written for the scene to sync to."""
    key = cls.KEY
    ids = ids or vo.ids(key)
    t = LEAD.get(key, LEAD_DEFAULT)
    starts, ends = [], []
    for k in ids:
        starts.append(round(t, 3))
        t += vo.dur[k]
        ends.append(round(t, 3))
        t += GAP
    return {"kind": "manim", "cls": cls.__name__, "key": key,
            "voice": list(zip(starts, ids)), "narr": {"starts": starts, "ends": ends}}


def seg_triangle(name, vo, plan, dots=DOTS, labels=TRIO["labels"], selection=TRIO["selection"]):
    """plan(vo) -> (voice placements, head keyframes, duration)."""
    at, head, dur = plan(vo)
    return {"kind": "triangle", "name": name, "voice": at, "duration": dur,
            "spec": {"kind": "triangle", "fps": FPS, "duration": dur, "dots": dots, "labels": labels,
                     "selection": selection, "head": head, "music_start": 0.15, "music_end": dur - 0.6,
                     "fade_in": 0.35, "fade_out": 0.4}}


def seg_roll(name, plan_path, meta_path, wav, title, subtitle, voice, cut=None):
    meta = json.load(open(meta_path))
    dur = min(cut or 1e9, meta["duration"])
    return {"kind": "roll", "name": name, "voice": voice, "duration": dur, "music": wav,
            "spec": {"kind": "roll", "fps": FPS, "duration": dur, "plan": plan_path, "meta": meta_path,
                     "audio": wav, "title": title, "subtitle": subtitle, "window": 8.0,
                     "fade_in": 0.35, "fade_out": 0.45}}


def seg_clip(name, path, cut, fade, voice=()):
    """An existing video (the kobimusic outro), cut at `cut` s with a fade to black."""
    return {"kind": "clip", "name": name, "voice": list(voice), "duration": cut, "src": path, "fade": fade}


def seg_card(name, dur, title, lines, voice=()):
    return {"kind": "card", "name": name, "voice": list(voice), "duration": dur,
            "spec": {"kind": "card", "fps": FPS, "duration": dur, "title": title, "lines": lines}}


# ----------------------------------------------------------------------------- the two cuts

def wander(dots, t0, t1, step, seed, alpha=2.0, start=None):
    """Head keyframes that keep moving through mixes: each stop is a random convex
    combination of the dots (Dirichlet(alpha) weights), reached by easing the whole way."""
    rng = np.random.default_rng(seed)
    d = np.asarray(dots, float)
    keys, t = [], t0
    if start is not None:
        keys.append([t0, tuple(start), 0.0])
    while t + step <= t1 + 1e-6:
        prev = t
        t = min(t1, t + step * rng.uniform(0.85, 1.15))
        p = rng.dirichlet([alpha] * len(d)) @ d
        # a move must fit inside its own gap, or Keyframes jumps when the last one lands
        keys.append([round(t, 3), (round(p[0], 2), round(p[1], 2)), round(0.92 * (t - prev), 3)])
    return keys


def tri_cold(vo):
    s = Seq(vo)
    a0, e0 = s.say("cold.0", 1.0)
    a1, e1 = s.say("cold.1")
    dur = e1 + 2.8
    head = [[0, CENTRE, 0]] + wander(DOTS, 0.0, dur - 0.4, 2.2, 11)
    return s.at, head, dur


def tri_demo(vo):
    """The walkthrough touches each corner only while the line names it, then mixes."""
    s = Seq(vo)
    a0, e0 = s.say("demo.0", 0.8)
    a1, e1 = s.say("demo.1")
    a2, e2 = s.say("demo.2", e1 + 0.8)
    a3, e3 = s.say("demo.3", e2 + 0.9)
    a4, e4 = s.say("demo.4", e3 + 0.8)
    mid_ab = ((DOTS[0][0] + DOTS[1][0]) / 2, (DOTS[0][1] + DOTS[1][1]) / 2 + 6)
    head = ([[0, CENTRE, 0]] + wander(DOTS, 0.0, a1, 2.0, 12) +
            [[a1 + 0.75 * (e1 - a1), DOTS[0], 1.8],                  # "onto the flute"
             [e1 + 0.5, mid_ab, 0.9],
             [a2 + 0.9, DOTS[1], 0.9],                               # "same thing with the violin"
             [a3 + 1.4, (54.0, 40.0), 1.6],                          # the harp line: a slow approach
             [e3 - 0.3, DOTS[2], e3 - 0.3 - (a3 + 1.4)]] +
            wander(DOTS, e3 + 0.3, e4 + 6.0, 2.0, 13, alpha=2.5))    # and the middle, moving
    return s.at, head, e4 + 6.4


def tri_short(vo):
    head = [[0, (46.0, 38.0), 0]] + wander(DOTS, 0.0, 7.2, 1.25, 14, alpha=1.6)
    return [], head, 7.4


# the six instruments from the site, on a hexagon (the stage is 100 x 75; y is drawn at 0.75)
HEX_DOTS = [(round(50 + 27 * np.cos(-np.pi / 2 + k * np.pi / 3), 2),
             round((37.5 + 27 * np.sin(-np.pi / 2 + k * np.pi / 3)) / 0.75, 2)) for k in range(6)]
HEX = {"labels": ["harp", "piano", "flute", "violin", "viola", "oboe"], "selection": [0, 1, 2, 3, 4, 5]}


def tri_six(vo):
    s = Seq(vo)
    a0, e0 = s.say("six.0", 0.8)
    a1, e1 = s.say("six.1")
    dur = e1 + 7.0
    head = [[0, (50.0, 50.0), 0]] + wander(HEX_DOTS, 0.0, dur - 0.5, 2.0, 15, alpha=0.45)
    return s.at, head, dur


def roll_voice(vo, wishes):
    """wishes: [(line id, earliest local time)]."""
    s = Seq(vo)
    for k, t in wishes:
        s.say(k, t)
    return s.at


def plan_short(vo, renders):
    R = lambda n, ext: os.path.join(renders, f"{n}.{ext}")
    P = lambda n: os.path.join(renders, "..", "plans", f"{n}.json")
    bench = json.load(open(os.path.join(WORK, "bench.json")))
    segs = [
        seg_card("title", 1.8, "making soundfont babies", [("babymaker", "accent")]),
        seg_triangle("tri_short", vo, tri_short),
        seg_roll("pianos_short", P("pianos_short"), R("pianos_short", "json"), R("pianos_short", "wav"),
                 "Satie, Gymnopédie No. 1", "three pianos from three soundfonts", [], cut=6.6),
        seg_roll("band_short", P("band_short"), R("band_short", "json"), R("band_short", "wav"),
                 "DOTABATA", "every track morphed", [], cut=4.4),
        seg_card("speed", 4.8, f"{bench['cuda']['render_s'] * 1000:.0f} ms per note",
                 [("no neural network, only DCTs", "accent"), ("no training, any three recordings", "faint")]),
        seg_clip("kobimusic", OUTRO, OUTRO_CUT, OUTRO_FADE),
    ]
    # the short's narration runs across segment boundaries, so it is placed on the global clock
    off = np.cumsum([0] + [s["duration"] for s in segs])
    g = Seq(vo)
    g.say("open.0", 0.25)
    g.say("open.1")
    g.say("pianos.0", off[2] + 0.2)
    g.say("band.0", off[3] + 0.15)
    _, end = g.say("speed.0", off[4] + 0.1)
    assert end <= off[5] + 0.3, f"the short's narration runs to {end:.2f}s, into the outro at {off[5]:.2f}s"
    return segs, g.at


def plan_long(vo, renders, scenes):
    R = lambda n, ext: os.path.join(renders, f"{n}.{ext}")
    P = lambda n: os.path.join(renders, "..", "plans", f"{n}.json")
    by_key = {c.KEY: c for c in scenes.ORDER}
    segs = [seg_triangle("cold", vo, tri_cold), seg_manim(by_key["why"], vo, None, None),
            seg_triangle("demo", vo, tri_demo),
            seg_triangle("six", vo, tri_six, dots=HEX_DOTS, labels=HEX["labels"], selection=HEX["selection"])]
    for k in ("gap", "transform", "split", "align", "average", "synth", "result", "dct", "speed"):
        segs.append(seg_manim(by_key[k], vo, None, None))
    segs.append(seg_roll("pianos_long", P("pianos_long"), R("pianos_long", "json"), R("pianos_long", "wav"),
                         "Satie, Gymnopédie No. 1", "each note is played by the piano at the playhead",
                         roll_voice(vo, [("pianos.0", 1.0), ("pianos.1", 0), ("pianos.2", 27.6),
                                         ("pianos.3", 54.0), ("pianos.4", 78.0)])))
    segs.append(seg_roll("band_long", P("band_long"), R("band_long", "json"), R("band_long", "wav"),
                         "DOTABATA", "Nena MIDI collection, all sixteen tracks morphed",
                         roll_voice(vo, [("band.0", 0.8), ("band.1", 0), ("band.2", 13.8), ("band.3", 38.5)])))
    segs.append(seg_manim(by_key["outro"], vo, None, None, ids=["outro.0"]))
    segs.append(seg_clip("kobimusic", OUTRO, OUTRO_CUT, OUTRO_FADE, voice=[(0.4, "outro.1")]))
    return segs, None


# ----------------------------------------------------------------------------- rendering

def render_segments(segs, out_dir, jobs, quality, only=None):
    os.makedirs(out_dir, exist_ok=True)
    narr = {s["key"]: s["narr"] for s in segs if s["kind"] == "manim"}
    narr_path = os.path.join(out_dir, "narration.json")
    json.dump(narr, open(narr_path, "w"), indent=1)
    cue_dir = os.path.join(out_dir, "cues")
    os.makedirs(cue_dir, exist_ok=True)
    media = os.path.join(out_dir, "manim")

    def one(s):
        name = s.get("cls") or s["name"]
        mp4 = os.path.join(out_dir, f"{name}.mp4")
        if only is not None and name not in only and os.path.exists(mp4):
            # reuse the segment rendered last time
            s["video"] = mp4
            if s["kind"] == "manim":
                s["duration"] = duration(mp4)
                cf = os.path.join(cue_dir, f"{s['key']}.json")
                s["cues"] = json.load(open(cf)) if os.path.exists(cf) else {}
            elif s["kind"] == "triangle":
                s["music"] = os.path.join(out_dir, f"{name}.music.wav")
            elif s["kind"] == "clip":
                s["music"], s["music_raw"] = os.path.join(out_dir, f"{name}.wav"), True
            print(f"  {name:14s} {s['duration']:6.1f}s (kept)", flush=True)
            return s
        if s["kind"] == "clip":
            run(["ffmpeg", "-loglevel", "error", "-y", "-i", s["src"], "-t", f"{s['duration']:.3f}",
                 "-vf", f"fps={FPS},scale=1920:1080,format=yuv420p,fade=t=out:st={s['duration'] - s['fade']:.3f}:d={s['fade']:.3f}",
                 "-an", "-c:v", "libx264", "-crf", "16", "-preset", "medium", mp4])
            wav = os.path.join(out_dir, f"{name}.wav")
            run(["ffmpeg", "-loglevel", "error", "-y", "-i", s["src"], "-t", f"{s['duration']:.3f}",
                 "-af", f"afade=t=out:st={s['duration'] - s['fade']:.3f}:d={s['fade']:.3f}", "-ar", str(SR), "-ac", "2", wav])
            s["music"] = wav
            s["music_raw"] = True
        elif s["kind"] == "manim":
            env = dict(os.environ, EXPLAINER_NARR=narr_path, EXPLAINER_CUES=cue_dir)
            run([sys.executable, "-m", "manim", "-q", quality, "--fps", str(FPS), "-r", "1920,1080",
                 "--format=mp4", "--media_dir", media, os.path.join(HERE, "scenes.py"), s["cls"]],
                env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            src = os.path.join(media, "videos", "scenes", "1080p30", f"{s['cls']}.mp4")
            os.replace(src, mp4)
            s["duration"] = duration(mp4)
            cf = os.path.join(cue_dir, f"{s['key']}.json")
            s["cues"] = json.load(open(cf)) if os.path.exists(cf) else {}
        else:
            spec = dict(s["spec"])
            if s["kind"] == "triangle":
                tl = os.path.join(out_dir, f"{name}.timeline.json")
                sp = os.path.join(out_dir, f"{name}.spec.json")
                json.dump(spec, open(sp, "w"))
                run([sys.executable, os.path.join(HERE, "frames.py"), sp, tl, "--timeline"])
                wav = os.path.join(out_dir, f"{name}.music.wav")
                run(["node", os.path.join(HERE, "demo_audio.mjs"), tl, wav], stderr=subprocess.DEVNULL)
                s["music"] = spec["audio"] = wav
            sp = os.path.join(out_dir, f"{name}.spec.json")
            json.dump(spec, open(sp, "w"))
            run([sys.executable, os.path.join(HERE, "frames.py"), sp, mp4])
        s["video"] = mp4
        print(f"  {name:14s} {s['duration']:6.1f}s", flush=True)
        return s

    with ThreadPoolExecutor(jobs) as ex:
        return list(ex.map(one, segs))


def moving_average(x, n):
    """A centred box filter by running sum (np.convolve with a 13k-tap kernel takes minutes)."""
    c = np.concatenate([[0.0], np.cumsum(x)])
    h = n // 2
    i = np.arange(len(x))
    lo, hi = np.clip(i - h, 0, len(x)), np.clip(i + h + 1, 0, len(x))
    return (c[hi] - c[lo]) / n


def mix(segs, vo, global_voice, out_wav, with_voice):
    total = sum(s["duration"] for s in segs)
    n = int((total + 1) * SR)
    voice, music, sfx = np.zeros(n), np.zeros((n, 2)), np.zeros(n)
    notes = {k: load(os.path.join(WORK, f"explainer-{k}.wav")) for k in ("flute", "violin", "harp", "morph")}
    placed = []

    def put(track, x, t, gain=1.0):
        i = int(round(t * SR))
        j = min(len(track), i + len(x))
        if j > i:
            track[i:j] += x[: j - i] * gain

    t0 = 0.0
    for s in segs:
        dur = s["duration"]
        for lt, k in s.get("voice", []):
            if with_voice:
                put(voice, vo.clip[k], t0 + lt)
            placed.append((t0 + lt, k))
        for name, ct in s.get("cues", {}).items():
            if name in notes:
                put(sfx, notes[name], t0 + ct)
        if s.get("music"):
            m = load(s["music"])
            m = m if m.ndim == 2 else np.stack([m, m], 1)
            m = m[: int(dur * SR)]
            if not s.get("music_raw"):
                m = m / (np.abs(m).max() + 1e-9) * MUSIC_PEAK
                fi, fo = int(0.05 * SR), int(0.45 * SR)
                m[:fi] *= np.linspace(0, 1, fi)[:, None]
                m[-fo:] *= np.linspace(1, 0, fo)[:, None]
            put(music, m, t0)
        t0 += dur
    for t, k in global_voice or []:
        if with_voice:
            put(voice, vo.clip[k], t)
        placed.append((t, k))

    voice *= VOICE_PEAK / (np.abs(voice).max() + 1e-9)
    sfx *= SFX_PEAK / (np.abs(sfx).max() + 1e-9)
    env = moving_average(np.abs(voice), int(0.3 * SR))
    ref = np.percentile(env[env > 1e-4], 60) if np.any(env > 1e-4) else 1.0
    duck = 1.0 - DUCK * np.clip(env / ref, 0, 1) if with_voice else np.ones(len(voice))
    out = music * duck[:, None] + (voice + sfx)[:, None]
    peak = np.abs(out).max()
    if peak > 0.97:
        out *= 0.97 / peak
    out = out[: int(total * SR)]
    sf.write(out_wav, out.astype(np.float32), SR)
    return sorted(placed), total


def loudnorm(wav, target=-16.0, tp=-1.5):
    """Two-pass EBU R128 normalisation, so both cuts play at the same level as other web video."""
    res = subprocess.run(["ffmpeg", "-hide_banner", "-i", wav, "-af",
                          f"loudnorm=I={target}:TP={tp}:LRA=11:print_format=json", "-f", "null", "-"],
                         capture_output=True, text=True, check=True)
    i = res.stderr.rindex("{")
    m = json.loads(res.stderr[i:res.stderr.index("}", i) + 1])
    out = wav.replace(".wav", ".norm.wav")
    run(["ffmpeg", "-loglevel", "error", "-y", "-i", wav, "-af",
         f"loudnorm=I={target}:TP={tp}:LRA=11:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
         f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true",
         "-ar", str(SR), out])
    return out


# ----------------------------------------------------------------------------- captions

def wrap(s, width=58):
    words, lines, cur = s.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    lines.append(cur)
    return lines


def rows2(s, width=58):
    """One row if it fits; two rows split at the space nearest the middle otherwise."""
    if len(s) <= width:
        return [s]
    sp = [i for i, ch in enumerate(s) if ch == " "]
    i = min(sp, key=lambda i: abs(i - len(s) / 2))
    return [s[:i], s[i + 1:]]


def chunks(cap, dur):
    """A long line becomes several captions of at most two rows: whole sentences where they
    fit, otherwise the words shared out evenly. Each is shown for its share of the words."""
    import re
    pieces = []
    for sent in re.split(r"(?<=[.?!])\s+", cap.strip()):
        if len(wrap(sent)) <= 2:
            pieces.append(sent.split())
        else:
            w = sent.split()
            k = -(-len(wrap(sent)) // 2)
            size = -(-len(w) // k)
            pieces += [w[i:i + size] for i in range(0, len(w), size)]
    groups = []
    for p in pieces:                       # pack short sentences together while they fit
        if groups and len(wrap(" ".join(groups[-1] + p))) <= 2:
            groups[-1] = groups[-1] + p
        else:
            groups.append(p)
    n = sum(len(g) for g in groups)
    t, out = 0.0, []
    for g in groups:
        d = dur * len(g) / n
        out.append((t, t + d, rows2(" ".join(g))))
        t += d
    return out


def clock(t, sep=","):
    return f"{int(t // 3600):02d}:{int(t // 60) % 60:02d}:{int(t % 60):02d}{sep}{int(round((t % 1) * 1000)) % 1000:03d}"


def write_captions(placed, vo, base):
    srt, ass = [], []
    n = 0
    for t, k in placed:
        for a, b, rows in chunks(vo.lines[k]["cap"], vo.dur[k]):
            n += 1
            srt += [str(n), f"{clock(t + a)} --> {clock(t + b + 0.15)}", *rows, ""]
            c = lambda x: f"{int(x // 3600)}:{int(x // 60) % 60:02d}:{x % 60:05.2f}"
            ass.append(f"Dialogue: 0,{c(t + a)},{c(t + b + 0.15)},Cap,,0,0,0,,{'\\N'.join(rows)}")
    open(base + ".srt", "w").write("\n".join(srt))
    head = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,EB Garamond,46,&H000D0D0D,&H000D0D0D,&H10FBFDFD,&H10FBFDFD,0,0,0,0,100,100,0,0,3,10,0,2,120,120,34,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    open(base + ".ass", "w").write(head + "\n".join(ass) + "\n")


# ----------------------------------------------------------------------------- main

def write_cue_sheet(placed, vo, segs, which, total):
    """The script with the time each line lands at, for recording a voiceover to the film."""
    path = os.path.join(HERE, "VOICEOVER.md")
    sections = {}
    if os.path.exists(path):
        txt = open(path).read()
        for part in txt.split("\n## ")[1:]:
            sections[part.split("\n", 1)[0]] = "## " + part.rstrip() + "\n"
    title = "Short (30 s)" if which == "short" else "Long (explained)"
    starts = np.cumsum([0] + [s["duration"] for s in segs])
    seg_at = lambda t: next((s.get("cls") or s["name"]) for s, a, b in zip(segs, starts, starts[1:]) if a <= t < b)
    rows = [f"## {title}", "",
            f"`babymaker-{'short' if which == 'short' else 'explained'}.mp4`, {total:.1f} s. The picture is timed for a "
            f"read at about {READ_WPS} words a second, so each line has room. Times are where each line starts.", "",
            "| time | on screen | line |", "|---|---|---|"]
    for t, k in placed:
        rows.append(f"| {int(t // 60)}:{t % 60:04.1f} | {seg_at(t)} | {vo.lines[k]['cap']} |")
    sections[title] = "\n".join(rows) + "\n"
    head = ("# Voiceover script\n\nWritten by build.py from video/script.py. The videos have no narration, only the "
            "music and the instrument sounds (plus silent copies), so this is the script to record over them. "
            "The captions in `out/*.srt` follow the same timing.\n\n")
    order = ["Short (30 s)", "Long (explained)"]
    open(path, "w").write(head + "\n".join(sections[k] for k in order if k in sections))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["short", "long"])
    ap.add_argument("--renders", default=os.path.expanduser("~/babymaker-video/renders"))
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--quality", default="h", choices=["l", "m", "h"])
    ap.add_argument("--voice", action="store_true", help="lay the TTS narration in (video/work/vo_*)")
    ap.add_argument("--burn", action="store_true", help="burn the captions in")
    ap.add_argument("--only", default=None, help="re-render only these segments (comma-separated names), keep the rest")
    a = ap.parse_args()

    vo = Voice(a.which, use_clips=a.voice)
    work = os.path.join(WORK, f"build_{a.which}")
    if a.which == "short":
        segs, gvoice = plan_short(vo, a.renders)
    else:
        import importlib.util
        spec = importlib.util.spec_from_file_location("scenes", os.path.join(HERE, "scenes.py"))
        # only the class list is needed here; manim itself renders in a subprocess
        scenes = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(scenes)
        segs, gvoice = plan_long(vo, a.renders, scenes)

    print(f"rendering {len(segs)} segments…", flush=True)
    segs = render_segments(segs, work, a.jobs, a.quality, a.only.split(",") if a.only else None)

    print("mixing…", flush=True)
    wav = os.path.join(work, "mix.wav")
    placed, total = mix(segs, vo, gvoice, wav, a.voice)
    wav = loudnorm(wav)

    os.makedirs(a.out, exist_ok=True)
    name = "babymaker-short" if a.which == "short" else "babymaker-explained"
    base = os.path.join(a.out, name)
    write_captions(placed, vo, base)
    write_cue_sheet(sorted(placed), vo, segs, a.which, total)

    listing = os.path.join(work, "parts.txt")
    open(listing, "w").write("".join(f"file '{s['video']}'\n" for s in segs))
    vf = ["fps=30", "format=yuv420p"]
    if a.burn:
        vf.insert(0, f"subtitles={base}.ass")
    silent = os.path.join(work, "picture.mp4")
    run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", listing,
         "-vf", ",".join(vf), "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-an",
         "-movflags", "+faststart", "-t", f"{total:.3f}", silent])
    run(["ffmpeg", "-loglevel", "error", "-y", "-i", silent, "-i", wav, "-c:v", "copy",
         "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-t", f"{total:.3f}", base + ".mp4"])
    os.replace(silent, base + "-silent.mp4")
    run(["ffmpeg", "-loglevel", "error", "-y", "-i", wav, "-c:a", "pcm_s16le", base + "-music.wav"])
    run(["ffmpeg", "-loglevel", "error", "-y", "-ss", "3", "-i", base + ".mp4", "-frames:v", "1", base + "-poster.png"])
    for f in (base + ".ass",):
        if os.path.exists(f) and not a.burn:
            os.remove(f)
    print(f"\nwrote {base}.mp4  {duration(base + '.mp4'):.1f}s  (+ -silent.mp4, -music.wav, .srt, VOICEOVER.md)")
    t = 0.0
    for s in segs:
        print(f"  {t:6.1f}  {(s.get('cls') or s['name']):14s} {s['duration']:5.1f}s")
        t += s["duration"]


if __name__ == "__main__":
    main()
