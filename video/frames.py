"""Frames for the demo segments, drawn with cairo and piped straight into ffmpeg.

Three kinds of segment, all in the site's look (paper, ink, one rose accent):

  triangle  the browser demo: three instruments on dots, the playhead dragged between
            them, weights by inverse distance exactly as the component computes them
  roll      a MIDI render through morphed soundfonts: the triangle of fonts on the left
            (the playhead at the weights in effect), a scrolling piano roll on the right
            with every note coloured by the weights it was played at
  card      a title or end card

    python3 video/frames.py spec.json out.mp4

The spec is written by video/build.py; see the draw_* functions for its fields.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys

import cairo
import numpy as np
import soundfile as sf

W, H = 1920, 1080
PAPER = (0xFD / 255, 0xFD / 255, 0xFB / 255)
INK = (0x0D / 255, 0x0D / 255, 0x0D / 255)
SOFT = (0x4A / 255, 0x4A / 255, 0x48 / 255)
FAINT = (0x8C / 255, 0x8C / 255, 0x8A / 255)
RULE = (0xDC / 255, 0xDC / 255, 0xDA / 255)
ACCENT = (0xB7 / 255, 0x3E / 255, 0x6A / 255)
# a roll's sources, in fixed order: the site's rose first, then the dataviz reference hues.
# Validated (validate_palette.js, light, on the paper): lightness, chroma and the normal-vision
# floor pass; the one CVD pair in the 6-8 band is legal because every source is labelled.
HEX = ["#B73E6A", "#2a78d6", "#c98500", "#008300", "#4a3aa7", "#eb6834", "#1baf7a", "#e34948"]
SLOT = [tuple(int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)) for h in HEX]
SERIF = "EB Garamond"
MONO = "DejaVu Sans Mono"


def ease(p):
    p = min(1.0, max(0.0, p))
    return p * p * (3 - 2 * p)


def mix_rgb(cols, ws):
    """Blend in linear light, so a 50/50 of two colours is not muddier than either."""
    lin = sum(w * np.asarray(c) ** 2.2 for c, w in zip(cols, ws))
    return tuple((lin / max(1e-9, sum(ws))) ** (1 / 2.2))


def text(ctx, s, x, y, size, color=INK, font=SERIF, align="left", alpha=1.0, weight=cairo.FONT_WEIGHT_NORMAL):
    ctx.select_font_face(font, cairo.FONT_SLANT_NORMAL, weight)
    ctx.set_font_size(size)
    ext = ctx.text_extents(s)
    if align == "center":
        x -= ext.x_advance / 2
    elif align == "right":
        x -= ext.x_advance
    ctx.set_source_rgba(*color, alpha)
    ctx.move_to(x, y)
    ctx.show_text(s)
    ctx.new_path()          # show_text leaves a current point; the next arc would join it
    return ext.x_advance


def levels(wav, fps, n):
    """Per-frame RMS of the music, for the playhead's pulse."""
    if not wav:
        return np.zeros(n)
    y, sr = sf.read(wav, always_2d=True)
    y = y.mean(1)
    hop = sr / fps
    out = np.zeros(n)
    for i in range(n):
        s = y[int(i * hop): int(i * hop + hop)]
        out[i] = float(np.sqrt(np.mean(s * s))) if len(s) else 0.0
    return out


# ----------------------------------------------------------------------------- triangle

class Keyframes:
    """[[t_arrive, [x, y], move_s]]: hold, then ease to the next point over move_s."""

    def __init__(self, keys):
        self.keys = [(float(t), np.asarray(p, float), float(m)) for t, p, m in keys]

    def at(self, t):
        pos = self.keys[0][1]
        for ta, p, m in self.keys[1:]:
            if t >= ta:
                pos = p
            elif t > ta - m:
                u = ease((t - (ta - m)) / m)
                return pos + (p - pos) * u
            else:
                break
        return pos


def weights_for(head, dots):
    """Inverse distance, exactly as the browser component does it."""
    raw = [1.0 / ((head[0] - d[0]) ** 2 + (head[1] - d[1]) ** 2 + 0.5) for d in dots]
    s = sum(raw)
    return [r / s for r in raw]


def draw_triangle(ctx, spec, t, level):
    dots = spec["dots"]
    labels = spec["labels"]
    head = spec["_head"].at(t)
    w = weights_for(head, dots)
    # the 100 x 75 stage, centred
    sc = spec.get("scale", 11.6)
    ox = (W - 100 * sc) / 2
    oy = spec.get("top", 60)
    P = lambda x, y: (ox + x * sc, oy + y * 0.75 * sc)
    hx, hy = P(*head)
    dash_off = -2.3 * sc * ((t / 1.4) % 1.0)
    for (dx, dy), wi in zip(dots, w):
        x, y = P(dx, dy)
        ctx.set_source_rgba(*INK, 0.08 + 0.92 * wi)
        ctx.set_line_width(0.35 * sc)
        ctx.set_dash([0.9 * sc, 1.4 * sc], dash_off)
        ctx.move_to(x, y)
        ctx.line_to(hx, hy)
        ctx.stroke()
    ctx.set_dash([])
    for (dx, dy), wi, lab in zip(dots, w, labels):
        x, y = P(dx, dy)
        ctx.new_sub_path()
        ctx.arc(x, y, 2.6 * sc, 0, 2 * math.pi)
        ctx.set_source_rgb(*PAPER)
        ctx.fill_preserve()
        ctx.set_source_rgb(*INK)
        ctx.set_line_width(0.45 * sc)
        ctx.stroke()
        vx, vy = x - hx, y - hy
        ln = math.hypot(vx, vy)
        if ln < 3 * sc:
            # the playhead is on this dot: put the label outside the triangle instead
            cxs, cys = P(sum(d[0] for d in dots) / len(dots), sum(d[1] for d in dots) / len(dots))
            vx, vy = x - cxs, y - cys
            ln = math.hypot(vx, vy)
        lx, ly = x + vx / ln * 7.5 * sc, y + vy / ln * 7.5 * sc
        text(ctx, lab, lx, ly - 0.6 * sc, 2.9 * sc, SOFT, align="center")
        text(ctx, f"{round(wi * 100)}%", lx, ly + 2.4 * sc, 2.1 * sc, SOFT, MONO, align="center")
    r = 2.2 + min(4.0, level * 22)
    ring = r + 1.6 + level * 30
    ctx.new_sub_path()
    ctx.arc(hx, hy, ring * sc, 0, 2 * math.pi)
    ctx.set_source_rgba(*ACCENT, 0.55)
    ctx.set_line_width(0.3 * sc)
    ctx.stroke()
    ctx.new_sub_path()
    ctx.arc(hx, hy, r * sc, 0, 2 * math.pi)
    ctx.set_source_rgb(*ACCENT)
    ctx.fill()
    return w


def triangle_timeline(spec):
    """What the audio renderer needs: the weights and the trio at every frame."""
    fps, dur = spec["fps"], spec["duration"]
    kf = Keyframes(spec["head"])
    frames = []
    for i in range(int(dur * fps) + 1):
        t = i / fps
        h = kf.at(t)
        frames.append({"t": round(t, 4), "weights": [round(x, 5) for x in weights_for(h, spec["dots"])],
                       "selection": spec["selection"]})
    return {"fps": fps, "duration": dur, "musicStart": spec.get("music_start", 0.0),
            "musicEnd": spec.get("music_end", dur), "frames": frames}


# ----------------------------------------------------------------------------- roll

class Path:
    """The arrange plan's weight keyframes, unquantised (for drawing)."""

    def __init__(self, keys):
        self.keys = [(float(t), dict(w)) for t, w in keys]

    def at(self, t):
        ks = self.keys
        if t <= ks[0][0]:
            return dict(ks[0][1])
        if t >= ks[-1][0]:
            return dict(ks[-1][1])
        i = max(j for j in range(len(ks)) if ks[j][0] <= t)
        (t0, a), (t1, b) = ks[i], ks[i + 1]
        u = (t - t0) / max(1e-9, t1 - t0)
        return {f: (1 - u) * a.get(f, 0) + u * b.get(f, 0) for f in set(a) | set(b)}


def trio_at(trios, t):
    for tr in trios:
        if tr["t0"] - 1e-6 <= t < tr["t1"]:
            return tr
    return trios[-1] if t >= trios[-1]["t0"] else trios[0]


def slot_weights(w, trio):
    ws = [max(0.0, w.get(f, 0.0)) for f in trio["fonts"]]
    s = sum(ws)
    return [x / s for x in ws] if s > 0 else [1 / len(ws)] * len(ws)


def polygon(n, cx=400, cy=590, R=235):
    """Where a set's sources sit: the old triangle for three, a regular n-gon otherwise."""
    if n == 3:
        return [(cx, cy - R), (cx - R * math.sin(math.pi / 3) * 1.08, cy + R * 0.62),
                (cx + R * math.sin(math.pi / 3) * 1.08, cy + R * 0.62)]
    r = 205
    return [(cx + r * math.cos(-math.pi / 2 + 2 * math.pi * k / n), cy - 15 + r * math.sin(-math.pi / 2 + 2 * math.pi * k / n))
            for k in range(n)]


def draw_roll(ctx, spec, t, level):
    """t is seconds into the segment; the music's own time is t + start."""
    start = spec["start"]
    tm = t + start
    path, trios, fonts = spec["_path"], spec["trios"], spec["fonts"]
    tr = trio_at(trios, tm)
    sw = slot_weights(path.at(tm), tr)

    # ---- header
    text(ctx, spec["title"], 90, 110, 46, INK)
    if spec.get("subtitle"):
        text(ctx, spec["subtitle"], 92, 152, 26, FAINT)
    # the trio's name, fading in when it changes
    if tr.get("name"):
        age = tm - tr["t0"]
        a = ease(age / 0.6) if age < 0.6 else 1.0
        text(ctx, tr["name"], 92, 200, 30, ACCENT, alpha=a)

    # ---- the polygon of fonts
    n = len(tr["fonts"])
    verts = polygon(n)
    cx = sum(v[0] for v in verts) / n
    cy = sum(v[1] for v in verts) / n
    ctx.set_source_rgb(*RULE)
    ctx.set_line_width(2.0)
    ctx.move_to(*verts[0])
    for v in verts[1:]:
        ctx.line_to(*v)
    ctx.close_path()
    ctx.stroke()
    hx = sum(w * v[0] for w, v in zip(sw, verts))
    hy = sum(w * v[1] for w, v in zip(sw, verts))
    for k, (v, wk) in enumerate(zip(verts, sw)):
        ctx.set_source_rgba(*SLOT[k], 0.12 + 0.88 * wk)
        ctx.set_line_width(3.2 if n <= 4 else 2.6)
        ctx.set_dash([7, 9], -16 * ((t / 1.4) % 1.0))
        ctx.move_to(*v)
        ctx.line_to(hx, hy)
        ctx.stroke()
    ctx.set_dash([])
    age = tm - tr["t0"]
    la = ease(age / 0.6) if (age < 0.6 and tr is not trios[0]) else 1.0
    for k, (v, f) in enumerate(zip(verts, tr["fonts"])):
        ctx.new_sub_path()
        ctx.arc(v[0], v[1], 17 if n <= 4 else 14, 0, 2 * math.pi)
        ctx.set_source_rgb(*PAPER)
        ctx.fill_preserve()
        ctx.set_source_rgb(*SLOT[k])
        ctx.set_line_width(4)
        ctx.stroke()
        pct = f"{round(sw[k] * 100)}%"
        if n == 3:
            lab = fonts[f]["label"]
            if k == 0:
                text(ctx, lab, v[0], v[1] - 66, 27, INK, align="center", alpha=la)
                text(ctx, pct, v[0], v[1] - 30, 21, SOFT, MONO, align="center")
            else:
                text(ctx, lab, v[0], v[1] + 62, 27, INK, align="center", alpha=la)
                text(ctx, pct, v[0], v[1] + 92, 21, SOFT, MONO, align="center")
        else:
            # labels pushed out along the spoke from the centre, away from the shape
            lab = fonts[f].get("short", fonts[f]["label"])
            ux, uy = v[0] - cx, v[1] - cy
            ul = math.hypot(ux, uy) or 1.0
            lx, ly = v[0] + ux / ul * 46, v[1] + uy / ul * 40
            al = "center" if abs(ux / ul) < 0.35 else ("left" if ux > 0 else "right")
            if al != "center":
                lx = v[0] + (26 if ux > 0 else -26)
                ly = v[1] + 7
            text(ctx, lab, lx, ly - (10 if al == "center" and uy < 0 else 0) + (14 if al == "center" and uy > 0 else 0),
                 23, INK, align=al, alpha=la)
            dy = 24
            text(ctx, pct, lx, ly + dy - (10 if al == "center" and uy < 0 else 0) + (14 if al == "center" and uy > 0 else 0),
                 18, SOFT, MONO, align=al, alpha=la)
    r = 12 + min(20, level * 110)
    ctx.new_sub_path()
    ctx.arc(hx, hy, r + 9 + level * 150, 0, 2 * math.pi)
    ctx.set_source_rgba(*ACCENT, 0.5)
    ctx.set_line_width(2)
    ctx.stroke()
    ctx.new_sub_path()
    ctx.arc(hx, hy, r, 0, 2 * math.pi)
    ctx.set_source_rgb(*ACCENT)
    ctx.fill()

    # ---- the piano roll
    x0, x1, y0, y1 = 790, 1840, 240, 880
    win = spec.get("window", 8.0)
    now_frac = 0.28
    tl = t - now_frac * win         # notes are timed from the start of the excerpt
    X = lambda s: x0 + (s - tl) / win * (x1 - x0)
    lo, hi = spec["_prange"]
    drum_h = 70 if spec["_has_drums"] else 0
    py1 = y1 - drum_h - (14 if drum_h else 0)
    rowh = (py1 - y0) / (hi - lo + 1)
    Y = lambda p: py1 - (p - lo + 1) * rowh
    ctx.set_source_rgb(*RULE)
    ctx.set_line_width(1.2)
    for p in range(lo, hi + 1):
        if p % 12 == 0:
            ctx.move_to(x0, Y(p) + rowh)
            ctx.line_to(x1, Y(p) + rowh)
    ctx.stroke()
    ctx.save()
    ctx.rectangle(x0, y0 - 4, x1 - x0, y1 - y0 + 8)
    ctx.clip()
    for nt in spec["_notes"]:
        if nt["end"] < tl - 0.5:
            continue
        if nt["t"] > tl + win:
            break
        ntr = trio_at(trios, nt["t"] + start)
        cw = slot_weights(nt["w"], ntr)
        if len(cw) > 3:
            # eight colours averaged by weight go grey; squaring lets the leading sources show
            cw = [x * x for x in cw]
        col = mix_rgb(SLOT[:len(ntr["fonts"])], cw)
        a0, a1 = X(nt["t"]), X(max(nt["end"], nt["t"] + 0.06))
        on = nt["t"] <= t <= nt["end"]
        past = nt["end"] < t
        alpha = (0.35 + 0.65 * nt["vel"] / 127) * (0.45 if past else 1.0)
        if nt["drum"]:
            yy = py1 + 14 + (nt["pitch"] - 35) / 47 * (drum_h - 8)
            ctx.new_sub_path()
            ctx.arc(a0, yy, 4.5 if not on else 6.5, 0, 2 * math.pi)
            ctx.set_source_rgba(*col, alpha)
            ctx.fill()
            continue
        yy = Y(nt["pitch"])
        hgt = max(3.0, rowh - 1.5)
        ctx.rectangle(a0, yy, max(3.0, a1 - a0), hgt)
        ctx.set_source_rgba(*col, alpha)
        ctx.fill()
        if on:
            ctx.rectangle(a0 - 2, yy - 2, max(3.0, a1 - a0) + 4, hgt + 4)
            ctx.set_source_rgba(*col, 0.35)
            ctx.set_line_width(2)
            ctx.stroke()
    ctx.restore()
    nx = X(t)
    ctx.set_source_rgba(*INK, 0.55)
    ctx.set_line_width(1.6)
    ctx.move_to(nx, y0 - 10)
    ctx.line_to(nx, y1 + 10)
    ctx.stroke()


def prep_roll(spec):
    meta = json.load(open(spec["meta"]))
    plan = json.load(open(spec["plan"]))
    drums = {int(k) for k, v in meta["tracks"].items() if v.get("drum")}
    notes = sorted(({**n, "drum": n["track"] in drums} for n in meta["notes"]), key=lambda n: n["t"])
    pitches = [n["pitch"] for n in notes if not n["drum"]]
    spec["_notes"] = notes
    spec["_prange"] = (min(pitches) - 1, max(pitches) + 1)
    spec["_has_drums"] = bool(drums)
    spec["_path"] = Path(plan["path"])
    spec["trios"] = plan["trios"]
    spec["fonts"] = plan["fonts"]
    spec["start"] = plan["start"]


# ----------------------------------------------------------------------------- cards

def draw_card(ctx, spec, t, level):
    dur = spec["duration"]
    a = min(ease(t / 0.5), ease((dur - t) / 0.45)) if spec.get("fade", True) else 1.0
    y = H / 2 - 20
    text(ctx, spec["title"], W / 2, y, spec.get("size", 92), INK, align="center", alpha=a)
    for k, (s, col) in enumerate(spec.get("lines", [])):
        text(ctx, s, W / 2, y + 78 + k * 50, 36 if k == 0 else 30,
             ACCENT if col == "accent" else FAINT, align="center", alpha=a * ease((t - 0.25) / 0.5))


# ----------------------------------------------------------------------------- driver

def render(spec, out):
    fps = spec.get("fps", 30)
    dur = spec["duration"]
    n = int(round(dur * fps))
    kind = spec["kind"]
    if kind == "triangle":
        spec["_head"] = Keyframes(spec["head"])
        draw = draw_triangle
    elif kind == "roll":
        prep_roll(spec)
        draw = draw_roll
    else:
        draw = draw_card
    lev = levels(spec.get("audio"), fps, n)
    lev = np.convolve(lev, np.ones(3) / 3, mode="same")
    surf = cairo.ImageSurface(cairo.FORMAT_RGB24, W, H)
    ctx = cairo.Context(surf)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgra",
                           "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "medium",
                           "-crf", "16", "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    for i in range(n):
        t = i / fps
        ctx.set_source_rgb(*PAPER)
        ctx.paint()
        draw(ctx, spec, t, float(lev[i]))
        fi, fo = spec.get("fade_in", 0), spec.get("fade_out", 0)
        veil = max(1 - ease(t / fi) if fi else 0, 1 - ease((dur - t) / fo) if fo else 0)
        if veil > 0:
            ctx.set_source_rgba(*PAPER, veil)
            ctx.paint()
        surf.flush()
        ff.stdin.write(bytes(surf.get_data()))
    ff.stdin.close()
    ff.wait()
    if ff.returncode:
        raise RuntimeError(f"ffmpeg failed on {out}")


if __name__ == "__main__":
    spec = json.load(open(sys.argv[1]))
    if len(sys.argv) > 3 and sys.argv[3] == "--timeline":
        json.dump(triangle_timeline(spec), open(sys.argv[2], "w"))
    else:
        render(spec, sys.argv[2])
