"""Manim scenes for the babymaker explainer.

Every curve, matrix and number on screen is read from video/work/explainer.json,
which video/explainer_data.py writes by running the real algorithm on the demo's
three instruments.

Timing comes from video/build.py: it synthesises the narration first, writes the
length of each line to a JSON, and each scene waits out its own lines with
`self.sync(i)`.  So the picture never runs ahead of the voice, and the voice never
waits on the picture.
"""
import json
import os

import numpy as np
from manim import (
    DOWN, DL, LEFT, ORIGIN, RIGHT, UP, UR,
    Arc, Arrow, Circle, Create, CurvedArrow, Dot, FadeIn, FadeOut, Group, ImageMobject, Line, Polygon,
    Rectangle, RoundedRectangle, Scene, Text, Transform, VGroup, VMobject, Write, rate_functions,
)

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "work", "explainer.json")))
NARR = json.load(open(os.environ["EXPLAINER_NARR"])) if os.environ.get("EXPLAINER_NARR") else {}

# the site's tokens, as manim needs them
PAPER, INK, SOFT, FAINT, RULE, ACCENT = "#FDFDFB", "#0D0D0D", "#4A4A48", "#8C8C8A", "#DCDCDA", "#B73E6A"
SERIF = "EB Garamond"
TONE = {"flute": ACCENT, "violin": INK, "harp": "#9A9A97"}

NAMES = D["names"]
HZ = np.array(D["hz"])
SRC = {s["name"]: s for s in D["sources"]}
TOP = 3.05          # content stays below the chapter head


# ----------------------------------------------------------------- text helpers

def title(s, size=34):
    return Text(s, font=SERIF, color=INK, font_size=size)


def note(s, size=21, color=SOFT):
    return Text(s, font=SERIF, color=color, font_size=size)


def caption(label, sub, size=22):
    """A panel's name over its one-line gloss."""
    return VGroup(note(label, size, INK), note(sub, size - 4, FAINT)) \
        .arrange(DOWN, buff=0.12, aligned_edge=LEFT)


# ----------------------------------------------------------------- plotting

class Box:
    """A rectangle of screen space with data coordinates mapped onto it.

    `frame()` is built once and kept: a fresh copy each call would leave the
    axes on screen when the caller fades out what it thinks is the frame.
    """

    def __init__(self, w, h, xr, yr, center=ORIGIN):
        self.w, self.h, self.xr, self.yr = w, h, xr, yr
        self.c = np.asarray(center, float)
        self._frame = None

    def pt(self, x, y):
        (x0, x1), (y0, y1) = self.xr, self.yr
        return self.c + np.array([(x - x0) / (x1 - x0) - 0.5, (y - y0) / (y1 - y0) - 0.5, 0]) * \
            np.array([self.w, self.h, 0])

    def curve(self, xs, ys, color=INK, sw=2.0, opacity=1.0):
        m = VMobject(stroke_color=color, stroke_width=sw, stroke_opacity=opacity)
        m.set_points_as_corners([self.pt(x, y) for x, y in zip(xs, ys)])
        return m

    def frame(self, xlabel=None, ylabel=None):
        if self._frame is not None:
            return self._frame
        (x0, x1), (y0, y1) = self.xr, self.yr
        g = VGroup(
            Line(self.pt(x0, y0), self.pt(x1, y0), stroke_color=RULE, stroke_width=1.4),
            Line(self.pt(x0, y0), self.pt(x0, y1), stroke_color=RULE, stroke_width=1.4),
        )
        if xlabel:
            g.add(note(xlabel, 17, FAINT).next_to(self.pt((x0 + x1) / 2, y0), DOWN, buff=0.16))
        if ylabel:
            g.add(note(ylabel, 17, FAINT).rotate(np.pi / 2)
                  .next_to(self.pt(x0, (y0 + y1) / 2), LEFT, buff=0.16))
        self._frame = g
        return g

    def above_left(self, mob, buff=0.22):
        """Put a caption just above the box's top-left corner."""
        return mob.next_to(self.pt(self.xr[0], self.yr[1]), UP, buff=buff, aligned_edge=LEFT)


def wave_curve(box, vals, color, sw=1.1):
    """A decimated waveform drawn across the whole box, scaled to fill it."""
    v = np.asarray(vals, float)
    xs = np.linspace(box.xr[0], box.xr[1], len(v))
    return box.curve(xs, v / (np.abs(v).max() or 1.0) * box.yr[1], color, sw)


def level_curve(box, db, color, sw=2.2, floor=60, opacity=1.0):
    """One source's level, in dB below its own peak, against seconds."""
    y = np.asarray(db, float)
    y = np.maximum(y - y.max(), -floor)
    return box.curve(np.arange(len(y)) * D["hopS"], y, color, sw, opacity)


# ----------------------------------------------------------------- scene base

class Chapter(Scene):
    """A section of the film, kept in step with its own narration."""
    KEY = ""

    def setup(self):
        self.camera.background_color = PAPER
        self.ends = NARR.get(self.KEY, {}).get("ends", [])
        self.cues = {}

    @property
    def t(self):
        """Seconds of finished film.  `Scene.wait` is itself a `play`, so
        counting run times by hand would count every wait twice; the
        renderer's own clock is the one that matches the output."""
        return self.renderer.time

    def cue(self, name):
        """Mark where a sound belongs; the driver reads these back to place it."""
        self.cues[name] = round(self.t, 3)
        d = os.environ.get("EXPLAINER_CUES")
        if d:
            json.dump(self.cues, open(os.path.join(d, f"{self.KEY}.json"), "w"))

    def sync(self, i, tail=0.0):
        """Hold until narration line `i` has finished (plus `tail` seconds)."""
        if i < len(self.ends):
            left = self.ends[i] + tail - self.t
            if left > 0.02:
                self.wait(left)
        elif tail:
            self.wait(tail)

    def chapter_head(self, n, name):
        head = VGroup(note(n, 19, FAINT), title(name, 30)).arrange(RIGHT, buff=0.42) if n \
            else VGroup(title(name, 30))
        head.to_corner(UR).shift(LEFT * 0.2 + DOWN * 0.1)
        self.add(head)
        return head

    def clear_out(self, *mobs, run_time=0.7):
        # Group, not VGroup: the DTW cost matrix is an ImageMobject
        self.play(FadeOut(Group(*[m for m in mobs if m is not None])), run_time=run_time)


# ----------------------------------------------------------------- 1. the gap

class S1Gap(Chapter):
    KEY = "gap"

    def construct(self):
        head = self.chapter_head("i", "the gap")

        # the three real recordings, one at a time, each with its note sounding
        rows = VGroup()
        for k, n in enumerate(NAMES):
            b = Box(8.8, 1.4, (0, 1), (-1, 1), UP * (1.75 - k * 1.75) + RIGHT * 0.5)
            row = VGroup(wave_curve(b, SRC[n]["wave"], TONE[n]),
                         note(n, 23, TONE[n]).next_to(b.pt(0, 0), LEFT, buff=0.45))
            rows.add(row)
            self.cue(n)
            self.play(Create(row), run_time=1.35)
        self.sync(0, 0.3)
        self.play(FadeOut(rows), run_time=0.6)

        # their levels laid over one another: three notes of three lengths
        lv = Box(8.8, 3.6, (0, 1.5), (-60, 0), DOWN * 0.5 + RIGHT * 0.3)
        lines = VGroup(*[level_curve(lv, SRC[n]["levelDb"], TONE[n]) for n in NAMES])
        labs = VGroup(*[note(f"{n}  {len(SRC[n]['levelDb']) * D['hopS']:.2f}s", 20, TONE[n])
                        for n in NAMES]).arrange(RIGHT, buff=0.8)
        lv.above_left(labs, 0.3)
        self.play(FadeIn(lv.frame("seconds", "level, dB")), run_time=0.5)
        self.play(*[Create(l) for l in lines], FadeIn(labs), run_time=1.8)
        self.sync(1, 0.4)

        cap = note("three attacks, three decays, no one note", 25, SOFT)
        cap.next_to(lv.pt(0.75, -60), DOWN, buff=0.75)
        self.play(FadeIn(cap), run_time=0.7)
        self.sync(2, 0.3)
        self.clear_out(lines, labs, cap, head, lv.frame())


# ----------------------------------------------------------------- 2. the transform

class S2Transform(Chapter):
    KEY = "transform"

    def construct(self):
        head = self.chapter_head("ii", "the transform")
        M, hop, sr = D["M"], D["hop"], D["sr"]

        b = Box(9.6, 1.9, (0, 1), (-1, 1), UP * 2.0)
        w = wave_curve(b, SRC["flute"]["wave"], TONE["flute"], 1.1)
        self.add(b.frame(), w)

        # a window sliding along it, one frame wide
        frac = 2 * M / sr / SRC["flute"]["durS"]
        win = Polygon(b.pt(0, -1), b.pt(frac, -1), b.pt(frac, 1), b.pt(0, 1),
                      stroke_color=ACCENT, stroke_width=1.2, stroke_opacity=0.5,
                      fill_color=ACCENT, fill_opacity=0.09)
        wlab = note(f"window {2 * M} samples · hop {hop} · 50% overlap", 20, FAINT)
        wlab.next_to(b.pt(0.5, -1), DOWN, buff=0.35)
        self.play(FadeIn(win), FadeIn(wlab), run_time=0.5)
        self.play(win.animate.shift(RIGHT * (1 - frac) * b.w), run_time=2.4,
                  rate_func=rate_functions.linear)
        self.sync(0, 0.2)

        f = VGroup(
            note("X[k]  =  Σ  w[n] · x[n] · cos( π/M · (n + ½ + M/2)(k + ½) )", 27, INK),
            note("Y[k]  =  Σ  w[n] · x[n] · sin( π/M · (n + ½ + M/2)(k + ½) )", 27, SOFT),
        ).arrange(DOWN, buff=0.32, aligned_edge=LEFT).move_to(DOWN * 0.55)
        legend = note("magnitude  √(X² + Y²)          phase  atan2(Y, X)", 22, SOFT)
        legend.next_to(f, DOWN, buff=0.6)
        self.play(Write(f[0]), run_time=1.3)
        self.play(FadeIn(f[1]), run_time=0.7)
        self.play(FadeIn(legend), run_time=0.6)
        self.sync(1, 0.2)

        stats = note(f"M = {M} bins     hop = {hop}     E5, {D['f0']:.0f} Hz", 22, FAINT)
        stats.next_to(legend, DOWN, buff=0.55)
        self.play(FadeIn(stats), run_time=0.6)
        self.sync(2, 0.3)
        self.clear_out(w, win, wlab, f, legend, stats, head, b.frame())


# ----------------------------------------------------------------- 3. one frame

class S3Split(Chapter):
    KEY = "split"

    def construct(self):
        head = self.chapter_head("iii", "one frame")
        s = SRC["flute"]
        lev = np.array(s["levelDb"])
        ti = int(round(D["frameT"] / D["hopS"]))

        top = Box(9.4, 3.1, (0, HZ[-1]), (-7.4, 3.9), UP * 1.35 + RIGHT * 0.5)
        raw = top.curve(HZ, s["logShape"], INK, 1.4)
        self.play(FadeIn(top.frame("frequency, Hz", "log magnitude")), run_time=0.5)
        self.play(Create(raw), run_time=1.4)
        self.sync(0, 0.2)

        # level: one number for the whole frame
        span = len(lev) * D["hopS"]
        lb = Box(4.6, 1.6, (0, span), (-60, 0), DOWN * 2.3 + LEFT * 2.3)
        lcurve = level_curve(lb, lev, SOFT, 2.0)
        marker = Dot(lb.pt(ti * D["hopS"], lev[ti] - lev.max()), radius=0.07, color=ACCENT)
        llab = caption("level", f"one number: {lev[ti] - lev.max():.1f} dB below peak")
        llab.next_to(lb.pt(span, 0), RIGHT, buff=0.7)
        self.play(FadeIn(lb.frame("seconds")), Create(lcurve), run_time=1.0)
        self.play(FadeIn(marker), FadeIn(llab), run_time=0.5)
        self.sync(1, 0.3)
        self.play(FadeOut(VGroup(lb.frame(), lcurve, marker, llab)), run_time=0.5)

        # envelope: the iterated cepstral smoothing climbing over the peaks
        it = [np.array(v) for v in D["envIters"]]
        cur = top.curve(HZ, it[0], ACCENT, 2.6)
        step = note(f"one cepstral smoothing, q = {D['q']}", 21, FAINT)
        step.next_to(top.pt(HZ[-1] * 0.5, 3.9), UP, buff=0.28)
        self.play(Create(cur), FadeIn(step), run_time=1.0)
        self.sync(2, 0.1)

        for k, e in zip(D["envIterN"][1:], it[1:]):
            self.play(Transform(cur, top.curve(HZ, e, ACCENT, 2.6)),
                      Transform(step, note(f"iteration {k}", 21, FAINT).move_to(step)),
                      run_time=0.6)
        self.play(Transform(cur, top.curve(HZ, s["env"], ACCENT, 2.6)),
                  Transform(step, note("true envelope", 21, ACCENT).move_to(step)), run_time=0.8)
        self.sync(3, 0.3)

        # fine structure: what the envelope leaves behind
        fb = Box(9.4, 1.9, (0, HZ[-1]), (-7.6, 2.6), DOWN * 2.45 + RIGHT * 0.5)
        fcurve = fb.curve(HZ, s["fine"], INK, 1.3)
        flab = note("fine structure  =  spectrum − envelope", 22, SOFT)
        fb.above_left(flab, 0.22)
        self.play(FadeIn(fb.frame("frequency, Hz")), run_time=0.4)
        self.play(Create(fcurve), FadeIn(flab), run_time=1.3)
        self.sync(4, 0.4)
        self.clear_out(raw, cur, step, fcurve, flab, head, top.frame(), fb.frame())


# ----------------------------------------------------------------- 4. alignment

class S4Align(Chapter):
    KEY = "align"

    def construct(self):
        head = self.chapter_head("iv", "alignment")

        lv = Box(8.6, 3.4, (0, 1.5), (-60, 0), UP * 0.5 + RIGHT * 0.3)
        lines = VGroup(*[level_curve(lv, SRC[n]["levelDb"], TONE[n], 2.0) for n in NAMES])
        labs = VGroup(*[note(n, 20, TONE[n]) for n in NAMES]).arrange(RIGHT, buff=0.9)
        lv.above_left(labs, 0.28)
        self.play(FadeIn(lv.frame("seconds", "level")), Create(lines), FadeIn(labs), run_time=1.4)
        self.sync(0, 0.2)

        feat = note("features per frame:  level  +  the first 12 cepstral orders", 22, SOFT)
        feat.next_to(lv.pt(0.75, -60), DOWN, buff=0.75)
        self.play(FadeIn(feat), run_time=0.6)
        self.sync(1, 0.2)
        self.play(FadeOut(VGroup(lines, labs, feat, lv.frame())), run_time=0.6)

        # the real cost matrix, and the real path through it
        cost = np.array(D["dtw"]["cost"])
        n_a, n_b = cost.shape
        shade = np.clip(cost / np.percentile(cost, 92), 0, 1) ** 0.7
        img = ImageMobject((shade * 232 + 18).astype(np.uint8))
        img.height = 4.9
        img.width = 4.9 * n_b / n_a
        img.move_to(LEFT * 3.5 + DOWN * 0.35)
        dl = img.get_corner(DL)

        def cell(i, j):
            return dl + RIGHT * (j / (n_b - 1)) * img.width + UP * (i / (n_a - 1)) * img.height

        ax = VGroup(
            note(D["dtw"]["b"], 21, TONE[D["dtw"]["b"]]).next_to(img, DOWN, buff=0.24),
            note(D["dtw"]["a"], 21, TONE[D["dtw"]["a"]]).rotate(np.pi / 2).next_to(img, LEFT, buff=0.24),
        )
        clab = note("DTW cost", 21, SOFT).next_to(img, UP, buff=0.3)
        self.play(FadeIn(img), FadeIn(ax), FadeIn(clab), run_time=0.9)

        path = np.array(D["dtw"]["path"])
        pline = VMobject(stroke_color=ACCENT, stroke_width=3.4)
        pline.set_points_as_corners([cell(i, j) for i, j in path])
        self.play(Create(pline), run_time=1.8)
        self.sync(2, 0.2)

        # the path, read as a map
        mp = np.array(D["dtw"]["map1"])
        mb = Box(4.2, 4.2, (0, len(mp) - 1), (0, mp.max()), RIGHT * 3.9 + DOWN * 0.35)
        mdiag = Line(mb.pt(0, 0), mb.pt(len(mp) - 1, mp.max()), stroke_color=INK,
                     stroke_width=1.2, stroke_opacity=0.28)
        mline = mb.curve(np.arange(len(mp)), mp, ACCENT, 2.6)
        mlab = note("flute frame  →  violin frame", 21, SOFT).next_to(mb.pt(len(mp) / 2, mp.max()), UP, buff=0.3)
        self.play(FadeIn(mb.frame()), Create(mdiag), run_time=0.5)
        self.play(Create(mline), FadeIn(mlab), run_time=1.3)
        self.sync(3, 0.5)
        self.clear_out(ax, clab, pline, mline, mdiag, mlab, head, mb.frame(), img)


# ----------------------------------------------------------------- 5. the average

class S5Average(Chapter):
    KEY = "average"

    def construct(self):
        head = self.chapter_head("v", "the average")
        w = D["weights"]
        M = D["morph"]

        # the weights, as the demo's own triangle
        tri = Polygon(UP * 1.25, DOWN * 1.0 + LEFT * 1.45, DOWN * 1.0 + RIGHT * 1.45,
                      stroke_color=RULE, stroke_width=1.6, fill_opacity=0)
        tri.move_to(LEFT * 4.8 + DOWN * 0.2)
        verts = list(tri.get_vertices()[:3])
        dots = VGroup(*[Dot(v, radius=0.07, color=TONE[n]) for v, n in zip(verts, NAMES)])
        vlab = VGroup(*[note(f"{n} {int(round(wi * 100))}%", 19, TONE[n])
                        .next_to(v, UP if k == 0 else DOWN, buff=0.22)
                        for k, (v, n, wi) in enumerate(zip(verts, NAMES, w))])
        pos = sum(np.array(v) * wi for v, wi in zip(verts, w))
        spokes = VGroup(*[Line(pos, v, stroke_color=TONE[n], stroke_width=1.2,
                               stroke_opacity=0.2 + 0.7 * wi)
                          for v, n, wi in zip(verts, NAMES, w)])
        play = Dot(pos, radius=0.1, color=ACCENT)
        self.play(Create(tri), FadeIn(dots), FadeIn(vlab), run_time=0.9)
        self.play(FadeIn(spokes), FadeIn(play), run_time=0.6)
        self.sync(0, 0.2)

        # three rows, one per domain, stacked down the right
        def row(y, h, xr, yr):
            return Box(7.4, h, xr, yr, RIGHT * 2.9 + UP * y)

        # 1. the timeline
        b1 = row(2.05, 1.0, (0, 1), (0, 1))
        ends, mixlen = M["posEnds"], M["J"]
        span = max(max(ends), mixlen)
        bars = VGroup(*[Line(b1.pt(0, 0.85 - k * 0.22), b1.pt(ends[k] / span, 0.85 - k * 0.22),
                            stroke_color=TONE[n], stroke_width=4, stroke_opacity=0.5)
                        for k, n in enumerate(NAMES)])
        bars.add(Line(b1.pt(0, 0.1), b1.pt(mixlen / span, 0.1), stroke_color=ACCENT, stroke_width=6))
        c1 = b1.above_left(caption("timeline", "aligned frames, w-averaged"), 0.2)
        self.play(FadeIn(c1), Create(bars), run_time=1.2)
        self.sync(1, 0.2)

        # 2. level, in dB
        lsrc, lmix = np.array(M["levelSrc"]), np.array(M["levelMix"])
        peak = lsrc.max()
        b2 = row(-0.15, 1.3, (0, lsrc.shape[1]), (peak - 55, peak))
        g2 = VGroup(*[b2.curve(np.arange(lsrc.shape[1]), np.maximum(lsrc[i], peak - 55),
                               TONE[n], 1.5, 0.4) for i, n in enumerate(NAMES)])
        g2.add(b2.curve(np.arange(len(lmix)), np.maximum(lmix, peak - 55), ACCENT, 2.8))
        c2 = b2.above_left(caption("level", "averaged in dB"), 0.2)
        self.play(FadeIn(c2), Create(g2), run_time=1.2)
        self.sync(2, 0.2)

        # 3. envelope, log against linear
        esrc = np.array(M["envSrc"])
        b3 = row(-2.55, 1.5, (0, HZ[-1]), (-4.6, 4.2))
        g3 = VGroup(*[b3.curve(HZ, esrc[i], TONE[n], 1.3, 0.35) for i, n in enumerate(NAMES)])
        g3.add(b3.curve(HZ, M["envMix"], ACCENT, 2.4))
        c3 = b3.above_left(caption("envelope", "averaged in the log domain"), 0.2)
        self.play(FadeIn(c3), Create(g3), run_time=1.2)
        xf = b3.curve(HZ, M["crossfadeEnv"], INK, 1.8, 0.5)
        xlab = note("amplitude average", 18, FAINT)
        xlab.next_to(b3.pt(HZ[-1], 4.2), UP, buff=0.2).align_to(b3.pt(HZ[-1], 0), RIGHT)
        self.play(Create(xf), FadeIn(xlab), run_time=1.0)
        self.sync(3, 0.3)

        # 4. fine structure, on its own
        self.play(FadeOut(VGroup(c1, bars, c2, g2, c3, g3, xf, xlab)), run_time=0.6)
        b4 = Box(7.4, 3.4, (0, HZ[-1]), (-8.6, 2.6), RIGHT * 2.9 + DOWN * 0.6)
        fsrc = np.array(M["fineSrc"])
        g4 = VGroup(*[b4.curve(HZ, fsrc[i], TONE[n], 1.1, 0.32) for i, n in enumerate(NAMES)])
        g4.add(b4.curve(HZ, M["fineMix"], ACCENT, 1.9))
        c4 = b4.above_left(caption("fine structure", "averaged as |a| to the 0.6"), 0.24)
        self.play(FadeIn(b4.frame("frequency, Hz")), FadeIn(c4), Create(g4), run_time=1.4)
        self.sync(4, 0.3)

        aside = VGroup(note("offline, the envelope's formant band is", 19, FAINT),
                       note("interpolated by optimal transport:", 19, FAINT),
                       note("peaks move rather than fade", 19, FAINT))
        aside.arrange(DOWN, buff=0.14).next_to(tri, DOWN, buff=0.95)
        self.play(FadeIn(aside), run_time=0.7)
        self.sync(5, 0.3)
        self.clear_out(tri, dots, vlab, play, spokes, g4, c4, aside, head, b4.frame())


# ----------------------------------------------------------------- 6. back to sound

class S6Synth(Chapter):
    KEY = "synth"

    def construct(self):
        head = self.chapter_head("vi", "back to sound")

        q = VGroup(note("a magnitude for every bin of every frame.", 28, INK),
                   note("no phase.", 28, ACCENT)).arrange(DOWN, buff=0.3)
        self.play(FadeIn(q[0]), run_time=0.8)
        self.play(FadeIn(q[1]), run_time=0.6)
        self.sync(0, 0.2)
        self.play(FadeOut(q), run_time=0.5)

        # WSOLA: blocks slid to where they best continue the one before
        b = Box(9.4, 1.8, (0, 1), (-1, 1), UP * 1.85)
        w = wave_curve(b, SRC["flute"]["wave"], TONE["flute"], 1.0)
        self.play(FadeIn(b.frame()), Create(w), run_time=0.9)
        blocks = VGroup(*[
            Polygon(b.pt(x, -1), b.pt(x + 0.15, -1), b.pt(x + 0.15, 1), b.pt(x, 1),
                    stroke_color=ACCENT, stroke_width=1.1, stroke_opacity=0.45,
                    fill_color=ACCENT, fill_opacity=0.07)
            for x in [0.06 + k * 0.16 for k in range(5)]])
        self.play(FadeIn(blocks), run_time=0.6)
        self.play(*[blk.animate.shift(RIGHT * s * b.w)
                    for blk, s in zip(blocks, [0.0, 0.02, -0.013, 0.028, -0.009])], run_time=1.2)
        wl = note("WSOLA — every block slid to the lag where it best continues the last", 21, SOFT)
        wl.next_to(b.pt(0.5, -1), DOWN, buff=0.4)
        self.play(FadeIn(wl), run_time=0.5)
        self.sync(1, 0.3)
        self.play(FadeOut(VGroup(w, blocks, wl, b.frame())), run_time=0.5)

        # the Griffin-Lim loop
        r = 1.5
        c = DOWN * 0.25
        ring = Circle(radius=r, stroke_color=RULE, stroke_width=1.4).move_to(c)
        steps = ["coherent mix\n→ take its phase", "morphed magnitudes\n→ onto that phase",
                 "inverse MCLT\n→ audio", "MCLT again\n→ keep the phase"]
        labs, arcs = VGroup(), VGroup()
        for k, s in enumerate(steps):
            a = np.pi / 2 - k * np.pi / 2
            labs.add(note(s, 21, INK if k % 2 == 0 else SOFT)
                     .move_to(c + np.array([np.cos(a), np.sin(a), 0]) * (r + 1.35)))
            arc = Arc(radius=r, start_angle=a - 0.28, angle=-(np.pi / 2 - 0.56),
                      stroke_color=ACCENT, stroke_width=2.6, arc_center=c)
            arcs.add(arc.add_tip(tip_length=0.2, tip_width=0.16))
        gl = note("×2", 24, ACCENT).move_to(c)
        self.play(Create(ring), FadeIn(labs), run_time=1.1)
        self.play(*[Create(a) for a in arcs], FadeIn(gl), run_time=1.1)
        self.sync(2, 0.4)
        self.clear_out(ring, labs, arcs, gl, head)


# ----------------------------------------------------------------- 7. the result

class S7Result(Chapter):
    KEY = "result"

    def construct(self):
        head = self.chapter_head("vii", "the result")

        rows = VGroup()
        for k, n in enumerate(NAMES):
            b = Box(6.4, 0.85, (0, 1), (-1, 1), UP * (2.7 - k * 1.0) + RIGHT * 0.4)
            rows.add(VGroup(wave_curve(b, SRC[n]["wave"], TONE[n], 0.9),
                            note(n, 19, TONE[n]).next_to(b.pt(0, 0), LEFT, buff=0.4)))
        self.play(FadeIn(rows), run_time=0.9)

        arrow = Line(UP * 0.35, DOWN * 0.5, stroke_color=RULE, stroke_width=1.6).add_tip(tip_length=0.18)
        ob = Box(7.6, 1.9, (0, 1), (-1, 1), DOWN * 1.9)
        olab = note("45% flute · 35% violin · 20% harp", 23, ACCENT)
        olab.next_to(ob.pt(0.5, -1), DOWN, buff=0.4)
        self.play(FadeIn(arrow), run_time=0.4)
        self.cue("morph")
        self.play(Create(wave_curve(ob, D["morph"]["wave"], ACCENT, 1.2)),
                  FadeIn(olab), run_time=1.8)
        self.sync(0, 0.8)

        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)
        end = VGroup(note("one attack, one decay, one timbre", 32, INK),
                     note("in a web worker, between key presses", 23, SOFT))
        end.arrange(DOWN, buff=0.32)
        self.play(FadeIn(end, shift=UP * 0.15), run_time=1.0)
        self.sync(1, 0.6)
        self.play(FadeOut(end), run_time=0.6)
        self.remove(head)


# ----------------------------------------------------------------- 0. why (the long video's opening)

SOFTFONTS = ["FluidR3", "MS GS Wavetable", "Yamaha FB-01"]


def chip(s, color=INK):
    t = note(s, 21, color)
    box = RoundedRectangle(corner_radius=0.16, width=t.width + 0.5, height=t.height + 0.34,
                           stroke_color=color, stroke_width=1.3)
    return VGroup(box, t.move_to(box))


def arrow(a, b):
    return Arrow(a, b, buff=0.12, stroke_color=FAINT, stroke_width=2.2, tip_length=0.18,
                 max_tip_length_to_length_ratio=0.3)


class S0Why(Chapter):
    KEY = "why"

    def construct(self):
        card = VGroup(title("making soundfont babies", 46),
                      note("morphing sampled instruments in the lapped cosine domain", 23, SOFT))
        card.arrange(DOWN, buff=0.32)
        self.play(FadeIn(card[0], shift=UP * 0.2), run_time=1.2)
        self.play(FadeIn(card[1]), run_time=0.8)
        self.wait(1.0)
        self.play(FadeOut(card), run_time=0.6)
        head = self.chapter_head("", "why")

        # audio -> model -> notes
        wb = Box(2.6, 1.0, (0, 1), (-1, 1), UP * 1.6 + LEFT * 4.6)
        wave = wave_curve(wb, SRC["violin"]["wave"], INK, 1.0)
        model = chip("transcription model", ACCENT).move_to(UP * 1.6)
        rng = np.random.default_rng(3)
        roll = VGroup(*[Rectangle(width=w, height=0.1, stroke_width=0, fill_color=INK, fill_opacity=0.75)
                        .move_to(UP * (1.6 + y) + RIGHT * (4.1 + x))
                        for x, y, w in zip(rng.uniform(-0.9, 0.9, 11), rng.choice(np.arange(-0.4, 0.45, 0.13), 11),
                                           rng.uniform(0.18, 0.55, 11))])
        a1, a2 = arrow(wb.pt(1, 0) + RIGHT * 0.1, model.get_left()), arrow(model.get_right(), RIGHT * 3.0 + UP * 1.6)
        self.play(Create(wave), run_time=0.8)
        self.play(FadeIn(a1), FadeIn(model), run_time=0.6)
        self.play(FadeIn(a2), FadeIn(roll, lag_ratio=0.1), run_time=0.9)
        self.sync(0, 0.2)

        # where the training audio comes from
        midi = chip("MIDI").move_to(DOWN * 0.9 + LEFT * 4.6)
        fonts = VGroup(*[chip(f, SOFT) for f in SOFTFONTS]).arrange(DOWN, buff=0.18).move_to(DOWN * 0.9)
        audio = chip("training audio").move_to(DOWN * 0.9 + RIGHT * 4.3)
        b1 = arrow(midi.get_right(), fonts.get_left())
        b2 = arrow(fonts.get_right(), audio.get_left())
        back = CurvedArrow(audio.get_top() + UP * 0.05, wb.pt(1, -1) + RIGHT * 1.3 + DOWN * 0.1, angle=0.5,
                           stroke_color=RULE, stroke_width=1.6, tip_length=0.16)
        self.play(FadeIn(midi), run_time=0.4)
        self.play(FadeIn(b1), FadeIn(fonts, lag_ratio=0.2), run_time=0.9)
        self.play(FadeIn(b2), FadeIn(audio), run_time=0.5)
        self.play(Create(back), run_time=0.7)
        self.sync(1, 0.2)

        # only so many
        top = VGroup(wave, a1, model, a2, roll, midi, b1, b2, audio, back)
        self.play(FadeOut(top), fonts.animate.move_to(UP * 0.6).scale(1.15), run_time=0.8)
        cnt = note("the same few pianos, in every render", 25, SOFT).next_to(fonts, DOWN, buff=0.7)
        self.play(FadeIn(cnt), run_time=0.6)
        self.sync(2, 0.2)

        # a triangle of them, and everything inside it
        tri_pts = [UP * 1.7 + LEFT * 1.4, DOWN * 1.9 + LEFT * 4.0, DOWN * 1.9 + RIGHT * 1.2]
        tri = Polygon(*tri_pts, stroke_color=RULE, stroke_width=1.6)
        targets = VGroup(*[chip(f, SOFT).scale(0.85).next_to(p, UP if k == 0 else DOWN, buff=0.18)
                           for k, (f, p) in enumerate(zip(SOFTFONTS, tri_pts))])
        self.play(FadeOut(cnt), Transform(fonts, targets), Create(tri), run_time=1.0)
        pts = rng.dirichlet([0.8, 0.8, 0.8], 160)
        babies = VGroup(*[Dot(sum(w * p for w, p in zip(ws, tri_pts)), radius=0.035, color=ACCENT,
                              fill_opacity=0.75) for ws in pts])
        self.play(FadeIn(babies, lag_ratio=0.02), run_time=2.0)
        lab = note("each point is a new instrument", 22, ACCENT).next_to(tri, RIGHT, buff=0.6)
        self.play(FadeIn(lab), run_time=0.5)
        self.sync(3, 0.6)
        self.clear_out(tri, fonts, babies, lab, head)


# ----------------------------------------------------------------- 8. what's next

class S8Outro(Chapter):
    KEY = "outro"

    def construct(self):
        head = self.chapter_head("", "next")
        # two formants on a log-frequency axis: the dB average vs the transport barycentre
        b = Box(8.6, 3.0, (np.log2(200), np.log2(3000)), (0, 1.1), UP * 0.4)
        f = np.linspace(np.log2(200), np.log2(3000), 400)
        bump = lambda c: np.exp(-0.5 * ((f - np.log2(c)) / 0.16) ** 2)
        a, c = bump(500), bump(1000)
        mid_db = 0.5 * a + 0.5 * c
        ot = bump(np.sqrt(500 * 1000))
        ga = b.curve(f, a, SOFT, 1.8, 0.7)
        gc = b.curve(f, c, "#9A9A97", 1.8, 0.7)
        gdb = b.curve(f, mid_db, INK, 2.2)
        got = b.curve(f, ot, ACCENT, 2.8)
        ticks = VGroup(*[note(t, 17, FAINT).next_to(b.pt(np.log2(v), 0), DOWN, buff=0.18)
                         for t, v in (("500 Hz", 500), ("700 Hz", 707), ("1 kHz", 1000))])
        xlab = note("frequency, log scale", 17, FAINT).next_to(ticks, DOWN, buff=0.2)
        self.play(FadeIn(b.frame()), FadeIn(ticks), FadeIn(xlab), Create(ga), Create(gc), run_time=1.0)
        l1 = note("dB average: two half-height bumps", 21, INK).next_to(b.pt(np.log2(200), 1.1), UP, buff=0.3, aligned_edge=LEFT)
        self.play(Create(gdb), FadeIn(l1), run_time=0.9)
        l2 = note("optimal transport: one peak, moved", 21, ACCENT).next_to(l1, DOWN, buff=0.12, aligned_edge=LEFT)
        self.play(Create(got), FadeIn(l2), run_time=0.9)
        self.sync(0, 0.3)
        self.clear_out(ga, gc, gdb, got, l1, l2, ticks, xlab, head, b.frame())

        end = VGroup(title("babymaker", 48),
                     note("github.com/hidude562/babymaker", 24, ACCENT),
                     note("python  ·  typescript port  ·  browser demo", 20, FAINT)).arrange(DOWN, buff=0.3)
        self.play(FadeIn(end, shift=UP * 0.15), run_time=1.0)
        self.sync(1, 2.2)
        self.play(FadeOut(end), run_time=0.8)


ORDER = [S0Why, S1Gap, S2Transform, S3Split, S4Align, S5Average, S6Synth, S7Result, S8Outro]

import sys  # noqa: E402
sys.path.insert(0, HERE)
from script import LONG  # noqa: E402

SCRIPT = {k: [l if isinstance(l, str) else l["cap"] for l in v] for k, v in LONG.items()}
