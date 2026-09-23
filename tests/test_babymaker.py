"""babymaker tests. Run: python3 -m pytest tests/test_babymaker.py -q  (or python3 tests/test_babymaker.py)"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from babymaker import Morpher, MorphConfig, Source, parse_source_spec  # noqa: E402
from babymaker.lapped import MDCT, true_envelope  # noqa: E402
from babymaker.morph import dtw_path, path_to_map, simplex_grid, smooth_map  # noqa: E402
from babymaker.sources import estimate_f0, midi_to_hz, prepare  # noqa: E402

SR = 22050
SF2_A = "/usr/share/sounds/sf2/FluidR3_GM.sf2"
SF2_B = "/usr/share/sounds/sf2/TimGM6mb.sf2"


def _tone(f0, dur, decay, harmonics, sr=SR, inharm=0.0, attack=0.005, seed=0):
    t = np.arange(int(dur * sr)) / sr
    x = np.zeros_like(t)
    for h, a in harmonics.items():
        f = f0 * h * np.sqrt(1 + inharm * h * h)
        x += a * np.sin(2 * np.pi * f * t + h)
    env = np.exp(-t / decay) * np.minimum(1.0, t / attack)
    x = x * env + 1e-4 * np.random.default_rng(seed).standard_normal(len(t))
    return x


def test_mdct_reconstruction():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(7001)
    for M, hop in ((256, 256), (256, 128), (256, 64), (1024, 256)):
        m = MDCT(M, hop)
        X, _ = m.analyze(x)
        assert np.abs(m.synthesize(X, len(x)) - x).max() < 1e-9


def test_mclt_instantaneous_frequency():
    sr, f = 8000, 1234.5
    m = MDCT(256, 64)
    x = np.sin(2 * np.pi * f * np.arange(8000) / sr)
    mag, ph = m.mclt(x)
    k = int(np.argmax(mag[40]))
    d = np.diff(ph[:, k])
    dev = (d[40] - m.bin_omega[k] * m.hop + np.pi) % (2 * np.pi) - np.pi
    assert abs((m.bin_omega[k] + dev / m.hop) * sr / (2 * np.pi) - f) < 0.05
    assert mag[30:60, k].std() / mag[30:60, k].mean() < 1e-3


def test_true_envelope_covers_peaks():
    sr, f0 = 8000, 200.0
    m = MDCT(256, 64)
    x = _tone(f0, 2.0, 10.0, {h: 1.0 / h for h in range(1, 15)}, sr=sr)
    mag, _ = m.mclt(x)
    lm = np.log(mag[60] + 1e-9)
    env = true_envelope(lm, int(0.45 * sr / f0), iters=40, tol_db=0.1)
    hb = [int(round(h * f0 / (sr / 2) * 256 - 0.5)) for h in range(2, 15)]
    peaks = [i - 1 + int(np.argmax(lm[i - 1:i + 2])) for i in hb]                 # harmonic bins
    assert max(lm[i] - env[i] for i in peaks) * 20 / np.log(10) < 4.0            # at most 4 dB above


def test_dtw_and_maps():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((30, 3))
    b = np.concatenate([a[:10], a[10:20], a[10:20], a[20:]])          # middle part twice as long
    g = path_to_map(dtw_path(a, b), 30, 40)
    assert g[0] == 0 and g[-1] == 39 and np.all(np.diff(g) >= 0)
    assert abs(g[25] - 35) <= 1
    gs = smooth_map(g, 2.0)
    assert gs[0] == 0 and abs(gs[-1] - 39) < 1e-9 and np.all(np.diff(gs) >= 0)


def test_simplex_grid():
    for n, steps in ((2, 4), (3, 3), (4, 2)):
        W = simplex_grid(n, steps)
        assert np.allclose(W.sum(1), 1) and W.min() >= 0
        from math import comb
        assert W.shape == (comb(steps + n - 1, n - 1), n)


def test_source_spec():
    d = parse_source_spec("~/x/font.sf2:8:24")
    assert d["kind"] == "sf2" and d["bank"] == 8 and d["preset"] == 24
    assert parse_source_spec("font.sf2:5")["preset"] == 5 and parse_source_spec("font.sf2")["preset"] == 0
    assert parse_source_spec("a.wav:C4")["note"] == 60 and parse_source_spec("a.wav:F#3")["note"] == 54
    assert parse_source_spec("a.wav:61.5")["note"] == 61.5 and parse_source_spec("a.wav")["note"] is None


def test_f0_estimate_inharmonic():
    f0 = midi_to_hz(60) * 2 ** (17 / 1200)                              # +17 cents
    x = _tone(f0, 1.5, 3.0, {h: 1.0 / h for h in range(1, 12)}, inharm=4e-4)   # piano-like stretch
    f = estimate_f0(x, SR, midi_to_hz(60))
    assert abs(1200 * np.log2(f / f0)) < 2.0                              # partial-1 based: not biased by stretch
    assert estimate_f0(np.random.default_rng(1).standard_normal(SR), SR, midi_to_hz(60)) is None


def test_prepare_aligns_pitch_and_level():
    x = _tone(midi_to_hz(58), 1.0, 2.0, {1: 1.0, 2: 0.5, 3: 0.3})
    x = np.concatenate([np.zeros(SR // 10), 0.05 * x])
    y, info = prepare(x, SR, 58, 60)
    f = estimate_f0(y, SR, midi_to_hz(60))
    assert abs(1200 * np.log2(f / midi_to_hz(60))) < 3.0
    assert np.abs(y[:5]).max() < 0.05
    assert info["trim_samples"][0] >= SR // 10 / info["speed_ratio"] - 100      # leading silence gone
    assert 0.9 < np.sqrt(np.convolve(y ** 2, np.ones(441) / 441, mode="same").max()) < 1.1


def _synthetic_sources():
    # A: bright, all harmonics, fast decay.  B: odd harmonics only, dark, slow decay, slow attack.
    a = _tone(midi_to_hz(60), 1.2, 0.35, {h: 1.0 / h for h in range(1, 20)}, seed=1)
    b = _tone(midi_to_hz(60), 2.4, 3.0, {h: 1.0 / h ** 2 for h in range(1, 20, 2)}, attack=0.3, seed=2)
    out = []
    for name, x in (("A", a), ("B", b)):
        y, info = prepare(x, SR, 60, 60)
        out.append(Source(name, y, SR, 60.0, info))
    return out


def test_morph_identity_and_midpoint():
    srcs = _synthetic_sources()
    mp = Morpher(srcs, MorphConfig(sr=SR))
    for i, s in enumerate(srcs):
        w = np.eye(2)[i]
        y = mp.render(w)
        n = min(len(y), len(s.audio))
        ref = s.audio[:n] * s.meta["peak_rms_in"]
        assert not np.isnan(y).any()
        assert abs(len(y) - len(s.audio)) < SR * 0.05
        assert np.corrcoef(y[:n], ref)[0, 1] > 0.85, f"identity render of {s.name} drifted"
    y, info = mp.render([0.5, 0.5], return_info=True)
    assert not np.isnan(y).any() and np.abs(y).max() > 1e-3
    da, db = srcs[0].duration, srcs[1].duration
    assert min(da, db) + 0.05 < info["duration_s"] < max(da, db) - 0.05
    f = estimate_f0(y, SR, midi_to_hz(60))
    assert f is not None and abs(1200 * np.log2(f / midi_to_hz(60))) < 5.0
    # decay: the midpoint's level 0.8 s in sits between the parents'
    def level_at(x, t):
        seg = x[int(t * SR):int(t * SR) + 2205]
        return 20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-9)
    la, lb, lm = (level_at(srcs[0].audio, 0.8) - level_at(srcs[0].audio, 0.1),
                  level_at(srcs[1].audio, 0.8) - level_at(srcs[1].audio, 0.1),
                  level_at(y, 0.8) - level_at(y, 0.1))
    assert min(la, lb) < lm < max(la, lb)
    # even harmonics: present in A, absent in B -> in between at the midpoint
    m = mp.mdct
    def h2_ratio(x):
        mag, _ = m.mclt(x)
        j = int((0.3 * SR + m.pad - m.M) / m.hop)
        f0 = midi_to_hz(60)
        def hb(h):
            lo = int(h * f0 * 0.97 / (SR / 2) * m.M)
            hi = int(h * f0 * 1.03 / (SR / 2) * m.M) + 1
            return mag[j, lo:hi].max()
        return 20 * np.log10(hb(2) / hb(1))
    ra, rb, rm = h2_ratio(srcs[0].audio), h2_ratio(srcs[1].audio), h2_ratio(y)
    assert rb + 3 < rm < ra - 3, (ra, rb, rm)


def test_torch_backend_matches_numpy():
    try:
        import torch  # noqa: F401
    except Exception:
        print("skipping torch test (torch not installed)")
        return
    srcs = _synthetic_sources()
    ref = Morpher(srcs, MorphConfig(sr=SR, device="cpu"))
    tor = Morpher(srcs, MorphConfig(sr=SR, device="torch"))
    assert tor.dev is not None and ref.dev is None
    m = ref.mdct
    for w in ([1, 0], [0.5, 0.5], [0.2, 0.8]):
        a, b = ref.render(w), tor.render(w)
        n = min(len(a), len(b))
        assert abs(len(a) - len(b)) <= 2
        if w == [1, 0]:                      # identity: the warp is an exact copy on both backends
            c = np.corrcoef(a[:n], b[:n])[0, 1]
            assert c > 0.97, f"torch backend differs from numpy for {w}: corr {c:.3f}"
        else:                                # the batched warp may pick other (equally valid) block
            la = np.log(m.mclt(a[:n])[0] + 1e-4)   # phases: compare magnitude spectrograms
            lb = np.log(m.mclt(b[:n])[0] + 1e-4)
            c = np.corrcoef(la.ravel(), lb.ravel())[0, 1]
            assert c > 0.97, f"torch backend spectrogram differs from numpy for {w}: corr {c:.3f}"


def test_weights_validation():
    srcs = _synthetic_sources()
    mp = Morpher(srcs, MorphConfig(sr=SR))
    y1 = mp.render([2, 2])           # normalised -> same as [0.5, 0.5]
    y2 = mp.render([0.5, 0.5])
    assert np.allclose(y1, y2)
    try:
        mp.render([1, 2, 3])
        assert False, "should reject wrong length"
    except ValueError:
        pass


def test_smooth_map_shorter_than_kernel():
    # a short source (a staccato note) has fewer frames than the smoothing kernel reaches
    g = np.cumsum(np.r_[0.0, np.full(29, 1.3)])
    gs = smooth_map(g, 13.8)
    assert gs.shape == g.shape
    assert gs[0] == g[0] and abs(gs[-1] - g[-1]) < 1e-9
    assert np.all(np.diff(gs) >= 0)


def test_end_to_end_sf2():
    if not (os.path.exists(SF2_A) and os.path.exists(SF2_B)):
        print("skipping sf2 test (fonts not found)")
        return
    import subprocess
    import tempfile
    import json
    with tempfile.TemporaryDirectory() as td:
        cmd = [sys.executable, "-m", "babymaker", "grid", "--src", f"{SF2_A}:0:0", "--src", f"{SF2_B}:0:48",
               "--key", "64", "--steps", "2", "--hold", "1.0", "--tail", "0.5", "--out", td]
        subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)
        idx = json.load(open(os.path.join(td, "index.json")))
        assert len(idx) == 3 and all(os.path.exists(os.path.join(td, m["file"])) for m in idx)
        import soundfile as sf
        y, sr = sf.read(os.path.join(td, "key064_w0.5_0.5.wav"))
        assert sr == 44100 and not np.isnan(y).any() and np.abs(y).max() <= 1.0
        f = estimate_f0(y, sr, midi_to_hz(64))
        assert f is not None and abs(1200 * np.log2(f / midi_to_hz(64))) < 10


if __name__ == "__main__":
    import inspect
    for name, fn in list(globals().items()):
        if name.startswith("test_") and inspect.isfunction(fn):
            fn()
            print("ok", name)
