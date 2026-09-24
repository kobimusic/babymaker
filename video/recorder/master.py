#!/usr/bin/env python3
"""Clean up and master the recorded voiceover.

Rebuilds the voice from the original takes (newer takes win where they overlap; each take's
DC offset removed), brings it up to a working level, then: DeepFilterNet3 noise removal
(capped at 40 dB) -> EQ (80 Hz high-pass, a little less 300 Hz mud, presence at 3.5 kHz,
some air) -> de-esser -> 3:1 compression -> limiter -> -16 LUFS -> a gentle expander that
only lowers what sits well under the speech.

Tuned with DNSMOS (P.835) on these takes, raw -> cleaned: short OVRL 3.48 -> 3.92
(SIG 4.14 -> 4.31, BAK 3.37 -> 4.30); long, 60-180 s, OVRL 3.55 -> 3.94.

    PYTHONPATH=~/voice-env/site python3 video/recorder/master.py short long --out ~/Downloads

Writes <cut>-voice-clean.mp3 (the voice alone, full length, on the film's timeline) and
<cut>-mix.mp3 (the voice over the music, ducked), and the same voice as a float WAV in
video/out for the recorder's export. DeepFilterNet is optional: without it the chain
runs without the noise removal and says so.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

# DeepFilterNet lives in its own folder (see video/README.md)
_DF = os.path.expanduser("~/voice-env/site")
if os.path.isdir(_DF) and _DF not in sys.path:
    sys.path.append(_DF)

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO = os.path.dirname(HERE)
OUT = os.path.join(VIDEO, "out")
STORE = os.environ.get("VO_STORE", os.path.join(VIDEO, "voiceover"))
NAMES = {"short": "babymaker-short", "long": "babymaker-explained"}
SR = 48000

# EQ, de-esser, compressor, limiter (ffmpeg; levels are linear where ffmpeg wants them)
CHAIN = ",".join([
    "highpass=f=80:poles=2",
    "equalizer=f=300:t=q:w=1.0:g=-2.5",           # mud
    "equalizer=f=3500:t=q:w=0.9:g=3.5",           # presence: this mic is dark (1-4 kHz sits 15 dB down)
    "highshelf=f=10000:g=2",                      # air
    "deesser=i=0.35:m=0.5:f=0.5:s=o",
    "acompressor=threshold=0.063:ratio=3:attack=8:release=200:knee=4:makeup=2",   # -24 dB, 3:1
    "alimiter=limit=0.89:attack=3:release=60:level=disabled",
])
# after loudness: -38 dB, 2:1, never more than 12 dB down. Lowers the room between words.
EXPANDER = "agate=threshold=0.0126:ratio=2:range=0.25:attack=4:release=220:knee=3:detection=rms"


def takes(cut):
    return json.load(open(os.path.join(STORE, cut, "takes.json")))


def assemble(cut, n, sr=SR, offset_ms=0.0):
    """The voice on the film's timeline, from the original takes; a newer take wins."""
    import soxr
    voice = np.zeros(n)
    covered = np.zeros(n, bool)
    xf = int(0.012 * sr)
    for tk in sorted(takes(cut), key=lambda t: t["created"]):
        x, tsr = sf.read(os.path.join(STORE, cut, "takes", tk["file"]), dtype="float64", always_2d=True)
        x = x.mean(1)
        # the mic records with a large DC offset (+0.10); a zero-phase high-pass takes it out
        # per take, so the offset cannot step at take boundaries
        from scipy.signal import butter, sosfiltfilt
        x = sosfiltfilt(butter(2, 30, "hp", fs=tsr, output="sos"), x)
        if tsr != sr:
            x = soxr.resample(x, tsr, sr, quality="VHQ")
        t0 = tk["t0"] + offset_ms / 1000.0
        a = max(0, int(round(max(tk["keepA"], t0) * sr)))
        b = min(n, int(round(min(tk["keepB"], t0 + len(x) / sr) * sr)))
        if b <= a:
            continue
        s0 = int(round(t0 * sr))
        seg = x[a - s0: b - s0]
        seg = np.pad(seg, (0, max(0, b - a - len(seg))))[: b - a]
        ramp = np.ones(b - a)
        k = min(xf, (b - a) // 2)
        if k:
            ramp[:k] = np.linspace(0, 1, k)
            ramp[-k:] = np.linspace(1, 0, k)
        voice[a:b] = voice[a:b] * (1 - ramp) + seg * ramp
        covered[a:b] = True
    return voice, covered


def spans(mask, sr, pad_s=0.25):
    """Contiguous covered regions (padded a little, joined when close)."""
    idx = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    out = []
    for a, b in zip(idx[::2], idx[1::2]):
        a, b = max(0, a - int(pad_s * sr)), min(len(mask), b + int(pad_s * sr))
        if out and a <= out[-1][1]:
            out[-1][1] = b
        else:
            out.append([a, b])
    return out


def denoise(x, covered, sr=SR, atten_db=40.0, chunk_s=30.0, overlap_s=1.0):
    """DeepFilterNet3 over each recorded region, in overlapping chunks for memory."""
    try:
        import torch
        from df.enhance import enhance, init_df
    except ImportError:
        print("  (DeepFilterNet not available: skipping noise removal)")
        return x, False
    model, st, _ = init_df(log_level="ERROR")
    y = x.copy()
    C, O = int(chunk_s * sr), int(overlap_s * sr)
    for a, b in spans(covered, sr):
        seg = x[a:b]
        out = np.zeros_like(seg)
        wsum = np.zeros_like(seg)
        i = 0
        while i < len(seg):
            j = min(len(seg), i + C)
            lo = max(0, i - O)
            piece = torch.as_tensor(seg[lo:j], dtype=torch.float32)[None]
            en = enhance(model, st, piece, atten_lim_db=atten_db)[0].numpy().astype(np.float64)
            w = np.ones(j - lo)
            if lo < i:
                w[: i - lo] = np.linspace(0, 1, i - lo)         # fade in over the overlap
                out[lo:i] *= 1 - w[: i - lo]                    # and the previous chunk out
                wsum[lo:i] *= 1 - w[: i - lo]
            out[lo:j] += en * w
            wsum[lo:j] += w
            i = j
        y[a:b] = out / np.maximum(wsum, 1e-9)
    return y, True


def ffmpeg_chain(x, sr, chain):
    with tempfile.TemporaryDirectory() as td:
        a, b = os.path.join(td, "in.wav"), os.path.join(td, "out.wav")
        sf.write(a, x.astype(np.float32), sr, subtype="FLOAT")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", a, "-af", chain, "-c:a", "pcm_f32le", b], check=True)
        y, _ = sf.read(b, dtype="float64")
    return y


def loudnorm(x, sr, target=-16.0, tp=-1.5, channels=1):
    """Two-pass EBU R128, linear (no dynamic processing, just the gain)."""
    with tempfile.TemporaryDirectory() as td:
        a, b = os.path.join(td, "in.wav"), os.path.join(td, "out.wav")
        sf.write(a, x.astype(np.float32), sr, subtype="FLOAT")
        r = subprocess.run(["ffmpeg", "-hide_banner", "-i", a, "-af",
                            f"loudnorm=I={target}:TP={tp}:LRA=11:print_format=json", "-f", "null", "-"],
                           capture_output=True, text=True, check=True)
        i = r.stderr.rindex("{")
        m = json.loads(r.stderr[i:r.stderr.index("}", i) + 1])
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", a, "-af",
                        f"loudnorm=I={target}:TP={tp}:LRA=11:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
                        f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}"
                        ":linear=true", "-ar", str(sr), "-c:a", "pcm_f32le", b], check=True)
        y, _ = sf.read(b, dtype="float64")
    return y


def moving_average(x, n):
    c = np.concatenate([[0.0], np.cumsum(x)])
    h = n // 2
    i = np.arange(len(x))
    lo, hi = np.clip(i - h, 0, len(x)), np.clip(i + h + 1, 0, len(x))
    return (c[hi] - c[lo]) / n


def master(cut, offset_ms=0.0, log=print):
    """-> (clean voice at 44.1 kHz, the ducked mix at 44.1 kHz stereo, a dict of measurements)."""
    import soxr
    name = NAMES[cut]
    music, msr = sf.read(os.path.join(OUT, f"{name}-music.wav"), dtype="float64", always_2d=True)
    n48 = int(round(len(music) / msr * SR))
    raw, covered = assemble(cut, n48, SR, offset_ms)
    if not covered.any():
        raise ValueError("no takes to master yet")
    log(f"  {cut}: {covered.sum() / SR:.0f} s of voice from {len(takes(cut))} takes")
    # the takes are quiet (speech around -44 dB): bring the peak to -6 dBFS before the denoiser,
    # which is trained on normally levelled speech
    pre = 10 ** (-6 / 20) / (np.abs(raw).max() + 1e-12)
    raw = raw * pre
    log(f"  gain before cleanup: {20 * np.log10(pre):+.1f} dB")
    x, dn = denoise(raw, covered)
    x = ffmpeg_chain(x, SR, CHAIN)
    x[~np.convolve(covered, np.ones(int(0.3 * SR)), "same").astype(bool)] = 0.0   # keep the gaps silent
    x = loudnorm(x, SR)
    x = ffmpeg_chain(x, SR, EXPANDER)
    v = soxr.resample(x, SR, msr, quality="VHQ")[: len(music)]
    v = np.pad(v, (0, len(music) - len(v)))
    # the mix: the music ducks under the voice
    env = moving_average(np.abs(v), int(0.3 * msr))
    ref = np.percentile(env[env > 1e-4], 60) if np.any(env > 1e-4) else 1.0
    gain = 1.0 - 0.6 * np.clip(env / ref, 0, 1)
    mix = music * gain[:, None] + v[:, None]
    mix = loudnorm(mix, msr)
    return v, mix, {"denoised": dn, "raw48": raw, "clean48": x, "covered48": covered}


def measure(x, covered, sr):
    """Room noise between words, speech level and tone, peaks: what the chain should change."""
    from scipy.signal import welch
    win = int(0.05 * sr)
    n = len(x) // win
    blocks = x[: n * win].reshape(n, win)
    cov = covered[: n * win].reshape(n, win).all(1)
    db = 20 * np.log10(np.sqrt((blocks[cov] ** 2).mean(1)) + 1e-12)
    speech = blocks[cov][db >= np.percentile(db, 60)].ravel()
    fr, p = welch(speech, sr, nperseg=4096)
    band = lambda a, b: 10 * np.log10(p[(fr >= a) & (fr < b)].mean() + 1e-20)
    return {"noise_db": float(np.percentile(db, 10)), "speech_db": float(np.percentile(db, 80)),
            "snr_db": float(np.percentile(db, 80) - np.percentile(db, 10)),
            "presence_vs_body_db": float(band(1000, 4000) - band(250, 1000)),
            "peak_dbfs": float(20 * np.log10(np.abs(x).max() + 1e-12)),
            "clipped": int((np.abs(x) >= 0.999).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cuts", nargs="+", choices=list(NAMES))
    ap.add_argument("--out", default=os.path.expanduser("~/Downloads"))
    ap.add_argument("--offset-ms", type=float, default=0.0)
    a = ap.parse_args()
    for cut in a.cuts:
        name = NAMES[cut]
        v, mix, info = master(cut, a.offset_ms)
        before = measure(info["raw48"], info["covered48"], SR)
        after = measure(info["clean48"], info["covered48"], SR)
        for k in before:
            print(f"    {k:20s} {before[k]:9.1f} -> {after[k]:9.1f}")
        sf.write(os.path.join(OUT, f"{name}-voice.wav"), v.astype(np.float32), 44100, subtype="FLOAT")
        sf.write(os.path.join(OUT, f"{name}-mix.wav"), mix.astype(np.float32), 44100, subtype="FLOAT")
        for src, dst, ch in ((f"{name}-voice.wav", f"{name}-voice-clean.mp3", "1"), (f"{name}-mix.wav", f"{name}-mix.mp3", "2")):
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", os.path.join(OUT, src), "-ac", ch,
                            "-c:a", "libmp3lame", "-q:a", "0", os.path.join(a.out, dst)], check=True)
        print(f"  wrote {a.out}/{name}-voice-clean.mp3 and {name}-mix.mp3")


if __name__ == "__main__":
    main()
