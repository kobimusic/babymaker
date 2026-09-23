"""Source notes for the morph: render an .sf2 preset note through the bundled
TinySoundFont renderer, or load a WAV; then prepare (mono, resample, pitch-correct,
trim, level-match) so every source sits at the same pitch, level and sample rate."""
from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field

import numpy as np
import soundfile as sf
import soxr

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE_ROOT = os.path.dirname(HERE)
RENDER_DIR = os.path.join(PIPE_ROOT, "render")
TSFRENDER = os.path.join(RENDER_DIR, "tsfrender")


def ensure_renderer() -> str:
    src = os.path.join(RENDER_DIR, "tsfrender.c")
    if not os.path.exists(TSFRENDER) or os.path.getmtime(TSFRENDER) < os.path.getmtime(src):
        subprocess.run(["gcc", "-O2", "-o", TSFRENDER, src, "-lm"], check=True)
    return TSFRENDER


def midi_to_hz(note: float) -> float:
    return 440.0 * 2.0 ** ((note - 69.0) / 12.0)


def hz_to_midi(hz: float) -> float:
    return 69.0 + 12.0 * np.log2(hz / 440.0)


@dataclass
class Source:
    name: str
    audio: np.ndarray            # mono float64 at `sr`, prepared
    sr: int
    note: float                  # MIDI note the audio is at (after correction)
    meta: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return len(self.audio) / self.sr


# ----------------------------------------------------------------------------- spec parsing

def parse_source_spec(spec: str) -> dict:
    """`font.sf2:bank:preset`, `font.sf2:preset` (bank 0), `sound.wav`, `sound.wav:note`
    (note = MIDI number or name like C4/F#3 the recording is at; default = target key)."""
    parts = spec.split(":")
    # windows-style drive letters or paths with ':' are not expected; first part is the path
    path = os.path.expanduser(parts[0])
    ext = os.path.splitext(path)[1].lower()
    if ext == ".sf2":
        if len(parts) == 1:
            bank, preset = 0, 0
        elif len(parts) == 2:
            bank, preset = 0, int(parts[1])
        else:
            bank, preset = int(parts[1]), int(parts[2])
        return {"kind": "sf2", "path": path, "bank": bank, "preset": preset}
    note = _parse_note(parts[1]) if len(parts) > 1 and parts[1] else None
    return {"kind": "wav", "path": path, "note": note}


_NOTE_NAMES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def _parse_note(s: str) -> float:
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        pass
    name = s[0].upper()
    i = 1
    acc = 0
    while i < len(s) and s[i] in "#b":
        acc += 1 if s[i] == "#" else -1
        i += 1
    octave = int(s[i:])
    return 12 * (octave + 1) + _NOTE_NAMES[name] + acc


# ----------------------------------------------------------------------------- loading

def render_sf2_notes(sf2: str, bank: int, preset: int, keys, vel: int, hold_s: float,
                     tail_s: float, sr: int) -> dict:
    """Render several notes (each held hold_s + release tail) with ONE renderer call - the
    soundfont (up to hundreds of MB) is loaded once.  Notes are spaced so that no tail
    overlaps the next onset.  Returns {key: mono float64}."""
    exe = ensure_renderer()
    keys = list(keys)
    slot = hold_s + tail_s + 0.25
    total = slot * len(keys)
    with tempfile.TemporaryDirectory() as td:
        ev = os.path.join(td, "events.txt")
        wav = os.path.join(td, "notes.wav")
        lines = [f"K 0 {bank} {preset}", "R 0 2.000"]
        v = np.clip(vel / 127.0, 0.02, 1.0)
        for i, key in enumerate(keys):
            lines.append(f"N {i * slot:.6f} 0 {key} {v:.4f}")
            lines.append(f"F {i * slot + hold_s:.6f} 0 {key}")
        with open(ev, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        res = subprocess.run([exe, sf2, ev, wav, str(sr), f"{total:.4f}", "0.0", "mono"],
                             capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"tsfrender failed on {sf2} {bank}:{preset}: {res.stderr.strip()}")
        audio, file_sr = sf.read(wav, dtype="float64", always_2d=True)
    assert file_sr == sr
    mono = audio.mean(axis=1)
    n_slot = int(round(slot * sr))
    n_note = int(round((hold_s + tail_s) * sr))
    return {key: mono[i * n_slot:i * n_slot + n_note].copy() for i, key in enumerate(keys)}


def render_sf2_note(sf2: str, bank: int, preset: int, key: int, vel: int, hold_s: float,
                    tail_s: float, sr: int) -> np.ndarray:
    """Render one note (key, vel) held for hold_s, plus release tail. Returns mono float64."""
    return render_sf2_notes(sf2, bank, preset, [key], vel, hold_s, tail_s, sr)[key]


def load_wav(path: str, sr: int) -> np.ndarray:
    audio, file_sr = sf.read(path, dtype="float64", always_2d=True)
    x = audio.mean(axis=1)
    if file_sr != sr:
        x = soxr.resample(x, file_sr, sr, quality="VHQ")
    return x


# ----------------------------------------------------------------------------- pitch

def estimate_f0(x: np.ndarray, sr: int, f_expected: float, search_semitones: float = 1.5,
                min_rel_db: float = -30.0, min_prominence_db: float = 15.0) -> float | None:
    """Fine pitch of the sustained part: the lowest strong partial h = 1..4 found within
    +-search_semitones of h * f_expected in a long zero-padded FFT, divided by h.  Partial
    peaks (not autocorrelation) so that inharmonic sounds such as pianos are not biased
    sharp by their stretched partials.  Returns None when no partial is strong enough."""
    n = len(x)
    if n < sr // 10:
        return None
    env = np.abs(x)
    start = min(int(np.argmax(env)) + int(0.05 * sr), max(0, n - int(0.2 * sr)))
    seg = x[start:start + int(1.0 * sr)]
    if len(seg) < int(0.08 * sr):
        seg = x[:int(1.0 * sr)]
    seg = seg - seg.mean()
    if np.max(np.abs(seg)) < 1e-6:
        return None
    seg = seg * np.hanning(len(seg))
    nfft = max(1 << 17, 1 << int(np.ceil(np.log2(4 * len(seg)))))
    S = np.abs(np.fft.rfft(seg, nfft))
    df = sr / nfft
    ref = S[int(20 / df):int(min(sr / 2 - 1, 6000) / df)].max()
    ratio = 2 ** (search_semitones / 12)
    wide = 2 ** (6 / 12)
    for h in range(1, 5):
        lo, hi = int(h * f_expected / ratio / df), int(h * f_expected * ratio / df)
        if hi + 1 >= len(S) or hi - lo < 3:
            break
        k = lo + int(np.argmax(S[lo:hi]))
        if S[k] < ref * 10 ** (min_rel_db / 20):
            continue
        # a partial must stand out of the local floor (noise gives ~10 dB peak-to-median)
        floor = np.median(S[int(h * f_expected / wide / df):int(min(len(S) - 1, h * f_expected * wide / df))])
        if S[k] < floor * 10 ** (min_prominence_db / 20):
            continue
        # peak must be a local maximum with a parabolic refinement
        if not (S[k] >= S[k - 1] and S[k] >= S[k + 1]):
            continue
        a, b, c = np.log(S[k - 1:k + 2] + 1e-20)
        den = a - 2 * b + c
        d = 0.5 * (a - c) / den if abs(den) > 1e-12 else 0.0
        return float((k + d) * df / h)
    return None


def pitch_shift_resample(x: np.ndarray, sr: int, ratio: float) -> np.ndarray:
    """Speed change by `ratio` (>1 = higher pitch, shorter)."""
    if abs(ratio - 1.0) < 1e-6:
        return x
    return soxr.resample(x, sr * ratio, sr, quality="VHQ")


# ----------------------------------------------------------------------------- preparation

def prepare(x: np.ndarray, sr: int, nominal_note: float, target_note: float, *,
            trim_db: float = -60.0, correct_pitch: bool = True, max_correction_cents: float = 150.0,
            fade_out_s: float = 0.01) -> tuple[np.ndarray, dict]:
    """Trim leading silence, move to target_note (nominal + measured fine error), strip
    trailing silence, normalise peak-RMS to 1.  Returns (audio, info)."""
    info = {"nominal_note": nominal_note, "target_note": target_note}
    x = np.asarray(x, np.float64)
    if len(x) == 0 or np.max(np.abs(x)) < 1e-9:
        raise ValueError("source is silent")
    # 1) resample to target pitch
    ratio = 2 ** ((target_note - nominal_note) / 12.0)
    measured = None
    if correct_pitch:
        f = estimate_f0(x, sr, midi_to_hz(nominal_note))
        if f is not None:
            cents = 1200 * np.log2(f / midi_to_hz(nominal_note))
            measured = float(hz_to_midi(f))
            if abs(cents) <= max_correction_cents:
                ratio = midi_to_hz(target_note) / f
            info["measured_cents_off"] = float(cents)
    info["measured_note"] = measured
    info["speed_ratio"] = float(ratio)
    x = pitch_shift_resample(x, sr, ratio)
    # 2) trim leading silence (relative to peak); keep 2 ms of pre-roll
    peak = np.max(np.abs(x))
    thr = peak * 10 ** (trim_db / 20)
    above = np.nonzero(np.abs(x) > thr)[0]
    lo = max(0, int(above[0]) - int(0.002 * sr))
    hi = min(len(x), int(above[-1]) + int(0.02 * sr))
    x = x[lo:hi]
    info["trim_samples"] = [int(lo), int(hi)]
    # 3) short fade-out to avoid a click at the end
    nf = min(len(x), int(fade_out_s * sr))
    if nf > 1:
        x[-nf:] *= np.linspace(1, 0, nf)
    # 4) level: peak short-time RMS -> 1.0
    win = max(1, int(0.02 * sr))
    e = np.convolve(x ** 2, np.ones(win) / win, mode="same")
    prms = float(np.sqrt(e.max()))
    info["peak_rms_in"] = prms
    x = x / (prms + 1e-12)
    return x, info


def load_source(spec: str, sr: int, key: int, vel: int = 100, hold_s: float = 2.0, tail_s: float = 1.5,
                correct_pitch: bool = True, raw: np.ndarray | None = None) -> Source:
    """Load + prepare one source at `key`. `raw` (from render_sf2_notes / load_wav) skips the
    render/read step."""
    d = parse_source_spec(spec)
    if d["kind"] == "sf2":
        raw = render_sf2_note(d["path"], d["bank"], d["preset"], key, vel, hold_s, tail_s, sr) if raw is None else raw
        nominal = float(key)
        name = f"{os.path.splitext(os.path.basename(d['path']))[0]}:{d['bank']}:{d['preset']}"
    else:
        raw = load_wav(d["path"], sr) if raw is None else raw
        nominal = float(d["note"]) if d["note"] is not None else float(key)
        name = os.path.splitext(os.path.basename(d["path"]))[0]
    audio, info = prepare(raw, sr, nominal, float(key), correct_pitch=correct_pitch)
    info.update(d)
    info["raw_duration_s"] = len(raw) / sr
    return Source(name=name, audio=audio, sr=sr, note=float(key), meta=info)
