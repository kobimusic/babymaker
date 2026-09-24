#!/usr/bin/env python3
"""A small local app for recording the voiceover over the finished videos.

    python3 video/recorder/server.py            # then open http://localhost:8773

It serves the page, the videos in video/out (with byte ranges, so the player can seek),
the script with each line's timecode (parsed from VOICEOVER.md), and keeps every take on
disk the moment it is recorded (video/voiceover/<cut>/). Export lays the takes on the
film's timeline (a newer take wins where two overlap), ducks the music under the voice,
normalises to -16 LUFS and muxes a new mp4 next to the others.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO = os.path.dirname(HERE)
OUT = os.path.join(VIDEO, "out")
STORE = os.environ.get("VO_STORE", os.path.join(VIDEO, "voiceover"))
sys.path.insert(0, VIDEO)
import script  # noqa: E402

PORT = int(os.environ.get("PORT", 8773))
READ_WPS = 2.8
CUTS = {
    "short": {"title": "Short (30 s)", "file": "babymaker-short", "script": script.SHORT},
    "long": {"title": "Long (explained)", "file": "babymaker-explained", "script": script.LONG},
}
LOCK = threading.Lock()


# ----------------------------------------------------------------------------- the script

def cues(cut):
    """Each line of the cut with the time it lands at, from VOICEOVER.md (build.py writes it)."""
    md = open(os.path.join(VIDEO, "VOICEOVER.md")).read()
    title = CUTS[cut]["title"]
    part = md.split(f"## {title}", 1)[1].split("\n## ", 1)[0]
    by_cap = {v["cap"]: (k, v) for k, v in script.lines(CUTS[cut]["script"]).items()}
    out = []
    for m in re.finditer(r"^\| (\d+):(\d+(?:\.\d+)?) \| ([^|]+) \| (.+) \|$", part, re.M):
        t = int(m.group(1)) * 60 + float(m.group(2))
        cap = m.group(4).strip()
        k, v = by_cap.get(cap, (None, {"cap": cap, "say": cap}))
        out.append({"id": k or f"line{len(out)}", "t": round(t, 2), "segment": m.group(3).strip(),
                    "cap": cap, "say": v["say"], "dur": round(len(v["say"].split()) / READ_WPS + 0.3, 2)})
    return out


def duration(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                       capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


# ----------------------------------------------------------------------------- takes

def take_dir(cut):
    d = os.path.join(STORE, cut, "takes")
    os.makedirs(d, exist_ok=True)
    return d


def takes_index(cut):
    p = os.path.join(STORE, cut, "takes.json")
    return json.load(open(p)) if os.path.exists(p) else []


def save_index(cut, takes):
    p = os.path.join(STORE, cut, "takes.json")
    json.dump(takes, open(p + ".tmp", "w"), indent=1)
    os.replace(p + ".tmp", p)


# ----------------------------------------------------------------------------- export

def _resample(x, sr, to):
    if sr == to:
        return x
    import soxr
    return soxr.resample(x, sr, to, quality="VHQ")


def _moving_average(x, n):
    c = np.concatenate([[0.0], np.cumsum(x)])
    h = n // 2
    i = np.arange(len(x))
    lo, hi = np.clip(i - h, 0, len(x)), np.clip(i + h + 1, 0, len(x))
    return (c[hi] - c[lo]) / n


def export(cut, offset_ms, duck=0.6):
    name = CUTS[cut]["file"]
    music, sr = sf.read(os.path.join(OUT, f"{name}-music.wav"), dtype="float64", always_2d=True)
    n = len(music)
    voice = np.zeros(n)
    covered = np.zeros(n, bool)
    xf = int(0.012 * sr)
    for tk in sorted(takes_index(cut), key=lambda t: t["created"]):     # newer takes win
        x, tsr = sf.read(os.path.join(take_dir(cut), tk["file"]), dtype="float64", always_2d=True)
        x = _resample(x.mean(1), tsr, sr)
        t0 = tk["t0"] + offset_ms / 1000.0
        a = int(round(max(tk["keepA"], t0) * sr))
        b = int(round(min(tk["keepB"], t0 + len(x) / sr) * sr))
        a, b = max(0, a), min(n, b)
        if b <= a:
            continue
        seg = x[a - int(round(t0 * sr)): b - int(round(t0 * sr))]
        seg = seg[: b - a]
        if len(seg) < b - a:
            seg = np.pad(seg, (0, b - a - len(seg)))
        ramp = np.ones(b - a)
        k = min(xf, (b - a) // 2)
        if k:
            ramp[:k] = np.linspace(0, 1, k)
            ramp[-k:] = np.linspace(1, 0, k)
        voice[a:b] = voice[a:b] * (1 - ramp) + seg * ramp
        covered[a:b] = True
    if not covered.any():
        raise ValueError("no takes to export yet")
    # a gentle high-pass (rumble, handling noise), then the voice to a steady level
    from scipy.signal import butter, sosfilt
    voice = sosfilt(butter(2, 70, "hp", fs=sr, output="sos"), voice)
    act = np.abs(voice) > 1e-4
    rms = np.sqrt(np.mean(voice[act] ** 2)) if act.any() else 1.0
    voice *= 0.12 / (rms + 1e-9)
    env = _moving_average(np.abs(voice), int(0.3 * sr))
    ref = np.percentile(env[env > 1e-4], 60) if np.any(env > 1e-4) else 1.0
    gain = 1.0 - duck * np.clip(env / ref, 0, 1)
    mix = music * gain[:, None] + voice[:, None]
    peak = np.abs(mix).max()
    if peak > 0.97:
        mix *= 0.97 / peak
    d = os.path.join(STORE, cut)
    raw = os.path.join(d, "mix.wav")
    sf.write(raw, mix.astype(np.float32), sr)
    sf.write(os.path.join(OUT, f"{name}-voice.wav"), voice.astype(np.float32), sr)
    # two-pass R128 to -16 LUFS, as the videos are
    r = subprocess.run(["ffmpeg", "-hide_banner", "-i", raw, "-af", "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json",
                        "-f", "null", "-"], capture_output=True, text=True, check=True)
    i = r.stderr.rindex("{")
    m = json.loads(r.stderr[i:r.stderr.index("}", i) + 1])
    norm = os.path.join(d, "mix.norm.wav")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", raw, "-af",
                    f"loudnorm=I=-16:TP=-1.5:LRA=11:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
                    f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}"
                    ":linear=true", "-ar", str(sr), norm], check=True)
    mp4 = os.path.join(OUT, f"{name}-vo.mp4")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", os.path.join(OUT, f"{name}-silent.mp4"), "-i", norm,
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-movflags", "+faststart", "-shortest", mp4], check=True)
    return {"mp4": f"/out/{name}-vo.mp4", "voice": f"/out/{name}-voice.wav",
            "path": mp4, "voice_path": os.path.join(OUT, f"{name}-voice.wav"),
            "covered_s": round(covered.sum() / sr, 1)}


# ----------------------------------------------------------------------------- http

class Handler(SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        if "/api/" in self.path:
            sys.stderr.write("%s %s\n" % (time.strftime("%H:%M:%S"), fmt % args))

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def _file(self, path, ctype):
        """Serve a file, honouring a single byte range (video seeking needs it)."""
        size = os.path.getsize(path)
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        if rng and rng.startswith("bytes="):
            a, _, b = rng[6:].partition("-")
            if a:
                start = int(a)
                end = int(b) if b else size - 1
            else:
                start = size - int(b)
            end = min(end, size - 1)
            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with open(path, "rb") as fh:
            fh.seek(start)
            left = end - start + 1
            try:
                while left > 0:
                    chunk = fh.read(min(1 << 20, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = dict(urllib.parse.parse_qsl(u.query))
        if u.path == "/":
            return self._file(os.path.join(HERE, "index.html"), "text/html; charset=utf-8")
        if u.path in ("/app.js", "/style.css", "/capture.js"):
            ct = "text/css" if u.path.endswith(".css") else "text/javascript"
            return self._file(os.path.join(HERE, u.path[1:]), ct + "; charset=utf-8")
        if u.path.startswith("/out/"):
            p = os.path.join(OUT, os.path.basename(u.path))
            if not os.path.exists(p):
                return self.send_error(404)
            ct = {"mp4": "video/mp4", "wav": "audio/wav", "png": "image/png"}.get(p.rsplit(".", 1)[-1],
                                                                                  "application/octet-stream")
            return self._file(p, ct)
        if u.path.startswith("/take/"):
            cut, f = u.path.split("/")[2:4]
            p = os.path.join(take_dir(cut), os.path.basename(f))
            return self._file(p, "audio/wav") if os.path.exists(p) else self.send_error(404)
        if u.path == "/api/cuts":
            res = []
            for k, c in CUTS.items():
                mp4 = os.path.join(OUT, f"{c['file']}.mp4")
                if os.path.exists(mp4):
                    res.append({"id": k, "title": c["title"], "video": f"/out/{c['file']}.mp4",
                                "duration": duration(mp4), "lines": cues(k),
                                "exported": os.path.exists(os.path.join(OUT, f"{c['file']}-vo.mp4"))})
            return self._json(res)
        if u.path == "/api/takes":
            return self._json(takes_index(q["cut"]))
        return self.send_error(404)

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        q = dict(urllib.parse.parse_qsl(u.query))
        if u.path == "/api/takes":
            cut = q["cut"]
            body = self.rfile.read(int(self.headers["Content-Length"]))
            with LOCK:
                tid = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
                f = f"{tid}.wav"
                open(os.path.join(take_dir(cut), f), "wb").write(body)
                info = sf.info(os.path.join(take_dir(cut), f))
                tk = {"id": tid, "file": f, "t0": float(q["t0"]), "keepA": float(q["keepA"]),
                      "keepB": float(q["keepB"]), "dur": round(info.duration, 3), "sr": info.samplerate,
                      "line": q.get("line") or None, "created": time.time()}
                takes = takes_index(cut) + [tk]
                save_index(cut, takes)
            return self._json(tk)
        if u.path == "/api/export":
            try:
                return self._json(export(q["cut"], float(q.get("offset_ms", 0))))
            except Exception as e:  # shown in the page
                return self._json({"error": str(e)}, 400)
        return self.send_error(404)

    def do_DELETE(self):
        u = urllib.parse.urlparse(self.path)
        q = dict(urllib.parse.parse_qsl(u.query))
        if u.path == "/api/takes":
            cut = q["cut"]
            with LOCK:
                takes = takes_index(cut)
                keep = [t for t in takes if t["id"] != q["id"]]
                for t in takes:
                    if t["id"] == q["id"]:
                        # deleted takes go to a bin, not away
                        binp = os.path.join(STORE, cut, "deleted")
                        os.makedirs(binp, exist_ok=True)
                        src = os.path.join(take_dir(cut), t["file"])
                        if os.path.exists(src):
                            shutil.move(src, os.path.join(binp, t["file"]))
                save_index(cut, keep)
            return self._json({"ok": True})
        return self.send_error(404)


def main():
    os.makedirs(STORE, exist_ok=True)
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"voiceover recorder on http://localhost:{PORT}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
