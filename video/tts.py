#!/usr/bin/env python3
"""Narration in Nathan's voice: Qwen3-TTS-12Hz-1.7B-Base, cloned from a clip of him talking.

Every line is generated TAKES times; each take is transcribed by Whisper and the one whose
words best match the script wins (sampling TTS now and then drops or repeats a word). Takes
are cached by text, so editing one line only regenerates that line.

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=~/qwen-tts-env/site:~/vid-env/site \
        python3 video/tts.py lines.json out_dir/

lines.json: {"id": "text", ...}. Writes out_dir/<id>.wav (24 kHz mono) and out_dir/index.json.
"""
import difflib
import hashlib
import json
import os
import re
import sys

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
REF_WAV = os.path.join(HERE, "work", "voice_ref_clip.wav")
REF_TXT = os.path.join(HERE, "work", "voice_ref_clip.txt")
MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
TAKES = 3
BATCH = 4
MAX_CHARS = 420


def words(s):
    return re.findall(r"[a-z0-9']+", s.lower().replace("-", " "))


def match(a, b):
    return difflib.SequenceMatcher(None, words(a), words(b)).ratio()


def trim(x, sr, db=-45):
    """Cut the silence the model leaves at either end, keeping 60 ms of air."""
    env = np.abs(x)
    thr = env.max() * 10 ** (db / 20)
    idx = np.where(env > thr)[0]
    if not len(idx):
        return x
    pad = int(0.06 * sr)
    return x[max(0, idx[0] - pad): idx[-1] + pad]


def main():
    lines = json.load(open(sys.argv[1]))
    out = sys.argv[2]
    os.makedirs(out, exist_ok=True)
    cache = os.path.join(out, "cache")
    os.makedirs(cache, exist_ok=True)

    todo = {}
    for k, text in lines.items():
        h = hashlib.sha1(text.encode()).hexdigest()[:12]
        if not os.path.exists(os.path.join(cache, f"{h}.wav")):
            todo[k] = (text, h)
    print(f"{len(lines)} lines, {len(todo)} to generate", flush=True)

    if todo:
        import torch
        from qwen_tts import Qwen3TTSModel
        from faster_whisper import WhisperModel
        tts = Qwen3TTSModel.from_pretrained(MODEL, device_map="cuda:0", dtype=torch.bfloat16)
        prompt = tts.create_voice_clone_prompt(ref_audio=REF_WAV, ref_text=open(REF_TXT).read().strip())
        asr = WhisperModel("medium.en", device="cpu", compute_type="int8", cpu_threads=12)
        def take(texts, seed):
            torch.manual_seed(seed)
            wavs, sr = tts.generate_voice_clone(text=texts, language=["English"] * len(texts),
                                                voice_clone_prompt=prompt * len(texts) if isinstance(prompt, list) else [prompt] * len(texts))
            return [trim(np.asarray(w, dtype=np.float32), sr) for w in wavs], sr

        def score(text, x, sr):
            segs, _ = asr.transcribe(_to16k(x, sr), language="en")
            heard = " ".join(s.text for s in segs)
            sc = match(text, heard)
            if len(x) / sr / max(1, len(words(text))) > 0.62:      # stalled or looped
                sc -= 0.3
            return sc, heard

        def save(k, text, h, sc, heard, x, sr):
            sf.write(os.path.join(cache, f"{h}.wav"), x, sr)
            json.dump({"text": text, "heard": heard, "score": round(sc, 3)},
                      open(os.path.join(cache, f"{h}.json"), "w"))
            print(f"  {k}: {len(x) / sr:5.1f}s  match {sc:.2f}", flush=True)

        def batches(items):
            """Similar lengths together, and no more text per batch than an 8 GB card decodes."""
            items = sorted(items, key=lambda kv: len(kv[1][0]))
            cur, n = [], 0
            for kv in items:
                if cur and (n + len(kv[1][0]) > MAX_CHARS or len(cur) >= BATCH):
                    yield cur
                    cur, n = [], 0
                cur.append(kv)
                n += len(kv[1][0])
            if cur:
                yield cur

        def run(batch, seed):
            try:
                return [take([t for _, (t, _) in batch], seed)]
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                return [take([t], seed) for _, (t, _) in batch]

        best = {}
        pending = list(todo.items())
        for attempt in range(TAKES):
            redo = []
            for batch in batches(pending):
                outs = run(batch, 7 + 1000 * attempt + len(best))
                xs = [x for o in outs for x in o[0]]
                sr = outs[0][1]
                for (k, (text, h)), x in zip(batch, xs):
                    sc, heard = score(text, x, sr)
                    if k not in best or sc > best[k][0]:
                        best[k] = (sc, heard, x, sr)
                    if best[k][0] > 0.97 or attempt == TAKES - 1:
                        save(k, text, h, *best[k])
                    else:
                        redo.append((k, (text, h)))
            if not redo:
                break
            print(f"  retaking {len(redo)} lines", flush=True)
            pending = redo

    index = {}
    for k, text in lines.items():
        h = hashlib.sha1(text.encode()).hexdigest()[:12]
        x, sr = sf.read(os.path.join(cache, f"{h}.wav"))
        sf.write(os.path.join(out, f"{k}.wav"), x, sr)
        meta = json.load(open(os.path.join(cache, f"{h}.json")))
        index[k] = {"text": text, "dur": round(len(x) / sr, 3), "score": meta["score"], "heard": meta["heard"]}
    json.dump(index, open(os.path.join(out, "index.json"), "w"), indent=1)
    worst = sorted(index.items(), key=lambda kv: kv[1]["score"])[:5]
    print("lowest matches:")
    for k, m in worst:
        print(f"  {m['score']:.2f} {k}: heard {m['heard']!r}")


def _to16k(x, sr):
    import soxr
    return soxr.resample(x, sr, 16000)


if __name__ == "__main__":
    main()
