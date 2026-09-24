# Video

The two videos about babymaker. One is a 30 second short. The other is a longer cut that goes over how the algorithm works, with the demos in between.

Everything in them is real output. The triangle demo's audio comes from the TypeScript morph in `web/lib`. The Satie and DOTABATA clips are rendered with `babymaker.arrange`. The explainer's curves and matrices come from `explainer_data.py`, which runs the Python morph on the demo's three instruments. The speed and DCT numbers come from `bench.py`.

The videos have no narration by default. I record the voiceover myself from `VOICEOVER.md`, which `build.py` writes with the time each line lands at (the picture is timed for a read at about 2.8 words a second). `--voice` lays in the cloned TTS narration instead. Both cuts end on the kobimusic outro, cut at 5 s with a fade to black.

## Files

| file | what it does |
|---|---|
| `script.py` | the narration for both cuts |
| `tts.py` | reads the narration in my voice (Qwen3-TTS-12Hz-1.7B-Base). Each line is checked by Whisper and retaken if the words do not match |
| `explainer_data.py` | dumps the algorithm's intermediates for the manim chapters |
| `scenes.py` | the manim chapters (why, the seven parts of the algorithm, what is next) |
| `frames.py` | the demo scenes (the triangle, the piano roll, the cards), drawn with cairo and piped into ffmpeg |
| `plans/make_plans.py` | the Satie and DOTABATA plans: the playhead wanders through random mixes and only touches a single source where the script names it. The long Satie ends on all eight pianos |
| `demo_audio.mjs` | sequences Alla Turca through the TypeScript morph along the triangle's timeline |
| `bench.py` | times the morph on CPU, GPU and in the TypeScript port, and measures the DCT facts the video quotes |
| `build.py` | lays out the segments, times them to the script, mixes the music and encodes. Writes `VOICEOVER.md` |
| `VOICEOVER.md` | the script to record, with a timecode per line |

## Run

I ran all of this on lanbox (RTX 5060, GPU 1). The TTS and the arrangement renders use the GPU, the rest runs on the CPU.

```bash
# 1. narration (needs qwen-tts, faster-whisper; the voice reference goes in video/work/voice_ref_clip.wav + .txt)
python3 -c "import json, sys; sys.path.insert(0, 'video'); import script; [json.dump({k: v['say'] for k, v in script.lines(s).items()}, open(f'video/work/say_{n}.json', 'w')) for n, s in (('short', script.SHORT), ('long', script.LONG))]"
CUDA_VISIBLE_DEVICES=1 python3 video/tts.py video/work/say_short.json video/work/vo_short
CUDA_VISIBLE_DEVICES=1 python3 video/tts.py video/work/say_long.json video/work/vo_long

# 2. the explainer's data, and the benchmark
python3 video/explainer_data.py
CUDA_VISIBLE_DEVICES=1 python3 video/bench.py

# 3. the MIDI renders (plans in video/plans/, written by make_plans.py; paths are relative to the plan)
python3 video/plans/make_plans.py
for p in pianos_short band_short pianos_long band_long; do
  CUDA_VISIBLE_DEVICES=1 python3 -m babymaker.arrange video/plans/$p.json
done

# 4. the videos
python3 video/build.py short --renders video/renders
python3 video/build.py long --renders video/renders
```

`build.py` writes `video/out/babymaker-short.mp4` and `babymaker-explained.mp4` (music and instrument sounds, no voice), a `-silent.mp4` copy with no audio, the `-music.wav` stem for mixing under a voiceover, and an `.srt` of the script. `--only S7cSpeed,demo` re-renders just those segments and keeps the rest.

## Inputs that are not in the repo

- The soundfonts: FluidR3_GM, MS_Basic (Microsoft GS Wavetable), A320U, TBOSf, FB01, OPL4 XI GM set, the Fairlight `piano.sf2` from the Nena pack, and the Mario Kart DS soundfont. The plans expect them in `fonts/` next to `plans/`.
- The MIDI: Satie's Gymnopédie No. 1 and `16 DOTABATA.MID` from the Nena MIDI collection.
- The voice reference (a minute of me talking, for the clone).
