# babymaker

Take the same note from N instruments and make the sound in between them.
Not a crossfade: the attack, the formants, the harmonics and the decay of the
new note each sit somewhere between its parents. Two sources give a slider,
three a triangle, N an (N-1)-simplex.

It was built for a synthetic-data pipeline that renders MIDI through sample
libraries to train transcription models. There are only so many distinct
instruments to render with, so a model can learn those exact sounds instead of
the instrument in general. Morphing between them gives many more timbres
without recording anything.

This repo has three parts:

| | |
|---|---|
| `babymaker/` | the Python library and CLI (numpy, with optional numba and a torch/CUDA backend) |
| `web/lib/babymaker/` | a line-for-line TypeScript port of the `synth="mix"`, `env_mode="log"` path |
| `web/` | the browser demo: six Sonatina Symphonic Orchestra instruments, three on a triangle, Alla Turca playing through whatever point you drag to |

## The demo

```bash
cd web
npm install
npm run dev          # http://localhost:5173
npm run build        # static site in web/dist/ (relative paths, host it anywhere)
```

Click the stage to load the instruments. The pink dot is the playhead;
wherever you drag it, the note Mozart is playing is rebuilt from the three
instruments at the weights shown. Click a dot to swap its instrument, drag the
dots to move them. Put a harp or piano on a dot and move toward it: the note
gets shorter, because the decay is interpolated too.

The six sources are analysed once, in a Web Worker, on load (about 1.5 s on a
desktop). After that each drag is one render (about 80-130 ms). Notes already
playing keep their sample; new notes get the new one.

## Python

```bash
pip install -e ".[fast]"          # numba kernels; add [gpu] for torch/CUDA
./render/build.sh                 # only needed for .sf2 sources (TinySoundFont, gcc)

# 50/50 between two pianos at C4
python3 -m babymaker render --src /usr/share/sounds/sf2/FluidR3_GM.sf2:0:0 \
    --src ~/Downloads/A320U.sf2:0:0 --key 60 --weights 0.5,0.5 --out baby.wav

# every point of a 3-source triangle (steps=3 -> 10 points), at two keys, plus the
# prepared sources; index.json / key060_morph.json describe everything
python3 -m babymaker grid --src FluidR3_GM.sf2:0:0 --src A320U.sf2:0:0 \
    --src TimGM6mb.sf2:0:48 --key 60 72 --steps 3 --sources-too -v --out out/morphs

# 20 Dirichlet-random weight vectors (alpha < 1 favours the corners)
python3 -m babymaker random --src ... --n 20 --seed 0 --alpha 0.7 --out out/morphs

python3 -m babymaker sources --src ... --out out/src     # dump the aligned sources
python3 -m babymaker list some.sf2                       # bank preset name
```

Source spec: `font.sf2:bank:preset`, `font.sf2:preset` (bank 0), `sound.wav`,
`sound.wav:note` (the note the recording is at, `60` or `C4`; default `--key`).
`--hold` / `--tail` set how long an sf2 note is held and how much release is
rendered; `--set key=value` overrides any `MorphConfig` field. The `render/`
renderer is only built (automatically, on first use) when a source is an .sf2.

```python
from babymaker import Morpher, MorphConfig, load_source
srcs = [load_source(s, 44100, key=60) for s in ("A.sf2:0:0", "B.sf2:0:0", "C.wav:C4")]
mp = Morpher(srcs, MorphConfig(sr=44100))     # analysis once
y = mp.render([0.2, 0.3, 0.5])                # float64 mono, any weights
```

## How it works

**Sources** (`sources.py`).  sf2 notes are rendered through the bundled
TinySoundFont renderer (`render/tsfrender`), so envelopes, filters, loops and
layered zones are exactly what a SoundFont player would play.  Each source is then
made mono, resampled to the target pitch (nominal note + measured fine error:
the lowest strong partial in a long FFT, so pianos are not read sharp by their
stretched partials), trimmed of leading/trailing silence and normalised to a
peak short-time RMS of 1 (the original level is kept and restored, dB-interpolated,
on output).

**Transform** (`lapped.py`).  The MDCT (a lapped DCT-IV, sine window, M = 1024
bins, 2048-sample window; 2048 bins below A2) and its MDST sibling form the
MCLT, giving a magnitude and a phase per bin per frame.  Frames are taken every
M/4 samples (5.8 ms): the four interleaved hop-M lattices each satisfy TDAC, so
overlap-add / 4 reconstructs exactly, and modified coefficients are averaged over
four lattices.  Synthesis writes back only the real (MDCT) part.

**Analysis** (`morph.py`, `Morpher._analyse`).  Every frame is split into

| part | what | how it is morphed |
|------|------|-------------------|
| level | frame RMS in dB | w-average in dB (decay rates interpolate) |
| envelope | DCT-cepstral *true envelope* (Röbel & Rodet) of the unit-RMS spectrum; cutoff q = 0.3·sr/f0, below the harmonic ripple | split by cepstral order: tilt (< 3) and per-harmonic detail (≥ 0.1·sr/f0) w-averaged in dB; the broad formant band in between treated as a distribution over log-frequency and combined as a 1-D **optimal-transport (Wasserstein) barycentre** (smoothed over 80 ms) — a peak at 500 Hz in A and 1 kHz in B lands near 700 Hz at w = ½ instead of becoming two half-height bumps (`env_mode=log` gives plain dB interpolation) |
| fine structure | spectrum / envelope: the harmonic comb and noise detail | w-average of \|a\|^0.6 (loudness-like; `fine_gamma=0` for dB) — pitch is common, so bins line up |
| phase | not modelled: taken from a real signal (below) | — |

**Time** — sources are aligned to source 0 by DTW on (level, low-order cepstrum);
the maps are anchored at the onset and Gaussian-smoothed (80 ms) so the warp
rate never jumps.  The output timeline is the w-average of the aligned
timelines (a 2 s decay and a 4 s sustain morph into 3 s); each source's warp
slope is clamped to [1/2.5, 2.5] and to exactly 1 for the first 100 ms, so an
attack is never stretched or repeated.

**Synthesis** (`synth="mix"`, default).  The magnitude model above says *what*
the morph should sound like; the waveform is built from real signal, not from
constructed phases: every source is warped onto the output timeline in the
time domain (WSOLA: 1024-sample Hann blocks re-picked within ±1.5 periods for
waveform continuity — ratio 1 is an exact copy), the warped sources are summed
with the weights, and the mix's MCLT phases are kept while its magnitudes are
replaced by the model's (gain capped at +20 dB).  Four Griffin-Lim iterations
(synthesize → re-analyse → re-impose the target magnitude) then make magnitude
and phase agree.  Because the phase always comes from a coherent mix of the
actual sources, attacks are real attacks and the result never has the
phase-vocoder "underwater" quality.  `synth="pv"` keeps the older phase-vocoder
construction (mean instantaneous frequency, peak-locked lobes) for comparison.

Identity weights (`[1,0,0]`) reproduce the source exactly (waveform correlation 1.000); a source morphed with a 7 ms shifted copy of itself comes back at 0.985; the level, duration and per-harmonic balance of a midpoint sit
between the parents (`tests/test_babymaker.py`).

## In the browser

`web/lib/babymaker/` is a port of `morph.py` to TypeScript, in the
configuration the demo runs: `synth="mix"`, `env_mode="log"`, `oversample=2`,
`gl_iters=2`. The DCT-IV and the cepstral DCTs go through a radix-2 FFT
(`fft.ts`, `dct.ts`), the MCLT is in `mdct.ts`, the time warp is the same
WSOLA search (`wsola.ts`), and `morph.worker.ts` runs it all off the main
thread.

What it leaves out: the optimal-transport barycentre for the formant band.
The browser interpolates the envelope in plain dB. Between instruments as
different as a harp and an oboe that costs some quality; it is the first thing
to port next.

The port is checked against Python renders of the same six sources
(`reference/`, written by `tools/prep-babies.py`): ten Dirichlet-random six-way
weight vectors, compared by waveform correlation at zero lag. All ten currently
come out at 1.0000 (the test requires > 0.99), and a one-hot weight returns the
source exactly.

## Tests

```bash
python3 -m pytest tests/test_babymaker.py -q     # 12 tests; the sf2 one needs FluidR3_GM + TimGM6mb in /usr/share/sounds/sf2
cd web && npm test                               # DCT/MDCT/MIDI + the Python parity test
```

## Rebuilding the demo's instruments

```bash
SSO="/path/to/Sonatina Symphonic Orchestra/Samples" python3 tools/prep-babies.py
```

Takes the SSO sample nearest E5 for each instrument, cuts it to 1.7 s, runs
`babymaker random --sources-too` on the six, writes the prepared sources to
`web/public/media/` (16-bit, with `manifest.json`) and the Python reference
renders to `reference/`. Rerunning it reproduces the shipped audio sample for
sample.

## Output

`render` writes one float32 WAV; `grid` / `random` write
`key<KKK>_w<weights>.wav` files, `key<KKK>_morph.json` (sources, measured pitch
errors, config, every render's weights / duration / level) and `index.json`.
Weights outside the simplex are accepted (extrapolation) but not guarded.

## Config (`MorphConfig`)

`frame` (0 = auto), `oversample` (hop = M / oversample), `cep_frac`, `cep_min/max`,
`env_iters`, `floor_db`, `fine_gamma`, `env_mode` (`ot` | `log`), `ot_log_freq`,
`ot_tilt_q`, `ot_q_frac`, `env_smooth_s`, `ot_grid`, `dtw_level_w`, `dtw_cep_w`, `dtw_ncep`,
`dtw_offdiag_penalty`, `dtw_smooth_s`, `synth` (`mix` | `pv`), `max_stretch`,
`attack_s`, `wsola_frame`, `mix_gain_max_db`, `gl_iters`, `device`, `fine_align`
(experimental), `phase_lock`, `onset_db`, `level_floor_db`, `normalize_weights`.

## Notes / limits

- Mono. Sources are aligned to the key of the morph; use sources within an
  octave or so of the key (large resampling ratios shift formants).
- The morph is per key. For a playable instrument build one morph per key
  (`--key 36 48 60 72 84`).
- Unpitched sources (drums, noise) get no pitch correction and morph on
  envelope / level / fine structure only.
- A morph between very different instruments is limited by its *target*: the
  blend of two magnitude spectra whose partials do not coincide is not the
  spectrum of any signal, so 2–3 dB of magnitude error remain on such partials
  whatever the phase reconstruction (see `fine_align` for an experimental
  partial-displacement fix).
## Performance

Measured on a 3-source C4 morph (3.5 s sources, 44.1 kHz):

| | laptop (Ryzen 8845HS), numpy | desktop (Core Ultra 265K), numpy | desktop, RTX 5060 (`--device cuda`) |
|---|---|---|---|
| analysis, 3 sources | 0.5 s | 0.35 s | **0.03 s** |
| render, one weight vector | 0.27 s | 0.13 s | **0.008 s** |

**GPU backend** (`gpu.py`, used automatically when torch sees a CUDA device;
`--device cpu` forces numpy, `--device torch` runs the torch code on the CPU
for testing).  Same maths, batched: MDCT/MDST/IMDCT as a zero-padded complex
FFT with DCT-IV twiddles (one cuFFT call over all frames), cepstral smoothing
as a projection on the first q DCT-II basis vectors (two thin matrix products,
which also makes the true-envelope iterations cheap), the optimal-transport
barycentre with batched `searchsorted` interpolation over all frames at once,
and a batched WSOLA: block positions are predicted pitch-synchronously (each
block starts at the sample near its nominal position whose phase matches the
output position) and refined in two parallel rounds of FFT cross-correlation
against the predicted neighbour, instead of the sequential search.  On real
morphs the GPU and CPU renders match to 0.99 spectrogram correlation with the
same HNR and harmonic-wobble figures; a one-hot render is exact on both.  Only
the DTW alignment and the ~kB of timeline bookkeeping stay on the CPU.  The
`pv` synthesis and `fine_align` use the numpy path.

What keeps the CPU path fast:

- MDCT / MDST / IMDCT run as a DCT-IV of the TDAC-folded block (O(M log M),
  `scipy.fft`), not as a dense 2M×M matrix product;
- the DTW core and the WSOLA block search are `numba` kernels (cached on disk;
  the first run after an edit pays ~0.2 s of compilation); WSOLA searches every
  4th lag and then refines;
- the optimal-transport barycentre is solved on every 4th–5th frame and
  interpolated (its output is smoothed over 80 ms anyway);
- per-source features are stored as float32;
- an sf2 source is rendered for all requested keys with one renderer call, so a
  300 MB soundfont is loaded once per source, not once per key;
- `grid` / `random` render the weight vectors of a key in parallel forked
  workers (`--jobs`, default half the cores; the analysed morph is inherited,
  nothing is pickled).  Output is bit-identical to a serial run.

## Credits

- Samples in `web/public/media/` and `reference/`: [Sonatina Symphonic
  Orchestra](https://github.com/peastman/sso) by Mattias Westlund,
  CC Sampling Plus 1.0.
- `render/tsf.h`: [TinySoundFont](https://github.com/schellingb/TinySoundFont)
  by Bernhard Schelling, MIT.
- True envelope: A. Röbel and X. Rodet, "Efficient spectral envelope
  estimation and its application to pitch shifting and envelope preservation",
  DAFx 2005.
- `web/public/media/alla-turca.mid`: a loop from Mozart's Rondo alla Turca
  (K. 331).
