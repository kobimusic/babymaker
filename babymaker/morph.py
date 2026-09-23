"""N-way morphing of prepared sources in the MCLT (lapped DCT) domain.

Per source, every frame is split into
  * level          – frame RMS in dB                       (temporal envelope)
  * envelope       – DCT-cepstral true envelope of the frame's spectral shape (timbre / formants)
  * fine structure – shape / envelope: the harmonic comb and noise detail
  * phase and per-bin instantaneous frequency (from the MCLT phase advance)
Sources are aligned in time by DTW on (level, low-order cepstrum) against source 0.
For a weight vector w (one entry per source) the morph is
  * time:      the output timeline is the w-average of the aligned timelines, so attack /
               decay / release lengths interpolate instead of cross-fading;
  * level:     w-average in dB (decay rates interpolate);
  * envelope:  1-D optimal-transport (Wasserstein) barycentre on a log-frequency axis:
               a formant at 500 Hz in A and 1 kHz in B lands near 700 Hz at w = ½ instead of
               becoming two half-height bumps; the envelope's total energy interpolates in dB;
  * fine:      w-average in the loudness-like domain |a|^gamma (gamma = 0.6; 0 = dB);
  * phase:     phase-vocoder propagation from the magnitude-weighted mean instantaneous
               frequency, identity phase locking around spectral peaks, reset on onsets.
The morphed MDCT (magnitude * cos(phase)) is overlap-added back to audio.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np

from .lapped import MDCT, cepstrum, lifter, true_envelope
from .sources import Source, midi_to_hz

try:
    import numba as _nb
except Exception:  # pragma: no cover
    _nb = None


# ----------------------------------------------------------------------------- config

@dataclass
class MorphConfig:
    sr: int = 44100
    frame: int = 0               # MDCT M (window 2M, bins M); 0 = auto from pitch (1024 / 2048)
    oversample: int = 4          # hop = M / oversample
    cep_frac: float = 0.3        # cepstral cutoff q = cep_frac * sr / f0 (below the harmonic ripple at sr/f0)
    cep_min: int = 12
    cep_max: int = 160
    env_iters: int = 30
    floor_db: float = -100.0     # magnitude floor (below the source's peak bin)
    fine_gamma: float = 0.6      # fine-structure interpolation domain exponent; 0 = log
    env_mode: str = "ot"         # "ot" (optimal transport) or "log" (cepstral / dB-linear)
    ot_log_freq: bool = True
    ot_grid: int = 1024
    dtw_level_w: float = 0.12    # DTW feature scale for dB level (1 unit ~ 8 dB)
    dtw_cep_w: float = 1.0
    dtw_ncep: int = 12
    dtw_offdiag_penalty: float = 0.15
    dtw_smooth_s: float = 0.08   # Gaussian smoothing of the time maps (s): no abrupt warp-rate changes
    ot_tilt_q: int = 3           # cepstral orders below this = "tilt", dB-interpolated
    ot_q_frac: float = 0.10      # cepstral orders in [ot_tilt_q, ot_q_frac*sr/f0) = broad formants, transported;
                                 # orders above (up to the envelope cutoff) = per-harmonic detail, dB-interpolated
    env_smooth_s: float = 0.08   # temporal smoothing (s) of the transported part
    phase_lock: bool = True
    onset_db: float = 8.0        # level rise per hop that triggers a phase reset
    level_floor_db: float = -80.0
    normalize_weights: bool = True
    synth: str = "mix"           # "mix": phases from a time-aligned coherent mix of the sources (+ Griffin-Lim
                                 # refinement); "pv": phase-vocoder construction from averaged phases
    max_stretch: float = 2.5     # slope limit of each source's time warp (mix synth)
    attack_s: float = 0.1        # sources play at natural speed for this long after the onset (mix synth)
    wsola_frame: int = 1024      # time-domain warp block (samples)
    mix_gain_max_db: float = 20.0  # cap on |target| / |mix| when taking the mix's phase
    gl_iters: int = 4            # magnitude-consistency (Griffin-Lim) iterations after the mix phase
    device: str = "auto"         # "auto" (cuda if available), "cuda", "cpu" (numpy), "torch" (torch on cpu)
    fine_align: bool = False     # (experimental) move each source's partial to the w-mean position within its harmonic
                                 # slot before blending, so blended partials are single lobes

    def to_dict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------------------- DTW

def _dtw_py(cost: np.ndarray, pen: float):
    n, m = cost.shape
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    step = np.zeros((n, m), np.int8)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = cost[i - 1, j - 1]
            a, b, d = D[i - 1, j - 1], D[i - 1, j] + pen, D[i, j - 1] + pen
            if a <= b and a <= d:
                D[i, j], step[i - 1, j - 1] = a + c, 0
            elif b <= d:
                D[i, j], step[i - 1, j - 1] = b + c, 1
            else:
                D[i, j], step[i - 1, j - 1] = d + c, 2
    return D, step


if _nb is not None:
    _dtw_core = _nb.njit(cache=True)(_dtw_py)
else:  # pragma: no cover
    _dtw_core = _dtw_py


def dtw_path(a: np.ndarray, b: np.ndarray, pen: float = 0.1) -> np.ndarray:
    """DTW between feature sequences a (n, d) and b (m, d); returns path (L, 2) of index pairs."""
    diff = a[:, None, :] - b[None, :, :]
    cost = np.sqrt(np.sum(diff * diff, axis=-1))
    D, step = _dtw_core(cost.astype(np.float64), float(pen))
    i, j = a.shape[0] - 1, b.shape[0] - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        s = step[i, j]
        if s == 0 and i > 0 and j > 0:
            i, j = i - 1, j - 1
        elif s == 1 and i > 0:
            i -= 1
        elif j > 0:
            j -= 1
        else:
            i -= 1
        path.append((i, j))
    return np.array(path[::-1], dtype=np.int64)


def path_to_map(path: np.ndarray, n_ref: int, n_other: int) -> np.ndarray:
    """Monotone map g[t_ref] -> fractional frame of the other sequence."""
    sums = np.zeros(n_ref)
    cnt = np.zeros(n_ref)
    np.add.at(sums, path[:, 0], path[:, 1])
    np.add.at(cnt, path[:, 0], 1)
    g = sums / np.maximum(cnt, 1)
    g[0] = 0.0                         # onsets coincide by construction (sources are onset-trimmed)
    g = np.maximum.accumulate(g)
    return np.clip(g, 0, n_other - 1)


def smooth_map(g: np.ndarray, sigma_frames: float) -> np.ndarray:
    """Gaussian-smooth a monotone time map, keeping both end points and monotonicity."""
    n = g.shape[0]
    if sigma_frames < 0.5 or n < 3:
        return g
    r = min(int(3 * sigma_frames), n - 1)     # a map shorter than the kernel: reflect only what exists
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_frames) ** 2)
    k /= k.sum()
    padded = np.concatenate([2 * g[0] - g[r:0:-1], g, 2 * g[-1] - g[-2:-r - 2:-1]])   # odd reflection
    gs = np.convolve(padded, k, mode="valid")
    gs = gs + np.linspace(g[0] - gs[0], g[-1] - gs[-1], n)                          # pin the ends
    gs = np.maximum.accumulate(gs)
    return np.clip(gs, g.min(), g.max())


def _wsola_py(xp, csum, src_of_out, n_out, N, H, dmax, win):
    y = np.zeros(n_out + N)
    nx = src_of_out.shape[0]
    prev = -1
    for s0 in range(0, n_out, H):
        si = s0 if s0 < nx else nx - 1
        nom = int(round(src_of_out[si]))
        if prev < 0:
            p = nom
        else:
            nat = prev + H                                      # natural continuation
            lo = nom - dmax
            if lo < 0:
                lo = 0
            hi = nom + dmax
            # coarse search every 4th lag on a grid that contains the natural continuation
            # (so ratio 1 is always exact), then refine around the best
            best = -1e300
            p = nat
            first = nat - ((nat - lo) // 4) * 4
            for d in range(first, hi + 1, 4):
                dot = 0.0
                for n in range(N):
                    dot += xp[d + n] * xp[nat + n]
                val = dot / (np.sqrt(csum[d + N] - csum[d]) + 1e-9)
                if val > best:
                    best = val
                    p = d
            c0 = p
            for d in range(max(lo, c0 - 3), min(hi, c0 + 3) + 1):
                if d == c0:
                    continue
                dot = 0.0
                for n in range(N):
                    dot += xp[d + n] * xp[nat + n]
                val = dot / (np.sqrt(csum[d + N] - csum[d]) + 1e-9)
                if val > best:
                    best = val
                    p = d
        for n in range(N):
            y[s0 + n] += win[n] * xp[p + n]
        prev = p
    return y


if _nb is not None:
    _wsola_core = _nb.njit(cache=True, fastmath=True)(_wsola_py)
else:  # pragma: no cover
    _wsola_core = _wsola_py


def wsola_warp(x: np.ndarray, src_of_out: np.ndarray, n_out: int, period: float, frame: int = 1024) -> np.ndarray:
    """Time-domain warp: output sample s is taken from around source sample src_of_out[s].
    WSOLA - each Hann block is picked within +-1.5 periods of its nominal position so that it
    continues the previous block's waveform (max normalised cross-correlation); 50 % overlap-add.
    Ratio 1 reproduces x exactly; moderate ratios keep the waveform (and hence the phases) intact."""
    N = frame
    H = N // 2
    win = np.hanning(N + 1)[:-1]
    dmax = int(np.clip(1.5 * period, 32, H))
    xp = np.concatenate([np.asarray(x, np.float64), np.zeros(N + 2 * dmax + H)])
    csum = np.concatenate([[0.0], np.cumsum(xp * xp)])
    pos = np.clip(np.asarray(src_of_out, np.float64), 0, len(x) - 1)
    y = _wsola_core(xp, csum, pos, int(n_out), N, H, dmax, win)
    return y[:n_out]


# ----------------------------------------------------------------------------- analysis

@dataclass
class _Analysis:
    name: str
    level_db: np.ndarray     # (T,)
    env: np.ndarray          # (T, M) log envelope of the unit-RMS shape
    fine: np.ndarray         # (T, M) log fine structure
    env_lin: np.ndarray      # (T, M) envelope minus its broad formant band (tilt + detail): dB-interpolated
    broad: np.ndarray        # (T, M) the broad formant band (cepstral orders [ot_tilt_q, q_ot)): transported
    fine_pow: np.ndarray     # (T, M) exp(fine_gamma * fine) (or fine itself when fine_gamma == 0)
    phase: np.ndarray | None # (T, M)                         (synth="pv" only)
    omega: np.ndarray | None # (T, M) instantaneous frequency (synth="pv" only)
    shape: np.ndarray | None # (T, M) linear unit-RMS magnitude (synth="pv" only)
    dtw_feat: np.ndarray     # (T, d)
    n_samples: int
    peak_rms_in: float


class Morpher:
    """Analyse N prepared sources once; `render(weights)` synthesises any point of the N-dim morph."""

    def __init__(self, sources: list[Source], cfg: MorphConfig | None = None):
        if len(sources) < 1:
            raise ValueError("need at least one source")
        self.cfg = cfg or MorphConfig()
        self.sources = sources
        sr = self.cfg.sr
        for s in sources:
            if s.sr != sr:
                raise ValueError(f"source {s.name} is at {s.sr} Hz, morph is at {sr} Hz")
        self.note = float(np.mean([s.note for s in sources]))
        self.f0 = midi_to_hz(self.note)
        M = self.cfg.frame or (2048 if self.f0 < 110 else 1024)
        self.mdct = MDCT(M, M // self.cfg.oversample)
        self.q = int(np.clip(self.cfg.cep_frac * sr / self.f0, self.cfg.cep_min, self.cfg.cep_max))
        self.q_ot = int(np.clip(self.cfg.ot_q_frac * sr / self.f0, self.cfg.ot_tilt_q + 2, self.q))
        # harmonic slots: bin k belongs to slot round(f_k / f0); partials of different sources that
        # sit in the same slot are the "same" partial and get displaced to a common position
        self.slot_id = np.rint(self.mdct.bin_hz(sr) / self.f0).astype(int)
        self.n_slots = int(self.slot_id.max()) + 1
        self._slot_onehot = np.zeros((M, self.n_slots))
        self._slot_onehot[np.arange(M), self.slot_id] = 1.0
        self._slot_center = (self._slot_onehot * np.arange(M)[:, None]).sum(0) / np.maximum(self._slot_onehot.sum(0), 1)
        self._slot_half = 0.5 * self.f0 / (sr / (2 * M))            # half a harmonic spacing, in bins
        from . import gpu as _gpu
        self.dev = None if (self.cfg.synth == "pv" or self.cfg.fine_align) else _gpu.pick_device(self.cfg.device)
        if self.dev is not None:
            self.tl = _gpu.TorchLapped(M, self.mdct.hop, self.dev, q_max=max(self.q, self.cfg.dtw_ncep + 1) + 1)
            self.analyses = [self._analyse_torch(s) for s in sources]
        else:
            self.analyses = [self._analyse(s) for s in sources]
        self._align()

    # -- analysis of one source
    def _analyse(self, src: Source) -> _Analysis:
        cfg, m = self.cfg, self.mdct
        mag, phase = m.mclt(src.audio)
        T, M = mag.shape
        rms = np.sqrt(np.mean(mag ** 2, axis=1))                      # per-frame RMS (Parseval-ish)
        level_db = 20 * np.log10(np.maximum(rms, 1e-12))
        peak_db = level_db.max()
        level_db = np.maximum(level_db, peak_db + cfg.level_floor_db)
        shape = mag / np.maximum(rms, 1e-12)[:, None]
        floor = shape.max() * 10 ** (cfg.floor_db / 20)
        log_shape = np.log(np.maximum(shape, floor))
        env = true_envelope(log_shape, self.q, iters=cfg.env_iters)
        fine = log_shape - env
        # instantaneous frequency from the phase advance between consecutive frames
        expected = m.bin_omega * m.hop
        dphi = np.diff(phase, axis=0) - expected[None, :]
        dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
        omega = np.empty_like(phase)                                  # omega[t]: advance from t-1 to t
        omega[1:] = m.bin_omega[None, :] + dphi / m.hop
        omega[0] = omega[1] if T > 1 else m.bin_omega
        # envelope split by cepstral order (done once here, not per render)
        cep = cepstrum(env)
        tilt = lifter(cep, cfg.ot_tilt_q) if cfg.ot_tilt_q > 0 else np.zeros_like(env)
        broad = lifter(cep, self.q_ot) - tilt
        env_lin = env - broad
        fine_pow = np.exp(cfg.fine_gamma * fine) if cfg.fine_gamma > 0 else fine
        # DTW features: level + low-order cepstrum of the envelope
        feat = np.concatenate([(level_db - peak_db)[:, None] * cfg.dtw_level_w,
                               cep[:, 1:1 + cfg.dtw_ncep] * cfg.dtw_cep_w], axis=1)
        f32 = np.float32
        return _Analysis(src.name, level_db, env.astype(f32), fine.astype(f32), env_lin.astype(f32),
                         broad.astype(f32), fine_pow.astype(f32), phase.astype(f32),
                         omega.astype(f32), shape.astype(f32), feat, len(src.audio),
                         float(src.meta.get("peak_rms_in", 1.0)))

    def _analyse_torch(self, src: Source) -> _Analysis:
        """Same features as _analyse, computed with the torch backend; the (T, M) features stay
        on the device as float32 tensors (the pv-only phase features are not computed)."""
        import torch
        cfg, tl = self.cfg, self.tl
        x = torch.as_tensor(np.asarray(src.audio, np.float32), device=self.dev)
        X, Y = tl.analyze(x)
        mag = torch.hypot(X, Y)
        rms = torch.sqrt((mag * mag).mean(1))
        level = 20 * torch.log10(rms.clamp_min(1e-12))
        peak_db = float(level.max())
        level_db = np.maximum(level.cpu().numpy().astype(np.float64), peak_db + cfg.level_floor_db)
        shape = mag / rms.clamp_min(1e-12)[:, None]
        floor = float(shape.max()) * 10 ** (cfg.floor_db / 20)
        log_shape = torch.log(shape.clamp_min(floor))
        env = tl.true_envelope(log_shape, self.q, iters=cfg.env_iters)
        fine = log_shape - env
        tilt = tl.smooth(env, cfg.ot_tilt_q) if cfg.ot_tilt_q > 0 else torch.zeros_like(env)
        broad = tl.smooth(env, self.q_ot) - tilt
        env_lin = env - broad
        fine_pow = torch.exp(cfg.fine_gamma * fine) if cfg.fine_gamma > 0 else fine
        cep = tl.cepstrum(env, 1 + cfg.dtw_ncep)[:, 1:].cpu().numpy().astype(np.float64)
        feat = np.concatenate([(level_db - peak_db)[:, None] * cfg.dtw_level_w, cep * cfg.dtw_cep_w], axis=1)
        return _Analysis(src.name, level_db, env, fine, env_lin, broad, fine_pow, None, None, None, feat,
                         len(src.audio), float(src.meta.get("peak_rms_in", 1.0)))

    # -- time alignment against source 0
    def _align(self):
        ref = self.analyses[0]
        n_ref = ref.level_db.shape[0]
        self.maps = []
        for a in self.analyses:
            if a is ref:
                self.maps.append(np.arange(n_ref, dtype=np.float64))
                continue
            path = dtw_path(ref.dtw_feat, a.dtw_feat, self.cfg.dtw_offdiag_penalty)
            g = path_to_map(path, n_ref, a.level_db.shape[0])
            self.maps.append(smooth_map(g, self.cfg.dtw_smooth_s * self.cfg.sr / self.mdct.hop))
        self.maps = np.stack(self.maps)      # (N, T_ref)

    # -- helpers
    @property
    def n(self) -> int:
        return len(self.sources)

    def _weights(self, w) -> np.ndarray:
        w = np.asarray(w, np.float64).ravel()
        if w.shape[0] != self.n:
            raise ValueError(f"expected {self.n} weights, got {w.shape[0]}")
        if self.cfg.normalize_weights:
            s = w.sum()
            if abs(s) < 1e-9:
                raise ValueError("weights sum to zero")
            w = w / s
        return w

    @staticmethod
    def _interp_rows(arr: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """Linear interpolation of rows of arr (T, ...) at fractional positions pos (J,)."""
        i0 = np.clip(np.floor(pos).astype(int), 0, arr.shape[0] - 1)
        i1 = np.clip(i0 + 1, 0, arr.shape[0] - 1)
        f = (pos - i0)[:, None] if arr.ndim > 1 else (pos - i0)
        return arr[i0] * (1 - f) + arr[i1] * f

    @staticmethod
    def _nearest_rows(arr: np.ndarray, pos: np.ndarray) -> np.ndarray:
        return arr[np.clip(np.rint(pos).astype(int), 0, arr.shape[0] - 1)]

    def timeline(self, w) -> tuple[np.ndarray, np.ndarray]:
        """For weights w: (n_out_frames, positions (N, J)) – source frame position for each output frame."""
        w = self._weights(w)
        n_ref = self.maps.shape[1]
        tau = np.einsum("i,it->t", w, self.maps)                     # morphed time of each ref frame
        tau = np.maximum.accumulate(tau)
        tau = tau + np.arange(n_ref) * 1e-6                          # strictly increasing
        J = int(np.floor(tau[-1])) + 1
        t_ref = np.interp(np.arange(J), tau, np.arange(n_ref))
        pos = np.stack([np.interp(t_ref, np.arange(n_ref), g) for g in self.maps])
        return J, pos

    # -- envelope morph
    def _ot_barycenter(self, broad: np.ndarray, w: np.ndarray) -> np.ndarray:
        """broad (N, J, M): each source's broad formant band (log amplitude, cepstral orders
        [ot_tilt_q, q_ot)) -> (J, M) Wasserstein barycentre per frame.  The band is treated as a
        distribution over log-frequency, so formant peaks move to interpolated positions
        instead of fading; its total energy interpolates in dB.  Transporting only the broad
        band keeps the transport stable from frame to frame (moving per-harmonic bumps by
        fractions of a bin would wobble harmonic levels by ~1 dB)."""
        N, J, M = broad.shape
        cfg = self.cfg
        p = np.exp(2.0 * broad.astype(np.float64))                    # power of the formant band
        tot = p.sum(axis=2)                                          # (N, J)
        cdf = np.cumsum(p / tot[:, :, None], axis=2)
        cdf = np.concatenate([np.zeros((N, J, 1)), cdf], axis=2)     # at bin edges 0..M
        edges = np.arange(M + 1, dtype=np.float64)
        z_edges = np.log(edges + 2.0) if cfg.ot_log_freq else edges
        grid = (np.arange(cfg.ot_grid) + 0.5) / cfg.ot_grid          # uniform quantile levels
        eps = np.arange(M + 1) * 1e-12
        dens = np.empty((J, M))
        for j in range(J):
            cs = [np.maximum.accumulate(cdf[i, j] + eps) for i in range(N)]   # strictly increasing
            # evaluate at every source's own CDF knots too: a one-hot w then reproduces that
            # source's band exactly, and low-mass regions keep their resolution
            levels = np.unique(np.concatenate([grid] + cs))
            zq = np.zeros(levels.shape[0])
            for i in range(N):
                zq += w[i] * np.interp(levels, cs[i], z_edges)
            zq = np.maximum.accumulate(zq)
            F = np.interp(z_edges, zq, levels, left=0.0, right=1.0)
            dens[j] = np.diff(F)
        tot_m = np.exp(np.einsum("i,ij->j", w, np.log(tot)))          # (J,)
        broad_m = 0.5 * np.log(np.maximum(dens * tot_m[:, None], 1e-30))
        return lifter(cepstrum(broad_m), self.q_ot)                   # back into its cepstral band

    def _smooth_time(self, x: np.ndarray) -> np.ndarray:
        """Moving average along frames (axis 0) over env_smooth_s."""
        n = int(round(self.cfg.env_smooth_s * self.cfg.sr / self.mdct.hop))
        if n < 2 or x.shape[0] < 3:
            return x
        c = np.cumsum(np.concatenate([np.zeros((1, x.shape[1])), x], axis=0), axis=0)
        half = n // 2
        idx = np.arange(x.shape[0])
        lo = np.clip(idx - half, 0, x.shape[0])
        hi = np.clip(idx + half + 1, 0, x.shape[0])
        return (c[hi] - c[lo]) / (hi - lo)[:, None]

    # -- rendering
    def _limit_warp(self, pos: np.ndarray) -> np.ndarray:
        """Clamp each source's warp slope to [1/max_stretch, max_stretch] frames per output frame,
        and to exactly 1 during the attack, so that no transient is ever stretched or repeated."""
        cfg, m = self.cfg, self.mdct
        out = pos.copy()
        n_att = int(round(cfg.attack_s * cfg.sr / m.hop))
        smax = max(1.0, cfg.max_stretch)
        for i in range(out.shape[0]):
            last = self.analyses[i].level_db.shape[0] - 1
            out[i, 0] = 0.0
            for j in range(1, out.shape[1]):
                s_ = 1.0 if j <= n_att else smax
                out[i, j] = min(max(pos[i, j], out[i, j - 1] + 1.0 / s_), out[i, j - 1] + s_, last)
        return out

    def _align_partials(self, fines: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Per frame and harmonic slot, shift each source's fine structure (log) along frequency so
        that its slot centroid lands on the w-mean of the sources' centroids.  Partials at slightly
        different frequencies (inharmonicity, detuning, vibrato phase) then blend into one lobe at
        the interpolated frequency instead of a double hump that no signal can realise."""
        N, J, M = fines.shape
        p = np.exp(2.0 * fines)                                       # power
        mass = p @ self._slot_onehot                                  # (N, J, S)
        moment = (p * np.arange(M)) @ self._slot_onehot
        c = np.where(mass > 1e-9, moment / np.maximum(mass, 1e-30), self._slot_center)
        c_t = np.einsum("i,ijs->js", w, c)                            # target centroid per slot
        delta = np.clip(c_t[None] - c, -self._slot_half, self._slot_half)   # (N, J, S)
        d = delta[:, :, self.slot_id]                                 # per-bin shift (N, J, M)
        q = np.arange(M)[None, None, :] - d
        q0 = np.floor(q)
        frac = q - q0
        i0 = np.clip(q0.astype(int), 0, M - 1)
        i1 = np.clip(i0 + 1, 0, M - 1)
        return np.take_along_axis(fines, i0, axis=2) * (1 - frac) + np.take_along_axis(fines, i1, axis=2) * frac

    def _coherent_mix(self, w: np.ndarray, pos: np.ndarray, n_out: int) -> np.ndarray:
        """Weighted sum of the sources, each warped in the time domain onto the output timeline."""
        m = self.mdct
        J = pos.shape[1]
        out_centers = np.arange(J) * m.hop - m.M
        s_idx = np.arange(n_out)
        mix = np.zeros(n_out)
        period = self.cfg.sr / self.f0
        for i, src in enumerate(self.sources):
            if w[i] == 0:
                continue
            src_of_out = np.interp(s_idx, out_centers, pos[i] * m.hop - m.M)
            src_of_out = np.clip(src_of_out, 0, len(src.audio) - 1)
            mix += w[i] * wsola_warp(src.audio, src_of_out, n_out, period, self.cfg.wsola_frame)
        return mix

    def render(self, weights, return_info: bool = False):
        """Synthesise the morph for `weights` (length N). Returns float64 mono audio at cfg.sr."""
        cfg, m = self.cfg, self.mdct
        w = self._weights(weights)
        N, A, M = self.n, self.analyses, m.M
        J, pos = self.timeline(w)
        n_out = int(round(sum(w[i] * A[i].n_samples for i in range(N))))
        n_out = max(m.hop, n_out)
        if cfg.synth == "mix":
            pos = self._limit_warp(pos)
            J2 = m.n_frames(n_out)                           # the analysis grid of an n_out signal
            if J2 > J:
                pos = np.concatenate([pos, np.repeat(pos[:, -1:], J2 - J, axis=1)], axis=1)
            pos, J = pos[:, :J2], J2
        else:
            n_out = min(n_out, (J - 1) * m.hop + 2 * M - m.pad)
        if self.dev is not None:
            y = self._render_torch(w, J, pos, n_out)
            return self._finish(y, w, J, n_out, return_info)
        # gather time-warped features (precomputed per source; only gathers + weighted sums here)
        level = np.zeros(J)
        env = np.zeros((J, M))
        for i in range(N):
            level += w[i] * self._interp_rows(A[i].level_db, pos[i])
            env += w[i] * self._interp_rows(A[i].env_lin, pos[i])
        if cfg.env_mode == "ot" and N > 1:
            # the transported band is smoothed over env_smooth_s afterwards, so solving the
            # transport on every `stride`-th frame and interpolating in between loses nothing
            stride = max(1, int(round(cfg.env_smooth_s * cfg.sr / m.hop / 3)))
            js = np.unique(np.concatenate([np.arange(0, J, stride), [J - 1]]))
            broad_s = np.stack([self._interp_rows(A[i].broad, pos[i][js]) for i in range(N)])
            broad_m = self._ot_barycenter(broad_s, w)
            if js.shape[0] < J:
                broad_m = self._interp_rows(broad_m, np.interp(np.arange(J), js, np.arange(js.shape[0])))
            env += self._smooth_time(broad_m)
        else:
            for i in range(N):
                env += w[i] * self._interp_rows(A[i].broad, pos[i])
        # fine structure
        g = cfg.fine_gamma
        if cfg.fine_align and N > 1:
            fines = np.stack([self._interp_rows(A[i].fine, pos[i]) for i in range(N)])
            fines = self._align_partials(fines, w)
            fine = (np.log(np.maximum(np.einsum("i,ijk->jk", w, np.exp(fines * g)), 1e-12)) / g if g > 0
                    else np.einsum("i,ijk->jk", w, fines))
        else:
            fp = np.zeros((J, M))
            for i in range(N):
                fp += w[i] * self._interp_rows(A[i].fine_pow, pos[i])
            fine = np.log(np.maximum(fp, 1e-12)) / g if g > 0 else fp
        mag = np.exp(env + fine) * (10 ** (level / 20))[:, None]
        if cfg.synth == "mix":
            mix = self._coherent_mix(w, pos, n_out)
            Xm, Ym = m.analyze(mix)
            mag_mix = np.hypot(Xm, Ym)
            gmax = 10 ** (cfg.mix_gain_max_db / 20)
            X = Xm * np.minimum(mag / np.maximum(mag_mix, 1e-12), gmax)   # target magnitude, mix phase
            for _ in range(cfg.gl_iters):                                 # make magnitude and phase agree
                Xr, Yr = m.analyze(m.synthesize(X, n_out))
                r = np.hypot(Xr, Yr)
                X = Xr * (mag / np.maximum(r, 1e-12))
            y = m.synthesize(X, n_out)
        else:
            omegas = np.empty((N, J, M))
            phases = np.empty((N, J, M))
            shapes = np.empty((N, J, M))
            for i in range(N):
                omegas[i] = self._interp_rows(A[i].omega, pos[i])
                phases[i] = self._nearest_rows(A[i].phase, pos[i])
                shapes[i] = self._interp_rows(A[i].shape, pos[i])
            v = np.maximum(w[:, None, None], 0) * shapes
            vs = v.sum(axis=0)
            omega = np.where(vs > 1e-12, (v * omegas).sum(axis=0) / np.maximum(vs, 1e-12), m.bin_omega[None, :])
            phase = self._propagate(mag, omega, phases, v, level, int(np.argmax(w)))
            y = m.from_polar(mag, phase, n_out)
        return self._finish(y, w, J, n_out, return_info)

    def _render_torch(self, w: np.ndarray, J: int, pos: np.ndarray, n_out: int) -> np.ndarray:
        """The default (mix) synthesis on the torch device; returns float64 numpy audio."""
        import torch
        from . import gpu as _gpu
        cfg, tl, A, N, M = self.cfg, self.tl, self.analyses, self.n, self.mdct.M
        dev = self.dev
        posT = torch.as_tensor(pos.astype(np.float32), device=dev)
        level = torch.zeros(J, device=dev)
        env = torch.zeros(J, M, device=dev)
        for i in range(N):
            level += float(w[i]) * _gpu.interp_rows(torch.as_tensor(A[i].level_db.astype(np.float32), device=dev), posT[i])
            env += float(w[i]) * _gpu.interp_rows(A[i].env_lin, posT[i])
        if cfg.env_mode == "ot" and N > 1:
            stride = max(1, int(round(cfg.env_smooth_s * cfg.sr / self.mdct.hop / 3)))
            js = np.unique(np.concatenate([np.arange(0, J, stride), [J - 1]]))
            jsT = torch.as_tensor(js, device=dev)
            broad_s = torch.stack([_gpu.interp_rows(A[i].broad, posT[i][jsT]) for i in range(N)])
            broad_m = _gpu.ot_barycenter(broad_s, w, self.q_ot, tl, cfg.ot_log_freq, cfg.ot_grid)
            if js.shape[0] < J:
                fi = torch.as_tensor(np.interp(np.arange(J), js, np.arange(js.shape[0])).astype(np.float32), device=dev)
                broad_m = _gpu.interp_rows(broad_m, fi)
            env += _gpu.smooth_time(broad_m, int(round(cfg.env_smooth_s * cfg.sr / self.mdct.hop)))
        else:
            for i in range(N):
                env += float(w[i]) * _gpu.interp_rows(A[i].broad, posT[i])
        g = cfg.fine_gamma
        fp = torch.zeros(J, M, device=dev)
        for i in range(N):
            fp += float(w[i]) * _gpu.interp_rows(A[i].fine_pow, posT[i])
        fine = torch.log(fp.clamp_min(1e-12)) / g if g > 0 else fp
        mag = torch.exp(env + fine) * (10 ** (level / 20))[:, None]
        mix = self._coherent_mix_torch(w, pos, n_out)
        Xm, Ym = tl.analyze(mix)
        mag_mix = torch.hypot(Xm, Ym)
        gmax = 10 ** (cfg.mix_gain_max_db / 20)
        X = Xm * torch.minimum(mag / mag_mix.clamp_min(1e-12), torch.tensor(gmax, device=dev))
        for _ in range(cfg.gl_iters):
            Xr, Yr = tl.analyze(tl.synthesize(X, n_out))
            X = Xr * (mag / torch.hypot(Xr, Yr).clamp_min(1e-12))
        return tl.synthesize(X, n_out).cpu().numpy().astype(np.float64)

    def _coherent_mix_torch(self, w: np.ndarray, pos: np.ndarray, n_out: int):
        """_coherent_mix on the device (batched WSOLA)."""
        import torch
        from . import gpu as _gpu
        m, dev = self.mdct, self.dev
        J = pos.shape[1]
        out_centers = np.arange(J) * m.hop - m.M
        s_idx = np.arange(n_out)
        mix = torch.zeros(n_out, device=dev)
        period = self.cfg.sr / self.f0
        for i, src in enumerate(self.sources):
            if w[i] == 0:
                continue
            src_of_out = np.clip(np.interp(s_idx, out_centers, pos[i] * m.hop - m.M), 0, len(src.audio) - 1)
            if not hasattr(src, "_dev_audio") or src._dev_audio.device != dev:
                src._dev_audio = torch.as_tensor(np.asarray(src.audio, np.float32), device=dev)
            mix += float(w[i]) * _gpu.wsola_warp(src._dev_audio, torch.as_tensor(src_of_out.astype(np.float32), device=dev),
                                                n_out, period, self.cfg.wsola_frame)
        return mix

    def _finish(self, y: np.ndarray, w: np.ndarray, J: int, n_out: int, return_info: bool):
        cfg, A = self.cfg, self.analyses
        # level: sources were normalised to peak-RMS 1; restore the dB-interpolated original level
        win = max(1, int(0.02 * cfg.sr))
        cs = np.concatenate([[0.0], np.cumsum(y * y)])
        e = (cs[win:] - cs[:-win]) / win if len(y) > win else cs[-1:] / max(len(y), 1)
        prms = float(np.sqrt(e.max())) + 1e-12
        target = float(np.exp(np.sum(w * np.log(np.maximum([a.peak_rms_in for a in A], 1e-6)))))
        y = y * (target / prms)
        # tidy the tail
        nf = min(len(y), int(0.01 * cfg.sr))
        if nf > 1:
            y[-nf:] *= np.linspace(1, 0, nf)
        if return_info:
            return y, {"weights": w.tolist(), "n_frames": int(J), "n_samples": int(n_out), "peak_rms": target,
                       "duration_s": len(y) / cfg.sr, "synth": cfg.synth,
                       "device": str(self.dev) if self.dev is not None else "numpy"}
        return y

    def _propagate(self, mag: np.ndarray, omega: np.ndarray, phases: np.ndarray, v: np.ndarray,
                   level: np.ndarray, dom: int) -> np.ndarray:
        """Phase-vocoder accumulation with identity phase locking and onset resets.

        Peaks of the morphed magnitude carry the accumulated phase; the bins around a peak
        take the peak's phase plus the lobe offset (phi[k] - phi[peak]) averaged (circularly,
        magnitude-weighted) over the sources -- each source's own lobe structure is coherent,
        the average of raw phases from unrelated sources is not.  At an onset the phases of
        the dominant source are copied whole, which keeps its attack crisp."""
        cfg, m = self.cfg, self.mdct
        J, M = mag.shape
        hop = m.hop
        phase = np.empty_like(mag)
        prev = phases[dom, 0]
        phase[0] = prev
        for j in range(1, J):
            if (level[j] - level[j - 1]) > cfg.onset_db:
                cur = phases[dom, j].copy()
            else:
                cur = prev + omega[j] * hop
                if cfg.phase_lock:
                    row = mag[j]
                    is_peak = np.zeros(M, bool)
                    is_peak[1:-1] = (row[1:-1] > row[:-2]) & (row[1:-1] >= row[2:])
                    is_peak[0] = row[0] > row[1]
                    is_peak[-1] = row[-1] > row[-2]
                    peaks = np.nonzero(is_peak)[0]
                    if peaks.size:
                        mids = (peaks[:-1] + peaks[1:] + 1) // 2
                        owner = peaks[np.searchsorted(mids, np.arange(M), side="right")]
                        ph = phases[:, j, :]                                   # (N, M)
                        z = (v[:, j, :] * np.exp(1j * (ph - ph[:, owner]))).sum(axis=0)
                        off = np.where(np.abs(z) > 1e-12, np.angle(z), ph[dom] - ph[dom][owner])
                        cur = cur[owner] + off
            phase[j] = cur
            prev = cur
        return np.angle(np.exp(1j * phase))

    # -- diagnostics
    def describe(self) -> dict:
        return {"note": self.note, "f0_hz": self.f0, "frame": self.mdct.M, "hop": self.mdct.hop,
                "cepstral_q": self.q, "device": str(self.dev) if self.dev is not None else "numpy",
                "config": self.cfg.to_dict(),
                "sources": [{"name": s.name, "duration_s": s.duration, "frames": int(a.level_db.shape[0]),
                             **{k: v for k, v in s.meta.items() if k not in ("kind",)}}
                            for s, a in zip(self.sources, self.analyses)]}


# ----------------------------------------------------------------------------- weight sets

def simplex_grid(n: int, steps: int) -> np.ndarray:
    """All weight vectors with entries in {0, 1/steps, ..., 1} summing to 1, shape (P, n)."""
    if n == 1:
        return np.ones((1, 1))
    out = []

    def rec(prefix, remaining, k):
        if k == 1:
            out.append(prefix + [remaining])
            return
        for a in range(remaining + 1):
            rec(prefix + [a], remaining - a, k - 1)

    rec([], steps, n)
    return np.array(out, dtype=np.float64) / steps


def random_weights(n: int, count: int, rng: np.random.Generator, alpha: float = 1.0) -> np.ndarray:
    return rng.dirichlet(np.full(n, alpha), size=count)
