"""torch backend for babymaker: the MCLT analysis, the envelope model and the render on a GPU
(or torch-on-CPU for testing).  Same maths as lapped.py / morph.py, batched:

* DCT-IV through a zero-padded complex FFT of length 2M with pre/post twiddles, so the
  MDCT / MDST / IMDCT are cuFFT calls over all frames at once;
* cepstral smoothing (lifter o cepstrum with q coefficients) as a projection on the first q
  DCT-II basis vectors, x @ B_q @ B_q^T - two thin matrix products instead of two full DCTs,
  which also makes the true-envelope iterations cheap;
* the 1-D optimal-transport barycentre with batched torch.searchsorted interpolation over all
  frames at once instead of a Python loop.
The time-domain WSOLA warp stays on the CPU (numba); only the ~150k-sample mix crosses over.
"""
from __future__ import annotations

import math

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None


def available() -> bool:
    return torch is not None and torch.cuda.is_available()


def pick_device(device: str):
    """'auto' -> cuda if available else None (numpy path); 'cuda'; 'torch' = torch on cpu."""
    if torch is None:
        if device in ("cuda", "torch"):
            raise RuntimeError("torch is not installed")
        return None
    if device == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else None
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        return torch.device("cuda")
    if device == "torch":
        torch.set_num_threads(max(1, min(4, (__import__("os").cpu_count() or 4))))   # tiny ops: many threads only add overhead
        return torch.device("cpu")
    return None


class TorchLapped:
    """MDCT / MDST / IMDCT + cepstral projections on a torch device (float32)."""

    def __init__(self, M: int, hop: int, device, q_max: int = 512):
        self.M, self.hop, self.device = M, hop, device
        self.os = M // hop
        self.pad = 2 * M
        n = torch.arange(2 * M, device=device, dtype=torch.float32)
        self.window = torch.sin(math.pi * (n + 0.5) / (2 * M))
        m = torch.arange(M, device=device, dtype=torch.float32)
        self.pre = torch.exp(-1j * math.pi * m / (2 * M)).to(torch.complex64)
        self.post = (math.sqrt(2.0 / M) * torch.exp(-1j * math.pi * (2 * m + 1) / (4 * M))).to(torch.complex64)
        self.alt = torch.where(m.long() % 2 == 0, -1.0, 1.0)                 # -(-1)^k
        # orthonormal DCT-II basis, first q_max vectors: (M, q_max)
        k = torch.arange(q_max, device=device, dtype=torch.float32)
        B = torch.cos(math.pi * (m[:, None] + 0.5) * k[None, :] / M) * math.sqrt(2.0 / M)
        B[:, 0] = math.sqrt(1.0 / M)
        self.B = B

    # -- DCT-IV (orthonormal, self-inverse) along the last axis
    def dct4(self, x):
        M = self.M
        xt = torch.zeros(x.shape[:-1] + (2 * M,), dtype=torch.complex64, device=self.device)
        xt[..., :M] = x.to(torch.float32) * self.pre
        Fx = torch.fft.fft(xt, dim=-1)[..., :M]
        return (self.post * Fx).real

    def n_frames(self, n_samples: int) -> int:
        return int(math.ceil((self.pad + n_samples + self.M) / self.hop)) + 1

    def frames(self, x):
        M, hop = self.M, self.hop
        T = self.n_frames(x.shape[0])
        padded = torch.zeros((T - 1) * hop + 2 * M, device=self.device, dtype=torch.float32)
        padded[self.pad:self.pad + x.shape[0]] = x
        return padded.unfold(0, 2 * M, hop) * self.window

    def _fold(self, fr):
        h = self.M // 2
        a, b, c, d = fr[..., :h], fr[..., h:2 * h], fr[..., 2 * h:3 * h], fr[..., 3 * h:]
        return torch.cat([-c.flip(-1) - d, a - b.flip(-1)], dim=-1)

    def analyze(self, x):
        fr = self.frames(x)
        X = self.dct4(self._fold(fr))
        Y = self.alt * self.dct4(self._fold(fr.flip(-1)))
        return X, Y

    def synthesize(self, X, n_samples: int):
        M, hop = self.M, self.hop
        h = M // 2
        T = X.shape[0]
        v = self.dct4(X)
        v1, v2 = v[:, :h], v[:, h:]
        blocks = torch.cat([v2, -v2.flip(-1), -v1.flip(-1), -v1], dim=1) * self.window   # (T, 2M)
        L = (T - 1) * hop + 2 * M
        y = F.fold(blocks.t().unsqueeze(0), output_size=(1, L), kernel_size=(1, 2 * M), stride=(1, hop))
        y = y.reshape(-1)[self.pad:] / self.os
        return y[:n_samples]

    # -- cepstral projections
    def smooth(self, log_mag, q: int):
        """lifter(cepstrum(x), q) = x @ B_q @ B_q^T  (last axis)."""
        Bq = self.B[:, :q]
        return (log_mag @ Bq) @ Bq.t()

    def cepstrum(self, log_mag, q: int):
        return log_mag @ self.B[:, :q]

    def true_envelope(self, log_mag, q: int, iters: int = 30, tol_db: float = 0.3):
        env = self.smooth(log_mag, q)
        tol = tol_db * math.log(10) / 20.0
        for _ in range(iters):
            new = self.smooth(torch.maximum(log_mag, env), q)
            done = bool((new - env).abs().max() < tol)
            env = new
            if done:
                break
        return env


def interp_rows(arr, pos):
    """Linear interpolation of rows of arr (T, ...) at fractional positions pos (J,) (torch)."""
    T = arr.shape[0]
    i0 = pos.floor().long().clamp(0, T - 1)
    i1 = (i0 + 1).clamp(0, T - 1)
    f = (pos - i0.to(pos.dtype))
    if arr.dim() > 1:
        f = f[:, None]
    return arr[i0] * (1 - f) + arr[i1] * f


def interp_batched(x, xp, fp, left=None, right=None):
    """np.interp along the last axis for batches: xp (B, K) increasing, fp (B, K) or (K,), x (B, L)."""
    K = xp.shape[-1]
    idx = torch.searchsorted(xp.contiguous(), x.contiguous(), right=True).clamp(1, K - 1)
    x0 = xp.gather(-1, idx - 1)
    x1 = xp.gather(-1, idx)
    if fp.dim() == 1:
        f0, f1 = fp[idx - 1], fp[idx]
    else:
        f0, f1 = fp.gather(-1, idx - 1), fp.gather(-1, idx)
    t = ((x - x0) / (x1 - x0).clamp_min(1e-30)).clamp(0, 1)
    y = f0 + t * (f1 - f0)
    lo = fp[..., 0] if fp.dim() == 1 else fp[..., :1]
    hi = fp[..., -1] if fp.dim() == 1 else fp[..., -1:]
    y = torch.where(x < xp[..., :1], torch.as_tensor(left, device=x.device, dtype=x.dtype) if left is not None else lo, y)
    y = torch.where(x > xp[..., -1:], torch.as_tensor(right, device=x.device, dtype=x.dtype) if right is not None else hi, y)
    return y


def ot_barycenter(broad, w, q_ot: int, lap: TorchLapped, log_freq: bool = True, grid_n: int = 1024):
    """broad (N, J, M) log amplitude of the formant band -> (J, M) Wasserstein barycentre per
    frame (see Morpher._ot_barycenter), all frames at once."""
    N, J, M = broad.shape
    dev = broad.device
    p = torch.exp(2.0 * broad)
    tot = p.sum(-1)                                                       # (N, J)
    cdf = torch.cumsum(p / tot[..., None], dim=-1)
    cdf = torch.cat([torch.zeros(N, J, 1, device=dev), cdf], dim=-1)       # (N, J, M+1)
    edges = torch.arange(M + 1, device=dev, dtype=torch.float32)
    z_edges = torch.log(edges + 2.0) if log_freq else edges
    eps = torch.arange(M + 1, device=dev, dtype=torch.float32) * 1e-7
    cs = torch.cummax(cdf + eps, dim=-1).values                            # strictly increasing
    grid = ((torch.arange(grid_n, device=dev, dtype=torch.float32) + 0.5) / grid_n).expand(J, grid_n)
    levels = torch.cat([grid] + [cs[i] for i in range(N)], dim=-1)         # (J, L): every source's knots
    levels, _ = torch.sort(levels, dim=-1)
    zq = torch.zeros_like(levels)
    for i in range(N):
        zq = zq + float(w[i]) * interp_batched(levels, cs[i], z_edges)
    zq = torch.cummax(zq, dim=-1).values
    Fz = interp_batched(z_edges.expand(J, M + 1), zq, levels, left=0.0, right=1.0)
    dens = torch.diff(Fz, dim=-1)
    tot_m = torch.exp(sum(float(w[i]) * torch.log(tot[i]) for i in range(N)))   # (J,)
    broad_m = 0.5 * torch.log((dens * tot_m[:, None]).clamp_min(1e-30))
    return lap.smooth(broad_m, q_ot)


def smooth_time(x, n: int):
    """Moving average over frames (axis 0), window n, edge-shortened."""
    J = x.shape[0]
    if n < 2 or J < 3:
        return x
    c = torch.cat([torch.zeros(1, x.shape[1], device=x.device, dtype=x.dtype), torch.cumsum(x, 0)], 0)
    half = n // 2
    idx = torch.arange(J, device=x.device)
    lo = (idx - half).clamp(0, J)
    hi = (idx + half + 1).clamp(0, J)
    return (c[hi] - c[lo]) / (hi - lo).to(x.dtype)[:, None]


def wsola_warp(x, src_of_out, n_out: int, period: float, frame: int = 1024, passes: int = 2):
    """Batched WSOLA on the device (see morph.wsola_warp for the sequential version).

    The sequential algorithm continues the previously *chosen* block, a dependency that
    cannot be batched.  Here the block positions are first predicted pitch-synchronously -
    block b starts at the source sample nearest its nominal position whose phase (sample
    index mod the period) equals that of the output position b*H, which is where a
    phase-continuous output would take it - and then refined in `passes` parallel rounds:
    every block re-picks its start within +-1.5 periods of its current position to best
    continue the *predicted* previous block (batched FFT cross-correlation).  The
    prediction is only wrong by the slow drift of the true period from `period`, so the
    refinements of neighbouring blocks agree to a fraction of a sample and the joints stay
    continuous.  Ratio 1 is an exact copy."""
    dev = x.device
    N = frame
    H = N // 2
    nx = x.shape[0]
    dmax = int(np.clip(1.5 * period, 32, H))
    win = torch.hann_window(N, periodic=True, device=dev)
    xp = torch.cat([x, torch.zeros(N + 2 * dmax + H, device=dev)])
    csum = torch.cat([torch.zeros(1, device=dev), torch.cumsum(xp * xp, 0)])
    starts = torch.arange(0, n_out, H, device=dev)
    B = starts.shape[0]
    nom = src_of_out[starts.clamp(max=src_of_out.shape[0] - 1)].clamp(0, nx - 1)             # (B,) float
    out_pos = starts.to(torch.float32)
    p = (nom - torch.remainder(nom - out_pos, period)).round().long().clamp(0, nx - 1)     # phase rule
    ar_n = torch.arange(N, device=dev)
    ar_c = torch.arange(2 * dmax + 1, device=dev)
    L = 1 << int(math.ceil(math.log2(N + 2 * dmax + N)))
    for _ in range(passes):
        tmpl_start = torch.cat([p[:1], p[:-1] + H])
        lo = (p - dmax).clamp_min(0)
        region = xp[lo[:, None] + torch.arange(N + 2 * dmax, device=dev)[None, :]]        # (B, N+2dmax)
        tmpl = xp[tmpl_start[:, None] + ar_n[None, :]]                                    # (B, N)
        corr = torch.fft.irfft(torch.fft.rfft(region, L) * torch.conj(torch.fft.rfft(tmpl, L)), L)[:, :2 * dmax + 1]
        cand = lo[:, None] + ar_c[None, :]
        en = csum[cand + N] - csum[cand]
        score = corr / (torch.sqrt(en) + 1e-9)
        score[cand > p[:, None] + dmax] = -1e30
        newp = lo + score.argmax(1)
        newp[0] = p[0]
        p = newp
    blocks = xp[p[:, None] + ar_n[None, :]] * win                                          # (B, N)
    Lout = (B - 1) * H + N
    y = F.fold(blocks.t().unsqueeze(0), output_size=(1, Lout), kernel_size=(1, N), stride=(1, H)).reshape(-1)
    return y[:n_out]
