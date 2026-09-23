"""Lapped DCT transforms: MDCT / MDST (the MCLT) and a DCT-II cepstral "true envelope".

MDCT frame of a 2M-sample block x (sine window w):
    X[k] = sqrt(2/M) * sum_n w[n] x[n] cos(pi/M (n + 1/2 + M/2)(k + 1/2)),  k = 0..M-1
The MDST uses sin() in place of cos(); X + jY is Malvar's modulated complex lapped
transform (MCLT), whose magnitude and phase behave like an STFT's.  With the sine
window (Princen-Bradley) the IMDCT + overlap-add reconstructs x exactly (TDAC), so a
morph built in the MCLT domain is written back through the real (DCT) part alone.

Both transforms are computed through a DCT-IV of the TDAC-folded block (O(M log M)):
    MDCT(a, b, c, d) = DCT-IV(-c_r - d, a - b_r)          (quarters a..d of the windowed block)
    MDST(x)[k]       = -(-1)^k MDCT(reverse(x))[k]        (the sine window is symmetric)
and the IMDCT unfolds DCT-IV(X) = (v1, v2) to (v2, -v2_r, -v1_r, -v1) before windowing.

`hop` may be M / os for an integer `os` (default 4): the os interleaved hop-M lattices
each reconstruct on their own, so the overlap-add divided by os is still exact, and
modified coefficients (time-warped, re-phased) are averaged over os lattices, which
tames the time-domain aliasing that a single-lattice MDCT would leak.
"""
from __future__ import annotations

import numpy as np
from scipy.fft import dct, idct


class MDCT:
    def __init__(self, M: int = 1024, hop: int | None = None):
        self.M = M
        self.hop = hop or M // 4
        if M % self.hop != 0:
            raise ValueError("hop must divide M")
        self.os = M // self.hop
        n = np.arange(2 * M)
        k = np.arange(M)
        self.window = np.sin(np.pi * (n + 0.5) / (2 * M)).astype(np.float64)
        self.pad = 2 * M                                           # leading zeros (see module doc)
        self.bin_omega = np.pi * (k + 0.5) / M                     # rad/sample centre of each bin
        self._alt = np.where(k % 2 == 0, -1.0, 1.0)                # -(-1)^k

    def bin_hz(self, sr: int) -> np.ndarray:
        return self.bin_omega * sr / (2 * np.pi)

    def n_frames(self, n_samples: int) -> int:
        return int(np.ceil((self.pad + n_samples + self.M) / self.hop)) + 1

    def frame_time(self, t) -> np.ndarray:
        """Centre of frame t in signal samples."""
        return np.asarray(t, np.float64) * self.hop + self.M - self.pad

    def frames(self, x: np.ndarray) -> np.ndarray:
        M, hop = self.M, self.hop
        T = self.n_frames(len(x))
        padded = np.zeros((T - 1) * hop + 2 * M, np.float64)
        padded[self.pad:self.pad + len(x)] = x
        idx = np.arange(2 * M)[None, :] + (np.arange(T) * hop)[:, None]
        return padded[idx] * self.window[None, :]

    def _fold(self, fr: np.ndarray) -> np.ndarray:
        h = self.M // 2
        a, b, c, d = fr[..., :h], fr[..., h:2 * h], fr[..., 2 * h:3 * h], fr[..., 3 * h:]
        return np.concatenate([-c[..., ::-1] - d, a - b[..., ::-1]], axis=-1)

    def mdct_of_frames(self, fr: np.ndarray) -> np.ndarray:
        return dct(self._fold(fr), type=4, norm="ortho", axis=-1)

    def analyze(self, x: np.ndarray):
        """Return (mdct, mdst), each (T, M)."""
        fr = self.frames(np.asarray(x, np.float64))
        X = self.mdct_of_frames(fr)
        Y = self._alt * self.mdct_of_frames(fr[:, ::-1])
        return X, Y

    def mclt(self, x: np.ndarray):
        """Return (magnitude, phase) of the MCLT, each (T, M)."""
        X, Y = self.analyze(x)
        return np.hypot(X, Y), np.arctan2(-Y, X)   # sign: phase advances by +omega*hop per frame

    def synthesize(self, X: np.ndarray, n_samples: int | None = None) -> np.ndarray:
        """IMDCT + overlap-add of (T, M) MDCT coefficients (frames at `hop`)."""
        M, hop = self.M, self.hop
        h = M // 2
        T = X.shape[0]
        v = dct(np.asarray(X, np.float64), type=4, norm="ortho", axis=-1)
        v1, v2 = v[:, :h], v[:, h:]
        blocks = np.concatenate([v2, -v2[:, ::-1], -v1[:, ::-1], -v1], axis=1) * self.window[None, :]
        y = np.zeros((T - 1) * hop + 2 * M, np.float64)
        # overlap-add: the os interleaved lattices are each a plain hop-M OLA
        for j in range(self.os):
            bl = blocks[j::self.os]
            L = bl.shape[0]
            if L == 0:
                continue
            seg = np.zeros((L + 1) * M, np.float64)
            seg[:L * M] += bl[:, :M].reshape(-1)
            seg[M:(L + 1) * M] += bl[:, M:].reshape(-1)
            y[j * hop:j * hop + (L + 1) * M] += seg
        y = y[self.pad:] / self.os
        if n_samples is not None:
            y = y[:n_samples]
        return y

    def from_polar(self, mag: np.ndarray, phase: np.ndarray, n_samples: int | None = None) -> np.ndarray:
        return self.synthesize(mag * np.cos(phase), n_samples)


def cepstrum(log_mag: np.ndarray) -> np.ndarray:
    """DCT-II of a log-magnitude spectrum along the last axis (orthonormal)."""
    return dct(log_mag, type=2, norm="ortho", axis=-1)


def lifter(c: np.ndarray, q: int) -> np.ndarray:
    """Keep cepstral coefficients [0, q) and return the smoothed log spectrum."""
    return idct(c[..., :q], type=2, norm="ortho", n=c.shape[-1], axis=-1)


def true_envelope(log_mag: np.ndarray, q: int, iters: int = 30, tol_db: float = 0.3) -> np.ndarray:
    """Roebel & Rodet's true envelope: cepstral smoothing iterated so the envelope rides
    over the harmonic peaks instead of through them. `log_mag` is a natural log,
    shape (..., K)."""
    env = lifter(cepstrum(log_mag), q)
    tol = tol_db * np.log(10) / 20.0
    for _ in range(iters):
        new = lifter(cepstrum(np.maximum(log_mag, env)), q)
        done = np.max(np.abs(new - env)) < tol
        env = new
        if done:
            break
    return env
