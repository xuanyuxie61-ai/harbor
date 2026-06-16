# -*- coding: utf-8 -*-
"""
perturbation_engine.py
======================
Stochastic and fractal perturbations for the stellar evolution model.

Three perturbation mechanisms are provided, each inspired by a
different seed project:

  1. Fractal composition mixing (seed 446_fractal_coastline)
     The midpoint-subdivision perturbation algorithm is used to
     generate self-similar fluctuations in the composition profile
     X(m), mimicking the effect of turbulent convective mixing.

  2. 1/f (pink) noise in the energy generation rate
     (seed 870_pink_noise)
     Multi-octave superposition of white noise processes produces
     a 1/f^alpha spectrum that models the low-frequency drift
     observed in stellar luminosities (e.g. solar-cycle-like
     variability in active stars).

  3. Threshold-triggered mixing events ("dredge-up")
     (seed 1029_Fdl1989_TimingofOneShotInterventions)
     When a shell-burning front reaches a composition threshold
     (analogous to the SIR infection threshold), a sudden mixing
     event redistributes composition over a specified mass range.

Physical background
-------------------
In realistic stars, mixing processes operate across a wide range of
scales:
  - Convective eddies (local, fast, stochastic)
  - Shear instabilities at composition gradients (intermittent)
  - Thermohaline mixing (double-diffusive, slow)
  - Dredge-up events (global, episodic, on AGB)

We capture these with a layered stochastic model whose three
components correspond to the three seed-project inspirations.

Key formulae
------------
Fractal perturbation (Kahaner-Moler-Nash 1989):
    q_{2n} = 0.5 (p_n + p_{n+1}) + w_n (p_n - p_{n+1})
    with   w_n = mu + mu^2 N(0,1).
1/f noise generation (Orfanidis 1995):
    z(t) = (1/B) sum_{i=1}^B u_i(t)
    where u_i is a white process updated every 2^{i-1} samples.
Threshold dredge-up:
    when  |X_shell - X_threshold| < eps,
    mix the shell over a mass range [m0, m1] by averaging.
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import math
import random


# =====================================================================
# 1) Fractal composition perturbation (seed 446)
# =====================================================================

def coastline_perturb(p: List[float], mu: float,
                       rng: Optional[random.Random] = None) -> List[float]:
    """Fractal midpoint perturbation of a 1D profile.

    This is the direct analogue of `coastline_perturb` from seed
    446_fractal_coastline, which inserts intermediate points in a
    closed polygon by perturbing the midpoint with a random weight:

        q[2i+1] = 0.5 (p[i] + p[i+1])
                + w * (p[i] - p[i+1])
                - w * (p[i-1] - p[i+2])

    where  w = mu + mu^2 * N(0, 1).

    For a stellar composition profile X(m), this produces a
    self-similar (fractal) modulation whose roughness is controlled
    by mu.  The fractal dimension D of the resulting profile is
    approximately  D = 2 - mu  for small mu.
    """
    if rng is None:
        rng = random.Random()
    n = len(p)
    if n < 2:
        return p[:]
    sig = mu * mu
    q = [0.0] * (2 * n - 1)
    for i in range(n):
        q[2*i] = p[i]
    for i in range(n - 1):
        w = mu + sig * rng.gauss(0.0, 1.0)
        mid = 0.5 * (p[i] + p[i+1])
        term1 = w * (p[i] - p[i+1])
        # Extended neighbours with wrap-around for a closed curve;
        # for a non-closed (stellar) profile we clamp at the ends.
        p_left  = p[i-1] if i > 0 else p[i]
        p_right = p[i+2] if i+2 < n else p[i+1]
        term2 = w * (p_left - p_right)
        q[2*i + 1] = mid + term1 + term2
    return q


def fractal_mix_profile(X: List[float], levels: int = 3,
                        mu: float = 0.05, seed: int = 42
                        ) -> List[float]:
    """Apply `levels` recursive fractal perturbations to the
    composition profile X.  Each level doubles the number of points,
    then we downsample back to the original size by averaging pairs.
    """
    rng = random.Random(seed)
    profile = X[:]
    for _ in range(levels):
        profile = coastline_perturb(profile, mu, rng)
        # Downsample back to original length
        n = len(X)
        new = [0.0] * n
        for i in range(n):
            idx = int(round(i * (len(profile) - 1) / max(n - 1, 1)))
            idx = max(0, min(len(profile) - 1, idx))
            new[i] = profile[idx]
        profile = new
    # Normalise: preserve total sum (mass conservation)
    s0 = sum(X)
    s1 = sum(profile)
    if s1 > 0 and s0 > 0:
        profile = [x * s0 / s1 for x in profile]
    return profile


# =====================================================================
# 2) 1/f (pink) noise generator (seed 870)
# =====================================================================

class PinkNoiseGenerator:
    """Voss-McCartney 1/f noise generator.

    Implements the algorithm of `ran1f` from seed 870_pink_noise:
    combine B white noise signals, where signal i is refreshed every
    2^{i-1} samples using the circular-buffer delay of `cdelay2`.

    The resulting process has power spectral density S(f) ~ 1/f^alpha
    with alpha close to 1 for B >= 4.
    """

    def __init__(self, B: int = 6, sigma: float = 1.0, seed: int = 123):
        if B > 31:
            raise ValueError("PinkNoiseGenerator: B must be <= 31")
        self.B = B
        self.sigma = sigma
        self.rng = random.Random(seed)
        # U[i] holds the current value of the i-th white process
        self.U = [self.rng.gauss(0.0, sigma) for _ in range(B)]
        # Q[i] is the countdown counter for the i-th process
        self.Q = [1 << i for i in range(B)]

    def _cdelay2(self, m: int, q: int) -> int:
        """Circular-buffer decrement:  q' = (q - 1) mod (m + 1).
        This is the `cdelay2` function from seed 870."""
        q = q - 1
        if q < 0:
            q = m
        return q

    def sample(self) -> float:
        """Return one sample of 1/f noise."""
        z = 0.0
        for i in range(self.B):
            # Decrement the counter
            self.Q[i] = self._cdelay2((1 << i) - 1, self.Q[i])
            if self.Q[i] == (1 << i) - 1:
                # Refresh this octave
                self.U[i] = self.rng.gauss(0.0, self.sigma)
            z += self.U[i]
        return z / max(self.B, 1)

    def series(self, n: int) -> List[float]:
        """Return a series of n samples."""
        return [self.sample() for _ in range(n)]


def correlation(signal: List[float], max_lag: int) -> List[float]:
    """Compute the sample auto-correlation function
        R(k) = (1/N) sum_{j=0}^{N-k-1} (x_{j+k} - xbar)(x_j - xbar)
    for k = 0, ..., max_lag.  This is the `correlation` function
    from seed 870_pink_noise.
    """
    N = len(signal)
    if N == 0 or max_lag < 0:
        return []
    xbar = sum(signal) / N
    R = [0.0] * (max_lag + 1)
    for k in range(max_lag + 1):
        s = 0.0
        for j in range(N - k):
            s += (signal[j+k] - xbar) * (signal[j] - xbar)
        R[k] = s / N
    return R


# =====================================================================
# 3) Threshold-triggered mixing event (seed 1029)
# =====================================================================

def dredge_up_event(X: List[float], m_grid: List[float],
                    shell_m: float, mixing_width: float,
                    efficiency: float = 0.5) -> List[float]:
    """Mix the composition profile X over a mass range
    [shell_m - mixing_width, shell_m + mixing_width] using a
    Gaussian window of that width.

    This is the direct analogue of the SIR threshold intervention
    (seed 1029_Fdl1989_TimingofOneShotInterventions): when the
    burning shell reaches a critical composition, a sudden mixing
    event redistributes material.

    Parameters
    ----------
    X : composition values at m_grid points.
    m_grid : mass grid (same length as X).
    shell_m : location of the burning shell.
    mixing_width : half-width of the mixing region.
    efficiency : mixing efficiency in [0, 1]; 1 = full homogenisation.

    Returns
    -------
    X_new : mixed composition profile.
    """
    N = len(X)
    if N != len(m_grid):
        raise ValueError("dredge_up_event: X and m_grid must have same length")
    X_new = X[:]
    # Gaussian mixing weight
    sigma = max(mixing_width, 1.0e-30)
    weights = [math.exp(-((m - shell_m) ** 2) / (2.0 * sigma * sigma))
               for m in m_grid]
    total_w = sum(weights)
    if total_w <= 0.0:
        return X_new
    mean_X = sum(w * x for w, x in zip(weights, X)) / total_w
    for i in range(N):
        X_new[i] = (1.0 - efficiency * weights[i] / max(
                     max(weights), 1.0e-30)) * X[i] + (
                     efficiency * weights[i] / max(max(weights), 1.0e-30)) * mean_X
    return X_new


def check_threshold(X: List[float], m_grid: List[float],
                    species_idx_name: str,
                    abundances: dict,
                    threshold: float) -> Optional[Tuple[float, float]]:
    """Check whether a shell-burning threshold has been crossed.

    Returns (m_shell, X_value) if the threshold is crossed, else None.

    The rule mirrors the `check_popthreshold` function of the SIR
    intervention framework: when the local value of a species drops
    below `threshold`, return the location.
    """
    if species_idx_name not in abundances:
        return None
    for i, m in enumerate(m_grid):
        if abundances[species_idx_name][i] < threshold:
            return (m, abundances[species_idx_name][i])
    return None


# =====================================================================
# Diagnostic
# =====================================================================

def _self_test():
    print("perturbation_engine self-test:")
    # Fractal mixing
    X = [1.0] * 16
    Xf = fractal_mix_profile(X, levels=2, mu=0.05, seed=1)
    print(f"  fractal_mix_profile: sum_orig={sum(X):.3f}  sum_new={sum(Xf):.3f}  "
          f"max_diff={max(abs(a-b) for a,b in zip(X,Xf)):.3e}")
    # Pink noise
    png = PinkNoiseGenerator(B=6, sigma=1.0, seed=7)
    samples = png.series(500)
    R = correlation(samples, max_lag=10)
    print(f"  pink noise: mean={sum(samples)/len(samples):.3e}  "
          f"R(0)={R[0]:.3f}  R(1)/R(0)={R[1]/max(R[0],1e-30):.3f}")
    # Dredge-up
    X = [0.0] * 20 + [1.0] * 20 + [0.0] * 20
    mg = [i * 0.05 for i in range(60)]
    Xm = dredge_up_event(X, mg, shell_m=1.5, mixing_width=0.3, efficiency=0.8)
    print(f"  dredge_up_event: sum_before={sum(X):.3f}  sum_after={sum(Xm):.3f}")
    print("perturbation_engine self-test OK")


if __name__ == "__main__":
    _self_test()
