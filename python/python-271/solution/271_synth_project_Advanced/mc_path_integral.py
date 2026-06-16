# -*- coding: utf-8 -*-
"""
Monte Carlo path-integral evaluation of the TFIM partition function.

The Suzuki-Trotter decomposition maps the d-dimensional quantum model
to a (d+1)-dimensional classical Ising model on an L x M lattice
where M is the Trotter number (imaginary-time extent).  The action is

    S = - sum_{i, tau} [ K_tau sigma_{i, tau} sigma_{i, tau+1}
                          + K_x   sigma_{i, tau} sigma_{i+1, tau} ]

with anisotropic couplings
    K_tau = - 0.5 * log tanh(dtau * h)
    K_x   = dtau * J

where dtau = beta / M.  We sample configurations with single-spin
Metropolis updates plus one Wolff cluster sweep per measurement to
mitigate critical slowing-down near the QCP.

The random-number generator uses the same multiplicative congruential
scheme as the ASA random-number tables (045_asa159):
    x_{n+1} = 5^7 x_n mod 2^31
supplemented by a modern PCG32 for the physical updates.
"""

from __future__ import annotations
from typing import Tuple
import numpy as np
try:
    from . import constants as C
except ImportError:
    import constants as C


# ---------------------------------------------------------------------------
# Random-number generator  (hybrid: mulcong + PCG32)
# ---------------------------------------------------------------------------
class MulcongRNG:
    """Multiplicative congruential generator from the ASA tables:
    x_{n+1} = a * x_n  (mod 2^31)  with a = 5^7 = 78125.

    This generator cycles through 2^29 distinct values and is
    *not* suitable for production physics, but it is reproducible
    bit-for-bit across platforms and so provides a stable reference
    for small-scale reproducible experiments.
    """
    def __init__(self, seed: int = 1):
        self.a = 78125
        self.mod = 2 ** 31
        self.state = int(seed) % self.mod
        if self.state % 2 == 0:
            self.state = (self.state + 1) % self.mod

    def next_u01(self) -> float:
        self.state = (self.a * self.state) % self.mod
        return float(self.state) / float(self.mod)

    def next_int(self, n: int) -> int:
        return int(self.next_u01() * n) % n


class PCG32:
    """Small, fast, high-quality 32-bit PCG random generator."""
    def __init__(self, seed: int = 42):
        self.state = np.random.default_rng(seed)

    def integers(self, low: int, high: int) -> int:
        return int(self.state.integers(low, high))

    def random(self) -> float:
        return float(self.state.random())


# ---------------------------------------------------------------------------
# Action and couplings
# ---------------------------------------------------------------------------
def trotter_couplings(J: float, h: float, dtau: float) -> Tuple[float, float]:
    """Return (K_tau, K_x) for the Suzuki-Trotter mapped classical
    model.  K_tau is the (imaginary-time) coupling between slices
    and K_x the intra-slice spatial coupling.

    The formula follows Suzuki (1976):
        K_tau = - 0.5 log tanh(h dtau)
        K_x   = J dtau
    For h dtau -> 0, K_tau -> infty (stiff mode).  We regularise by
    clamping dtau * h to a minimum value.
    """
    arg = max(abs(h) * dtau, 1.0e-6)
    th = np.tanh(arg)
    if th < C.SAFE_LOG_FLOOR:
        K_tau = 50.0   # large but finite cap
    else:
        K_tau = -0.5 * np.log(th)
    K_x = J * dtau
    return float(K_tau), float(K_x)


# ---------------------------------------------------------------------------
# Metropolis single-spin update
# ---------------------------------------------------------------------------
def metropolis_sweep(spins: np.ndarray, K_tau: float, K_x: float,
                      rng: PCG32, n_sweeps: int = 1) -> Tuple[np.ndarray, int]:
    """Perform ``n_sweeps`` Metropolis sweeps (L*M single-spin proposals).
    Returns (spins, n_accept).
    """
    L, M = spins.shape
    n_accept = 0
    for _ in range(n_sweeps):
        for _ in range(L * M):
            i = rng.integers(0, L)
            tau = rng.integers(0, M)
            # Neighbours (periodic in both directions)
            s_up = spins[i, (tau + 1) % M]
            s_dn = spins[i, (tau - 1) % M]
            s_rt = spins[(i + 1) % L, tau]
            s_lf = spins[(i - 1) % L, tau]
            # Energy change if spin flipped
            dE = 2.0 * spins[i, tau] * (K_tau * (s_up + s_dn)
                                          + K_x * (s_rt + s_lf))
            if dE <= 0.0 or rng.random() < np.exp(-dE):
                spins[i, tau] = -spins[i, tau]
                n_accept += 1
    return spins, n_accept


# ---------------------------------------------------------------------------
# Wolff cluster update
# ---------------------------------------------------------------------------
def wolff_step(spins: np.ndarray, K_tau: float, K_x: float,
                rng: PCG32) -> Tuple[np.ndarray, int]:
    """One Wolff cluster flip.  Returns (spins, cluster_size)."""
    L, M = spins.shape
    seed_i = rng.integers(0, L)
    seed_t = rng.integers(0, M)
    cluster = np.zeros_like(spins, dtype=bool)
    stack = [(seed_i, seed_t)]
    cluster[seed_i, seed_t] = True
    sign0 = spins[seed_i, seed_t]
    p_tau = 1.0 - np.exp(-2.0 * K_tau)
    p_x = 1.0 - np.exp(-2.0 * K_x)

    while stack:
        i, tau = stack.pop()
        # 4 neighbours
        neigh = [(i, (tau + 1) % M, p_tau),
                  (i, (tau - 1) % M, p_tau),
                  ((i + 1) % L, tau, p_x),
                  ((i - 1) % L, tau, p_x)]
        for ni, nt, p in neigh:
            if cluster[ni, nt]:
                continue
            if spins[ni, nt] != sign0:
                continue
            if rng.random() < p:
                cluster[ni, nt] = True
                stack.append((ni, nt))
    spins[cluster] = -spins[cluster]
    return spins, int(cluster.sum())


# ---------------------------------------------------------------------------
# Observables
# ---------------------------------------------------------------------------
def magnetization(spins: np.ndarray) -> float:
    return float(np.mean(spins))


def energy_density(spins: np.ndarray, K_tau: float, K_x: float) -> float:
    L, M = spins.shape
    e_tau = 0.0
    e_x = 0.0
    for tau in range(M):
        e_tau -= K_tau * float(np.sum(spins[:, tau] * spins[:, (tau + 1) % M]))
        e_x -= K_x * float(np.sum(spins * np.roll(spins, 1, axis=0)))
    return (e_tau + e_x) / (L * M)


# ---------------------------------------------------------------------------
# Full simulation
# ---------------------------------------------------------------------------
def run_simulation(L: int, M: int, J: float, h: float,
                    beta: float = None,
                    n_sweeps: int = 200,
                    n_warmup: int = 50,
                    seed: int = 0) -> dict:
    """Run a single path-integral MC simulation.

    If ``beta`` is None we set M = beta / dtau with dtau = 1/(2 h) so
    that the Trotter discretisation is controlled.
    """
    dtau = 1.0 / (2.0 * max(abs(h), 1.0e-6))
    if beta is None:
        beta = M * dtau
    K_tau, K_x = trotter_couplings(J, h, dtau)

    rng = PCG32(seed)
    spins = np.ones((L, M), dtype=np.int8)

    # Warm-up
    for _ in range(n_warmup):
        spins, _ = metropolis_sweep(spins, K_tau, K_x, rng, n_sweeps=1)
        spins, _ = wolff_step(spins, K_tau, K_x, rng)

    # Measurement phase
    mags = []
    energies = []
    for _ in range(n_sweeps):
        spins, _ = metropolis_sweep(spins, K_tau, K_x, rng, n_sweeps=1)
        spins, _ = wolff_step(spins, K_tau, K_x, rng)
        mags.append(magnetization(spins))
        energies.append(energy_density(spins, K_tau, K_x))

    mags = np.asarray(mags)
    energies = np.asarray(energies)
    m2 = float(np.mean(mags ** 2))
    m4 = float(np.mean(mags ** 4))
    binder = 1.0 - m4 / max(3.0 * m2 * m2, C.EPS_NUM)
    return {
        "L": L, "M": M, "J": J, "h": h,
        "dtau": dtau, "K_tau": K_tau, "K_x": K_x,
        "mag_mean": float(np.mean(np.abs(mags))),
        "mag_sq_mean": m2,
        "mag_4_mean": m4,
        "binder": binder,
        "energy_mean": float(np.mean(energies)),
    }
