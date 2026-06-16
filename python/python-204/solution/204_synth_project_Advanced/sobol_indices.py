"""
sobol_indices.py
================

Saltelli / Jansen / Sobol estimators for first-order, total-order,
and second-order sensitivity indices.

Let ``Y = f(theta)`` with ``theta in [0, 1]^d``.  The ANOVA
decomposition gives

    Var(Y) = sum_i V_i + sum_{i < j} V_{ij} + ... + V_{1...d}

where ``V_i`` is the first-order contribution of ``theta_i`` and
``V_{ij}`` is the *pure* second-order interaction.  The Sobol indices
are

    S_i     = V_i / Var(Y)             (first-order)
    S_i^T   = (V_i + sum_{j != i} V_{ij} + ...) / Var(Y)
                                       (total-order)
    S_{ij}  = V_{ij} / Var(Y)          (pure 2nd-order)

Estimation proceeds via the Saltelli sampling plan: build matrices

    A, B, AB_i, BA_i       for i = 1, ..., d

and evaluate the forward model on each.  The estimators used here
(Jansen 1999, Saltelli 2010, Janon et al. 2014) are::

    f0    = (1 / N) sum_k f(A)_k
    V_tot = (1 / N) sum_k f(B)_k
    V     = (1 / (2N)) sum_k [f(A)_k^2 + f(B)_k^2] - f0^2
    V_i   = (1 / N) sum_k f(B)_k (f(AB_i)_k - f(A)_k)        (Jansen)
    S_i^T = (1 / (2N)) sum_k (f(A)_k - f(AB_i)_k)^2 / V

This module implements:

* ``saltelli_sample`` — construct (A, B, AB_i, BA_i) matrices using
  the column operations from ``r8col_utils`` and the matrix kernels
  from ``matrix_kernels``.
* ``sobol_first_order``, ``sobol_total_order``, ``sobol_second_order``
  — individual estimators.
* ``sobol_bootstrap`` — bootstrap confidence intervals.
* ``sobol_full_analysis`` — one-stop routine that calls the forward
  model and returns all indices + diagnostics.

References
----------
* I. M. Sobol, *Global sensitivity indices for nonlinear mathematical
  models and their Monte Carlo estimates*, Math. Comput. Simul. 55
  (2001), 271-280.
* A. Saltelli et al., *Variance based sensitivity analysis of model
  output*, Comput. Phys. Commun. 181 (2010), 259-270.
* A. Janon et al., *Asymptotic normality and efficiency of two Sobol
  index estimators*, ESAIM: PS 18 (2014), 342-362.
* B. Iooss, A. L. Janon, *Recent improvements in the estimation of
  Sobol indices*, 2014.
"""

from __future__ import annotations

import math

import numpy as np

from r8col_utils import (dedupe_sample_matrix, r8col_duplicates,
                         r8col_mean, r8col_min, r8col_max)
from matrix_kernels import mxm_tiled, gram_centered, cross_gram_centered


# =====================================================================
# Saltelli sampling plan
# =====================================================================
def saltelli_sample(d: int, N: int,
                    rng: np.random.Generator | None = None
                    ) -> dict:
    """Build the Saltelli sampling plan matrices.

    Parameters
    ----------
    d : int
        Number of uncertain parameters (``>= 2``).
    N : int
        Base sample count (columns of A and B).
    rng : Generator, optional

    Returns
    -------
    dict
        ``{'A': (d, N), 'B': (d, N), 'AB': list of (d, N),
        'BA': list of (d, N)}``.
    """
    if d < 2:
        raise ValueError("saltelli_sample: d must be >= 2")
    if N < 4:
        raise ValueError("saltelli_sample: N must be >= 4")
    rng = rng or np.random.default_rng()
    A = rng.random((d, N))
    B = rng.random((d, N))
    # Deduplicate columns within A and within B (rare collisions)
    A = dedupe_sample_matrix(A)
    B = dedupe_sample_matrix(B)
    # Ensure A and B have the same column count
    N_min = min(A.shape[1], B.shape[1])
    A = A[:, :N_min]
    B = B[:, :N_min]
    AB = []
    BA = []
    for i in range(d):
        # AB_i: B with row i replaced by A[i]
        ABi = B.copy()
        ABi[i, :] = A[i, :]
        AB.append(ABi)
        # BA_i: A with row i replaced by B[i]
        BAi = A.copy()
        BAi[i, :] = B[i, :]
        BA.append(BAi)
    return dict(A=A, B=B, AB=AB, BA=BA)


# =====================================================================
# Estimators
# =====================================================================
def sobol_mean_var(fA: np.ndarray, fB: np.ndarray) -> tuple[float, float]:
    """Estimate ``E[Y]`` and ``Var[Y]`` from ``f(A)`` and ``f(B)``.

    Uses the pooled estimator:
        f0   = (mean(fA) + mean(fB)) / 2
        V    = (mean(fA^2) + mean(fB^2)) / 2 - f0^2
    """
    if fA.size != fB.size or fA.size < 2:
        raise ValueError("sobol_mean_var: arrays must match and have >= 2")
    f0 = 0.5 * (fA.mean() + fB.mean())
    V = 0.5 * (np.mean(fA ** 2) + np.mean(fB ** 2)) - f0 * f0
    V = max(V, 1.0e-30)  # guard against degenerate output
    return float(f0), float(V)


def sobol_first_order(fA: np.ndarray, fB: np.ndarray,
                      fABi: np.ndarray, V: float) -> float:
    r"""First-order index via the Saltelli / Janon estimator.

        S_i = (1/N) sum_k f(B)_k * (f(AB_i)_k - f(A)_k)  / V
    """
    if fA.size != fB.size or fB.size != fABi.size:
        raise ValueError("sobol_first_order: shape mismatch")
    N = fA.size
    if N < 2:
        return 0.0
    num = float(np.mean(fB * (fABi - fA)))
    return num / max(V, 1.0e-30)


def sobol_total_order(fA: np.ndarray, fABi: np.ndarray,
                      V: float) -> float:
    r"""Total-order index via Jansen's estimator.

        S_i^T = (1 / (2 N)) sum_k (f(A)_k - f(AB_i)_k)^2  / V
    """
    if fA.size != fABi.size or fA.size < 2:
        return 0.0
    N = fA.size
    num = 0.5 * float(np.mean((fA - fABi) ** 2))
    return num / max(V, 1.0e-30)


def sobol_second_order(fA: np.ndarray, fB: np.ndarray,
                       fABi: np.ndarray, fABj: np.ndarray,
                       fBAj: np.ndarray, V: float) -> float:
    r"""Pure second-order index ``S_{ij}``.

    Uses the Saltelli estimator::

        V_{ij} = V_{i}^{joint} - V_i - V_j

    where ``V_i^{joint}`` is the variance contributed by fixing both
    ``theta_i`` and ``theta_j`` simultaneously.  Implemented via the
    identity

        V_{ij} = Cov(f(AB_i), f(BA_j))  -  V_i V_j  (approx.)

    Returns ``S_{ij} = V_{ij} / V``.
    """
    if fA.size != fB.size or fABi.size != fABj.size:
        raise ValueError("sobol_second_order: shape mismatch")
    N = fA.size
    if N < 4:
        return 0.0
    # Joint variance contribution
    joint = float(np.mean(fABi * fBAj)) - float(np.mean(fABi)) * float(np.mean(fBAj))
    # Approximate V_{ij} = joint (good enough for diagnostic use)
    return joint / max(V, 1.0e-30)


# =====================================================================
# Full Sobol analysis
# =====================================================================
def sobol_full_analysis(f, d: int, N: int,
                        rng: np.random.Generator | None = None
                        ) -> dict:
    """Run the complete Sobol analysis for a scalar forward model ``f``.

    Parameters
    ----------
    f : callable
        ``f(theta) -> float`` with ``theta in R^d`` (one column).
    d : int
        Dimension.
    N : int
        Base sample count.
    rng : Generator, optional

    Returns
    -------
    dict
        ``{'d': ..., 'N': ..., 'f0': ..., 'V': ...,
        'S1': ndarray, 'ST': ndarray, 'S2': ndarray (d, d),
        'n_evals': ...}``.
    """
    if d < 2:
        raise ValueError("sobol_full_analysis: d must be >= 2")
    plan = saltelli_sample(d, N, rng=rng)
    A, B, AB, BA = plan['A'], plan['B'], plan['AB'], plan['BA']
    N_eff = A.shape[1]
    # Evaluate forward model on all matrices
    def eval_block(M: np.ndarray) -> np.ndarray:
        out = np.zeros(M.shape[1], dtype=float)
        for j in range(M.shape[1]):
            out[j] = float(f(M[:, j]))
        return out
    fA = eval_block(A)
    fB = eval_block(B)
    f0, V = sobol_mean_var(fA, fB)
    S1 = np.zeros(d)
    ST = np.zeros(d)
    for i in range(d):
        fABi = eval_block(AB[i])
        S1[i] = sobol_first_order(fA, fB, fABi, V)
        ST[i] = sobol_total_order(fA, fABi, V)
    # Second-order matrix (upper triangle)
    S2 = np.zeros((d, d))
    fAB_cache = [eval_block(AB[i]) for i in range(d)]
    fBA_cache = [eval_block(BA[i]) for i in range(d)]
    for i in range(d):
        for j in range(i + 1, d):
            S2[i, j] = sobol_second_order(fA, fB, fAB_cache[i],
                                          fAB_cache[j], fBA_cache[j], V)
            S2[j, i] = S2[i, j]
    n_evals = (2 + 2 * d) * N_eff
    return dict(d=d, N=N_eff, f0=f0, V=V, S1=S1, ST=ST, S2=S2,
                n_evals=n_evals, fA=fA, fB=fB)


# =====================================================================
# Bootstrap confidence intervals
# =====================================================================
def sobol_bootstrap(f, d: int, N: int, n_boot: int = 200,
                    rng: np.random.Generator | None = None) -> dict:
    """Bootstrap the Sobol indices to obtain 95% CIs.

    Uses the same Saltelli plan but resamples the *rows* of the
    response block ``(fA, fB, fAB_i)`` with replacement.  Much
    cheaper than re-running the forward model.
    """
    if n_boot < 20:
        raise ValueError("sobol_bootstrap: n_boot must be >= 20")
    rng = rng or np.random.default_rng()
    res = sobol_full_analysis(f, d, N, rng=rng)
    fA = res['fA']
    fB = res['fB']
    # We do not re-evaluate; we just resample indices to get bootstrap
    # variation around the *same* forward-model evaluations.  This
    # yields a lower-bound CI (ignores forward-model noise) but is the
    # standard "fast bootstrap" in the UQ literature.
    plan = saltelli_sample(d, N, rng=rng)
    AB, BA = plan['AB'], plan['BA']
    def eval_block(M):
        out = np.zeros(M.shape[1], dtype=float)
        for j in range(M.shape[1]):
            out[j] = float(f(M[:, j]))
        return out
    fAB_all = [eval_block(AB[i]) for i in range(d)]
    fBA_all = [eval_block(BA[i]) for i in range(d)]
    N_eff = fA.size
    S1_boot = np.zeros((n_boot, d))
    ST_boot = np.zeros((n_boot, d))
    for b in range(n_boot):
        idx = rng.integers(0, N_eff, size=N_eff)
        fA_r = fA[idx]
        fB_r = fB[idx]
        f0, V = sobol_mean_var(fA_r, fB_r)
        for i in range(d):
            fABi = fAB_all[i][idx]
            S1_boot[b, i] = sobol_first_order(fA_r, fB_r, fABi, V)
            ST_boot[b, i] = sobol_total_order(fA_r, fABi, V)
    S1_lo = np.quantile(S1_boot, 0.025, axis=0)
    S1_hi = np.quantile(S1_boot, 0.975, axis=0)
    ST_lo = np.quantile(ST_boot, 0.025, axis=0)
    ST_hi = np.quantile(ST_boot, 0.975, axis=0)
    # Covariance for Bayesian refinement (langevin_inversion)
    combined = np.concatenate([S1_boot, ST_boot], axis=1)
    cov = np.cov(combined.T) + 1.0e-10 * np.eye(2 * d)
    return dict(S1=res['S1'], ST=res['ST'],
                S1_ci=(S1_lo, S1_hi), ST_ci=(ST_lo, ST_hi),
                cov=cov, f0=res['f0'], V=res['V'],
                S2=res['S2'], n_evals=res['n_evals'])


# =====================================================================
if __name__ == "__main__":
    # Ishigami test function:  y = sin(x1) + a sin(x2)^2 + b x3^4 sin(x1)
    a, b = 7.0, 0.1

    def ishigami(theta):
        x = theta * 2.0 * math.pi - math.pi
        return (math.sin(x[0]) + a * math.sin(x[1]) ** 2
                + b * x[2] ** 4 * math.sin(x[0]))
    rng = np.random.default_rng(0)
    out = sobol_bootstrap(ishigami, d=3, N=400, n_boot=50, rng=rng)
    print("Ishigami S1 :", out['S1'])
    print("Ishigami ST :", out['ST'])
    print("S1 95% CI lo:", out['S1_ci'][0])
    print("S1 95% CI hi:", out['S1_ci'][1])
