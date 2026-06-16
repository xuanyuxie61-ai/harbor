"""
adaptive_fidelity_sampler.py
============================

Scan-adaptive fidelity selection driven by an ICD-style (Iterated
Conditional Design) acquisition function.  Borrowed from the SUNO MRI
undersampling paper (Gautam et al., IEEE TCI 2026; project 1228), where
a scan-adaptive mask is jointly optimized with a reconstructor.

Analogous construction
----------------------
In MRI, the joint optimization is:
    min_{theta, M_i}  sum_i || f_theta(A_i^H M_i y_i) - x_i ||_2^2
where M_i is the sampling mask and f_theta is the reconstructor.

In multi-fidelity UQ, the analog is:
    min_{model params, level(xi)}  sum_{k=1}^N || F_mf(xi_k) - F_H(xi_k) ||_2^2
where `level(xi) in {0, 1, 2, 3}` is the fidelity mask selecting which
simulator to query at each training input xi_k, and F_mf is the multi-
fidelity predictor (GP posterior mean).

Acquisition function
--------------------
We select the next training point (xi*, l*) by maximizing

    a(xi, l) = rho_l(x) * sigma_mf(x) - lambda * c_l / C_budget

where
  - rho_l(x) is the correlation of fidelity l with the truth (from
    the fidelity manager's correlation matrix),
  - sigma_mf(x) is the current multi-fidelity GP posterior standard
    deviation at x,
  - c_l is the cost of fidelity l,
  - C_budget is the remaining cost budget,
  - lambda is a tunable exploration/exploitation trade-off.

The algorithm alternates between:
  1. greedy single-point selection (analog of the greedy MRI mask),
  2. batch ICD selection (analog of the ICD MRI optimizer).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ----------------------------------------------------------------------
# Dataclass for a sampled training point.
# ----------------------------------------------------------------------
@dataclass
class FidelitySample:
    """A single sampled (xi, level, y, cost) record."""
    xi: List[float]
    level: int
    y: float
    cost: float


# ----------------------------------------------------------------------
# Acquisition function.
# ----------------------------------------------------------------------
def fidelity_acquisition(
    xi: List[float],
    level: int,
    mf_mean: float,
    mf_std: float,
    rho_l: float,
    cost_l: float,
    C_budget: float,
    lambda_: float = 0.1,
) -> float:
    """Compute the acquisition value a(xi, l) for fidelity selection.

    a(xi, l) = rho_l * sigma_mf(xi) - lambda * c_l / max(C_budget, eps)

    Parameters
    ----------
    xi        : d-dimensional input vector.
    level     : proposed fidelity level (for cost lookup).
    mf_mean   : current multi-fidelity GP posterior mean at xi.
    mf_std    : current multi-fidelity GP posterior std at xi.
    rho_l     : correlation of `level` with the truth.
    cost_l    : cost of evaluating `level`.
    C_budget  : remaining cost budget.
    lambda_   : exploration / exploitation trade-off.
    """
    if mf_std < 0.0:
        raise ValueError("fidelity_acquisition: mf_std must be >= 0.")
    if C_budget <= 0.0:
        return -1.0e30
    return rho_l * mf_std - lambda_ * cost_l / max(C_budget, 1.0e-30)


# ----------------------------------------------------------------------
# Greedy fidelity selection (analog of the greedy MRI mask optimizer).
# ----------------------------------------------------------------------
def greedy_fidelity_selection(
    xi_candidates: List[List[float]],
    mf_predictor,              # callable (xi) -> (mean, std)
    fm,                        # FidelityManager instance
    C_budget: float,
    lambda_: float = 0.1,
) -> Tuple[List[float], int, float]:
    """Select the (xi, level) pair with the largest acquisition value.

    Returns (xi*, l*, a_max).
    """
    if not xi_candidates:
        raise ValueError("greedy_fidelity_selection: empty candidate set.")
    best_xi: Optional[List[float]] = None
    best_l = 0
    best_a = -1.0e30
    for xi in xi_candidates:
        mean, std = mf_predictor(xi)
        for l in range(fm.n_levels):
            rho_l = fm.correlation(l, fm.n_levels - 1)
            c_l = fm.cost(l)
            a = fidelity_acquisition(
                xi, l, mean, std, rho_l, c_l, C_budget, lambda_
            )
            if a > best_a:
                best_a = a
                best_xi = xi
                best_l = l
    if best_xi is None:
        raise RuntimeError("greedy_fidelity_selection: no candidate selected.")
    return best_xi, best_l, best_a


# ----------------------------------------------------------------------
# Batch ICD-style fidelity selection.
# ----------------------------------------------------------------------
def icd_batch_selection(
    xi_candidates: List[List[float]],
    mf_predictor,
    fm,
    C_budget: float,
    batch_size: int = 4,
    lambda_: float = 0.1,
    alpha1: float = 1.0,
    alpha2: float = 0.2,
) -> List[Tuple[List[float], int, float]]:
    """Select a batch of (xi, level) pairs via iterated conditional design.

    This is the direct analog of `icd_sampling_optimization` from the
    MRI adaptive-sampling paper.  The acquisition function is modified
    to include diversity penalties:

        a_batch(xi, l; B) = a(xi, l)
            - alpha1 * min_{b in B} || xi - xi_b ||_2
            - alpha2 * sum_{b in B} I(l == l_b)

    so that the batch avoids both spatial and fidelity-level redundancy.
    """
    if batch_size < 1:
        raise ValueError("icd_batch_selection: batch_size must be >= 1.")
    selected: List[Tuple[List[float], int, float]] = []
    remaining = list(xi_candidates)
    for _ in range(batch_size):
        if not remaining:
            break
        best_xi: Optional[List[float]] = None
        best_l = 0
        best_a = -1.0e30
        for xi in remaining:
            mean, std = mf_predictor(xi)
            for l in range(fm.n_levels):
                rho_l = fm.correlation(l, fm.n_levels - 1)
                c_l = fm.cost(l)
                a = fidelity_acquisition(
                    xi, l, mean, std, rho_l, c_l, C_budget, lambda_
                )
                # Diversity penalties.
                if selected:
                    min_dist = min(
                        math.sqrt(sum((xi[k] - b[0][k]) ** 2 for k in range(len(xi))))
                        for b in selected
                    )
                    a -= alpha1 * min_dist
                    same_level_count = sum(1 for b in selected if b[1] == l)
                    a -= alpha2 * same_level_count
                if a > best_a:
                    best_a = a
                    best_xi = xi
                    best_l = l
        if best_xi is None:
            break
        selected.append((best_xi, best_l, best_a))
        remaining.remove(best_xi)
    return selected


# ----------------------------------------------------------------------
# Top-level adaptive sampler: iteratively enrich the training set.
# ----------------------------------------------------------------------
def run_adaptive_sampling(
    fm,                           # FidelityManager
    mf_predictor,                 # callable (xi) -> (mean, std)
    budget: float,
    n_init: int = 8,
    n_iter: int = 20,
    seed: int = 42,
    n_candidates: int = 64,
    bounds: Optional[List[Tuple[float, float]]] = None,
) -> List[FidelitySample]:
    """Run the adaptive fidelity-selection loop.

    Parameters
    ----------
    fm            : FidelityManager (supplies cost and correlation).
    mf_predictor  : current multi-fidelity GP predictor (xi) -> (mean, std).
    budget        : total computational cost budget.
    n_init        : initial Latin-hypercube sample size.
    n_iter        : number of adaptive enrichment iterations.
    seed          : random seed for reproducibility.
    n_candidates  : number of candidate xi per iteration.
    bounds        : [(lo, hi)]^d parameter bounds.
    """
    if bounds is None:
        bounds = [(-3.0, -0.5), (-3.0, -1.0), (-3.0, -0.5), (0.2, 1.0)]
    rng = random.Random(seed)
    d = len(bounds)
    samples: List[FidelitySample] = []

    # Initial Latin hypercube at fidelity level 0 (cheapest).
    for i in range(n_init):
        xi = [rng.uniform(b[0], b[1]) for b in bounds]
        y = fm.evaluate(xi, 0)
        c = fm.cost(0)
        samples.append(FidelitySample(xi=xi, level=0, y=y, cost=c))
        budget -= c
        if budget <= 0:
            return samples

    # Adaptive enrichment.
    for it in range(n_iter):
        # Draw n_candidates new random candidates.
        xi_cands = [[rng.uniform(b[0], b[1]) for b in bounds] for _ in range(n_candidates)]
        selected = icd_batch_selection(
            xi_cands, mf_predictor, fm, budget,
            batch_size=2, lambda_=0.1,
        )
        for xi, l, a_val in selected:
            if budget <= 0:
                break
            c = fm.cost(l)
            if c > budget:
                # Fall back to cheaper fidelity.
                for l_try in range(l - 1, -1, -1):
                    if fm.cost(l_try) <= budget:
                        l = l_try
                        c = fm.cost(l)
                        break
                else:
                    continue
            y = fm.evaluate(xi, l)
            samples.append(FidelitySample(xi=xi, level=l, y=y, cost=c))
            budget -= c
    return samples
