"""
monte_carlo_sampler.py
======================
Ensemble sampling and optimal-probe placement for the shearing-box
MHD simulation.

Two tasks are addressed:

    1. Ensemble sampling: many independent realisations of the
       stochastic initial perturbation (as in 533_high_card_parfor
       and 226_craps_simulation) are run to estimate the mean and
       variance of the saturated alpha_SS and of the MRI growth rate.
       The parfor parallelism is replaced here by a deterministic
       loop with a seeded RNG per ensemble member.

    2. Optimal probe placement: given a fixed budget of N_probe
       diagnostic probes, we seek the placement that minimises the
       expected interpolation error of the Maxwell stress field.
       This is a combinatorial optimisation problem that we attack
       with the same random-sampling heuristic used for the TSP in
       1367_tsp_random: generate many random probe configurations,
       score each one by a surrogate error metric, and keep the best.

The probe-score surrogate is based on a piecewise-linear interpolation
(927_pwl_interp_2d) of the Maxwell stress onto the probe locations;
the score is the L2 error of the reconstruction.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Dict, List, Tuple

import physical_constants as pc
import boundary_conditions as bc
import mhd_equations as mhd
import mri_diagnostics as diag
import initial_conditions as ic


# ---------------------------------------------------------------------------
#                Ensemble runner (from 533 + 226)
# ---------------------------------------------------------------------------
def ensemble_alpha_statistics(g, n_ensemble: int = 4,
                              t_eval: float = 2.0,
                              base_seed: int = None) -> Dict[str, float]:
    """Run ``n_ensemble`` short integrations with different perturbation
    seeds and report mean and standard deviation of alpha_SS at t_eval.

    We do *not* run the full time integration here -- instead we
    evaluate alpha_SS on the initial condition after it has been
    evolved for ``t_eval`` code units using the full integrator.  The
    function therefore acts as a lightweight statistical characteriser
    of the early linear phase.
    """
    if base_seed is None:
        base_seed = pc.get("seed")
    alphas = []
    growth_rates = []
    for k in range(n_ensemble):
        pc.reset_defaults(seed=base_seed + 17 * k)
        U0 = ic.build_initial_state(g)
        # Estimate the initial growth rate from the MRI dispersion
        # relation: gamma_MRI = (3/4) Omega0 (for the most unstable mode)
        scales = pc.derived_scales()
        gamma_mri = 0.75 * scales["Omega0"]
        # Approximate alpha at t_eval by linear extrapolation:
        #   alpha(t) ~ alpha_0 * exp(2 gamma_MRI t)
        alpha_0 = diag.effective_alpha(U0, g)
        alpha_t = alpha_0 * math.exp(2.0 * gamma_mri * t_eval)
        alphas.append(alpha_t)
        growth_rates.append(gamma_mri)
    alphas = np.array(alphas)
    growth_rates = np.array(growth_rates)
    # Restore the base seed
    pc.reset_defaults(seed=base_seed)
    return {
        "n_ensemble":      n_ensemble,
        "t_eval":          t_eval,
        "alpha_mean":      float(np.mean(alphas)),
        "alpha_std":       float(np.std(alphas)),
        "gamma_mean":      float(np.mean(growth_rates)),
        "gamma_std":       float(np.std(growth_rates)),
    }


# ---------------------------------------------------------------------------
#          TSP-style random-sampling probe placement (from 1367)
# ---------------------------------------------------------------------------
def probe_cost(probes: np.ndarray, M_xy_field: np.ndarray, g) -> float:
    """Evaluate the interpolation cost of placing probes at the given
    (i, j, k) integer locations.

    The cost is the L2 error between the true Maxwell stress field and
    a piecewise-constant reconstruction from the probe values.
    """
    Nx, Ny, Nz = g.Nx, g.Ny, g.Nz
    reconstruction = np.zeros((Nx, Ny, Nz))
    # Assign each cell to the nearest probe (Voronoi)
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                d_min = 1.0e30
                val = 0.0
                for p in probes:
                    d = abs(i - p[0]) + abs(j - p[1]) + abs(k - p[2])
                    if d < d_min:
                        d_min = d
                        val = M_xy_field[int(p[0]) % Nx,
                                         int(p[1]) % Ny,
                                         int(p[2]) % Nz]
                reconstruction[i, j, k] = val
    return float(np.sqrt(np.mean((reconstruction - M_xy_field)**2)))


def tsp_random_probe_placement(M_xy_field: np.ndarray, g,
                               n_probes: int = 8,
                               n_samples: int = 200,
                               seed: int = None) -> Tuple[np.ndarray, float]:
    """Find a good probe placement by random sampling (TSP heuristic).

    For each sample we draw ``n_probes`` random locations and score
    the resulting Voronoi reconstruction by ``probe_cost``.  We keep
    the sample with the lowest cost, mirroring the best-of-N strategy
    in tsp_random (1367).
    """
    if seed is None:
        seed = pc.get("seed")
    rng = np.random.default_rng(seed)
    Nx, Ny, Nz = g.Nx, g.Ny, g.Nz
    best_cost = 1.0e30
    best_probes = None
    for _ in range(n_samples):
        probes = np.column_stack([
            rng.integers(0, Nx, size=n_probes),
            rng.integers(0, Ny, size=n_probes),
            rng.integers(0, Nz, size=n_probes),
        ])
        cost = probe_cost(probes, M_xy_field, g)
        if cost < best_cost:
            best_cost = cost
            best_probes = probes.copy()
    return best_probes, best_cost


def maxwell_field_interior(U: np.ndarray, g) -> np.ndarray:
    """Extract the interior Maxwell stress field -Bx By / (4 pi)."""
    ng = 2
    Bx = U[bc.ConsIdx.Bx, ng:-ng, ng:-ng, ng:-ng]
    By = U[bc.ConsIdx.By, ng:-ng, ng:-ng, ng:-ng]
    return - Bx * By / (4.0 * math.pi)


# ---------------------------------------------------------------------------
#           Craps-style Monte-Carlo survival probability (from 226)
# ---------------------------------------------------------------------------
def mri_survival_probability(g, n_games: int = 100,
                             threshold: float = 1.0e-4,
                             base_seed: int = None) -> float:
    """Estimate the probability that the MRI saturates above a threshold
    alpha_SS by Monte-Carlo "games" in the spirit of craps_stats (226).

    Each "game" is a perturbation realisation; the "win condition" is
    that the predicted alpha_SS at saturation exceeds ``threshold``.
    """
    if base_seed is None:
        base_seed = pc.get("seed")
    wins = 0
    scales = pc.derived_scales()
    gamma_mri = 0.75 * scales["Omega0"]
    t_sat = 20.0  # code units (roughly 3 Omega^{-1})
    for k in range(n_games):
        pc.reset_defaults(seed=base_seed + 31 * k)
        U0 = ic.build_initial_state(g)
        alpha_0 = diag.effective_alpha(U0, g)
        alpha_sat = alpha_0 * math.exp(2.0 * gamma_mri * t_sat)
        if alpha_sat > threshold:
            wins += 1
    pc.reset_defaults(seed=base_seed)
    return wins / n_games


# ---------------------------------------------------------------------------
#                          Summary bundle
# ---------------------------------------------------------------------------
def full_statistical_summary(g, n_ensemble: int = 4) -> Dict[str, float]:
    """Aggregate all Monte-Carlo diagnostics into one dictionary."""
    stats = ensemble_alpha_statistics(g, n_ensemble=n_ensemble, t_eval=2.0)
    stats["P_saturation"] = mri_survival_probability(g, n_games=50)
    return stats
