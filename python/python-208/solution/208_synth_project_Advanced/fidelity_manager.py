"""
fidelity_manager.py
===================

Hierarchical multi-fidelity model manager with NER-style model annotation.
Borrows the entity-recognition paradigm from HunFlair2 (project 1170) to
tag each fidelity level with structured metadata (fidelity class, cost,
accuracy, physical assumptions).

Hierarchy
---------
  LEVEL_0  (analytic power-law)            cost ~ 1 microsecond
  LEVEL_1  (Lagrange surrogate on sparse grid)  cost ~ 1 millisecond
  LEVEL_2  (reduced-chemistry ODE)          cost ~ 10 milliseconds
  LEVEL_3  (full disk + chemistry + planet) cost ~ 1 second

The manager tracks:
  - the model registry with NER-style annotation tags,
  - the correlation structure rho_{l,l'} = Corr(F_l, F_{l'}) between levels,
  - the cost ratio c_l / c_{l-1},
  - the variance ratio Var(F_l) / Var(F_{l-1}).

These are used downstream by the adaptive sampler and the multi-fidelity GP.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from physical_system import (
    AU_CGS,
    ChemistryParams,
    DiskParams,
    PlanetParams,
    evaluate_high_fidelity,
)


# ----------------------------------------------------------------------
# NER-style annotation record (inspired by HunFlair2 entity tags).
# ----------------------------------------------------------------------
@dataclass
class FidelityTag:
    """Structured annotation record for a fidelity level."""
    fidelity_class: str        # e.g. "Analytic", "Surrogate", "ReducedODE", "Full"
    resolution_tier: int       # 0 = coarsest, 3 = finest
    physics_tags: List[str] = field(default_factory=list)
    # e.g. ["power-law", "steady-state", "no-planet", "isothermal"]
    cost_exponent: float = 0.0       # log10(cost / 1s)
    nominal_rmse: float = 0.0        # reference RMSE on validation set
    bias_model: str = "additive"     # "additive" | "multiplicative" | "nonlinear"


# ----------------------------------------------------------------------
# Individual fidelity models.
# ----------------------------------------------------------------------
def fidelity_level_0(xi: List[float], planet: PlanetParams,
                     chem: ChemistryParams, t_s: float) -> float:
    """Analytic power-law approximation.

    We approximate the QoI as:
        log10 Q_L0(xi) ~ A_0 + a_1 log10(alpha) + a_2 log10(St)
                                + a_3 log10(Z_0) + a_4 zeta
    with coefficients calibrated from a single high-fidelity run.
    """
    A0 = 35.0
    a = [0.45, 0.30, 1.00, -2.0]
    val = A0
    for j, x in enumerate(xi):
        val += a[j] * x
    return val


def fidelity_level_1(
    xi: List[float],
    planet: PlanetParams,
    chem: ChemistryParams,
    t_s: float,
    calibration: Optional[Dict] = None,
) -> float:
    """Reduced-order surrogate: Lagrange interpolation on a sparse grid
    over the 4-D parameter space.  Built by `lagrange_surrogate.py`.
    The calibration dictionary is injected at runtime.
    """
    if calibration is None or "nodes" not in calibration:
        # Fall back to LEVEL_0 if surrogate is not yet built.
        return fidelity_level_0(xi, planet, chem, t_s)
    from lagrange_surrogate import lagrange_tensor_eval
    return lagrange_tensor_eval(
        xi, calibration["nodes"], calibration["values"]
    )


def fidelity_level_2(
    xi: List[float],
    planet: PlanetParams,
    chem: ChemistryParams,
    t_s: float,
    n_r_coarse: int = 16,
) -> float:
    """Reduced ODE: full physics on a coarse 16-cell radial grid."""
    return evaluate_high_fidelity(xi, planet, chem, t_s, n_r=n_r_coarse)


def fidelity_level_3(
    xi: List[float],
    planet: PlanetParams,
    chem: ChemistryParams,
    t_s: float,
    n_r_fine: int = 64,
) -> float:
    """High-fidelity reference: full physics on a 64-cell grid."""
    return evaluate_high_fidelity(xi, planet, chem, t_s, n_r=n_r_fine)


# ----------------------------------------------------------------------
# Fidelity registry.
# ----------------------------------------------------------------------
class FidelityManager:
    """Manages a hierarchy of fidelity levels with NER-style tags."""

    def __init__(
        self,
        planet: PlanetParams,
        chem: ChemistryParams,
        t_s: float,
        surrogate_calibration: Optional[Dict] = None,
    ) -> None:
        self.planet = planet
        self.chem = chem
        self.t_s = t_s
        self.surrogate_calibration = surrogate_calibration

        self.tags: List[FidelityTag] = [
            FidelityTag(
                fidelity_class="Analytic",
                resolution_tier=0,
                physics_tags=["power-law", "steady-state", "no-planet", "isothermal"],
                cost_exponent=-6.0,
                nominal_rmse=0.8,
                bias_model="additive",
            ),
            FidelityTag(
                fidelity_class="Surrogate",
                resolution_tier=1,
                physics_tags=["lagrange", "sparse-grid", "no-planet", "thermal"],
                cost_exponent=-3.0,
                nominal_rmse=0.35,
                bias_model="additive",
            ),
            FidelityTag(
                fidelity_class="ReducedODE",
                resolution_tier=2,
                physics_tags=["ode-chemistry", "coarse-grid", "migrating-planet"],
                cost_exponent=-2.0,
                nominal_rmse=0.10,
                bias_model="nonlinear",
            ),
            FidelityTag(
                fidelity_class="Full",
                resolution_tier=3,
                physics_tags=["full-physics", "fine-grid", "pebble-accretion",
                              "viscous-evolution"],
                cost_exponent=0.0,
                nominal_rmse=0.0,
                bias_model="none",
            ),
        ]
        self.n_levels = len(self.tags)

        # Cost ratios (c_l / c_{l-1}).  Determined empirically.
        self.cost_ratios: List[float] = [1.0, 50.0, 20.0, 30.0]

        # Correlation matrix rho_{l,l'} between fidelity outputs.
        # Default: strong correlation with nearest neighbor.
        self.correlation_matrix: List[List[float]] = [
            [1.00, 0.92, 0.80, 0.70],
            [0.92, 1.00, 0.96, 0.88],
            [0.80, 0.96, 1.00, 0.98],
            [0.70, 0.88, 0.98, 1.00],
        ]

    # ---- evaluation dispatch ------------------------------------------
    def evaluate(self, xi: List[float], level: int) -> float:
        """Evaluate the forward map at fidelity `level`."""
        if level < 0 or level >= self.n_levels:
            raise ValueError(
                f"FidelityManager.evaluate: level {level} out of range."
            )
        if level == 0:
            return fidelity_level_0(xi, self.planet, self.chem, self.t_s)
        if level == 1:
            return fidelity_level_1(
                xi, self.planet, self.chem, self.t_s,
                calibration=self.surrogate_calibration,
            )
        if level == 2:
            return fidelity_level_2(xi, self.planet, self.chem, self.t_s)
        return fidelity_level_3(xi, self.planet, self.chem, self.t_s)

    # ---- query cost ---------------------------------------------------
    def cost(self, level: int) -> float:
        """Relative computational cost of evaluating fidelity `level`."""
        return self.cost_ratios[level]

    # ---- correlation --------------------------------------------------
    def correlation(self, l1: int, l2: int) -> float:
        return self.correlation_matrix[l1][l2]

    # ---- NER-style tag dump ------------------------------------------
    def describe(self, level: int) -> FidelityTag:
        return self.tags[level]

    # ---- batch evaluation --------------------------------------------
    def batch_evaluate(
        self, xi_batch: List[List[float]], level: int
    ) -> List[float]:
        return [self.evaluate(xi, level) for xi in xi_batch]
