"""
grid_geometry.py
================

Pyramid-grid based adaptive-mesh-refinement (AMR) geometry for a protogalactic
simulation volume.  This module translates the pyramid-grid counting formula
(Burkardt 2014, 932_pyramid_grid) into a *refinement hierarchy*: the l-th
AMR level has a 3-D lattice of side 2^l times the base grid; the total number
of cells in a pyramid of depth L is

       N_total(L, N) = sum_{l=0}^{L} (2^l N)^3

which is the 3-D analogue of the 1-D pyramid sum  sum_{k=1}^{N+1} k^2 =
(N+1)(N+2)(2N+3)/6.  In the galaxy-formation context each level corresponds
to a nested spherical sub-volume around a protogalactic clump, so the
hierarchy follows the radial density profile rather than a Cartesian stacking.

Key formulae
------------
* Pyramid sum (Burkardt):

    S(N) = N(N+1)(2N+1)/6

* AMR cell count with geometric refinement ratio r=2:

    C(L, N) = sum_{l=0}^{L} (r^l N)^3
            = N^3 * (r^{3(L+1)} - 1) / (r^3 - 1)

* Effective resolution:

    dx_eff = L_box / (N * r^l_max)

* Cell volume at level l (CGS):

    V_l = dx_eff^3

Reference
---------
- Berger, M. J., & Colella, P. 1989, JCP 82, 64  (AMR foundation)
- Burkardt, J. 2014, pyramid_grid (MGM project 932)
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np

from astro_constants import (
    KILOPARSEC_CGS, MAX_AMR_LEVELS, BASELINE_GRID_CELLS_PER_AXIS,
    DOMAIN_SIZE_KPC,
)


# =====================================================================
#                   PYRAMID-GRID ANALYTIC COUNTING
# =====================================================================

def pyramid_grid_size_1d(n: int) -> int:
    """
    Number of nodes in a 1-D pyramid grid of depth n (Burkardt).

        S(n) = (n+1)(n+2)(2n+3) / 6

    This is the canonical formula that underpins all pyramid-grid
    constructions used later for AMR hierarchy sizing.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    np1 = n + 1
    return (np1 * (np1 + 1) * (2 * np1 + 1)) // 6


def pyramid_amr_cell_count(base_n: int, max_level: int, ratio: int = 2) -> int:
    """
    Total number of cells in a 3-D pyramid-AMR hierarchy.

        C(L, N, r) = N^3 * (r^{3(L+1)} - 1) / (r^3 - 1)

    For base_n = 16, max_level = 3, r = 2, this gives a count of order
    10^6 which is tractable for a small reproducibility experiment.
    """
    if max_level < 0 or base_n <= 0 or ratio <= 1:
        raise ValueError("invalid hierarchy parameters")
    r3 = ratio ** 3
    return (base_n ** 3) * (r3 ** (max_level + 1) - 1) // (r3 - 1)


def effective_dx_kpc(base_n: int, level: int, box_kpc: float, ratio: int = 2) -> float:
    """Physical cell size in kpc at refinement level `level`."""
    return box_kpc / (base_n * (ratio ** level))


def cell_volume_cgs(base_n: int, level: int, box_kpc: float, ratio: int = 2) -> float:
    """Cell volume in CGS at a given refinement level."""
    dx_kpc = effective_dx_kpc(base_n, level, box_kpc, ratio)
    dx_cgs = dx_kpc * KILOPARSEC_CGS
    return dx_cgs ** 3


# =====================================================================
#                  CELL METADATA AND LEVEL BOUNDARIES
# =====================================================================

@dataclass
class AMRCell:
    """Single AMR cell with physics state needed by the hydro solver."""
    level: int
    ijk: Tuple[int, int, int]
    density_cgs: float = 0.0
    momentum_cgs: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    energy_density_cgs: float = 0.0
    metal_mass_fraction: float = 0.0
    refinement_tag: int = 0

    @property
    def index_tuple(self) -> Tuple[int, int, int]:
        return self.ijk


@dataclass
class AMRLevel:
    """One level of the pyramid AMR hierarchy."""
    level: int
    ncells_per_axis: int
    refinement_ratio: int = 2

    @property
    def total_cells(self) -> int:
        return self.ncells_per_axis ** 3

    @property
    def dx_kpc(self, box_kpc: float = DOMAIN_SIZE_KPC) -> float:
        # note: we pass box_kpc via attribute below; default is DOMAIN_SIZE_KPC
        return box_kpc / self.ncells_per_axis

    def __repr__(self) -> str:
        return (f"AMRLevel(level={self.level}, "
                f"N={self.ncells_per_axis}, cells={self.total_cells})")


# =====================================================================
#                     PYRAMID-AMR GRID MANAGER
# =====================================================================

class PyramidGalaxyGrid:
    """
    Manages a 3-D pyramid-grid AMR hierarchy whose levels follow
    the radial density profile of a protogalactic gas cloud.

    The grid is *static* (not dynamically evolving) for the small-scale
    reproducibility experiment; the levels are determined once from an
    analytic density profile rho(r) ~ r^{-alpha} and the refinement
    criterion |grad rho| * dx / rho > eta_threshold.
    """

    def __init__(
        self,
        base_n: int = BASELINE_GRID_CELLS_PER_AXIS,
        max_level: int = MAX_AMR_LEVELS,
        box_kpc: float = DOMAIN_SIZE_KPC,
        ratio: int = 2,
        density_slope: float = 1.8,
        eta_threshold: float = 0.25,
    ) -> None:
        if max_level > MAX_AMR_LEVELS:
            raise ValueError(f"max_level cannot exceed {MAX_AMR_LEVELS}")
        self.base_n = base_n
        self.max_level = max_level
        self.box_kpc = box_kpc
        self.ratio = ratio
        self.density_slope = density_slope
        self.eta_threshold = eta_threshold

        self.levels: List[AMRLevel] = self._build_levels()
        self.level_cell_counts: List[int] = [lv.total_cells for lv in self.levels]
        self.cumulative_cells = sum(self.level_cell_counts)

    # -----------------------------------------------------------------
    #  construction
    # -----------------------------------------------------------------
    def _build_levels(self) -> List[AMRLevel]:
        """
        Analytic refinement criterion for a spherically-symmetric
        protogalactic profile:

            rho(r) = rho_0 * (r / r_core)^{-alpha}

        A cell at radius r is refined to level l if

            |grad rho| * dx / rho  >  eta_threshold

        which, for the power-law, reduces to

            alpha * dx / r  >  eta_threshold.
        """
        levels: List[AMRLevel] = []
        for l in range(self.max_level + 1):
            ncells = self.base_n * (self.ratio ** l)
            levels.append(AMRLevel(level=l, ncells_per_axis=ncells,
                                   refinement_ratio=self.ratio))
        return levels

    # -----------------------------------------------------------------
    #  geometric queries
    # -----------------------------------------------------------------
    def dx_cgs(self, level: int) -> float:
        """Cell size in cm at a given AMR level."""
        return effective_dx_kpc(self.base_n, level, self.box_kpc,
                                self.ratio) * KILOPARSEC_CGS

    def volume_cgs(self, level: int) -> float:
        return cell_volume_cgs(self.base_n, level, self.box_kpc, self.ratio)

    def total_cells(self) -> int:
        return sum(lv.total_cells for lv in self.levels)

    def analytic_pyramid_total(self) -> int:
        """Analytic prediction of the total cell count."""
        return pyramid_amr_cell_count(self.base_n, self.max_level, self.ratio)

    # -----------------------------------------------------------------
    #  protogalactic profile
    # -----------------------------------------------------------------
    def density_profile(self, r_kpc: np.ndarray, rho_0_cgs: float = 1.0e-24,
                        r_core_kpc: float = 0.5) -> np.ndarray:
        """
        Analytic spherically-symmetric density profile (CGS).

            rho(r) = rho_0 * (r / r_core + epsilon)^{-alpha}

        where epsilon is a small softening to regularise the cusp.
        """
        eps = 1.0e-3
        return rho_0_cgs * np.power(r_kpc / r_core_kpc + eps,
                                    -self.density_slope)

    def grad_rho_over_rho(self, r_kpc: np.ndarray) -> np.ndarray:
        """
        |grad rho| / rho for the power-law profile.

        For rho ~ r^{-alpha}:
            |grad rho|/rho = alpha / r.
        """
        return self.density_slope / np.maximum(r_kpc, 1.0e-6)

    def refine_criterion(self, r_kpc: np.ndarray, level: int) -> np.ndarray:
        """
        Refinement indicator:

            eta(r, l) = (|grad rho| / rho) * dx_l.

        Cells with eta > eta_threshold should be refined to level l+1.
        """
        dx_kpc = effective_dx_kpc(self.base_n, level, self.box_kpc, self.ratio)
        return self.grad_rho_over_rho(r_kpc) * dx_kpc

    # -----------------------------------------------------------------
    #  diagnostics
    # -----------------------------------------------------------------
    def summary(self) -> str:
        lines = [
            "PyramidGalaxyGrid summary:",
            f"  base grid:        {self.base_n}^3",
            f"  box size:         {self.box_kpc} kpc",
            f"  max AMR level:    {self.max_level}",
            f"  refinement ratio: {self.ratio}",
            f"  density slope:    alpha = {self.density_slope}",
            f"  total cells:      {self.total_cells()}",
            f"  analytic count:   {self.analytic_pyramid_total()}",
        ]
        for lv in self.levels:
            lines.append(
                f"    level {lv.level}: {lv.ncells_per_axis}^3 = "
                f"{lv.total_cells} cells, dx = "
                f"{effective_dx_kpc(self.base_n, lv.level, self.box_kpc, self.ratio):.4f} kpc"
            )
        return "\n".join(lines)
