"""
adaptive_refinement.py
======================

Adaptive mesh refinement (AMR) tag selection via the *rational
knapsack* algorithm.

The seed project 627_knapsack_rational solves the rational (fractional)
knapsack problem: given N items with profits P_i and weights W_i, and
a weight budget M, find x_i in [0, 1] that maximises sum P_i x_i
subject to sum W_i x_i <= M.  The solution is the greedy algorithm
that sorts items by profit density P_i / W_i and fills in decreasing
order, with a fractional last item.

Key mappings
------------
* Item                       -> grid cell candidate for refinement
* Profit P_i                 -> refinement benefit
                                = |grad rho|_i * dx_i^3 * indicator
* Weight W_i                 -> memory cost of refining cell i
                                = 8 (one cell -> 8 children in 3-D)
* Weight budget M            -> available memory cells for refinement
* Fractional x_i             -> partial refinement flag (used to set
                                intermediate refinement in AMR)

The *profit density*

    r_i = P_i / W_i

orders cells by *refinement efficiency* (benefit per unit memory).

Profit function
---------------
For a cell i at level l with density rho_i, the refinement profit is

    P_i = |grad rho|_i / rho_i * V_i * C_jeans

where

    V_i = dx_i^3                (cell volume)
    C_jeans = (L_J / dx_i)^2   (how many cells per Jeans length)

The first factor marks cells near density gradients (shocks, contact
discontinuities); the Jeans factor marks cells that are under-resolved
for gravitational collapse (Truelove et al. 1997).

References
----------
- Kreher, D., & Simpson, D. 1998, Combinatorial Algorithms, CRC
- Burkardt, J., knapsack_rational (seed project 627)
- Truelove, J. K. et al. 1997, ApJ 489, L179 (Jeans refinement)
"""

from __future__ import annotations
import math
import numpy as np
from typing import List, Tuple, Optional, Dict

from astro_constants import (
    GRAVITATIONAL_CGS, KILOPARSEC_CGS, KNAPSACK_WEIGHT_BUDGET_FRAC,
    MAX_AMR_LEVELS,
)


# =====================================================================
#                 RATIONAL KNAPSACK (seed 627)
# =====================================================================

def knapsack_rational(profits: np.ndarray, weights: np.ndarray,
                       mass_limit: float
                       ) -> Tuple[np.ndarray, float, float]:
    """
    Solve the rational knapsack problem.

    Given items with profits P_i and weights W_i, find x_i in [0, 1]
    that maximises sum P_i x_i subject to sum W_i x_i <= M.

    Assumes items are already sorted in decreasing order of P_i / W_i.

    Returns
    -------
    x : array of fractional selections
    total_profit : achieved profit
    total_weight : used weight
    """
    n = profits.size
    if n != weights.size:
        raise ValueError("profits and weights must have same size")
    x = np.zeros(n)
    remaining = mass_limit
    total_profit = 0.0
    total_weight = 0.0
    for i in range(n):
        if weights[i] <= 0.0:
            continue
        if remaining >= weights[i]:
            x[i] = 1.0
            remaining -= weights[i]
            total_profit += profits[i]
            total_weight += weights[i]
        else:
            # fractional last item
            x[i] = remaining / weights[i]
            total_profit += profits[i] * x[i]
            total_weight += weights[i] * x[i]
            remaining = 0.0
            break
    return x, total_profit, total_weight


def knapsack_reorder(profits: np.ndarray, weights: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reorder items by decreasing profit density P_i / W_i.

    Returns (sorted_profits, sorted_weights, order).
    """
    density = profits / np.maximum(weights, 1.0e-60)
    order = np.argsort(-density)
    return profits[order], weights[order], order


# =====================================================================
#                 AMR PROFIT FUNCTION
# =====================================================================

def refinement_profit(
    rho_3d: np.ndarray,
    dx_cgs: float,
    level: int,
    gamma: float = 5.0 / 3.0,
    temperature_k: float = 8000.0,
    mu: float = 1.22,
) -> np.ndarray:
    """
    Compute the per-cell refinement profit:

        P_i = (|grad rho| / rho) * V_cell * C_jeans

    where C_jeans = (L_J / dx)^2 is the square of the number of cells
    per Jeans length.  Cells with C_jeans < 4 are severely
    under-resolved and have large profit.
    """
    # density gradient magnitude (centred differences, periodic)
    grad_mag = np.zeros_like(rho_3d)
    for axis in range(3):
        f_plus = np.roll(rho_3d, shift=-1, axis=axis)
        f_minus = np.roll(rho_3d, shift=1, axis=axis)
        grad_mag += ((f_plus - f_minus) / (2.0 * dx_cgs)) ** 2
    grad_mag = np.sqrt(grad_mag)
    rho_safe = np.maximum(rho_3d, 1.0e-60)
    grad_over_rho = grad_mag / rho_safe
    V_cell = dx_cgs ** 3
    # Jeans length
    from astro_constants import BOLTZMANN_CGS, PROTON_MASS_CGS
    cs = math.sqrt(gamma * BOLTZMANN_CGS * temperature_k / (mu * PROTON_MASS_CGS))
    L_J = cs * np.sqrt(math.pi / (GRAVITATIONAL_CGS * rho_safe))
    C_jeans = (L_J / dx_cgs) ** 2
    return grad_over_rho * V_cell * C_jeans


# =====================================================================
#                  AMR TAG SELECTOR
# =====================================================================

class AMRTagger:
    """
    Select cells for refinement using the rational-knapsack algorithm.

    At each refinement pass we:
      1. compute the profit of each cell
      2. sort by profit density (profit / memory_cost)
      3. solve the rational knapsack with the memory budget
      4. tag cells with x_i > 0.5 for refinement
    """

    def __init__(
        self,
        memory_budget_cells: int = 10000,
        max_level: int = MAX_AMR_LEVELS,
        profit_threshold: float = 1.0e-30,
    ) -> None:
        self.memory_budget = memory_budget_cells
        self.max_level = max_level
        self.profit_threshold = profit_threshold
        self.history: List[Dict[str, float]] = []

    def select_cells_to_refine(
        self,
        rho_3d: np.ndarray,
        dx_cgs: float,
        current_level: int,
        level_map: Optional[np.ndarray] = None,
        temperature_k: float = 8000.0,
    ) -> np.ndarray:
        """
        Return a boolean mask of cells to refine.

        Parameters
        ----------
        rho_3d : 3-D density array
        dx_cgs : current cell size in cm
        current_level : current AMR level
        level_map : optional 3-D array of current cell levels
        temperature_k : assumed temperature for Jeans length
        """
        if current_level >= self.max_level:
            return np.zeros_like(rho_3d, dtype=bool)
        profit = refinement_profit(rho_3d, dx_cgs, current_level,
                                    temperature_k=temperature_k)
        # only consider cells above the profit threshold
        above = profit > self.profit_threshold
        if not np.any(above):
            return np.zeros_like(rho_3d, dtype=bool)
        # if level_map provided, skip cells already at max level
        if level_map is not None:
            above = above & (level_map < self.max_level)
        profits_sub = profit[above]
        # memory cost: 7 cells per refined cell (1 becomes 8, net +7)
        weights_sub = np.full(profits_sub.shape, 7.0)
        # sort by profit density (descending)
        p_sorted, w_sorted, order = knapsack_reorder(profits_sub, weights_sub)
        # solve rational knapsack
        x_sorted, total_profit, total_weight = knapsack_rational(
            p_sorted, w_sorted, self.memory_budget
        )
        # undo the sort
        x_original_order = np.empty_like(x_sorted)
        x_original_order[order] = x_sorted
        # map back to full grid
        x_full = np.zeros(profit.shape)
        x_full[above] = x_original_order
        # tag cells with x > 0.5
        tag = x_full > 0.5
        self.history.append({
            "level": current_level,
            "n_tagged": int(np.sum(tag)),
            "total_profit": total_profit,
            "total_weight": total_weight,
            "budget": self.memory_budget,
        })
        return tag

    def summary(self) -> str:
        if not self.history:
            return "AMRTagger: no refinement passes yet"
        last = self.history[-1]
        lines = [
            f"AMRTagger summary (last pass):",
            f"  level:         {last['level']}",
            f"  cells tagged:  {last['n_tagged']}",
            f"  profit:        {last['total_profit']:.3e}",
            f"  weight used:   {last['total_weight']:.0f} / {last['budget']}",
        ]
        return "\n".join(lines)


# =====================================================================
#                  CONVENIENCE: JEANS NUMBER CHECK
# =====================================================================

def jeans_number_check(rho_3d: np.ndarray, dx_cgs: float,
                       temperature_k: float = 8000.0,
                       critical_nj: float = 4.0) -> np.ndarray:
    """
    Return a boolean mask of cells where the Jeans number

        N_J = L_J / dx

    falls below the critical value (default 4).  These cells must be
    refined to avoid artificial fragmentation (Truelove et al. 1997).
    """
    from astro_constants import BOLTZMANN_CGS, PROTON_MASS_CGS
    gamma = 5.0 / 3.0
    mu = 1.22
    cs = math.sqrt(gamma * BOLTZMANN_CGS * temperature_k / (mu * PROTON_MASS_CGS))
    rho_safe = np.maximum(rho_3d, 1.0e-60)
    L_J = cs * np.sqrt(math.pi / (GRAVITATIONAL_CGS * rho_safe))
    N_J = L_J / dx_cgs
    return N_J < critical_nj
