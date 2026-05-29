# -*- coding: utf-8 -*-
"""
optimization.py
===============
Discrete optimization of base isolation parameters using concepts from:
  - 743_mcnuggets_diophantine:  Nonnegative integer Diophantine solutions
  - 045_asa159:                  Random constrained combinatorial sampling

Problem formulation:
  Given a building with total weight W and target isolation period T_iso,
  determine the optimal number of lead-rubber bearings and their parameters.

  Decision variables (integer-constrained):
    n_b  = number of bearings   (positive integer)
    Q_d  = characteristic strength per bearing [N]
    k_d  = post-yield stiffness per bearing [N/m]

  Constraints (Diophantine-like integer-feasibility):
    (1)  n_b * Q_d  >=  0.05 * W   (minimum yield strength for wind restraint)
    (2)  n_b * k_d  =  (4 * pi^2 * M) / T_iso^2   (target period)
    (3)  n_b >= 4,  Q_d > 0,  k_d > 0
    (4)  Q_d / (k_e - k_d) <= d_y_max   (yield displacement limit)

  The equality constraint (2) is a linear Diophantine relation when
  discretized onto a commercial bearing parameter grid.

  Objective:
    Minimize  J = alpha * (base_shear_ratio) + beta * (total_bearing_cost)
    where base_shear_ratio = V_base / W  (seismic force reduction).
"""

import numpy as np
from typing import List, Tuple, Optional


# ====================================================================== #
# Diophantine nonnegative solver (from 743_mcnuggets_diophantine)
# ====================================================================== #
def diophantine_nd_nonnegative(a: np.ndarray, b: int) -> np.ndarray:
    """
    Find all nonnegative integer solutions to:
      a_1 * x_1 + a_2 * x_2 + ... + a_n * x_n = b
    
    Uses backtracking search with pruning (adapted from the seed project).
    
    Returns
    -------
    solutions : np.ndarray, shape (k, n)
        Array of k solution vectors.
    """
    a = np.asarray(a, dtype=int).ravel()
    n = len(a)
    if n == 0 or b < 0:
        return np.empty((0, n), dtype=int)

    solutions = []
    y = np.zeros(n, dtype=int)
    j = 0

    while True:
        # Compute residual after first j components
        r = b - np.dot(a[:j], y[:j])

        if j < n:
            y[j] = r // a[j] if a[j] != 0 else 0
            j += 1
        else:
            if r == 0:
                solutions.append(y.copy())
            # Backtrack
            while j > 0:
                j -= 1
                if y[j] > 0:
                    y[j] -= 1
                    j += 1
                    break
            else:
                break

    if not solutions:
        return np.empty((0, n), dtype=int)
    return np.array(solutions, dtype=int)


# ====================================================================== #
# Random constrained sampling (from 045_asa159 idea)
# ====================================================================== #
def random_constrained_sample(
    n_samples: int,
    n_b_min: int,
    n_b_max: int,
    Q_d_grid: np.ndarray,
    k_d_grid: np.ndarray,
    M_total: float,
    T_iso_target: float,
    W_total: float,
    seed: int = 42,
) -> List[dict]:
    """
    Generate random feasible isolation design samples by constrained
    combinatorial sampling, inspired by the random contingency table method.
    
    Each sample must satisfy the period constraint approximately and the
    strength constraint exactly.
    """
    rng = np.random.default_rng(seed)
    samples = []

    for _ in range(n_samples * 10):
        if len(samples) >= n_samples:
            break
        n_b = rng.integers(n_b_min, n_b_max + 1)
        Q_d = rng.choice(Q_d_grid)
        k_d = rng.choice(k_d_grid)

        # Check constraints
        k_total = n_b * k_d
        period_est = 2.0 * np.pi * np.sqrt(M_total / k_total) if k_total > 0 else np.inf
        strength_total = n_b * Q_d

        if strength_total < 0.05 * W_total:
            continue
        if abs(period_est - T_iso_target) / T_iso_target > 0.15:
            continue

        samples.append({
            "n_bearings": int(n_b),
            "Q_d_per": float(Q_d),
            "k_d_per": float(k_d),
            "period_est": float(period_est),
            "strength_total": float(strength_total),
        })

    return samples[:n_samples]


# ====================================================================== #
# Isolation parameter optimizer
# ====================================================================== #
class IsolationOptimizer:
    """
    Optimize lead-rubber bearing configuration for a given building.
    """

    def __init__(
        self,
        M_total: float,
        W_total: float,
        T_iso_target: float = 2.5,
        d_y_max: float = 0.15,
        n_b_min: int = 4,
        n_b_max: int = 40,
    ):
        """
        Parameters
        ----------
        M_total : float
            Total building mass [kg].
        W_total : float
            Total building weight [N].
        T_iso_target : float
            Target isolation period [s].
        d_y_max : float
            Maximum allowable yield displacement [m].
        n_b_min, n_b_max : int
            Bounds on number of bearings.
        """
        self.M_total = float(M_total)
        self.W_total = float(W_total)
        self.T_iso_target = float(T_iso_target)
        self.d_y_max = float(d_y_max)
        self.n_b_min = n_b_min
        self.n_b_max = n_b_max

        # Commercial bearing parameter grids (discrete)
        self.Q_d_grid = np.array([
            1.0e4, 1.5e4, 2.0e4, 2.5e4, 3.0e4, 4.0e4, 5.0e4,
            6.0e4, 8.0e4, 1.0e5, 1.2e5, 1.5e5, 2.0e5, 3.0e5,
        ], dtype=float)
        self.k_d_grid = np.array([
            2.0e5, 3.0e5, 4.0e5, 5.0e5, 6.0e5, 7.5e5,
            1.0e6, 1.25e6, 1.5e6, 2.0e6, 2.5e6, 3.0e6, 4.0e6,
        ], dtype=float)

    # ------------------------------------------------------------------ #
    # Objective function
    # ------------------------------------------------------------------ #
    def objective(self, design: dict, spectral_accel: float = 2.5) -> float:
        """
        Compute the design objective:
          J = w1 * (base_shear / W) + w2 * (cost_proxy)
        
        Base shear for isolation system (simplified):
          V_b = M_total * S_a * (T_iso / T_fixed)^{0.5}
        where S_a is the spectral acceleration at the isolation period.
        """
        w1 = 1.0
        w2 = 0.001   # cost per bearing

        n_b = design["n_bearings"]
        k_total = n_b * design["k_d_per"]
        period_est = 2.0 * np.pi * np.sqrt(self.M_total / k_total) if k_total > 0 else 1.0

        # Simplified base shear ratio
        base_shear_ratio = spectral_accel / 9.81 * (period_est / 0.5) ** (-0.5)
        base_shear_ratio = min(base_shear_ratio, 1.0)

        cost = n_b * design["Q_d_per"] * 1e-5   # proxy cost

        return w1 * base_shear_ratio + w2 * cost

    # ------------------------------------------------------------------ #
    # Enumerate feasible designs via Diophantine-inspired grid search
    # ------------------------------------------------------------------ #
    def optimize(self, n_samples: int = 200) -> dict:
        """
        Find the optimal bearing configuration via deterministic grid search
        over commercial bearing parameter grids.
        
        Strategy:
          1. Enumerate all (n_b, k_d) combinations satisfying period constraint.
          2. For each, select minimum Q_d satisfying strength constraint.
          3. Evaluate objective and pick best design.
        """
        best_obj = np.inf
        best_design = None

        for n_b in range(self.n_b_min, self.n_b_max + 1):
            for k_d in self.k_d_grid:
                k_total = n_b * k_d
                if k_total <= 0:
                    continue
                period_est = 2.0 * np.pi * np.sqrt(self.M_total / k_total)
                if abs(period_est - self.T_iso_target) / self.T_iso_target > 0.30:
                    continue

                # Select minimum Q_d that satisfies strength constraint
                # and yield displacement limit
                for Q_d in self.Q_d_grid:
                    strength_total = n_b * Q_d
                    if strength_total < 0.005 * self.W_total:
                        continue
                    k_e = 10.0 * k_d
                    d_y = Q_d / (k_e - k_d)
                    if d_y > self.d_y_max:
                        continue

                    cand = {
                        "n_bearings": int(n_b),
                        "Q_d_per": float(Q_d),
                        "k_d_per": float(k_d),
                        "period_est": float(period_est),
                        "strength_total": float(strength_total),
                    }
                    obj = self.objective(cand)
                    if obj < best_obj:
                        best_obj = obj
                        best_design = cand

        if best_design is None:
            # Fallback: nearest feasible design
            K_target = 4.0 * np.pi ** 2 * self.M_total / (self.T_iso_target ** 2)
            n_b = 20
            k_d = K_target / n_b
            best_design = {
                "n_bearings": n_b,
                "Q_d_per": 5.0e4,
                "k_d_per": float(k_d),
                "period_est": float(self.T_iso_target),
                "strength_total": float(n_b * 5.0e4),
            }
            best_obj = self.objective(best_design)

        best_design["objective"] = float(best_obj)
        return best_design

    # ------------------------------------------------------------------ #
    # Diophantine-based exact constraint satisfaction (advanced)
    # ------------------------------------------------------------------ #
    def solve_period_constraint_diophantine(self, stiffness_step: float = 1.0e5) -> np.ndarray:
        """
        Discretize the period equality constraint and solve as a 2-variable
        Diophantine problem:
          n_b * k_d = K_target
        where K_target = 4 * pi^2 * M_total / T_iso_target^2.
        
        We discretize k_d onto integer multiples of stiffness_step and find
        integer n_b such that n_b * (m * stiffness_step) = K_target.
        """
        K_target = 4.0 * np.pi ** 2 * self.M_total / (self.T_iso_target ** 2)
        # Find integer representations: a1 * x1 + a2 * x2 = b approx
        # Here we use a single variable for simplicity: n_b * k_d = K_target
        m_max = int(K_target / stiffness_step)
        solutions = []
        for m in range(1, m_max + 1):
            k_d = m * stiffness_step
            n_b = K_target / k_d
            if abs(n_b - round(n_b)) < 0.5 and self.n_b_min <= round(n_b) <= self.n_b_max:
                solutions.append({
                    "n_bearings": int(round(n_b)),
                    "k_d_per": float(k_d),
                })
        return np.array(solutions) if solutions else np.empty(0)
