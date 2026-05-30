# -*- coding: utf-8 -*-

import numpy as np
from typing import List, Tuple, Optional





def diophantine_nd_nonnegative(a: np.ndarray, b: int) -> np.ndarray:
    a = np.asarray(a, dtype=int).ravel()
    n = len(a)
    if n == 0 or b < 0:
        return np.empty((0, n), dtype=int)

    solutions = []
    y = np.zeros(n, dtype=int)
    j = 0

    while True:

        r = b - np.dot(a[:j], y[:j])

        if j < n:
            y[j] = r // a[j] if a[j] != 0 else 0
            j += 1
        else:
            if r == 0:
                solutions.append(y.copy())

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
    rng = np.random.default_rng(seed)
    samples = []

    for _ in range(n_samples * 10):
        if len(samples) >= n_samples:
            break
        n_b = rng.integers(n_b_min, n_b_max + 1)
        Q_d = rng.choice(Q_d_grid)
        k_d = rng.choice(k_d_grid)


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





class IsolationOptimizer:

    def __init__(
        self,
        M_total: float,
        W_total: float,
        T_iso_target: float = 2.5,
        d_y_max: float = 0.15,
        n_b_min: int = 4,
        n_b_max: int = 40,
    ):
        self.M_total = float(M_total)
        self.W_total = float(W_total)
        self.T_iso_target = float(T_iso_target)
        self.d_y_max = float(d_y_max)
        self.n_b_min = n_b_min
        self.n_b_max = n_b_max


        self.Q_d_grid = np.array([
            1.0e4, 1.5e4, 2.0e4, 2.5e4, 3.0e4, 4.0e4, 5.0e4,
            6.0e4, 8.0e4, 1.0e5, 1.2e5, 1.5e5, 2.0e5, 3.0e5,
        ], dtype=float)
        self.k_d_grid = np.array([
            2.0e5, 3.0e5, 4.0e5, 5.0e5, 6.0e5, 7.5e5,
            1.0e6, 1.25e6, 1.5e6, 2.0e6, 2.5e6, 3.0e6, 4.0e6,
        ], dtype=float)




    def objective(self, design: dict, spectral_accel: float = 2.5) -> float:
        w1 = 1.0
        w2 = 0.001

        n_b = design["n_bearings"]
        k_total = n_b * design["k_d_per"]
        period_est = 2.0 * np.pi * np.sqrt(self.M_total / k_total) if k_total > 0 else 1.0


        base_shear_ratio = spectral_accel / 9.81 * (period_est / 0.5) ** (-0.5)
        base_shear_ratio = min(base_shear_ratio, 1.0)

        cost = n_b * design["Q_d_per"] * 1e-5

        return w1 * base_shear_ratio + w2 * cost




    def optimize(self, n_samples: int = 200) -> dict:
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




    def solve_period_constraint_diophantine(self, stiffness_step: float = 1.0e5) -> np.ndarray:
        K_target = 4.0 * np.pi ** 2 * self.M_total / (self.T_iso_target ** 2)


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
