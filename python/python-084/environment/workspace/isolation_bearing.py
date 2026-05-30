# -*- coding: utf-8 -*-

import numpy as np
from typing import Tuple


class LeadRubberBearing:

    def __init__(self, Q_d: float = 5.0e5, k_d: float = 1.5e7, k_e: float = None, n_wen: float = 1.0):
        if Q_d <= 0:
            raise ValueError("Characteristic strength Q_d must be positive")
        if k_d <= 0:
            raise ValueError("Post-yield stiffness k_d must be positive")

        self.Q_d = float(Q_d)
        self.k_d = float(k_d)
        self.k_e = float(k_e) if k_e is not None else 10.0 * k_d
        self.n_wen = float(n_wen)

        if self.k_e <= self.k_d:
            raise ValueError("Elastic stiffness k_e must exceed post-yield stiffness k_d")

        self.d_y = self.Q_d / (self.k_e - self.k_d)
        self._z = 0.0
        self._u_prev = 0.0
        self._F_prev = 0.0




    def force(self, u: float, du: float) -> float:

        raise NotImplementedError("Hole 2: Implement bilinear hysteresis force")

    def force_vectorized(self, u: np.ndarray, du: np.ndarray) -> np.ndarray:
        F = np.empty_like(u, dtype=float)
        for i in range(len(u)):
            F[i] = self.force(float(u[i]), float(du[i]))
        return F

    def reset_state(self):
        self._z = 0.0
        self._u_prev = 0.0
        self._F_prev = 0.0




    def energy_dissipation(self, d_max: float) -> float:
        if d_max <= self.d_y:
            return 0.0
        return 4.0 * self.Q_d * (d_max - self.d_y)




    def effective_stiffness(self, d_max: float) -> float:
        if abs(d_max) < 1e-12:
            return self.k_e
        F_max = self.Q_d * np.sign(d_max) + self.k_d * d_max
        return abs(F_max / d_max)

    def effective_damping_ratio(self, d_max: float) -> float:
        E_d = self.energy_dissipation(d_max)
        k_eff = self.effective_stiffness(d_max)
        if abs(d_max) < 1e-12 or k_eff <= 0:
            return 0.0
        zeta = E_d / (2.0 * np.pi * k_eff * d_max ** 2)
        return min(zeta, 0.5)




    def wen_dzdt(self, z: float, du_dt: float) -> float:
        if abs(self.d_y) < 1e-12:
            return 0.0
        term = 1.0 - (abs(z) ** self.n_wen) * np.sign(du_dt * z)
        return (du_dt / self.d_y) * term


class IsolationSystem:

    def __init__(self, n_bearings: int = 20, Q_d_per: float = 2.5e4, k_d_per: float = 7.5e5):
        self.n_bearings = n_bearings
        self.bearing = LeadRubberBearing(Q_d=Q_d_per, k_d=k_d_per)

        self.Q_d_total = n_bearings * Q_d_per
        self.k_d_total = n_bearings * k_d_per
        self.k_e_total = n_bearings * self.bearing.k_e

    def total_force(self, u: float, du: float) -> float:
        return self.n_bearings * self.bearing.force(u, du)

    def reset(self):
        self.bearing.reset_state()

    def effective_period(self, M_base: float) -> float:

        k_eff = self.k_d_total
        if k_eff <= 0:
            return np.inf
        return 2.0 * np.pi * np.sqrt(M_base / k_eff)
