# -*- coding: utf-8 -*-

import numpy as np
from typing import Tuple, Optional





def trapz_integral(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dx = np.diff(x)
    avg_y = 0.5 * (y[:-1] + y[1:])
    return float(np.sum(avg_y * dx))


def cumtrapz(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    result = np.zeros(len(x), dtype=float)
    for i in range(1, len(x)):
        result[i] = result[i - 1] + 0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1])
    return result





class ResponseAnalyzer:

    def __init__(
        self,
        U: np.ndarray,
        V: np.ndarray,
        A: np.ndarray,
        t: np.ndarray,
        M: np.ndarray,
        C: np.ndarray,
        K: np.ndarray,
        Gamma: np.ndarray,
        a_g: np.ndarray,
        story_heights: np.ndarray,
        iso_force_history: Optional[np.ndarray] = None,
    ):
        self.U = np.asarray(U, dtype=float)
        self.V = np.asarray(V, dtype=float)
        self.A = np.asarray(A, dtype=float)
        self.t = np.asarray(t, dtype=float)
        self.M = np.asarray(M, dtype=float)
        self.C = np.asarray(C, dtype=float)
        self.K = np.asarray(K, dtype=float)
        self.Gamma = np.asarray(Gamma, dtype=float)
        self.a_g = np.asarray(a_g, dtype=float)
        self.story_heights = np.asarray(story_heights, dtype=float)
        self.iso_force_history = iso_force_history

        self.n_time, self.n_dof = self.U.shape
        self.n_story = len(story_heights)




    def interstory_drift(self) -> np.ndarray:
        drift = np.zeros((self.n_time, self.n_story), dtype=float)
        for i in range(self.n_story):
            idx = i + 1
            du = self.U[:, idx] - self.U[:, idx - 1]
            h = self.story_heights[i]
            if h > 1e-12:
                drift[:, i] = du / h
            else:
                drift[:, i] = 0.0
        return drift

    def max_drift(self) -> np.ndarray:
        drift = self.interstory_drift()
        return np.max(np.abs(drift), axis=0)




    def story_shear(self) -> np.ndarray:
        V = np.zeros((self.n_time, self.n_story), dtype=float)
        for i in range(self.n_story):
            idx = i + 1
            du = self.U[:, idx] - self.U[:, idx - 1]

            k_i = self.K[idx, idx] + self.K[idx - 1, idx - 1]
            if k_i > 0:
                V[:, i] = k_i * du
        return V

    def base_shear(self) -> np.ndarray:
        V = self.story_shear()
        return np.sum(V, axis=1)

    def max_base_shear(self) -> float:
        return float(np.max(np.abs(self.base_shear())))




    def absolute_acceleration(self) -> np.ndarray:
        return self.A + np.outer(self.a_g, self.Gamma)

    def max_absolute_acceleration(self) -> np.ndarray:
        a_abs = self.absolute_acceleration()
        return np.max(np.abs(a_abs), axis=0)




    def displaced_coordinates(self, base_coords: np.ndarray) -> np.ndarray:
        base_coords = np.asarray(base_coords, dtype=float)
        coords_t = np.zeros((self.n_time, self.n_dof, 3), dtype=float)
        for it in range(self.n_time):
            coords_t[it, :, :] = base_coords.copy()

            coords_t[it, :, 0] += self.U[it, :]
        return coords_t




    def input_energy(self) -> np.ndarray:
        v_base = self.V[:, 0]
        M_total = float(np.sum(self.M))
        power = -v_base * M_total * self.a_g
        return cumtrapz(self.t, power)

    def kinetic_energy(self) -> np.ndarray:
        E_k = np.zeros(self.n_time, dtype=float)
        for it in range(self.n_time):
            v = self.V[it, :]
            E_k[it] = 0.5 * float(v @ self.M @ v)
        return E_k

    def strain_energy(self) -> np.ndarray:
        E_s = np.zeros(self.n_time, dtype=float)
        for it in range(self.n_time):
            u = self.U[it, :]
            E_s[it] = 0.5 * float(u @ self.K @ u)
        return E_s

    def damping_energy(self) -> np.ndarray:
        power = np.zeros(self.n_time, dtype=float)
        for it in range(self.n_time):
            v = self.V[it, :]
            power[it] = float(v @ self.C @ v)
        return cumtrapz(self.t, power)

    def hysteretic_energy(self) -> np.ndarray:
        if self.iso_force_history is None:
            return np.zeros(self.n_time, dtype=float)
        v_iso = self.V[:, 0]
        power = self.iso_force_history * v_iso
        return cumtrapz(self.t, power)

    def total_energy(self) -> np.ndarray:

        raise NotImplementedError("Hole 3: Implement total energy")

    def energy_balance_error(self) -> float:

        raise NotImplementedError("Hole 3: Implement energy balance error")




    def summary(self) -> dict:
        drift_max = self.max_drift()
        a_max = self.max_absolute_acceleration()
        return {
            "max_isolation_displacement_m": float(np.max(np.abs(self.U[:, 0]))),
            "max_roof_displacement_m": float(np.max(np.abs(self.U[:, -1]))),
            "max_drift_ratio": float(np.max(drift_max)),
            "max_base_shear_kN": self.max_base_shear() / 1000.0,
            "max_floor_accel_g": float(np.max(a_max[1:])) / 9.81,
            "energy_balance_error": self.energy_balance_error(),
        }
