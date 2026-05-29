# -*- coding: utf-8 -*-
"""
response_analysis.py
====================
Post-processing of seismic time-history responses, incorporating ideas from:
  - 017_area_under_curve:  Numerical integration for energy and cumulative
                           quantities (velocity/displacement from acceleration)
  - 1423_xyz_display:      3-D node coordinate transformations for drift and
                           inter-story displacement profiles

Quantities computed:
  - Floor displacements u(t)
  - Inter-story drift ratios:  drift_i = (u_i - u_{i-1}) / h_i
  - Floor absolute accelerations:  a_abs = a + Gamma * a_g
  - Story shear forces:  V_i = k_i * (u_i - u_{i-1})
  - Base shear:  V_b = sum_i V_i
  - Input energy:  E_in = -integral( M * a_g * v_base  dt )
  - Hysteretic energy:  E_h = integral( F_iso * v_iso  dt )
  - Kinetic energy:  E_k = 0.5 * v^T * M * v
  - Strain energy:   E_s = 0.5 * u^T * K * u
  - Damping energy:  E_d = integral( v^T * C * v  dt )
"""

import numpy as np
from typing import Tuple, Optional


# ====================================================================== #
# Numerical integration (from 017_area_under_curve seed)
# ====================================================================== #
def trapz_integral(x: np.ndarray, y: np.ndarray) -> float:
    """
    Trapezoidal rule for definite integral:
      I = sum_{i=1}^{N-1} 0.5 * (y_i + y_{i+1}) * (x_{i+1} - x_i)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dx = np.diff(x)
    avg_y = 0.5 * (y[:-1] + y[1:])
    return float(np.sum(avg_y * dx))


def cumtrapz(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integral."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    result = np.zeros(len(x), dtype=float)
    for i in range(1, len(x)):
        result[i] = result[i - 1] + 0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1])
    return result


# ====================================================================== #
# Response analyzer class
# ====================================================================== #
class ResponseAnalyzer:
    """
    Analyze seismic response time histories of a multi-DOF structure.
    """

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
        """
        Parameters
        ----------
        U, V, A : np.ndarray, shape (n_time, n_dof)
            Displacement, velocity, acceleration histories.
        t : np.ndarray, shape (n_time,)
            Time vector.
        M, C, K : np.ndarray
            Mass, damping, stiffness matrices.
        Gamma : np.ndarray
            Influence vector.
        a_g : np.ndarray
            Ground acceleration.
        story_heights : np.ndarray
            Story heights [m].
        iso_force_history : np.ndarray, optional
            Isolation force time history [N].
        """
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

    # ------------------------------------------------------------------ #
    # Inter-story drift
    # ------------------------------------------------------------------ #
    def interstory_drift(self) -> np.ndarray:
        """
        Compute inter-story drift ratios at each time step:
          drift_i(t) = (u_i(t) - u_{i-1}(t)) / h_i
        
        Returns
        -------
        drift : np.ndarray, shape (n_time, n_story)
        """
        drift = np.zeros((self.n_time, self.n_story), dtype=float)
        for i in range(self.n_story):
            idx = i + 1   # story i sits between DOF i and i+1
            du = self.U[:, idx] - self.U[:, idx - 1]
            h = self.story_heights[i]
            if h > 1e-12:
                drift[:, i] = du / h
            else:
                drift[:, i] = 0.0
        return drift

    def max_drift(self) -> np.ndarray:
        """Maximum absolute inter-story drift ratio per story."""
        drift = self.interstory_drift()
        return np.max(np.abs(drift), axis=0)

    # ------------------------------------------------------------------ #
    # Story shear forces
    # ------------------------------------------------------------------ #
    def story_shear(self) -> np.ndarray:
        """
        Compute story shear force time histories:
          V_i(t) = k_i * (u_i(t) - u_{i-1}(t))
        
        Returns
        -------
        V : np.ndarray, shape (n_time, n_story)
        """
        V = np.zeros((self.n_time, self.n_story), dtype=float)
        for i in range(self.n_story):
            idx = i + 1
            du = self.U[:, idx] - self.U[:, idx - 1]
            # Extract story stiffness from K matrix
            k_i = self.K[idx, idx] + self.K[idx - 1, idx - 1]
            if k_i > 0:
                V[:, i] = k_i * du
        return V

    def base_shear(self) -> np.ndarray:
        """Total base shear time history [N]."""
        V = self.story_shear()
        return np.sum(V, axis=1)

    def max_base_shear(self) -> float:
        return float(np.max(np.abs(self.base_shear())))

    # ------------------------------------------------------------------ #
    # Absolute floor accelerations
    # ------------------------------------------------------------------ #
    def absolute_acceleration(self) -> np.ndarray:
        """
        Absolute acceleration = relative acceleration + ground acceleration.
          a_abs(t) = a(t) + Gamma * a_g(t)
        """
        return self.A + np.outer(self.a_g, self.Gamma)

    def max_absolute_acceleration(self) -> np.ndarray:
        """Peak absolute acceleration per DOF."""
        a_abs = self.absolute_acceleration()
        return np.max(np.abs(a_abs), axis=0)

    # ------------------------------------------------------------------ #
    # Floor displacement profile (3-D coordinates, from xyz_display seed)
    # ------------------------------------------------------------------ #
    def displaced_coordinates(self, base_coords: np.ndarray) -> np.ndarray:
        """
        Compute displaced 3-D coordinates of the building at each time step.
        
        base_coords : np.ndarray, shape (n_dof, 3)
            Undisplaced node coordinates.
        
        Returns
        -------
        coords_t : np.ndarray, shape (n_time, n_dof, 3)
        """
        base_coords = np.asarray(base_coords, dtype=float)
        coords_t = np.zeros((self.n_time, self.n_dof, 3), dtype=float)
        for it in range(self.n_time):
            coords_t[it, :, :] = base_coords.copy()
            # Apply lateral displacement in x-direction
            coords_t[it, :, 0] += self.U[it, :]
        return coords_t

    # ------------------------------------------------------------------ #
    # Energy quantities (from area_under_curve numerical integration)
    # ------------------------------------------------------------------ #
    def input_energy(self) -> np.ndarray:
        """
        Cumulative input energy from ground motion:
          E_in(t) = -integral_0^t  v_base(tau) * M_total * a_g(tau)  dtau
        """
        v_base = self.V[:, 0]   # isolation layer velocity
        M_total = float(np.sum(self.M))
        power = -v_base * M_total * self.a_g
        return cumtrapz(self.t, power)

    def kinetic_energy(self) -> np.ndarray:
        """
        Kinetic energy:
          E_k(t) = 0.5 * v(t)^T * M * v(t)
        """
        E_k = np.zeros(self.n_time, dtype=float)
        for it in range(self.n_time):
            v = self.V[it, :]
            E_k[it] = 0.5 * float(v @ self.M @ v)
        return E_k

    def strain_energy(self) -> np.ndarray:
        """
        Elastic strain energy:
          E_s(t) = 0.5 * u(t)^T * K * u(t)
        """
        E_s = np.zeros(self.n_time, dtype=float)
        for it in range(self.n_time):
            u = self.U[it, :]
            E_s[it] = 0.5 * float(u @ self.K @ u)
        return E_s

    def damping_energy(self) -> np.ndarray:
        """
        Cumulative energy dissipated by viscous damping:
          E_d(t) = integral_0^t  v(tau)^T * C * v(tau)  dtau
        """
        power = np.zeros(self.n_time, dtype=float)
        for it in range(self.n_time):
            v = self.V[it, :]
            power[it] = float(v @ self.C @ v)
        return cumtrapz(self.t, power)

    def hysteretic_energy(self) -> np.ndarray:
        """
        Cumulative hysteretic energy dissipated by isolation bearings:
          E_h(t) = integral_0^t  F_iso(tau) * v_iso(tau)  dtau
        """
        if self.iso_force_history is None:
            return np.zeros(self.n_time, dtype=float)
        v_iso = self.V[:, 0]
        power = self.iso_force_history * v_iso
        return cumtrapz(self.t, power)

    def total_energy(self) -> np.ndarray:
        """Sum of kinetic + strain + damping + hysteretic energies."""
        # TODO: Hole 3 - Implement total energy summation
        raise NotImplementedError("Hole 3: Implement total energy")

    def energy_balance_error(self) -> float:
        """
        Check energy balance:
          E_in(t) ~= E_k(t) + E_s(t) + E_d(t) + E_h(t)
        Returns the maximum relative error over the time history.
        """
        # TODO: Hole 3 - Implement energy balance error computation
        raise NotImplementedError("Hole 3: Implement energy balance error")

    # ------------------------------------------------------------------ #
    # Summary metrics
    # ------------------------------------------------------------------ #
    def summary(self) -> dict:
        """Return a dictionary of key response metrics."""
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
