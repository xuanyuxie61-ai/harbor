# -*- coding: utf-8 -*-
"""
isolation_bearing.py
====================
Nonlinear hysteretic model for lead-core rubber base isolation bearings.

Core science:
  The lead-core rubber bearing (LRB) exhibits a bilinear hysteretic force-
  displacement relationship characterized by:
    - Characteristic strength:  Q_d  [N]
    - Post-elastic stiffness:   k_d  [N/m]
    - Initial (elastic) stiffness:  k_e  [N/m]
    - Yield displacement:       d_y = Q_d / (k_e - k_d)  [m]

  The restoring force is:
    F_iso = k_d * u + Q_d * z
  where z is a dimensionless hysteretic variable governed by:
    dz/dt = (1 / d_y) * ( du/dt - |du/dt| * |z|^{n} * z )
  (Wen-Bouc hysteresis model with n = 1 for bilinear approximation).

  For the bilinear simplification used here:
    - Loading branch:   F = k_e * u   for |u| <= d_y
    - Yielded branch:   F = Q_d * sgn(u) + k_d * u   for |u| > d_y

  Energy dissipation per cycle:
    E_d = 4 * Q_d * (d_max - d_y)
  where d_max is the maximum displacement.
"""

import numpy as np
from typing import Tuple


class LeadRubberBearing:
    """
    Lead-core rubber bearing (LRB) with bilinear hysteresis.
    
    Parameters
    ----------
    Q_d : float
        Characteristic strength [N].
    k_d : float
        Post-yield stiffness [N/m].
    k_e : float
        Initial elastic stiffness [N/m] (default 10 * k_d).
    n_wen : float
        Wen model exponent (default 1.0 for bilinear).
    """

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
        self._z = 0.0   # Hysteretic variable
        self._u_prev = 0.0
        self._F_prev = 0.0

    # ------------------------------------------------------------------ #
    # Bilinear force evaluation
    # ------------------------------------------------------------------ #
    def force(self, u: float, du: float) -> float:
        """
        Compute restoring force at displacement u with velocity du.
        
        Bilinear model:
          If loading and |u| <= d_y:    F = k_e * u
          If yielded:                    F = Q_d * sgn(u) + k_d * u
          If unloading:                  slope = k_e until |F - F_y| = 2*Q_d
        """
        # TODO: Hole 2 - Implement bilinear hysteresis force model
        raise NotImplementedError("Hole 2: Implement bilinear hysteresis force")

    def force_vectorized(self, u: np.ndarray, du: np.ndarray) -> np.ndarray:
        """Vectorized version for array inputs (stateless per call)."""
        F = np.empty_like(u, dtype=float)
        for i in range(len(u)):
            F[i] = self.force(float(u[i]), float(du[i]))
        return F

    def reset_state(self):
        """Reset hysteretic state to zero."""
        self._z = 0.0
        self._u_prev = 0.0
        self._F_prev = 0.0

    # ------------------------------------------------------------------ #
    # Energy dissipation
    # ------------------------------------------------------------------ #
    def energy_dissipation(self, d_max: float) -> float:
        """
        Approximate energy dissipated per cycle at amplitude d_max.
        
        E_d = 4 * Q_d * (d_max - d_y)   for d_max > d_y
        """
        if d_max <= self.d_y:
            return 0.0
        return 4.0 * self.Q_d * (d_max - self.d_y)

    # ------------------------------------------------------------------ #
    # Effective stiffness & damping
    # ------------------------------------------------------------------ #
    def effective_stiffness(self, d_max: float) -> float:
        """
        Secant stiffness at maximum displacement d_max:
          k_eff = F(d_max) / d_max
        """
        if abs(d_max) < 1e-12:
            return self.k_e
        F_max = self.Q_d * np.sign(d_max) + self.k_d * d_max
        return abs(F_max / d_max)

    def effective_damping_ratio(self, d_max: float) -> float:
        """
        Equivalent viscous damping ratio from energy dissipation:
                     E_d
          zeta_eff = -------------
                     2 * pi * k_eff * d_max^2
        """
        E_d = self.energy_dissipation(d_max)
        k_eff = self.effective_stiffness(d_max)
        if abs(d_max) < 1e-12 or k_eff <= 0:
            return 0.0
        zeta = E_d / (2.0 * np.pi * k_eff * d_max ** 2)
        return min(zeta, 0.5)   # Cap at 50% for numerical stability

    # ------------------------------------------------------------------ #
    # Wen-Bouc exact differential form (advanced)
    # ------------------------------------------------------------------ #
    def wen_dzdt(self, z: float, du_dt: float) -> float:
        """
        Wen-Bouc hysteretic evolution rate:
          dz/dt = (du/dt / d_y) * (1 - |z|^n * sgn(du/dt * z))
        
        For n=1 this produces a smooth transition between elastic and
        plastic behavior.
        """
        if abs(self.d_y) < 1e-12:
            return 0.0
        term = 1.0 - (abs(z) ** self.n_wen) * np.sign(du_dt * z)
        return (du_dt / self.d_y) * term


class IsolationSystem:
    """
    Array of lead-rubber bearings supporting the base isolation layer.
    """

    def __init__(self, n_bearings: int = 20, Q_d_per: float = 2.5e4, k_d_per: float = 7.5e5):
        self.n_bearings = n_bearings
        self.bearing = LeadRubberBearing(Q_d=Q_d_per, k_d=k_d_per)
        # Total isolation properties
        self.Q_d_total = n_bearings * Q_d_per
        self.k_d_total = n_bearings * k_d_per
        self.k_e_total = n_bearings * self.bearing.k_e

    def total_force(self, u: float, du: float) -> float:
        """Total restoring force from all bearings."""
        return self.n_bearings * self.bearing.force(u, du)

    def reset(self):
        self.bearing.reset_state()

    def effective_period(self, M_base: float) -> float:
        """
        Effective isolation period:
          T_eff = 2*pi * sqrt( M_base / k_eff )
        """
        # Assume small displacement for design estimate
        k_eff = self.k_d_total
        if k_eff <= 0:
            return np.inf
        return 2.0 * np.pi * np.sqrt(M_base / k_eff)
