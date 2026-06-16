"""
higgs_potential.py
==================
SM Higgs effective potential V(phi) at tree and one-loop (Coleman-Weinberg).

Tree-level potential in unitary gauge (phi = sqrt(2 phi^dagger phi)):
    V_0(phi) = -mu^2 phi^2 / 2 + lambda phi^4 / 4
    with mu^2 = lambda v^2 and v ~ 246.22 GeV.

One-loop Coleman-Weinberg correction in MS-bar:
    V_1(phi) = Sum_i  n_i M_i(phi)^4 / (64 pi^2) * [ln(M_i(phi)^2/Q^2) - c_i]

    where c_i = 3/2 in MS-bar for all SM particles.

Field-dependent masses:
    m_t(phi) = y_t phi / sqrt(2)
    m_W(phi) = g phi / 2
    m_Z(phi) = sqrt(g^2 + g'^2) phi / 2
    m_H(phi) = sqrt(-mu^2 + 3 lambda phi^2)    for phi > v / sqrt(3)
    m_G(phi) = sqrt(-mu^2 + lambda phi^2)      for phi > v

Goldstone modes become tachyonic below phi = v; we regulate via m^2 -> max(m^2, 0).

Minimum at phi = v is found via bisection on dV/dphi (incorporating seed 806_nonlin_bisect).
Exact analytical solution for mu^2 is: mu^2 = lambda v^2 (seed 762_mhd_exact).
"""
from __future__ import annotations
import numpy as np
from sm_constants import SMConstants


class HiggsPotential:
    """
    One-loop effective Higgs potential in the SM.
    """

    def __init__(self, sm: SMConstants | None = None) -> None:
        self.sm = sm if sm is not None else SMConstants()
        # Derived: mu^2 = lambda * v^2 (exact tree-level relation)
        self.mu2 = self.sm.lam_tree * self.sm.v ** 2

    # ------------------------------------------------------------------ #
    #                       Field-dependent masses                       #
    # ------------------------------------------------------------------ #
    def field_dependent_masses(self, phi: np.ndarray) -> dict:
        """
        Field-dependent particle masses M_i(phi).
        Scalar masses are regulated to avoid tachyonic regions below the VEV.
        """
        phi = np.asarray(phi, dtype=float)
        phi = np.clip(np.abs(phi), 0.0, 1e4)  # prevent overflow
        m_top = self.sm.y_t * phi / np.sqrt(2.0)
        m_W = self.sm.g * phi / 2.0
        m_Z = np.sqrt(self.sm.g ** 2 + self.sm.gp ** 2) * phi / 2.0
        m2_H = -self.mu2 + 3.0 * self.sm.lam_tree * phi ** 2
        m2_G = -self.mu2 + self.sm.lam_tree * phi ** 2
        m_H = np.sqrt(np.maximum(m2_H, 0.0))
        m_G = np.sqrt(np.maximum(m2_G, 0.0))
        return {"top": m_top, "W": m_W, "Z": m_Z, "H": m_H, "G": m_G}

    # ------------------------------------------------------------------ #
    #                    Potential: tree and 1-loop                      #
    # ------------------------------------------------------------------ #
    def tree_level(self, phi: np.ndarray) -> np.ndarray:
        """V_0(phi) = -mu^2 phi^2/2 + lambda phi^4/4."""
        phi = np.asarray(phi, dtype=float)
        phi = np.clip(phi, -1e5, 1e5)  # prevent overflow
        return -0.5 * self.mu2 * phi ** 2 + 0.25 * self.sm.lam_tree * phi ** 4

    def one_loop_coleman_weinberg(self, phi: np.ndarray) -> np.ndarray:
        """
        One-loop Coleman-Weinberg correction in MS-bar.
        V_1 = Sum_i n_i M_i^4 / (64 pi^2) * [ln(M_i^2/Q^2) - c_i]
        """
        phi = np.asarray(phi, dtype=float)
        phi = np.clip(phi, -1e4, 1e4)  # prevent overflow
        masses = self.field_dependent_masses(phi)
        Q = self.sm.Q_ref
        c_msbar = 1.5
        V1 = np.zeros_like(phi, dtype=float)
        for name, m in masses.items():
            m_safe = np.maximum(np.abs(m), 1e-6)
            m_safe = np.minimum(m_safe, 1e4)  # cap to prevent m^4 overflow
            ni = self.sm.dof[name]
            V1 = V1 + ni * m_safe ** 4 / (64.0 * np.pi ** 2) * (
                np.log(m_safe ** 2 / Q ** 2) - c_msbar
            )
        return V1

    def effective_potential(self, phi: np.ndarray) -> np.ndarray:
        """Full one-loop effective potential V_eff = V_0 + V_1."""
        return self.tree_level(phi) + self.one_loop_coleman_weinberg(phi)

    # ------------------------------------------------------------------ #
    #                  High-order finite-difference derivatives          #
    # ------------------------------------------------------------------ #
    def dV_dphi(self, phi: np.ndarray, order: int = 6, h: float = 1e-3) -> np.ndarray:
        """
        First derivative via central FD.
        4th order: f' ~ (-f(x+2h) + 8 f(x+h) - 8 f(x-h) + f(x-2h)) / (12h)
        6th order: f' ~ (f(x+3h) - 9 f(x+2h) + 45 f(x+h) - 45 f(x-h) + 9 f(x-2h) - f(x-3h)) / (60h)
        """
        phi = np.asarray(phi, dtype=float)
        f = self.effective_potential
        if order == 4:
            return (-f(phi + 2 * h) + 8.0 * f(phi + h) - 8.0 * f(phi - h) + f(phi - 2 * h)) / (12.0 * h)
        if order == 6:
            return (
                f(phi + 3 * h) - 9.0 * f(phi + 2 * h) + 45.0 * f(phi + h)
                - 45.0 * f(phi - h) + 9.0 * f(phi - 2 * h) - f(phi - 3 * h)
            ) / (60.0 * h)
        if order == 8:
            return (
                -f(phi + 4 * h) + (32.0 / 3.0) * f(phi + 3 * h)
                - 56.0 * f(phi + 2 * h) + (224.0) * f(phi + h)
                - (224.0) * f(phi - h) + 56.0 * f(phi - 2 * h)
                - (32.0 / 3.0) * f(phi - 3 * h) + f(phi - 4 * h)
            ) / (280.0 * h)
        raise ValueError(f"Unsupported FD order {order}")

    def d2V_dphi2(self, phi: np.ndarray, h: float = 5e-3) -> np.ndarray:
        """
        Second derivative via 4th-order central FD for the second derivative:
        f'' ~ (-f(x+2h) + 16 f(x+h) - 30 f(x) + 16 f(x-h) - f(x-2h)) / (12 h^2)
        """
        phi = np.asarray(phi, dtype=float)
        f = self.effective_potential
        return (
            -f(phi + 2 * h) + 16.0 * f(phi + h) - 30.0 * f(phi)
            + 16.0 * f(phi - h) - f(phi - 2 * h)
        ) / (12.0 * h ** 2)

    # ------------------------------------------------------------------ #
    #                          Vacuum structure                          #
    # ------------------------------------------------------------------ #
    def lambda_eff(self, phi: np.ndarray | float | None = None) -> np.ndarray:
        """
        Effective quartic coupling:
            lambda_eff(phi) = 4 V_eff(phi) / phi^4
        Vacuum stability requires lambda_eff(phi) > 0 for all phi.
        """
        if phi is None:
            phi = np.array([self.sm.v])
        phi = np.maximum(np.abs(np.asarray(phi, dtype=float)), 1e-6)
        return 4.0 * self.effective_potential(phi) / phi ** 4

    def find_vev_bisection(
        self,
        phi_lo: float = 200.0,
        phi_hi: float = 300.0,
        tol: float = 1e-10,
        max_iter: int = 200,
    ) -> float:
        """
        Find the VEV phi = v by bisection on V'(phi) = 0.
        Inspired by seed 806_nonlin_bisect (bisection for nonlinear equations).
        """
        fa = self.dV_dphi(np.array([phi_lo]), order=6)[0]
        for _ in range(max_iter):
            mid = 0.5 * (phi_lo + phi_hi)
            fm = self.dV_dphi(np.array([mid]), order=6)[0]
            if abs(fm) < tol or (phi_hi - phi_lo) < tol:
                return mid
            if fa * fm < 0.0:
                phi_hi = mid
            else:
                phi_lo = mid
                fa = fm
        return 0.5 * (phi_lo + phi_hi)

    def curvature_at_minimum(self) -> float:
        """V''(v) should equal m_h^2 at tree level; deviations encode loop corrections."""
        return float(self.d2V_dphi2(np.array([self.sm.v]))[0])

    def barrier_height(self) -> float:
        """
        Height of the potential barrier between origin and VEV (electroweak phase transition).
        Find maximum of V(phi) in (0, v) via golden-section search.
        """
        phi_g = (np.sqrt(5.0) - 1.0) / 2.0  # golden ratio
        a, b = 1e-3, self.sm.v  # avoid phi=0 (singular)
        for _ in range(100):
            x1 = b - phi_g * (b - a)
            x2 = a + phi_g * (b - a)
            v1 = float(self.effective_potential(np.array([x1]))[0])
            v2 = float(self.effective_potential(np.array([x2]))[0])
            if v1 < v2:
                a = x1
            else:
                b = x2
        peak = float(self.effective_potential(np.array([0.5 * (a + b)]))[0])
        V_origin = float(self.effective_potential(np.array([1e-3]))[0])
        V_vev = float(self.effective_potential(np.array([self.sm.v]))[0])
        # Barrier height relative to the VEV
        return max(peak - V_vev, 0.0)
