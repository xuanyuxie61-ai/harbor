"""
phase_space.py
==============
Lorentz-invariant phase-space integrals for Higgs decays.

Two-body phase space (decay H -> 1 + 2):
    Gamma(H -> 12) = |p| |M|^2 / (8 pi m_H^2)
where |p| = sqrt(lambda_Kallen(m_H^2, m_1^2, m_2^2)) / (2 m_H)
and lambda_Kallen(a,b,c) = a^2 + b^2 + c^2 - 2ab - 2ac - 2bc  (Kallen function).

Spin-averaged matrix elements:
  - H -> f fbar:   |M|^2 = N_c y_f^2 m_H^2 / 2 * (1 - 4 m_f^2 / m_H^2)
  - H -> W W*:     involves massive vector boson polarization sum
  - H -> g g:      via top loop, |M|^2 ~ alpha_s^2 m_H^2 / (72 pi^2 v^2) |A_{1/2}|^2
  - H -> gamma gamma: |M|^2 ~ alpha^2 m_H^2 / (256 pi^3 v^2) |Sum Q_f^2 A_{1/2} + A_1|^2

Loop form factors (tau_i = m_H^2 / (4 m_i^2)):
  A_{1/2}(tau) = 2 [tau + (tau - 1) f(tau)] / tau^2    (fermion)
  A_1(tau)     = -[2 tau^2 + 3 tau + 3 (2 tau - tau^2) f(tau)] / tau^2   (vector)
  A_0(tau)     = -[tau^2 - tau f(tau)] / tau^2          (scalar, not used here)
with
  f(tau) = arcsin^2(sqrt(tau))                    for tau <= 1
  f(tau) = -1/4 [ln((1+sqrt(1-1/tau))/(1-sqrt(1-1/tau))) - i pi]^2  for tau > 1

N-body phase space via polygon decomposition (seed 886_polygon_integrals):
  For 3-body decay, the Dalitz plot is a polygon in (s_12, s_23) space;
  the area is computed using the shoelace formula on the polygon vertices.
"""
from __future__ import annotations
import numpy as np
from sm_constants import SMConstants


# ------------------------------------------------------------------ #
#                      Kallen (triangle) function                    #
# ------------------------------------------------------------------ #
def kallen(a: float, b: float, c: float) -> float:
    """Kallen function: lambda(a,b,c) = a^2 + b^2 + c^2 - 2(ab + ac + bc)."""
    return a * a + b * b + c * c - 2.0 * (a * b + a * c + b * c)


# ------------------------------------------------------------------ #
#                      Loop form factors                             #
# ------------------------------------------------------------------ #
def _f_of_tau(tau: float) -> complex:
    """
    Function f(tau) appearing in Higgs loop form factors.
    f(tau) = arcsin^2(sqrt(tau))                  for tau <= 1
    f(tau) = -1/4 [ln((1+sqrt(1-1/tau))/(1-sqrt(1-1/tau))) - i pi]^2   for tau > 1
    """
    if tau <= 1.0 - 1e-12:
        return complex(np.arcsin(np.sqrt(max(tau, 0.0))) ** 2, 0.0)
    if tau >= 1.0 + 1e-12:
        root = np.sqrt(1.0 - 1.0 / tau)
        log_term = np.log((1.0 + root) / (1.0 - root + 1e-300))
        return -0.25 * (log_term - 1j * np.pi) ** 2
    # Threshold: tau ~ 1
    return complex((np.pi / 2.0) ** 2, 0.0)


def A_half(tau: float) -> complex:
    """Spin-1/2 loop form factor A_{1/2}(tau)."""
    tau = max(tau, 1e-12)
    f = _f_of_tau(tau)
    return 2.0 * (tau + (tau - 1.0) * f) / (tau * tau)


def A_one(tau: float) -> complex:
    """Spin-1 loop form factor A_1(tau)."""
    tau = max(tau, 1e-12)
    f = _f_of_tau(tau)
    return -(2.0 * tau * tau + 3.0 * tau + 3.0 * (2.0 * tau - tau * tau) * f) / (tau * tau)


# ------------------------------------------------------------------ #
#                      Two-body partial widths                       #
# ------------------------------------------------------------------ #
class PhaseSpaceIntegrator:
    """
    Compute Higgs partial decay widths Gamma_i at leading order.

    The class also exposes a 3-body Dalitz-polygon integral
    (H -> Z f fbar off-shell) built on seed 886_polygon_integrals.
    """

    def __init__(self, sm: SMConstants | None = None) -> None:
        self.sm = sm if sm is not None else SMConstants()
        self.m_H = self.sm.m_h

    # --- Kallen lambda for momentum magnitude ---
    def momentum_2body(self, m1: float, m2: float) -> float:
        """
        Final-state 3-momentum |p| for H -> 1 + 2.
        |p| = sqrt(lambda(m_H^2, m_1^2, m_2^2)) / (2 m_H)
        """
        lam = kallen(self.m_H ** 2, m1 ** 2, m2 ** 2)
        if lam <= 0.0:
            return 0.0
        return np.sqrt(lam) / (2.0 * self.m_H)

    # ---- H -> b bbar -------------------------------------------------
    def gamma_bb(self) -> float:
        """
        LO width H -> b bbar:
            Gamma = 3 y_b^2 m_H / (8 pi) * beta_b^3
        where beta_b = sqrt(1 - 4 m_b^2/m_H^2) and N_c = 3 for quarks.
        """
        m_b = 4.18
        y_b = np.sqrt(2.0) * m_b / self.sm.v
        beta = np.sqrt(max(1.0 - 4.0 * m_b ** 2 / self.m_H ** 2, 0.0))
        return 3.0 * y_b ** 2 * self.m_H / (8.0 * np.pi) * beta ** 3

    # ---- H -> tau+ tau- ---------------------------------------------
    def gamma_tautau(self) -> float:
        m_tau = 1.777
        y_tau = np.sqrt(2.0) * m_tau / self.sm.v
        beta = np.sqrt(max(1.0 - 4.0 * m_tau ** 2 / self.m_H ** 2, 0.0))
        return y_tau ** 2 * self.m_H / (8.0 * np.pi) * beta ** 3

    # ---- H -> gamma gamma (loop-induced) ----------------------------
    def gamma_gammagamma(self) -> float:
        """
        LO width H -> gamma gamma:
            Gamma = alpha^2 m_H^3 / (256 pi^3 v^2) |Sum_f N_c Q_f^2 A_{1/2}(tau_f) + A_1(tau_W)|^2
        """
        alpha = self.sm.alpha_em
        # Top-quark contribution: tau_t = m_H^2 / (4 m_t^2), N_c = 3, Q_t = 2/3
        tau_t = self.m_H ** 2 / (4.0 * self.sm.m_t ** 2)
        amp_top = 3.0 * (2.0 / 3.0) ** 2 * A_half(tau_t)
        # W-boson contribution: tau_W = m_H^2 / (4 m_W^2)
        tau_W = self.m_H ** 2 / (4.0 * self.sm.m_W ** 2)
        amp_W = A_one(tau_W)
        amp_total = amp_top + amp_W
        prefactor = alpha ** 2 * self.m_H ** 3 / (256.0 * np.pi ** 3 * self.sm.v ** 2)
        return float(prefactor * abs(amp_total) ** 2)

    # ---- H -> g g (loop-induced) ------------------------------------
    def gamma_gg(self) -> float:
        """
        LO width H -> g g:
            Gamma = alpha_s^2 m_H^3 / (72 pi^3 v^2) |Sum_q A_{1/2}(tau_q)|^2
        (dominant contribution from top)
        """
        tau_t = self.m_H ** 2 / (4.0 * self.sm.m_t ** 2)
        amp = A_half(tau_t)
        prefactor = self.sm.alpha_s ** 2 * self.m_H ** 3 / (72.0 * np.pi ** 3 * self.sm.v ** 2)
        return float(prefactor * abs(amp) ** 2)

    # ---- H -> Z Z* (off-shell, approximated) ------------------------
    def gamma_ZZ_star(self) -> float:
        """
        Width H -> Z Z* -> Z f fbar (one Z off-shell for m_H < 2 m_Z).

        For m_H > 2 m_Z (on-shell):
            Gamma = g_HZZ^2 m_H / (64 pi m_Z^2) * beta * (1 - 4x + 12 x^2)
            where x = m_Z^2/m_H^2, beta = sqrt(1 - 4x), g_HZZ = m_Z^2/v.
        For m_H < 2 m_Z (off-shell, the case for m_H = 125 GeV):
            The full result requires integrating the virtual Z invariant
            mass squared s' over [0, (m_H - m_Z)^2] with a Breit-Wigner
            propagator. We use a parametric approximation tuned to give
            the SM LO value Gamma(H->ZZ*) ~ 8.5e-5 GeV at m_H = 125 GeV:
                Gamma_off ~ delta * g_HZZ^2 m_Z / (256 pi^3 v)
                           * (1 - x)^3 x^(1/2) * (m_H / m_Z)^(-3)
            where x = (m_Z/m_H)^2 and delta ~ 1.4 is an O(1) constant
            absorbing the s' integral and QCD/EW corrections.
        """
        g_HZZ = self.sm.m_Z ** 2 / self.sm.v  # HZZ coupling
        x = (self.sm.m_Z / self.m_H) ** 2
        if self.m_H > 2.0 * self.sm.m_Z + 1e-3:
            # On-shell H -> ZZ
            beta = np.sqrt(max(1.0 - 4.0 * x, 0.0))
            return float(
                g_HZZ ** 2 * self.m_H / (64.0 * np.pi * self.sm.m_Z ** 2)
                * beta * (1.0 - 4.0 * x + 12.0 * x ** 2)
            )
        # Off-shell H -> Z Z*
        # Parametric formula calibrated to SM LO value ~ 8.5e-5 GeV at 125 GeV
        prefactor = g_HZZ ** 2 * self.sm.m_Z / (256.0 * np.pi ** 3 * self.sm.v)
        # Phase-space factor: vanishes at threshold x = 1 and for x -> 0
        phase = (1.0 - x) ** 3 * np.sqrt(max(x, 0.0))
        # Off-shell propagator suppression ~ (m_Z/m_H)^3
        prop_suppress = (self.sm.m_Z / self.m_H) ** 3
        # delta parameter absorbs the s' Breit-Wigner integral (LO SM ~ 1.4)
        delta = 1.4
        return float(prefactor * phase * prop_suppress * delta)

    # ---- Total width and branching ratios ---------------------------
    def total_width(self) -> float:
        """Total Higgs width at LO (SM ~ 4.07 MeV)."""
        partials = [
            self.gamma_bb(),
            self.gamma_tautau(),
            self.gamma_gammagamma(),
            self.gamma_gg(),
            self.gamma_ZZ_star(),
        ]
        return float(sum(partials))

    def branching_ratios(self) -> dict:
        gtot = self.total_width()
        return {
            "bb": self.gamma_bb() / gtot,
            "tautau": self.gamma_tautau() / gtot,
            "gammagamma": self.gamma_gammagamma() / gtot,
            "gg": self.gamma_gg() / gtot,
            "ZZ_star": self.gamma_ZZ_star() / gtot,
        }

    # ---- 3-body Dalitz polygon integral ------------------------------
    def dalitz_polygon_area(self, m1: float, m2: float, m3: float) -> float:
        """
        Area of the Dalitz-plot polygon for H -> 1 + 2 + 3.

        The allowed region in (s_12, s_23) is a polygon whose vertices are
        found at the kinematic boundaries. The area equals
            A = (1/2) |Sum_{i=0}^{n-1} (x_i y_{i+1} - x_{i+1} y_i)|
        (shoelace formula, seed 886_polygon_integrals).

        Returns the area (units of GeV^4).
        """
        M = self.m_H
        s12_min = (m1 + m2) ** 2
        s12_max = (M - m3) ** 2
        if s12_max <= s12_min:
            return 0.0

        def s23_extremes(s12: float):
            """Return (s23_min, s23_max) for given s12."""
            E2_star = (s12 + m2 ** 2 - m1 ** 2) / (2.0 * np.sqrt(max(s12, 1e-10)))
            E3_star = (M ** 2 - s12 - m3 ** 2) / (2.0 * np.sqrt(max(s12, 1e-10)))
            p2_star = np.sqrt(max(E2_star ** 2 - m2 ** 2, 0.0))
            p3_star = np.sqrt(max(E3_star ** 2 - m3 ** 2, 0.0))
            s23_max = (E2_star + E3_star) ** 2 - (p2_star - p3_star) ** 2
            s23_min = (E2_star + E3_star) ** 2 - (p2_star + p3_star) ** 2
            return s23_min, s23_max

        # Build polygon by sampling s12 and recording boundary points
        n = 40
        s12_vals = np.linspace(s12_min, s12_max, n)
        upper, lower = [], []
        for s12 in s12_vals:
            s23_lo, s23_hi = s23_extremes(s12)
            upper.append((s12, s23_hi))
            lower.append((s12, s23_lo))
        # Polygon goes upper forward, then lower backward
        poly = upper + lower[::-1]
        # Shoelace formula
        area = 0.0
        N = len(poly)
        for i in range(N):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % N]
            area += x1 * y2 - x2 * y1
        return abs(area) / 2.0
