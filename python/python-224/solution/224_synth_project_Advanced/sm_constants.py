"""
sm_constants.py
===============
Standard Model fundamental constants, couplings, and derived quantities.

All values follow the 2022 PDG review unless otherwise noted.
Key formulas:
    - Electromagnetic coupling: alpha_em(m_Z) = 1/127.955
    - Weak mixing angle: sin^2(theta_W) = 1 - (m_W/m_Z)^2
    - SU(2)_L coupling: g = e / sin(theta_W)
    - U(1)_Y coupling: g' = e / cos(theta_W)
    - Strong coupling: alpha_s(m_Z) = 0.1180
    - Top Yukawa: y_t = sqrt(2) * m_t / v
    - Higgs quartic: lambda = m_h^2 / (2 v^2)
    - VEV: v = (sqrt(2) G_F)^(-1/2) = 246.22 GeV

Vacuum stability conditions at tree level:
    lambda(mu) > 0          (stability)
    lambda(mu) + y_t^4/(4 pi^2) * log(...) > 0   (metastability)
"""
from __future__ import annotations
import numpy as np


class SMConstants:
    """
    Container for Standard Model parameters at the electroweak scale Q = m_Z.

    The class provides both input parameters (m_Z, m_W, m_t, m_h, alpha_em,
    alpha_s, G_F) and derived quantities (g, g', gs, y_t, lambda, v).
    """

    def __init__(
        self,
        m_Z: float = 91.1876,
        m_W: float = 80.385,
        m_t: float = 173.0,
        m_h: float = 125.09,
        alpha_em: float = 1.0 / 127.955,
        alpha_s: float = 0.1180,
        G_F: float = 1.1663788e-5,
        v: float | None = None,
    ) -> None:
        self.m_Z = float(m_Z)
        self.m_W = float(m_W)
        self.m_t = float(m_t)
        self.m_h = float(m_h)
        self.alpha_em = float(alpha_em)
        self.alpha_s = float(alpha_s)
        self.G_F = float(G_F)

        # VEV: either given or derived from G_F via v = (sqrt(2) G_F)^(-1/2)
        if v is None:
            self.v = 1.0 / np.sqrt(np.sqrt(2.0) * self.G_F)
        else:
            self.v = float(v)

        # Electromagnetic coupling e = sqrt(4 pi alpha_em)
        self.e = np.sqrt(4.0 * np.pi * self.alpha_em)

        # Weak mixing angle: sin^2 theta_W = 1 - (m_W/m_Z)^2 (on-shell scheme)
        self.sw2 = 1.0 - (self.m_W / self.m_Z) ** 2
        self.cw2 = 1.0 - self.sw2
        self.sw = np.sqrt(max(self.sw2, 0.0))
        self.cw = np.sqrt(max(self.cw2, 0.0))

        # Gauge couplings
        self.g = self.e / self.sw       # SU(2)_L
        self.gp = self.e / self.cw      # U(1)_Y
        self.gs = np.sqrt(4.0 * np.pi * self.alpha_s)  # SU(3)_c

        # Yukawa couplings (fermion masses: m_f = y_f v / sqrt(2))
        self.y_t = np.sqrt(2.0) * self.m_t / self.v
        self.y_b = 0.024   # approx y_b(m_b) ~ 0.024
        self.y_tau = 0.0102

        # Higgs quartic coupling at tree level: lambda = m_h^2 / (2 v^2)
        self.lam_tree = self.m_h ** 2 / (2.0 * self.v ** 2)

        # Renormalization scale Q (chosen as m_t by convention)
        self.Q_ref = self.m_t

        # Particle degrees of freedom for Coleman-Weinberg sum (sign: - for fermions)
        # n_i: dof * sign * color
        self.dof = {
            "top": -12.0,    # 3 colors * 2 (particle/antiparticle) * 2 (chiral) * (-1)
            "W": 6.0,        # 2 charges * 3 polarizations (massive)
            "Z": 3.0,
            "H": 1.0,        # physical Higgs
            "G": 3.0,        # 3 Goldstone bosons (would-be longitudinal W/Z)
        }

    def summary(self) -> str:
        lines = [
            "SMConstants @ Q = m_Z:",
            f"  m_Z = {self.m_Z:.4f} GeV, m_W = {self.m_W:.4f} GeV",
            f"  m_t = {self.m_t:.2f} GeV, m_h = {self.m_h:.2f} GeV",
            f"  v   = {self.v:.4f} GeV (from G_F = {self.G_F:.4e})",
            f"  alpha_em = {self.alpha_em:.6f}, alpha_s = {self.alpha_s:.4f}",
            f"  sin^2 theta_W = {self.sw2:.6f}",
            f"  g = {self.g:.6f}, g' = {self.gp:.6f}, gs = {self.gs:.6f}",
            f"  y_t = {self.y_t:.6f}, lambda = {self.lam_tree:.6f}",
        ]
        return "\n".join(lines)
