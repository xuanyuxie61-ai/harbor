"""
physics_models.py
=================
Physical constants, Bloch-Torrey relaxation kernels, and reaction kinetics
for the KKT-constrained optimal control problem.

Scientific background
---------------------
The state equation is a steady-state Bloch-Torrey-type reaction-diffusion PDE:

    -nu * Laplacian(y) + R(y; T1, T2) = u + f    in Omega = [0,1]^2
    y = g                                         on dOmega

where the relaxation operator R models T1/T2 spin-lattice / spin-spin decay
(inspired by quantitative MRI, cf. wastehling T1T2-mapping):

    R(y; T1, T2) = kappa_1 * y  +  kappa_2 * y^3 / (1 + ||y||^2)

The linear term kappa_1 = 1/T1 captures longitudinal recovery; the cubic
saturation term approximates steady-state SSFP signal behaviour where
kappa_2 ~ (1/T2 - 1/T1).

KKT role
--------
These physical parameters enter the PDE constraint block of the KKT system
and control the conditioning of the adjoint operator A^T.
"""

from __future__ import annotations
import math
import numpy as np


# ---------------------------------------------------------------------------
# Physical constants (SI-like, rescaled to unit domain)
# ---------------------------------------------------------------------------

# Diffusion coefficient (m^2 / s, rescaled)
NU_DEFAULT = 1.0e-1

# T1 / T2 relaxation times (s)
T1_DEFAULT = 1.0
T2_DEFAULT = 0.3

# Equilibrium magnetisation
M0_DEFAULT = 1.0

# Control (RF pulse amplitude) bounds (mu_T)
U_LOWER_DEFAULT = -2.0
U_UPPER_DEFAULT = 6.0

# Energy budget for integral constraint  integral u dx <= E_max
EMAX_DEFAULT = 3.0

# Tikhonov regularisation weight
ALPHA_DEFAULT = 1.0e-2

# Background-error covariance scale (from 4D-Var)
B_COV_SCALE_DEFAULT = 5.0e-2

# State constraint bound
Y_MAX_DEFAULT = 4.0


class PhysicalParameters:
    """Immutable container for the physical parameters of the PDE model.

    The parameters are deeply coupled to the KKT system: the reaction
    stiffness ``kappa_1`` enters the diagonal of the state matrix ``A``,
    the control bounds define the active-set partition, and the energy
    budget couples to the integral-constraint multiplier ``lambda``.
    """

    def __init__(
        self,
        nu: float = NU_DEFAULT,
        T1: float = T1_DEFAULT,
        T2: float = T2_DEFAULT,
        M0: float = M0_DEFAULT,
        u_lower: float = U_LOWER_DEFAULT,
        u_upper: float = U_UPPER_DEFAULT,
        E_max: float = EMAX_DEFAULT,
        alpha: float = ALPHA_DEFAULT,
        B_scale: float = B_COV_SCALE_DEFAULT,
        y_max: float = Y_MAX_DEFAULT,
    ):
        self.nu = float(nu)
        self.T1 = float(T1)
        self.T2 = float(T2)
        self.M0 = float(M0)
        self.u_lower = float(u_lower)
        self.u_upper = float(u_upper)
        self.E_max = float(E_max)
        self.alpha = float(alpha)
        self.B_scale = float(B_scale)
        self.y_max = float(y_max)

        # Derived physical quantities -----------------------------------
        # kappa_1 = 1/T1  (longitudinal relaxation rate, 1/s)
        self.kappa_1 = 1.0 / self.T1
        # kappa_2 = 1/T2 - 1/T1  (saturation rate, 1/s)
        self.kappa_2 = max(1.0 / self.T2 - 1.0 / self.T1, 0.0)
        # Peclet number  Pe = L * U_char / nu  (here L=1, U_char=M0)
        self.peclet = self.M0 / max(self.nu, 1.0e-14)
        # Damkohler number  Da = kappa_1 * L^2 / nu
        self.damkohler = self.kappa_1 / max(self.nu, 1.0e-14)

    # ------------------------------------------------------------------
    def reaction(self, y: np.ndarray) -> np.ndarray:
        """Evaluate the Bloch-type relaxation operator  R(y).

        R(y) = kappa_1 * y  +  kappa_2 * y^3 / (1 + y^2)

        The rational saturation term ensures bounded reaction even for
        large magnetisation, modelling the steady-state SSFP signal.
        """
        y = np.asarray(y, dtype=np.float64)
        lin = self.kappa_1 * y
        sat = self.kappa_2 * (y ** 3) / (1.0 + y * y)
        return lin + sat

    def reaction_derivative(self, y: np.ndarray) -> np.ndarray:
        """dR/dy needed for the Newton linearisation of the KKT system.

        dR/dy = kappa_1 + kappa_2 * (3y^2(1+y^2) - y^3*2y) / (1+y^2)^2
              = kappa_1 + kappa_2 * (3y^2 + y^4) / (1+y^2)^2
        Wait, let me redo this:
            d/dy [ y^3/(1+y^2) ]
            = [3y^2(1+y^2) - y^3 * 2y] / (1+y^2)^2
            = [3y^2 + 3y^4 - 2y^4] / (1+y^2)^2
            = [3y^2 + y^4] / (1+y^2)^2
            = y^2 * (3 + y^2) / (1+y^2)^2
        """
        y = np.asarray(y, dtype=np.float64)
        y2 = y * y
        denom = (1.0 + y2) ** 2
        sat_deriv = y2 * (3.0 + y2) / np.where(denom > 1.0e-30, denom, 1.0e-30)
        return self.kappa_1 + self.kappa_2 * sat_deriv

    # ------------------------------------------------------------------
    def bloch_steady_state(self, B1_amp: np.ndarray) -> np.ndarray:
        """Analytical steady-state M_z for a spoiled GRE sequence.

        M_z^ss = M0 * (1 - exp(-TR/T1)) / (1 - cos(flip) * exp(-TR/T1))

        Here we use B1_amp as a flip-angle surrogate (radians) and fix
        TR = T1 for simplicity.  This enters the target-state generation.
        """
        TR_over_T1 = 1.0
        E1 = math.exp(-TR_over_T1)
        cos_fa = np.cos(B1_amp)
        denom = 1.0 - cos_fa * E1
        denom = np.where(np.abs(denom) < 1.0e-14, 1.0e-14, denom)
        return self.M0 * (1.0 - E1) / denom

    # ------------------------------------------------------------------
    def gamma_function_stirling(self, x: float) -> float:
        """Stirling approximation of Gamma(x) for large-argument asymptotics.

        Gamma(x) ~ sqrt(2*pi/x) * (x/e)^x * (1 + 1/(12x) + 1/(288x^2) - ...)

        This is used for constructing the ASA-314-style statistical
        test quantities in the convergence analysis.
        """
        if x <= 0.0:
            return float("inf")
        if x < 0.5:
            # Reflection formula  Gamma(x)*Gamma(1-x) = pi/sin(pi*x)
            return math.pi / (math.sin(math.pi * x) * self.gamma_function_stirling(1.0 - x))
        # Stirling series (6 terms)
        inv_x = 1.0 / x
        stirling_coef = [
            1.0,
            1.0 / 12.0,
            1.0 / 288.0,
            -139.0 / 51840.0,
            -571.0 / 2488320.0,
            163879.0 / 209018880.0,
        ]
        corr = 1.0
        power = 1.0
        for c in stirling_coef[1:]:
            power *= inv_x
            corr += c * power
        return math.sqrt(2.0 * math.pi / x) * (x / math.e) ** x * corr


# ---------------------------------------------------------------------------
def setup_parameters(**overrides) -> PhysicalParameters:
    """Return a PhysicalParameters instance, applying any keyword overrides."""
    return PhysicalParameters(**overrides)
