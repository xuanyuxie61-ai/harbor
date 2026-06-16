"""
rg_flow.py
==========
Renormalization group (RG) flow of SM couplings with ETDRK4 integration.

The 1-loop SM beta functions (in MS-bar, t = ln(mu/mu_0)) are:

    16 pi^2 d g'/dt  = (41/6) g'^3
    16 pi^2 d g/dt   = -(19/6) g^3
    16 pi^2 d gs/dt  = -7 gs^3
    16 pi^2 d y_t/dt = y_t ( (9/2) y_t^2 - 8 gs^2 - (9/4) g^2 - (17/12) g'^2 )
    16 pi^2 d lambda/dt = 24 lambda^2
                          + lambda ( 12 y_t^2 - 9 g^2 - 3 g'^2 )
                          - 6 y_t^4
                          + (3/8) ( 2 g^4 + (g^2 + g'^2)^2 )

State vector y = (g', g, gs, y_t, lambda).

The system is stiff because the gauge couplings have large negative
coefficients while lambda has a positive quadratic term. Standard RK4
becomes unstable for large t-steps; exponential time differencing
treats the linear part exactly.

ETDRK4 (Cox & Matthews 2002; Kassam & Trefethen 2005):
    For y' = L y + N(y):
    E  = expm(h L)
    E2 = expm(h L / 2)
    Q, f1, f2, f3 are contour integrals of phi functions.

In this implementation we linearize around the current state at each
step:
    L = Jacobian at y_n
    N(y) = f(y) - L y

and use scipy.linalg.expm for the matrix exponential.

The stability of the EW vacuum under RG flow is determined by whether
lambda(mu) remains positive up to some high scale Lambda (typically
10^10 - 10^19 GeV). Current data suggests lambda turns negative around
10^10 GeV, implying metastability.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import expm


class RGFlowSolver:
    """
    RG flow solver for SM couplings using ETDRK4.
    """

    def __init__(self, g_prime: float = 0.357, g: float = 0.652, gs: float = 1.166,
                 y_t: float = 0.997, lam: float = 0.126) -> None:
        """Initial conditions at mu_0 = m_t ~ 173 GeV."""
        self.y0 = np.array([g_prime, g, gs, y_t, lam], dtype=float)

    # ------------------------------------------------------------------ #
    #                       Beta functions                               #
    # ------------------------------------------------------------------ #
    @staticmethod
    def beta_functions(y: np.ndarray) -> np.ndarray:
        """
        1-loop SM beta functions for (g', g, gs, y_t, lambda).
        Returns dy/dt with t = ln(mu/mu_0).
        """
        gp, g, gs, yt, lam = y
        inv16pi2 = 1.0 / (16.0 * np.pi ** 2)
        # U(1)_Y
        dgp = inv16pi2 * (41.0 / 6.0) * gp ** 3
        # SU(2)_L
        dg = inv16pi2 * (-19.0 / 6.0) * g ** 3
        # SU(3)_c
        dgs = inv16pi2 * (-7.0) * gs ** 3
        # top Yukawa
        dyt = inv16pi2 * yt * (
            4.5 * yt ** 2 - 8.0 * gs ** 2 - 2.25 * g ** 2 - (17.0 / 12.0) * gp ** 2
        )
        # Higgs quartic
        dlam = inv16pi2 * (
            24.0 * lam ** 2
            + lam * (12.0 * yt ** 2 - 9.0 * g ** 2 - 3.0 * gp ** 2)
            - 6.0 * yt ** 4
            + 0.375 * (2.0 * g ** 4 + (g ** 2 + gp ** 2) ** 2)
        )
        return np.array([dgp, dg, dgs, dyt, dlam])

    @staticmethod
    def jacobian(y: np.ndarray) -> np.ndarray:
        """
        Jacobian J_{ij} = d beta_i / d y_j (analytical).
        Used for ETDRK4 linearization.
        """
        gp, g, gs, yt, lam = y
        inv16pi2 = 1.0 / (16.0 * np.pi ** 2)
        J = np.zeros((5, 5))
        # d beta_gp / d y
        J[0, 0] = inv16pi2 * (41.0 / 6.0) * 3.0 * gp ** 2
        # d beta_g / d y
        J[1, 1] = inv16pi2 * (-19.0 / 6.0) * 3.0 * g ** 2
        # d beta_gs / d y
        J[2, 2] = inv16pi2 * (-7.0) * 3.0 * gs ** 2
        # d beta_yt / d y
        J[3, 0] = inv16pi2 * yt * (-(17.0 / 12.0) * 2.0 * gp)
        J[3, 1] = inv16pi2 * yt * (-2.25 * 2.0 * g)
        J[3, 2] = inv16pi2 * yt * (-8.0 * 2.0 * gs)
        J[3, 3] = inv16pi2 * (9.0 * yt ** 2 - 8.0 * gs ** 2 - 2.25 * g ** 2 - (17.0 / 12.0) * gp ** 2)
        # d beta_lam / d y
        J[4, 0] = inv16pi2 * (lam * (-3.0 * 2.0 * gp) + 0.375 * 2.0 * (g ** 2 + gp ** 2) * 2.0 * gp)
        J[4, 1] = inv16pi2 * (lam * (-9.0 * 2.0 * g) + 0.375 * (4.0 * 2.0 * g * 1 + 2.0 * (g ** 2 + gp ** 2) * 2.0 * g))
        J[4, 2] = 0.0
        J[4, 3] = inv16pi2 * (lam * 12.0 * 2.0 * yt - 6.0 * 4.0 * yt ** 3)
        J[4, 4] = inv16pi2 * (48.0 * lam + 12.0 * yt ** 2 - 9.0 * g ** 2 - 3.0 * gp ** 2)
        return J

    # ------------------------------------------------------------------ #
    #                      ETDRK4 time stepper                           #
    # ------------------------------------------------------------------ #
    def etdrk4_step(self, y: np.ndarray, h: float) -> np.ndarray:
        """
        Single ETDRK4 step for y' = f(y).
        Linearize: f(y) = L y + N(y) where L = Jacobian(y_n).

        Scheme (Kassam & Trefethen 2005):
            E  = expm(h L)
            E2 = expm(h L / 2)
            Q  = h * phi_1(hL)
            f1, f2, f3: phi-function coefficients

            Nv = f(y) - L y
            a  = E2 y + Q Nv
            Na = f(a) - L a
            b  = E2 y + Q Na
            Nb = f(b) - L b
            c  = E2 a + Q (2 Nb - Nv)
            Nc = f(c) - L c

            y_{n+1} = E y + h (f1 Nv + 2 f2 (Na + Nb) + f3 Nc)
        """
        L = self.jacobian(y)
        f0 = self.beta_functions(y)
        Nv = f0 - L @ y

        E = expm(h * L)
        E2 = expm(0.5 * h * L)

        # phi_1(z) = (exp(z) - 1) / z, implemented via matrix function
        # Q = h * phi_1(hL)
        I = np.eye(5)
        # For numerical stability, use the identity: phi_1(z) = (exp(z) - 1)/z
        # and h phi_1(hL) = L^{-1} (E - I) when L invertible
        try:
            L_inv = np.linalg.inv(L)
            Q = L_inv @ (E - I)
        except np.linalg.LinAlgError:
            # Fallback to Taylor: Q ~ h (I + hL/2 + h^2 L^2 / 6)
            Q = h * (I + 0.5 * h * L + (h ** 2 / 6.0) * L @ L)

        # Half-step Q for E2
        try:
            Q2 = L_inv @ (E2 - I)
        except np.linalg.LinAlgError:
            Q2 = 0.5 * h * (I + 0.25 * h * L + (h ** 2 / 24.0) * L @ L)

        # Stage a
        a = E2 @ y + Q2 @ Nv
        fa = self.beta_functions(a)
        Na = fa - L @ a

        # Stage b
        b = E2 @ y + Q2 @ Na
        fb = self.beta_functions(b)
        Nb = fb - L @ b

        # Stage c
        c = E2 @ a + Q2 @ (2.0 * Nb - Nv)
        fc = self.beta_functions(c)
        Nc = fc - L @ c

        # phi_2(z) = (phi_1(z) - 1) / z
        # phi_3(z) = (phi_2(z) - 1/2) / z
        # Coefficients (Kassam-Trefethen Table 1):
        #   f1 = phi_1 - 3 phi_2 + 4 phi_3
        #   f2 = phi_2 - 2 phi_3
        #   f3 = -phi_2 + 4 phi_3  ... but let's use the simpler formulation
        # via the identity: E y + ... directly
        try:
            phi1 = L_inv @ (E - I)
            phi2 = L_inv @ (phi1 - I)
            phi3 = L_inv @ (phi2 - 0.5 * I)
        except np.linalg.LinAlgError:
            phi1 = Q
            phi2 = h * 0.5 * I
            phi3 = h ** 2 * (1.0 / 6.0) * I

        f1 = phi1 - 3.0 * phi2 + 4.0 * phi3
        f2 = phi2 - 2.0 * phi3
        f3 = -phi2 + 4.0 * phi3

        y_new = E @ y + h * (f1 @ Nv + 2.0 * f2 @ (Na + Nb) + f3 @ Nc)
        return y_new

    # ------------------------------------------------------------------ #
    #                       Full flow integration                        #
    # ------------------------------------------------------------------ #
    def solve(self, mu_low: float = 91.1876, mu_high: float = 1e10, n_steps: int = 400) -> dict:
        """
        Integrate RG flow from mu_low to mu_high (GeV) in log-scale.

        Returns dict with:
            'mu_grid': array of scales (GeV)
            'trajectory': (n_steps, 5) array of (g', g, gs, y_t, lambda)
            'lambda_turns_negative': scale at which lambda crosses zero (or None)
            'lambda_at_high': lambda(mu_high)
        """
        t_low = np.log(mu_low)
        t_high = np.log(mu_high)
        t_grid = np.linspace(t_low, t_high, n_steps)
        h = t_grid[1] - t_grid[0]

        # Initial conditions at mu = m_t
        y = self.y0.copy()
        traj = np.zeros((n_steps, 5))
        traj[0] = y
        lambda_cross = None

        for i in range(1, n_steps):
            y = self.etdrk4_step(y, h)
            traj[i] = y
            if lambda_cross is None and y[4] < 0.0:
                # Linear interpolation for crossing scale
                if i >= 1 and traj[i - 1, 4] > 0.0:
                    frac = traj[i - 1, 4] / (traj[i - 1, 4] - y[4])
                    t_cross = t_grid[i - 1] + frac * h
                    lambda_cross = np.exp(t_cross)

        return {
            "mu_grid": np.exp(t_grid),
            "trajectory": traj,
            "lambda_turns_negative": lambda_cross,
            "lambda_at_high": float(traj[-1, 4]),
            "g_prime_at_high": float(traj[-1, 0]),
            "g_at_high": float(traj[-1, 1]),
            "gs_at_high": float(traj[-1, 2]),
            "y_t_at_high": float(traj[-1, 3]),
        }

    # ------------------------------------------------------------------ #
    #                   Vacuum stability criterion                       #
    # ------------------------------------------------------------------ #
    def vacuum_stable_up_to(self, mu_max: float = 1e10) -> bool:
        """Check if lambda(mu) > 0 for all mu in [m_t, mu_max]."""
        result = self.solve(mu_low=self.y0[3] * 246.22 / np.sqrt(2.0), mu_high=mu_max)
        return bool(np.all(result["trajectory"][:, 4] > 0.0))
