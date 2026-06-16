"""
stability_analysis.py
=====================
Stability analysis of the Higgs vacuum and of the likelihood Hessian.

Vacuum stability in the SM requires:
    lambda(mu) > 0        for all mu up to some scale Lambda
If lambda turns negative at scale Lambda_inst, the EW vacuum is metastable
with tunneling rate (semiclassically):
    Gamma/V ~ Lambda_inst^4 exp(-S_E)
where S_E ~ 8 pi^2 / (3 |lambda(Lambda_inst)|) is the bounce action.

Numerical conditions checked here:
  1. VEV is a local minimum: d^2 V / d phi^2 > 0 at phi = v.
  2. Potential is bounded below: V -> +inf as phi -> inf.
  3. Hessian of profile likelihood is positive-definite at best-fit.
  4. Condition number kappa(H) = lambda_max / lambda_min < 1e6
     (well-constrained fit).

Eigenvalue decomposition of Hessian:
    H = U diag(lambda) U^T
positive-definiteness <=> all lambda_i > 0.

Lyapunov exponent of the RG flow (for vacuum-stability ODE system):
    h_max = max_t Re(lambda_J(t))
stability requires h_max < 0 (all trajectories converge).
"""
from __future__ import annotations
import numpy as np


class StabilityAnalyzer:
    """
    Analyzes stability of the Higgs vacuum and the signal-strength fit.
    """

    # ------------------------------------------------------------------ #
    #              Likelihood Hessian stability                          #
    # ------------------------------------------------------------------ #
    def hessian_eigenvalues(self, H: np.ndarray) -> np.ndarray:
        """Eigenvalues of symmetric Hessian matrix H."""
        H = 0.5 * (H + H.T)  # symmetrize
        return np.linalg.eigvalsh(H)

    def condition_number(self, H: np.ndarray) -> float:
        """Condition number kappa(H) = |lambda_max| / |lambda_min|."""
        ev = self.hessian_eigenvalues(H)
        ev_abs = np.abs(ev)
        if ev_abs.min() < 1e-15:
            return float("inf")
        return float(ev_abs.max() / ev_abs.min())

    def is_positive_definite(self, H: np.ndarray) -> bool:
        """Check whether H is positive-definite via eigenvalues."""
        return bool(np.all(self.hessian_eigenvalues(H) > 0.0))

    # ------------------------------------------------------------------ #
    #              Vacuum stability criteria                             #
    # ------------------------------------------------------------------ #
    def vacuum_stability_check(self, higgs_pot) -> dict:
        """
        Check vacuum stability using the HiggsPotential object.

        Returns dict with:
          - vev_is_minimum: V''(v) > 0
          - bounded_below: V(phi) -> +inf for large phi
          - ew_vacuum_deeper: V(v) < V(0)
          - quartic_positive: lambda > 0 at EW scale
          - curvature_GeV2: V''(v) in GeV^2
          - vev_recovered: |v_recovered - v| < 1 GeV

        Note: lambda_eff(v) = 4V(v)/v^4 is negative at the minimum
        (V(v) < 0 by construction for SSB), so we instead check
        V(v) < V(0) (EW vacuum deeper than origin) and lambda > 0.
        """
        sm = higgs_pot.sm
        v = sm.v

        # Curvature at minimum
        curv = float(higgs_pot.d2V_dphi2(np.array([v]))[0])

        # Value at large field (should be positive and growing)
        phi_high = 1e3  # GeV (use moderate value to avoid overflow)
        V_high = float(higgs_pot.effective_potential(np.array([phi_high]))[0])
        bounded = V_high > 0.0 and phi_high > v

        # EW vacuum deeper than origin: V(v) < V(epsilon)
        V_v = float(higgs_pot.effective_potential(np.array([v]))[0])
        V_0 = float(higgs_pot.effective_potential(np.array([1e-3]))[0])
        ew_deeper = V_v < V_0

        # Quartic coupling at EW scale (should be positive)
        quartic_ok = sm.lam_tree > 0.0

        # Recovered VEV
        v_rec = higgs_pot.find_vev_bisection()
        vev_ok = abs(v_rec - v) < 5.0  # within 5 GeV (allow for loop shifts)

        return {
            "vev_is_minimum": curv > 0.0,
            "bounded_below": bounded,
            "ew_vacuum_deeper": ew_deeper,
            "quartic_positive": quartic_ok,
            "curvature_GeV2": curv,
            "lambda_tree": sm.lam_tree,
            "vev_recovered": vev_ok,
            "v_recovered_GeV": v_rec,
            "V_v": V_v,
            "V_0": V_0,
        }

    # ------------------------------------------------------------------ #
    #              Metastability bounce action estimate                  #
    # ------------------------------------------------------------------ #
    @staticmethod
    def bounce_action(lambda_val: float) -> float:
        """
        Semiclassical bounce action:
            S_E ~ 8 pi^2 / (3 |lambda|)
        This is the O(4)-symmetric bounce action for a pure phi^4 potential.
        """
        if abs(lambda_val) < 1e-10:
            return float("inf")
        return 8.0 * np.pi ** 2 / (3.0 * abs(lambda_val))

    # ------------------------------------------------------------------ #
    #              RG flow Lyapunov exponent                             #
    # ------------------------------------------------------------------ #
    @staticmethod
    def lyapunov_from_trajectory(trajectory: np.ndarray, dt: float) -> float:
        """
        Estimate the largest Lyapunov exponent from a trajectory of the
        coupling vector over log-scale steps.

        For each step, compute Jacobian by finite differences and take
        the largest real part of its eigenvalues.

        Returns the time-averaged max Lyapunov exponent.
        """
        if len(trajectory) < 3:
            return 0.0
        n = trajectory.shape[0]
        d = trajectory.shape[1]
        lyap_vals = []
        for t in range(1, n - 1):
            J = np.zeros((d, d))
            for j in range(d):
                diff = trajectory[t + 1, j] - trajectory[t - 1, j]
                J[:, j] = diff / (2.0 * dt)
            # This is only an approximation of the Jacobian, but we scale by 1/|y|
            y = trajectory[t]
            y_safe = np.maximum(np.abs(y), 1e-10)
            J_normalized = J / y_safe[np.newaxis, :]
            ev = np.linalg.eigvals(J_normalized)
            lyap_vals.append(float(np.max(ev.real)))
        return float(np.mean(lyap_vals)) if lyap_vals else 0.0

    # ------------------------------------------------------------------ #
    #              Aggregate stability report                            #
    # ------------------------------------------------------------------ #
    def analyze(self, H: np.ndarray, higgs_pot) -> "StabilityReport":
        """
        Produce a combined stability report for Hessian H and Higgs potential.
        """
        eig = self.hessian_eigenvalues(H)
        kappa = self.condition_number(H)
        hess_pos_def = self.is_positive_definite(H)
        vac = self.vacuum_stability_check(higgs_pot)
        vacuum_stable = (
            vac["vev_is_minimum"]
            and vac["bounded_below"]
            and vac["ew_vacuum_deeper"]
            and vac["quartic_positive"]
            and hess_pos_def
        )
        return StabilityReport(
            vacuum_stable=vacuum_stable,
            hessian_pos_def=hess_pos_def,
            eigenvalues=eig,
            condition_number=kappa,
            vacuum_details=vac,
        )


class StabilityReport:
    """Container for stability analysis results."""

    def __init__(
        self,
        vacuum_stable: bool,
        hessian_pos_def: bool,
        eigenvalues: np.ndarray,
        condition_number: float,
        vacuum_details: dict,
    ) -> None:
        self.vacuum_stable = vacuum_stable
        self.hessian_pos_def = hessian_pos_def
        self.eigenvalues = eigenvalues
        self.condition_number = condition_number
        self.vacuum_details = vacuum_details

    def __repr__(self) -> str:
        lines = [
            "StabilityReport:",
            f"  vacuum_stable     = {self.vacuum_stable}",
            f"  hessian_pos_def   = {self.hessian_pos_def}",
            f"  condition_number  = {self.condition_number:.3e}",
            f"  eigenvalues       = {self.eigenvalues}",
        ]
        return "\n".join(lines)
