"""
signal_strength.py
==================
Signal-strength parameterization and profile-likelihood fit for
multiple Higgs decay channels.

The signal strength modifier for channel i is:
    mu_i = (sigma x BR)_i / (sigma x BR)_i^{SM}
In the SM, mu_i = 1 for all channels by definition.

Production x decay decomposition:
    mu_i^{production, decay} = (sigma_p / sigma_p^SM) * (BR_d / BR_d^SM)

For the combined fit, the profile likelihood ratio is:
    lambda(mu) = L(mu, theta_hat_hat(mu)) / L(mu_hat, theta_hat)
where theta are nuisance parameters (systematics).

The test statistic (for upper limits / discovery):
    q_mu = -2 ln lambda(mu)    for mu_hat <= mu
         = 0                    otherwise

Under the Asimov dataset (data = expected background + signal),
mu_hat = 1 exactly in the SM.

This module implements:
  - Log-likelihood construction per channel
  - Asimov dataset generation
  - Profile likelihood scan over mu
  - Best-fit mu_hat via Newton minimization of -2 ln L

The signal migration between production modes is modeled as an
advection-diffusion process in mu-space (seed 352_fd1d_advection_diffusion_steady):
    -D d^2 p(mu)/d mu^2 + v d p(mu)/d mu = S(mu)
where D is diffusion (migration) strength, v is advection (bias),
S(mu) is the source from production x decay.

The steady solution of this PDE is obtained by finite-difference
discretization (seed 362_fd1d_heat_steady).
"""
from __future__ import annotations
import numpy as np
from sm_constants import SMConstants
from phase_space import PhaseSpaceIntegrator
from finite_difference import HighOrderFD


def _trapz(y, x):
    """Backward-compatible trapezoidal integration."""
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


class SignalStrengthFitter:
    """
    Signal-strength fit for a set of Higgs decay channels.
    """

    def __init__(self, channels: list, sm: SMConstants | None = None) -> None:
        self.channels = list(channels)
        self.sm = sm if sm is not None else SMConstants()
        self.ps = PhaseSpaceIntegrator(self.sm)
        self.fd = HighOrderFD()

        # Expected SM signal yields s_i and background b_i per channel
        # (units: events at 300 fb^-1, approximate LHC Run 3 values)
        self.s_SM = {
            "ggH_bb": 240.0,
            "ggH_tautau": 15.0,
            "VBF_hww": 85.0,
            "VH_zz": 12.0,
            "ttH_gammagamma": 1.4,
        }
        self.b = {
            "ggH_bb": 1800.0,
            "ggH_tautau": 90.0,
            "VBF_hww": 420.0,
            "VH_zz": 35.0,
            "ttH_gammagamma": 8.5,
        }
        # Fractional systematic uncertainty on background (nuisance)
        self.syst_frac = {
            "ggH_bb": 0.10,
            "ggH_tautau": 0.08,
            "VBF_hww": 0.05,
            "VH_zz": 0.06,
            "ttH_gammagamma": 0.12,
        }
        # Observed counts (set equal to Asimov expectation by default)
        self.n_obs = {ch: self.s_SM[ch] + self.b[ch] for ch in self.channels}

        # Nuisance parameters theta_j (per-channel background normalizations)
        self.n_nuisance = len(self.channels)
        self.theta_hat = np.zeros(self.n_nuisance)  # initialized at nominal

    # ------------------------------------------------------------------ #
    #                         Model expectations                         #
    # ------------------------------------------------------------------ #
    def expected_signal(self, mu_vec: np.ndarray, theta_vec: np.ndarray | None = None) -> np.ndarray:
        """
        Expected signal in each channel given signal strengths mu.
        Signal does NOT depend on nuisance theta (only background does).
        s_i(mu_i) = mu_i * s_i^SM
        """
        mu_vec = np.asarray(mu_vec, dtype=float)
        s = np.array([self.s_SM[ch] for ch in self.channels])
        return mu_vec * s

    def expected_background(self, theta_vec: np.ndarray) -> np.ndarray:
        """b_i(theta_i) = b_i^nominal * exp(theta_i * sigma_syst_i)."""
        theta_vec = np.asarray(theta_vec, dtype=float)
        b = np.array([self.b[ch] for ch in self.channels])
        syst = np.array([self.syst_frac[ch] for ch in self.channels])
        return b * np.exp(theta_vec * syst)

    # ------------------------------------------------------------------ #
    #                           Likelihood                               #
    # ------------------------------------------------------------------ #
    def log_likelihood(self, mu_vec: np.ndarray, theta_vec: np.ndarray) -> float:
        """
        Poisson log-likelihood plus Gaussian constraint on nuisances:
            ln L = Sum_i [n_i ln(s_i + b_i) - (s_i + b_i) - ln(n_i!)]
                   - 1/2 Sum_j theta_j^2
        The constant ln(n_i!) is dropped (does not affect profiling).
        """
        mu_vec = np.asarray(mu_vec, dtype=float)
        theta_vec = np.asarray(theta_vec, dtype=float)
        s = self.expected_signal(mu_vec)
        b = self.expected_background(theta_vec)
        lam = s + b
        lam = np.maximum(lam, 1e-10)
        n = np.array([self.n_obs[ch] for ch in self.channels])
        ll = float(np.sum(n * np.log(lam) - lam))
        ll -= 0.5 * float(np.sum(theta_vec ** 2))
        return ll

    def neg_log_likelihood(self, mu_vec: np.ndarray, theta_vec: np.ndarray) -> float:
        return -self.log_likelihood(mu_vec, theta_vec)

    # ------------------------------------------------------------------ #
    #                       Asimov dataset                               #
    # ------------------------------------------------------------------ #
    def generate_asimov_data(self) -> None:
        """
        Set observed counts to Asimov expectation: n_i = s_i^SM + b_i.
        In this case the best-fit mu_hat = (1, 1, ..., 1).
        """
        for ch in self.channels:
            self.n_obs[ch] = self.s_SM[ch] + self.b[ch]

    def generate_pseudo_data(self, mu_true: np.ndarray, seed: int = 224) -> None:
        """
        Generate pseudo-data from Poisson(mu_true * s_SM + b).
        """
        rng = np.random.default_rng(seed)
        for i, ch in enumerate(self.channels):
            mu_i = float(mu_true[i]) if np.ndim(mu_true) > 0 else float(mu_true)
            lam = mu_i * self.s_SM[ch] + self.b[ch]
            self.n_obs[ch] = int(rng.poisson(max(lam, 1e-3)))

    # ------------------------------------------------------------------ #
    #                    Profile likelihood fit                          #
    # ------------------------------------------------------------------ #
    def profile_over_theta(self, mu_vec: np.ndarray) -> float:
        """
        For fixed mu, maximize L over theta (profile likelihood).
        Uses simple gradient ascent on theta.
        """
        theta = self.theta_hat.copy()
        lr = 0.1
        for _ in range(200):
            # Numerical gradient of log L w.r.t. theta
            grad = np.zeros_like(theta)
            eps = 1e-4
            for j in range(len(theta)):
                tp = theta.copy()
                tm = theta.copy()
                tp[j] += eps
                tm[j] -= eps
                grad[j] = (self.log_likelihood(mu_vec, tp) - self.log_likelihood(mu_vec, tm)) / (2 * eps)
            theta = theta + lr * grad
            if np.linalg.norm(grad) < 1e-8:
                break
        self.theta_hat = theta
        return self.log_likelihood(mu_vec, theta)

    def profile_likelihood(self, mu_vec: np.ndarray) -> float:
        """Profile log-likelihood at given mu (returns ln L_prof(mu))."""
        return self.profile_over_theta(mu_vec)

    def profile_likelihood_fit(self) -> "FitResult":
        """
        Find best-fit mu_hat by minimizing -2 ln L_prof(mu).
        Uses Newton's method with FD Hessian.
        """
        mu = np.ones(len(self.channels))
        for _ in range(30):
            # Evaluate profile likelihood on a neighborhood for FD
            def nll(mu_):
                return -self.profile_over_theta(mu_)
            grad = self.fd.gradient(nll, mu, h=1e-3, order=4)
            hess = self.fd.hessian(nll, mu, h=5e-3, order=4)
            # Regularize Hessian for stability
            hess = hess + 1e-6 * np.eye(len(mu))
            try:
                step = np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:
                step = 0.1 * grad
            mu_new = mu - step
            # Clip to physical range (mu >= 0)
            mu_new = np.maximum(mu_new, 1e-3)
            if np.linalg.norm(mu_new - mu) < 1e-6:
                mu = mu_new
                break
            mu = mu_new
        nll_min = -self.profile_over_theta(mu)
        return FitResult(mu_hat=mu, nll_min=nll_min, theta_hat=self.theta_hat.copy())

    # ------------------------------------------------------------------ #
    #               Scan for test statistic q(mu)                        #
    # ------------------------------------------------------------------ #
    def scan_test_statistic(self, mu_scan: np.ndarray, mu_single: np.ndarray) -> np.ndarray:
        """
        Compute q(mu) = -2[ln L_prof(mu) - ln L_prof(mu_hat)]
        along a 1D scan of mu_single (one channel, others fixed at best fit).
        """
        res = self.profile_likelihood_fit()
        q_vals = []
        for m in mu_scan:
            mu_vec = res.mu_hat.copy()
            mu_vec[0] = m
            lnL = self.profile_over_theta(mu_vec)
            q = -2.0 * (lnL - (-res.nll_min))
            q_vals.append(max(q, 0.0))
        return np.array(q_vals)


class FitResult:
    """Container for the best-fit result."""

    def __init__(self, mu_hat: np.ndarray, nll_min: float, theta_hat: np.ndarray) -> None:
        self.mu_hat = mu_hat
        self.nll_min = nll_min
        self.theta_hat = theta_hat

    def __repr__(self) -> str:
        return f"FitResult(mu_hat={self.mu_hat}, nll_min={self.nll_min:.4f})"


# ---------------------------------------------------------------------- #
#       Signal migration via advection-diffusion (steady-state)          #
# ---------------------------------------------------------------------- #
class SignalMigration:
    """
    Steady-state advection-diffusion of signal strength in mu-space:
        -D d^2 p/d mu^2 + v d p/d mu = S(mu)
    with Dirichlet BC p(0) = p(1) = 0 (zero signal at boundaries).

    Discretized by 2nd-order FD (seed 362_fd1d_heat_steady):
        -D (p_{i-1} - 2 p_i + p_{i+1}) / h^2 + v (p_{i+1} - p_{i-1}) / (2h) = S_i

    Produces a tridiagonal system A p = S solved by Thomas algorithm.
    """

    def __init__(
        self,
        n_grid: int = 100,
        mu_min: float = 0.0,
        mu_max: float = 2.0,
        D: float = 0.01,
        v_adv: float = 0.05,
    ) -> None:
        self.n = n_grid
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.D = D
        self.v_adv = v_adv
        self.h = (mu_max - mu_min) / (n_grid - 1)
        self.mu_grid = np.linspace(mu_min, mu_max, n_grid)

    def _source(self, mu_center: float = 1.0, sigma: float = 0.1) -> np.ndarray:
        """Gaussian source S(mu) = exp(-(mu - mu_center)^2 / (2 sigma^2))."""
        return np.exp(-((self.mu_grid - mu_center) ** 2) / (2.0 * sigma ** 2))

    def solve_steady(self) -> np.ndarray:
        """Solve the steady advection-diffusion equation via tridiagonal solver."""
        n = self.n
        h = self.h
        D = self.D
        v = self.v_adv
        # Coefficients for interior points
        a = -D / h ** 2 - v / (2.0 * h)      # sub-diagonal
        b = 2.0 * D / h ** 2                 # main diagonal
        c = -D / h ** 2 + v / (2.0 * h)      # super-diagonal
        S = self._source()
        rhs = S.copy()
        rhs[0] = 0.0  # Dirichlet BC
        rhs[-1] = 0.0

        # Thomas algorithm
        c_p = np.zeros(n)
        d_p = np.zeros(n)
        c_p[0] = 0.0
        d_p[0] = rhs[0]
        for i in range(1, n):
            denom = b - a * c_p[i - 1] if i > 1 else b
            if abs(denom) < 1e-12:
                denom = 1e-12
            c_p[i] = c / denom if i < n - 1 else 0.0
            d_p[i] = (rhs[i] - a * d_p[i - 1]) / denom

        p = np.zeros(n)
        p[-1] = d_p[-1]
        for i in range(n - 2, -1, -1):
            p[i] = d_p[i] - c_p[i] * p[i + 1]
        return p

    def migration_fraction(self, mu_window=(0.9, 1.1)) -> float:
        """Fraction of signal migrating into [mu_lo, mu_hi]."""
        p = self.solve_steady()
        mask = (self.mu_grid >= mu_window[0]) & (self.mu_grid <= mu_window[1])
        num = _trapz(p[mask], self.mu_grid[mask])
        den = _trapz(p, self.mu_grid)
        if den <= 0:
            return 0.0
        return float(num / den)
