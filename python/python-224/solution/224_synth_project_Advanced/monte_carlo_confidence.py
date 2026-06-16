"""
monte_carlo_confidence.py
=========================
Monte Carlo integration over confidence ellipses in signal-strength space.

Confidence region at CL = alpha:
    (mu - mu_hat)^T F (mu - mu_hat) <= chi^2_{k, alpha}
where k = n_channels, and chi^2_{k, alpha} is the critical value.

For 2 channels: chi^2_{2, 0.95} = 5.991; for 5 channels: chi^2_{5, 0.95} = 11.070.

Area of the confidence ellipse in 2D (seed 331_ellipse_monte_carlo):
    Area = pi R^2 / sqrt(det A)
where A is the Fisher matrix (2x2) and R^2 = chi^2_{2, alpha}.

For high-dimensional integration, we use:
  - Importance sampling from the Gaussian N(mu_hat, F^{-1})
  - Hit-or-miss sampling inside the chi^2 ellipse
  - Coverage probability verification

The probability inside the confidence region under the true Gaussian:
    P(chi^2_k <= chi^2_{k,alpha}) = alpha  (exact for Gaussian)
Monte Carlo verifies this coverage empirically.
"""
from __future__ import annotations
import numpy as np
from math import gamma as gamma_fn


class MonteCarloConfidence:
    """
    Monte Carlo integration over the signal-strength confidence ellipse.
    """

    def __init__(
        self,
        fisher_matrix: np.ndarray,
        mu_hat: np.ndarray,
        seed: int = 224,
    ) -> None:
        self.F = np.asarray(fisher_matrix, dtype=float)
        self.mu_hat = np.asarray(mu_hat, dtype=float)
        self.rng = np.random.default_rng(seed)
        # Covariance = F^{-1}
        try:
            self.cov = np.linalg.inv(self.F)
        except np.linalg.LinAlgError:
            self.cov = np.linalg.pinv(self.F)
        # Ensure symmetric positive-definite
        self.cov = 0.5 * (self.cov + self.cov.T)
        eig = np.linalg.eigvalsh(self.cov)
        if eig.min() < 0:
            self.cov += (-eig.min() + 1e-8) * np.eye(len(self.cov))

    # ------------------------------------------------------------------ #
    #                   Critical chi^2 values                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def chi2_critical(k: int, alpha: float = 0.95) -> float:
        """
        Critical value of chi^2 distribution with k d.o.f. at CL alpha.
        Uses the Wilson-Hilferty approximation for chi^2 quantile:
            chi^2_{k, alpha} ~ k (1 - 2/(9k) + z_alpha sqrt(2/(9k)))^3
        where z_alpha = Phi^{-1}(alpha).
        """
        if alpha <= 0.0 or alpha >= 1.0:
            raise ValueError("alpha must be in (0, 1)")
        # Inverse normal (Abramowitz & Stegun rational approximation)
        if alpha > 0.5:
            p = 1.0 - alpha
            t = np.sqrt(-2.0 * np.log(p))
            z = t - (2.515517 + 0.802853 * t + 0.010328 * t * t) / (
                1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t
            )
        else:
            p = alpha
            t = np.sqrt(-2.0 * np.log(p))
            z = -(t - (2.515517 + 0.802853 * t + 0.010328 * t * t) / (
                1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t
            ))
        # Wilson-Hilferty cube transform
        term = 1.0 - 2.0 / (9.0 * k) + z * np.sqrt(2.0 / (9.0 * k))
        return float(k * term ** 3)

    # ------------------------------------------------------------------ #
    #                Ellipse area (2D projection)                        #
    # ------------------------------------------------------------------ #
    def ellipse_area_2d(self, axes: tuple = (0, 1), alpha: float = 0.95) -> float:
        """
        Area of the (1-alpha) confidence ellipse projected onto 2 axes.
        Area = pi * R^2 / sqrt(det F_2d)
        where R^2 = chi^2_{2, alpha}.
        (seed 331_ellipse_monte_carlo)
        """
        i, j = axes
        F_2d = self.F[np.ix_([i, j], [i, j])]
        det = np.linalg.det(F_2d)
        if det <= 0:
            return 0.0
        R2 = self.chi2_critical(2, alpha)
        return float(np.pi * R2 / np.sqrt(det))

    # ------------------------------------------------------------------ #
    #               Gaussian sampling + coverage                         #
    # ------------------------------------------------------------------ #
    def sample_gaussian(self, n_samples: int) -> np.ndarray:
        """Draw n_samples from N(mu_hat, F^{-1})."""
        return self.rng.multivariate_normal(self.mu_hat, self.cov, size=n_samples)

    def _delta_chi2(self, samples: np.ndarray) -> np.ndarray:
        """Compute Delta chi^2 = (mu - mu_hat)^T F (mu - mu_hat) for each sample."""
        dmu = samples - self.mu_hat[np.newaxis, :]
        return np.einsum("ij,jk,ik->i", dmu, self.F, dmu)

    def estimate_coverage(self, cl: float = 0.95, n_samples: int = 5000) -> float:
        """
        Empirically estimate coverage probability:
            P(chi^2_k <= chi^2_{k, alpha})  under N(mu_hat, F^{-1}).
        Should equal alpha for a Gaussian.
        """
        k = len(self.mu_hat)
        crit = self.chi2_critical(k, cl)
        samples = self.sample_gaussian(n_samples)
        dq = self._delta_chi2(samples)
        return float(np.mean(dq <= crit))

    # ------------------------------------------------------------------ #
    #              Hit-or-miss integration inside ellipse                #
    # ------------------------------------------------------------------ #
    def integrate_in_ellipse(self, integrand, alpha: float = 0.95, n_samples: int = 10000) -> float:
        """
        Integrate `integrand(mu)` over the confidence ellipse
        {mu: (mu - mu_hat)^T F (mu - mu_hat) <= chi^2_{k, alpha}}
        using hit-or-miss Monte Carlo with Gaussian proposal.

        The volume of the k-d confidence ellipse:
            V_k = (pi^{k/2} / Gamma(k/2+1)) * (chi^2_{k,alpha})^{k/2} / sqrt(det F)
        """
        k = len(self.mu_hat)
        crit = self.chi2_critical(k, alpha)

        # Generate Gaussian samples and keep those inside
        samples = self.sample_gaussian(n_samples)
        dq = self._delta_chi2(samples)
        inside = dq <= crit
        if not inside.any():
            return 0.0
        values = np.array([integrand(s) for s in samples[inside]])
        mean_inside = float(np.mean(values))

        # Volume of k-d ellipse
        vol = (
            (np.pi ** (k / 2.0))
            / gamma_fn(k / 2.0 + 1.0)
            * (crit ** (k / 2.0))
            / np.sqrt(max(np.linalg.det(self.F), 1e-300))
        )
        p_accept = float(np.mean(inside))
        return mean_inside * vol / max(p_accept, 1e-10)

    # ------------------------------------------------------------------ #
    #              Likelihood diffusion smearing (seed 1258)             #
    # ------------------------------------------------------------------ #
    def diffuse_likelihood(self, mu_grid: np.ndarray, sigma_diff: float = 0.05) -> np.ndarray:
        """
        Apply Gaussian diffusion to a 1D likelihood profile:
            L_smeared(mu) = Integral K(mu - mu', sigma) L(mu') d mu'
        where K is the Gaussian kernel.
        (seed 1258_VLOGroup_PoGMDM diffusion-style smearing)
        """
        mu_grid = np.asarray(mu_grid, dtype=float)
        L_raw = np.exp(-0.5 * (mu_grid - self.mu_hat[0]) ** 2 / max(self.cov[0, 0], 1e-10))
        L_smeared = np.zeros_like(L_raw)
        for i, mu_i in enumerate(mu_grid):
            K = np.exp(-0.5 * (mu_grid - mu_i) ** 2 / max(sigma_diff ** 2, 1e-10))
            if hasattr(np, "trapezoid"):
                Z = np.trapezoid(K, mu_grid)
            else:
                Z = np.trapz(K, mu_grid)
            if Z < 1e-300:
                L_smeared[i] = 0.0
            else:
                if hasattr(np, "trapezoid"):
                    L_smeared[i] = np.trapezoid(K * L_raw, mu_grid) / Z
                else:
                    L_smeared[i] = np.trapz(K * L_raw, mu_grid) / Z
        return L_smeared
