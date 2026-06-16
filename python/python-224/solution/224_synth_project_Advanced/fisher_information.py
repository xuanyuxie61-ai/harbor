"""
fisher_information.py
=====================
Fisher information matrix for signal-strength parameters.

Definition:
    F_{ij} = - E[ d^2 ln L / (d mu_i d mu_j) ]
evaluated at the true parameter values (here taken as mu = 1 = SM).

For Gaussian approximation (Asimov data):
    F_{ij} = Sum_k (1 / sigma_k^2) (d s_k / d mu_i) (d s_k / d mu_j)
where the sum runs over channels and sigma_k^2 = s_k + b_k + (syst_k * b_k)^2.

Asymptotic uncertainties:
    sigma(mu_i) = sqrt( (F^{-1})_{ii} )

Correlation matrix:
    rho_{ij} = (F^{-1})_{ij} / sqrt((F^{-1})_{ii} (F^{-1})_{jj})

The Fisher information encodes the Cramer-Rao lower bound:
    Var(mu_hat_i) >= (F^{-1})_{ii}

This implements the exact formulas from seed 433_fisher_exact
(applied here to Higgs signal strengths instead of KPP wave speeds).

Hyperspherical parameterization (seed 563_hypersphere_angle):
    mu_i = R * omega_i    where omega lies on the unit (n-1)-sphere
and angular statistics on omega provide rotation-invariant tests
of the signal-strength hypothesis.
"""
from __future__ import annotations
import numpy as np


class FisherMatrix:
    """
    Fisher information matrix for the multi-channel signal-strength fit.
    """

    def __init__(self, fitter) -> None:
        self.fitter = fitter
        self.channels = fitter.channels

    # ------------------------------------------------------------------ #
    #                   Fisher matrix at given mu                        #
    # ------------------------------------------------------------------ #
    def compute_fisher(self, mu_vec: np.ndarray, theta_vec: np.ndarray | None = None) -> np.ndarray:
        """
        Compute Fisher matrix at signal-strength vector mu.

        F_{ij} = Sum_k (d s_k / d mu_i) (d s_k / d mu_j) / sigma_k^2
        """
        mu_vec = np.asarray(mu_vec, dtype=float)
        if theta_vec is None:
            theta_vec = np.zeros(len(self.channels))

        n = len(self.channels)
        F = np.zeros((n, n))

        s = self.fitter.expected_signal(mu_vec)
        b = self.fitter.expected_background(theta_vec)
        syst = np.array([self.fitter.syst_frac[ch] for ch in self.channels])

        # Variance per channel (Poisson + systematic in quadrature)
        sigma2 = s + b + (syst * b) ** 2
        sigma2 = np.maximum(sigma2, 1e-10)

        # Derivatives ds_k / d mu_i = delta_{ki} * s_i^SM * exp(theta_i * syst_i)
        ds_dmu = np.diag([self.fitter.s_SM[ch] * np.exp(theta_vec[i] * syst[i])
                          for i, ch in enumerate(self.channels)])

        # F = (ds/dmu)^T diag(1/sigma^2) (ds/dmu)
        F = ds_dmu.T @ np.diag(1.0 / sigma2) @ ds_dmu
        return 0.5 * (F + F.T)  # symmetrize

    def asimov_fisher(self, mu_vec: np.ndarray | None = None) -> np.ndarray:
        """Fisher matrix evaluated at Asimov (mu = 1, theta = 0)."""
        if mu_vec is None:
            mu_vec = np.ones(len(self.channels))
        return self.compute_fisher(mu_vec, np.zeros(len(self.channels)))

    # ------------------------------------------------------------------ #
    #                    Derived quantities                              #
    # ------------------------------------------------------------------ #
    def uncertainties(self, F: np.ndarray) -> np.ndarray:
        """Asymptotic 1-sigma uncertainties: sigma_i = sqrt((F^{-1})_{ii})."""
        try:
            F_inv = np.linalg.inv(F)
        except np.linalg.LinAlgError:
            F_inv = np.linalg.pinv(F)
        diag = np.maximum(np.diag(F_inv), 0.0)
        return np.sqrt(diag)

    def correlation_matrix(self, F: np.ndarray) -> np.ndarray:
        """Correlation matrix rho_{ij} from Fisher matrix."""
        try:
            F_inv = np.linalg.inv(F)
        except np.linalg.LinAlgError:
            F_inv = np.linalg.pinv(F)
        d = np.sqrt(np.maximum(np.diag(F_inv), 1e-30))
        rho = F_inv / np.outer(d, d)
        return rho

    # ------------------------------------------------------------------ #
    #            Hyperspherical parameterization                         #
    # ------------------------------------------------------------------ #
    @staticmethod
    def cartesian_to_hyperspherical(mu_vec: np.ndarray) -> tuple:
        """
        Convert signal-strength vector mu to hyperspherical coordinates.
        Returns (R, theta_1, ..., theta_{n-2}, phi) where:
            R = |mu|
            theta_k in [0, pi] (polar angles)
            phi in [0, 2 pi)   (azimuthal angle)
        (seed 563_hypersphere_angle)
        """
        mu = np.asarray(mu_vec, dtype=float)
        R = np.linalg.norm(mu)
        if R < 1e-12:
            return (0.0,) + tuple([0.0] * (len(mu) - 1))
        n = len(mu)
        angles = []
        remaining = R
        for k in range(n - 1):
            cos_angle = mu[k] / max(remaining, 1e-12)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            angles.append(float(angle))
            remaining *= np.sin(angle)
        # Last angle (azimuthal, with sign from mu[-1])
        if remaining > 1e-12:
            phi = 2.0 * np.arcsin(np.clip(mu[-1] / max(remaining, 1e-12), -1.0, 1.0))
        else:
            phi = 0.0
        angles.append(float(phi))
        return (float(R),) + tuple(angles)

    @staticmethod
    def solid_angle_density(angles: list) -> float:
        """
        Solid-angle element on S^{n-1}:
            d Omega = sin^{n-2}(theta_1) sin^{n-3}(theta_2) ... sin(theta_{n-2}) d theta_1 ... d phi
        """
        if len(angles) < 2:
            return 1.0
        d = len(angles) - 1  # dimension of sphere = n-1
        sin_prod = 1.0
        for k, theta in enumerate(angles[:-1]):
            power = d - 1 - k
            sin_prod *= np.sin(theta) ** power
        return float(sin_prod)

    # ------------------------------------------------------------------ #
    #              Fisher-based chi^2 at given mu                        #
    # ------------------------------------------------------------------ #
    def chi2_at(self, mu_vec: np.ndarray, mu_ref: np.ndarray, F: np.ndarray) -> float:
        """Delta chi^2 = (mu - mu_ref)^T F (mu - mu_ref)."""
        dmu = np.asarray(mu_vec) - np.asarray(mu_ref)
        return float(dmu @ F @ dmu)
