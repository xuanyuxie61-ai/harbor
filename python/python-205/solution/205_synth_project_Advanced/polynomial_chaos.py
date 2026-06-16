"""
polynomial_chaos.py - Polynomial Chaos Expansion Core Module

This module implements the mathematical core of polynomial chaos
expansion (PCE) surrogate models for uncertainty quantification.

Mathematical Framework
----------------------
### Polynomial Chaos Expansion ###

Given a computational model Y = M(X) where X = (X_1, ..., X_d) is
a random input vector with joint PDF f_X, the PCE approximation is:

  Y ≈ Y^PCE = sum_{alpha in A} c_alpha * Psi_alpha(X)

where:
  - A is a multi-index set (total order, hyperbolic cross, etc.)
  - Psi_alpha(X) = prod_{i=1}^d phi_{alpha_i}(X_i) are multivariate
    orthogonal polynomials
  - c_alpha are the PCE coefficients

### Orthogonality ###

  E[Psi_alpha(X) * Psi_beta(X)] = delta_{alpha,beta} * <Psi_alpha, Psi_alpha>

For standard uniform X_i ~ U(-1,1), the basis polynomials are
normalized Legendre polynomials:
  integral_{-1}^{1} P_m(x) P_n(x) dx / 2 = delta_{mn} / (2n+1)

### Coefficient Computation ###

#### Projection (Non-Intrusive) ####
  c_alpha = E[Y * Psi_alpha(X)] / E[Psi_alpha^2(X)]
          ≈ (1/N) sum_{i=1}^N Y_i * Psi_alpha(X_i) / <Psi_alpha, Psi_alpha>

#### Regression (Non-Intrusive) ####
  min_c ||Y - Psi * c||_2^2 + lambda * ||c||_1  (LASSO for sparse PCE)
  or
  min_c ||Y - Psi * c||_2^2 + lambda * ||c||_2^2  (Ridge)

### Statistical Moments from PCE ###
  E[Y] = c_0
  Var[Y] = sum_{alpha != 0} c_alpha^2 * <Psi_alpha, Psi_alpha>

### Sobol Sensitivity Indices ###
  First-order: S_i = Var_{X_i}(E[Y|X_i]) / Var[Y]
               = sum_{alpha: alpha_i>0, alpha_j=0 for j!=i} c_alpha^2 * <Psi_alpha, Psi_alpha> / Var[Y]
  Total-order: S_{Ti} = sum_{alpha: alpha_i>0} c_alpha^2 * <Psi_alpha, Psi_alpha> / Var[Y]

References
----------
[1] Xiu & Karniadakis, "The Wiener-Askey Polynomial Chaos", SIAM Rev., 2002.
[2] Blatman & Sudret, "Adaptive sparse polynomial chaos expansion", IJNME, 2011.
[3] Sudret, "Global sensitivity analysis using polynomial chaos expansions",
    Reliability Engineering & System Safety, 2008.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, List, Dict, Optional, Any
import math
from numerical_utils import (total_order_multi_index, hyperbolic_cross_multi_index,
                              legendre_ek_compute, beta_function)


# ---------------------------------------------------------------------------
# Orthogonal Polynomial Basis
# ---------------------------------------------------------------------------

class LegendreBasis:
    """
    Univariate Legendre polynomial basis with evaluation and normalization.

    The normalized Legendre polynomials satisfy:
      integral_{-1}^{1} phi_m(x) * phi_n(x) * (1/2) dx = delta_{mn}

    Recurrence relation:
      (n+1) * P_{n+1}(x) = (2n+1) * x * P_n(x) - n * P_{n-1}(x)
    with P_0(x) = 1, P_1(x) = x.

    Normalized: phi_n(x) = sqrt(2n+1) * P_n(x)
    """

    def __init__(self, max_degree: int = 10):
        self.max_degree = max_degree

    def evaluate(self, x: NDArray, n: int) -> NDArray:
        """
        Evaluate n-th degree normalized Legendre polynomial at points x.

        Parameters
        ----------
        x : ndarray(m,)
            Evaluation points in [-1, 1].
        n : int
            Polynomial degree.

        Returns
        -------
        ndarray(m,)
            Values phi_n(x_i).
        """
        if n == 0:
            return np.ones_like(x)
        if n == 1:
            return np.sqrt(3.0) * x

        # Three-term recurrence
        P_prev = np.ones_like(x)           # P_0
        P_curr = x.copy()                   # P_1
        for k in range(1, n):
            P_next = ((2 * k + 1) * x * P_curr - k * P_prev) / (k + 1)
            P_prev = P_curr
            P_curr = P_next

        return math.sqrt(2 * n + 1) * P_curr

    def evaluate_all(self, x: NDArray, max_n: int) -> NDArray:
        """
        Evaluate all Legendre polynomials up to degree max_n.

        Returns
        -------
        ndarray(max_n + 1, m)
            phi[k, :] = phi_k(x).
        """
        m = len(x)
        result = np.zeros((max_n + 1, m))
        result[0] = np.ones(m)

        if max_n >= 1:
            result[1] = np.sqrt(3.0) * x

        P_prev = np.ones(m)
        P_curr = x.copy()

        for n in range(1, max_n):
            P_next = ((2 * n + 1) * x * P_curr - n * P_prev) / (n + 1)
            result[n + 1] = math.sqrt(2 * (n + 1) + 1) * P_next
            P_prev = P_curr
            P_curr = P_next

        return result

    def norm_squared(self, n: int) -> float:
        """
        Squared L2 norm of the normalized Legendre polynomial.

        For our normalization: ||phi_n||^2 = 1 (already normalized).
        For standard Legendre: ||P_n||^2 = 2/(2n+1).
        """
        return 1.0  # Our basis is orthonormal w.r.t. uniform measure on [-1,1]


class MultivariateBasis:
    """
    Multivariate polynomial basis via tensor product of univariate bases.

    For multi-index alpha = (alpha_1, ..., alpha_d):
      Psi_alpha(x) = prod_{i=1}^d phi_{alpha_i}(x_i)

    The squared norm is:
      ||Psi_alpha||^2 = prod_{i=1}^d ||phi_{alpha_i}||^2
    """

    def __init__(self, d: int, max_degree: int, basis_type: str = 'legendre'):
        self.d = d
        self.max_degree = max_degree
        self.basis_type = basis_type
        if basis_type == 'legendre':
            self.univariate = LegendreBasis(max_degree)
        else:
            raise ValueError(f"Unknown basis type: {basis_type}")

    def evaluate(self, x: NDArray, alpha: NDArray) -> NDArray:
        """
        Evaluate multivariate basis function Psi_alpha at points x.

        Parameters
        ----------
        x : ndarray(n, d)
            Sample points.
        alpha : ndarray(d,)
            Multi-index.

        Returns
        -------
        ndarray(n,)
            Psi_alpha(x_i) for each sample.
        """
        n = x.shape[0]
        result = np.ones(n)
        for j in range(self.d):
            result *= self.univariate.evaluate(x[:, j], alpha[j])
        return result

    def evaluate_all(self, x: NDArray,
                     multi_index: NDArray) -> NDArray:
        """
        Evaluate all basis functions for given multi-index set.

        Parameters
        ----------
        x : ndarray(n, d)
            Sample points.
        multi_index : ndarray(m, d)
            Multi-index set.

        Returns
        -------
        Psi : ndarray(n, m)
            Evaluation matrix: Psi[i, j] = Psi_{alpha_j}(x_i).
        """
        n = x.shape[0]
        m = multi_index.shape[0]
        Psi = np.ones((n, m))

        # Evaluate univariate polynomials for each dimension
        for j in range(self.d):
            phi_all = self.univariate.evaluate_all(
                x[:, j], self.max_degree)
            for k in range(m):
                Psi[:, k] *= phi_all[multi_index[k, j], :]

        return Psi

    def norm_squared(self, alpha: NDArray) -> float:
        """Squared norm of Psi_alpha (product of univariate norms)."""
        result = 1.0
        for j in range(self.d):
            result *= self.univariate.norm_squared(alpha[j])
        return result


# ---------------------------------------------------------------------------
# Polynomial Chaos Surrogate
# ---------------------------------------------------------------------------

class PolynomialChaosSurrogate:
    """
    Polynomial Chaos Expansion surrogate model.

    Attributes
    ----------
    d : int
        Input dimension.
    max_degree : int
        Maximum polynomial degree.
    index_set_type : str
        'total_order' or 'hyperbolic_cross'.
    multi_index : ndarray(m, d)
        Multi-index set.
    coefficients : ndarray(m,)
        PCE coefficients.
    basis : MultivariateBasis
        Polynomial basis object.
    is_fitted : bool
        Whether coefficients have been computed.
    """

    def __init__(self, d: int, max_degree: int = 3,
                 index_set_type: str = 'total_order',
                 hc_q: float = 0.5):
        self.d = d
        self.max_degree = max_degree
        self.index_set_type = index_set_type
        self.hc_q = hc_q

        # Build multi-index set
        if index_set_type == 'total_order':
            self.multi_index = total_order_multi_index(d, max_degree)
        elif index_set_type == 'hyperbolic_cross':
            self.multi_index = hyperbolic_cross_multi_index(d, max_degree, hc_q)
        else:
            raise ValueError(f"Unknown index set type: {index_set_type}")

        self.n_terms = self.multi_index.shape[0]
        self.basis = MultivariateBasis(d, max_degree)
        self.coefficients = np.zeros(self.n_terms)
        self.is_fitted = False

    def fit_projection(self, X: NDArray, Y: NDArray,
                       weights: Optional[NDArray] = None) -> 'PolynomialChaosSurrogate':
        """
        Fit PCE coefficients via projection (quadrature-based).

        c_alpha = sum_{i=1}^N w_i * Y_i * Psi_alpha(X_i) / ||Psi_alpha||^2

        This is the non-intrusive projection method, optimal when
        the quadrature rule is exact for the product Psi_alpha * M(X).

        Parameters
        ----------
        X : ndarray(n, d)
            Input sample points.
        Y : ndarray(n,)
            Model outputs at sample points.
        weights : ndarray(n,), optional
            Quadrature weights. If None, uses uniform weights 1/n.

        Returns
        -------
        self
        """
        n = X.shape[0]
        if weights is None:
            weights = np.ones(n) / n

        Psi = self.basis.evaluate_all(X, self.multi_index)

        for j in range(self.n_terms):
            norm_sq = self.basis.norm_squared(self.multi_index[j])
            if norm_sq < 1e-30:
                norm_sq = 1.0
            self.coefficients[j] = np.sum(weights * Y * Psi[:, j]) / norm_sq

        self.is_fitted = True
        return self

    def fit_regression(self, X: NDArray, Y: NDArray,
                       regularization: str = 'ridge',
                       lambda_reg: float = 1e-6) -> 'PolynomialChaosSurrogate':
        """
        Fit PCE coefficients via least-squares regression.

        min_c ||Y - Psi * c||_2^2 + lambda * penalty(c)

        For Ridge (L2): penalty = ||c||_2^2
          c = (Psi^T Psi + lambda I)^{-1} Psi^T Y

        For LASSO (L1): uses iterative soft-thresholding (ISTA).
          c^{k+1} = S_{lambda/L}(c^k - (1/L) Psi^T (Psi c^k - Y))
          where S_t(x) = sign(x) * max(|x| - t, 0)

        Parameters
        ----------
        X : ndarray(n, d)
            Input points.
        Y : ndarray(n,)
            Outputs.
        regularization : str
            'ridge' or 'lasso'.
        lambda_reg : float
            Regularization parameter.

        Returns
        -------
        self
        """
        Psi = self.basis.evaluate_all(X, self.multi_index)
        n, m = Psi.shape

        if regularization == 'ridge':
            # Normal equations: (Psi^T Psi + lambda I) c = Psi^T Y
            A = Psi.T @ Psi + lambda_reg * np.eye(m)
            b = Psi.T @ Y
            try:
                self.coefficients = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse
                self.coefficients = np.linalg.lstsq(A, b, rcond=None)[0]

        elif regularization == 'lasso':
            # ISTA (Iterative Soft-Thresholding Algorithm)
            L = np.linalg.norm(Psi.T @ Psi, 2)  # Lipschitz constant
            if L < 1e-30:
                L = 1.0
            step = 1.0 / L
            c = np.zeros(m)

            for iteration in range(1000):
                grad = Psi.T @ (Psi @ c - Y) / n
                c_tilde = c - step * grad
                # Soft thresholding
                self.coefficients = np.sign(c_tilde) * np.maximum(
                    np.abs(c_tilde) - lambda_reg * step, 0.0)
                if np.linalg.norm(self.coefficients - c) < 1e-10:
                    break
                c = self.coefficients.copy()

        else:
            # No regularization: least squares
            self.coefficients, _, _, _ = np.linalg.lstsq(Psi, Y, rcond=None)

        self.is_fitted = True
        return self

    def predict(self, X: NDArray) -> NDArray:
        """
        Predict model output at new input points.

        Y_pred = sum_j c_j * Psi_{alpha_j}(X)

        Parameters
        ----------
        X : ndarray(n, d)
            Input points.

        Returns
        -------
        ndarray(n,)
            Predicted outputs.
        """
        if not self.is_fitted:
            raise RuntimeError("Surrogate not fitted. Call fit_*() first.")

        Psi = self.basis.evaluate_all(X, self.multi_index)
        return Psi @ self.coefficients

    def statistical_moments(self) -> Dict[str, float]:
        """
        Compute statistical moments from PCE coefficients.

        Mean:     E[Y] = c_0
        Variance: Var[Y] = sum_{j>0} c_j^2 * ||Psi_{alpha_j}||^2
        The PCE provides exact moments (up to approximation error)
        without additional model evaluations.

        Returns
        -------
        dict with 'mean', 'variance', 'std', 'skewness', 'kurtosis'
        """
        if not self.is_fitted:
            raise RuntimeError("Surrogate not fitted.")

        mean = self.coefficients[0] if self.n_terms > 0 else 0.0

        variance = 0.0
        third_moment = 0.0
        fourth_moment = 0.0

        for j in range(1, self.n_terms):
            norm_sq = self.basis.norm_squared(self.multi_index[j])
            c_j = self.coefficients[j]
            variance += c_j ** 2 * norm_sq
            third_moment += c_j ** 3 * norm_sq ** 1.5
            fourth_moment += c_j ** 4 * norm_sq ** 2

        std = math.sqrt(max(variance, 0.0))
        skewness = third_moment / (variance ** 1.5 + 1e-30) if variance > 1e-30 else 0.0
        kurtosis = fourth_moment / (variance ** 2 + 1e-30) if variance > 1e-30 else 0.0

        return {
            'mean': float(mean),
            'variance': float(variance),
            'std': float(std),
            'skewness': float(skewness),
            'kurtosis': float(kurtosis)
        }

    def sobol_indices(self) -> Dict[str, NDArray]:
        """
        Compute Sobol sensitivity indices from PCE coefficients.

        First-order index S_i measures the main effect of X_i:
          S_i = V_i / V
        where V_i = sum_{alpha: alpha_i>0, alpha_j=0 for j!=i} c_alpha^2 * ||Psi_alpha||^2

        Total-order index S_{Ti} measures total effect (including interactions):
          S_{Ti} = V_{Ti} / V
        where V_{Ti} = sum_{alpha: alpha_i>0} c_alpha^2 * ||Psi_alpha||^2

        The sum S_1 + S_2 + ... + S_d <= 1 (equality if no interactions).
        The sum S_{T1} + S_{T2} + ... + S_{Td} >= 1.

        Returns
        -------
        dict with:
            'first_order': ndarray(d,) - First-order Sobol indices
            'total_order': ndarray(d,) - Total-order Sobol indices
            'interaction_order': ndarray(d,) - S_Ti - S_i (interaction strength)
        """
        if not self.is_fitted:
            raise RuntimeError("Surrogate not fitted.")

        moments = self.statistical_moments()
        total_var = moments['variance']

        if total_var < 1e-30:
            return {
                'first_order': np.zeros(self.d),
                'total_order': np.zeros(self.d),
                'interaction_order': np.zeros(self.d)
            }

        S_first = np.zeros(self.d)
        S_total = np.zeros(self.d)

        for j in range(1, self.n_terms):
            alpha = self.multi_index[j]
            norm_sq = self.basis.norm_squared(alpha)
            c_sq = self.coefficients[j] ** 2 * norm_sq

            total_degree = int(np.sum(alpha))

            # First-order: only alpha_i > 0 for exactly one i
            non_zero_dims = np.where(alpha > 0)[0]
            if len(non_zero_dims) == 1:
                S_first[non_zero_dims[0]] += c_sq

            # Total-order: alpha_i > 0 for dimension i
            for dim in non_zero_dims:
                S_total[dim] += c_sq

        S_first /= total_var
        S_total /= total_var
        S_interaction = S_total - S_first

        # Clamp to [0, 1]
        S_first = np.clip(S_first, 0.0, 1.0)
        S_total = np.clip(S_total, 0.0, 1.0)

        return {
            'first_order': S_first,
            'total_order': S_total,
            'interaction_order': S_interaction
        }

    def leave_one_out_error(self, X: NDArray, Y: NDArray) -> Dict[str, float]:
        """
        Compute leave-one-out cross-validation error.

        For PCE, LOO error can be computed analytically without
        refitting:
          LOO_i = (Y_i - Y_hat_i) / (1 - h_ii)
        where h_ii is the i-th diagonal of the hat matrix:
          H = Psi (Psi^T Psi)^{-1} Psi^T
          Q2 = 1 - (1/N) sum_i (LOO_i / (Y_i - mean(Y)))^2

        Returns
        -------
        dict with 'loo_q2', 'loo_rmse', 'loo_max_error'
        """
        if not self.is_fitted:
            raise RuntimeError("Surrogate not fitted.")

        Psi = self.basis.evaluate_all(X, self.multi_index)
        n, m = Psi.shape

        # Predictions
        Y_pred = Psi @ self.coefficients

        # Hat matrix diagonal
        try:
            PtP_inv = np.linalg.inv(Psi.T @ Psi + 1e-12 * np.eye(m))
            H = Psi @ PtP_inv @ Psi.T
            h_diag = np.diag(H)
        except np.linalg.LinAlgError:
            h_diag = np.ones(n) * m / n

        # LOO errors
        residuals = Y - Y_pred
        denom = 1.0 - h_diag
        denom = np.where(np.abs(denom) < 1e-14, 1e-14, denom)
        loo_errors = residuals / denom

        # Q2 metric
        Y_mean = np.mean(Y)
        ss_res = np.sum(loo_errors ** 2)
        ss_tot = np.sum((Y - Y_mean) ** 2)
        Q2 = 1.0 - ss_res / (ss_tot + 1e-30)

        return {
            'loo_q2': float(Q2),
            'loo_rmse': float(np.sqrt(np.mean(loo_errors ** 2))),
            'loo_max_error': float(np.max(np.abs(loo_errors)))
        }

    def get_info(self) -> Dict[str, Any]:
        """Get surrogate model information."""
        info = {
            'dimension': self.d,
            'max_degree': self.max_degree,
            'index_set_type': self.index_set_type,
            'n_terms': self.n_terms,
            'is_fitted': self.is_fitted,
            'multi_index': self.multi_index
        }
        if self.is_fitted:
            info['coefficients'] = self.coefficients
            info['moments'] = self.statistical_moments()
            info['sobol'] = self.sobol_indices()
        return info
