"""
surrogate_fitting.py - Advanced Surrogate Fitting Strategies

This module implements advanced fitting strategies for polynomial chaos
surrogates, including:
  - Iterative smoothing for noisy training data (from polygon_average)
  - Cross-validated model selection
  - Error estimation and validation metrics
  - Multi-response surrogate fitting

Mathematical Framework
----------------------
### Ridge Regression with Generalized Cross-Validation ###

  c_ridge = argmin_c { ||Y - Psi*c||^2 + lambda * ||c||^2 }

  GCV(lambda) = (1/n) * ||Y - Psi*c_lambda||^2 / (1 - tr(H_lambda)/n)^2
  where H_lambda = Psi (Psi^T Psi + lambda I)^{-1} Psi^T

### Iterative Smoothing for Noisy Data ###

  v_i^{k+1} = (1 - alpha) * v_i^k + alpha * (v_{i-1}^k + v_{i+1}^k) / 2

  This is equivalent to applying a low-pass filter with transfer function:
    H(omega) = 1 - alpha * (1 - cos(omega))
  High frequencies (omega near pi) are attenuated most strongly.

### Multi-Response Fitting ###

  For q outputs, fit q separate PCEs or use a joint formulation:
    min_{C in R^{m x q}} ||Y - Psi*C||_F^2 + lambda * ||C||_F^2
  which decouples into q independent ridge regressions.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Dict, List, Optional, Tuple, Any
from numerical_utils import iterative_smoothing_1d, convergence_ratio
from polynomial_chaos import PolynomialChaosSurrogate


# ---------------------------------------------------------------------------
# Model Selection via Cross-Validation
# ---------------------------------------------------------------------------

def generalized_cross_validation(Psi: NDArray, Y: NDArray,
                                 lambda_values: NDArray) -> Tuple[float, float]:
    """
    Select regularization parameter via Generalized Cross-Validation.

    GCV(lambda) = (1/n) * ||r||^2 / (1 - tr(H)/n)^2
    where r = Y - Psi * c_lambda and H = Psi(Psi^T Psi + lambda I)^{-1} Psi^T

    The effective degrees of freedom: df(lambda) = tr(H_lambda)
    As lambda -> 0: df -> rank(Psi) (interpolation)
    As lambda -> inf: df -> 0 (mean model)

    Parameters
    ----------
    Psi : ndarray(n, m)
        Design matrix.
    Y : ndarray(n,)
        Observations.
    lambda_values : ndarray(k,)
        Candidate regularization parameters.

    Returns
    -------
    best_lambda : float
        Optimal regularization parameter.
    best_gcv : float
        GCV score at optimal lambda.
    """
    n, m = Psi.shape
    best_lambda = lambda_values[0]
    best_gcv = float('inf')

    for lam in lambda_values:
        # Solve regularized system
        A = Psi.T @ Psi + lam * np.eye(m)
        try:
            c = np.linalg.solve(A, Psi.T @ Y)
        except np.linalg.LinAlgError:
            c = np.linalg.lstsq(A, Psi.T @ Y, rcond=None)[0]

        residuals = Y - Psi @ c
        ss_res = np.sum(residuals ** 2)

        # Trace of hat matrix
        try:
            A_inv = np.linalg.inv(A)
            H_trace = np.trace(Psi @ A_inv @ Psi.T)
        except np.linalg.LinAlgError:
            H_trace = m  # Approximation

        denom = (1.0 - H_trace / n) ** 2
        if denom < 1e-14:
            denom = 1e-14

        gcv = ss_res / (n * denom)

        if gcv < best_gcv:
            best_gcv = gcv
            best_lambda = lam

    return best_lambda, best_gcv


# ---------------------------------------------------------------------------
# Multi-Response Surrogate
# ---------------------------------------------------------------------------

class MultiResponseSurrogate:
    """
    Multi-response polynomial chaos surrogate.

    Fits separate PCEs for each output quantity of interest (QoI),
    with shared input sampling and optional joint regularization.

    Attributes
    ----------
    surrogates : dict
        Maps QoI names to PolynomialChaosSurrogate objects.
    qoi_names : list of str
        Names of output quantities.
    """

    def __init__(self, d: int, max_degree: int = 3,
                 index_set_type: str = 'total_order'):
        self.d = d
        self.max_degree = max_degree
        self.index_set_type = index_set_type
        self.surrogates: Dict[str, PolynomialChaosSurrogate] = {}
        self.qoi_names: List[str] = []
        self._fitted = False

    def add_qoi(self, name: str):
        """Register a new QoI."""
        if name not in self.surrogates:
            self.surrogates[name] = PolynomialChaosSurrogate(
                self.d, self.max_degree, self.index_set_type)
            self.qoi_names.append(name)

    def fit_all(self, X: NDArray, Y_dict: Dict[str, NDArray],
                method: str = 'ridge',
                lambda_reg: float = 1e-6) -> Dict[str, Dict[str, float]]:
        """
        Fit surrogates for all QoIs.

        Parameters
        ----------
        X : ndarray(n, d)
            Input sample points.
        Y_dict : dict
            Maps QoI names to output arrays.
        method : str
            Fitting method: 'ridge', 'lasso', 'projection'.
        lambda_reg : float
            Regularization parameter.

        Returns
        -------
        errors : dict
            Maps QoI names to error metrics.
        """
        errors = {}

        for name in self.qoi_names:
            if name not in Y_dict:
                continue
            Y = Y_dict[name]
            surrogate = self.surrogates[name]

            if method == 'projection':
                surrogate.fit_projection(X, Y)
            elif method == 'ridge':
                surrogate.fit_regression(X, Y, 'ridge', lambda_reg)
            elif method == 'lasso':
                surrogate.fit_regression(X, Y, 'lasso', lambda_reg)
            else:
                surrogate.fit_regression(X, Y, 'ridge', lambda_reg)

            # Compute validation metrics
            Y_pred = surrogate.predict(X)
            ss_res = float(np.sum((Y - Y_pred) ** 2))
            ss_tot = float(np.sum((Y - np.mean(Y)) ** 2))
            rmse = float(np.sqrt(np.mean((Y - Y_pred) ** 2)))
            # Numerical guard for near-constant outputs
            if ss_tot < 1e-20:
                r2 = 1.0 if ss_res < 1e-20 else 0.0
            else:
                r2 = 1.0 - ss_res / ss_tot

            try:
                loo = surrogate.leave_one_out_error(X, Y)
            except Exception:
                loo = {'loo_q2': 0.0, 'loo_rmse': rmse, 'loo_max_error': rmse}

            errors[name] = {
                'rmse': rmse,
                'r2': float(r2),
                'loo_q2': loo['loo_q2'],
                'loo_rmse': loo['loo_rmse'],
                'n_terms': surrogate.n_terms
            }

        self._fitted = True
        return errors

    def predict_all(self, X: NDArray) -> Dict[str, NDArray]:
        """Predict all QoIs at new input points."""
        result = {}
        for name in self.qoi_names:
            if self.surrogates[name].is_fitted:
                result[name] = self.surrogates[name].predict(X)
        return result

    def statistical_moments_all(self) -> Dict[str, Dict[str, float]]:
        """Compute statistical moments for all QoIs."""
        result = {}
        for name in self.qoi_names:
            if self.surrogates[name].is_fitted:
                result[name] = self.surrogates[name].statistical_moments()
        return result

    def sobol_indices_all(self) -> Dict[str, Dict[str, NDArray]]:
        """Compute Sobol indices for all QoIs."""
        result = {}
        for name in self.qoi_names:
            if self.surrogates[name].is_fitted:
                result[name] = self.surrogates[name].sobol_indices()
        return result


# ---------------------------------------------------------------------------
# Noisy Data Preprocessing
# ---------------------------------------------------------------------------

def preprocess_noisy_data(Y: NDArray, smoothing_iterations: int = 3,
                          smoothing_alpha: float = 0.3) -> NDArray:
    """
    Preprocess noisy training data with iterative smoothing.

    Applied before surrogate fitting to reduce the effect of
    numerical noise in expensive simulations. The smoothing
    preserves the overall trend while dampening high-frequency
    oscillations that would be fit by the surrogate.

    Parameters
    ----------
    Y : ndarray(n,)
        Raw (potentially noisy) output values.
    smoothing_iterations : int
        Number of smoothing passes.
    smoothing_alpha : float
        Smoothing strength (0 < alpha <= 1).

    Returns
    -------
    ndarray(n,)
        Smoothed output values.
    """
    return iterative_smoothing_1d(Y, smoothing_iterations, smoothing_alpha)


# ---------------------------------------------------------------------------
# Degree Selection
# ---------------------------------------------------------------------------

def select_optimal_degree(X: NDArray, Y: NDArray,
                          max_degree_test: int = 6,
                          d: Optional[int] = None) -> int:
    """
    Select optimal polynomial degree via cross-validation.

    Tests degrees 1, 2, ..., max_degree_test and selects the one
    minimizing the LOO Q2 score.

    Parameters
    ----------
    X : ndarray(n, d)
        Training inputs.
    Y : ndarray(n,)
        Training outputs.
    max_degree_test : int
        Maximum degree to test.
    d : int, optional
        Input dimension (inferred from X if not given).

    Returns
    -------
    int
        Optimal polynomial degree.
    """
    if d is None:
        d = X.shape[1]

    best_degree = 1
    best_q2 = -float('inf')

    for p in range(1, max_degree_test + 1):
        try:
            surrogate = PolynomialChaosSurrogate(d, p, 'total_order')
            if surrogate.n_terms > X.shape[0]:
                break  # Too many terms for available data
            surrogate.fit_regression(X, Y, 'ridge', 1e-6)
            loo = surrogate.leave_one_out_error(X, Y)
            if loo['loo_q2'] > best_q2:
                best_q2 = loo['loo_q2']
                best_degree = p
        except Exception:
            break

    return best_degree


# ---------------------------------------------------------------------------
# Validation Metrics
# ---------------------------------------------------------------------------

def compute_validation_metrics(Y_true: NDArray, Y_pred: NDArray) -> Dict[str, float]:
    """
    Compute comprehensive validation metrics.

    Metrics:
      RMSE: sqrt(mean((Y_true - Y_pred)^2))
      MAE: mean(|Y_true - Y_pred|)
      R2: 1 - SS_res/SS_tot (coefficient of determination)
      MaxAE: max(|Y_true - Y_pred|)
      RelRMSE: RMSE / std(Y_true) (relative error)
    """
    residuals = Y_true - Y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((Y_true - np.mean(Y_true)) ** 2)

    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))
    r2 = float(1.0 - ss_res / (ss_tot + 1e-30))
    max_ae = float(np.max(np.abs(residuals)))
    rel_rmse = rmse / (float(np.std(Y_true)) + 1e-30)

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'max_abs_error': max_ae,
        'relative_rmse': rel_rmse
    }
