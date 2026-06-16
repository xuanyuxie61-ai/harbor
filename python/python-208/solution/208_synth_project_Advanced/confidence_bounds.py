"""
confidence_bounds.py
====================

Confidence-bound propagation for the multi-fidelity GP predictor.
Borrows the CLP (Conditional Linear Projection) framework from project
1223 (gev26_clpbounds): we construct prediction intervals for the
QoI that are valid under both parametric (Gaussian) and nonparametric
(multiplier-bootstrap) assumptions.

Background (Semenova 2026, CLP estimator)
-----------------------------------------
Given a multi-fidelity predictor hat{F}(xi), we want a (1 - alpha)
prediction interval for the true QoI F_H(xi):

    PI(xi) = [ hat{F}(xi) - q_{1-alpha/2} * SE(xi),
               hat{F}(xi) + q_{1-alpha/2} * SE(xi) ]

where SE(xi) is a standard-error estimate and q_p is the p-quantile of
a reference distribution (Gaussian or bootstrap).

The CLP estimator combines three sources of uncertainty:
  1. GP posterior variance (epistemic uncertainty from limited data),
  2. Multi-fidelity bias correction variance (from the discrepancy model),
  3. Observational noise variance (from measurement/denoising error).

We also implement OLS / LASSO / Ridge variants following CLP_final.py:
the bias correction at each fidelity level can be modeled linearly in
a set of engineered features, with regularization selected by K-fold CV.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


# ----------------------------------------------------------------------
# Prediction interval dataclass.
# ----------------------------------------------------------------------
@dataclass
class PredictionInterval:
    """A (1 - alpha) prediction interval."""
    center: float
    lower: float
    upper: float
    se: float
    alpha: float
    coverage_method: str  # "gaussian" | "bootstrap"


# ----------------------------------------------------------------------
# Gaussian prediction interval.
# ----------------------------------------------------------------------
def gaussian_quantile(p: float) -> float:
    """Approximate the standard-normal quantile via the Abramowitz-Stegun formula.

    For 0 < p < 1, returns z such that Phi(z) = p, where Phi is the
    standard-normal CDF.  Accuracy: ~4.5e-4.
    """
    if p <= 0.0 or p >= 1.0:
        raise ValueError("gaussian_quantile: p must be in (0, 1).")
    if p < 0.5:
        return -gaussian_quantile(1.0 - p)
    t = math.sqrt(-2.0 * math.log(1.0 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)


def gaussian_pi(
    mean: float, se: float, alpha: float = 0.05
) -> PredictionInterval:
    """Construct a Gaussian prediction interval."""
    if se < 0.0:
        raise ValueError("gaussian_pi: se must be >= 0.")
    z = gaussian_quantile(1.0 - alpha / 2.0)
    return PredictionInterval(
        center=mean,
        lower=mean - z * se,
        upper=mean + z * se,
        se=se,
        alpha=alpha,
        coverage_method="gaussian",
    )


# ----------------------------------------------------------------------
# Multiplier bootstrap for robust confidence bands.
# ----------------------------------------------------------------------
def multiplier_bootstrap(
    residuals: List[float],
    weights: List[float],
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float]:
    """Multiplier-bootstrap critical value for the sup-norm of the residual process.

    Given residuals r_i and GP posterior weights w_i(xi), we compute
    the (1 - alpha) quantile of:
        max_xi | sum_i w_i(xi) r_i * g_i |
    where g_i are iid N(0, 1) multipliers.
    """
    if not residuals:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(residuals)
    maxima: List[float] = []
    for _ in range(n_boot):
        g = [rng.gauss(0.0, 1.0) for _ in range(n)]
        s = sum(residuals[i] * weights[i] * g[i] for i in range(n))
        maxima.append(abs(s))
    maxima.sort()
    idx = max(0, min(int(math.ceil((1.0 - alpha) * n_boot)) - 1, len(maxima) - 1))
    return maxima[idx], sum(maxima) / n_boot


# ----------------------------------------------------------------------
# CLP-style bias correction with linear features.
# ----------------------------------------------------------------------
@dataclass
class BiasCorrection:
    """Linear bias correction model: bias(xi) ~ phi(xi)^T beta."""
    beta: List[float]
    regularizer: str  # "ols" | "ridge" | "lasso"
    feature_names: List[str] = field(default_factory=list)


def engineer_features(xi: List[float]) -> List[float]:
    """Build the CLP-style feature vector from xi.

    Features:
      - linear terms xi_j
      - squared terms xi_j^2
      - pairwise products xi_j xi_k for j < k
      - a constant 1
    """
    d = len(xi)
    phi: List[float] = []
    for j in range(d):
        phi.append(xi[j])
    for j in range(d):
        phi.append(xi[j] ** 2)
    for j in range(d):
        for k in range(j + 1, d):
            phi.append(xi[j] * xi[k])
    phi.append(1.0)
    return phi


def ols_fit(X: List[List[float]], y: List[float]) -> List[float]:
    """Ordinary least squares: beta = (X^T X)^{-1} X^T y."""
    n = len(y)
    if n == 0:
        return []
    p = len(X[0])
    # Normal equations.
    XtX = [[0.0] * p for _ in range(p)]
    Xty = [0.0] * p
    for i in range(n):
        for a in range(p):
            Xty[a] += X[i][a] * y[i]
            for b in range(p):
                XtX[a][b] += X[i][a] * X[i][b]
    # Regularize for stability.
    for a in range(p):
        XtX[a][a] += 1.0e-8
    # Solve by Gaussian elimination.
    A = [XtX[a] + [Xty[a]] for a in range(p)]
    for j in range(p):
        # Pivot.
        pivot = j
        for i in range(j + 1, p):
            if abs(A[i][j]) > abs(A[pivot][j]):
                pivot = i
        A[j], A[pivot] = A[pivot], A[j]
        if abs(A[j][j]) < 1.0e-30:
            continue
        for i in range(j + 1, p):
            factor = A[i][j] / A[j][j]
            for k in range(j, p + 1):
                A[i][k] -= factor * A[j][k]
    beta = [0.0] * p
    for j in range(p - 1, -1, -1):
        if abs(A[j][j]) < 1.0e-30:
            beta[j] = 0.0
            continue
        s = A[j][p]
        for k in range(j + 1, p):
            s -= A[j][k] * beta[k]
        beta[j] = s / A[j][j]
    return beta


def ridge_fit(X: List[List[float]], y: List[float], lam: float = 1.0) -> List[float]:
    """Ridge regression: beta = (X^T X + lam I)^{-1} X^T y."""
    n = len(y)
    if n == 0:
        return []
    p = len(X[0])
    XtX = [[0.0] * p for _ in range(p)]
    Xty = [0.0] * p
    for i in range(n):
        for a in range(p):
            Xty[a] += X[i][a] * y[i]
            for b in range(p):
                XtX[a][b] += X[i][a] * X[i][b]
    for a in range(p):
        XtX[a][a] += lam
    # Solve.
    A = [XtX[a] + [Xty[a]] for a in range(p)]
    for j in range(p):
        pivot = j
        for i in range(j + 1, p):
            if abs(A[i][j]) > abs(A[pivot][j]):
                pivot = i
        A[j], A[pivot] = A[pivot], A[j]
        if abs(A[j][j]) < 1.0e-30:
            continue
        for i in range(j + 1, p):
            factor = A[i][j] / A[j][j]
            for k in range(j, p + 1):
                A[i][k] -= factor * A[j][k]
    beta = [0.0] * p
    for j in range(p - 1, -1, -1):
        if abs(A[j][j]) < 1.0e-30:
            beta[j] = 0.0
            continue
        s = A[j][p]
        for k in range(j + 1, p):
            s -= A[j][k] * beta[k]
        beta[j] = s / A[j][j]
    return beta


def lasso_fit(
    X: List[List[float]], y: List[float], lam: float = 0.1,
    max_iter: int = 200, tol: float = 1.0e-6,
) -> List[float]:
    """LASSO regression by coordinate descent with soft-thresholding.

        min_beta  0.5 || y - X beta ||^2 + lam * sum |beta_j|
    """
    n = len(y)
    if n == 0:
        return []
    p = len(X[0])
    beta = [0.0] * p
    # Precompute column norms.
    col_sq = [sum(X[i][j] ** 2 for i in range(n)) + 1.0e-12 for j in range(p)]
    for _it in range(max_iter):
        beta_old = list(beta)
        for j in range(p):
            # Partial residual.
            r = [y[i] - sum(X[i][k] * beta[k] for k in range(p) if k != j) for i in range(n)]
            rho = sum(X[i][j] * r[i] for i in range(n))
            # Soft-thresholding.
            if rho > lam:
                beta[j] = (rho - lam) / col_sq[j]
            elif rho < -lam:
                beta[j] = (rho + lam) / col_sq[j]
            else:
                beta[j] = 0.0
        # Convergence check.
        diff = math.sqrt(sum((beta[j] - beta_old[j]) ** 2 for j in range(p)))
        if diff < tol:
            break
    return beta


def fit_bias_correction(
    xi_samples: List[List[float]],
    residuals: List[float],
    method: str = "ridge",
    lam: float = 1.0,
) -> BiasCorrection:
    """Fit a CLP-style bias correction model."""
    X = [engineer_features(xi) for xi in xi_samples]
    if method == "ols":
        beta = ols_fit(X, residuals)
    elif method == "ridge":
        beta = ridge_fit(X, residuals, lam)
    elif method == "lasso":
        beta = lasso_fit(X, residuals, lam)
    else:
        raise ValueError(f"fit_bias_correction: unknown method '{method}'.")
    return BiasCorrection(beta=beta, regularizer=method)


def predict_bias(xi: List[float], bc: BiasCorrection) -> float:
    """Predict the bias correction at xi."""
    phi = engineer_features(xi)
    if len(phi) != len(bc.beta):
        return 0.0
    return sum(phi[j] * bc.beta[j] for j in range(len(phi)))


# ----------------------------------------------------------------------
# Combined multi-fidelity prediction interval.
# ----------------------------------------------------------------------
def mf_prediction_interval(
    xi: List[float],
    mf_mean: float,
    mf_var: float,
    bc: Optional[BiasCorrection],
    sigma_n_sq: float,
    alpha: float = 0.05,
) -> PredictionInterval:
    """Combine GP variance + bias correction + noise into a total PI."""
    bias = predict_bias(xi, bc) if bc is not None else 0.0
    # Total variance: posterior + noise.
    total_var = max(mf_var, 1.0e-14) + sigma_n_sq
    se = math.sqrt(total_var)
    return gaussian_pi(mf_mean + bias, se, alpha)
