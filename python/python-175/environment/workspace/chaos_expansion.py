"""
chaos_expansion.py
==================
Generalized Polynomial Chaos (gPC) expansion core: projection, Galerkin,
and coefficient computation for stochastic PDE solutions.

Fused from seed projects:
- 990_r8poly   : 1-D orthogonal polynomial evaluations
- 893_polynomial : multi-dimensional multi-index algebra
- 1130_sphere_triangle_quad : spherical triangle sampling for quasi-Monte Carlo baseline

Mathematical foundation
-----------------------
Let xi = (xi_1, ..., xi_d) be independent random variables with known PDFs.
The gPC expansion of a random quantity u(xi) is
    u(xi) \approx \sum_{\alpha \in \Lambda} u_\alpha \Psi_\alpha(xi)
where \Lambda is a finite multi-index set and \Psi_\alpha are multivariate
orthonormal polynomials satisfying
    E[\Psi_\alpha(xi) \Psi_\beta(xi)] = \delta_{\alpha\beta}.

For uniform xi_k ~ U(-1,1), \Psi_\alpha are products of normalized Legendre
polynomials.  For Gaussian xi_k ~ N(0,1), they are products of normalized
physicists' Hermite polynomials (the original Wiener-Hermite chaos).

Projection (pseudo-spectral) coefficients:
    u_\alpha = E[u(xi) \Psi_\alpha(xi)] \approx \sum_q w^{(q)} u(xi^{(q)}) \Psi_\alpha(xi^{(q)})

Galerkin formulation for the stochastic PDE:
    \sum_{\alpha} u_\alpha(x) E[ a(x,xi) \Psi_\alpha(xi) \Psi_\beta(xi) ] = ...
This yields a coupled deterministic system whose size is N_x * |\Lambda|.
"""

import numpy as np
from orthogonal_polynomials import legendre_eval, normalized_legendre_eval, hermite_eval
from multidim_polynomial import (enumerate_multi_indices_total_degree, sparse_grid_index_set,
                                  multivariate_orthogonal_basis)
from quadrature_rules import gauss_legendre_tensor, smolyak_sparse_grid


def normalized_legendre_1d(m, x):
    """Normalized Legendre polynomial on [-1,1] with weight 1/2."""
    if m == 0:
        return np.ones_like(x)
    return legendre_eval(m, x) * np.sqrt((2.0 * m + 1.0) / 2.0)


def normalized_hermite_1d(m, x):
    """Normalized physicists' Hermite polynomial on (-inf,inf) with weight exp(-x^2/2)/sqrt(2pi)."""
    if m == 0:
        return np.ones_like(x)
    return hermite_eval(m, x) / np.sqrt((2.0 ** m) * np.math.factorial(m) * np.sqrt(np.pi))


def gpc_basis_eval(alpha, xi, distribution="uniform"):
    """
    Evaluate the multivariate gPC basis Psi_alpha(xi).

    Parameters
    ----------
    alpha : array-like of ints, shape (d,)
    xi : ndarray, shape (n_samples, d) or (d,)
    distribution : str
        "uniform"  -> normalized Legendre on [-1,1]
        "gaussian" -> normalized Hermite on (-inf,inf)

    Returns
    -------
    values : ndarray, shape (n_samples,)
    """
    if distribution == "uniform":
        poly_func = normalized_legendre_1d
    elif distribution == "gaussian":
        poly_func = normalized_hermite_1d
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    return multivariate_orthogonal_basis(alpha, xi, poly_func)


def gpc_projection_coefficients(xi_samples, weights, func_values, index_set,
                                distribution="uniform"):
    """
    Compute gPC coefficients by numerical projection (pseudo-spectral approach).

    Parameters
    ----------
    xi_samples : ndarray, shape (n_q, d)
    weights : ndarray, shape (n_q,)
        Quadrature weights (must sum to 1 for probability measure).
    func_values : ndarray, shape (n_q,) or (n_q, n_x)
        Function values u(xi^{(q)}) at quadrature points.
    index_set : ndarray, shape (N, d)
        Multi-index set.
    distribution : str

    Returns
    -------
    coeffs : ndarray, shape (N,) or (N, n_x)
    """
    n_q = xi_samples.shape[0]
    N = index_set.shape[0]
    func_values = np.asarray(func_values, dtype=float)
    single_output = (func_values.ndim == 1)
    if single_output:
        func_values = func_values.reshape(-1, 1)
    n_x = func_values.shape[1]
    coeffs = np.zeros((N, n_x), dtype=float)

    # TODO: Hole 3a - Compute gPC projection coefficients
    # For a normalized orthonormal basis Psi_alpha, the coefficient u_alpha is the
    # weighted inner product  E[u * Psi_alpha] = sum_q w_q * u(xi_q) * Psi_alpha(xi_q).
    # Implement the projection for each multi-index alpha and each spatial degree of freedom.
    for i, alpha in enumerate(index_set):
        psi = gpc_basis_eval(alpha, xi_samples, distribution)
        # Weighted inner product
        for j in range(n_x):
            coeffs[i, j] = None  # FIX: compute projection coefficient

    if single_output:
        return coeffs[:, 0]
    return coeffs


def gpc_mean_variance(coeffs, index_set):
    """
    Compute mean and variance from gPC coefficients.
    Mean = u_0 (alpha = all zeros)
    Variance = sum_{alpha != 0} u_\alpha^2
    """
    N = index_set.shape[0]
    zero_alpha = np.zeros(index_set.shape[1], dtype=int)
    mean = 0.0
    for i, alpha in enumerate(index_set):
        if np.array_equal(alpha, zero_alpha):
            mean = coeffs[i]
            break
    # TODO: Hole 3b - Compute variance from gPC coefficients
    # For an orthonormal gPC basis, Var[u] = sum_{alpha != 0} u_alpha^2.
    # Accumulate the squared coefficients over all non-zero multi-indices.
    var = None  # FIX: compute variance using Parseval identity for orthonormal basis
    return mean, var


def gpc_sobol_sensitivity(coeffs, index_set, dim):
    """
    Compute first-order Sobol sensitivity indices from gPC coefficients.

    For dimension k, the first-order index is
        S_k = Var[E[u | xi_k]] / Var[u]
             = (sum_{alpha: alpha_k>0, alpha_j=0 for j!=k} c_\alpha^2) / Var[u]
    """
    N, d = index_set.shape
    mean, total_var = gpc_mean_variance(coeffs, index_set)
    if total_var < 1e-30:
        return np.zeros(d)
    S = np.zeros(d)
    for k in range(d):
        partial_var = 0.0
        for i, alpha in enumerate(index_set):
            if alpha[k] > 0 and np.sum(alpha) == alpha[k]:
                partial_var += coeffs[i] ** 2
        S[k] = partial_var / total_var
    return S


def gpc_total_order_sobol(coeffs, index_set, dim):
    """
    Compute total-order Sobol sensitivity indices.
    S_k^T = (sum_{alpha: alpha_k > 0} c_\alpha^2) / Var[u]
    """
    N, d = index_set.shape
    mean, total_var = gpc_mean_variance(coeffs, index_set)
    if total_var < 1e-30:
        return np.zeros(d)
    S = np.zeros(d)
    for k in range(d):
        partial_var = 0.0
        for i, alpha in enumerate(index_set):
            if alpha[k] > 0:
                partial_var += coeffs[i] ** 2
        S[k] = partial_var / total_var
    return S


def gpc_reconstruct(xi_new, coeffs, index_set, distribution="uniform"):
    """
    Reconstruct u(xi) at new random variable samples from gPC coefficients.
    """
    xi_new = np.asarray(xi_new, dtype=float)
    if xi_new.ndim == 1:
        xi_new = xi_new.reshape(1, -1)
    N = index_set.shape[0]
    single = (coeffs.ndim == 1)
    if single:
        coeffs = coeffs.reshape(-1, 1)
    n_out = coeffs.shape[1]
    result = np.zeros((xi_new.shape[0], n_out), dtype=float)
    for i, alpha in enumerate(index_set):
        psi = gpc_basis_eval(alpha, xi_new, distribution)
        for j in range(n_out):
            result[:, j] += coeffs[i, j] * psi
    if single:
        return result[:, 0]
    return result


def test_chaos_expansion():
    """Test gPC on a simple analytical function: f(xi) = xi_1^2 + xi_2^2."""
    d = 2
    max_deg = 2
    idx_set = enumerate_multi_indices_total_degree(d, max_deg)
    # Use tensor Gauss-Legendre quadrature
    xi_q, w_q = gauss_legendre_tensor(d, 5)
    # Normalize weights for probability measure on [-1,1]^d (measure = 1/2^d)
    w_q = w_q / (2.0 ** d)
    f_vals = xi_q[:, 0] ** 2 + xi_q[:, 1] ** 2
    coeffs = gpc_projection_coefficients(xi_q, w_q, f_vals, idx_set, "uniform")
    mean, var = gpc_mean_variance(coeffs, idx_set)
    # Analytical: E[xi_k^2] = 1/3, E[f] = 2/3
    assert np.isclose(mean, 2.0 / 3.0, atol=1e-12)
    # Var[xi_k^2] = E[xi_k^4] - (E[xi_k^2])^2 = 1/5 - 1/9 = 4/45
    assert np.isclose(var, 8.0 / 45.0, atol=1e-12)
    # Reconstruction test
    xi_test = np.array([[0.5, -0.3], [0.0, 0.0]])
    rec = gpc_reconstruct(xi_test, coeffs, idx_set, "uniform")
    exact = xi_test[:, 0] ** 2 + xi_test[:, 1] ** 2
    assert np.allclose(rec, exact, atol=1e-10)
    print("chaos_expansion: all self-tests passed")


if __name__ == "__main__":
    test_chaos_expansion()
