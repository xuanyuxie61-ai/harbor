
import numpy as np
from orthogonal_polynomials import legendre_eval, normalized_legendre_eval, hermite_eval
from multidim_polynomial import (enumerate_multi_indices_total_degree, sparse_grid_index_set,
                                  multivariate_orthogonal_basis)
from quadrature_rules import gauss_legendre_tensor, smolyak_sparse_grid


def normalized_legendre_1d(m, x):
    if m == 0:
        return np.ones_like(x)
    return legendre_eval(m, x) * np.sqrt((2.0 * m + 1.0) / 2.0)


def normalized_hermite_1d(m, x):
    if m == 0:
        return np.ones_like(x)
    return hermite_eval(m, x) / np.sqrt((2.0 ** m) * np.math.factorial(m) * np.sqrt(np.pi))


def gpc_basis_eval(alpha, xi, distribution="uniform"):
    if distribution == "uniform":
        poly_func = normalized_legendre_1d
    elif distribution == "gaussian":
        poly_func = normalized_hermite_1d
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    return multivariate_orthogonal_basis(alpha, xi, poly_func)


def gpc_projection_coefficients(xi_samples, weights, func_values, index_set,
                                distribution="uniform"):
    n_q = xi_samples.shape[0]
    N = index_set.shape[0]
    func_values = np.asarray(func_values, dtype=float)
    single_output = (func_values.ndim == 1)
    if single_output:
        func_values = func_values.reshape(-1, 1)
    n_x = func_values.shape[1]
    coeffs = np.zeros((N, n_x), dtype=float)





    for i, alpha in enumerate(index_set):
        psi = gpc_basis_eval(alpha, xi_samples, distribution)

        for j in range(n_x):
            coeffs[i, j] = None

    if single_output:
        return coeffs[:, 0]
    return coeffs


def gpc_mean_variance(coeffs, index_set):
    N = index_set.shape[0]
    zero_alpha = np.zeros(index_set.shape[1], dtype=int)
    mean = 0.0
    for i, alpha in enumerate(index_set):
        if np.array_equal(alpha, zero_alpha):
            mean = coeffs[i]
            break



    var = None
    return mean, var


def gpc_sobol_sensitivity(coeffs, index_set, dim):
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
    d = 2
    max_deg = 2
    idx_set = enumerate_multi_indices_total_degree(d, max_deg)

    xi_q, w_q = gauss_legendre_tensor(d, 5)

    w_q = w_q / (2.0 ** d)
    f_vals = xi_q[:, 0] ** 2 + xi_q[:, 1] ** 2
    coeffs = gpc_projection_coefficients(xi_q, w_q, f_vals, idx_set, "uniform")
    mean, var = gpc_mean_variance(coeffs, idx_set)

    assert np.isclose(mean, 2.0 / 3.0, atol=1e-12)

    assert np.isclose(var, 8.0 / 45.0, atol=1e-12)

    xi_test = np.array([[0.5, -0.3], [0.0, 0.0]])
    rec = gpc_reconstruct(xi_test, coeffs, idx_set, "uniform")
    exact = xi_test[:, 0] ** 2 + xi_test[:, 1] ** 2
    assert np.allclose(rec, exact, atol=1e-10)
    print("chaos_expansion: all self-tests passed")


if __name__ == "__main__":
    test_chaos_expansion()
