
import numpy as np


def hep_coefficients(n):
    if n < 0:
        raise ValueError("hep_coefficients: n 必须非负")
    if n == 0:
        return np.array([1.0])


    ct = np.zeros((n + 1, n + 1), dtype=float)
    ct[0, 0] = 1.0
    ct[1, 1] = 1.0

    for i in range(1, n):

        ct[i + 1, 1:i + 2] = ct[i, 0:i + 1]
        ct[i + 1, 0:i] -= i * ct[i - 1, 0:i]

    return ct[n, 0:n + 1]


def hep_value(x, degree):
    if degree < 0:
        raise ValueError("hep_value: degree 必须非负")
    x = np.asarray(x, dtype=float)
    if degree == 0:
        return np.ones_like(x)
    if degree == 1:
        return x.copy()

    v_prev2 = np.ones_like(x)
    v_prev1 = x.copy()
    v_curr = None
    for j in range(1, degree):
        v_curr = x * v_prev1 - j * v_prev2
        v_prev2 = v_prev1
        v_prev1 = v_curr
    return v_curr


def hep_values(x, max_degree):
    if max_degree < 0:
        raise ValueError("hep_values: max_degree 必须非负")
    x = np.asarray(x, dtype=float).reshape(-1)
    n = x.shape[0]
    v = np.zeros((n, max_degree + 1), dtype=float)
    v[:, 0] = 1.0
    if max_degree >= 1:
        v[:, 1] = x
    for j in range(1, max_degree):
        v[:, j + 1] = x * v[:, j] - j * v[:, j - 1]
    return v


def hermite_product_polynomial_value(m, degrees, x):
    if len(degrees) != m:
        raise ValueError("hermite_product_polynomial_value: degrees 长度必须与 m 一致")
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[1] != m:
        raise ValueError("hermite_product_polynomial_value: x 的列数必须与 m 一致")

    result = np.ones(x.shape[0], dtype=float)
    for i in range(m):
        result *= hep_value(x[:, i], degrees[i])
    return result


def polynomial_chaos_expand(coeffs, multi_indices, xi_samples):
    coeffs = np.asarray(coeffs, dtype=float)
    multi_indices = np.asarray(multi_indices, dtype=int)
    xi_samples = np.asarray(xi_samples, dtype=float)

    n_terms = coeffs.shape[0]
    n_samples = xi_samples.shape[0]
    d = multi_indices.shape[1]

    if xi_samples.shape[1] != d:
        raise ValueError("polynomial_chaos_expand: xi_samples 维度与 multi_indices 不匹配")

    result = np.zeros(n_samples, dtype=float)
    for k in range(n_terms):
        alpha = multi_indices[k]
        val = np.ones(n_samples, dtype=float)
        for j in range(d):
            val *= hep_value(xi_samples[:, j], alpha[j])
        result += coeffs[k] * val
    return result


def generate_multi_indices(d, p):
    if d <= 0 or p < 0:
        raise ValueError("generate_multi_indices: d > 0 且 p >= 0")

    indices = []
    def recurse(current, dim, remaining):
        if dim == d - 1:
            current.append(remaining)
            indices.append(current.copy())
            current.pop()
            return
        for k in range(remaining + 1):
            current.append(k)
            recurse(current, dim + 1, remaining - k)
            current.pop()

    for total in range(p + 1):
        recurse([], 0, total)
    return np.array(indices, dtype=int)


def sobol_sensitivity(coeffs, multi_indices):
    coeffs = np.asarray(coeffs, dtype=float)
    multi_indices = np.asarray(multi_indices, dtype=int)

    n_terms = coeffs.shape[0]
    d = multi_indices.shape[1]

    from scipy.special import factorial as sp_factorial

    factorial_alpha = np.prod(sp_factorial(multi_indices, exact=True), axis=1)

    total_variance = np.sum(factorial_alpha[1:] * coeffs[1:] ** 2)
    if total_variance < 1e-30:
        total_variance = 1e-30

    sobol_main = np.zeros(d, dtype=float)
    for i in range(d):
        mask = (multi_indices[:, i] > 0) & (np.sum(multi_indices[:, :i], axis=1) + np.sum(multi_indices[:, i+1:], axis=1) == 0)
        mask[0] = False
        sobol_main[i] = np.sum(factorial_alpha[mask] * coeffs[mask] ** 2) / total_variance

    return total_variance, sobol_main
