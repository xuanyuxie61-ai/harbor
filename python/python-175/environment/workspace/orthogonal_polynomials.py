
import numpy as np
from numpy.polynomial import polynomial as P


def legendre_coeffs(n):
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return np.array([1.0])
    if n == 1:
        return np.array([0.0, 1.0])
    p_nm2 = np.array([1.0])
    p_nm1 = np.array([0.0, 1.0])
    for k in range(1, n):

        xpk = np.concatenate(([0.0], p_nm1))
        p_n = ((2.0 * k + 1.0) * xpk - k * np.concatenate((p_nm2, [0.0, 0.0]))[:len(xpk)]) / (k + 1.0)
        p_nm2, p_nm1 = p_nm1, p_n
    return p_nm1


def chebyshev_coeffs(n):
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return np.array([1.0])
    if n == 1:
        return np.array([0.0, 1.0])
    t_nm2 = np.array([1.0])
    t_nm1 = np.array([0.0, 1.0])
    for k in range(1, n):
        xtm1 = np.concatenate(([0.0], t_nm1))
        t_n = 2.0 * xtm1 - np.concatenate((t_nm2, [0.0, 0.0]))[:len(xtm1)]
        t_nm2, t_nm1 = t_nm1, t_n
    return t_nm1


def hermite_coeffs(n):
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return np.array([1.0])
    if n == 1:
        return np.array([0.0, 2.0])
    h_nm2 = np.array([1.0])
    h_nm1 = np.array([0.0, 2.0])
    for k in range(1, n):
        xhm1 = np.concatenate(([0.0], h_nm1))
        h_n = 2.0 * xhm1 - 2.0 * k * np.concatenate((h_nm2, [0.0, 0.0]))[:len(xhm1)]
        h_nm2, h_nm1 = h_nm1, h_n
    return h_nm1


def poly_horner(coeffs, x):
    coeffs = np.asarray(coeffs, dtype=float)
    x = np.asarray(x, dtype=float)
    if coeffs.size == 0:
        return np.zeros_like(x)
    result = np.full_like(x, coeffs[-1])
    for c in coeffs[-2::-1]:
        result = result * x + c
    return result


def legendre_eval(n, x):
    x = np.asarray(x, dtype=float)
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x.copy()
    p_nm2 = np.ones_like(x)
    p_nm1 = x.copy()
    for k in range(1, n):
        p_n = ((2.0 * k + 1.0) * x * p_nm1 - k * p_nm2) / (k + 1.0)
        p_nm2, p_nm1 = p_nm1, p_n
    return p_nm1


def chebyshev_eval(n, x):
    x = np.asarray(x, dtype=float)
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x.copy()
    t_nm2 = np.ones_like(x)
    t_nm1 = x.copy()
    for _ in range(1, n):
        t_n = 2.0 * x * t_nm1 - t_nm2
        t_nm2, t_nm1 = t_nm1, t_n
    return t_nm1


def hermite_eval(n, x):
    x = np.asarray(x, dtype=float)
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2.0 * x
    h_nm2 = np.ones_like(x)
    h_nm1 = 2.0 * x
    for k in range(1, n):
        h_n = 2.0 * x * h_nm1 - 2.0 * k * h_nm2
        h_nm2, h_nm1 = h_nm1, h_n
    return h_nm1


def jacobi_matrix_legendre(n):
    if n < 1:
        raise ValueError("n must be positive")
    J = np.zeros((n, n))
    for i in range(1, n):
        beta = i / np.sqrt(4.0 * i * i - 1.0)
        J[i - 1, i] = beta
        J[i, i - 1] = beta
    return J


def jacobi_matrix_chebyshev(n):
    if n < 1:
        raise ValueError("n must be positive")
    J = np.zeros((n, n))
    for i in range(1, n):
        J[i - 1, i] = 0.5
        J[i, i - 1] = 0.5
    return J


def gauss_legendre_nodes_weights(n):
    J = jacobi_matrix_legendre(n)
    eigvals, eigvecs = np.linalg.eigh(J)

    idx = np.argsort(eigvals)
    x = eigvals[idx]
    v1 = eigvecs[0, idx]
    w = 2.0 * v1 ** 2
    return x, w


def gauss_chebyshev_nodes_weights(n):
    J = jacobi_matrix_chebyshev(n)
    eigvals, eigvecs = np.linalg.eigh(J)
    idx = np.argsort(eigvals)
    x = eigvals[idx]
    v1 = eigvecs[0, idx]
    w = np.pi * v1 ** 2
    return x, w


def normalized_legendre_eval(n, x):
    return legendre_eval(n, x) / np.sqrt(2.0 / (2.0 * n + 1.0))


def polynomial_roots_via_companion(coeffs):
    coeffs = np.asarray(coeffs, dtype=complex)
    n = len(coeffs) - 1
    if n < 1:
        return np.array([], dtype=complex)

    if abs(coeffs[-1]) < 1e-15:
        raise ValueError("Leading coefficient is numerically zero")
    coeffs = coeffs / coeffs[-1]
    C = np.zeros((n, n), dtype=complex)
    if n > 1:
        C[1:, :-1] = np.eye(n - 1)
    C[-1, :] = -coeffs[:-1]
    roots = np.linalg.eigvals(C)
    return roots


def test_orthogonal_polynomials():
    x, w = gauss_legendre_nodes_weights(20)
    for m in range(6):
        for n in range(6):
            val = np.sum(legendre_eval(m, x) * legendre_eval(n, x) * w)
            expect = 2.0 / (2.0 * m + 1.0) if m == n else 0.0
            assert np.isclose(val, expect, atol=1e-12), f"Legendre ortho fail ({m},{n})"
    print("orthogonal_polynomials: all self-tests passed")


if __name__ == "__main__":
    test_orthogonal_polynomials()
