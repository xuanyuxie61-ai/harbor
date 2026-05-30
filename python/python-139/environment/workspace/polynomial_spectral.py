
import numpy as np
from scipy.special import eval_legendre, eval_chebyt, factorial


def build_legendre_matrix(n, x_nodes):
    x = np.asarray(x_nodes, dtype=float)
    m = len(x)
    V = np.zeros((m, n + 1), dtype=float)
    for k in range(n + 1):
        V[:, k] = eval_legendre(k, x)
    return V


def monomial_to_legendre_matrix(n):


    A = np.eye(n + 1, dtype=float)


    C = np.zeros((n + 1, n + 1), dtype=float)
    for k in range(n + 1):



        if k == 0:
            C[0, 0] = 1.0
        elif k == 1:
            C[1, 0] = -1.0
            C[1, 1] = 2.0
        else:


            prev = np.zeros(n + 1, dtype=float)
            prev2 = np.zeros(n + 1, dtype=float)
            prev[:k] = C[k - 1, :k]
            prev2[:k - 1] = C[k - 2, :k - 1]

            temp = np.zeros(n + 1, dtype=float)
            temp[0] = -prev[0]
            for j in range(1, n + 1):
                temp[j] = 2.0 * prev[j - 1] - (prev[j] if j <= k - 1 else 0.0)
            temp = temp * (2.0 * k - 1.0)
            temp[:k - 1] -= (k - 1.0) * prev2[:k - 1]
            C[k, :] = temp / float(k)




    A = np.linalg.inv(C.T)
    return A


def legendre_to_monomial_matrix(n):
    A = monomial_to_legendre_matrix(n)
    return np.linalg.inv(A)


def bernstein_to_monomial_matrix(n):
    B = np.zeros((n + 1, n + 1), dtype=float)
    for i in range(n + 1):

        for k in range(n - i + 1):
            coeff = factorial(n) / (factorial(i) * factorial(n - i))
            sign = 1.0 if k % 2 == 0 else -1.0
            coeff *= sign * (factorial(n - i) / (factorial(k) * factorial(n - i - k)))
            B[i, i + k] = coeff




    return np.linalg.inv(B.T)


def monomial_to_bernstein_matrix(n):
    return np.linalg.inv(bernstein_to_monomial_matrix(n))


def chebyshev_to_monomial_matrix(n):
    C = np.zeros((n + 1, n + 1), dtype=float)
    C[0, 0] = 1.0
    if n >= 1:
        C[1, 1] = 1.0
    for k in range(2, n + 1):

        prev = np.zeros(n + 1, dtype=float)
        prev2 = np.zeros(n + 1, dtype=float)
        prev[:k] = C[k - 1, :k]
        prev2[:k - 1] = C[k - 2, :k - 1]

        temp = np.zeros(n + 1, dtype=float)
        temp[1:k + 1] = 2.0 * prev[:k]
        temp[:k - 1] -= prev2[:k - 1]
        C[k, :] = temp

    return np.linalg.inv(C.T)


def monomial_to_chebyshev_matrix(n):
    return np.linalg.inv(chebyshev_to_monomial_matrix(n))


def gegenbauer_to_monomial_matrix(n, alpha):
    C = np.zeros((n + 1, n + 1), dtype=float)
    C[0, 0] = 1.0
    if n >= 1:
        C[1, 1] = 2.0 * alpha
    for k in range(2, n + 1):
        prev = np.zeros(n + 1, dtype=float)
        prev2 = np.zeros(n + 1, dtype=float)
        prev[:k] = C[k - 1, :k]
        prev2[:k - 1] = C[k - 2, :k - 1]
        temp = np.zeros(n + 1, dtype=float)
        temp[1:k + 1] = 2.0 * (k + alpha - 1.0) * prev[:k]
        temp[:k - 1] -= (k + 2.0 * alpha - 2.0) * prev2[:k - 1]
        C[k, :] = temp / float(k)
    return np.linalg.inv(C.T)


def monomial_to_gegenbauer_matrix(n, alpha):
    return np.linalg.inv(gegenbauer_to_monomial_matrix(n, alpha))


def spectral_interpolate_legendre(coeffs, x_query):
    x = np.asarray(x_query, dtype=float)
    n = len(coeffs) - 1
    val = np.zeros_like(x, dtype=float)
    for k in range(n + 1):
        val += coeffs[k] * eval_legendre(k, x)
    return val


def map_domain_01_to_m1p1(x):
    return 2.0 * np.asarray(x, dtype=float) - 1.0


def map_domain_m1p1_to_01(x):
    return 0.5 * (np.asarray(x, dtype=float) + 1.0)
