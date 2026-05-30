
import numpy as np


def hermite_he_prob(n, x):
    x = np.asarray(x, dtype=float)
    if n < 0:
        return np.zeros_like(x)
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x
    
    he_prev2 = np.ones_like(x)
    he_prev1 = x
    
    for k in range(1, n):
        he_curr = x * he_prev1 - k * he_prev2
        he_prev2 = he_prev1
        he_prev1 = he_curr
    
    return he_prev1


def hermite_he_prob_matrix(degree, x):
    x = np.asarray(x, dtype=float)
    n_points = x.size if x.ndim == 0 else x.shape[0]
    if x.ndim == 0:
        x = np.array([x])
        n_points = 1
    
    H = np.zeros((n_points, degree + 1))
    H[:, 0] = 1.0
    if degree >= 1:
        H[:, 1] = x
    for k in range(1, degree):
        H[:, k + 1] = x * H[:, k] - k * H[:, k - 1]
    return H


def he_double_product_integral(i, j):
    if i != j:
        return 0.0

    import math
    return float(math.factorial(i))


def he_triple_product_integral(i, j, k):
    if (i + j + k) % 2 == 1:
        return 0.0
    s = (i + j + k) // 2
    if s < i or s < j or s < k:
        return 0.0
    
    a = s - k
    b = s - i
    c = s - j
    


    import math
    num = math.factorial(i) * math.factorial(j)
    den = math.factorial(a) * math.factorial(b) * math.factorial(c) * math.factorial(k)
    return float(num) / float(den)


def build_pce_galerkin_matrix(degree, alpha_mu, alpha_sigma):







    raise NotImplementedError("HOLE 1: build_pce_galerkin_matrix 待修复")


def vandermonde_matrix(n, x):
    x = np.asarray(x, dtype=float)
    V = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == 0:
                V[i, j] = 1.0
            else:
                V[i, j] = x[j] ** i
    return V


def vandermonde_solve(V, b):
    cond = np.linalg.cond(V)
    if cond > 1e14:

        c, residuals, rank, s = np.linalg.lstsq(V, b, rcond=1e-14)
        return c
    return np.linalg.solve(V, b)


def newton_basis_vandermonde(degree, nodes):
    nodes = np.asarray(nodes, dtype=float)
    m = len(nodes)
    n = degree + 1
    V = np.zeros((m, n))
    V[:, 0] = 1.0
    for j in range(1, n):
        V[:, j] = V[:, j - 1] * (nodes - nodes[j - 1])
    return V
