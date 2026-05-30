
import numpy as np
from typing import Tuple, List


def legendre_polynomial(x: np.ndarray, n: int) -> np.ndarray:
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x.copy()
    P_nm2 = np.ones_like(x)
    P_nm1 = x.copy()
    P_n = np.zeros_like(x)
    for k in range(1, n):
        P_n = ((2 * k + 1) * x * P_nm1 - k * P_nm2) / (k + 1)
        P_nm2, P_nm1 = P_nm1, P_n
    return P_n


def legendre_derivative(x: np.ndarray, n: int) -> np.ndarray:
    if n == 0:
        return np.zeros_like(x)
    Pn = legendre_polynomial(x, n)
    Pnm1 = legendre_polynomial(x, n - 1)
    eps = 1e-14
    denom = 1.0 - x ** 2
    denom = np.where(np.abs(denom) < eps, eps, denom)
    return n * (Pnm1 - x * Pn) / denom


def gll_nodes_weights(p: int) -> Tuple[np.ndarray, np.ndarray]:
    if p < 1:
        raise ValueError("Degree p must be >= 1.")
    n_nodes = p + 1
    xi = np.zeros(n_nodes, dtype=float)
    w = np.zeros(n_nodes, dtype=float)


    xi[0] = -1.0
    xi[-1] = 1.0

    if p == 1:
        w[:] = 1.0
        return xi, w



    for i in range(1, p):
        xi[i] = -np.cos(np.pi * i / p)

    max_iter = 100
    tol = 1e-14
    for i in range(1, p):
        x_old = xi[i] + 2 * tol
        x = xi[i]
        it = 0
        while abs(x - x_old) > tol and it < max_iter:
            x_old = x
            dP = legendre_derivative(x, p)
            ddP = (2.0 * x * dP - p * (p + 1) * legendre_polynomial(x, p)) / (1.0 - x ** 2 + 1e-15)
            if abs(ddP) < 1e-15:
                break
            x = x - dP / ddP
            it += 1
        xi[i] = x


    xi.sort()


    for i in range(n_nodes):
        Pp = legendre_polynomial(xi[i], p)
        w[i] = 2.0 / (p * (p + 1) * Pp ** 2 + 1e-15)

    return xi, w


def affine_map(xi: np.ndarray, a: float, b: float) -> np.ndarray:
    return 0.5 * (b - a) * xi + 0.5 * (a + b)


def fekete_line_rule(a: float, b: float, p: int) -> Tuple[np.ndarray, np.ndarray]:
    xi, wi = gll_nodes_weights(p)
    x = affine_map(xi, a, b)

    w = wi * 0.5 * (b - a)
    return x, w


def fekete_triangle_nodes_weights(p: int) -> Tuple[np.ndarray, np.ndarray]:
    if p > 8:

        p = 8


    xi1d, w1d = gll_nodes_weights(p)


    n1d = p + 1
    pts_sq = []
    for i in range(n1d):
        for j in range(n1d):
            pts_sq.append([xi1d[i], xi1d[j]])
    pts_sq = np.array(pts_sq, dtype=float)






    xi01, w01 = gll_nodes_weights(p)
    xi01 = 0.5 * (xi01 + 1.0)
    w01 = 0.5 * w01

    pts_tri = []
    w_tri = []
    for i in range(n1d):
        for j in range(n1d - i):



            x = xi01[i]
            y = xi01[j] * (1.0 - x)

            w = w01[i] * w01[j] * (1.0 - x)
            pts_tri.append([x, y])
            w_tri.append(w)

    pts_tri = np.array(pts_tri, dtype=float)
    w_tri = np.array(w_tri, dtype=float)

    total = np.sum(w_tri)
    if total > 0:
        w_tri *= 0.5 / total
    return pts_tri, w_tri


def vandermonde_1d(nodes: np.ndarray, p: int) -> np.ndarray:
    n = nodes.shape[0]
    V = np.zeros((n, p + 1), dtype=float)
    for j in range(p + 1):
        V[:, j] = legendre_polynomial(nodes, j)
    return V


def fekete_approximate_1d(a: float, b: float, p: int) -> Tuple[np.ndarray, np.ndarray]:
    return fekete_line_rule(a, b, p)


if __name__ == "__main__":

    for p in [1, 2, 3, 4, 5]:
        x, w = fekete_line_rule(0.0, 1.0, p)
        print(f"p={p}: nodes={x}, weights={w}, sum={np.sum(w):.6f}")
