
import numpy as np
from typing import Tuple





def clenshaw_curtis_compute(order: int) -> Tuple[np.ndarray, np.ndarray]:
    if order < 1:
        raise ValueError("clenshaw_curtis_compute: order must be >= 1.")

    x = np.zeros(order, dtype=float)
    w = np.zeros(order, dtype=float)

    if order == 1:
        x[0] = 0.0
        w[0] = 2.0
        return x, w

    for i in range(order):
        x[i] = np.cos((order - 1 - i) * np.pi / (order - 1))

    x[0] = -1.0
    if order % 2 == 1:
        x[(order - 1) // 2] = 0.0
    x[-1] = 1.0




    raise NotImplementedError("Hole 1: Clenshaw-Curtis weight calculation not implemented.")





def jacobi_compute(n: int, alpha: float, beta: float) -> Tuple[np.ndarray, np.ndarray]:
    if n < 1:
        raise ValueError("jacobi_compute: n must be >= 1.")
    if alpha <= -1.0 or beta <= -1.0:
        raise ValueError("jacobi_compute: alpha and beta must be > -1.")

    from scipy.special import gamma as gamma_func

    zemu = (2.0 ** (alpha + beta + 1.0)) * gamma_func(alpha + 1.0) * gamma_func(beta + 1.0) / gamma_func(2.0 + alpha + beta)

    if n == 1:
        x = np.array([(beta - alpha) / (2.0 + alpha + beta)], dtype=float)
        w = np.array([zemu], dtype=float)
        return x, w

    diag = np.zeros(n, dtype=float)
    offdiag = np.zeros(n, dtype=float)

    diag[0] = (beta - alpha) / (2.0 + alpha + beta)
    offdiag[0] = 4.0 * (1.0 + alpha) * (1.0 + beta) / ((3.0 + alpha + beta) * (2.0 + alpha + beta) ** 2)

    for i in range(1, n):
        abi = 2.0 * (i + 1) + alpha + beta
        diag[i] = (beta + alpha) * (beta - alpha) / ((abi - 2.0) * abi)
        offdiag[i] = (4.0 * (i + 1) * (i + 1 + alpha) * (i + 1 + beta) * (i + 1 + alpha + beta)
                      / ((abi - 1.0) * (abi + 1.0) * abi * abi))

    offdiag = np.sqrt(offdiag)
    offdiag[-1] = 0.0


    x, eigvecs = _sym_tridiag_eig(diag, offdiag[:-1])
    w = eigvecs[0, :] ** 2 * zemu

    return x, w





def gen_hermite_compute(n: int, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    if n < 1:
        raise ValueError("gen_hermite_compute: n must be >= 1.")
    if alpha <= -1.0:
        raise ValueError("gen_hermite_compute: alpha must be > -1.")

    from scipy.special import gamma as gamma_func

    zemu = gamma_func((alpha + 1.0) / 2.0)

    offdiag = np.zeros(n, dtype=float)
    for i in range(n):
        idx = i + 1
        if idx % 2 == 1:
            offdiag[i] = (idx + alpha) / 2.0
        else:
            offdiag[i] = idx / 2.0

    offdiag = np.sqrt(offdiag)
    diag = np.zeros(n, dtype=float)

    x, eigvecs = _sym_tridiag_eig(diag, offdiag[:-1])
    w = eigvecs[0, :] ** 2 * zemu
    return x, w





def laguerre_quadrature_rule(n: int) -> Tuple[np.ndarray, np.ndarray]:
    if n < 1:
        raise ValueError("laguerre_quadrature_rule: n must be >= 1.")

    zemu = 1.0
    offdiag = np.sqrt(np.arange(1, n + 1, dtype=float))
    diag = np.array([2.0 * i - 1.0 for i in range(1, n + 1)], dtype=float)

    x, eigvecs = _sym_tridiag_eig(diag, offdiag[:-1])
    w = eigvecs[0, :] ** 2 * zemu
    return x, w





def wandzura_rule(rule_index: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    if rule_index == 1:


        a1 = 0.816847572980458
        b1 = 0.091576213509771
        w1 = 0.109951743655322

        a2 = 0.108103018168070
        b2 = 0.445948490915965
        w2 = 0.223381589678011

        xy = np.array([
            [a1, b1, b1],
            [b1, a1, b1],
            [b1, b1, a1],
            [a2, b2, b2],
            [b2, a2, b2],
            [b2, b2, a2],
        ], dtype=float)
        w = np.array([w1, w1, w1, w2, w2, w2], dtype=float)

        w = w / np.sum(w)
        return xy, w

    elif rule_index == 2:

        a1 = 0.873821971016996
        b1 = 0.063089014491502
        w1 = 0.050844906370207

        a2 = 0.501426509658179
        b2 = 0.249286745170910
        w2 = 0.116786275726379

        a3 = 0.636502499121399
        b3 = 0.310352451033785
        c3 = 0.053145049844816
        w3 = 0.082851075618374

        xy = np.array([
            [a1, b1, b1],
            [b1, a1, b1],
            [b1, b1, a1],
            [a2, b2, b2],
            [b2, a2, b2],
            [b2, b2, a2],
            [a3, b3, c3],
            [a3, c3, b3],
            [b3, a3, c3],
            [b3, c3, a3],
            [c3, a3, b3],
            [c3, b3, a3],
        ], dtype=float)
        w = np.array([w1]*3 + [w2]*3 + [w3]*6, dtype=float)
        w = w / np.sum(w)
        return xy, w

    else:
        raise ValueError("wandzura_rule: rule_index must be 1 or 2.")


def integrate_triangle(f, vertices: np.ndarray, rule_index: int = 1) -> float:
    if vertices.shape != (3, 2):
        raise ValueError("integrate_triangle: vertices must have shape (3, 2).")

    xy_bary, w = wandzura_rule(rule_index)



    phys = xy_bary @ vertices


    J = np.linalg.det(np.column_stack((vertices[1] - vertices[0], vertices[2] - vertices[0])))
    area = 0.5 * abs(J)

    vals = f(phys)
    return float(area * np.dot(w, vals))





def _sym_tridiag_eig(diag: np.ndarray, offdiag: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = diag.shape[0]
    T = np.diag(diag) + np.diag(offdiag, k=1) + np.diag(offdiag, k=-1)
    w, v = np.linalg.eigh(T)

    idx = np.argsort(w)
    return w[idx], v[:, idx]
