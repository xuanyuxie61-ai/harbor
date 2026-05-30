
import numpy as np
from scipy.linalg import eig_banded
from typing import Tuple, Optional





def gegenbauer_rule(n: int, lambda_param: float = 0.5,
                    a: float = -1.0, b: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    if n < 1:
        raise ValueError("n must be >= 1")
    if lambda_param <= -0.5:
        raise ValueError("lambda must be > -0.5")


    aj, bj = _class_matrix_gegenbauer(n, lambda_param)




    ab = np.zeros((2, n))
    ab[0, :] = aj
    ab[1, :-1] = bj[1:]

    w, v = eig_banded(ab, lower=True)
    x = np.real(w)

    mu0 = _gegenbauer_moment0(lambda_param)
    weights = np.real(v[0, :] ** 2) * mu0


    idx = np.argsort(x)
    x = x[idx]
    weights = weights[idx]


    if a != -1.0 or b != 1.0:
        scale = (b - a) / 2.0
        shift = (a + b) / 2.0
        x = scale * x + shift
        weights = weights * scale

    return x, weights


def _class_matrix_gegenbauer(n: int, lambda_param: float) -> Tuple[np.ndarray, np.ndarray]:
    aj = np.zeros(n)
    bj = np.zeros(n)

    bj[0] = 0.0
    for i in range(1, n):
        num = i * (i + 2.0 * lambda_param - 1.0)
        den = 4.0 * (i + lambda_param - 1.0) * (i + lambda_param)
        if den <= 0:
            raise ValueError("Invalid Gegenbauer parameters")
        bj[i] = np.sqrt(num / den)

    return aj, bj


def _gegenbauer_moment0(lambda_param: float) -> float:
    from math import gamma, sqrt, pi
    if lambda_param == 0.5:
        return 2.0
    return sqrt(pi) * gamma(lambda_param + 0.5) / gamma(lambda_param + 1.0)





def triangle_gauss_rule(order: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if order == 1:

        x = np.array([1.0 / 3.0])
        y = np.array([1.0 / 3.0])
        w = np.array([0.5])
    elif order == 2:

        x = np.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0])
        y = np.array([1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0])
        w = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    elif order == 3:

        x = np.array([0.5, 0.5, 0.0])
        y = np.array([0.0, 0.5, 0.5])
        w = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    elif order == 4:

        a1 = 0.445948490915965
        b1 = 0.091576213509771
        w1 = 0.111690794839005
        w2 = 0.054975871827661
        x = np.array([a1, 1.0 - 2.0 * a1, a1,
                      b1, 1.0 - 2.0 * b1, b1])
        y = np.array([a1, a1, 1.0 - 2.0 * a1,
                      b1, b1, 1.0 - 2.0 * b1])
        w = np.array([w1, w1, w1, w2, w2, w2])
    elif order == 5:

        a1 = 0.470142064105115
        w1 = 0.066197076394253
        w2 = 0.062969590272413
        w3 = 0.1125
        x = np.array([a1, 1.0 - 2.0 * a1, a1,
                      1.0 / 3.0, 0.059715871789770,
                      0.797426985353087, 0.059715871789770])
        y = np.array([a1, a1, 1.0 - 2.0 * a1,
                      1.0 / 3.0, 0.059715871789770,
                      0.059715871789770, 0.797426985353087])
        w = np.array([w1, w1, w1, w3, w2, w2, w2])
    elif order == 6:

        a1 = 0.249286745170910
        b1 = 0.501426509658179
        a2 = 0.063089014491502
        b2 = 0.873821971016996
        a3 = 0.310352451033785
        b3 = 0.053145049844816
        w1 = 0.058393137863189
        w2 = 0.025422453185104
        w3 = 0.041425537809187
        x = np.array([a1, b1, a1, a2, b2, a2, a3, b3, 1.0 - a3 - b3,
                      a3, 1.0 - a3 - b3, b3])
        y = np.array([a1, a1, b1, a2, a2, b2, a3, a3, a3,
                      b3, b3, 1.0 - a3 - b3])
        w = np.array([w1, w1, w1, w2, w2, w2, w3, w3, w3, w3, w3, w3])
    elif order == 7:

        a1 = 0.260345966079038
        b1 = 0.065130102902216
        w1 = 0.087977301162222
        w2 = 0.008744311553736
        w3 = 0.038081799045199
        w4 = 0.018855448056131
        w5 = -0.002166998150765
        x = np.array([1.0 / 3.0, a1, 1.0 - 2.0 * a1, a1,
                      b1, 1.0 - 2.0 * b1, b1,
                      0.312865496004874, 0.638444188569809,
                      0.048690315425317, 0.638444188569809,
                      0.048690315425317, 0.312865496004874])
        y = np.array([1.0 / 3.0, a1, a1, 1.0 - 2.0 * a1,
                      b1, b1, 1.0 - 2.0 * b1,
                      0.638444188569809, 0.048690315425317,
                      0.312865496004874, 0.312865496004874,
                      0.638444188569809, 0.048690315425317])
        w = np.array([w5, w1, w1, w1, w2, w2, w2,
                      w3, w3, w3, w3, w3, w3])
    else:
        raise ValueError(f"Unsupported order {order}. Supported: 1-7.")

    return x, y, w


def map_triangle_quad_points(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray,
                             x_ref: np.ndarray, y_ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x_phys = p1[0] + (p2[0] - p1[0]) * x_ref + (p3[0] - p1[0]) * y_ref
    y_phys = p1[1] + (p2[1] - p1[1]) * x_ref + (p3[1] - p1[1]) * y_ref
    return x_phys, y_phys





def tetrahedron_unit_monomial_integral(alpha: int, beta: int, gamma: int,
                                       delta: int) -> float:
    from math import factorial
    if alpha < 0 or beta < 0 or gamma < 0 or delta < 0:
        raise ValueError("Exponents must be non-negative")
    num = factorial(alpha) * factorial(beta) * factorial(gamma) * factorial(delta)
    den = factorial(alpha + beta + gamma + delta + 3)
    return float(num) / float(den)


def monomial_value(m: int, e: np.ndarray, x: np.ndarray) -> float:
    val = 1.0
    for i in range(m):
        if e[i] != 0:
            val *= x[i] ** e[i]
    return val





def comp_next(n: int, k: int, a: np.ndarray, more: bool,
              h: int, t: int) -> Tuple[np.ndarray, bool, int, int]:
    if not more:
        a[:] = 0
        a[0] = n
        more = True
        h = 0
        t = n
        if k == 1:
            more = False
        return a, more, h, t

    if 1 < t:
        h = 0

    h = h + 1
    t = a[h - 1]
    a[h - 1] = 0
    a[0] = t - 1
    a[h] = a[h] + 1

    if a[k - 1] == n:
        more = False

    return a, more, h, t





def l2_error_estimate(uh: np.ndarray, uexact: np.ndarray,
                      weights: np.ndarray, areas: np.ndarray) -> float:
    err2 = np.sum(weights * (uh - uexact) ** 2 * areas)
    return float(np.sqrt(err2))


def h1_seminorm_error_estimate(duh: np.ndarray, duexact: np.ndarray,
                               weights: np.ndarray, areas: np.ndarray) -> float:
    diff2 = np.sum((duh[:, 0] - duexact[:, 0]) ** 2 +
                   (duh[:, 1] - duexact[:, 1]) ** 2)

    return float(np.sqrt(diff2))
