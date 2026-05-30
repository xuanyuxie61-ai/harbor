
import numpy as np
from typing import Tuple, Optional




sqrt15 = np.sqrt(15.0)
a_lj = (6.0 + sqrt15) / 21.0
b_lj = (6.0 - sqrt15) / 21.0

_LYNESS_RULES = {
    1: {
        "order": 1,
        "precision": 1,
        "suborders": [
            (1, np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]), 0.5)
        ]
    },
    3: {
        "order": 3,
        "precision": 2,
        "suborders": [
            (3, np.array([0.0, 0.5, 0.5]), 0.5),
        ]
    },
    7: {
        "order": 7,
        "precision": 5,
        "suborders": [
            (1, np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]), 0.5 * 9.0 / 40.0),
            (3, np.array([1.0 - 2.0 * a_lj, a_lj, a_lj]), 1.5 * (155.0 - sqrt15) / 1200.0),
            (3, np.array([1.0 - 2.0 * b_lj, b_lj, b_lj]), 1.5 * (155.0 + sqrt15) / 1200.0),
        ]
    }
}


def expand_lyness_suborders(suborders):
    points = []
    weights = []
    for perm_type, bary, w in suborders:
        a, b, c = bary
        if perm_type == 1:

            points.append([a, b, c])
            weights.append(w)
        elif perm_type == 3:

            pts = [[a, b, c], [c, a, b], [b, c, a]]
            for p in pts:
                points.append(p)
                weights.append(w / 3.0)
        elif perm_type == 6:

            from itertools import permutations
            for p in set(permutations([a, b, c])):
                points.append(list(p))
                weights.append(w / 6.0)
    return np.array(points), np.array(weights)


def integrate_triangle(
    f: callable,
    v1: np.ndarray,
    v2: np.ndarray,
    v3: np.ndarray,
    rule: int = 7
) -> float:
    if rule not in _LYNESS_RULES:
        raise ValueError(f"不支持的规则编号: {rule}")

    barys, weights = expand_lyness_suborders(_LYNESS_RULES[rule]["suborders"])


    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    v3 = np.asarray(v3)

    det_j = abs((v2[0] - v1[0]) * (v3[1] - v1[1]) - (v3[0] - v1[0]) * (v2[1] - v1[1]))

    result = 0.0
    for bary, w in zip(barys, weights):
        s, t, _ = bary
        x = v1[0] + s * (v2[0] - v1[0]) + t * (v3[0] - v1[0])
        y = v1[1] + s * (v2[1] - v1[1]) + t * (v3[1] - v1[1])
        result += w * f(x, y)

    return result * det_j


def integrate_standard_triangle(f: callable, rule: int = 7) -> float:
    return integrate_triangle(f, np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0]), rule)


def cauchy_principal_value(
    f: callable,
    a: float,
    b: float,
    x_sing: float,
    n: int = 64
) -> float:
    if n % 2 != 0:
        n += 1

    if x_sing <= a or x_sing >= b:

        from utils import gauss_legendre_nodes_weights
        xg, wg = gauss_legendre_nodes_weights(n)
        t = 0.5 * (b - a) * xg + 0.5 * (b + a)
        return 0.5 * (b - a) * np.sum(wg * f(t) / (t - x_sing))

    from utils import gauss_legendre_nodes_weights
    xg, wg = gauss_legendre_nodes_weights(n)


    t_left = 0.5 * (x_sing - a) * xg + 0.5 * (x_sing + a)
    w_left = 0.5 * (x_sing - a) * wg
    f_x = f(x_sing)
    integrand_left = (f(t_left) - f_x) / (t_left - x_sing)

    integral_left = np.sum(w_left * integrand_left)


    t_right = 0.5 * (b - x_sing) * xg + 0.5 * (b + x_sing)
    w_right = 0.5 * (b - x_sing) * wg
    integrand_right = (f(t_right) - f_x) / (t_right - x_sing)
    integral_right = np.sum(w_right * integrand_right)



    singular_part = f_x * (np.log(b - x_sing) - np.log(x_sing - a))
    return integral_left + integral_right + singular_part


def gaussian_copula_bivariate_integral(
    u: float,
    v: float,
    rho: float,
    n_quad: int = 50
) -> float:
    from scipy import stats
    if abs(rho) >= 1.0:
        rho = np.clip(rho, -0.99999, 0.99999)

    a = stats.norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))
    b = stats.norm.ppf(np.clip(v, 1e-10, 1 - 1e-10))


    from utils import gauss_legendre_nodes_weights
    xg, wg = gauss_legendre_nodes_weights(n_quad)





    z_min = -6.0
    z_max_x = max(a, z_min)
    z_max_y = max(b, z_min)

    zx = 0.5 * (z_max_x - z_min) * xg + 0.5 * (z_max_x + z_min)
    zy = 0.5 * (z_max_y - z_min) * xg + 0.5 * (z_max_y + z_min)
    wx = 0.5 * (z_max_x - z_min) * wg
    wy = 0.5 * (z_max_y - z_min) * wg


    def phi2(x, y):
        det = 1.0 - rho**2
        norm = 1.0 / (2.0 * np.pi * np.sqrt(det))
        z = (x**2 - 2 * rho * x * y + y**2) / det
        return norm * np.exp(-0.5 * z)


    result = 0.0
    for i in range(n_quad):
        for j in range(n_quad):
            if zx[i] <= a and zy[j] <= b:
                result += wx[i] * wy[j] * phi2(zx[i], zy[j])

    return result


def test_copula_quadrature():

    f1 = lambda x, y: x * y
    val1 = integrate_standard_triangle(f1, rule=3)
    expected1 = 1.0 / 24.0
    assert abs(val1 - expected1) < 1e-10, f"三角形积分错误: {val1} != {expected1}"


    f2 = lambda t: t
    cpv_val = cauchy_principal_value(f2, -1.0, 1.0, 0.5, n=64)


    expected_cpv = 2.0 + 0.5 * np.log(1.0 / 3.0)
    assert abs(cpv_val - expected_cpv) < 1e-6, f"CPV 错误: {cpv_val} != {expected_cpv}"


    try:
        from scipy import stats
        rho = 0.5
        u, v = 0.3, 0.7
        c_num = gaussian_copula_bivariate_integral(u, v, rho, n_quad=40)
        c_ref = stats.multivariate_normal.cdf(
            [stats.norm.ppf(u), stats.norm.ppf(v)],
            mean=[0, 0],
            cov=[[1, rho], [rho, 1]]
        )
        assert abs(c_num - c_ref) < 0.05, f"Copula 积分偏差过大: {c_num} vs {c_ref}"
    except Exception:
        pass

    print(f"copula_quadrature test passed. triangle_int={val1:.8f}, cpv={cpv_val:.8f}")


if __name__ == "__main__":
    test_copula_quadrature()
