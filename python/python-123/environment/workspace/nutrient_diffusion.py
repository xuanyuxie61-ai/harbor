
import numpy as np
from typing import Callable, Tuple


def laplace_radial_2d_exact(
    x: np.ndarray, y: np.ndarray, a: float, b: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r2 = x ** 2 + y ** 2
    r2 = np.where(r2 < 1e-15, 1e-15, r2)
    r = np.sqrt(r2)

    u = a * np.log(r) + b
    ux = a * x / r2
    uy = a * y / r2
    uxx = a * (-2.0 * x ** 2 / r2 + 1.0) / r2
    uxy = -2.0 * a * x * y / (r2 ** 2)
    uyy = a * (-2.0 * y ** 2 / r2 + 1.0) / r2

    return u, ux, uy, uxx, uxy, uyy


def laplace_radial_3d_exact(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, a: float, b: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r2 = x ** 2 + y ** 2 + z ** 2
    r2 = np.where(r2 < 1e-15, 1e-15, r2)
    r = np.sqrt(r2)

    u = a / r + b
    ux = -a * x / (r2 * r)
    uy = -a * y / (r2 * r)
    uz = -a * z / (r2 * r)
    return u, ux, uy, uz


def oxygen_diffusion_steady_state_radial(
    r: np.ndarray, R_tumor: float, C_boundary: float,
    D: float = 1.0e-5, consumption_rate: float = 0.01
) -> np.ndarray:
    from scipy.special import i0, k0, i1

    r = np.asarray(r, dtype=float)
    r = np.clip(r, 1e-12, R_tumor)

    alpha = np.sqrt(consumption_rate / D)




    i0_alphaR = i0(alpha * R_tumor)
    if abs(i0_alphaR) < 1e-15:
        i0_alphaR = 1e-15

    A = C_boundary / i0_alphaR
    C = A * i0(alpha * r)


    C = np.clip(C, 0.0, C_boundary)
    return C


def cc_abscissa(order: int, i: int) -> float:
    if order < 1:
        raise ValueError("cc_abscissa: order 必须 >= 1")
    if i < 1 or i > order:
        raise ValueError("cc_abscissa: i 超出范围")
    if order == 1:
        return 0.0
    if 2 * (order - i) == order - 1:
        return 0.0
    return np.cos((order - i) * np.pi / (order - 1))


def cc_weights(n: int) -> np.ndarray:
    if n < 1:
        raise ValueError("cc_weights: n 必须 >= 1")
    w = np.zeros(n)
    if n == 1:
        w[0] = 2.0
        return w

    theta = np.zeros(n)
    for i in range(n):
        theta[i] = i * np.pi / (n - 1)

    for i in range(n):
        w[i] = 1.0
        for j in range(1, (n - 1) // 2 + 1):
            if 2 * j == n - 1:
                b = 1.0
            else:
                b = 2.0
            w[i] -= b * np.cos(2.0 * j * theta[i]) / (4.0 * j * j - 1.0)

    w[0] = w[0] / (n - 1)
    w[1:n - 1] = 2.0 * w[1:n - 1] / (n - 1)
    w[n - 1] = w[n - 1] / (n - 1)
    return w


def clenshaw_curtis_integrate(f: Callable[[np.ndarray], np.ndarray],
                               a: float, b: float, n: int) -> float:
    if n < 1:
        raise ValueError("clenshaw_curtis_integrate: n 必须 >= 1")
    if b <= a:
        raise ValueError("clenshaw_curtis_integrate: 需要 b > a")

    t = np.array([cc_abscissa(n, i + 1) for i in range(n)])
    w = cc_weights(n)
    x = 0.5 * ((1.0 - t) * a + (t + 1.0) * b)
    fx = f(x)
    fx = np.asarray(fx).ravel()
    if fx.shape[0] != n:
        raise ValueError("clenshaw_curtis_integrate: f 输出维度不匹配")

    q = np.sum(w * fx) * (b - a) / 2.0
    return float(q)


def sparse_grid_monomial_integral(dim: int, level: int,
                                   exponents: np.ndarray) -> float:
    if dim < 1:
        raise ValueError("sparse_grid_monomial_integral: dim >= 1")
    if level < 0:
        raise ValueError("sparse_grid_monomial_integral: level >= 0")
    exponents = np.asarray(exponents, dtype=int)
    if exponents.shape[0] != dim:
        raise ValueError("sparse_grid_monomial_integral: exponents 长度不匹配")

    def _level_to_order_closed(lv):
        if lv == 0:
            return 1
        return 2 ** lv + 1

    def _cc_rule(lv):
        o = _level_to_order_closed(lv)
        pts = np.array([cc_abscissa(o, i + 1) for i in range(o)])
        ws = cc_weights(o)

        pts = 0.5 * (pts + 1.0)
        ws = ws * 0.5
        return pts, ws

    def _eval_1d(pts, alpha):
        return pts ** alpha

    def _tensor_product_integral(l1, l2):
        p1, w1 = _cc_rule(l1)
        p2, w2 = _cc_rule(l2)
        f1 = _eval_1d(p1, exponents[0])
        f2 = _eval_1d(p2, exponents[1]) if dim > 1 else np.ones_like(p2)

        val = 0.0
        for i in range(len(p1)):
            for j in range(len(p2)):
                val += w1[i] * w2[j] * f1[i] * f2[j]
        return val

    if dim == 1:

        total = 0.0
        for l1 in range(level + 1):
            p1, w1 = _cc_rule(l1)
            f1 = _eval_1d(p1, exponents[0])
            if l1 == 0:
                total += np.sum(w1 * f1)
            else:
                total += np.sum(w1 * f1) - np.sum(_cc_rule(l1 - 1)[1] * _eval_1d(_cc_rule(l1 - 1)[0], exponents[0]))
        return total

    if dim == 2:




        from math import comb
        total = 0.0
        max_sum = level + 1
        for l1 in range(1, max_sum + 1):
            for l2 in range(1, max_sum + 1):
                s = l1 + l2
                if s > max_sum + 1:
                    continue
                k = max_sum + 1 - s
                if k < 0 or k > 1:
                    continue
                alpha = ((-1) ** k) * comb(1, k)
                q = _tensor_product_integral(l1, l2)
                total += alpha * q
        return total


    total = 1.0
    for d_idx in range(dim):
        p, w = _cc_rule(level)
        f = _eval_1d(p, exponents[d_idx])
        total *= np.sum(w * f)
    return total


def michaelis_menten_consumption(C: np.ndarray, rho: np.ndarray,
                                  Vmax: float, Km: float) -> np.ndarray:


    raise NotImplementedError("Hole_1: michaelis_menten_consumption 待实现")



def hypoxia_region_fraction(C: np.ndarray, threshold: float = 0.02) -> float:
    if C.size == 0:
        return 0.0
    return float(np.mean(C < threshold))
