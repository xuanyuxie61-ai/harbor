
import numpy as np
from typing import Tuple






AMU_KG = 1.66053906660e-27


M_PROTON_GEV = 0.938272


M_NEUTRON_GEV = 0.939565


C_M_S = 299792458.0


GEV_TO_KG = 1.78266192e-27


KEV_TO_JOULE = 1.602176634e-16


FERMI_TO_M = 1.0e-15


RHO_LOCAL_GEV_CM3 = 0.3


V0_KM_S = 220.0


VE_KM_S = 232.0


VESC_KM_S = 544.0


KB_J_K = 1.380649e-23


H_PLANCK = 6.62607015e-34


H_BAR = H_PLANCK / (2.0 * np.pi)


KM_S_TO_M_S = 1.0e3


def gev_to_kg(m_gev: float) -> float:
    return m_gev * GEV_TO_KG






def spherical_bessel_j1(x: float) -> float:
    if abs(x) < 1.0e-8:
        return x / 3.0 - x**3 / 30.0 + x**5 / 840.0
    sx = np.sin(x)
    cx = np.cos(x)
    return sx / (x * x) - cx / x


def double_factorial(n: int) -> float:
    if n < 0:
        raise ValueError("double_factorial: n 必须非负")
    result = 1.0
    while n > 1:
        result *= float(n)
        n -= 2
    return result


def erf_approx(x: float) -> float:
    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429

    sign = 1.0 if x >= 0 else -1.0
    x = abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return sign * y


def erfc_approx(x: float) -> float:
    return 1.0 - erf_approx(x)






def r8vec_bracket4(nx: int, x: np.ndarray, xval: float) -> int:
    if nx < 2:
        raise ValueError("r8vec_bracket4: 数组长度至少为 2")
    if xval <= x[0]:
        return 0
    if xval >= x[-1]:
        return nx - 2

    lo = 0
    hi = nx - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xval < x[mid]:
            hi = mid
        else:
            lo = mid
    return lo


def r8_uniform_01(seed: int) -> Tuple[float, int]:
    IA = 16807
    IM = 2147483647
    seed = (IA * seed) % IM
    if seed == 0:
        seed = 1
    r = seed / IM
    return r, seed


def r8vec_linspace(a: float, b: float, n: int) -> np.ndarray:
    if n < 2:
        return np.array([a])
    return np.linspace(a, b, n)






def gauss_hermite_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
    if n < 1:
        raise ValueError("gauss_hermite_nodes_weights: n 必须 >= 1")
    if n == 1:
        return np.array([0.0]), np.array([np.sqrt(np.pi)])


    i = np.arange(1, n, dtype=float)
    beta = np.sqrt(i / 2.0)
    J = np.diag(beta, 1) + np.diag(beta, -1)

    eigvals, eigvecs = np.linalg.eigh(J)
    x = eigvals

    w = np.sqrt(np.pi) * (eigvecs[0, :]) ** 2
    return x, w


def gauss_hermite_quadrature(f, n: int) -> float:
    x, w = gauss_hermite_nodes_weights(n)
    return float(np.sum(w * f(x)))






def triangle_area_2d(vertices: np.ndarray) -> float:
    if vertices.shape != (3, 2):
        raise ValueError("triangle_area_2d: vertices 必须为 (3,2) 数组")
    x1, y1 = vertices[0]
    x2, y2 = vertices[1]
    x3, y3 = vertices[2]
    area = 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    return area


def barycentric_to_cartesian(tri: np.ndarray, bary: np.ndarray) -> np.ndarray:
    return bary[:, 0:1] * tri[0] + bary[:, 1:2] * tri[1] + bary[:, 2:3] * tri[2]






if __name__ == "__main__":

    assert abs(spherical_bessel_j1(0.0)) < 1e-12
    assert abs(spherical_bessel_j1(1.0) - 0.3011686789) < 1e-6


    assert double_factorial(7) == 105.0
    assert double_factorial(8) == 384.0


    x, w = gauss_hermite_nodes_weights(16)
    integral = np.sum(w * x**2)
    assert abs(integral - np.sqrt(np.pi) / 2.0) < 1e-12


    arr = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    assert r8vec_bracket4(5, arr, 1.5) == 1
    assert r8vec_bracket4(5, arr, -1.0) == 0
    assert r8vec_bracket4(5, arr, 5.0) == 3

    print("utils.py: 所有自测通过")
