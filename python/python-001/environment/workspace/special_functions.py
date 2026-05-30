
import numpy as np
from typing import Tuple


class SpecialFunctionError(Exception):
    pass


def lngamma_lanczos(z: float) -> Tuple[float, int]:
    if z <= 0.0:
        return 0.0, 1

    a = np.array([
        0.9999999999995183,
        676.5203681218835,
        -1259.139216722289,
        771.3234287757674,
        -176.6150291498386,
        12.50734324009056,
        -0.1385710331296526,
        0.9934937113930748e-05,
        0.1659470187408462e-06
    ], dtype=float)

    lnsqrt2pi = 0.9189385332046727
    value = 0.0
    tmp = z + 7.0
    for j in range(8, 0, -1):
        value += a[j] / tmp
        tmp -= 1.0
    value += a[0]
    value = np.log(value) + lnsqrt2pi - (z + 6.5) + (z - 0.5) * np.log(z + 6.5)
    return value, 0


def gamma_lanczos(z: float) -> float:
    if z <= 0.0:
        if np.isclose(z, 0.0):
            return np.inf

        if not np.isclose(z, round(z)):
            lz, ier = lngamma_lanczos(1.0 - z)
            if ier != 0:
                return np.nan
            return np.pi / (np.sin(np.pi * z) * np.exp(lz))
        return np.nan
    lz, ier = lngamma_lanczos(z)
    if ier != 0:
        return np.nan
    return np.exp(lz)


def factorial_ratio(n: int, m: int) -> float:
    if n < 0 or m < 0:
        raise SpecialFunctionError("阶乘自变量必须非负")
    if n == m:
        return 1.0
    if n > m:
        ln_val = 0.0
        for k in range(m + 1, n + 1):
            ln_val += np.log(k)
        return np.exp(ln_val)
    else:
        ln_val = 0.0
        for k in range(n + 1, m + 1):
            ln_val += np.log(k)
        return np.exp(-ln_val)


def legendre_to_monomial_matrix(n_max: int) -> np.ndarray:
    if n_max < 1:
        return np.ones((1, 1))

    n = n_max
    A = np.zeros((n, n))
    A[0, 0] = 1.0
    if n > 1:
        A[1, 1] = 1.0

    for k in range(1, n - 1):

        coeff = (2.0 * k + 1.0) / (k + 1.0)
        for i in range(k + 1):
            A[i + 1, k + 1] += coeff * A[i, k]
        coeff_prev = -k / (k + 1.0)
        for i in range(k):
            A[i, k + 1] += coeff_prev * A[i, k - 1]

    return A


def monomial_to_legendre_matrix(n_max: int) -> np.ndarray:
    A = legendre_to_monomial_matrix(n_max)


    try:
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:

        return np.linalg.pinv(A)


def gegenbauer_to_monomial_matrix(n_max: int, alpha: float = 0.5) -> np.ndarray:
    if n_max < 1:
        return np.ones((1, 1))
    n = n_max
    A = np.zeros((n, n))
    A[0, 0] = 1.0
    if n > 1:
        A[1, 1] = 2.0 * alpha

    for k in range(1, n - 1):
        coeff1 = 2.0 * (k + alpha) / (k + 1.0)
        for i in range(k + 1):
            A[i + 1, k + 1] += coeff1 * A[i, k]
        coeff2 = -(k + 2.0 * alpha - 1.0) / (k + 1.0)
        for i in range(k):
            A[i, k + 1] += coeff2 * A[i, k - 1]
    return A


def associated_legendre_normalized(n: int, m: int, x: float) -> float:
    if np.abs(x) > 1.0:
        x = np.clip(x, -1.0, 1.0)
    if m > n:
        return 0.0
    if m < 0:
        m = -m


    def norm_coeff(n0, m0):
        delta = 1.0 if m0 == 0 else 0.0
        ratio = factorial_ratio(n0 - m0, n0 + m0)
        return np.sqrt((2.0 * n0 + 1.0) * (2.0 - delta) * ratio)

    p_mm = 1.0
    if m > 0:
        somx2 = np.sqrt(max(0.0, 1.0 - x * x))
        fact = 1.0
        for i in range(1, m + 1):
            p_mm *= -fact * somx2
            fact += 2.0
    p_mm *= norm_coeff(m, m)

    if n == m:
        return p_mm

    p_m1m = norm_coeff(m + 1, m) * (2.0 * m + 1.0) * x * p_mm / norm_coeff(m, m)

    if n == m + 1:
        return p_m1m

    p_nm_prev2 = p_mm
    p_nm_prev1 = p_m1m
    p_nm = 0.0
    for nn in range(m + 2, n + 1):
        n_nm = norm_coeff(nn, m)
        n_nm1 = norm_coeff(nn - 1, m)
        n_nm2 = norm_coeff(nn - 2, m)
        a = (2.0 * nn - 1.0) * x * p_nm_prev1 / n_nm1
        b = (nn + m - 1.0) * p_nm_prev2 / n_nm2
        p_nm = n_nm * (a - b) / (nn - m)
        p_nm_prev2 = p_nm_prev1
        p_nm_prev1 = p_nm

    return p_nm


def associated_legendre_schmidt(n: int, m: int, x: float) -> float:
    pbar = associated_legendre_normalized(n, m, x)
    if m == 0:
        return pbar / np.sqrt(2.0 * n + 1.0)
    else:
        return pbar / np.sqrt(2.0 * (2.0 * n + 1.0))
