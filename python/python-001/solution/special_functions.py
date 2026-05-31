"""
special_functions.py

基于 asa245 (Lanczos Gamma 对数近似) 与多项式转换核心算法，
提供小行星引力场建模所需的特殊函数与正交多项式工具。

科学背景：
- 球谐系数估计中涉及 Gamma 函数（统计分布、阶乘推广）。
- Legendre / Chebyshev / Gegenbauer 多项式用于引力势展开：
  U(r,θ,φ) = (GM/r) Σ_{n=0}^{∞} Σ_{m=0}^{n} (R_e/r)^n P_{nm}(sinφ)
             × [C_{nm} cos(mλ) + S_{nm} sin(mλ)]
"""

import numpy as np
from typing import Tuple


class SpecialFunctionError(Exception):
    pass


def lngamma_lanczos(z: float) -> Tuple[float, int]:
    """
    Lanczos 近似计算 ln(Gamma(z))，精度达 14 位以上。

    公式（Lanczos 级数）：
        ln Γ(z) ≈ ln[√(2π)] + (z−0.5) ln(z+6.5) − (z+6.5)
                  + ln[ Σ_{j=1}^{9} a_j / (z+j−1) ]

    参数:
        z: 实数自变量，必须 > 0。

    返回:
        (value, ier): ier=0 成功，ier=1 表示 z ≤ 0。
    """
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
    """
    通过 Lanczos 近似计算 Gamma(z)。
    边界处理：对非正实数返回 nan 并发出警告。
    """
    if z <= 0.0:
        if np.isclose(z, 0.0):
            return np.inf
        # 对负非整数，利用反射公式计算
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
    """
    计算 n! / m! 的数值稳定版本，使用对数域运算。
    用于球谐函数归一化系数中的阶乘比。
    """
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
    """
    构建 Legendre 基到单项式基的转换矩阵 A，使得
        mcoef = A @ lcoef
    其中 A[i,j] 表示 Legendre_j 在单项式 x^i 中的系数。

    Legendre 多项式的三项递推关系：
        (n+1) P_{n+1}(x) = (2n+1) x P_n(x) − n P_{n−1}(x)
    """
    if n_max < 1:
        return np.ones((1, 1))

    n = n_max
    A = np.zeros((n, n))
    A[0, 0] = 1.0  # P_0(x) = 1
    if n > 1:
        A[1, 1] = 1.0  # P_1(x) = x

    for k in range(1, n - 1):
        # P_{k+1} 的系数由 P_k 和 P_{k-1} 递推得到
        coeff = (2.0 * k + 1.0) / (k + 1.0)
        for i in range(k + 1):
            A[i + 1, k + 1] += coeff * A[i, k]
        coeff_prev = -k / (k + 1.0)
        for i in range(k):
            A[i, k + 1] += coeff_prev * A[i, k - 1]

    return A


def monomial_to_legendre_matrix(n_max: int) -> np.ndarray:
    """
    构建单项式基到 Legendre 基的转换矩阵，
    利用正交性:  ∫_{-1}^{1} P_m(x) P_n(x) dx = 2/(2n+1) δ_{mn}
    """
    A = legendre_to_monomial_matrix(n_max)
    # A 将 Legendre 系数映射到单项式系数: m = A @ l
    # 因此 l = A^{-1} @ m
    try:
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:
        # 对于病态矩阵使用伪逆
        return np.linalg.pinv(A)


def gegenbauer_to_monomial_matrix(n_max: int, alpha: float = 0.5) -> np.ndarray:
    """
    Gegenbauer (超球) 多项式到单项式的转换矩阵。
    在轨道力学中，Gegenbauer 多项式出现在展开式
        1/|r - r'| = Σ_{n=0}^{∞} (r_<^n / r_>^{n+1}) C_n^{(1/2)}(cosγ)
    其中 C_n^{(1/2)} 就是 Legendre 多项式。
    """
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
    """
    计算全归一化缔合 Legendre 函数 P̄_{nm}(x)。
    使用递推公式（Heiskanen & Moritz, 1967）。

    归一化系数:
        N_{nm} = √[(2n+1)(2−δ_{m0})(n−m)! / (n+m)!]

    递推关系（列递推，m→m+1）:
        P̄_{mm}(x) = N_{mm} (2m−1)!! (1−x²)^{m/2}
        P̄_{m+1,m}(x) = N_{m+1,m} (2m+1) x P_{mm}(x)
        P̄_{n,m}(x) = [ (2n−1) x P̄_{n−1,m} − (n+m−1) P̄_{n−2,m} ] / (n−m)
    """
    if np.abs(x) > 1.0:
        x = np.clip(x, -1.0, 1.0)
    if m > n:
        return 0.0
    if m < 0:
        m = -m

    # 计算 N_{nm}
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
    """
    Schmidt 半归一化缔合 Legendre 函数，常用于地学和行星科学。
    """
    pbar = associated_legendre_normalized(n, m, x)
    if m == 0:
        return pbar / np.sqrt(2.0 * n + 1.0)
    else:
        return pbar / np.sqrt(2.0 * (2.0 * n + 1.0))
