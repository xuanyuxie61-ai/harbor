"""
special_functions.py
=====================
核物理高精度特殊函数计算模块

基于种子项目 323_e_spigot 的任意精度算法思想 (spigot 算法
通过整数数组和混合进制运算逐位计算常数)，本模块实现
核反应光学模型中所需的高精度特殊函数：
1. Coulomb 波函数 F_L(η, ρ), G_L(η, ρ)
2. 球 Bessel 函数 j_l(x), n_l(x) 的高精度版本
3. Gamma 函数对数 (用于核统计模型)
4. 误差函数的复数延拓 (用于光学势的色散关系)

核心公式
--------
Coulomb 波函数递推 (对带电粒子光学模型):
    d²F/dρ² + [1 - 2η/ρ - L(L+1)/ρ²] F = 0

递推关系:
    F_{L+1} = [(2L+1)(1 + η/L(L+1))ρ F_L - L F_{L-1}] / (L+1)

Gamma 函数对数 Stirling 渐近展开:
    ln Γ(z) ≈ (z - 1/2) ln z - z + (1/2) ln(2π)
             + Σ_{k=1}^n B_{2k} / (2k(2k-1) z^{2k-1})

其中 B_{2k} 为 Bernoulli 数。

误差函数复数延拓:
    erf(z) = (2/√π) Σ_{n=0}^∞ (-1)^n z^{2n+1} / [n!(2n+1)]
"""

import numpy as np


# Bernoulli 数 (用于 Stirling 展开)
_BERNOULLI_NUMBERS = [
    1.0 / 6.0,
    -1.0 / 30.0,
    1.0 / 42.0,
    -1.0 / 30.0,
    5.0 / 66.0,
    -691.0 / 2730.0,
    7.0 / 6.0,
    -3617.0 / 510.0,
]


def log_gamma_stirling(z, n_terms=4):
    """
    使用 Stirling 渐近展开计算 ln Γ(z)。

    适用于 |z| 较大或 Re(z) > 0 的情况。

    Parameters
    ----------
    z : complex or float
        自变量。
    n_terms : int
        渐近展开的项数。

    Returns
    -------
    ln_Gamma : complex or float
        ln Γ(z)。
    """
    z = complex(z)
    if z.real <= 0:
        # 使用反射公式: Γ(z)Γ(1-z) = π / sin(πz)
        # ln Γ(z) = ln π - ln sin(πz) - ln Γ(1-z)
        return np.log(np.pi) - np.log(np.sin(np.pi * z)) - log_gamma_stirling(1.0 - z, n_terms)

    # Stirling 主项
    result = (z - 0.5) * np.log(z) - z + 0.5 * np.log(2.0 * np.pi)

    # Bernoulli 修正项
    for k in range(1, min(n_terms + 1, len(_BERNOULLI_NUMBERS) + 1)):
        B2k = _BERNOULLI_NUMBERS[k - 1]
        result += B2k / (2.0 * k * (2.0 * k - 1.0) * (z ** (2.0 * k - 1.0)))

    return result


def gamma_function(z):
    """Gamma 函数 Γ(z) = exp(ln Γ(z))。"""
    return np.exp(log_gamma_stirling(z))


def coulomb_wave_function_series(L, eta, rho, max_iter=1000, tol=1e-14):
    """
    使用级数展开计算正规 Coulomb 波函数 F_L(η, ρ)。

    F_L(η, ρ) = C_L(η) ρ^{L+1} Σ_{k=0}^∞ a_k ρ^k

    其中 C_L(η) = 2^L e^{-πη/2} |Γ(L+1+iη)| / Γ(2L+2)
    递推系数:
        a_0 = 1
        a_1 = η / (L+1)
        a_{k} = [2η a_{k-1} - a_{k-2}] / [k(k + 2L + 1)]

    Parameters
    ----------
    L : int
        角动量量子数。
    eta : float
        Sommerfeld 参数 η = Z1 Z2 e² / (ħv)。
    rho : float
        无量纲径向坐标 ρ = kr。
    max_iter : int
        最大迭代次数。
    tol : float
        收敛容差。

    Returns
    -------
    F_L : float
        正规 Coulomb 波函数值。
    """
    if rho <= 0:
        return 0.0

    # 归一化常数 C_L(η)
    log_cl = L * np.log(2.0) - 0.5 * np.pi * eta
    log_cl += log_gamma_stirling(L + 1.0 + 1j * eta).real
    log_cl -= log_gamma_stirling(2.0 * L + 2.0).real
    C_L = np.exp(log_cl)

    # 级数系数递推
    a_km2 = 1.0
    a_km1 = eta / (L + 1.0)

    sum_val = a_km2 + a_km1 * rho
    rho_pow = rho ** 2

    for k in range(2, max_iter + 1):
        denom = k * (k + 2 * L + 1)
        a_k = (2.0 * eta * a_km1 - a_km2) / denom
        term = a_k * rho_pow
        sum_val += term

        if abs(term) < tol * abs(sum_val):
            break

        a_km2 = a_km1
        a_km1 = a_k
        rho_pow *= rho

    F_L = C_L * (rho ** (L + 1.0)) * sum_val
    return F_L


def spherical_bessel_jn_highprecision(x, n, max_iter=1000):
    """
    高精度球 Bessel 函数 j_n(x) 计算。

    使用 Miller 算法 (向下递推) 提高大 n 时的数值稳定性。

    核心思想: 从远高于目标阶数的 M 开始向下递推
    j_M 和 j_{M+1} 的初值设为 0 和 1，递推后归一化。

    j_{n-1}(x) = (2n+1)/x j_n(x) - j_{n+1}(x)
    """
    x = float(x)
    if x < 1e-15:
        return 1.0 if n == 0 else 0.0

    if n == 0:
        return np.sin(x) / x
    if n == 1:
        return np.sin(x) / (x ** 2) - np.cos(x) / x

    # Miller 算法
    M = n + int(np.sqrt(10.0 * n)) + 20  # 起始阶数
    j = np.zeros(M + 2)
    j[M] = 1.0
    j[M + 1] = 0.0

    # 向下递推
    for k in range(M, 0, -1):
        j[k - 1] = (2.0 * k + 1.0) / x * j[k] - j[k + 1]

    # 归一化: j_0(x) 应该等于 sin(x)/x
    scale = np.sin(x) / x / j[0]
    return j[n] * scale


def spherical_neumann_nn_highprecision(x, n):
    """
    高精度球 Neumann 函数 n_n(x)。

    使用向上递推 (n_0, n_1 已知)。
    """
    x = float(x)
    if x < 1e-15:
        return -1e10
    if n == 0:
        return -np.cos(x) / x
    if n == 1:
        return -np.cos(x) / (x ** 2) - np.sin(x) / x

    nm2 = -np.cos(x) / x
    nm1 = -np.cos(x) / (x ** 2) - np.sin(x) / x
    for k in range(2, n + 1):
        nn = (2.0 * k - 1.0) / x * nm1 - nm2
        nm2 = nm1
        nm1 = nn
    return nm1


def complex_error_function(z, n_terms=20):
    """
    复误差函数 erf(z) 的级数计算。

    erf(z) = (2/√π) Σ_{n=0}^∞ (-1)^n z^{2n+1} / [n! (2n+1)]

    用于光学势的色散关系计算 (如 Lane 一致性)。
    """
    z = complex(z)
    result = 0.0 + 0.0j
    z2 = z * z
    term = z
    factorial_n = 1.0

    for n in range(n_terms):
        result += term / (factorial_n * (2.0 * n + 1.0))
        term *= -z2
        factorial_n *= (n + 1.0)

    return (2.0 / np.sqrt(np.pi)) * result


def coulomb_phase_shift(L, eta):
    """
    计算 Coulomb 相移 σ_L = arg Γ(L+1+iη)。

    σ_L = Σ_{k=1}^L arctan(η/k) + σ_0
    σ_0 = arg Γ(1+iη)
    """
    sigma0 = log_gamma_stirling(1.0 + 1j * eta).imag
    sigma = sigma0
    for k in range(1, L + 1):
        sigma += np.arctan(eta / k)
    return sigma


def penetration_factor(l, k, R, eta=0.0):
    """
    计算势垒穿透因子 (WKB 近似)。

    P_l = kR / [F_l²(η, kR) + G_l²(η, kR)]

    简化版本：使用离心势垒近似。
    """
    rho = k * R
    if eta == 0.0:
        # 中性粒子，使用球 Bessel 函数
        jl = spherical_bessel_jn_highprecision(rho, l)
        nl = spherical_neumann_nn_highprecision(rho, l)
        denominator = (rho * jl) ** 2 + (rho * nl) ** 2
    else:
        # 带电粒子，使用 Coulomb 函数的近似
        FL = coulomb_wave_function_series(l, eta, rho)
        # G_L 的近似 (使用渐近行为)
        GL = 1.0 / FL if abs(FL) > 1e-10 else 1e10
        denominator = FL ** 2 + GL ** 2

    if denominator < 1e-30:
        denominator = 1e-30
    return rho / denominator


if __name__ == "__main__":
    # 自检
    print("ln Γ(5.5) =", log_gamma_stirling(5.5).real, "期望 ~3.958")
    print("Γ(5.5) =", gamma_function(5.5).real, "期望 ~52.34")

    print("j_5(3.0) =", spherical_bessel_jn_highprecision(3.0, 5))
    print("n_2(2.0) =", spherical_neumann_nn_highprecision(2.0, 2))

    print("erf(1+1j) =", complex_error_function(1.0 + 1.0j))
    print("Coulomb σ_2(η=1) =", coulomb_phase_shift(2, 1.0))

    F = coulomb_wave_function_series(0, 1.0, 2.0)
    print("F_0(η=1, ρ=2) =", F)
