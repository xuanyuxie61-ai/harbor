"""
quadrature_physics.py
=====================
高维动量空间积分：Gegenbauer-Clenshaw-Curtis 积分与 Alpert 奇异积分规则。

原项目映射：
  - 460_gegenbauer_cc：Gegenbauer 权函数积分
  - 004_alpert_rule：Alpert 奇异/振荡积分规则

物理背景
--------
在格点 QCD 中，许多物理量（如衰变常数、形状因子、圈图修正）
需要对布里渊区（Brillouin zone）内的动量进行积分：

    I = ∫_{-π}^{π} d^4k / (2π)^4  f(k)

对于具有球对称性的问题，可将积分分解为径向与角向部分：

    ∫ d^4k f(|k|, k̂) = ∫_0^∞ |k|^3 d|k| ∫ dΩ_3 f(|k|, k̂)

其中三维角向积分可利用 Gegenbauer 多项式展开。
Gegenbauer 权函数为：

    w_λ(x) = (1 - x^2)^{λ - 1/2}

积分公式：

    ∫_{-1}^{1} w_λ(x) f(x) dx ≈ Γ(λ + 1/2) √π / Γ(λ + 1) * u

其中 u 由 Chebyshev 系数递推计算。

对于格点传播子中的奇异核（如 1/(p^2 + m^2) 在 p→0 时），
Alpert 规则通过对数/幂律权函数的精确处理，消除数值发散。

核心公式
--------
1. Gegenbauer 积分：
   设 f(x) 的 Chebyshev 偶展开为
        f(x) = Σ_{r=0}^{s} a_{2r} T_{2r}(x)
   则
        I_λ[f] = Γ(λ+1/2)√π / Γ(λ+1) * u_0
   其中 u_r 满足递推：
        u_{s-1} = 0.5 (σ+1) a_{2s}
        u_{r-1} = (r - λ)/(r + λ + 1) u_r + a_{2r},  r = s-1, ..., 1
        u_0 = -λ/(λ+1) u_1 + 0.5 a_0

2. Alpert 对数奇异积分：
   对于 ∫_0^1 x^α log(x) f(x) dx，使用广义 Gauss-Legendre 节点与权重，
   其中 f(x) 为光滑函数。
"""

import numpy as np


def chebyshev_even_coeffs(n: int, f):
    """
    计算 f(x) 在 [-1,1] 上的偶 Chebyshev 系数（实用节点）。

    节点：x_j = cos(j π / n), j = 0, ..., n
    利用 DCT 计算系数 a_{2r}。
    """
    j = np.arange(n + 1)
    x = np.cos(j * np.pi / n)
    fx = f(x)
    # 简化的 DCT 计算偶系数
    s = n // 2
    a2 = np.zeros(s + 1)
    for r in range(s + 1):
        val = 0.0
        for j_idx in range(n + 1):
            weight = 1.0 if (j_idx == 0 or j_idx == n) else 2.0
            val += weight * fx[j_idx] * np.cos(2 * r * j_idx * np.pi / n)
        a2[r] = val / (2 * n)
    return a2


def gegenbauer_cc(n: int, lambda_param: float, f) -> float:
    """
    Gegenbauer-Clenshaw-Curtis 积分。

    计算 ∫_{-1}^{1} (1 - x^2)^{λ - 1/2} f(x) dx

    Parameters
    ----------
    n : int
        节点数。
    lambda_param : float
        Gegenbauer 参数 λ > -0.5。
    f : callable
        被积函数。

    Returns
    -------
    value : float
        积分估计值。
    """
    if lambda_param <= -0.5:
        raise ValueError("lambda must be > -0.5")
    a2 = chebyshev_even_coeffs(n, f)
    s = n // 2
    sigma = n % 2
    u = 0.5 * (sigma + 1.0) * a2[s]
    for rh in range(s - 1, 0, -1):
        u = (rh - lambda_param) / (rh + lambda_param + 1.0) * u + a2[rh]
    u = -lambda_param * u / (lambda_param + 1.0) + 0.5 * a2[0]

    from math import gamma, sqrt, pi
    value = gamma(lambda_param + 0.5) * sqrt(np.pi) * u / gamma(lambda_param + 1.0)
    return value


def alpert_log_rule(f, n: int = 8) -> float:
    """
    Alpert 对数奇异积分规则。

    计算 ∫_0^1 log(x) f(x) dx。

    对于小 n，使用预先计算的节点和权重（基于广义 Gauss 规则）。

    Parameters
    ----------
    f : callable
        光滑部分 f(x)。
    n : int
        节点数（简化实现使用复合 Simpson + 端点修正）。

    Returns
    -------
    value : float
        积分值。
    """
    # 分段积分：在 [0, ε] 使用解析主值，在 [ε, 1] 使用标准求积
    eps = 1e-6
    # [eps, 1] 上的复合 Simpson
    m = max(n, 4)
    if m % 2 == 1:
        m += 1
    h = (1.0 - eps) / m
    x = np.linspace(eps, 1.0, m + 1)
    y = np.log(x) * f(x)
    # Simpson 规则
    integral = y[0] + y[-1]
    integral += 4.0 * np.sum(y[1:m:2])
    integral += 2.0 * np.sum(y[2:m-1:2])
    integral *= h / 3.0
    # [0, eps] 主值：∫_0^ε log(x) f(0) dx = ε (log(ε) - 1) f(0)
    pv = eps * (np.log(eps) - 1.0) * f(0.0)
    return integral + pv


def alpert_power_rule(f, alpha: float, n: int = 8) -> float:
    """
    Alpert 幂律奇异积分规则。

    计算 ∫_0^1 x^α f(x) dx，其中 α > -1 可能非整数。

    解析延拓：∫_0^ε x^α f(0) dx = ε^{α+1} / (α+1) * f(0)
    """
    if alpha <= -1.0:
        raise ValueError("alpha must be > -1")
    eps = 1e-6
    m = max(n, 4)
    if m % 2 == 1:
        m += 1
    h = (1.0 - eps) / m
    x = np.linspace(eps, 1.0, m + 1)
    y = np.power(x, alpha) * f(x)
    integral = y[0] + y[-1]
    integral += 4.0 * np.sum(y[1:m:2])
    integral += 2.0 * np.sum(y[2:m-1:2])
    integral *= h / 3.0
    pv = (eps ** (alpha + 1.0)) / (alpha + 1.0) * f(0.0)
    return integral + pv


def decay_constant_integral(meson_mass: float, pion_mass: float,
                            lattice_spacing: float = 1.0) -> float:
    """
    利用 Gegenbauer 积分计算格点上的 π 介子衰变常数 f_π。

    PCAC 关系：
        f_π m_π^2 = (m_u + m_d) ⟨0 | ū iγ_5 d | π⟩

    在格点上，衰变常数可通过矢量-轴矢 Ward 恒等式提取：

        f_π = Z_A / m_π * ⟨ 0 | A_4 | π ⟩

    这里用简化的一维动量积分模型：

        f_π ≈ C ∫_0^π dk / (2π)  (1 - cos(k)) / (m_π^2 + 2(1 - cos(k)))

    Parameters
    ----------
    meson_mass : float
        介子质量（格点单位）。
    pion_mass : float
        π 介子质量。
    lattice_spacing : float
        晶格间距 a。

    Returns
    -------
    f_pi : float
        衰变常数估计（格点单位）。
    """
    def integrand(k):
        return (1.0 - np.cos(k)) / (pion_mass ** 2 + 2.0 * (1.0 - np.cos(k)) + 1e-10)

    # 使用 Gegenbauer 积分（λ=0.5 对应 Legendre 权）
    # 先做变量替换 k = π x，将积分域变为 [-1, 1]
    def f_g(x):
        return integrand(0.5 * np.pi * (x + 1.0))

    try:
        val = gegenbauer_cc(32, 0.5, f_g)
    except Exception:
        # 回退到标准数值积分
        x = np.linspace(-1, 1, 1000)
        val = np.trapezoid(f_g(x), x)

    f_pi = 0.5 * val / np.pi
    # 无量纲化：f_pi * a，其中 a 为晶格间距
    return f_pi * lattice_spacing


def self_energy_integral(mass: float, cutoff: float = np.pi) -> float:
    """
    计算简化的单圈自能积分（用于夸克质量重整化）。

    Σ(m) = ∫ d^4k / (2π)^4  1 / (k^2 + m^2)

    对于小动量，使用 Alpert 对数规则处理红外奇异性。
    """
    def radial_integrand(k):
        # 4D 球坐标：体积元 ∝ k^3 dk
        return k ** 3 / (k ** 2 + mass ** 2 + 1e-10)

    # 分解为 [0, m] 和 [m, cutoff]
    # [0, m] 使用对数奇异规则（变量替换 k = m x）
    def f_low(x):
        x = np.atleast_1d(x)
        result = np.zeros_like(x, dtype=float)
        mask = x > 1e-15
        result[mask] = radial_integrand(mass * x[mask]) / (mass ** 3 + 1e-15)
        result[~mask] = 1.0
        return result if result.size > 1 else result.item()

    val_low = alpert_log_rule(f_low, n=16)
    val_low *= mass ** 4

    # [m, cutoff] 标准积分
    n_seg = 100
    k = np.linspace(mass, cutoff, n_seg)
    y = radial_integrand(k)
    val_high = np.trapezoid(y, k)

    # 归一化因子 1/(2π)^4 * 2π^2 (3D 球面积)
    total = (val_low + val_high) / (2.0 * np.pi ** 2)
    return total
