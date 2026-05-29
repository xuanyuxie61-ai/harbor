"""
matrix_elements.py
==================
核矩阵元计算与 Cauchy 主值积分模块

本模块实现：
1. 核子-核子相互作用矩阵元的数值积分
2. 自能计算中的 Cauchy 主值积分
3. 电磁多极算符矩阵元

数学基础：
1. Cauchy 主值积分：
   P ∫_a^b f(x)/(x - x₀) dx = lim_{ε→0} [∫_a^{x₀-ε} + ∫_{x₀+ε}^b] f(x)/(x-x₀) dx

   数值计算采用 Gauss-Legendre 求积，利用对称性消去奇异性：
   P ∫_{-1}^1 g(s)/s ds ≈ Σ w_i [g(s_i) - g(0)] / s_i
   当 N 为偶数时，Σ w_i/s_i = 0，因此 g(0) 项自动消失。

2. 多极算符矩阵元：
   ⟨n'l'j'||Q_λ||nlj⟩ = e_λ ∫ u_{n'l'}(r) r^λ u_{nl}(r) dr
   其中 u(r) = r R(r) 为约化径向波函数。

3. 自能（Self-Energy）：
   Σ(E) = P ∫_{E_min}^{E_max} |V_{nk}|² / (E - E_k) dE_k
   其中 V_{nk} 为耦合矩阵元，E_k 为中间态能量。
"""

import numpy as np
from math import sqrt, pi, cos, sin


def gauss_legendre_nodes_weights(n):
    """
    计算 n 点 Gauss-Legendre 求积的节点与权重。

    对于积分 ∫_{-1}^1 f(x) dx ≈ Σ_{i=1}^n w_i f(x_i)
    """
    if n <= 0:
        raise ValueError("n 必须为正整数")
    nodes, weights = np.polynomial.legendre.leggauss(n)
    return nodes, weights


def cauchy_principal_value(f, a, b, x0, n=64):
    """
    计算 Cauchy 主值积分 P ∫_a^b f(x) / (x - x₀) dx。

    算法（基于 cauchy_principal_value 的 Python 实现）：
    1. 将积分区间分解为 [a, x₀-δ] ∪ [x₀-δ, x₀+δ] ∪ [x₀+δ, b]
    2. 两侧正则积分用标准 Gauss-Legendre
    3. 中心奇异积分通过对称求积：
       P ∫_{x₀-δ}^{x₀+δ} f(x)/(x-x₀) dx
       = ∫_{-1}^1 [f(x₀ + δ s) - f(x₀)] / s ds
       ≈ Σ w_i f(x₀ + δ s_i) / s_i   （N 为偶数时）

    参数
    ----
    f : callable
        被积函数（光滑部分）
    a, b : float
        积分区间
    x0 : float
        奇点位置，必须满足 a < x0 < b
    n : int
        Gauss-Legendre 点数（必须为偶数）

    返回
    ----
    cpv : float
        Cauchy 主值积分结果
    """
    if n % 2 != 0:
        n += 1

    if not (a < x0 < b):
        nodes, weights = gauss_legendre_nodes_weights(n)
        x_mapped = 0.5 * ((b - a) * nodes + a + b)
        w_mapped = 0.5 * (b - a) * weights
        return np.sum(w_mapped * f(x_mapped) / (x_mapped - x0))

    delta = min(x0 - a, b - x0) * 0.5
    delta = max(delta, 1e-10)

    nodes, weights = gauss_legendre_nodes_weights(n)

    if x0 - delta > a:
        x_left = 0.5 * ((x0 - delta - a) * nodes + x0 - delta + a)
        w_left = 0.5 * (x0 - delta - a) * weights
        I_left = np.sum(w_left * f(x_left) / (x_left - x0))
    else:
        I_left = 0.0

    if b > x0 + delta:
        x_right = 0.5 * ((b - x0 - delta) * nodes + b + x0 + delta)
        w_right = 0.5 * (b - x0 - delta) * weights
        I_right = np.sum(w_right * f(x_right) / (x_right - x0))
    else:
        I_right = 0.0

    I_center = 0.0
    for i in range(n):
        s = nodes[i]
        if abs(s) < 1e-15:
            continue
        x_s = x0 + delta * s
        I_center += weights[i] * f(x_s) / s

    return I_left + I_center + I_right


def self_energy_integral(coupling_squared, energy_levels, E, n_quad=64):
    """
    计算单粒子自能 Σ(E)。

    Σ(E) = Σ_k |V_{nk}|² · P ∫ dE' ρ(E') / (E - E')

    这里将离散能级近似为 δ 函数加 Lorentz 展宽：
    ρ(E') ≈ (1/π) Σ_k Γ_k / [(E' - E_k)² + Γ_k²]

    参数
    ----
    coupling_squared : ndarray
        |V_{nk}|² 数组
    energy_levels : ndarray
        中间态能量 E_k (MeV)
    E : float
        入射能量 (MeV)
    n_quad : int
        积分点数

    返回
    ----
    sigma : float
        自能 (MeV)
    """
    sigma = 0.0
    gamma_width = 0.5

    for k in range(len(energy_levels)):
        E_k = energy_levels[k]
        V2 = coupling_squared[k]

        def integrand(Ep):
            lorentz = (1.0 / pi) * gamma_width / ((Ep - E_k) ** 2 + gamma_width ** 2)
            return V2 * lorentz

        E_min = E_k - 10.0 * gamma_width
        E_max = E_k + 10.0 * gamma_width

        try:
            contrib = cauchy_principal_value(integrand, E_min, E_max, E, n_quad)
        except Exception:
            nodes, weights = gauss_legendre_nodes_weights(n_quad)
            x_mapped = 0.5 * ((E_max - E_min) * nodes + E_max + E_min)
            w_mapped = 0.5 * (E_max - E_min) * weights
            lorentz = (1.0 / pi) * gamma_width / ((x_mapped - E_k) ** 2 + gamma_width ** 2)
            contrib = np.sum(w_mapped * V2 * lorentz / (E - x_mapped))

        sigma += contrib

    return sigma


def electric_multipole_matrix_element(r_grid, u_i, u_f, lambda_order):
    """
    计算电多极跃迁矩阵元。

    ⟨f||Q_λ||i⟩ = e_eff ∫_0^∞ u_f(r) r^λ u_i(r) dr

    参数
    ----
    r_grid : ndarray
        径向格点
    u_i, u_f : ndarray
        初态与末态约化波函数 u(r) = r R(r)
    lambda_order : int
        多极阶数 λ

    返回
    ----
    me : float
        约化矩阵元 (e·fm^λ)
    """
    integrand = u_f * (r_grid ** lambda_order) * u_i
    return np.trapezoid(integrand, r_grid)


def transition_probability(lambda_order, me, E_gamma, mass_number, Ji):
    """
    计算约化跃迁几率 B(λ) 与半寿命估计。

    B(λ) = |⟨f||Q_λ||i⟩|² / (2J_i + 1)

    Weisskopf 单位：
    B_W(λ) = (1/4π) [3/(λ+3)]² (1.2 A^{1/3})^{2λ}

    参数
    ----
    lambda_order : int
        多极阶数
    me : float
        约化矩阵元
    E_gamma : float
        γ 射线能量 (MeV)
    mass_number : int
        质量数 A
    Ji : float
        初态总角动量

    返回
    ----
    B_lambda : float
        约化跃迁几率
    B_W : float
        Weisskopf 单位
    tau_half : float
        估算半寿命 (秒)
    """
    B_lambda = me ** 2 / (2.0 * Ji + 1.0)

    R = 1.2 * (mass_number ** (1.0 / 3.0))
    B_W = (1.0 / (4.0 * pi)) * (3.0 / (lambda_order + 3.0)) ** 2 * R ** (2 * lambda_order)

    if E_gamma > 1e-6:
        tau_half = 1e-16 / (E_gamma ** (2 * lambda_order + 1) * B_lambda)
    else:
        tau_half = 1e10

    return B_lambda, B_W, tau_half


def overlap_integral(r_grid, u1, u2):
    """
    计算两个径向波函数的重叠积分 ⟨1|2⟩ = ∫ u_1(r) u_2(r) dr。
    """
    return np.trapezoid(u1 * u2, r_grid)


def spectroscopic_factor(r_grid, u_orbital, u_residual, A_core, n, l, j):
    """
    计算谱学因子 S。

    S = |⟨A+1 | a^†_{nlj} | A⟩|²
      ≈ |∫ u_{res}(r) u_{orb}(r) dr|² · (2j + 1)

    参数
    ----
    r_grid : ndarray
    u_orbital : ndarray
        轨道波函数（剥离/拾取核子）
    u_residual : ndarray
        剩余核波函数
    A_core : int
        核心核子数
    n, l : int
        主量子数、轨道角动量
    j : float
        总角动量

    返回
    ----
    S : float
        谱学因子
    """
    overlap = overlap_integral(r_grid, u_orbital, u_residual)
    return overlap ** 2 * (2.0 * j + 1.0)
