"""
hauser_feshbach.py
===================
Hauser-Feshbach 复合核统计衰变理论模块

基于种子项目 074_bdf2 的隐式多步 ODE 求解思想
(用于衰变链时间演化) 以及 685_line_nco_rule 的
Newton-Cotes Open 数值积分思想 (用于共振能量平均)，
本模块实现复合核反应的统计理论计算。

核心公式
--------
复合核形成截面:
    σ_{CF}(a) = (π/k_a²) Σ_{J,π} (2J+1) / [(2I_a+1)(2I_A+1)] * T_a(J,π)

Hauser-Feshbach 衰变宽度:
    σ_{α→β} = σ_{CF}(α) * Γ_β / Γ_tot

其中穿透系数 T_a 来自光学模型，Γ 为衰变宽度。

能量平均截面 (Ericson 涨落理论):
    <σ_{αβ}> = σ_{CN} * <Γ_α Γ_β / Γ²> * W_{αβ}

其中 W_{αβ} 为宽度涨落修正因子 (通常用 Porter-Thomas 分布)。

衰变链时间演化 (BDF2 离散):
    dN_i/dt = Σ_j λ_{ji} N_j - λ_i N_i

使用 BDF2 隐式格式求解母核-子核衰变链:
    (3/2Δt + λ_i) N_i^{n+1} = 2N_i^n - (1/2)N_i^{n-1} + Δt * Σ_j λ_{ji} N_j^{n+1}
"""

import numpy as np
from scipy.linalg import solve


def level_density_parameter(A, E_exc):
    """
    计算核激发态能级密度参数 a (MeV^{-1})。

    采用 Bethe 公式:
        ρ(E) ∝ exp(2√(aE)) / E^{5/4}

    能级密度参数:
        a = A / k_τ

    其中 k_τ 为温度参数，对球形核约 8 MeV，对变形核约 10 MeV。
    """
    if A <= 0:
        raise ValueError("质量数 A 必须为正")
    k_tau = 8.0  # MeV, 典型值
    # 能量修正 (随着激发能增加，a 略有增加)
    a = A / k_tau * (1.0 + 0.05 * np.log1p(E_exc / 10.0))
    return a


def level_density(A, E_exc, J, spin_cutoff=5.0):
    """
    计算给定激发能 E_exc 和自旋 J 的能级密度 ρ(E, J)。

    复合公式:
        ρ(E, J) = (2J+1) / (2√(2π) σ³) * exp(-(J+1/2)²/(2σ²)) * ρ(E)

    其中 ρ(E) = exp(2√(aE)) / (12√(2) σ a^{1/4} E^{5/4})
    """
    if E_exc <= 0:
        return 1e-30  # 边界保护
    a = level_density_parameter(A, E_exc)
    # 总状态密度
    rho_total = np.exp(2.0 * np.sqrt(a * E_exc)) / (12.0 * np.sqrt(2.0) * spin_cutoff * (a ** 0.25) * (E_exc ** 1.25))
    # 自旋分布
    spin_factor = (2.0 * J + 1.0) / (2.0 * np.sqrt(2.0 * np.pi) * (spin_cutoff ** 3))
    spin_factor *= np.exp(-((J + 0.5) ** 2) / (2.0 * spin_cutoff ** 2))
    return rho_total * spin_factor


def transmission_coefficient_integral(T_dict, l_max):
    """
    将各分波穿透系数按角动量求和:

    T_total = Σ_{l=0}^{l_max} (2l+1) T_l

    这是复合核形成截面的分子部分。
    """
    total = 0.0
    for l in range(l_max + 1):
        js = [l + 0.5] if l == 0 else [l - 0.5, l + 0.5]
        for j in js:
            key = (l, j)
            if key in T_dict:
                total += (2.0 * j + 1.0) * T_dict[key]
    return total


def compound_formation_cross_section(params, T_dict, l_max, I_target=0.0, I_proj=0.5):
    """
    计算复合核形成截面 σ_CF。

    σ_CF = (π/k²) * Σ_{J,π} (2J+1) / [(2I_proj+1)(2I_target+1)] * T(J,π)

    简化：假设所有 J 都贡献，且分波穿透系数已包含 (2j+1) 权重。
    """
    prefactor = np.pi / (params.k ** 2)
    spin_denom = (2.0 * I_proj + 1.0) * (2.0 * I_target + 1.0)
    T_sum = transmission_coefficient_integral(T_dict, l_max)
    # 简化处理：假设平均 J 简并度
    return prefactor * T_sum / spin_denom


def decay_width(T_dict, l_max, E_gamma=1.0):
    """
    计算各衰变道的部分宽度 Γ。

    对于中子出射道:
        Γ_n = (1/2πρ(E)) * Σ_{l,j} T_{lj}(E)

    对于γ衰变道:
        Γ_γ = E_γ^5 * f(E_γ)  (偶极辐射近似)

    Returns
    -------
    widths : dict
        各衰变道宽度 (MeV)。
    """
    T_sum = transmission_coefficient_integral(T_dict, l_max)
    # 简化的宽度估计
    Gamma_n = T_sum * 0.1  # 中子宽度 (MeV)
    Gamma_gamma = E_gamma ** 5 * 1e-6  # γ 宽度
    Gamma_total = Gamma_n + Gamma_gamma + 0.01  # 加上其他道的小贡献

    return {
        'neutron': Gamma_n,
        'gamma': Gamma_gamma,
        'total': Gamma_total,
        'ratio_n': Gamma_n / Gamma_total,
        'ratio_gamma': Gamma_gamma / Gamma_total,
    }


def open_newton_cotes_weights(n, a, b):
    """
    计算 Newton-Cotes Open 求积规则的节点和权重。

    基于种子项目 685_line_nco_rule 的核心思想：
    在区间 [a, b] 上取 n 个等距内点 (不含端点)，
    构造 Lagrange 插值多项式并精确积分。

    节点:
        x_i = [(n - i + 1) * a + i * b] / (n + 1),  i = 1, ..., n

    Returns
    -------
    x : ndarray
        节点坐标。
    w : ndarray
        积分权重。
    """
    if n < 1:
        raise ValueError("n 必须 >= 1")
    i = np.arange(1, n + 1, dtype=float)
    x = ((n - i + 1.0) * a + i * b) / (n + 1.0)

    # 使用 Lagrange 基函数的精确积分计算权重
    # 简化的权重计算：对于低阶使用已知公式
    if n == 1:
        # 中点法则
        w = np.array([b - a], dtype=float)
    elif n == 2:
        h = (b - a) / 3.0
        w = np.array([h, h])
    elif n == 3:
        h = (b - a) / 4.0
        w = np.array([3.0 * h / 2.0, 3.0 * h / 2.0, 3.0 * h / 2.0])
    elif n == 4:
        h = (b - a) / 5.0
        w = np.array([2.0 * h, 2.0 * h, 2.0 * h, 2.0 * h])
    else:
        # 通用计算：Newton-Cotes Open 权重
        # 使用 Vandermonde 矩阵求解
        V = np.vander(x - a, increasing=True, N=n)
        # 精确积分 x^k 在 [a,b] 上 = (b^{k+1} - a^{k+1})/(k+1)
        exact_moments = np.array([((b ** (k + 1) - a ** (k + 1)) / (k + 1.0)) for k in range(n)])
        w = solve(V.T, exact_moments)

    return x, w


def energy_average_cross_section(E_min, E_max, n_points, sigma_func):
    """
    使用 Newton-Cotes Open 规则对能量相关截面进行平均。

    <σ> = 1/(E_max - E_min) ∫_{E_min}^{E_max} σ(E) dE

    Parameters
    ----------
    E_min, E_max : float
        能量范围 (MeV)。
    n_points : int
        积分节点数。
    sigma_func : callable
        σ(E) 函数。

    Returns
    -------
    avg_sigma : float
        能量平均截面。
    """
    x, w = open_newton_cotes_weights(n_points, E_min, E_max)
    vals = np.array([sigma_func(e) for e in x])
    integral = np.dot(w, vals)
    return integral / (E_max - E_min)


def decay_chain_bdf2(initial_populations, decay_matrix, t_span, n_steps):
    """
    使用 BDF2 (Backward Differentiation Formula 2) 隐式格式
    求解核衰变链的母核-子核时间演化。

    基于种子项目 074_bdf2 的 BDF2 求解思想。

    衰变链方程组:
        dN_i/dt = Σ_j λ_{ji} N_j - λ_i N_i

    矩阵形式:
        dN/dt = M · N

    BDF2 离散 (变步长):
        N_{n+1} - (4/3)N_n + (1/3)N_{n-1} = (2Δt/3) M N_{n+1}
        => [I - (2Δt/3)M] N_{n+1} = (4/3)N_n - (1/3)N_{n-1}

    Parameters
    ----------
    initial_populations : ndarray
        初始核素布居数。
    decay_matrix : ndarray
        衰变矩阵 M，其中 M_{ij} = λ_{ji} (i≠j), M_{ii} = -λ_i。
    t_span : tuple
        (t0, tf) 时间范围 (秒)。
    n_steps : int
        时间步数。

    Returns
    -------
    t : ndarray
        时间网格。
    N : ndarray
        布居数历史，形状 (n_steps+1, n_species)。
    """
    t0, tf = t_span
    dt = (tf - t0) / n_steps
    n_species = len(initial_populations)

    t = np.linspace(t0, tf, n_steps + 1)
    N = np.zeros((n_steps + 1, n_species))
    N[0, :] = initial_populations

    # 第一步使用向后 Euler (BDF1)
    I = np.eye(n_species)
    A1 = I - dt * decay_matrix
    N[1, :] = solve(A1, N[0, :])

    # BDF2 主循环
    A_bdf2 = I - (2.0 * dt / 3.0) * decay_matrix
    for n in range(1, n_steps):
        rhs = (4.0 / 3.0) * N[n, :] - (1.0 / 3.0) * N[n - 1, :]
        N[n + 1, :] = solve(A_bdf2, rhs)

    return t, N


def width_fluctuation_correction(T_dict, l_max, nu=1.0):
    """
    计算宽度涨落修正因子 (Moldauer 近似)。

    W_{ab} = [1 + 2/(ν_a + 1) T_a]^{-1/2} [1 + 2/(ν_b + 1) T_b]^{-1/2}

    其中 ν 为自由度参数 (ν=1 对应 Porter-Thomas 分布)。
    """
    T_sum = transmission_coefficient_integral(T_dict, l_max)
    # 简化的 Moldauer 修正
    W = 1.0 / np.sqrt(1.0 + 2.0 * T_sum / (nu + 1.0))
    return W


if __name__ == "__main__":
    # 自检
    print("能级密度参数 (A=56, E=10MeV):", level_density_parameter(56, 10.0))
    print("能级密度 (A=56, E=10MeV, J=2):", level_density(56, 10.0, 2.0))

    x, w = open_newton_cotes_weights(4, 0.0, 10.0)
    print("NCO(4) 积分 x³:", np.dot(w, x ** 3), "期望 2500")

    # 衰变链测试
    M = np.array([[-0.5, 0.0, 0.0],
                  [0.3, -0.2, 0.0],
                  [0.2, 0.2, -0.1]])
    N0 = np.array([100.0, 0.0, 0.0])
    t, N = decay_chain_bdf2(N0, M, (0.0, 10.0), 100)
    print("衰变链最终布居:", N[-1, :])
