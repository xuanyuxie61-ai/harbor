"""
density_evolution.py
====================
核密度演化反应-扩散 PDE 求解模块

本模块实现核物质密度 ρ(r, t) 的时间演化，基于 FTCS (Forward Time Centered Space) 格式
求解广义 Fisher-KPP 型反应-扩散方程：

  ∂ρ/∂t = D ∇²ρ + f(ρ)

其中非线性源项 f(ρ) 模拟核子间的短程关联与配对效应：
  f(ρ) = ρ (1 - ρ/ρ₀) [1 + α (ρ/ρ₀ - 1/2)²] - β ρ³

该方程在核物理中的物理意义：
- 扩散项 D ∇²ρ：核子由于剩余相互作用在核内的迁移
- 反应项 ρ(1-ρ/ρ₀)：类似于逻辑斯蒂增长，ρ₀ 为饱和密度 (~0.16 fm⁻³)
- 高阶修正：模拟核物质的不可压缩性

数值方法：
- 空间：中心差分 (二阶精度)
- 时间：前向欧拉 (FTCS)，要求 D Δt / Δx² ≤ 0.5 以保证稳定性
- 边界条件：Neumann（零流）在 r=0 和 r=R_max
"""

import numpy as np
from math import sqrt, pi


# 饱和核密度
RHO0 = 0.16  # fm⁻³
# 扩散系数（核子平均自由程估计）
DIFFUSION_COEF = 2.5  # fm²/MeV (自然单位)


def reaction_source(rho, rho0=RHO0, alpha=0.5, beta=0.1):
    """
    反应源项：f(ρ) = ρ(1 - ρ/ρ₀)[1 + α(ρ/ρ₀ - 1/2)²] - β ρ³

    参数
    ----
    rho : float 或 ndarray
        局部密度
    rho0 : float
        饱和密度
    alpha, beta : float
        非线性修正系数

    返回
    ----
    float 或 ndarray
        源项值
    """
    x = rho / rho0
    logistic = rho * (1.0 - x)
    correction = 1.0 + alpha * (x - 0.5) ** 2
    cubic_damping = beta * rho ** 3
    return logistic * correction - cubic_damping


def reaction_source_derivative(rho, rho0=RHO0, alpha=0.5, beta=0.1):
    """
    反应源项对 ρ 的导数，用于线性稳定性分析。

    df/dρ = (1 - 2ρ/ρ₀)[1 + α(ρ/ρ₀ - 1/2)²]
            + ρ(1 - ρ/ρ₀) · 2α(ρ/ρ₀ - 1/2)/ρ₀
            - 3β ρ²
    """
    x = rho / rho0
    term1 = (1.0 - 2.0 * x) * (1.0 + alpha * (x - 0.5) ** 2)
    term2 = rho * (1.0 - x) * 2.0 * alpha * (x - 0.5) / rho0
    term3 = 3.0 * beta * rho ** 2
    return term1 + term2 - term3


def ftcs_density_evolution_1d(r_grid, rho_initial, D, t_max, nt,
                               rho0=RHO0, alpha=0.5, beta=0.1,
                               left_bc='neumann', right_bc='neumann'):
    """
    一维球对称 FTCS 求解核密度演化。

    球坐标下拉普拉斯算子（仅径向依赖）：
    ∇²ρ = (1/r²) ∂/∂r (r² ∂ρ/∂r)
        = ∂²ρ/∂r² + (2/r) ∂ρ/∂r

    参数
    ----
    r_grid : ndarray
        径向格点 (fm)
    rho_initial : ndarray
        初始密度分布
    D : float
        扩散系数
    t_max : float
        总演化时间
    nt : int
        时间步数
    left_bc, right_bc : str
        边界条件类型

    返回
    ----
    rho_final : ndarray
        最终密度分布
    rho_history : ndarray
        密度演化历史（每隔一定步数保存）
    stability_parameter : float
        D Δt / Δr²
    """
    N = len(r_grid)
    dr = r_grid[1] - r_grid[0]
    dt = t_max / nt

    # 稳定性判据
    s = D * dt / (dr ** 2)
    if s > 0.5:
        # 自动调整时间步数以满足稳定性
        nt = int(np.ceil(t_max * D / (0.45 * dr ** 2)))
        dt = t_max / nt
        s = D * dt / (dr ** 2)

    rho = rho_initial.copy()
    save_interval = max(1, nt // 20)
    history = [rho.copy()]

    for step in range(nt):
        rho_new = rho.copy()

        for i in range(1, N - 1):
            r = r_grid[i]
            # 二阶径向导数
            d2rho = (rho[i + 1] - 2.0 * rho[i] + rho[i - 1]) / (dr ** 2)

            # 一阶径向导数（带 2/r 因子）
            if r > 1e-6:
                drho = (rho[i + 1] - rho[i - 1]) / (2.0 * dr)
                laplacian = d2rho + (2.0 / r) * drho
            else:
                # r=0 处的极限：∇²ρ = 3 ∂²ρ/∂r²
                laplacian = 3.0 * d2rho

            source = reaction_source(rho[i], rho0, alpha, beta)
            rho_new[i] = rho[i] + dt * (D * laplacian + source)

            # 密度非负约束
            if rho_new[i] < 0:
                rho_new[i] = 0.0
            # 密度不超过某个上限（泡利阻塞）
            if rho_new[i] > 3.0 * rho0:
                rho_new[i] = 3.0 * rho0

        # 边界条件
        if left_bc == 'neumann':
            rho_new[0] = rho_new[1]
        elif left_bc == 'dirichlet':
            rho_new[0] = rho0

        if right_bc == 'neumann':
            rho_new[N - 1] = rho_new[N - 2]
        elif right_bc == 'dirichlet':
            rho_new[N - 1] = 0.0

        rho = rho_new
        if (step + 1) % save_interval == 0:
            history.append(rho.copy())

    return rho, np.array(history), s


def total_nucleon_number(r_grid, rho):
    """
    计算球对称密度分布对应的总核子数。

    A = 4π ∫₀^∞ ρ(r) r² dr
    """
    integrand = 4.0 * pi * rho * r_grid ** 2
    return np.trapezoid(integrand, r_grid)


def rms_radius(r_grid, rho):
    """
    计算密度分布的均方根半径。

    ⟨r²⟩ = ∫ ρ(r) r⁴ dr / ∫ ρ(r) r² dr
    R_rms = √⟨r²⟩
    """
    num = np.trapezoid(rho * r_grid ** 4, r_grid)
    den = np.trapezoid(rho * r_grid ** 2, r_grid)
    if den < 1e-15:
        return 0.0
    return sqrt(num / den)


def surface_thickness(r_grid, rho, rho0=RHO0):
    """
    计算核表面厚度（密度从 0.9ρ₀ 降到 0.1ρ₀ 的距离）。
    """
    # 找到密度下降到 0.9 rho0 和 0.1 rho0 的半径
    r90 = None
    r10 = None
    for i in range(len(r_grid) - 1, 0, -1):
        if r90 is None and rho[i] < 0.9 * rho0:
            r90 = r_grid[i]
        if r10 is None and rho[i] < 0.1 * rho0:
            r10 = r_grid[i]
            break
    if r90 is not None and r10 is not None:
        return r10 - r90
    return 0.0


def density_moment(r_grid, rho, n):
    """
    计算密度分布的第 n 阶径向矩。

    M_n = 4π ∫ ρ(r) r^{n+2} dr
    """
    integrand = 4.0 * pi * rho * r_grid ** (n + 2)
    return np.trapezoid(integrand, r_grid)
