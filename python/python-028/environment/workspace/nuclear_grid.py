"""
nuclear_grid.py
===============
原子核三维网格生成与最优采样模块

本模块实现：
1. 环形/壳层网格生成（基于 annulus_grid 思想）—— 用于变形核的柱对称坐标网格
2. CVT (Centroidal Voronoi Tessellation) 三维采样 —— 用于核体积内最优求积点的分布
3. 形变核表面网格生成 —— 基于 β, γ 形变参数的核表面参数化

数学模型：
- 核表面参数化（Bohr 形变）：
  R(θ, φ) = R₀ [1 + β₂ cos γ Y₂₀ + β₂ sin γ (Y₂₂ + Y₂₋₂)/√2 + ...]
- CVT 能量泛函：E = Σᵢ ∫_{Vᵢ} ρ(x) ||x - zᵢ||² dx
  其中 Vᵢ 为 Voronoi 区域，zᵢ 为生成元。
"""

import numpy as np
from math import sin, cos, sqrt, pi


def deformed_nuclear_surface_grid(beta2, gamma, R0, n_theta, n_phi):
    """
    生成变形核表面网格点。

    核表面由 Bohr 形变参数化：
    R(θ, φ) = R₀ [1 + β₂ cosγ · Y₂₀(θ, φ)
                  + β₂ sinγ · (Y₂₂(θ, φ) + Y₂₋₂(θ, φ))/√2]

    参数
    ----
    beta2 : float
        四极形变参数
    gamma : float
        四极形变角 (弧度)
    R0 : float
        平均核半径 (fm)
    n_theta : int
        极角方向网格数
    n_phi : int
        方位角方向网格数

    返回
    ----
    grid : ndarray, shape (n_points, 3)
        表面网格点的笛卡尔坐标 (x, y, z)
    areas : ndarray
        每个网格面对应的微元面积
    """
    theta = np.linspace(0, pi, n_theta)
    phi = np.linspace(0, 2 * pi, n_phi)
    dtheta = pi / (n_theta - 1) if n_theta > 1 else pi
    dphi = 2 * pi / (n_phi - 1) if n_phi > 1 else 2 * pi

    grid = []
    areas = []

    for i in range(n_theta):
        for j in range(n_phi):
            th = theta[i]
            ph = phi[j]
            # Y_20 = √(5/16π) (3cos²θ - 1)
            Y20 = sqrt(5.0 / (16.0 * pi)) * (3.0 * cos(th) ** 2 - 1.0)
            # Y_22 + Y_2-2 的实部 ∝ sin²θ cos(2φ)
            Y22_real = sqrt(15.0 / (32.0 * pi)) * sin(th) ** 2 * cos(2.0 * ph)

            R = R0 * (1.0 + beta2 * (cos(gamma) * Y20 + sin(gamma) * Y22_real))
            x = R * sin(th) * cos(ph)
            y = R * sin(th) * sin(ph)
            z = R * cos(th)
            grid.append([x, y, z])

            # 微元面积 dA ≈ R² sinθ dθ dφ (一阶近似)
            dA = max(R, 0.0) ** 2 * sin(th) * dtheta * dphi
            areas.append(dA)

    return np.array(grid), np.array(areas)


def annular_shell_grid(n_r, n_theta, r_inner, r_outer, center=(0.0, 0.0)):
    """
    在二维环形区域 [r_inner, r_outer] × [0, 2π) 内生成均匀网格。

    该网格用于柱对称变形核的截面计算，对应 annulus_grid 的升级。
    每个网格点带有径向权重 w = r，用于柱坐标体积元 r dr dθ。

    参数
    ----
    n_r : int
        径向网格数
    n_theta : int
        角向网格数
    r_inner, r_outer : float
        内外半径 (fm)
    center : tuple
        中心坐标 (x0, y0)

    返回
    ----
    points : ndarray, shape (n_points, 2)
        网格点坐标
    weights : ndarray
        每个点的体积权重
    """
    if r_inner < 0 or r_outer <= r_inner:
        raise ValueError("必须满足 0 ≤ r_inner < r_outer")

    dr = (r_outer - r_inner) / n_r
    dtheta = 2 * pi / n_theta
    points = []
    weights = []

    for i in range(n_r):
        r = r_inner + (i + 0.5) * dr
        for j in range(n_theta):
            theta = (j + 0.5) * dtheta
            x = center[0] + r * cos(theta)
            y = center[1] + r * sin(theta)
            points.append([x, y])
            weights.append(r * dr * dtheta)

    return np.array(points), np.array(weights)


def cvt_3d_sample(n_generators, n_iterations, n_samples, density_fn=None,
                  bounds=((-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)), seed=42):
    """
    在三维有界区域内执行 CVT (Centroidal Voronoi Tessellation) 采样。

    算法：Lloyd 迭代
    1. 初始化生成元 g_i（随机或根据密度）
    2. 在区域内按密度 ρ(x) 采样大量点 s_k
    3. 对每个 s_k 找到最近生成元 g_{i(k)}
    4. 更新 g_i = Σ_{k∈V_i} ρ(s_k) s_k / Σ_{k∈V_i} ρ(s_k)
    5. 重复直到收敛

    参数
    ----
    n_generators : int
        生成元数量（即求积点数）
    n_iterations : int
        Lloyd 迭代次数
    n_samples : int
        每次迭代采样点数
    density_fn : callable
        密度函数 ρ(x, y, z)，默认为均匀密度
    bounds : tuple of tuple
        各维边界
    seed : int
        随机种子

    返回
    ----
    generators : ndarray, shape (n_generators, 3)
        最终生成元位置（最优求积点）
    energy_history : list
        每次迭代的 CVT 能量
    """
    rng = np.random.default_rng(seed)

    # 初始化生成元
    generators = rng.random((n_generators, 3))
    for d in range(3):
        lo, hi = bounds[d]
        generators[:, d] = lo + generators[:, d] * (hi - lo)

    if density_fn is None:
        def density_fn(x, y, z):
            return 1.0

    energy_history = []

    for it in range(n_iterations):
        # 在边界内采样
        samples = rng.random((n_samples, 3))
        for d in range(3):
            lo, hi = bounds[d]
            samples[:, d] = lo + samples[:, d] * (hi - lo)

        # 计算每个采样点的密度
        rho = np.array([density_fn(s[0], s[1], s[2]) for s in samples])

        # 为每个采样点找到最近生成元（暴力搜索，保证鲁棒性）
        nearest = np.zeros(n_samples, dtype=int)
        for k in range(n_samples):
            dists = np.sum((generators - samples[k]) ** 2, axis=1)
            nearest[k] = np.argmin(dists)

        # 计算 CVT 能量
        energy = 0.0
        for k in range(n_samples):
            energy += rho[k] * np.sum((samples[k] - generators[nearest[k]]) ** 2)
        energy /= n_samples
        energy_history.append(energy)

        # 更新生成元为 Voronoi 区域的质心
        new_generators = np.zeros_like(generators)
        counts = np.zeros(n_generators)
        for k in range(n_samples):
            i = nearest[k]
            new_generators[i] += rho[k] * samples[k]
            counts[i] += rho[k]

        # 处理空区域：保持原生成元
        for i in range(n_generators):
            if counts[i] > 0:
                new_generators[i] /= counts[i]
            else:
                new_generators[i] = generators[i]

        generators = new_generators

    return generators, energy_history


def nuclear_volume_cvt_quadrature(A, beta2=0.0, gamma=0.0, n_points=200,
                                   n_iter=30, R0=1.2):
    """
    利用 CVT 在原子核体积内生成最优求积点。

    密度分布采用 Fermi 分布：
    ρ(r) = ρ₀ / [1 + exp((r - R) / a)]
    其中 R = R₀ A^{1/3}, a ≈ 0.52 fm (弥散参数)

    参数
    ----
    A : int
        质量数
    beta2, gamma : float
        形变参数
    n_points : int
        求积点数
    n_iter : int
        CVT 迭代次数
    R0 : float
        核半径参数

    返回
    ----
    points : ndarray
        求积点坐标 (x, y, z)
    weights : ndarray
        求积权重（通过 Voronoi 体积近似）
    """
    R = R0 * (A ** (1.0 / 3.0))
    a_diff = 0.52  # 弥散参数 fm

    def fermi_density(x, y, z):
        # 简化为球形 Fermi 分布（CVT 在球内近似）
        r = sqrt(x * x + y * y + z * z)
        return 1.0 / (1.0 + np.exp((r - R) / a_diff))

    bounds = ((-1.5 * R, 1.5 * R),) * 3
    generators, _ = cvt_3d_sample(
        n_generators=n_points,
        n_iterations=n_iter,
        n_samples=50000,
        density_fn=fermi_density,
        bounds=bounds,
        seed=42
    )

    # 计算近似权重：每个点对应 Voronoi 区域的体积
    # 采用蒙特卡洛估计
    rng = np.random.default_rng(123)
    n_mc = 200000
    mc_samples = rng.random((n_mc, 3))
    for d in range(3):
        lo, hi = bounds[d]
        mc_samples[:, d] = lo + mc_samples[:, d] * (hi - lo)

    weights = np.zeros(n_points)
    total_vol = 0.0
    for k in range(n_mc):
        x, y, z = mc_samples[k]
        rho = fermi_density(x, y, z)
        if rho < 1e-6:
            continue
        dists = np.sum((generators - mc_samples[k]) ** 2, axis=1)
        i = np.argmin(dists)
        weights[i] += rho
        total_vol += rho

    if total_vol > 0:
        weights *= ((3.0 * R) ** 3) / n_mc  # 缩放至实际体积

    return generators, weights
