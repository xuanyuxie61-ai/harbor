"""
cvt_composition_mesh.py
=======================
基于 Lloyd 算法的非均匀质心 Voronoi 剖分 (CVT),
用于在成分空间生成最优计算网格。

算法 (种子项目 259_cvt_square_nonuniform):
  1. 初始化 N 个生成点 (generators) p_i ∈ [0, 1]
  2. 重复 Lloyd 迭代:
     a. 在 [0,1] 上生成 M 个采样点, 按密度函数 rho(x) 加权
     b. 将每个采样点分配给最近的生成点
     c. 将每个生成点移动到其 Voronoi 单元的质心
  3. 收敛后, 生成点构成最优非均匀网格

密度函数 rho(x_C):
  rho(x_C) ∝ |d²G/dc²| + epsilon
  即在 Gibbs 能曲率大的区域 (spinodal 附近) 加密网格

这在 CALPHAD 相图计算中用于:
  - 在混溶隙边界附近集中计算资源
  - 提高相边界定位精度
  - 自适应网格细化
"""

import numpy as np
from calphad_fec_constants import N_X_GRID, X_C_MIN, X_C_MAX


def density_function(x, T, phase):
    """
    密度函数: 在 Gibbs 能曲率大的区域分配更多网格点。

    rho(x) = |d²G/dx²| / (max|d²G/dx²| + eps)

    Parameters
    ----------
    x : np.ndarray
        碳摩尔分数
    T : float
        温度 (K)
    phase : str
        相名称

    Returns
    -------
    np.ndarray
        密度值 (归一化)
    """
    from gibbs_energy_calphad import second_derivative_G
    d2G = second_derivative_G(x, T, phase)
    rho = np.abs(d2G)
    rho_max = np.max(rho)
    if rho_max > 0:
        rho = rho / rho_max
    rho += 0.01  # 避免零密度区域
    return rho


def lloyd_cvt_1d(T, phase, n_generators=50, n_samples=2000,
                 max_iter=100, tol=1.0e-6):
    """
    1D Lloyd 算法计算 CVT 网格。

    Parameters
    ----------
    T : float
        温度 (K)
    phase : str
        相名称
    n_generators : int
        生成点数目
    n_samples : int
        每轮 Monte Carlo 采样数
    max_iter : int
        Lloyd 迭代最大次数
    tol : float
        收敛容差 (生成点最大位移)

    Returns
    -------
    dict
        {
            'generators': np.ndarray,  # CVT 生成点 (排序后)
            'weights': np.ndarray,     # 每个 Voronoi 单元的权重
            'energy': float,           # CVT 能量 (量化误差)
            'n_iter': int,             # 迭代次数
        }
    """
    # 初始化: 均匀分布 + 微扰
    rng = np.random.RandomState(278)
    gen = np.linspace(X_C_MIN, X_C_MAX, n_generators + 2)[1:-1]
    gen += rng.randn(n_generators) * (X_C_MAX - X_C_MIN) / (10 * n_generators)
    gen = np.sort(np.clip(gen, X_C_MIN, X_C_MAX))

    for iteration in range(max_iter):
        gen_old = gen.copy()

        # 生成加权采样点
        # 使用拒绝采样按密度函数采样
        samples = np.linspace(X_C_MIN, X_C_MAX, n_samples)
        rho = density_function(samples, T, phase)
        rho /= np.sum(rho)

        # Voronoi 分配: 每个采样点到最近生成点的距离
        # 1D 情况简化为排序比较
        assignments = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            dists = np.abs(samples[i] - gen)
            assignments[i] = np.argmin(dists)

        # 质心更新
        new_gen = np.zeros(n_generators)
        weights = np.zeros(n_generators)
        for k in range(n_generators):
            mask = assignments == k
            w = rho[mask]
            s = samples[mask]
            total_w = np.sum(w)
            if total_w > 1e-30:
                new_gen[k] = np.sum(w * s) / total_w
                weights[k] = total_w
            else:
                new_gen[k] = gen[k]
                weights[k] = 0.0

        # 边界保护
        new_gen = np.sort(np.clip(new_gen, X_C_MIN, X_C_MAX))
        gen = new_gen

        # 收敛检查
        max_disp = np.max(np.abs(gen - gen_old))
        if max_disp < tol:
            break

    # 计算 CVT 能量 (量化误差)
    samples_full = np.linspace(X_C_MIN, X_C_MAX, 5000)
    rho_full = density_function(samples_full, T, phase)
    rho_full /= np.sum(rho_full)
    energy = 0.0
    for i in range(len(samples_full)):
        dist = np.min(np.abs(samples_full[i] - gen))
        energy += rho_full[i] * dist * dist * (samples_full[1] - samples_full[0])

    return {
        'generators': gen,
        'weights': weights,
        'energy': float(energy),
        'n_iter': iteration + 1,
    }


def adaptive_cvt_refinement(T, phase, initial_cvt, refinement_factor=2):
    """
    CVT 网格自适应细化。

    在 Voronoi 单元中曲率变化最大的区域插入新的生成点。

    Parameters
    ----------
    T : float
        温度 (K)
    phase : str
        相名称
    initial_cvt : dict
        初始 CVT 结果
    refinement_factor : int
        细化倍数

    Returns
    -------
    dict
        细化后的 CVT 结果
    """
    gen = initial_cvt['generators']
    n_old = len(gen)
    n_new = n_old * refinement_factor

    # 计算每个单元的曲率变化
    from gibbs_energy_calphad import second_derivative_G
    d2G_vals = second_derivative_G(gen, T, phase)
    curvature_grad = np.abs(np.gradient(d2G_vals, gen))

    # 按曲率梯度分配新生成点
    p = curvature_grad / np.sum(curvature_grad + 1e-30)
    n_per_cell = np.maximum(1, np.round(p * n_new).astype(int))

    new_gen = []
    for i in range(n_old):
        # 在当前单元内均匀插入
        if i == 0:
            left = X_C_MIN
        else:
            left = (gen[i - 1] + gen[i]) / 2.0
        if i == n_old - 1:
            right = X_C_MAX
        else:
            right = (gen[i] + gen[i + 1]) / 2.0

        n_insert = min(n_per_cell[i], 20)  # 上限保护
        pts = np.linspace(left, right, n_insert + 2)[1:-1]
        new_gen.extend(pts)

    new_gen = np.sort(np.array(new_gen))
    new_gen = np.clip(new_gen, X_C_MIN, X_C_MAX)

    # 使用 Lloyd 算法松弛
    result = lloyd_cvt_1d(T, phase, n_generators=len(new_gen),
                          n_samples=3000, max_iter=50)
    # 覆盖为精细化的初始点
    result['generators'] = new_gen
    return result
