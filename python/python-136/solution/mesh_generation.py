"""
mesh_generation.py
==================
催化剂模拟的自适应网格生成模块。

基于种子项目 246_cvt_1d_sampling 与 261_cvt_square_uniform 重构：
- cvt_1d_sampling 使用 Lloyd 算法迭代优化一维生成器位置；
- cvt_square_uniform 在二维正方形区域生成质心 Voronoi 镶嵌。

在本系统中：
1. 一维 CVT 用于催化剂颗粒径向自适应网格加密（在反应剧烈区域自动聚集节点）；
2. 二维 CVT 用于催化剂截面多孔域的网格剖分，与有限元/有限差分耦合；
3. 所有可视化内容已移除，仅保留节点坐标与 Voronoi 单元信息。
"""

import numpy as np


class MeshGenerationError(Exception):
    """网格生成异常。"""
    pass


def cvt_1d_lloyd(n_generators, n_iterations, n_samples,
                 density_func=None, domain=(0.0, 1.0)):
    r"""
    一维质心 Voronoi 镶嵌（CVT）网格生成器。

    Lloyd 算法迭代：
        1. 在域内按密度函数生成大量随机样本点；
        2. 将每个样本点分配到最近的生成器（Voronoi 单元）；
        3. 将每个生成器更新为其 Voronoi 单元内样本点的质心；
        4. 重复直到收敛。

    密度函数 ρ(x) 控制网格疏密：ρ(x) 越大，该区域网格越密。
    样本点的接受-拒绝采样用于实现非均匀密度。

    数学上，生成器 g_i 的更新规则为：
        g_i^{new} = \frac{\int_{V_i} x \rho(x) dx}{\int_{V_i} \rho(x) dx}

    离散形式（蒙特卡洛近似）：
        g_i^{new} = \frac{\sum_{x_j \in V_i} x_j}{|V_i|}

    Parameters
    ----------
    n_generators : int
        生成器数量（内部节点数）。
    n_iterations : int
        Lloyd 迭代次数。
    n_samples : int
        每次迭代的样本点数。
    density_func : callable, optional
        密度函数 ρ(x) ≥ 0。若 None，默认为均匀密度 ρ(x)=1。
    domain : tuple of float
        计算域 (a, b)。

    Returns
    -------
    generators : ndarray, shape (n_generators,)
        优化后的生成器位置。
    energy_history : list of float
        每次迭代的 CVT 能量（量化误差）。
    """
    a, b = domain
    if a >= b:
        raise MeshGenerationError("domain 必须满足 a < b")
    if n_generators < 2:
        raise MeshGenerationError("生成器数量至少为 2")

    if density_func is None:
        density_func = lambda x: np.ones_like(x)

    # 初始化生成器：在域内均匀分布
    generators = np.linspace(a + 0.01 * (b - a), b - 0.01 * (b - a),
                             n_generators)
    generators = np.sort(generators)
    energy_history = []

    for _ in range(n_iterations):
        # 接受-拒绝采样，按密度函数生成样本
        samples = np.empty(0)
        batch = min(n_samples * 5, 1000000)
        while samples.size < n_samples:
            cand = np.random.uniform(a, b, size=batch)
            rho_cand = density_func(cand)
            rho_max = np.max(rho_cand)
            if rho_max <= 0:
                raise MeshGenerationError("密度函数在非零测集上为零")
            accept = np.random.uniform(0, rho_max, size=batch) < rho_cand
            accepted = cand[accept]
            samples = np.concatenate([samples, accepted])
        samples = samples[:n_samples]
        samples = np.sort(samples)

        # 计算 Voronoi 边界：相邻生成器的中点
        boundaries = np.empty(n_generators + 1)
        boundaries[0] = a
        boundaries[-1] = b
        if n_generators > 1:
            boundaries[1:-1] = 0.5 * (generators[:-1] + generators[1:])

        # 分配样本到 Voronoi 单元并计算质心
        new_generators = np.zeros(n_generators)
        counts = np.zeros(n_generators)
        energy = 0.0

        # 利用排序后样本的高效区间搜索
        idx = np.searchsorted(samples, boundaries)
        for i in range(n_generators):
            lo = idx[i]
            hi = idx[i + 1]
            cell_samples = samples[lo:hi]
            counts[i] = cell_samples.size
            if counts[i] > 0:
                new_generators[i] = np.mean(cell_samples)
                energy += np.sum((cell_samples - generators[i]) ** 2)
            else:
                # 空单元保持原位置（边界情况）
                new_generators[i] = generators[i]

        energy = energy / n_samples
        energy_history.append(energy)
        generators = np.sort(new_generators)

    return generators, energy_history


def cvt_square_uniform_2d(n_generators, n_iterations, n_samples,
                          domain=(0.0, 1.0, 0.0, 1.0)):
    """
    二维正方形区域 CVT 网格生成。

    基于 cvt_square_uniform 的核心思想，使用采样法近似 Voronoi 单元，
    并通过最近邻分配更新生成器到质心。

    Parameters
    ----------
    n_generators : int
        生成器数量。
    n_iterations : int
        Lloyd 迭代次数。
    n_samples : int
        每次迭代的样本数。
    domain : tuple
        (xmin, xmax, ymin, ymax)。

    Returns
    -------
    generators : ndarray, shape (n_generators, 2)
        二维生成器坐标。
    energy_history : list of float
    """
    xmin, xmax, ymin, ymax = domain
    if n_generators < 2:
        raise MeshGenerationError("生成器数量至少为 2")

    # 随机初始化
    generators = np.random.rand(n_generators, 2)
    generators[:, 0] = xmin + generators[:, 0] * (xmax - xmin)
    generators[:, 1] = ymin + generators[:, 1] * (ymax - ymin)

    energy_history = []

    for _ in range(n_iterations):
        # 均匀采样
        samples = np.random.rand(n_samples, 2)
        samples[:, 0] = xmin + samples[:, 0] * (xmax - xmin)
        samples[:, 1] = ymin + samples[:, 1] * (ymax - ymin)

        # 最近邻分配（欧氏距离）
        # 使用广播计算距离矩阵 (n_samples, n_generators)
        dx = samples[:, 0:1] - generators[:, 0].reshape(1, -1)
        dy = samples[:, 1:2] - generators[:, 1].reshape(1, -1)
        dists = dx ** 2 + dy ** 2
        nearest = np.argmin(dists, axis=1)

        # 计算每个单元的质心与能量
        new_generators = np.zeros_like(generators)
        counts = np.zeros(n_generators)
        energy = 0.0

        for i in range(n_generators):
            mask = nearest == i
            cell_samples = samples[mask]
            counts[i] = cell_samples.shape[0]
            if counts[i] > 0:
                new_generators[i] = np.mean(cell_samples, axis=0)
                energy += np.sum(
                    np.sum((cell_samples - generators[i]) ** 2, axis=1)
                )
            else:
                new_generators[i] = generators[i]

        energy = energy / n_samples
        energy_history.append(energy)
        generators = new_generators.copy()

    return generators, energy_history


def adaptive_radial_mesh(R, n_nodes, reaction_steepness=5.0):
    """
    针对催化剂颗粒径向扩散-反应问题的自适应网格生成。

    在颗粒表面（r ≈ R）反应物浓度梯度通常最大，因此需要更密的网格。
    密度函数设计为：
        ρ(r) = 1 + s * (r/R)^2
    其中 s 为陡峭度参数。

    Parameters
    ----------
    R : float
        颗粒半径。
    n_nodes : int
        内部节点数。
    reaction_steepness : float
        密度陡峭度。

    Returns
    -------
    nodes : ndarray
        径向节点坐标（包含边界 0 和 R）。
    """
    if R <= 0:
        raise MeshGenerationError("R 必须为正")

    def density(r):
        return 1.0 + reaction_steepness * (r / R) ** 2

    generators, _ = cvt_1d_lloyd(
        n_generators=n_nodes - 2,
        n_iterations=30,
        n_samples=50000,
        density_func=density,
        domain=(0.0, R)
    )
    nodes = np.concatenate([[0.0], generators, [R]])
    nodes = np.sort(nodes)
    # 去重
    nodes = np.unique(nodes)
    # 若去重后节点数不足，线性插值补充
    if nodes.size < n_nodes:
        nodes = np.linspace(0.0, R, n_nodes)
    return nodes
