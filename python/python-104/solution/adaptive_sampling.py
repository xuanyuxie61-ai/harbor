"""
adaptive_sampling.py — 自适应采样与最优网格生成

融合原项目:
  - 262_cvt_triangle_uniform (质心Voronoi剖分/CVT)
  - 718_matlab_commandline (文件日志与数据记录)

功能:
  - 基于CVT的三角形域上最优采样点分布
  - Lloyd迭代优化采样点位置
  - 自适应网格加密 (基于残差估计)
  - 采样数据日志记录

物理模型:
  1. 质心Voronoi剖分 (CVT):
       给定区域 Omega 和 N 个生成点 {z_i},
       Voronoi单元: V_i = { x in Omega | ||x - z_i|| <= ||x - z_j|| for all j }
       CVT条件: z_i = centroid(V_i) = (1/|V_i|) * integral_{V_i} x dA

     Lloyd算法:
       重复: 计算Voronoi单元 -> 将z_i更新为单元质心

  2. 在AO中, CVT用于:
       - 波前传感器子孔径的最优布局
       - 相位屏采样点的自适应分布 (高梯度区域加密)

  3. 三角形均匀采样 (源自262):
       给定三角形顶点 v0, v1, v2,
       均匀随机点: p = (1 - sqrt(u1)) * v0 + sqrt(u1)*(1-u2)*v1 + sqrt(u1)*u2*v2,
       其中 u1, u2 ~ Uniform(0,1).
"""

import numpy as np


# --- 三角形域操作 ---

def triangle_area(v0, v1, v2):
    """
    计算三角形面积 (二维).

    A = 0.5 * | (v1-v0) x (v2-v0) |
    """
    v0, v1, v2 = np.array(v0), np.array(v1), np.array(v2)
    cross = (v1[0] - v0[0]) * (v2[1] - v0[1]) - (v1[1] - v0[1]) * (v2[0] - v0[0])
    return 0.5 * abs(cross)


def sample_triangle_uniform(v0, v1, v2, n_samples, seed=None):
    """
    在三角形内均匀随机采样 (源自262_cvt_triangle_uniform).

    算法:
      u1, u2 ~ Uniform(0,1)
      alpha = sqrt(u1)
      p = (1-alpha)*v0 + alpha*(1-u2)*v1 + alpha*u2*v2
    """
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1.")
    if seed is not None:
        np.random.seed(seed)
    v0, v1, v2 = np.array(v0), np.array(v1), np.array(v2)
    u1 = np.random.rand(n_samples)
    u2 = np.random.rand(n_samples)
    alpha = np.sqrt(u1)
    points = ((1.0 - alpha)[:, None] * v0[None, :]
              + alpha[:, None] * (1.0 - u2)[:, None] * v1[None, :]
              + alpha[:, None] * u2[:, None] * v2[None, :])
    return points


def triangle_centroid(points):
    """
    计算点集的质心.
    """
    if len(points) == 0:
        return np.array([0.0, 0.0])
    return np.mean(points, axis=0)


# --- CVT优化 (源自262_cvt_triangle_uniform) ---

def cvt_triangle_uniform(triangle_vertices, n_generators, n_samples_per_iter=5000,
                          n_iterations=50, seed=None):
    """
    在三角形区域内计算质心Voronoi剖分.

    算法:
      1. 随机初始化生成点
      2. 在三角形内均匀采样大量点
      3. 对每个采样点, 找到最近的生成点
      4. 将每个生成点更新为其Voronoi单元内采样点的质心
      5. 重复2-4

    返回: generators (N x 2 数组)
    """
    if n_generators < 1:
        raise ValueError("n_generators must be >= 1.")
    if n_iterations < 1:
        raise ValueError("n_iterations must be >= 1.")

    v0, v1, v2 = np.array(triangle_vertices[0]), np.array(triangle_vertices[1]), np.array(triangle_vertices[2])

    # 初始化生成点
    if seed is not None:
        np.random.seed(seed)
    generators = sample_triangle_uniform(v0, v1, v2, n_generators)

    for it in range(n_iterations):
        # 采样大量点
        samples = sample_triangle_uniform(v0, v1, v2, n_samples_per_iter)

        # 分配最近生成点
        assignments = np.zeros(n_samples_per_iter, dtype=int)
        for i, samp in enumerate(samples):
            dists = np.sum((generators - samp) ** 2, axis=1)
            assignments[i] = np.argmin(dists)

        # 更新质心
        new_generators = np.zeros_like(generators)
        counts = np.zeros(n_generators, dtype=int)
        for k in range(n_generators):
            mask = assignments == k
            if np.sum(mask) > 0:
                new_generators[k] = np.mean(samples[mask], axis=0)
                counts[k] = np.sum(mask)
            else:
                # 空单元: 重新随机初始化
                new_generators[k] = sample_triangle_uniform(v0, v1, v2, 1)[0]
                counts[k] = 1

        # 检查收敛
        shift = np.max(np.linalg.norm(new_generators - generators, axis=1))
        generators = new_generators
        if shift < 1e-8:
            break

    return generators


def cvt_disk_uniform(n_generators, radius=1.0, n_iterations=30, seed=None):
    """
    在单位圆盘上计算CVT.

    将圆盘剖分为多个扇形三角形, 分别计算CVT后合并.
    """
    if n_generators < 1:
        raise ValueError("n_generators must be >= 1.")

    n_sectors = max(6, int(np.sqrt(n_generators)))
    generators_per_sector = max(1, n_generators // n_sectors)
    all_generators = []

    angles = np.linspace(0, 2 * np.pi, n_sectors + 1)
    for s in range(n_sectors):
        theta0, theta1 = angles[s], angles[s + 1]
        v0 = np.array([0.0, 0.0])
        v1 = np.array([radius * np.cos(theta0), radius * np.sin(theta0)])
        v2 = np.array([radius * np.cos(theta1), radius * np.sin(theta1)])

        gens = cvt_triangle_uniform([v0, v1, v2], generators_per_sector,
                                     n_iterations=n_iterations, seed=seed)
        all_generators.append(gens)
        if seed is not None:
            seed += 1

    generators = np.vstack(all_generators)
    # 如果过多, 随机选取
    if len(generators) > n_generators:
        idx = np.random.choice(len(generators), n_generators, replace=False)
        generators = generators[idx]
    return generators


# --- 自适应采样 ---

def adaptive_phase_sampling(phase, mask, n_target_points, n_iterations=20):
    """
    基于CVT的自适应相位采样.

    在高相位梯度区域自动加密采样点.
    方法:
      1. 计算相位梯度幅值
      2. 将梯度幅值作为密度函数
      3. 使用拒绝采样+CVT迭代生成非均匀分布
    """
    if n_target_points < 1:
        raise ValueError("n_target_points must be >= 1.")

    grid_size = phase.shape[0]
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    X, Y = np.meshgrid(x, y)

    # 计算梯度幅值
    dphidx = np.zeros_like(phase)
    dphidy = np.zeros_like(phase)
    dphidx[:, 1:-1] = (phase[:, 2:] - phase[:, :-2]) / (2.0 * (x[1] - x[0]))
    dphidy[1:-1, :] = (phase[2:, :] - phase[:-2, :]) / (2.0 * (y[1] - y[0]))
    grad_amp = np.sqrt(dphidx ** 2 + dphidy ** 2)
    grad_amp[~mask] = 0.0
    grad_max = np.max(grad_amp)
    if grad_max < 1e-20:
        grad_amp = np.ones_like(grad_amp) * mask

    # 收集瞳孔内坐标
    coords = np.column_stack([X[mask].ravel(), Y[mask].ravel()])
    weights = grad_amp[mask].ravel()
    weights = weights / np.sum(weights)

    if len(coords) < n_target_points:
        n_target_points = len(coords)

    # 使用加权采样初始化
    indices = np.random.choice(len(coords), size=n_target_points, p=weights)
    generators = coords[indices].copy()

    # Lloyd迭代 (带权重)
    for _ in range(n_iterations):
        # 分配
        assignments = np.zeros(len(coords), dtype=int)
        for i, pt in enumerate(coords):
            dists = np.sum((generators - pt) ** 2, axis=1)
            assignments[i] = np.argmin(dists)

        # 加权质心更新
        new_gens = np.zeros_like(generators)
        for k in range(n_target_points):
            mask_k = assignments == k
            if np.sum(mask_k) > 0:
                w = weights[mask_k]
                pts = coords[mask_k]
                new_gens[k] = np.sum(pts * w[:, None], axis=0) / np.sum(w)
            else:
                new_gens[k] = coords[np.random.randint(len(coords))]

        generators = new_gens

    return generators


# --- 日志记录 (源自718_matlab_commandline) ---

def log_sampling_info(filepath, generators, iteration, residual=None):
    """
    记录采样点位置和迭代信息.
    """
    with open(filepath, 'w') as f:
        f.write("# Adaptive Sampling Log\n")
        f.write(f"# Iteration: {iteration}\n")
        if residual is not None:
            f.write(f"# Residual: {residual:.6e}\n")
        f.write("# x y\n")
        for pt in generators:
            f.write(f"{pt[0]:.12e} {pt[1]:.12e}\n")
