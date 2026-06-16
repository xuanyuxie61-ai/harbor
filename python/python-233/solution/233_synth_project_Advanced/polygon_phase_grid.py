"""
polygon_phase_grid.py — 多边形相位空间网格与六边形布里渊区采样
============================================================
本模块实现基于多边形扇形三角化的相空间网格生成方法,
用于顶夸克质量测量中的多维参数空间离散化。

核心方法:
  对于 d 维参数空间中的多边形区域 Ω,
  使用从质心扇形三角化生成规则三角网格:

  1. 计算质心: v_c = (1/N_v) Σ v_i
  2. 对每条边 (v_l, v_{l+1}), 形成三角形 (v_l, v_{l+1}, v_c)
  3. 在每个三角形内用重心坐标生成网格点:
     x_g = (i × v_l + j × v_{l+1} + k × v_c) / n
     其中 i+j+k = n, i≥1, j≥0, k≥0

  总网格点数:
    N_g = 1 + N_edges × n(n+1)/2

  这给出了 O(n²) 的均匀覆盖, 适用于:
  - 系统参数 (JES, b-energy scale) 的二维扫描
  - 相位空间角度 (cosθ, φ) 的网格化
  - 二维 nuisance parameter 空间的离散化

  映射种子项目:
    - 885_polygon_grid: 多边形扇形三角化网格生成
"""

import numpy as np


def polygon_centroid(vertices):
    """
    计算多边形质心 (算术平均)。

    v_c = (1/N_v) Σ_{i=1}^{N_v} v_i

    注意: 这是顶点质心, 非面积质心。
    对于凸多边形两者相同, 非凸多边形可能不同。

    参数:
        vertices: (N_v, 2) 顶点坐标

    返回:
        centroid: (2,) 质心坐标
    """
    return np.mean(vertices, axis=0)


def polygon_grid_count(n_subdivisions, n_vertices):
    """
    计算多边形网格的总点数。

    N_g = 1 + N_v × n(n+1)/2

    其中:
      - 1 对应质心点
      - N_v × n(n+1)/2 对应 N_v 个扇形三角形中的点

    参数:
        n_subdivisions: 每条边的细分段数 n
        n_vertices: 多边形顶点数 N_v

    返回:
        总网格点数
    """
    return 1 + n_vertices * n_subdivisions * (n_subdivisions + 1) // 2


def polygon_grid_points(n_subdivisions, vertices):
    """
    在多边形内生成规则三角网格。

    算法步骤:
    1. 计算质心 v_c
    2. 对每条边 l = 0, ..., N_v-1:
       三角形 (v_l, v_{l+1}, v_c)
       对 i = 1..n, j = 0..n-i, k = n-i-j:
         x_g = (i × v_l + j × v_{l+1} + k × v_c) / n

    重心坐标保证网格点在三角形内,
    且边界点精确落在多边形边上。

    映射种子项目:
      - 885_polygon_grid: 扇形三角化 + 重心坐标

    参数:
        n_subdivisions: 细分段数 n
        vertices: (N_v, 2) 多边形顶点

    返回:
        grid_points: (N_g, 2) 网格点坐标
    """
    vertices = np.asarray(vertices, dtype=float)
    n_v = len(vertices)
    n = n_subdivisions

    n_total = polygon_grid_count(n, n_v)
    grid_points = np.zeros((n_total, 2))

    # 质心
    v_c = polygon_centroid(vertices)
    grid_points[0] = v_c

    idx = 1
    for l in range(n_v):
        v_l = vertices[l]
        v_next = vertices[(l + 1) % n_v]

        for i in range(1, n + 1):
            for j in range(0, n - i + 1):
                k = n - i - j
                point = (i * v_l + j * v_next + k * v_c) / float(n)
                if idx < n_total:
                    grid_points[idx] = point
                    idx += 1

    return grid_points[:idx]


def hexagonal_brillouin_grid(n_subdivisions):
    """
    生成六边形布里渊区网格 (用于周期系统参数扫描)。

    六边形顶点 (正六边形, 外接圆半径 = 1):
      v_k = (cos(2πk/6), sin(2πk/6)), k = 0,...,5

    应用场景:
    - 系统误差参数 (α₁, α₂, ..., α₆) 的周期扫描
    - 探测器校准参数的六角对称性

    参数:
        n_subdivisions: 细分段数

    返回:
        grid_points: (N_g, 2) 六边形内网格点
    """
    n_vertices = 6
    angles = np.array([2.0 * np.pi * k / n_vertices for k in range(n_vertices)])
    vertices = np.column_stack([np.cos(angles), np.sin(angles)])

    return polygon_grid_points(n_subdivisions, vertices)


def rectangular_parameter_grid(param_ranges, n_points_per_dim):
    """
    生成矩形参数空间网格 (张量积网格)。

    对于 d 维参数空间, 每维 n 个点:
      总点数 = n^d

    用于系统误差参数的规则扫描。

    参数:
        param_ranges: [(min_1, max_1), ..., (min_d, max_d)]
        n_points_per_dim: 每维点数 (标量或列表)

    返回:
        grid: (N_total, d) 参数网格点
    """
    d = len(param_ranges)
    if np.isscalar(n_points_per_dim):
        n_points_per_dim = [n_points_per_dim] * d

    axes = [np.linspace(r[0], r[1], n)
            for r, n in zip(param_ranges, n_points_per_dim)]

    mesh = np.meshgrid(*axes, indexing='ij')
    grid = np.column_stack([m.ravel() for m in mesh])

    return grid


def adaptive_polygon_refinement(vertices, func, tol=1e-4, max_depth=5):
    """
    自适应多边形网格细化 (基于函数梯度)。

    在函数变化剧烈的区域增加网格密度:
    1. 在粗网格上计算 |∇f|
    2. 对 |∇f| > threshold 的三角形进行 1→4 细分
    3. 重复直到所有区域 |∇f| < tol 或达到 max_depth

    参数:
        vertices: 多边形顶点
        func: 标量函数 f: R² → R
        tol: 梯度阈值
        max_depth: 最大细化深度

    返回:
        refined_points: 细化后的网格点
        func_values: 函数值
    """
    # 初始粗网格
    points = polygon_grid_points(4, vertices)
    values = np.array([func(p) for p in points])

    for depth in range(max_depth):
        # 估计梯度 (有限差分)
        n_pts = len(points)
        if n_pts < 3:
            break

        # 简单差分梯度
        h = 0.01
        grad_mag = np.zeros(n_pts)
        for i in range(n_pts):
            fx_plus = func(points[i] + np.array([h, 0]))
            fx_minus = func(points[i] - np.array([h, 0]))
            fy_plus = func(points[i] + np.array([0, h]))
            fy_minus = func(points[i] - np.array([0, h]))
            grad_mag[i] = np.sqrt(((fx_plus - fx_minus) / (2 * h))**2 +
                                  ((fy_plus - fy_minus) / (2 * h))**2)

        if np.max(grad_mag) < tol:
            break

        # 增加细分
        n_new = 4 + 2 * (depth + 1)
        points = polygon_grid_points(n_new, vertices)
        values = np.array([func(p) for p in points])

    return points, values


def nuisance_parameter_hex_scan(nuisance_names, scan_range=2.0, n_rings=3):
    """
    在六边形网格上扫描 nuisance 参数对。

    对于每对 nuisance 参数 (α_i, α_j),
    在 [-scan_range, scan_range]² 内生成六边形网格,
    评估似然函数 L(α_i, α_j)。

    参数:
        nuisance_names: 参数名列表
        scan_range: 扫描范围 (以 σ 为单位)
        n_rings: 径向环数

    返回:
        scan_grids: 字典 { (name_i, name_j): grid_points }
    """
    n_params = len(nuisance_names)
    scan_grids = {}

    for i in range(n_params):
        for j in range(i + 1, n_params):
            # 六边形网格
            grid = hexagonal_brillouin_grid(n_rings * 3)

            # 缩放到扫描范围
            grid_scaled = grid * scan_range

            scan_grids[(nuisance_names[i], nuisance_names[j])] = grid_scaled

    return scan_grids
