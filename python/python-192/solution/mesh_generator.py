"""
================================================================================
自适应网格生成模块 (mesh_generator.py)
================================================================================
融合项目:
  - 1398_voronoi_plot: Voronoi图生成与距离计算
  - 1290_tree_chaos: 迭代函数系统（IFS）分形生成
  - 585_image_sample: 边界采样与几何提取

在可压缩湍流CFD中，近壁面区域需要极精细的网格以解析边界层。
本模块提供：
  1. Voronoi非结构化网格生成（用于复杂几何区域）
  2. IFS自适应分形细化（基于混沌迭代的局部加密）
  3. 边界层采样点提取（壁面法向网格拉伸）
================================================================================
"""

import numpy as np
from utils_numerical import safe_divide


def generate_voronoi_mesh(nc: int, bounds: tuple = ((0.0, 1.0), (0.0, 1.0)),
                          m: int = 100, n: int = 100, p_norm: float = 2.0) -> dict:
    """
    基于Voronoi图的非结构化网格生成器

    对于点集 {g_i}，Voronoi单元定义为：

        V_i = { x ∈ Ω : ||x - g_i||_p ≤ ||x - g_j||_p, ∀j≠i }

    在CFD中，Voronoi单元可用于有限体积法的控制体，
    其对偶图Delaunay三角化则用于有限元/谱元离散。

    参数:
        nc: 生成点数量
        bounds: ((xmin, xmax), (ymin, ymax))
        m, n: 背景网格分辨率（用于离散化Voronoi区域）
        p_norm: 距离范数 (2=欧氏, 1=曼哈顿, np.inf=切比雪夫)

    返回:
        dict 包含生成点、Voronoi单元索引、背景网格坐标
    """
    xmin, xmax = bounds[0]
    ymin, ymax = bounds[1]

    # 生成随机生成点（添加边界点确保覆盖）
    np.random.seed(42)
    generators = np.random.rand(2, nc)
    generators[0, :] = xmin + generators[0, :] * (xmax - xmin)
    generators[1, :] = ymin + generators[1, :] * (ymax - ymin)

    # 添加边界点
    boundary_pts = np.array([
        [xmin, xmin, xmax, xmax],
        [ymin, ymax, ymin, ymax]
    ])
    generators = np.hstack([generators, boundary_pts])
    nc_total = generators.shape[1]

    # 背景网格
    x_grid = np.linspace(xmin, xmax, n)
    y_grid = np.linspace(ymin, ymax, m)
    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]

    # 计算每个背景网格点最近的生成点
    voronoi_map = np.zeros((m, n), dtype=int)
    for i in range(m):
        y = y_grid[i]
        for j in range(n):
            x = x_grid[j]

            min_dist = np.inf
            nearest = 0
            for k in range(nc_total):
                gx, gy = generators[:, k]
                if p_norm == np.inf:
                    dist = max(abs(x - gx), abs(y - gy))
                elif p_norm == 1.0:
                    dist = abs(x - gx) + abs(y - gy)
                elif p_norm == 2.0:
                    dist = (x - gx) ** 2 + (y - gy) ** 2
                else:
                    dx_ = abs(x - gx)
                    dy_ = abs(y - gy)
                    dist = (dx_ ** p_norm + dy_ ** p_norm) ** (1.0 / p_norm)

                if dist < min_dist:
                    min_dist = dist
                    nearest = k

            voronoi_map[i, j] = nearest

    # 提取每个Voronoi单元的面积
    cell_areas = np.zeros(nc_total)
    for k in range(nc_total):
        cell_areas[k] = np.sum(voronoi_map == k) * dx * dy

    return {
        'generators': generators[:, :nc],
        'voronoi_map': voronoi_map,
        'x_grid': x_grid,
        'y_grid': y_grid,
        'cell_areas': cell_areas[:nc],
        'dx': dx,
        'dy': dy
    }


def ifs_adaptive_refinement(n_iterations: int = 5000, bounds: tuple = ((0.0, 1.0), (0.0, 1.0)),
                            refinement_regions: list = None) -> np.ndarray:
    """
    基于迭代函数系统（IFS）的自适应网格细化点生成

    IFS通过仿射变换的随机迭代生成分形集：

        x_{k+1} = A_j · x_k + b_j,   j ~ 离散概率分布

    在CFD中，利用IFS的吸引子特性在边界层/激波区域生成
    高密度采样点，实现自适应网格加密。

    Barnsley分形蕨的变换矩阵:
        A₀ = [[0, 0], [0, 0.5]],    b₀ = [0.5, 0]
        A₁ = [[0.1, 0], [0, 0.1]],  b₁ = [0.45, 0.15]
        A₂ = [[0.42, -0.42], [0.42, 0.42]],  b₂ = [0.29, -0.01]
        A₃ = [[0.42, 0.42], [-0.42, 0.42]],  b₃ = [0.29, 0.41]

    参数:
        n_iterations: 迭代次数（生成点数）
        bounds: 物理区域边界
        refinement_regions: 需要加密的矩形区域列表 [(xmin,xmax,ymin,ymax), ...]

    返回:
        points: (2 x n_iterations) 采样点坐标
    """
    # IFS变换定义（树形混沌吸引子）
    transforms = [
        {'A': np.array([[0.0, 0.0], [0.0, 0.5]]), 'b': np.array([0.5, 0.0]), 'prob': 0.25},
        {'A': np.array([[0.1, 0.0], [0.0, 0.1]]), 'b': np.array([0.45, 0.15]), 'prob': 0.25},
        {'A': np.array([[0.42, -0.42], [0.42, 0.42]]), 'b': np.array([0.29, -0.01]), 'prob': 0.25},
        {'A': np.array([[0.42, 0.42], [-0.42, 0.42]]), 'b': np.array([0.29, 0.41]), 'prob': 0.25}
    ]

    # 累积概率
    cum_prob = np.cumsum([t['prob'] for t in transforms])

    points = np.zeros((2, n_iterations))
    x = np.random.rand(2)

    # 预热
    for _ in range(100):
        r = np.random.rand()
        idx = np.searchsorted(cum_prob, r)
        x = transforms[idx]['A'] @ x + transforms[idx]['b']

    # 主迭代
    accepted = 0
    i = 0
    while accepted < n_iterations and i < n_iterations * 10:
        i += 1
        r = np.random.rand()
        idx = np.searchsorted(cum_prob, r)
        x = transforms[idx]['A'] @ x + transforms[idx]['b']

        # 映射到物理区域
        px = bounds[0][0] + x[0] * (bounds[0][1] - bounds[0][0])
        py = bounds[1][0] + x[1] * (bounds[1][1] - bounds[1][0])

        # 区域加密策略：在refinement_regions内接受概率更高
        if refinement_regions:
            in_refined = any(
                rx[0] <= px <= rx[1] and ry[0] <= py <= ry[1]
                for rx, ry in refinement_regions
            )
            if in_refined:
                # 加密区域以更高概率接受
                if np.random.rand() < 0.8:
                    points[:, accepted] = [px, py]
                    accepted += 1
            else:
                if np.random.rand() < 0.3:
                    points[:, accepted] = [px, py]
                    accepted += 1
        else:
            points[:, accepted] = [px, py]
            accepted += 1

    return points[:, :accepted]


def sample_boundary_points(n_points: int = 50, boundary_type: str = 'plate',
                           Re: float = 1e5, x_range: tuple = (0.0, 1.0)) -> tuple:
    """
    边界层几何采样（基于image_sample思想）

    对于平板边界层，壁面法向坐标按双曲正切拉伸：

        y_j = y_max · tanh(β · j / N) / tanh(β)

    其中 β 控制拉伸率，需满足：

        y₁⁺ ≈ 1  (第一个网格点的壁面单位距离)
        y₁ = y⁺ · ν / u_τ

    参数:
        n_points: 法向采样点数
        boundary_type: 'plate' 或 'airfoil'
        Re: 雷诺数
        x_range: 流向范围

    返回:
        x, y: 边界采样点坐标
        dy_wall: 壁面处网格间距
    """
    if boundary_type == 'plate':
        # 平板边界层：Blasius解参考厚度
        # δ ≈ 5.0 · x / √Re_x
        x_sample = np.linspace(x_range[0] + 0.01, x_range[1], n_points)
        delta = 5.0 * x_sample / np.sqrt(np.maximum(Re * x_sample, 1.0))

        # 法向坐标（双曲正切拉伸）
        beta = 2.5
        eta = np.linspace(0.0, 1.0, n_points)
        y_norm = np.tanh(beta * eta) / np.tanh(beta)

        # 生成二维网格点
        x_grid = np.tile(x_sample, n_points)
        y_grid = np.outer(delta, y_norm).flatten()

        # 壁面间距（第一个法向层）
        dy_wall = delta[0] * (np.tanh(beta / n_points) / np.tanh(beta))

        return x_grid, y_grid, float(dy_wall)

    else:
        # 简化翼型型线 (NACA 0012近似)
        theta = np.linspace(0.0, 2.0 * np.pi, n_points)
        x_airfoil = 0.5 * (1.0 + np.cos(theta))
        y_airfoil = 0.06 * (0.2969 * np.sqrt(x_airfoil) - 0.1260 * x_airfoil
                          - 0.3516 * x_airfoil ** 2 + 0.2843 * x_airfoil ** 3
                          - 0.1015 * x_airfoil ** 4)

        # 对称翼型
        x_points = np.concatenate([x_airfoil, x_airfoil[::-1]])
        y_points = np.concatenate([y_airfoil, -y_airfoil[::-1]])

        return x_points, y_points, 0.01


def generate_spectral_element_mesh(nx: int = 16, ny: int = 16,
                                   x_bounds: tuple = (0.0, 1.0),
                                   y_bounds: tuple = (0.0, 1.0),
                                   stretch_y: bool = True) -> dict:
    """
    生成谱元法计算网格

    使用Gauss-Lobatto-Legendre (GLL) 点分布：

        x_i = (1 - ξ_i)/2 · x_min + (1 + ξ_i)/2 · x_max

    其中 ξ_i 为 [-1, 1] 上的GLL点，满足：

        (1 - ξ²) P'_{N-1}(ξ) = 0

    P_N 为N阶Legendre多项式。

    参数:
        nx, ny: x和y方向的单元数
        x_bounds, y_bounds: 区域边界
        stretch_y: 是否对y方向进行边界层拉伸

    返回:
        dict 包含坐标数组、Jacobian、网格间距
    """
    # 使用Chebyshev点作为GLL点的近似（计算更简单）
    def chebyshev_nodes(n):
        return np.cos(np.pi * np.arange(n + 1) / n)

    # 每个谱元内部的GLL点数
    n_gll = 8
    xi = chebyshev_nodes(n_gll)

    x_nodes = np.linspace(x_bounds[0], x_bounds[1], nx + 1)
    y_nodes = np.linspace(y_bounds[0], y_bounds[1], ny + 1)

    # 全局网格坐标
    npx = nx * n_gll + 1
    npy = ny * n_gll + 1
    x = np.zeros(npx)
    y = np.zeros(npy)

    # x方向均匀分布
    for i in range(nx):
        for j in range(n_gll + 1):
            idx = i * n_gll + j
            if idx < npx:
                x_local = 0.5 * ((1 - xi[j]) * x_nodes[i] + (1 + xi[j]) * x_nodes[i + 1])
                x[idx] = x_local

    # y方向（可选拉伸）
    for i in range(ny):
        for j in range(n_gll + 1):
            idx = i * n_gll + j
            if idx < npy:
                y_local = 0.5 * ((1 - xi[j]) * y_nodes[i] + (1 + xi[j]) * y_nodes[i + 1])
                if stretch_y:
                    # 指数拉伸：壁面附近加密
                    y_local = y_bounds[1] * (np.exp(2.0 * y_local / y_bounds[1]) - 1.0) / (np.exp(2.0) - 1.0)
                y[idx] = y_local

    # 去重并排序
    x = np.unique(np.round(x, 12))
    y = np.unique(np.round(y, 12))

    # 生成二维网格
    X, Y = np.meshgrid(x, y)

    # 计算Jacobian和网格间距
    dx = np.diff(x)
    dy = np.diff(y)

    return {
        'x': x,
        'y': y,
        'X': X,
        'Y': Y,
        'dx_min': float(np.min(dx)) if len(dx) > 0 else 1e-3,
        'dy_min': float(np.min(dy)) if len(dy) > 0 else 1e-3,
        'nx': len(x),
        'ny': len(y)
    }
