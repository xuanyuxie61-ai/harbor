"""
grid_generator.py
=================
非均匀网格生成模块，整合三种网格策略:

  1. CVT (Centroidal Voronoi Tessellation) 非均匀网格 (源自 245_cvt_1d_nonuniform)
     - Lloyd 迭代算法使生成点收敛到密度函数的质心
     - 用于在离子浓度梯度大的区域加密网格

  2. 圆弧网格 (源自 176_circle_arc_grid)
     - 沿圆形/柱形样品边界生成等弧长网格
     - 用于圆柱形电池电极的截面离散

  3. 三角形邻接拓扑 (源自 1354_triangulation_triangle_neighbors)
     - 构建三角形网格的邻接关系
     - 用于非结构网格上的有限差分/有限体积法

数学背景:
  CVT 能量泛函 (Lloyd 1982):
    E(V, Z) = sum_i integral_{V_i} rho(x) * ||x - z_i||^2 dx

  Lloyd 迭代:
    z_i^{(k+1)} = integral_{V_i} x * rho(x) dx / integral_{V_i} rho(x) dx

  密度函数 rho(x) 可选:
    - 常数 (均匀网格)
    - sqrt(|grad c|) (基于浓度梯度自适应)
    - |d^2V/dx^2| (基于电势曲率)

  圆弧网格参数化:
    x(theta) = R*cos(theta), y(theta) = R*sin(theta)
    theta_j = theta_0 + j * dtheta, j = 0, ..., N-1
    dtheta = (theta_end - theta_start) / (N-1)

  三角形邻接: 若两个三角形共享两个节点，则它们邻接。
"""

import numpy as np
from material_constants import SMALL_NUMBER, CONVERGENCE_TOL


# ============================================================================
# CVT 非均匀网格 (1D)
# ============================================================================
def cvt_density_function(x, mode='gradient'):
    """
    密度函数 rho(x) 定义网格疏密。

    mode='gradient': rho ~ sqrt(|dc/dx|) 模拟浓度梯度
    mode='curvature': rho ~ |d^2phi/dx^2| 模拟电势曲率
    mode='sqrt': rho = sqrt(x)
    mode='chebyshev': rho = sin(pi*(x-0.5)) (Chebyshev-like)
    """
    x = np.asarray(x, dtype=np.float64)
    if mode == 'sqrt':
        return np.sqrt(np.abs(x) + SMALL_NUMBER)
    elif mode == 'chebyshev':
        return np.sin(np.pi * (x - 0.5)) + 1.0 + SMALL_NUMBER
    elif mode == 'gradient':
        # 模拟离子浓度梯度: 在 x=0.3 和 x=0.7 处梯度最大
        sigma = 0.1
        g = np.exp(-((x - 0.3)**2) / (2*sigma**2)) + np.exp(-((x - 0.7)**2) / (2*sigma**2))
        return np.sqrt(g + SMALL_NUMBER)
    elif mode == 'curvature':
        # 模拟 Debye 层附近的电势曲率
        kappa = 10.0
        return np.abs(kappa**2 * np.exp(-kappa * x) + kappa**2 * np.exp(-kappa * (1.0 - x))) + SMALL_NUMBER
    else:
        return np.ones_like(x)


def cvt_1d_lloyd(n_generators, n_samples=2000, n_steps=100, density_mode='gradient', seed=42):
    """
    1D CVT Lloyd 算法。

    算法:
    1. 初始化生成点 z_i (均匀或随机)
    2. 在 [0,1] 上生成 n_samples 个采样点, 权重为 rho(x)
    3. Voronoi 分配: 每个采样点分配给最近的生成点
    4. 更新生成点为各自 Voronoi 区域的质心
    5. 重复步骤 2-4

    返回:
        generators: 排序后的 CVT 生成点 [n_generators]
        energy: CVT 能量序列 (收敛历史)
    """
    rng = np.random.RandomState(seed)
    z = np.sort(rng.uniform(0.05, 0.95, size=n_generators))
    energy_history = []

    for step in range(n_steps):
        # 重要性采样: 使用拒绝采样
        x_samples = rng.uniform(0.0, 1.0, size=n_samples * 3)
        rho_vals = cvt_density_function(x_samples, mode=density_mode)
        rho_max = np.max(rho_vals)
        accept = rng.uniform(0, 1, size=len(x_samples)) < (rho_vals / (rho_max + SMALL_NUMBER))
        x_weighted = x_samples[accept][:n_samples]
        if len(x_weighted) < n_samples:
            x_weighted = np.sort(rng.uniform(0, 1, n_samples))
        rho_weighted = cvt_density_function(x_weighted, mode=density_mode)

        # Voronoi 边界为相邻生成点的中点
        bounds = 0.5 * (z[:-1] + z[1:])
        bounds = np.concatenate([[0.0], bounds, [1.0]])
        cell_energy = 0.0
        z_new = np.zeros_like(z)
        for i in range(n_generators):
            mask = (x_weighted >= bounds[i]) & (x_weighted < bounds[i+1])
            if np.sum(mask) == 0:
                z_new[i] = z[i]
                continue
            x_cell = x_weighted[mask]
            w_cell = rho_weighted[mask]
            total_w = np.sum(w_cell)
            if total_w < SMALL_NUMBER:
                z_new[i] = z[i]
            else:
                z_new[i] = np.sum(x_cell * w_cell) / total_w
                cell_energy += np.sum(w_cell * (x_cell - z_new[i])**2)
        energy_history.append(cell_energy)
        z = np.sort(z_new)

    return z, energy_history


# ============================================================================
# 圆弧网格
# ============================================================================
def circle_arc_grid(R, theta_start=0.0, theta_end=2*np.pi, n_points=50,
                    clustering='uniform', concentration_param=2.0):
    """
    圆弧网格生成。

    参数化: x(theta) = R*cos(theta), y(theta) = R*sin(theta)

    clustering:
      'uniform': 等角距 theta_j = theta_start + j*dtheta
      'tanh': 双曲正切聚集 (两端加密)
        theta(s) = theta_start + (theta_end-theta_start)/2 * (1 + tanh(a*(2s-1))/tanh(a))
        s in [0, 1], j/(N-1)
      'cosine': 余弦聚集 (中间加密)
        theta_j = theta_start + (theta_end-theta_start)*(1 - cos(pi*j/(N-1)))/2
    """
    j = np.arange(n_points, dtype=np.float64)
    s = j / max(n_points - 1, 1)

    if clustering == 'tanh':
        a = concentration_param
        s_mapped = 0.5 * (1.0 + np.tanh(a * (2.0*s - 1.0)) / np.tanh(a))
    elif clustering == 'cosine':
        s_mapped = 0.5 * (1.0 - np.cos(np.pi * s))
    else:  # uniform
        s_mapped = s

    theta = theta_start + (theta_end - theta_start) * s_mapped
    x = R * np.cos(theta)
    y = R * np.sin(theta)
    return x, y, theta


# ============================================================================
# 三角形邻接关系
# ============================================================================
def triangulation_triangle_neighbors(triangle_node, node_num_total=None):
    """
    构建三角形邻接表 (源自 1354)。

    算法:
    1. 对每个三角形提取3条边 (节点对)
    2. 将边排序 (小节点号在前)
    3. 按字典序排序所有边
    4. 相邻的相同边对应的两个三角形互为邻接

    参数:
        triangle_node: [3, n_tri] 整数数组, 每个三角形的3个节点 (1-indexed)
        node_num_total: 总节点数 (可选)

    返回:
        neighbors: [3, n_tri] 整数数组
          neighbors(k, e) = 与三角形 e 的第 k 条边相邻的三角形编号
          = 0 表示边界边

    三角形的边约定:
      边 1: 节点 2-3 (对面为节点 1)
      边 2: 节点 1-3 (对面为节点 2)
      边 3: 节点 1-2 (对面为节点 3)
    """
    triangle_node = np.asarray(triangle_node, dtype=np.int64)
    if triangle_node.shape[0] != 3:
        triangle_node = triangle_node.T
    n_tri = triangle_node.shape[1]

    # 提取所有边 (每边附带三角形编号和局部边号)
    edge_list = []
    for e in range(n_tri):
        n1, n2, n3 = triangle_node[:, e]
        # 边 1: (n2, n3), 边 2: (n1, n3), 边 3: (n1, n2)
        edge_list.append((min(n2, n3), max(n2, n3), e + 1, 1))
        edge_list.append((min(n1, n3), max(n1, n3), e + 1, 2))
        edge_list.append((min(n1, n2), max(n1, n2), e + 1, 3))

    # 按 (node_a, node_b) 字典序排序
    edge_list.sort(key=lambda x: (x[0], x[1]))

    neighbors = np.zeros((3, n_tri), dtype=np.int64)

    # 遍历排序后的边, 相邻的同键边为邻接三角形
    i = 0
    n_edges = len(edge_list)
    while i < n_edges:
        j = i + 1
        while j < n_edges and edge_list[j][0] == edge_list[i][0] and edge_list[j][1] == edge_list[i][1]:
            j += 1
        # 边 i..j-1 是相同的键
        if j - i == 2:
            _, _, tri_a, side_a = edge_list[i]
            _, _, tri_b, side_b = edge_list[j-1]
            neighbors[side_a - 1, tri_a - 1] = tri_b
            neighbors[side_b - 1, tri_b - 1] = tri_a
        # j-i > 2 为非流形边, 不处理
        i = j

    return neighbors


def structured_triangle_mesh(nx, ny, Lx=1.0, Ly=1.0):
    """
    在矩形区域 [0,Lx] x [0,Ly] 上生成结构化三角形网格。

    每个矩形单元切分为 2 个三角形 (对角线方向一致)。

    返回:
        node_xy: [2, n_node] 节点坐标
        triangle_node: [3, n_tri] 三角形节点编号 (1-indexed)
    """
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    node_xy = np.vstack([X.flatten(), Y.flatten()])

    # 三角形生成
    triangles = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            n1 = i * ny + j + 1        # (i, j)
            n2 = (i + 1) * ny + j + 1  # (i+1, j)
            n3 = (i + 1) * ny + (j + 1) + 1  # (i+1, j+1)
            n4 = i * ny + (j + 1) + 1  # (i, j+1)
            triangles.append([n1, n2, n3])
            triangles.append([n1, n3, n4])

    triangle_node = np.array(triangles, dtype=np.int64).T
    return node_xy, triangle_node


def grid_quality_metrics(node_xy, triangle_node):
    """
    网格质量评估指标:

    1.  aspect_ratio = 最长边 / 最短边 (理想值: 1)
    2.  min_angle: 最小内角 [deg] (理想: 60)
    3.  area: 三角形面积 (应全为正)

    三角形面积 (叉积):
      A = 0.5 * |(v2-v1) x (v3-v1)|
    """
    triangle_node = np.asarray(triangle_node, dtype=np.int64)
    if triangle_node.shape[0] != 3:
        triangle_node = triangle_node.T
    n_tri = triangle_node.shape[1]

    areas = np.zeros(n_tri)
    aspect_ratios = np.zeros(n_tri)
    min_angles = np.zeros(n_tri)

    for e in range(n_tri):
        n1, n2, n3 = triangle_node[:, e]
        v1 = node_xy[:, n1 - 1]
        v2 = node_xy[:, n2 - 1]
        v3 = node_xy[:, n3 - 1]
        e1 = v2 - v1
        e2 = v3 - v1
        e3 = v3 - v2
        # 2D 叉积
        cross = e1[0]*e2[1] - e1[1]*e2[0]
        areas[e] = 0.5 * abs(cross)
        # 边长
        L1 = np.linalg.norm(e1)
        L2 = np.linalg.norm(e2)
        L3 = np.linalg.norm(e3)
        Lmin = min(L1, L2, L3)
        Lmax = max(L1, L2, L3)
        aspect_ratios[e] = Lmax / max(Lmin, SMALL_NUMBER)
        # 角度
        if L1 > SMALL_NUMBER and L2 > SMALL_NUMBER and L3 > SMALL_NUMBER:
            cos_A = (L1**2 + L2**2 - L3**2) / (2*L1*L2)
            cos_B = (L1**2 + L3**2 - L2**2) / (2*L1*L3)
            cos_C = (L2**2 + L3**2 - L1**2) / (2*L2*L3)
            cos_A = np.clip(cos_A, -1, 1)
            cos_B = np.clip(cos_B, -1, 1)
            cos_C = np.clip(cos_C, -1, 1)
            angles = np.degrees([np.arccos(cos_A), np.arccos(cos_B), np.arccos(cos_C)])
            min_angles[e] = np.min(angles)

    return {
        'areas': areas,
        'aspect_ratios': aspect_ratios,
        'min_angles': min_angles,
        'min_area': np.min(areas),
        'max_aspect': np.max(aspect_ratios),
        'min_angle_deg': np.min(min_angles),
    }
