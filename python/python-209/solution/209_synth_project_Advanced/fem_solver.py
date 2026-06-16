"""
fem_solver.py — 确定性椭圆 PDE 有限元求解器
==============================================

本模块实现随机参数椭圆 PDE 的确定性求解器:
    -∇·(a(x)∇u(x)) = f(x)  in Ω = [0,1]²
    u(x) = 0                 on ∂Ω

核心方法:
  1. 结构化三角网格生成
  2. P1 有限元离散化
  3. 稀疏刚度矩阵组装
  4. 共轭梯度法 (PCG) 求解

数学框架:
  弱形式: ∫_Ω a∇u·∇v dx = ∫_Ω fv dx  ∀v ∈ H¹₀(Ω)
  离散:   K(ξ)·U(ξ) = F
  其中 K_{ij}(ξ) = ∫_Ω a(x,ξ)∇φᵢ·∇φⱼ dx

映射种子项目:
  - 865_percolation_simulation: 渗流拓扑分析随机介质连通性
  - 190_closest_pair_brute: 最近点对搜索优化节点分布
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import cg


# ============================================================
# 第1部分: 结构化三角网格
# ============================================================

def generate_triangular_mesh(nx, ny, domain=(0.0, 1.0, 0.0, 1.0)):
    """
    在矩形域 [x0,x1]×[y0,y1] 上生成结构化三角网格。
    每个矩形单元剖分为两个三角形 (对角线方向交替)。

    返回:
        nodes: (N_nodes, 2) 节点坐标
        elements: (N_elems, 3) 三角形节点索引 (逆时针)
        boundary_nodes: 边界节点索引列表
    """
    x0, x1, y0, y1 = domain
    x = np.linspace(x0, x1, nx + 1)
    y = np.linspace(y0, y1, ny + 1)
    X, Y = np.meshgrid(x, y)
    nodes = np.column_stack([X.ravel(), Y.ravel()])

    elements = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n2 = n0 + (nx + 1)
            n3 = n2 + 1
            if (i + j) % 2 == 0:
                elements.append([n0, n1, n2])
                elements.append([n1, n3, n2])
            else:
                elements.append([n0, n1, n3])
                elements.append([n0, n3, n2])
    elements = np.array(elements, dtype=int)

    # 边界节点: x=x0, x=x1, y=y0, y=y1
    tol = 1e-10
    boundary = set()
    for idx, (px, py) in enumerate(nodes):
        if abs(px - x0) < tol or abs(px - x1) < tol or \
           abs(py - y0) < tol or abs(py - y1) < tol:
            boundary.add(idx)
    return nodes, elements, sorted(boundary)


def compute_triangle_geometry(nodes, element):
    """
    计算三角形几何量:
      - 面积: A = |det([x1-x0, x2-x0; y1-y0, y2-y0])|/2
      - 梯度: ∇φᵢ = (1/(2A)) [y_{i+1}-y_{i+2}; x_{i+2}-x_{i+1}]

    返回:
        area: 三角形面积
        grad_phi: (3, 2) 三个基函数的梯度
    """
    i0, i1, i2 = element
    x0, y0 = nodes[i0]
    x1, y1 = nodes[i1]
    x2, y2 = nodes[i2]
    area = 0.5 * abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
    if area < 1e-15:
        return 0.0, np.zeros((3, 2))
    grad = np.zeros((3, 2))
    grad[0] = [y1 - y2, x2 - x1] / (2.0 * area)
    grad[1] = [y2 - y0, x0 - x2] / (2.0 * area)
    grad[2] = [y0 - y1, x1 - x0] / (2.0 * area)
    return area, grad


# ============================================================
# 第2部分: 刚度矩阵组装
# ============================================================

def assemble_stiffness_matrix(nodes, elements, diffusivity_func, boundary_nodes):
    """
    组装刚度矩阵:
        K_{ij} = ∫_Ω a(x)∇φᵢ·∇φⱼ dx ≈ Σ_T a(x_T) (∇φᵢ·∇φⱼ) |T|

    参数:
        nodes: (N, 2) 节点
        elements: (M, 3) 单元
        diffusivity_func: 扩散系数函数 a(x) → 标量
        boundary_nodes: 边界节点列表

    返回:
        K: (N, N) 稀疏刚度矩阵
        F: (N,) 载荷向量
    """
    N = len(nodes)
    K = lil_matrix((N, N))
    F = np.zeros(N)

    for elem in elements:
        area, grad = compute_triangle_geometry(nodes, elem)
        if area < 1e-15:
            continue
        # 单元中心
        center = np.mean(nodes[elem], axis=0)
        a_val = diffusivity_func(center)
        # 单元刚度矩阵: K^e_{ij} = a · (∇φᵢ·∇φⱼ) · |T|
        K_e = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                K_e[i, j] = a_val * np.dot(grad[i], grad[j]) * area
        # 组装到全局
        for i_loc in range(3):
            for j_loc in range(3):
                i_glob = elem[i_loc]
                j_glob = elem[j_loc]
                K[i_glob, j_glob] += K_e[i_loc, j_loc]

    # 载荷向量: f(x) = 1 (均匀源)
    for elem in elements:
        area, _ = compute_triangle_geometry(nodes, elem)
        if area < 1e-15:
            continue
        for i_loc in range(3):
            F[elem[i_loc]] += area / 3.0

    K = csr_matrix(K)
    return K, F


def apply_dirichlet_bc(K, F, boundary_nodes, bc_value=0.0):
    """
    施加 Dirichlet 边界条件 (大数法):
        K_{ii} → α (大数)
        F_i → α · bc_value
    """
    K = K.tolil()
    alpha = 1.0e30
    for idx in boundary_nodes:
        K[idx, :] = 0.0
        K[:, idx] = 0.0
        K[idx, idx] = alpha
        F[idx] = alpha * bc_value
    return csr_matrix(K), F


# ============================================================
# 第3部分: 线性系统求解
# ============================================================

def solve_linear_system(K, F, tol=1e-8, max_iter=5000):
    """
    共轭梯度法 (PCG) 求解:
        K·U = F

    误差估计: ||r||/||b|| < tol
    返回:
        U: (N,) 解向量
        info: 收敛标志 (0=成功)
        residual_norm: 最终残差范数
    """
    U, info = cg(K, F, rtol=tol, maxiter=max_iter)
    residual = np.linalg.norm(K @ U - F)
    return U, info, residual


# ============================================================
# 第4部分: 渗流拓扑分析
# (映射自 865_percolation_simulation)
# ============================================================

def percolation_analysis(diffusivity_field, nodes, elements, threshold=0.5):
    """
    渗流理论分析随机介质的连通性:
        1. 根据阈值将扩散系数分为"高导"和"低导"区域
        2. BFS 洪水填充识别连通分量
        3. 判断是否存在贯穿渗流通路

    物理含义:
        若高导区域连通, PDE 解呈现通道效应;
        若低导区域连通, 解呈现屏障效应;
        渗流阈值 p_c ≈ 0.593 (二维正方格点渗流)。

    返回:
        n_components: 连通分量数
        largest_size: 最大分量尺寸
        percolates: 是否渗流贯穿
        component_sizes: 各分量尺寸列表
    """
    N = len(nodes)
    # 节点分类: 高于阈值为"导通"
    occupied = np.zeros(N, dtype=bool)
    for idx in range(N):
        if idx < len(diffusivity_field):
            occupied[idx] = diffusivity_field[idx] > threshold

    # BFS 洪水填充
    visited = np.zeros(N, dtype=bool)
    labels = -np.ones(N, dtype=int)
    component_id = 0
    component_sizes = []

    for start in range(N):
        if not occupied[start] or visited[start]:
            continue
        # BFS
        queue = [start]
        visited[start] = True
        size = 0
        while queue:
            node = queue.pop(0)
            labels[node] = component_id
            size += 1
            # 4邻域 (基于节点索引的邻近关系)
            for neighbor in _get_neighbors(node, nodes, elements):
                if not visited[neighbor] and occupied[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        component_sizes.append(size)
        component_id += 1

    # 渗流判断: 检查最大分量是否连接左右边界
    percolates = False
    if component_sizes:
        largest_label = np.argmax(component_sizes)
        left_connected = any(labels[i] == largest_label
                             for i in range(N) if abs(nodes[i][0]) < 1e-10)
        right_connected = any(labels[i] == largest_label
                              for i in range(N) if abs(nodes[i][0] - 1.0) < 1e-10)
        percolates = left_connected and right_connected

    return (component_id,
            max(component_sizes) if component_sizes else 0,
            percolates,
            component_sizes)


def _get_neighbors(node_idx, nodes, elements, radius_factor=1.5):
    """获取节点的邻近节点 (基于单元连接性)"""
    x, y = nodes[node_idx]
    # 计算平均单元尺寸
    avg_h = 1.0 / max(1, int(np.sqrt(len(nodes))))
    r2 = (radius_factor * avg_h) ** 2
    neighbors = []
    for elem in elements:
        if node_idx in elem:
            for n_idx in elem:
                if n_idx != node_idx and n_idx not in neighbors:
                    dx = nodes[n_idx][0] - x
                    dy = nodes[n_idx][1] - y
                    if dx * dx + dy * dy <= r2:
                        neighbors.append(n_idx)
    return neighbors


# ============================================================
# 第5部分: 最近点对搜索
# (映射自 190_closest_pair_brute)
# ============================================================

def closest_pair_brute(nodes):
    """
    暴力法最近点对:
        d_min = min_{i<j} ||x_i - x_j||
    返回: (d_min, i, j)
    用途: 检测网格质量 (最小边长)
    """
    n = len(nodes)
    d_min = float('inf')
    best_i, best_j = 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = nodes[i][0] - nodes[j][0]
            dy = nodes[i][1] - nodes[j][1]
            d2 = dx * dx + dy * dy
            if d2 < d_min:
                d_min = d2
                best_i, best_j = i, j
    return np.sqrt(d_min), best_i, best_j


def mesh_quality_indicator(nodes, elements):
    """
    网格质量指标:
        - 最小边长 (最近点对)
        - 最大面积比 (最大/最小三角形面积)
        - 最小角度估计
    """
    min_dist, _, _ = closest_pair_brute(nodes[:min(200, len(nodes))])
    areas = []
    for elem in elements:
        area, _ = compute_triangle_geometry(nodes, elem)
        areas.append(area)
    areas = np.array(areas)
    areas_pos = areas[areas > 1e-15]
    if len(areas_pos) > 0:
        area_ratio = np.max(areas_pos) / max(np.min(areas_pos), 1e-15)
    else:
        area_ratio = float('inf')
    return {
        'min_edge_length': min_dist,
        'area_ratio': area_ratio,
        'n_elements': len(elements),
        'n_nodes': len(nodes),
        'mean_area': np.mean(areas_pos) if len(areas_pos) > 0 else 0.0
    }


# ============================================================
# 第6部分: 完整随机 PDE 求解流程
# ============================================================

def solve_stochastic_pde_realization(nx, ny, diffusivity_samples,
                                     domain=(0.0, 1.0, 0.0, 1.0)):
    """
    求解单次随机 PDE 实现:
        -∇·(a(x,ω)∇u) = f, u|∂Ω=0

    参数:
        nx, ny: 网格密度
        diffusivity_samples: (N_nodes,) 扩散系数场采样
        domain: 计算域

    返回:
        solution: (N_nodes,) PDE 解
        quality: 网格质量字典
    """
    nodes, elements, boundary = generate_triangular_mesh(nx, ny, domain)
    quality = mesh_quality_indicator(nodes, elements)

    # 将扩散系数插值到节点
    if len(diffusivity_samples) >= len(nodes):
        a_field = diffusivity_samples[:len(nodes)]
    else:
        a_field = np.ones(len(nodes))
        a_field[:len(diffusivity_samples)] = diffusivity_samples

    def diffusivity_func(x):
        # 最近邻插值
        dists = np.sum((nodes - x) ** 2, axis=1)
        nearest = np.argmin(dists)
        return max(a_field[nearest], 1e-6)

    K, F = assemble_stiffness_matrix(nodes, elements, diffusivity_func, boundary)
    K, F = apply_dirichlet_bc(K, F, boundary)
    U, info, residual = solve_linear_system(K, F)

    # 渗流分析
    median_a = np.median(a_field)
    n_comp, largest, percolates, sizes = percolation_analysis(a_field, nodes, elements, median_a)

    return {
        'solution': U,
        'nodes': nodes,
        'elements': elements,
        'quality': quality,
        'solver_info': info,
        'residual': residual,
        'percolation': {
            'n_components': n_comp,
            'largest_size': largest,
            'percolates': percolates,
            'sizes': sizes
        }
    }
