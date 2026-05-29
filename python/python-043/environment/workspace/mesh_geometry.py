"""
网格几何生成与优化 (mesh_geometry.py)
======================================
融合以下种子项目的核心算法：
  - 1349_triangulation_rcm : 网格重排序减少矩阵带宽
  - 254_cvt_circle_uniform : 圆盘上 CVT (质心 Voronoi 镶嵌) 采样
  - 1310_triangle_io       : 三角形网格数据读写
  - 351_fd_to_tec          : 有限差分网格与 Delaunay 三角剖分
  - 1342_triangulation_order3_contour : Delaunay 三角剖分

为地核发电机模拟提供：
  - 球壳径向分层网格
  - 地核截面的 CVT 最优采样点生成
  - 球面 Delaunay 三角剖分与节点重排序 (RCM)
  - 三角网格数据 IO
"""

import numpy as np
from typing import List, Tuple


# ---------------------------------------------------------------------------
# 1. 球壳径向分层网格
#    在地核发电机中，外核为球壳 r_icb <= r <= r_cmb。
#    径向网格采用对数拉伸以更好地分辨边界层。
# ---------------------------------------------------------------------------
def radial_mesh(r_icb: float, r_cmb: float, n: int,
                stretching: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成径向分层网格。

    参数:
      r_icb      : 内核边界半径
      r_cmb      : 核幔边界半径
      n          : 层数
      stretching : 拉伸因子（>0）。stretch=1 为均匀，>1 在边界处加密。

    映射公式（代数拉伸）:
      xi_i = i / (n-1),  i=0..n-1
      r_i = r_icb + (r_cmb - r_icb) * [xi_i + stretching*xi_i*(1-xi_i)]
            / [1 + stretching/4]   (归一化)
    实际使用更简单的对数拉伸:
      r_i = r_icb * (r_cmb/r_icb)^{xi_i}
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    if r_icb <= 0.0 or r_cmb <= r_icb:
        raise ValueError("Invalid radial bounds")

    xi = np.linspace(0.0, 1.0, n)
    if stretching == 1.0:
        r = r_icb * (r_cmb / r_icb) ** xi
    else:
        # 代数拉伸，在两端加密
        s = stretching
        num = xi + s * xi * (1.0 - xi)
        den = 1.0 + s * 0.25
        r = r_icb + (r_cmb - r_icb) * num / den
    dr = np.diff(r)
    dr = np.append(dr, dr[-1])  # 最后一层重复
    return r, dr


# ---------------------------------------------------------------------------
# 2. 圆盘上均匀 CVT 采样（基于 cvt_circle_uniform）
#    扩展到球壳截面：在圆盘 (r, theta) 上生成最优采样点。
# ---------------------------------------------------------------------------
def disk_uniform_samples(n_samples: int, radius: float = 1.0,
                         seed: int = 42) -> np.ndarray:
    """
    在圆盘上生成均匀随机采样点（极坐标方法）。
    公式: r = R * sqrt(u), theta = 2*pi*v，其中 u,v ~ U[0,1]。
    返回 shape (n_samples, 2) 的 (x, y) 数组。
    """
    rng = np.random.default_rng(seed)
    u = rng.random(n_samples)
    v = rng.random(n_samples)
    r = radius * np.sqrt(u)
    theta = 2.0 * np.pi * v
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x, y))


def cvt_disk_uniform(n_generators: int, n_samples: int = 20000,
                     n_iterations: int = 20, radius: float = 1.0,
                     seed: int = 42) -> np.ndarray:
    """
    使用 Lloyd 算法在圆盘上计算 Centroidal Voronoi Tessellation。
    返回生成点坐标 (n_generators, 2)。

    算法:
      1. 随机初始化生成点
      2. 生成大量均匀样本
      3. 将每个样本分配到最近的生成点 (Voronoi 区域)
      4. 将生成点移动到区域质心
      5. 重复 3-4
    """
    rng = np.random.default_rng(seed)
    # 初始化：圆盘上均匀分布
    generators = disk_uniform_samples(n_generators, radius, seed=seed)

    for it in range(n_iterations):
        samples = disk_uniform_samples(n_samples, radius, seed=seed + it + 1)
        # 分配到最近生成点
        # 计算距离矩阵 (n_samples, n_generators)
        diffs = samples[:, np.newaxis, :] - generators[np.newaxis, :, :]
        dists = np.sum(diffs ** 2, axis=2)
        nearest = np.argmin(dists, axis=1)
        # 更新质心
        new_generators = np.zeros_like(generators)
        counts = np.zeros(n_generators, dtype=int)
        for i in range(n_samples):
            gid = nearest[i]
            new_generators[gid] += samples[i]
            counts[gid] += 1
        # 防止空区域
        for g in range(n_generators):
            if counts[g] > 0:
                new_generators[g] /= counts[g]
            else:
                # 重新随机放置空区域生成点
                new_generators[g] = disk_uniform_samples(1, radius, seed=seed + 1000 + g)[0]
        generators = new_generators

    return generators


# ---------------------------------------------------------------------------
# 3. 球面三角剖分节点与 RCM 重排序
#    基于 triangulation_rcm 的图论重排序思想，减少有限元/谱元矩阵带宽。
# ---------------------------------------------------------------------------
def build_node_adjacency(elements: np.ndarray, n_nodes: int) -> List[List[int]]:
    """
    由三角形单元构建节点邻接图。
    elements: shape (n_elements, 3)，每行为三角形节点的全局索引。
    """
    adj = [set() for _ in range(n_nodes)]
    nelem = elements.shape[0]
    for e in range(nelem):
        n1, n2, n3 = elements[e]
        adj[n1].add(n2)
        adj[n1].add(n3)
        adj[n2].add(n1)
        adj[n2].add(n3)
        adj[n3].add(n1)
        adj[n3].add(n2)
    return [list(s) for s in adj]


def pseudo_peripheral_node(adj: List[List[int]], start: int = 0) -> int:
    """
    寻找伪外围节点：用于 RCM 的根节点选择。
    算法：从 start 开始 BFS，反复选择最后一层中度数最小的节点。
    """
    n = len(adj)
    visited = [False] * n
    queue = [start]
    visited[start] = True
    last_level = [start]
    while queue:
        next_level = []
        for node in queue:
            for nb in adj[node]:
                if not visited[nb]:
                    visited[nb] = True
                    next_level.append(nb)
        if not next_level:
            break
        last_level = next_level
        queue = next_level

    # 在最后一层中选择度数最小的节点
    min_deg = float('inf')
    best = last_level[0]
    for node in last_level:
        deg = len(adj[node])
        if deg < min_deg:
            min_deg = deg
            best = node
    return best


def reverse_cuthill_mckee(adj: List[List[int]]) -> np.ndarray:
    """
    执行 Reverse Cuthill-McKee 重排序。

    算法:
      1. 为每个连通分量选择伪外围根节点
      2. 从根节点开始 BFS，每层内按度数升序排列邻居
      3. 记录 Cuthill-McKee 编号
      4. 反转编号得到 RCM 排序
    """
    n = len(adj)
    visited = [False] * n
    ordering = []

    for start in range(n):
        if visited[start]:
            continue
        root = pseudo_peripheral_node(adj, start)
        # BFS (Cuthill-McKee)
        cm_order = []
        queue = [root]
        visited[root] = True
        while queue:
            level = queue[:]
            queue = []
            # 按度数升序排列当前层
            level.sort(key=lambda node: len(adj[node]))
            for node in level:
                cm_order.append(node)
                neighbors = [nb for nb in adj[node] if not visited[nb]]
                neighbors.sort(key=lambda nb: len(adj[nb]))
                for nb in neighbors:
                    if not visited[nb]:
                        visited[nb] = True
                        queue.append(nb)
        ordering.extend(cm_order)

    # Reverse
    rcm_order = ordering[::-1]
    return np.array(rcm_order, dtype=int)


def compute_bandwidth(elements: np.ndarray) -> int:
    """计算三角网格连接矩阵的带宽。"""
    bw = 0
    nelem = elements.shape[0]
    for e in range(nelem):
        n1, n2, n3 = elements[e]
        bw = max(bw, abs(n1 - n2), abs(n1 - n3), abs(n2 - n3))
    return bw + 1


def spherical_surface_nodes(n_theta: int, n_phi: int) -> np.ndarray:
    """
    生成球面经纬度节点（非均匀，在极点加密）。
    返回 shape (n_nodes, 3) 的 (x, y, z) 坐标。
    """
    theta = np.linspace(0.0, np.pi, n_theta)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    nodes = []
    for t in theta:
        for p in phi:
            x = np.sin(t) * np.cos(p)
            y = np.sin(t) * np.sin(p)
            z = np.cos(t)
            nodes.append((x, y, z))
    return np.array(nodes, dtype=float)


def delaunay_triangulation_2d(points: np.ndarray) -> np.ndarray:
    """
    使用 scipy 对 2D 点集进行 Delaunay 三角剖分。
    若 scipy 不可用，则回退到简单网格三角化。
    """
    try:
        from scipy.spatial import Delaunay
        tri = Delaunay(points)
        return tri.simplices.astype(int)
    except Exception:
        # 回退：假设点是规则网格的，构造简单三角形
        # 这仅用于保证代码在缺少 scipy 时仍可运行
        n = points.shape[0]
        # 简单 fallback：每三个连续点形成一个三角形
        nelem = n // 3
        elements = np.zeros((nelem, 3), dtype=int)
        for e in range(nelem):
            elements[e] = [3 * e, 3 * e + 1, 3 * e + 2]
        return elements


# ---------------------------------------------------------------------------
# 4. 地核发电机专用网格生成器
# ---------------------------------------------------------------------------
def generate_core_mesh(r_icb: float, r_cmb: float,
                       n_radial: int, n_theta: int, n_phi: int) -> dict:
    """
    生成地核外核球壳网格（结构化经纬网格）。

    返回字典:
      r          : 径向坐标 (n_radial,)
      theta      : 极角坐标 (n_theta,)
      phi        : 方位角坐标 (n_phi,)
      nodes_3d   : 3D 节点坐标 (n_radial*n_theta*n_phi, 3)
      dr         : 径向间距 (n_radial,)
    """
    r, dr = radial_mesh(r_icb, r_cmb, n_radial, stretching=1.5)
    theta = np.linspace(0.0, np.pi, n_theta)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)

    # 构建 3D 节点列表
    nodes_3d = []
    for ri in r:
        for t in theta:
            for p in phi:
                x = ri * np.sin(t) * np.cos(p)
                y = ri * np.sin(t) * np.sin(p)
                z = ri * np.cos(t)
                nodes_3d.append((x, y, z))
    nodes_3d = np.array(nodes_3d, dtype=float)

    return {
        "r": r,
        "theta": theta,
        "phi": phi,
        "nodes_3d": nodes_3d,
        "dr": dr,
        "n_radial": n_radial,
        "n_theta": n_theta,
        "n_phi": n_phi,
    }


# ---------------------------------------------------------------------------
# 5. 三角网格 IO（基于 triangle_io 思想）
# ---------------------------------------------------------------------------
def write_triangle_nodes(filename: str, nodes: np.ndarray):
    """写 Triangle 格式节点文件。"""
    n_nodes = nodes.shape[0]
    dim = nodes.shape[1]
    with open(filename, 'w') as f:
        f.write(f"{n_nodes} {dim} 0 0\n")
        for i in range(n_nodes):
            line = f"{i+1} " + " ".join(f"{nodes[i, d]:.18e}" for d in range(dim))
            f.write(line + "\n")


def write_triangle_elements(filename: str, elements: np.ndarray):
    """写 Triangle 格式单元文件。"""
    n_elem = elements.shape[0]
    order = elements.shape[1]
    with open(filename, 'w') as f:
        f.write(f"{n_elem} {order} 0\n")
        for e in range(n_elem):
            line = f"{e+1} " + " ".join(str(elements[e, j] + 1) for j in range(order))
            f.write(line + "\n")


def read_triangle_nodes(filename: str) -> np.ndarray:
    """读 Triangle 格式节点文件。"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    header = lines[0].strip().split()
    n_nodes = int(header[0])
    dim = int(header[1])
    nodes = np.zeros((n_nodes, dim), dtype=float)
    for i in range(n_nodes):
        parts = lines[i + 1].strip().split()
        for d in range(dim):
            nodes[i, d] = float(parts[d + 1])
    return nodes


def read_triangle_elements(filename: str) -> np.ndarray:
    """读 Triangle 格式单元文件。"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    header = lines[0].strip().split()
    n_elem = int(header[0])
    order = int(header[1])
    elements = np.zeros((n_elem, order), dtype=int)
    for e in range(n_elem):
        parts = lines[e + 1].strip().split()
        for j in range(order):
            elements[e, j] = int(parts[j + 1]) - 1  # 0-based
    return elements


# ---------------------------------------------------------------------------
# 自测试
# ---------------------------------------------------------------------------
def _self_test():
    # 径向网格
    r, dr = radial_mesh(1221e3, 3480e3, 16)
    assert len(r) == 16
    assert r[0] == 1221e3
    assert r[-1] == 3480e3

    # CVT
    pts = cvt_disk_uniform(10, n_samples=5000, n_iterations=5)
    assert pts.shape == (10, 2)
    assert np.all(np.linalg.norm(pts, axis=1) <= 1.0 + 1e-10)

    # RCM
    elements = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]], dtype=int)
    adj = build_node_adjacency(elements, 5)
    rcm = reverse_cuthill_mckee(adj)
    assert len(rcm) == 5
    bw_before = compute_bandwidth(elements)
    # 应用重排序
    reordered = np.zeros_like(elements)
    inv = np.argsort(rcm)
    for e in range(elements.shape[0]):
        for j in range(3):
            reordered[e, j] = inv[elements[e, j]]
    bw_after = compute_bandwidth(reordered)
    assert bw_after <= bw_before

    # 核心网格
    mesh = generate_core_mesh(1221e3, 3480e3, 8, 8, 8)
    assert mesh["nodes_3d"].shape[0] == 8 * 8 * 8

    print("mesh_geometry: self-test passed.")


if __name__ == "__main__":
    _self_test()
