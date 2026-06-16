"""
lattice_geometry.py -  Hubbard 模型的晶格几何与布里渊区离散化
==============================================================

科学背景 (Scientific Background):
    强关联 Hubbard 模型的哈密顿量定义在离散晶格上:
        H = -t Σ_{<i,j>,σ} c†_{iσ} c_{jσ} + U Σ_i n_{i↑} n_{i↓} - μ Σ_i (n_{i↑}+n_{i↓})

    其中 t 为近邻跳跃积分, U 为在位库仑排斥, μ 为化学势.
    本项目聚焦 **三角晶格** (triangular lattice) —— 几何阻挫系统的原型.
    三角晶格的阻挫导致自旋液体、超导等非平庸量子态.

本模块融合的种子项目:
    - 1408_wedge_grid:     楔形体网格 → 三维布里渊区棱柱采样
    - 293_disk_grid:       圆盘网格 → 二维费米面圆盘采样 (Fibonacci 网格)
    - 1320_triangle_to_fem: 三角网格 → 实空间三角晶格 FEM 节点映射

核心公式 (Key Formulas):
    三角晶格基矢:
        a1 = a (1, 0),  a2 = a (1/2, √3/2)

    倒格矢 (reciprocal vectors):
        b1 = (2π/a)(1, -1/√3),  b2 = (2π/a)(0, 2/√3)

    紧束缚色散关系 (tight-binding dispersion):
        ε(k) = -2t [cos(k·a1) + cos(k·a2) + cos(k·(a1-a2))]

    范霍夫奇点 (van Hove singularity) 位于:
        k_vH = (π/a)(1, 1/√3) → ε_vH = t  (半填充时 μ = U/2)
"""

import numpy as np
from typing import Tuple, List, Optional


# ==========================================================================
#  物理常数与晶格参数
# ==========================================================================

def triangular_lattice_vectors(a: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回三角晶格实空间基矢.

    数学公式:
        a1 = a * (1, 0)
        a2 = a * (1/2, sqrt(3)/2)

    参数:
        a: 晶格常数 (默认 1.0, 以 t=1 为单位)

    返回:
        (a1, a2): 两个基矢, 形状 (2,)
    """
    a1 = np.array([a, 0.0])
    a2 = np.array([a * 0.5, a * np.sqrt(3.0) / 2.0])
    return a1, a2


def reciprocal_vectors(a: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回三角晶格倒格矢.

    由 a_i · b_j = 2π δ_{ij} 解出:
        b1 = (2π/a) * (1, -1/sqrt(3))
        b2 = (2π/a) * (0,  2/sqrt(3))
    """
    c = 2.0 * np.pi / a
    b1 = np.array([c, -c / np.sqrt(3.0)])
    b2 = np.array([0.0, 2.0 * c / np.sqrt(3.0)])
    return b1, b2


def first_brizouin_corner_points(a: float = 1.0) -> dict:
    """
    返回第一布里渊区高对称点 (Γ, K, K', M).

    三角晶格的第一布里渊区为正六边形.
    高对称点:
        Γ = (0, 0)
        K = (4π/3a, 0)
        K' = (2π/3a, 2π/(a√3))
        M = (π/a, π/(a√3))
    """
    b1, b2 = reciprocal_vectors(a)
    Gamma = np.array([0.0, 0.0])
    K = (b1 + b2) / 3.0
    Kp = (2.0 * b1 + b2) / 3.0
    M = (b1 + b2) / 2.0
    return {'Gamma': Gamma, 'K': K, 'Kp': Kp, 'M': M}


# ==========================================================================
#  实空间三角晶格构造 (融合 1320_triangle_to_fem)
# ==========================================================================

def build_triangular_cluster(Lx: int, Ly: int,
                             a: float = 1.0,
                             periodic: bool = True
                             ) -> dict:
    """
    构造 Lx × Ly 三角晶格有限簇 (cluster).

    融合种子项目 1320_triangle_to_fem 的思想: 将三角网格的
    node-element 描述转换为紧束缚模型所需的格点-跳跃表.

    在 DQMC 中, 我们处理有限尺寸簇 (通常 4×4, 6×6, 8×8).
    总格点数 Ns = Lx * Ly.

    返回 dict:
        'positions': (Ns, 2) 格点坐标
        'hopping':   list of (i, j, t_ij) 跳跃连接
        'adjacency': (Ns, Ns) 邻接矩阵
        'Ns':        总格点数
        'Lx', 'Ly':  簇尺寸
    """
    a1, a2 = triangular_lattice_vectors(a)
    Ns = Lx * Ly
    positions = np.zeros((Ns, 2))
    adjacency = np.zeros((Ns, Ns), dtype=int)
    hopping = []

    # 格点索引: i = ix + iy * Lx
    for iy in range(Ly):
        for ix in range(Lx):
            idx = ix + iy * Lx
            pos = ix * a1 + iy * a2
            positions[idx] = pos

    # 最近邻跳跃 (三角晶格配位数 z = 6)
    neighbor_shifts = [
        (1, 0), (0, 1), (-1, 1),    # 正方向
        (-1, 0), (0, -1), (1, -1),  # 负方向
    ]
    for iy in range(Ly):
        for ix in range(Lx):
            i = ix + iy * Lx
            for dx, dy in neighbor_shifts:
                if periodic:
                    jx = ix + dx
                    jy = iy + dy
                    # 对三角晶格的非正交周期性做修正
                    # 沿 a2 方向平移时, x 方向有半个基矢偏移
                    if jy < 0 or jy >= Ly:
                        if periodic:
                            jy_mod = jy % Ly
                            # 三角晶格斜向周期性: 跨 Ly 边界时 x 偏移 Ly/2
                            jx_mod = (jx + (jy // Ly) * (Ly // 2)) % Lx
                        else:
                            continue
                    else:
                        jx_mod = (ix + dx) % Lx if periodic else ix + dx
                        jy_mod = jy
                    if not periodic:
                        if jx_mod < 0 or jx_mod >= Lx:
                            continue
                else:
                    jx_mod = ix + dx
                    jy_mod = iy + dy
                    if jx_mod < 0 or jx_mod >= Lx or jy_mod < 0 or jy_mod >= Ly:
                        continue
                j = (jx_mod % Lx) + (jy_mod % Ly) * Lx
                if i != j and adjacency[i, j] == 0:
                    adjacency[i, j] = 1
                    adjacency[j, i] = 1
                    hopping.append((i, j, -1.0))  # t = 1 为单位

    return {
        'positions': positions,
        'hopping': hopping,
        'adjacency': adjacency,
        'Ns': Ns,
        'Lx': Lx,
        'Ly': Ly,
    }


def build_kinetic_matrix(cluster: dict, a: float = 1.0) -> np.ndarray:
    """
    在 k 空间或实空间构造动能矩阵 T_{ij}.

    实空间: T 为 Ns×Ns 稀疏矩阵, T_{ij} = -t 若 <i,j> 为近邻.
    本征值即紧束缚能带 ε_n(k).

    对于 DQMC, 我们需要完整的 T 矩阵用于:
        B(τ) = exp(-Δτ * (T + V(τ)))
    其中 V(τ) 为 HS 辅助场构型决定的对角势能矩阵.
    """
    Ns = cluster['Ns']
    T = np.zeros((Ns, Ns))
    for i, j, t_ij in cluster['hopping']:
        T[i, j] += t_ij
    # 确保厄米性
    T = 0.5 * (T + T.T)
    return T


# ==========================================================================
#  二维圆盘布里渊区网格 (融合 293_disk_grid)
# ==========================================================================

def disk_grid_fibonacci(n_points: int, radius: float = 1.0,
                        center: Optional[np.ndarray] = None
                        ) -> np.ndarray:
    """
    用 Fibonacci 螺旋在二维圆盘中生成准均匀网格点.

    融合种子项目 293_disk_grid 的 Fibonacci 圆盘采样方法.
    在布里渊区积分中, 我们需要对费米面附近的圆盘区域做积分:
        ∫_{BZ} d²k f(k) ≈ (1/N_k) Σ_{k_i ∈ disk} w_i f(k_i)

    Fibonacci 网格的优势:
        - 低偏差 (low discrepancy) → 准蒙特卡洛收敛更快
        - 避免方形网格在高对称方向上的假各向异性

    数学构造:
        黄金角 θ_g = π(3 - √5)
        第 i 个点:
            r_i = R √(i/N)
            θ_i = i * θ_g
            (x_i, y_i) = center + r_i (cos θ_i, sin θ_i)
    """
    if center is None:
        center = np.array([0.0, 0.0])
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    indices = np.arange(n_points)
    r = radius * np.sqrt((indices + 0.5) / n_points)
    theta = indices * golden_angle
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    return np.column_stack([x, y])


def disk_grid_count(n_radial: int, n_angular: int) -> int:
    """
    极坐标圆盘网格的总点数.

    N_total = 1 + Σ_{i=1}^{n_radial} (i * n_angular)
            = 1 + n_angular * n_radial * (n_radial + 1) / 2
    """
    return 1 + n_angular * n_radial * (n_radial + 1) // 2


def build_bz_disk_grid(n_k: int, a: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    构造第一布里渊区内的圆盘采样网格.

    返回:
        k_points: (n_k, 2) k 空间坐标
        weights:  (n_k,) 积分权重 (等权 1/n_k)
    """
    # 第一布里渊区内切圆半径 (三角晶格)
    k_inner = 2.0 * np.pi / (a * np.sqrt(3.0))
    pts = disk_grid_fibonacci(n_k, radius=k_inner)
    weights = np.ones(n_k) / n_k
    return pts, weights


# ==========================================================================
#  三维楔形体布里渊区网格 (融合 1408_wedge_grid)
# ==========================================================================

def wedge_grid_size(n: int) -> int:
    """
    三维楔形体网格的总点数.

    楔形体 = 三角形底面 × 一维高度.
    三角形方向点数为 (n+1)(n+2)/2, 高度方向为 (n+1).

    N_total = (n+1) * (n+1)(n+2)/2

    物理应用:
        对于层状 Hubbard 模型 (bilayer / multilayer),
        第三维代表层间耦合 k_z 方向.
    """
    tri_count = (n + 1) * (n + 2) // 2
    return (n + 1) * tri_count


def wedge_grid_3d(n: int) -> np.ndarray:
    """
    在单位楔形体 0≤x, 0≤y, x+y≤1, -1≤z≤1 中生成均匀网格.

    融合种子项目 1408_wedge_grid.
    物理映射: (x,y) → 布里渊区三角底面, z → 层间色散 k_z.
    """
    ng = wedge_grid_size(n)
    g = np.zeros((ng, 3))
    idx = 0
    for iz in range(n + 1):
        z = -1.0 + 2.0 * iz / n if n > 0 else 0.0
        for iy in range(n + 1):
            for ix in range(n + 1 - iy):
                x = ix / n if n > 0 else 0.0
                y = iy / n if n > 0 else 0.0
                g[idx] = [x, y, z]
                idx += 1
    return g


# ==========================================================================
#  紧束缚色散 (Tight-Binding Dispersion)
# ==========================================================================

def tight_binding_dispersion(kx: np.ndarray, ky: np.ndarray,
                             t: float = 1.0, t_prime: float = 0.0,
                             a: float = 1.0) -> np.ndarray:
    """
    三角晶格紧束缚色散 ε(k).

    数学公式 (包含次近邻跳跃 t'):
        ε(k) = -2t [cos(k·a1) + cos(k·a2) + cos(k·(a1-a2))]
               -2t' [cos(k·(a1+a2)) + cos(k·(2a1-a2)) + cos(k·(a1-2a2))]

    对于纯三角晶格 (t'=0):
        ε(k) = -2t [cos(kx*a) + 2 cos(kx*a/2) cos(√3 ky*a/2)]

    范霍夫奇点条件:
        ∇_k ε(k) = 0  →  态密度出现对数发散

    参数:
        t: 最近邻跳跃积分 (能量单位)
        t_prime: 次近邻跳跃积分 (破坏粒子-空穴对称性)
        a: 晶格常数
    """
    arg1 = kx * a
    arg2 = kx * a * 0.5 + ky * a * np.sqrt(3.0) / 2.0
    arg3 = kx * a * 0.5 - ky * a * np.sqrt(3.0) / 2.0
    eps = -2.0 * t * (np.cos(arg1) + np.cos(arg2) + np.cos(arg3))
    if abs(t_prime) > 1e-15:
        # 次近邻: a1+a2, 2a1-a2, a1-2a2
        arg4 = kx * a * 1.5 + ky * a * np.sqrt(3.0) / 2.0
        arg5 = kx * a * 1.5 - ky * a * np.sqrt(3.0) / 2.0
        arg6 = ky * a * np.sqrt(3.0)
        eps -= 2.0 * t_prime * (np.cos(arg4) + np.cos(arg5) + np.cos(arg6))
    return eps


def density_of_states_histogram(eps_array: np.ndarray,
                                n_bins: int = 100,
                                energy_range: Optional[Tuple[float, float]] = None
                                ) -> Tuple[np.ndarray, np.ndarray]:
    """
    由蒙特卡洛采样的 ε(k) 估计态密度 N(ε).

    N(ε) = (1/N_k) Σ_k δ(ε - ε(k))
         ≈ (1/N_k) Σ_k (1/η√π) exp(-(ε-ε(k))²/η²)

    实际用直方图近似.
    """
    if energy_range is None:
        emin, emax = eps_array.min() - 0.1, eps_array.max() + 0.1
    else:
        emin, emax = energy_range
    dos, bin_edges = np.histogram(eps_array, bins=n_bins,
                                  range=(emin, emax), density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return bin_centers, dos


# ==========================================================================
#  自洽化学势求解 (半填充条件)
# ==========================================================================

def chemical_potential_half_filling(temperature: float,
                                    dos_centers: np.ndarray,
                                    dos_values: np.ndarray,
                                    tolerance: float = 1e-8,
                                    max_iter: int = 200) -> float:
    """
    二分法求解半填充 (n=1) 化学势 μ.

    粒子数方程:
        n = ∫ dε N(ε) f(ε, μ, T)
    其中费米-狄拉克分布:
        f(ε, μ, T) = 1 / (exp((ε-μ)/T) + 1)

    半填充条件: n = 1 (每个格点一个电子, 考虑自旋)

    在粒子-空穴对称的三角晶格中, μ = U/2 (严格值).
    当 t' ≠ 0 时, 对称性被破坏, 需要数值求解.
    """
    mu_lo, mu_hi = dos_centers[0] - 5.0, dos_centers[-1] + 5.0
    dE = dos_centers[1] - dos_centers[0] if len(dos_centers) > 1 else 1.0

    for iteration in range(max_iter):
        mu_mid = 0.5 * (mu_lo + mu_hi)
        # 防止数值溢出
        arg = np.clip((dos_centers - mu_mid) / max(temperature, 1e-15), -500, 500)
        fdist = 1.0 / (np.exp(arg) + 1.0)
        n_elec = np.trapz(dos_values * fdist, dos_centers)

        if abs(n_elec - 1.0) < tolerance:
            return mu_mid
        if n_elec < 1.0:
            mu_lo = mu_mid
        else:
            mu_hi = mu_mid
    return mu_mid


# ==========================================================================
#  晶格连通性与图分割 (融合 796_neighbors_to_metis_graph)
# ==========================================================================

def adjacency_to_metis_format(adjacency: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    将邻接矩阵转换为 METIS 图格式.

    融合种子项目 796_neighbors_to_metis_graph.
    METIS 格式用于图分割, 在并行 DQMC 中将晶格分为
    若干子域 (domain), 每个子域由一个 MPI 进程处理.

    METIS 图文件:
        第一行: N  E  (节点数, 边数)
        第 i+1 行: 节点 i 的邻居列表 (1-indexed)

    返回:
        xadj: (N+1,) 压缩行指针
        adjncy: (2E,) 邻接列表
    """
    N = adjacency.shape[0]
    xadj = np.zeros(N + 1, dtype=int)
    adjncy_list = []
    for i in range(N):
        neighbors = np.where(adjacency[i] != 0)[0]
        adjncy_list.extend((neighbors + 1).tolist())  # 1-indexed
        xadj[i + 1] = xadj[i] + len(neighbors)
    adjncy = np.array(adjncy_list, dtype=int)
    return xadj, adjncy


def compute_graph_diameter(adjacency: np.ndarray) -> int:
    """
    BFS 计算图的直径 (最大最短路径长度).

    物理意义: 晶格上两个最远格点间的"量子距离"
    决定了关联函数衰减的特征长度尺度.
    """
    N = adjacency.shape[0]
    max_dist = 0
    for source in range(N):
        dist = np.full(N, -1, dtype=int)
        dist[source] = 0
        queue = [source]
        while queue:
            current = queue.pop(0)
            for neighbor in np.where(adjacency[current] != 0)[0]:
                if dist[neighbor] == -1:
                    dist[neighbor] = dist[current] + 1
                    queue.append(neighbor)
        max_dist = max(max_dist, int(dist.max()))
    return max_dist
