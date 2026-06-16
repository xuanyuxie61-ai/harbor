"""
lattice_geometry.py — 量子晶格几何构造模块

融合种子项目:
  - 256_cvt_corn_movie: Centroidal Voronoi Tessellation (CVT) 用于非均匀晶格位点生成
  - 109_boundary_word_right: Polyabolo边界字表示用于拓扑边界条件
  - 785_naca: NACA翼型共形映射用于晶格变形
  - 185_circles: 圆形域构造用于自旋关联域
  - 314_double_c_data: 双C形数据用于畴壁测试构型

本模块实现:
  1. 正方形/三角/Voronoi晶格的统一构造
  2. 周期性与开放边界条件的拓扑分类
  3. 基于CVT的非均匀晶格位点松弛
  4. 共形映射晶格变形（用于研究几何对量子相变的影响）
  5. 自旋簇域的Voronoi分割
"""

import numpy as np
from typing import Tuple, List, Optional, Dict


# ============================================================================
#  第一部分: 基础晶格构造 (融合 185_circles 的离散圆域概念)
# ============================================================================

def build_square_lattice(Lx: int, Ly: int,
                         boundary: str = 'periodic') -> Dict:
    """
    构造 Lx x Ly 正方形晶格

    物理背景:
        二维横场Ising模型 (TFIM) 的标准晶格几何。
        Hamiltonian: H = -J Σ_{<ij>} σ_i^z σ_j^z - Γ Σ_i σ_i^x

    参数:
        Lx, Ly: 两个方向的晶格尺寸
        boundary: 'periodic' 或 'open' 或 'cylinder'

    返回:
        包含 sites, bonds, neighbors 的字典

    数学表达:
        格点集合 Λ = {(i,j) : 0 ≤ i < Lx, 0 ≤ j < Ly}
        键集合 E = {((i,j),(i',j')) : |i-i'|+|j-j'| = 1 (mod boundary)}

        对于周期边界: (i+Lx) mod Lx, (j+Ly) mod Ly
        配位数 z = 4 (周期), z = 2 or 3 (开放)
    """
    sites = []
    bonds = []
    adjacency = {}

    N = Lx * Ly
    for i in range(Lx):
        for j in range(Ly):
            idx = i * Ly + j
            sites.append((i, j))
            adjacency[idx] = []

    # 构造近邻键
    for i in range(Lx):
        for j in range(Ly):
            idx = i * Ly + j

            # x方向近邻
            if boundary == 'periodic' or boundary == 'cylinder':
                i_next = (i + 1) % Lx
                if boundary == 'cylinder' and i == Lx - 1:
                    pass  # cylinder在x方向开放
                else:
                    idx_next = i_next * Ly + j
                    bonds.append((idx, idx_next))
                    adjacency[idx].append(idx_next)
                    adjacency[idx_next].append(idx)
            elif i + 1 < Lx:
                idx_next = (i + 1) * Ly + j
                bonds.append((idx, idx_next))
                adjacency[idx].append(idx_next)
                adjacency[idx_next].append(idx)

            # y方向近邻
            if boundary == 'periodic':
                j_next = (j + 1) % Ly
                idx_next = i * Ly + j_next
                if j != j_next or Ly > 1:  # 避免Ly=1时重复
                    bonds.append((idx, idx_next))
                    adjacency[idx].append(idx_next)
                    adjacency[idx_next].append(idx)
            elif j + 1 < Ly:
                idx_next = i * Ly + (j + 1)
                bonds.append((idx, idx_next))
                adjacency[idx].append(idx_next)
                adjacency[idx_next].append(idx)

    # 计算配位数统计
    coord_numbers = [len(adjacency[s]) for s in range(N)]
    z_mean = np.mean(coord_numbers)
    z_var = np.var(coord_numbers)

    # 结构因子所需的倒格矢
    # G = (2π/Lx * nx, 2π/Ly * ny), nx ∈ [-Lx/2, Lx/2), ny ∈ [-Ly/2, Ly/2)
    qx = np.fft.fftfreq(Lx, d=1.0) * 2 * np.pi
    qy = np.fft.fftfreq(Ly, d=1.0) * 2 * np.pi

    return {
        'Lx': Lx, 'Ly': Ly, 'N': N,
        'sites': sites, 'bonds': bonds,
        'adjacency': adjacency,
        'coord_numbers': coord_numbers,
        'z_mean': z_mean, 'z_var': z_var,
        'boundary': boundary,
        'qx': qx, 'qy': qy,
    }


def build_triangular_lattice(L: int, boundary: str = 'periodic') -> Dict:
    """
    构造 L x L 三角晶格 (用于研究几何阻挫对量子相变的影响)

    物理背景:
        三角晶格上的反铁磁Ising模型具有宏观基态简并,
        引入横场后呈现丰富的量子相图。

        配位数 z = 6 (周期边界)

    晶格矢量:
        a₁ = (1, 0), a₂ = (1/2, √3/2)
        倒格矢: b₁ = 2π(1, -1/√3), b₂ = 2π(0, 2/√3)
    """
    sites = []
    bonds = []
    adjacency = {}

    N = L * L
    for i in range(L):
        for j in range(L):
            idx = i * L + j
            x = i + 0.5 * j
            y = (np.sqrt(3) / 2) * j
            sites.append((x, y))
            adjacency[idx] = []

    for i in range(L):
        for j in range(L):
            idx = i * L + j

            # 三个方向的近邻
            neighbors_raw = [
                ((i + 1) % L, j),
                (i, (j + 1) % L),
                ((i - 1) % L, (j + 1) % L),  # 对角方向
            ]

            for ni, nj in neighbors_raw:
                if boundary == 'periodic':
                    nidx = ni * L + nj
                else:
                    if 0 <= ni < L and 0 <= nj < L:
                        nidx = ni * L + nj
                    else:
                        continue

                if nidx > idx:  # 避免重复
                    bonds.append((idx, nidx))
                    adjacency[idx].append(nidx)
                    adjacency[nidx].append(idx)

    coord_numbers = [len(adjacency[s]) for s in range(N)]

    return {
        'Lx': L, 'Ly': L, 'N': N,
        'sites': sites, 'bonds': bonds,
        'adjacency': adjacency,
        'coord_numbers': coord_numbers,
        'z_mean': np.mean(coord_numbers),
        'z_var': np.var(coord_numbers),
        'boundary': boundary,
        'lattice_type': 'triangular',
    }


# ============================================================================
#  第二部分: CVT非均匀晶格 (融合 256_cvt_corn_movie)
# ============================================================================

def cvt_lattice_relaxation(seed_points: np.ndarray,
                           domain: Tuple[float, float, float, float],
                           n_samples: int = 5000,
                           n_iterations: int = 30,
                           tolerance: float = 1e-8) -> np.ndarray:
    """
    Centroidal Voronoi Tessellation (CVT) 晶格松弛

    算法 (Lloyd迭代, 融合 256_cvt_corn_movie 的 cvt_disk):
        1. 在域Ω内生成n_samples个采样点
        2. 将每个采样点分配给最近的生成点
        3. 将每个生成点移到其Voronoi域的质心
        4. 重复直到收敛

    物理应用:
        非均匀晶格的构造,用于研究无序对量子相变的影响。
        CVT生成的晶格具有最优的L²量化性质,
        即 ∫_Ω |x - nearest(x)|² dx 最小化。

    能量泛函:
        E_CVT = Σ_i ∫_{V_i} ρ(x) |x - g_i|² dx
        其中 V_i 是生成点 g_i 的 Voronoi 域,
        ρ(x) 是密度函数。

    参数:
        seed_points: 初始生成点 (n_gen, 2)
        domain: (xmin, xmax, ymin, ymax)
        n_samples: Monte Carlo采样点数
        n_iterations: Lloyd迭代次数
        tolerance: 收敛阈值

    返回:
        松弛后的生成点 (n_gen, 2)
    """
    xmin, xmax, ymin, ymax = domain
    points = seed_points.copy()
    n_gen = len(points)

    # Monte Carlo采样 (融合 941_quad_monte_carlo 的随机采样思想)
    samples_x = np.random.uniform(xmin, xmax, n_samples)
    samples_y = np.random.uniform(ymin, ymax, n_samples)
    samples = np.column_stack([samples_x, samples_y])

    for iteration in range(n_iterations):
        # 将采样点分配给最近的生成点
        assignments = np.zeros(n_samples, dtype=int)
        min_dists = np.full(n_samples, np.inf)

        for gi in range(n_gen):
            dists = np.sum((samples - points[gi]) ** 2, axis=1)
            closer = dists < min_dists
            assignments[closer] = gi
            min_dists[closer] = dists[closer]

        # 计算Voronoi域质心
        new_points = np.zeros_like(points)
        for gi in range(n_gen):
            mask = assignments == gi
            if np.sum(mask) > 0:
                new_points[gi] = np.mean(samples[mask], axis=0)
            else:
                new_points[gi] = points[gi]

        # 检查收敛
        max_shift = np.max(np.sqrt(np.sum((new_points - points) ** 2, axis=1)))
        points = new_points

        if max_shift < tolerance:
            break

    return points


def build_voronoi_lattice(n_sites: int,
                          domain: Tuple[float, float, float, float] = (0, 1, 0, 1),
                          cutoff_radius: float = None) -> Dict:
    """
    基于CVT构造非均匀Voronoi晶格

    物理背景:
        无序晶格上的量子Ising模型:
        H = -Σ_{<ij>} J_ij σ_i^z σ_j^z - Σ_i Γ_i σ_i^x
        其中 J_ij 依赖于位点间距: J_ij = J₀ exp(-|r_i - r_j|/λ)

    参数:
        n_sites: 位点数
        domain: 计算域
        cutoff_radius: 近邻截断半径 (默认自动计算)
    """
    # 随机初始种子点
    xmin, xmax, ymin, ymax = domain
    seed_points = np.column_stack([
        np.random.uniform(xmin, xmax, n_sites),
        np.random.uniform(ymin, ymax, n_sites),
    ])

    # CVT松弛
    points = cvt_lattice_relaxation(seed_points, domain)

    # 构造近邻关系 (基于距离截断)
    if cutoff_radius is None:
        # 自动截断: 取平均最近邻距离的1.5倍
        from scipy.spatial import distance_matrix
        dist_matrix = distance_matrix(points, points)
        np.fill_diagonal(dist_matrix, np.inf)
        min_dists = np.min(dist_matrix, axis=1)
        cutoff_radius = 1.5 * np.mean(min_dists)

    bonds = []
    adjacency = {i: [] for i in range(n_sites)}
    dist_matrix = np.sqrt(np.sum(
        (points[:, np.newaxis, :] - points[np.newaxis, :, :]) ** 2, axis=2
    ))

    for i in range(n_sites):
        for j in range(i + 1, n_sites):
            if dist_matrix[i, j] < cutoff_radius:
                bonds.append((i, j))
                adjacency[i].append(j)
                adjacency[j].append(i)

    coord_numbers = [len(adjacency[s]) for s in range(n_sites)]

    return {
        'N': n_sites,
        'sites': points,
        'bonds': bonds,
        'adjacency': adjacency,
        'coord_numbers': coord_numbers,
        'z_mean': np.mean(coord_numbers),
        'z_var': np.var(coord_numbers),
        'cutoff_radius': cutoff_radius,
        'domain': domain,
        'lattice_type': 'voronoi_cvt',
    }


# ============================================================================
#  第三部分: 共形映射晶格变形 (融合 785_naca 的翼型几何)
# ============================================================================

def conformal_mapping_deformation(sites: List, mapping_type: str = 'joukowski',
                                  params: Dict = None) -> np.ndarray:
    """
    对晶格施加共形映射变形

    物理背景:
        共形映射保持局部角度但改变晶格常数,
        用于研究空间变化的耦合强度对量子临界行为的影响。

    Joukowski映射 (融合 785_naca 的翼型概念):
        w = z + a²/z
        其中 z = x + iy, a 为映射参数

    其他映射:
        - 指数映射: w = exp(αz), 用于半无穷几何
        - 平方根映射: w = √z, 用于楔形几何

    参数:
        sites: 原始格点列表 [(x,y), ...]
        mapping_type: 'joukowski', 'exponential', 'sqrt'
        params: 映射参数
    """
    if params is None:
        params = {}

    z = np.array([complex(s[0], s[1]) for s in sites])

    if mapping_type == 'joukowski':
        a = params.get('a', 0.3)
        # Joukowski: w = z + a²/z
        # 避免除零: 添加小偏移
        z_safe = np.where(np.abs(z) < 1e-10, z + 1e-10, z)
        w = z_safe + a ** 2 / z_safe

    elif mapping_type == 'exponential':
        alpha = params.get('alpha', 0.5 + 0.1j)
        w = np.exp(alpha * z)

    elif mapping_type == 'sqrt':
        # 平方根映射,选择分支切割沿负实轴
        w = np.sqrt(z + 1.0)  # 偏移避免分支点

    else:
        raise ValueError(f"Unknown mapping type: {mapping_type}")

    return np.column_stack([w.real, w.imag])


# ============================================================================
#  第四部分: 边界拓扑分析 (融合 109_boundary_word_right)
# ============================================================================

def classify_boundary_topology(lattice: Dict) -> Dict:
    """
    分析晶格边界的拓扑性质

    物理背景:
        边界条件对量子相变的影响:
        - 周期边界 (torus): 保持平移对称性, 有限尺寸修正 ~ 1/L
        - 开放边界 (open): 边界态可能改变低能谱
        - 螺旋边界 (Möbius): 引入拓扑相位

    融合 109_boundary_word_right 的边界字表示:
        边界字 w = b₁b₂...bₙ, bᵢ ∈ {N, E, S, W}
        边界字的等价类对应不同的拓扑 sector

    返回:
        boundary_sites: 边界位点索引
        boundary_word: 边界字编码
        euler_characteristic: 欧拉示性数 χ = V - E + F
        genus: 亏格 g = (2 - χ) / 2
    """
    N = lattice['N']
    bonds = lattice['bonds']
    adjacency = lattice['adjacency']
    boundary_type = lattice.get('boundary', 'open')

    # 识别边界位点 (配位数 < 最大配位数)
    coord_numbers = lattice['coord_numbers']
    z_max = max(coord_numbers) if coord_numbers else 0
    boundary_sites = [i for i in range(N) if coord_numbers[i] < z_max]

    # 边界字编码 (融合 boundary_word_right 的8方向编码)
    # 方向: 0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW
    direction_map = {0: 'N', 1: 'NE', 2: 'E', 3: 'SE',
                     4: 'S', 5: 'SW', 6: 'W', 7: 'NW'}

    boundary_word = []
    if boundary_sites:
        # 按角度排序边界位点
        sites = lattice['sites']
        center = np.mean([sites[s] for s in boundary_sites], axis=0)
        angles = np.arctan2(
            [sites[s][1] - center[1] for s in boundary_sites],
            [sites[s][0] - center[0] for s in boundary_sites]
        )
        sorted_bs = [boundary_sites[i] for i in np.argsort(angles)]

        for k in range(len(sorted_bs) - 1):
            s1, s2 = sorted_bs[k], sorted_bs[k + 1]
            dx = sites[s2][0] - sites[s1][0]
            dy = sites[s2][1] - sites[s1][1]
            angle = np.arctan2(dy, dx)
            # 量化为8方向
            direction = int(np.round((angle + np.pi) / (np.pi / 4))) % 8
            boundary_word.append(direction_map[direction])

    # 欧拉示性数计算
    V = N
    E = len(bonds)
    # 面数 (对于平面嵌入): F = E - V + 2 (Euler公式)
    F = E - V + 2  # 对于球面拓扑
    euler_characteristic = V - E + F

    # 亏格 (对于闭合可定向曲面)
    if boundary_type == 'periodic':
        genus = 1  # 环面
    elif boundary_type == 'cylinder':
        genus = 0  # 圆柱
    else:
        genus = 0  # 圆盘

    return {
        'boundary_sites': boundary_sites,
        'n_boundary': len(boundary_sites),
        'boundary_word': ''.join(boundary_word) if boundary_word else '',
        'euler_characteristic': euler_characteristic,
        'genus': genus,
        'boundary_type': boundary_type,
        'V': V, 'E': E, 'F': F,
    }


# ============================================================================
#  第五部分: 自旋簇域检测 (融合 314_double_c_data 的聚类思想)
# ============================================================================

def detect_spin_clusters(spins: np.ndarray, lattice: Dict,
                         cluster_threshold: float = 0.5) -> Dict:
    """
    检测自旋簇域 (SW团簇算法的基础)

    物理背景:
        量子Ising模型中的自旋簇与几何相变相关。
        在临界点, 簇的大小分布呈现幂律:
            n_s ~ s^{-τ} exp(-s/s_ξ)
        其中 s_ξ ~ ξ^{D_f}, D_f 为分形维数。

        Fortuin-Kasteleyn (FK) 团簇:
        键占据概率 p_bond = 1 - exp(-2J/kT) (经典)
        量子推广: p_bond = 1 - exp(-2JΔτ/ℏ)

    融合 314_double_c_data 的聚类测试思想:
        使用BFS/DFS进行连通分量分析

    参数:
        spins: 自旋构型 σ ∈ {+1, -1}^N
        lattice: 晶格字典
        cluster_threshold: FK键概率阈值

    返回:
        cluster_labels: 每个位点的簇标签
        cluster_sizes: 各簇的大小
        n_clusters: 簇数量
        percolation_strength: 最大簇的相对大小 (序参量)
    """
    N = lattice['N']
    adjacency = lattice['adjacency']

    # Fortuin-Kasteleyn键占据
    # 对于量子Ising: p = 1 - exp(-2βJ/(n_slices))
    # 这里使用经典近似
    p_bond = 1.0 - np.exp(-2.0 * cluster_threshold)

    # BFS 寻找连通分量
    visited = np.zeros(N, dtype=bool)
    cluster_labels = np.full(N, -1, dtype=int)
    cluster_id = 0
    cluster_sizes = []

    for start in range(N):
        if visited[start]:
            continue

        # BFS
        queue = [start]
        visited[start] = True
        cluster_labels[start] = cluster_id
        size = 0

        while queue:
            node = queue.pop(0)
            size += 1

            for neighbor in adjacency[node]:
                if visited[neighbor]:
                    continue
                # FK键条件: 同向自旋且随机数 < p_bond
                if spins[node] == spins[neighbor]:
                    r = np.random.random()
                    if r < p_bond:
                        visited[neighbor] = True
                        cluster_labels[neighbor] = cluster_id
                        queue.append(neighbor)

        cluster_sizes.append(size)
        cluster_id += 1

    # 计算物理量
    cluster_sizes = np.array(cluster_sizes)
    n_clusters = cluster_id
    percolation_strength = np.max(cluster_sizes) / N if n_clusters > 0 else 0

    # 矩分析 (用于有限尺寸标度)
    # 平均簇大小: χ = Σ s² n_s / Σ s n_s
    if np.sum(cluster_sizes) > 0:
        mean_cluster_size = np.sum(cluster_sizes ** 2) / np.sum(cluster_sizes)
    else:
        mean_cluster_size = 0

    return {
        'cluster_labels': cluster_labels,
        'cluster_sizes': cluster_sizes,
        'n_clusters': n_clusters,
        'percolation_strength': percolation_strength,
        'mean_cluster_size': mean_cluster_size,
        'max_cluster_size': np.max(cluster_sizes) if n_clusters > 0 else 0,
    }


# ============================================================================
#  第六部分: 双C形畴壁测试构型 (融合 314_double_c_data)
# ============================================================================

def generate_domain_wall_test_config(lattice: Dict,
                                     wall_type: str = 'straight') -> np.ndarray:
    """
    生成包含畴壁的测试自旋构型

    物理背景:
        畴壁能是表征对称性破缺的关键量。
        对于2D Ising模型, 畴壁能:
            σ(T) = 2J - kT ln(sinh(2J/kT))  (精确结果)

        在量子情况下, 横场降低畴壁能:
            σ(Γ) = σ(0) √(1 - (Γ/Γ_c)²)  (平均场近似)

    融合 314_double_c_data 的几何构型:
        使用交错C形区域构造复杂畴壁

    参数:
        lattice: 晶格字典
        wall_type: 'straight', 'diagonal', 'double_c'
    """
    N = lattice['N']
    sites = lattice['sites']
    Lx, Ly = lattice['Lx'], lattice['Ly']

    spins = np.ones(N)

    if wall_type == 'straight':
        # 垂直畴壁: 左半 +1, 右半 -1
        for idx in range(N):
            if isinstance(sites[idx], tuple) and len(sites[idx]) == 2:
                if sites[idx][0] >= Lx / 2:
                    spins[idx] = -1
            else:
                # 非均匀晶格
                if sites[idx][0] >= (sites[0][0] + sites[-1][0]) / 2:
                    spins[idx] = -1

    elif wall_type == 'diagonal':
        # 对角畴壁
        for idx in range(N):
            if isinstance(sites[idx], tuple) and len(sites[idx]) == 2:
                if sites[idx][0] + sites[idx][1] >= (Lx + Ly) / 2:
                    spins[idx] = -1

    elif wall_type == 'double_c':
        # 双C形畴壁 (融合 double_c_data 的几何)
        center = np.array([np.mean([s[0] for s in sites]),
                           np.mean([s[1] for s in sites])])
        for idx in range(N):
            pos = np.array(sites[idx])
            r = np.sqrt(np.sum((pos - center) ** 2))
            theta = np.arctan2(pos[1] - center[1], pos[0] - center[0])
            # C形区域: 半径2-5, 角度π/2到3π/2
            if 0.2 < r < 0.5 and np.pi / 2 < theta < 3 * np.pi / 2:
                spins[idx] = -1

    return spins
