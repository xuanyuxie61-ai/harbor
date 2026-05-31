"""
halo_finder.py
==============
暗物质晕识别与统计分析模块

采用 Friends-of-Friends (FOF) 算法与球形过密度判据识别暗物质晕，
融入 levels（水平集/等值面分析）与 sphere_positive_distance（球面随机采样与统计）
的核心算法，为大尺度结构模拟提供晕目录生成与质量函数统计。

核心物理公式
------------
Friends-of-Friends (FOF) 算法:
    两个粒子 p, q 若满足 |x_p - x_q| < b L / N^{1/3} 则被连接为友邻，
    其中 b 为连接参数（通常取 0.2）。

    友邻关系具有传递性，所有通过友邻链相连的粒子构成一个暗物质晕。

球形过密度判据 (SO):
    以某点为中心，在半径 R_Δ 内的平均密度满足:
        ρ̄(<R_Δ) = Δ · ρ_{crit}(z)

    其中 Δ 通常取 200（对应 virialized 区域）。
    晕质量:
        M_Δ = (4π/3) R_Δ³ Δ ρ_{crit}(z)

密度场的水平集分析（融入 levels 核心思想）:
    对密度场 δ(x)，选取一系列阈值水平 δ_i，
    计算每个水平上的等值面所包围的体积:
        V(>δ_i) = ∫ Θ(δ(x) - δ_i) d³x

    其中 Θ 为 Heaviside 阶跃函数。
    该体积分布可用于分析宇宙 Web 结构（结节、纤维、面、空腔）。

球面距离统计（融入 sphere_positive_distance）:
    对晕中心位置 {x_i}，计算天球坐标系（或局部球坐标系）中的角向分布:
        随机方向 n̂ 在单位球面 S² 上均匀采样:
            n̂ = (x, y, z) / r,  x,y,z ~ N(0,1)
        正象限采样:
            n̂ = |n̂| / ||n̂||

    两两角距:
        θ_{ij} = arccos(n̂_i · n̂_j)

    用于检验晕分布的各向同性。
"""

import numpy as np
from typing import Tuple, List
from collections import deque


class HaloFinder:
    """
    暗物质晕识别器。
    """

    def __init__(self, L: float, linking_length: float = None):
        """
        Parameters
        ----------
        L : float
            模拟盒子边长
        linking_length : float, optional
            FOF 连接长度，默认 0.2 * (L³/N_part)^{1/3}
        """
        self.L = L
        self.linking_length = linking_length

    def _distance_periodic(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """
        考虑周期性边界的最小距离。
        """
        diff = np.abs(p1 - p2)
        diff = np.minimum(diff, self.L - diff)
        return np.sqrt(np.sum(diff ** 2))

    def fof_groups(
        self, pos: np.ndarray, mass: np.ndarray = None
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Friends-of-Friends 晕识别。

        采用 brute-force 最近邻搜索（适用于小规模模拟）。

        Parameters
        ----------
        pos : np.ndarray, shape (N_p, 3)
            粒子位置
        mass : np.ndarray, optional
            粒子质量（用于计算晕质量）

        Returns
        -------
        groups : list of np.ndarray
            每个晕的粒子索引列表
        halo_mass : np.ndarray
            每个晕的总质量
        """
        n_part = pos.shape[0]
        if self.linking_length is None:
            self.linking_length = 0.2 * (self.L ** 3 / n_part) ** (1.0 / 3.0)

        visited = np.zeros(n_part, dtype=bool)
        groups = []
        halo_mass = []

        # 构建简单空间索引：按网格分箱
        n_bins = max(1, int(self.L / self.linking_length))
        bin_size = self.L / n_bins
        bins = {}
        for i in range(n_part):
            bx = int(pos[i, 0] / bin_size) % n_bins
            by = int(pos[i, 1] / bin_size) % n_bins
            bz = int(pos[i, 2] / bin_size) % n_bins
            key = (bx, by, bz)
            if key not in bins:
                bins[key] = []
            bins[key].append(i)

        def get_neighbors(idx: int) -> List[int]:
            """获取在连接长度内的邻居粒子索引。"""
            p = pos[idx]
            bx = int(p[0] / bin_size) % n_bins
            by = int(p[1] / bin_size) % n_bins
            bz = int(p[2] / bin_size) % n_bins
            neighbors = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        key = (
                            (bx + dx) % n_bins,
                            (by + dy) % n_bins,
                            (bz + dz) % n_bins,
                        )
                        if key in bins:
                            for j in bins[key]:
                                if j != idx and not visited[j]:
                                    dist = self._distance_periodic(p, pos[j])
                                    if dist < self.linking_length:
                                        neighbors.append(j)
            return neighbors

        for i in range(n_part):
            if visited[i]:
                continue
            # BFS 找连通分量
            queue = deque([i])
            visited[i] = True
            group = [i]
            while queue:
                cur = queue.popleft()
                for nb in get_neighbors(cur):
                    if not visited[nb]:
                        visited[nb] = True
                        group.append(nb)
                        queue.append(nb)
            groups.append(np.array(group))
            if mass is not None:
                halo_mass.append(mass[group].sum())
            else:
                halo_mass.append(len(group))

        return groups, np.array(halo_mass)

    def spherical_overdensity_mass(
        self,
        pos: np.ndarray,
        mass: np.ndarray,
        center: np.ndarray,
        rho_crit: float,
        Delta: float = 200.0,
    ) -> Tuple[float, float]:
        """
        计算以 center 为中心的球形过密度质量与半径。

        算法:
            将粒子按距 center 的距离排序，逐个累加质量，
            直到平均密度 ρ̄ = M(<R) / (4π R³/3) < Δ ρ_{crit}

        公式:
            M_Δ = Σ_{|x_i - c| < R_Δ} m_i
            R_Δ = (3 M_Δ / (4π Δ ρ_{crit}))^{1/3}
        """
        # 周期性距离
        diff = np.abs(pos - center)
        diff = np.minimum(diff, self.L - diff)
        dist = np.sqrt(np.sum(diff ** 2, axis=1))
        sort_idx = np.argsort(dist)
        sorted_dist = dist[sort_idx]
        sorted_mass = mass[sort_idx]
        cum_mass = np.cumsum(sorted_mass)
        # 平均密度
        volumes = (4.0 / 3.0) * np.pi * sorted_dist ** 3
        volumes = np.clip(volumes, 1e-30, None)
        rho_avg = cum_mass / volumes
        mask = rho_avg >= Delta * rho_crit
        if mask.sum() == 0:
            return 0.0, 0.0
        idx = np.where(mask)[0][-1]
        return cum_mass[idx], sorted_dist[idx]


def level_set_volume_analysis(
    delta_grid: np.ndarray,
    L: float,
    n_levels: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    密度场的水平集体积分析（融入 levels 核心思想）。

    Parameters
    ----------
    delta_grid : np.ndarray
        密度对比度场
    L : float
        盒子边长
    n_levels : int
        水平数

    Returns
    -------
    levels : np.ndarray
        阈值水平数组
    volumes : np.ndarray
        每水平上等值面所包围的体积分数
    """
    delta_min = delta_grid.min()
    delta_max = delta_grid.max()
    levels = np.linspace(delta_min, delta_max, n_levels)
    total_volume = L ** 3
    volumes = np.zeros(n_levels)
    dx3 = (L / delta_grid.shape[0]) ** 3
    for i, lev in enumerate(levels):
        mask = delta_grid >= lev
        volumes[i] = mask.sum() * dx3 / total_volume
    return levels, volumes


def sample_sphere_positive_distance(n_samples: int, rng: np.random.Generator = None) -> np.ndarray:
    """
    单位球面正象限随机采样（融入 sphere_positive_sample 核心算法）。

    算法:
        1. 生成三维标准正态分布随机向量 x ~ N(0, I)
        2. 归一化: n̂ = |x| / ||x||

    在宇宙学中用于:
        - 随机视线方向的生成
        - 角向分布统计检验
    """
    if rng is None:
        rng = np.random.default_rng(seed=42)
    x = rng.standard_normal((n_samples, 3))
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-15, None)
    return np.abs(x) / norms


def angular_distance_histogram(
    directions: np.ndarray, n_bins: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算球面上随机方向对的角距离分布。

    各向同性分布的理论概率密度:
        P(θ) dθ = sin(θ) dθ / 2

    归一化 histogram 应接近 sin(θ)/2。
    """
    n = len(directions)
    if n < 2:
        return np.array([]), np.array([])
    # 随机抽取若干对
    max_pairs = min(n * (n - 1) // 2, 50000)
    rng = np.random.default_rng(seed=42)
    angles = []
    for _ in range(max_pairs):
        i, j = rng.integers(0, n, 2)
        if i != j:
            cos_theta = np.clip(np.dot(directions[i], directions[j]), -1.0, 1.0)
            angles.append(np.arccos(cos_theta))
    angles = np.array(angles)
    hist, edges = np.histogram(angles, bins=n_bins, range=(0.0, np.pi))
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    # 归一化
    bin_width = edges[1] - edges[0]
    pdf = hist / (hist.sum() * bin_width)
    return bin_centers, pdf


def halo_mass_function_from_groups(
    halo_mass: np.ndarray, volume: float, n_bins: int = 15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从晕质量列表估计质量函数 dn/dlnM。

    公式:
        dn/dlnM = (1/V) · N_i / ΔlnM_i
    """
    if len(halo_mass) == 0:
        return np.array([]), np.array([]), np.array([])
    logM = np.log10(halo_mass[halo_mass > 0])
    if len(logM) == 0:
        return np.array([]), np.array([]), np.array([])
    bins = np.linspace(logM.min(), logM.max(), n_bins + 1)
    counts, edges = np.histogram(logM, bins=bins)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    dlnM = (edges[1:] - edges[:-1]) * np.log(10)
    dn_dlnM = counts / (volume * dlnM)
    # Poisson 误差
    err = np.sqrt(np.clip(counts, 0, None)) / (volume * dlnM)
    return bin_centers, dn_dlnM, err


if __name__ == "__main__":
    # 自检
    pos = np.random.rand(1000, 3) * 100.0
    mass = np.ones(1000) * 1e10
    finder = HaloFinder(L=100.0)
    groups, hmass = finder.fof_groups(pos, mass)
    print(f"识别到 {len(groups)} 个 FOF 晕")
    print(f"最大晕质量: {hmass.max():.3e}")

    # 水平集分析
    delta = np.random.randn(32, 32, 32) * 0.5
    lev, vol = level_set_volume_analysis(delta, 100.0, n_levels=10)
    print(f"水平集体积分数范围: [{vol.min():.4f}, {vol.max():.4f}]")

    # 球面采样
    dirs = sample_sphere_positive_distance(1000)
    bc, pdf = angular_distance_histogram(dirs)
    print(f"角距离均值: {bc.mean():.4f} (理论 π/2 ≈ {np.pi/2:.4f})")
