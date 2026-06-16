# -*- coding: utf-8 -*-
"""
grain_structure.py
==================
多晶晶粒结构与位错塞积模拟模块

本模块融合以下种子项目算法:
- 1398_voronoi_plot: Voronoi图 → 多晶晶粒几何生成
- 1229_maidens_2017-LCSS: LCSS时间序列分析 → 位错速度信号分析

核心物理:
---------
多晶材料中，晶界是位错运动的主要障碍。
位错在晶界前塞积 (pile-up)，产生应力集中:

σ_pileup(n) = n σ_applied

其中 n 是塞积位错数量:
n = (π(1-ν) L τ) / (μ b)

L: 晶粒尺寸
τ: 施加剪应力

Hall-Petch关系:
σ_y = σ₀ + k_HP / √d

d: 晶粒尺寸
k_HP: Hall-Petch系数 (≈ 0.1 MPa·m^{1/2} for Al)

Voronoi晶粒生成:
1. 随机分布种子点
2. 计算每个像素到最近种子的距离
3. 分配给最近的种子
"""

import math
import random
from physical_constants import PI, DEFAULT_MATERIAL


# ============================================================================
# Voronoi晶粒结构 — 来自 1398_voronoi_plot
# ============================================================================

class VoronoiGrainStructure:
    """
    2D Voronoi多晶晶粒结构生成

    算法:
    1. 在域内随机分布 N 个种子点
    2. 对每个网格点 (x, y)，找到最近种子
    3. 分配晶粒编号
    4. 识别晶界 (不同晶粒相邻的网格点)

    物理应用:
    - 多晶塑性模拟的初始微结构
    - 晶界位错塞积分析
    - 晶粒尺寸对强度的影响 (Hall-Petch)
    """

    def __init__(self, domain_size, n_grains, seed=None):
        """
        初始化Voronoi晶粒结构

        Args:
            domain_size: 域尺寸 (Lx, Ly) (m)
            n_grains: 晶粒数量
            seed: 随机种子
        """
        self.Lx, self.Ly = domain_size
        self.n_grains = n_grains
        self.rng = random.Random(seed)

        self.seeds = []
        self.grain_map = None
        self.grain_boundaries = []
        self.grain_areas = []
        self.grain_sizes = []

        self._generate_seeds()

    def _generate_seeds(self):
        """生成随机种子点"""
        self.seeds = []
        for _ in range(self.n_grains):
            x = self.rng.uniform(0, self.Lx)
            y = self.rng.uniform(0, self.Ly)
            self.seeds.append((x, y))

    def build_grain_map(self, resolution=50):
        """
        构建晶粒映射

        对每个网格点分配最近种子的编号

        Args:
            resolution: 网格分辨率

        Returns:
            list: resolution × resolution 晶粒编号矩阵
        """
        self.resolution = resolution
        dx = self.Lx / resolution
        dy = self.Ly / resolution

        grain_map = [[0]*resolution for _ in range(resolution)]
        self.grain_boundaries = []

        for i in range(resolution):
            for j in range(resolution):
                x = (i + 0.5) * dx
                y = (j + 0.5) * dy

                # 找最近种子
                min_dist = float('inf')
                nearest = 0
                for k, (sx, sy) in enumerate(self.seeds):
                    dist = (x - sx)**2 + (y - sy)**2
                    if dist < min_dist:
                        min_dist = dist
                        nearest = k

                grain_map[i][j] = nearest

                # 检测晶界
                if i > 0 and grain_map[i-1][j] != nearest:
                    self.grain_boundaries.append(((i, j), (i-1, j)))
                if j > 0 and grain_map[i][j-1] != nearest:
                    self.grain_boundaries.append(((i, j), (i, j-1)))

        self.grain_map = grain_map
        self._compute_grain_statistics()
        return grain_map

    def _compute_grain_statistics(self):
        """计算晶粒统计信息"""
        # 计算每个晶粒的面积
        self.grain_areas = [0.0] * self.n_grains
        cell_area = (self.Lx / self.resolution) * (self.Ly / self.resolution)

        for i in range(self.resolution):
            for j in range(self.resolution):
                gid = self.grain_map[i][j]
                self.grain_areas[gid] += cell_area

        # 等效晶粒尺寸 (√area)
        self.grain_sizes = [math.sqrt(a) for a in self.grain_areas]

    def get_average_grain_size(self):
        """计算平均晶粒尺寸"""
        if not self.grain_sizes:
            return 0.0
        return sum(self.grain_sizes) / len(self.grain_sizes)

    def get_grain_size_distribution(self):
        """计算晶粒尺寸分布"""
        if not self.grain_sizes:
            return {}

        d_min = min(self.grain_sizes)
        d_max = max(self.grain_sizes)
        n_bins = 10
        bin_width = (d_max - d_min) / n_bins if d_max > d_min else 1.0

        distribution = {i: 0 for i in range(n_bins)}
        for d in self.grain_sizes:
            bin_idx = min(int((d - d_min) / bin_width), n_bins - 1)
            distribution[bin_idx] += 1

        return {
            'distribution': distribution,
            'mean': sum(self.grain_sizes) / len(self.grain_sizes),
            'std': math.sqrt(sum((d - sum(self.grain_sizes)/len(self.grain_sizes))**2
                                 for d in self.grain_sizes) / len(self.grain_sizes)),
            'min': d_min,
            'max': d_max,
        }

    def get_neighbors(self, grain_id):
        """获取给定晶粒的邻居列表"""
        neighbors = set()
        for (i1, j1), (i2, j2) in self.grain_boundaries:
            g1 = self.grain_map[i1][j1]
            g2 = self.grain_map[i2][j2]
            if g1 == grain_id and g2 != grain_id:
                neighbors.add(g2)
            elif g2 == grain_id and g1 != grain_id:
                neighbors.add(g1)
        return list(neighbors)


# ============================================================================
# 位错塞积模型
# ============================================================================

class DislocationPileup:
    """
    位错在晶界前的塞积模型

    Eshelby-Frank-Nabarro塞积理论:

    n条位错在长度L上塞积:
    - 位置: x_k = L cos²(kπ/(2n)), k = 1, ..., n
    - 应力集中: σ_tip = n σ_applied
    - 总Burgers矢量: B_total = n b

    塞积导致的Hall-Petch强化:
    σ_y = σ₀ + k_HP / √d

    其中:
    - σ₀: 晶格摩擦应力 (Peierls应力)
    - k_HP = √(4 μ γ_s b) (γ_s: 表面能)
    - d: 晶粒尺寸
    """

    def __init__(self, material=None):
        self.mat = material or DEFAULT_MATERIAL

    def pileup_positions(self, n_dislocations, pileup_length):
        """
        计算塞积位错的位置

        x_k = (L/2) * (1 + cos((2k-1)π/(2n))), k = 1, ..., n

        位错在领先位错 (x₁ ≈ L) 和尾部位错 (x_n ≈ 0) 之间分布

        Args:
            n_dislocations: 塞积位错数
            pileup_length: 塞积长度 (m)

        Returns:
            list: 各条位错的位置 (m)
        """
        positions = []
        for k in range(1, n_dislocations + 1):
            x_k = (pileup_length / 2.0) * (1.0 + math.cos((2*k - 1) * PI / (2 * n_dislocations)))
            positions.append(x_k)
        return positions

    def pileup_stress_concentration(self, n_dislocations, sigma_applied):
        """
        计算塞积尖端应力集中

        σ_tip = n σ_applied

        更精确的公式 (Eshelby):
        σ_tip(r) = σ_applied * √(L/r)

        其中 r 是距塞积尖端的距离

        Args:
            n_dislocations: 位错数
            sigma_applied: 施加应力 (Pa)

        Returns:
            float: 尖端应力 (Pa)
        """
        return n_dislocations * sigma_applied

    def number_of_dislocations(self, grain_size, sigma_applied):
        """
        计算塞积位错数量

        n = π(1-ν) L τ / (μ b)

        其中 L ≈ d (晶粒尺寸)

        Args:
            grain_size: 晶粒尺寸 (m)
            sigma_applied: 施加剪应力 (Pa)

        Returns:
            int: 塞积位错数
        """
        n = PI * (1.0 - self.mat.nu) * grain_size * sigma_applied / \
            (self.mat.mu * self.mat.b_magnitude)
        return max(1, int(round(n)))

    def hall_petch_strength(self, grain_size, sigma_0=None, k_hp=None):
        """
        Hall-Petch屈服强度

        σ_y = σ₀ + k_HP / √d

        Args:
            grain_size: 晶粒尺寸 (m)
            sigma_0: 摩擦应力 (Pa, 默认用Peierls应力)
            k_hp: Hall-Petch系数 (Pa·m^{1/2})

        Returns:
            float: 屈服强度 (Pa)
        """
        if sigma_0 is None:
            sigma_0 = self.mat.peierls_stress()
        if k_hp is None:
            # 估算: k_HP ≈ √(4 μ γ_s b)
            gamma_s = 1.0  # 表面能 ~1 J/m²
            k_hp = math.sqrt(4.0 * self.mat.mu * gamma_s * self.mat.b_magnitude)

        d = max(grain_size, 1e-12)  # 防止奇异
        return sigma_0 + k_hp / math.sqrt(d)

    def inverse_hall_petch(self, grain_size):
        """
        反Hall-Petch效应 (纳米晶)

        当晶粒尺寸 < 临界值 (~10-20 nm):
        位错塞积不可能 (n < 1)
        变形机制转变为晶界滑移

        σ_y = σ₀ + k₁ d (正比于d)

        交叉尺寸: d_c ≈ 10-20 nm

        Args:
            grain_size: 晶粒尺寸 (m)

        Returns:
            float: 屈服强度 (Pa)
        """
        d_c = 15e-9  # 临界晶粒尺寸
        sigma_0 = self.mat.peierls_stress()

        if grain_size >= d_c:
            return self.hall_petch_strength(grain_size)
        else:
            # 反Hall-Petch
            k_inv = self.mat.mu / 10.0  # 估算系数
            return sigma_0 + k_inv * grain_size / d_c


# ============================================================================
# LCSS时间序列分析 — 来自 1229_maidens_2017-LCSS
# ============================================================================

class LCSSTimeSeriesAnalysis:
    """
    最长公共子序列 (LCSS) 时间序列分析

    用于比较不同应力水平或不同晶粒中位错速度信号的相似性。

    LCSS算法:
    给定两个时间序列 A = [a₁, ..., aₘ] 和 B = [b₁, ..., bₙ]:
    LCSS(A, B) = 最长公共子序列的长度

    其中元素匹配条件:
    |aᵢ - bⱼ| ≤ ε 且 |i - j| ≤ δ

    ε: 幅度容差
    δ: 时间偏移容差

    LCSS相似度:
    sim(A, B) = LCSS(A, B) / min(m, n)

    物理应用:
    - 比较不同晶粒中的位错速度模式
    - 识别周期性位错运动
    - 检测位错脱钉扎事件的时间相关性
    """

    def __init__(self, epsilon=0.1, delta=5):
        """
        初始化LCSS分析器

        Args:
            epsilon: 幅度容差
            delta: 时间偏移容差
        """
        self.epsilon = epsilon
        self.delta = delta

    def compute_lcss(self, seq_a, seq_b):
        """
        计算两个序列的LCSS

        动态规划:
        L[i,j] = L[i-1, j-1] + 1  if |aᵢ-bⱼ|≤ε and |i-j|≤δ
        L[i,j] = max(L[i-1,j], L[i,j-1])  otherwise

        Args:
            seq_a: 序列A
            seq_b: 序列B

        Returns:
            dict: LCSS结果
        """
        m = len(seq_a)
        n = len(seq_b)

        # DP表
        L = [[0]*(n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if abs(i - j) <= self.delta and abs(seq_a[i-1] - seq_b[j-1]) <= self.epsilon:
                    L[i][j] = L[i-1][j-1] + 1
                else:
                    L[i][j] = max(L[i-1][j], L[i][j-1])

        lcss_length = L[m][n]
        similarity = lcss_length / min(m, n) if min(m, n) > 0 else 0.0

        return {
            'lcss_length': lcss_length,
            'similarity': similarity,
            'len_a': m,
            'len_b': n,
        }

    def compute_distance_matrix(self, sequences):
        """
        计算多个序列间的LCSS距离矩阵

        d(A, B) = 1 - sim(A, B)

        Args:
            sequences: 序列列表

        Returns:
            list: N×N距离矩阵
        """
        n_seq = len(sequences)
        distance_matrix = [[0.0]*n_seq for _ in range(n_seq)]

        for i in range(n_seq):
            for j in range(i + 1, n_seq):
                result = self.compute_lcss(sequences[i], sequences[j])
                dist = 1.0 - result['similarity']
                distance_matrix[i][j] = dist
                distance_matrix[j][i] = dist

        return distance_matrix

    def analyze_velocity_patterns(self, velocity_signals):
        """
        分析位错速度信号模式

        比较不同条件下的位错速度时间序列

        Args:
            velocity_signals: 速度信号列表

        Returns:
            dict: 模式分析结果
        """
        n_signals = len(velocity_signals)

        # 归一化信号
        normalized = []
        for sig in velocity_signals:
            if sig:
                max_val = max(abs(s) for s in sig) if sig else 1.0
                if max_val < 1e-30:
                    max_val = 1.0
                normalized.append([s / max_val for s in sig])
            else:
                normalized.append([0.0])

        # 计算距离矩阵
        dist_matrix = self.compute_distance_matrix(normalized)

        # 找最相似和最不相似的信号对
        min_dist = float('inf')
        max_dist = 0.0
        min_pair = (0, 0)
        max_pair = (0, 0)

        for i in range(n_signals):
            for j in range(i + 1, n_signals):
                d = dist_matrix[i][j]
                if d < min_dist:
                    min_dist = d
                    min_pair = (i, j)
                if d > max_dist:
                    max_dist = d
                    max_pair = (i, j)

        return {
            'distance_matrix': dist_matrix,
            'most_similar_pair': min_pair,
            'most_similar_distance': min_dist,
            'most_dissimilar_pair': max_pair,
            'most_dissimilar_distance': max_dist,
            'n_signals': n_signals,
        }


if __name__ == '__main__':
    print("=" * 70)
    print("多晶晶粒结构与位错塞积验证")
    print("=" * 70)

    # 测试Voronoi晶粒
    print("\nVoronoi多晶晶粒:")
    voronoi = VoronoiGrainStructure(
        domain_size=(1e-6, 1e-6),
        n_grains=12,
        seed=42
    )
    voronoi.build_grain_map(resolution=30)
    d_avg = voronoi.get_average_grain_size()
    print(f"  晶粒数: {voronoi.n_grains}")
    print(f"  平均晶粒尺寸: {d_avg:.4e} m")
    print(f"  晶界数量: {len(voronoi.grain_boundaries)}")
    dist = voronoi.get_grain_size_distribution()
    print(f"  尺寸标准差: {dist['std']:.4e} m")

    # 测试塞积
    print("\n位错塞积分析:")
    pileup = DislocationPileup()
    d = 1e-6  # 1 μm晶粒
    sigma = 100e6  # 100 MPa

    n_disl = pileup.number_of_dislocations(d, sigma)
    positions = pileup.pileup_positions(n_disl, d)
    sigma_tip = pileup.pileup_stress_concentration(n_disl, sigma)

    print(f"  晶粒尺寸: {d:.2e} m")
    print(f"  施加应力: {sigma:.2e} Pa")
    print(f"  塞积位错数: {n_disl}")
    print(f"  尖端应力集中: {sigma_tip:.2e} Pa")

    # Hall-Petch
    print("\nHall-Petch关系:")
    for d_nm in [100, 500, 1000, 5000, 10000]:
        d_m = d_nm * 1e-9
        sigma_y = pileup.hall_petch_strength(d_m)
        print(f"  d = {d_nm:5d} nm: σ_y = {sigma_y:.4e} Pa")

    # 反Hall-Petch
    for d_nm in [5, 10, 15, 20, 30]:
        d_m = d_nm * 1e-9
        sigma_y = pileup.inverse_hall_petch(d_m)
        print(f"  d = {d_nm:2d} nm: σ_y = {sigma_y:.4e} Pa (nano)")

    # 测试LCSS
    print("\nLCSS时间序列分析:")
    lcss = LCSSTimeSeriesAnalysis(epsilon=0.2, delta=3)
    seq1 = [math.sin(2*PI*i/20) for i in range(40)]
    seq2 = [math.sin(2*PI*i/20 + 0.1) for i in range(40)]
    seq3 = [math.cos(2*PI*i/20) for i in range(40)]

    result12 = lcss.compute_lcss(seq1, seq2)
    result13 = lcss.compute_lcss(seq1, seq3)
    print(f"  sin vs sin(+φ): sim = {result12['similarity']:.4f}")
    print(f"  sin vs cos: sim = {result13['similarity']:.4f}")
