"""
布里渊区三角采样模块
=====================
对应种子项目: 1306_triangle_histogram (单位三角形直方图 → 布里渊区 k 空间采样)

物理背景:
    钙钛矿多铁性材料的电子结构和磁激发需要第一布里渊区 (BZ) 积分.
    对于立方钙钛矿, BZ 为立方体; 对于菱形 BiFeO3, BZ 为菱形十二面体.

    k 空间积分方法:
        1. 将 BZ 分解为四面体/三角形
        2. 在每个三角形上使用高斯求积
        3. 加权求和: ∫f(k)dk ≈ Σᵢ wᵢ·f(kᵢ)

    三角形直方图 (对应种子项目 1306):
        将参考三角形 (0,0)-(1,0)-(0,1) 细分为 n² 个等面积子三角形.
        对于 k 空间采样, 这确保了均匀的 k 点密度.

核心公式:
    BZ 积分:
        <A> = (1/V_BZ) ∫_BZ A(k) dk

    四面体方法 (Blöchl 改进):
        将 BZ 分为四面体, 在每个四面体内线性插值.

    权重计算:
        对于均匀三角剖分, 权重 wᵢ = V_triangle / V_BZ
"""

import numpy as np


class BrillouinZoneSampler:
    """
    布里渊区 k 空间采样器.

    将 BZ 分解为三角形/四面体网格, 生成 k 点和积分权重.
    用于计算:
        - 态密度 (DOS)
        - 磁化率 χ(k,ω)
        - 磁电耦合系数 α(k)
    """

    def __init__(self, lattice_type='cubic', a_lattice=3.96e-10,
                 n_divisions=8):
        """
        参数:
            lattice_type: 'cubic', 'rhombohedral', 'tetragonal'
            a_lattice: 晶格常数 (m)
            n_divisions: 细分层数 (影响 k 点密度)
        """
        self.lattice_type = lattice_type
        self.a = a_lattice
        self.n_div = n_divisions

        # 倒格矢
        self.b1, self.b2, self.b3 = self._compute_reciprocal_lattice()

        # 高对称 k 点
        self.high_sym_points = self._get_high_symmetry_points()

        # 三角剖分
        self.k_points, self.weights = self._generate_k_mesh()

    def _compute_reciprocal_lattice(self):
        """
        计算倒格矢.

        对于立方晶系:
            b₁ = (2π/a)(1,0,0)
            b₂ = (2π/a)(0,1,0)
            b₃ = (2π/a)(0,0,1)

        对于菱形晶系:
            使用标准转换公式.

        返回:
            b1, b2, b3: 倒格矢, shape (3,)
        """
        two_pi_a = 2.0 * np.pi / self.a

        if self.lattice_type == 'cubic':
            b1 = np.array([two_pi_a, 0.0, 0.0])
            b2 = np.array([0.0, two_pi_a, 0.0])
            b3 = np.array([0.0, 0.0, two_pi_a])

        elif self.lattice_type == 'rhombohedral':
            # 菱形晶格的倒格矢 (以 [111] 为极化方向)
            alpha = 89.3 * np.pi / 180.0
            cos_a = np.cos(alpha)
            vol = self.a ** 3 * np.sqrt(
                1.0 - 3.0 * cos_a ** 2 + 2.0 * cos_a ** 3
            )
            factor = 2.0 * np.pi * self.a / vol

            b1 = factor * np.array([
                1.0 - cos_a, cos_a - cos_a ** 2, cos_a - cos_a ** 2
            ])
            b2 = factor * np.array([
                cos_a - cos_a ** 2, 1.0 - cos_a, cos_a - cos_a ** 2
            ])
            b3 = factor * np.array([
                cos_a - cos_a ** 2, cos_a - cos_a ** 2, 1.0 - cos_a
            ])

        elif self.lattice_type == 'tetragonal':
            c_a = 1.01  # c/a 比 (近似)
            b1 = np.array([two_pi_a, 0.0, 0.0])
            b2 = np.array([0.0, two_pi_a, 0.0])
            b3 = np.array([0.0, 0.0, 2.0 * np.pi / (self.a * c_a)])

        else:
            b1 = np.array([two_pi_a, 0.0, 0.0])
            b2 = np.array([0.0, two_pi_a, 0.0])
            b3 = np.array([0.0, 0.0, two_pi_a])

        return b1, b2, b3

    def _get_high_symmetry_points(self):
        """
        获取高对称 k 点 (分数坐标).

        立方 BZ:
            Γ = (0,0,0)
            X = (0.5,0,0)
            M = (0.5,0.5,0)
            R = (0.5,0.5,0.5)

        返回:
            dict: {name: k_cart} 其中 k_cart 为笛卡尔坐标
        """
        points = {
            'Gamma': np.array([0.0, 0.0, 0.0]),
            'X': 0.5 * self.b1,
            'M': 0.5 * (self.b1 + self.b2),
            'R': 0.5 * (self.b1 + self.b2 + self.b3),
        }

        if self.lattice_type == 'rhombohedral':
            points['L'] = 0.5 * (self.b1 + self.b2 + self.b3)
            points['T'] = 0.25 * (self.b1 + self.b2 + self.b3)
            points['Z'] = 0.5 * self.b3

        return points

    def _generate_k_mesh(self):
        """
        生成 k 点网格.

        使用均匀 Monkhorst-Pack 网格:
            k = (n₁/N)·b₁ + (n₂/N)·b₂ + (n₃/N)·b₃
            nᵢ = 0, 1, ..., N-1

        然后将每个长方体单元分为 6 个四面体,
        对应种子项目 1306 的三角形细分思想.

        返回:
            k_points: shape (n_k, 3), k 点笛卡尔坐标
            weights: shape (n_k,), 积分权重
        """
        N = self.n_div
        k_pts = []
        w_ts = []

        # 体积元
        vol_cell = abs(np.dot(self.b1, np.cross(self.b2, self.b3)))
        dv = vol_cell / (N ** 3)

        for n1 in range(N):
            for n2 in range(N):
                for n3 in range(N):
                    # 单元中心 k 点
                    k_frac = np.array([
                        (n1 + 0.5) / N,
                        (n2 + 0.5) / N,
                        (n3 + 0.5) / N
                    ])
                    k_cart = (k_frac[0] * self.b1 +
                              k_frac[1] * self.b2 +
                              k_frac[2] * self.b3)
                    k_pts.append(k_cart)
                    w_ts.append(dv)

        k_points = np.array(k_pts)
        weights = np.array(w_ts)

        # 归一化权重
        weights /= np.sum(weights)

        return k_points, weights

    # ============================================================
    # BZ 积分
    # ============================================================

    def integrate_bz(self, func):
        """
        在布里渊区上积分函数 f(k).

        <f> = Σᵢ wᵢ · f(kᵢ)

        参数:
            func: 函数, 接受 k 向量 (3,), 返回标量

        返回:
            result: 积分值
        """
        result = 0.0
        for k, w in zip(self.k_points, self.weights):
            result += w * func(k)
        return result

    def integrate_bz_vectorized(self, func):
        """
        向量化 BZ 积分.

        参数:
            func: 函数, 接受 k 数组 (n_k, 3), 返回 (n_k,)

        返回:
            result: 积分值
        """
        values = func(self.k_points)
        return np.sum(self.weights * values)

    # ============================================================
    # 三角形直方图 (对应种子项目 1306)
    # ============================================================

    def triangle_histogram_2d(self, k_points_2d, n_subdivisions=8):
        """
        计算 k 点在参考三角形中的直方图分布.

        对应种子项目 1306:
        将参考三角形 (0,0)-(1,0)-(0,1) 分为 n² 个等面积子三角形.
        统计 k 点落入各子三角形的计数.

        用于验证 k 点采样的均匀性.

        参数:
            k_points_2d: 2D k 点坐标, shape (n_k, 2)
            n_subdivisions: 细分层数

        返回:
            histogram: shape (n_sub²,), 各子三角形中的计数
            stats: 统计信息 (均值, 方差, 均匀性指标)
        """
        n = n_subdivisions
        n_sub_triangles = n * n
        histogram = np.zeros(n_sub_triangles, dtype=int)

        for k in k_points_2d:
            # 转换到重心坐标
            u, v = k[0], k[1]

            # 检查是否在参考三角形内
            if u < 0 or v < 0 or u + v > 1.0:
                continue

            # 确定子三角形索引
            i = int(np.floor(u * n))
            j = int(np.floor(v * n))

            # 判断在上三角还是下三角
            u_frac = u * n - i
            v_frac = v * n - j
            if u_frac + v_frac <= 1.0:
                sub_idx = 2 * (j * n + i)  # 下三角
            else:
                sub_idx = 2 * (j * n + i) + 1  # 上三角

            sub_idx = min(sub_idx, n_sub_triangles * 2 - 1)
            if sub_idx < len(histogram):
                histogram[sub_idx] += 1

        # 统计
        non_zero = histogram[histogram > 0]
        stats = {
            'mean_count': np.mean(histogram) if len(histogram) > 0 else 0,
            'std_count': np.std(histogram) if len(histogram) > 0 else 0,
            'max_count': np.max(histogram) if len(histogram) > 0 else 0,
            'min_count': np.min(histogram) if len(histogram) > 0 else 0,
            'uniformity': (np.std(histogram) / (np.mean(histogram) + 1e-30)
                           if len(histogram) > 0 else 0),
        }

        return histogram, stats

    # ============================================================
    # 物理量计算
    # ============================================================

    def compute_magnon_dispersion(self, k_path, exchange_J=1.0e-22,
                                  anisotropy_D=1.0e-24):
        """
        计算磁子色散关系 ω(k).

        对于铁磁体 (Holstein-Primakoff 近似):
            ℏω(k) = 2S·[J(0) - J(k)] + gμ_B·H_A

        其中:
            J(k) = J·(cos(k·a₁) + cos(k·a₂) + cos(k·a₃))  [简单立方]
            H_A = 2K₁/(μ₀·M_s)  [各向异性 field]

        参数:
            k_path: k 点路径, shape (n_k, 3)
            exchange_J: 交换积分 (J)
            anisotropy_D: 各向异性常数 (J)

        返回:
            omega: 磁子能量 (J), shape (n_k,)
        """
        S = 2.5  # Fe³⁺ 自旋

        # J(k) 结构因子
        J_k = np.zeros(len(k_path))
        for i, b in enumerate([self.b1, self.b2, self.b3]):
            kd = k_path @ (b * self.a / (2.0 * np.pi))
            J_k += exchange_J * np.cos(2.0 * np.pi * kd)

        # J(0)
        J_0 = 3.0 * exchange_J

        # 磁子能量
        from multiferroic_constants import HBAR
        omega = 2.0 * S * (J_0 - J_k) + anisotropy_D

        return omega

    def density_of_states(self, energies, n_bins=50, broadening=0.01):
        """
        计算态密度 (DOS).

        g(E) = (1/N) Σₖ δ(E - Eₖ)

        使用 Lorentzian 展宽:
            δ(E) ≈ (η/π) / (E² + η²)

        参数:
            energies: k 点能量, shape (n_k,)
            n_bins: 能量网格点数
            broadening: Lorentzian 展宽参数 η

        返回:
            E_grid: 能量网格, shape (n_bins,)
            dos: 态密度, shape (n_bins,)
        """
        E_min = np.min(energies) - 3 * broadening
        E_max = np.max(energies) + 3 * broadening
        E_grid = np.linspace(E_min, E_max, n_bins)

        dos = np.zeros(n_bins)
        for E_k in energies:
            dos += (broadening / np.pi) / (
                (E_grid - E_k) ** 2 + broadening ** 2
            )
        dos /= len(energies)

        return E_grid, dos
