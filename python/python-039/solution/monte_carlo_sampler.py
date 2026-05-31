"""
monte_carlo_sampler.py
QGP蒙特卡洛部分子采样与组合计数

基于种子项目:
- 202_combo: 组合数学（Stirling数、Bell数、子集枚举、分割）→ 部分子多重数分布
- 540_histogram_display: 直方图统计 → 物理量分布分析

物理应用:
1. 部分子簇射中的组合计数：将n个部分子分配到k个喷注的方式数 = S(n,k) (Stirling数第二类)
2. 色单态组合与色分解：n个胶子的色单态组合数 = Bell(n)
3. 喷注能量谱的直方图分析
4. 事件多重数分布: Poisson-Bose组合统计
"""

import numpy as np
from typing import Tuple, List, Dict
from collections import Counter


class CombinatorialPhysics:
    """
    将组合数学应用于QGP部分子系统的计数问题。
    """

    @staticmethod
    def stirling_numbers_second_kind(m: int, n: int) -> np.ndarray:
        """
        计算Stirling数第二类 S(i,j)。

        递推关系:
        S(n,k) = k · S(n-1,k) + S(n-1,k-1)
        S(0,0) = 1, S(n,0) = 0 (n>0)

        物理意义: 将n个不可区分部分子分配到k个可区分喷注的方式数。

        Parameters
        ----------
        m : int
            最大n值
        n : int
            最大k值

        Returns
        -------
        np.ndarray
            Stirling数表 (m+1, n+1)
        """
        if m < 0 or n < 0:
            return np.zeros((1, 1))
        s = np.zeros((m + 1, n + 1), dtype=np.int64)
        s[0, 0] = 1
        for i in range(1, m + 1):
            for j in range(1, min(i, n) + 1):
                s[i, j] = j * s[i - 1, j] + s[i - 1, j - 1]
        return s

    @staticmethod
    def bell_numbers(m: int) -> np.ndarray:
        """
        计算Bell数 B(n)。

        B(n) = Σ_{k=0}^n S(n,k)

        物理意义: n个胶子形成色单态的总方式数。

        Parameters
        ----------
        m : int
            最大n值

        Returns
        -------
        np.ndarray
            Bell数数组
        """
        s = CombinatorialPhysics.stirling_numbers_second_kind(m, m)
        bell = np.sum(s, axis=1)
        return bell

    @staticmethod
    def partition_function(n: int, max_k: int) -> np.ndarray:
        """
        计算整数分拆函数 p(n,k): 将n分拆为最多k个正整数之和的方式数。

        物理意义: 将总能量E分配到k个部分子的相空间体积。

        Parameters
        ----------
        n : int
            待分拆的整数
        max_k : int
            最大部分数

        Returns
        -------
        np.ndarray
            分拆数表
        """
        p = np.zeros((n + 1, max_k + 1), dtype=np.int64)
        p[0, 0] = 1
        for i in range(1, n + 1):
            for k in range(1, min(i, max_k) + 1):
                p[i, k] = p[i - 1, k - 1] + p[i - k, k]
        return p

    @staticmethod
    def subset_sum_count(weights: np.ndarray, target: float,
                         tolerance: float = 1e-6) -> int:
        """
        计算子集和等于目标值的子集数量 (基于背包问题/backtrack思想)。

        物理意义: 给定一组部分子能量，求总能量为E的子集数。

        Parameters
        ----------
        weights : np.ndarray
            权重数组
        target : float
            目标和
        tolerance : float
            容差

        Returns
        -------
        int
            子集数量
        """
        n = len(weights)
        count = 0
        # 使用动态规划替代回溯以提高效率
        dp = {0.0: 1}
        for w in weights:
            new_dp = dict(dp)
            for s, c in dp.items():
                new_sum = s + w
                if abs(new_sum - target) <= tolerance:
                    count += c
                new_dp[new_sum] = new_dp.get(new_sum, 0) + c
            dp = new_dp
        # 精确匹配已在循环中计数
        return count


class HistogramAnalysis:
    """
    物理量分布的直方图分析。
    """

    def __init__(self, data: np.ndarray, n_bins: int = 50,
                 range_limits: Tuple[float, float] = (0.0, 1.0)):
        """
        初始化直方图分析器。

        Parameters
        ----------
        data : np.ndarray
            输入数据
        n_bins : int
            直方图 bins 数量
        range_limits : Tuple[float, float]
            数据范围
        """
        self.data = np.asarray(data)
        self.n_bins = max(2, n_bins)
        self.range_limits = range_limits
        self.counts, self.bin_edges = np.histogram(
            self.data, bins=self.n_bins, range=self.range_limits
        )
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        self.bin_width = self.bin_edges[1] - self.bin_edges[0]

    def probability_density(self) -> np.ndarray:
        """
        计算概率密度函数估计。

        f(x) = counts / (N · Δx)

        Returns
        -------
        np.ndarray
            概率密度
        """
        n_total = np.sum(self.counts)
        if n_total == 0 or self.bin_width == 0:
            return np.zeros_like(self.counts, dtype=float)
        return self.counts.astype(float) / (n_total * self.bin_width)

    def cumulative_distribution(self) -> np.ndarray:
        """
        计算累积分布函数。

        F(x) = ∫_{-∞}^x f(x') dx'

        Returns
        -------
        np.ndarray
            CDF值
        """
        pdf = self.probability_density()
        cdf = np.cumsum(pdf) * self.bin_width
        cdf = np.clip(cdf, 0.0, 1.0)
        return cdf

    def mean(self) -> float:
        """
        计算样本均值。
        """
        return float(np.mean(self.data))

    def variance(self) -> float:
        """
        计算样本方差。
        """
        return float(np.var(self.data, ddof=1))

    def skewness(self) -> float:
        """
        计算偏度 (三阶标准化矩)。

        γ₁ = ⟨(x - μ)³⟩ / σ³
        """
        mu = self.mean()
        std = np.sqrt(self.variance())
        if std < 1e-15:
            return 0.0
        gamma1 = np.mean((self.data - mu) ** 3) / (std ** 3)
        return float(gamma1)

    def kurtosis(self) -> float:
        """
        计算超额峰度。

        κ = ⟨(x - μ)⁴⟩ / σ⁴ - 3
        """
        mu = self.mean()
        var = self.variance()
        if var < 1e-15:
            return 0.0
        kappa = np.mean((self.data - mu) ** 4) / (var ** 2) - 3.0
        return float(kappa)

    def moments(self, max_order: int = 4) -> Dict[int, float]:
        """
        计算各阶中心矩。

        Parameters
        ----------
        max_order : int
            最大阶数

        Returns
        -------
        Dict[int, float]
            阶数到矩值的映射
        """
        mu = self.mean()
        result = {}
        for k in range(1, max_order + 1):
            result[k] = float(np.mean((self.data - mu) ** k))
        return result


class PartonCascade:
    """
    部分子级联的蒙特卡洛模拟。
    """

    def __init__(self, alpha_s: float = 0.3, q0: float = 1.0):
        """
        初始化级联参数。

        Parameters
        ----------
        alpha_s : float
            强耦合常数
        q0 : float
            红外截断 [GeV]
        """
        self.alpha_s = alpha_s
        self.q0 = q0

    def splitting_probability(self, z: float, t: float,
                              splitting_type: str = 'gg') -> float:
        """
        DGLAP分裂函数 (Leading Order近似)。

        P_{g→gg}(z) = 2C_A [z/(1-z)_+ + (1-z)/z + z(1-z)]
        P_{q→qg}(z) = C_F [(1+z²)/(1-z)_+]

        Parameters
        ----------
        z : float
            能量份额
        t : float
            演化参数
        splitting_type : str
            分裂类型

        Returns
        -------
        float
            分裂概率密度
        """
        if z <= 0.0 or z >= 1.0:
            return 0.0
        if splitting_type == 'gg':
            ca = 3.0
            p = 2.0 * ca * (z / (1.0 - z + 1e-10) +
                            (1.0 - z) / (z + 1e-10) +
                            z * (1.0 - z))
        elif splitting_type == 'qg':
            cf = 4.0 / 3.0
            p = cf * ((1.0 + z ** 2) / (1.0 - z + 1e-10))
        else:
            p = 0.0
        # 包含alpha_s/t演化核
        return self.alpha_s * p / (2.0 * np.pi * max(t, 1e-10))

    def multiplicity_distribution(self, E_init: float,
                                  n_events: int = 1000) -> np.ndarray:
        """
        模拟部分子多重数分布。

        简化模型: 每次分裂产生2个部分子，总多重数服从
        近似的Poisson-like分布。

        Parameters
        ----------
        E_init : float
            初始能量 [GeV]
        n_events : int
            事件数

        Returns
        -------
        np.ndarray
            多重数数组
        """
        multiplicities = []
        for _ in range(n_events):
            # 简化: 平均多重数 ~ log(E/q0)
            if E_init <= self.q0:
                multiplicities.append(1)
                continue
            n_avg = 2.0 * np.log(E_init / self.q0)
            n_avg = max(1.0, n_avg)
            # Poisson抽样
            n = np.random.poisson(n_avg)
            n = max(1, n)
            multiplicities.append(n)
        return np.array(multiplicities)

    def jet_energy_spectrum(self, energies: np.ndarray,
                            n_bins: int = 40) -> HistogramAnalysis:
        """
        分析喷注能量谱。

        Parameters
        ----------
        energies : np.ndarray
            喷注能量样本 [GeV]
        n_bins : int
            bin数量

        Returns
        -------
        HistogramAnalysis
            直方图分析器
        """
        e_max = np.max(energies) * 1.1 if len(energies) > 0 else 10.0
        hist = HistogramAnalysis(energies, n_bins=n_bins,
                                  range_limits=(0.0, e_max))
        return hist

    def color_singlet_combinatorics(self, n_gluons: int) -> Dict[str, int]:
        """
        计算n个胶子的色单态组合数。

        在SU(3)色群中，n个胶子形成单态的方式数:
        N_singlet = B(n) · f_{SU(3)}(n)

        Parameters
        ----------
        n_gluons : int
            胶子数

        Returns
        -------
        Dict[str, int]
            组合数信息
        """
        if n_gluons < 0:
            n_gluons = 0
        bell = CombinatorialPhysics.bell_numbers(n_gluons)
        # SU(3)色因子近似
        color_factor = 1 if n_gluons == 0 else 8 ** (n_gluons - 1)
        return {
            'n_gluons': n_gluons,
            'bell_number': int(bell[n_gluons]),
            'color_configurations': color_factor,
            'total_singlets_estimate': int(min(bell[n_gluons] * color_factor, 2**63 - 1))
        }
