# -*- coding: utf-8 -*-
"""
dos_histogram.py — 态密度计算与直方图统计
============================================
核心科学问题: 从离散能级计算二维异质结的态密度 (DOS),
使用直方图方法和 Gaussian 展宽.

融合种子项目:
  - 539_histogram_discrete: 离散直方图 PDF/CDF 构建
  - 542_histogram_pdf_2d_sample: 2D 直方图采样

数学基础:
  二维自由粒子态密度 (抛物线带):
    g_2D(E) = m* / (π·ℏ²)  (E > E_c, 阶梯函数)

  量子约束子带:
    g_2D(E) = sum_n (m*/(π·ℏ²)) · θ(E - E_n)

  Gaussian 展宽:
    g(E) = sum_n (1/(σ·√(2π))) · exp(-(E-E_n)²/(2σ²))
"""

import numpy as np
from high_order_fd import HBAR, M0, E0


class DOSCalculator:
    """
    态密度计算器.

    支持:
    1. 理想 2D 阶梯 DOS
    2. 离散能级 Gaussian 展宽
    3. 直方图统计方法
    4. 联合态密度 (JDOS) 用于光学跃迁
    """

    def __init__(self, energies, degeneracy=None, broadening=None):
        """
        参数
        ----
        energies : ndarray
            子带能级 [eV]
        degeneracy : ndarray or None
            简并度 (默认全1)
        broadening : float or None
            Gaussian 展宽 [eV] (默认自动)
        """
        self.energies = np.sort(np.asarray(energies, dtype=np.float64))
        self.N = len(energies)
        self.degeneracy = degeneracy if degeneracy is not None else np.ones(self.N)

        if broadening is None:
            # 自动展宽: 最小能级间距的 1/5
            if self.N >= 2:
                dE_min = np.min(np.diff(self.energies))
                self.sigma = max(dE_min / 5.0, 1e-5)
            else:
                self.sigma = 0.01
        else:
            self.sigma = broadening

    def ideal_2d_dos(self, E_array, m_star):
        """
        理想二维抛物线带态密度:
            g_2D(E) = g_v * g_s * m* / (π·ℏ²)

        其中:
          g_v = 谷简并 (TMDC: 2, K/K' 谷)
          g_s = 自旋简并 (= 1 若无自旋简并)
          m* = 有效质量 [m0]

        返回 [1/(eV·m²)]
        """
        g_valley = 2  # K, K' 谷
        g_spin = 1    # 自旋极化

        m_kg = m_star * M0
        # g_2D = m* / (π·ℏ²) [1/(J·m²)] → [1/(eV·m²)]
        g0 = g_valley * g_spin * m_kg / (np.pi * HBAR**2) * E0

        return np.full_like(E_array, g0)

    def subband_dos(self, E_array, m_star):
        """
        量子约束子带态密度 (阶梯函数 + 展宽):
            g(E) = sum_n g_n · θ(E - E_n) · m*/(πℏ²)

        展宽后:
            g(E) = sum_n g_n/(σ√(2π)) · exp(-(E-E_n)²/(2σ²)) · m*/(πℏ²)
        """
        m_kg = m_star * M0
        g_2d_prefactor = m_kg / (np.pi * HBAR**2) * E0  # [1/(eV·m²)]

        dos = np.zeros_like(E_array)
        for i, (E_n, g_n) in enumerate(zip(self.energies, self.degeneracy)):
            # Gaussian 展宽
            x = (E_array - E_n) / self.sigma
            gaussian = np.exp(-0.5 * x**2) / (self.sigma * np.sqrt(2 * np.pi))
            dos += g_n * gaussian

        dos *= g_2d_prefactor
        return dos

    def histogram_dos(self, E_array, n_bins=None):
        """
        直方图方法计算 DOS (融合 539_histogram_discrete).

        将能级分布构建为分段线性 PDF:
          1. 统计每个 bin 中的能级数
          2. 归一化得到概率密度
          3. 转换为态密度

        参数
        ----
        E_array : ndarray
            能量网格
        n_bins : int or None
            直方图 bin 数
        """
        if n_bins is None:
            n_bins = max(int(np.sqrt(self.N)) * 2, 10)

        E_min = np.min(self.energies) - 3 * self.sigma
        E_max = np.max(self.energies) + 3 * self.sigma

        # 构建直方图
        counts, bin_edges = np.histogram(
            self.energies, bins=n_bins,
            weights=self.degeneracy,
            range=(E_min, E_max))

        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        dE = bin_edges[1] - bin_edges[0]

        # 归一化为密度
        total_weight = np.sum(self.degeneracy)
        if total_weight > 0:
            pdf = counts / (total_weight * dE)
        else:
            pdf = counts / max(dE, 1e-30)

        # 插值到目标能量网格
        dos = np.interp(E_array, bin_centers, pdf, left=0, right=0)

        return dos

    def cdf(self, E_array):
        """
        累积态密度 (CDF):
            N(E) = integral_{-inf}^{E} g(E') dE'
        融合 539_histogram_discrete 的 CDF 计算.
        """
        dos = self.subband_dos(E_array, m_star=1.0)  # 归一化
        # 梯形积分
        cdf = np.zeros_like(E_array)
        for i in range(1, len(E_array)):
            dE = E_array[i] - E_array[i - 1]
            cdf[i] = cdf[i - 1] + 0.5 * (dos[i] + dos[i - 1]) * dE
        # 归一化
        if cdf[-1] > 0:
            cdf /= cdf[-1]
        return cdf

    def joint_dos(self, E_c_energies, E_v_energies, E_array,
                  m_e_star, m_h_star, matrix_element=1.0):
        """
        联合态密度 (JDOS) 用于光学跃迁:
            J(E) = sum_{c,v} |M_cv|² · δ(E - (E_c - E_v))

        对于抛物线带:
            J_2D(E) = m_r / (π·ℏ²) · θ(E - E_g)
        其中 m_r = m_e*m_h/(m_e+m_h) 是约化质量.

        参数
        ----
        E_c_energies : ndarray
            导带子带能级
        E_v_energies : ndarray
            价带子带能级
        E_array : ndarray
            光子能量网格
        matrix_element : float
            动量矩阵元 (相对单位)
        """
        m_r_m0 = (m_e_star * m_h_star) / (m_e_star + m_h_star)
        m_r_kg = m_r_m0 * M0
        jdos_prefactor = m_r_kg / (np.pi * HBAR**2) * E0 * matrix_element**2

        jdos = np.zeros_like(E_array)
        for Ec in E_c_energies:
            for Ev in E_v_energies:
                E_transition = Ec - Ev  # 跃迁能量
                # Gaussian 展宽 δ 函数
                x = (E_array - E_transition) / self.sigma
                delta = np.exp(-0.5 * x**2) / (self.sigma * np.sqrt(2 * np.pi))
                jdos += delta

        jdos *= jdos_prefactor
        return jdos

    def fermi_level(self, T, carrier_density, m_star, band_edge):
        """
        费米能级确定 (二维):
            n_2D = integral g(E) · f(E, E_F, T) dE

        其中 f 是费米-狄拉克分布:
            f(E) = 1 / (exp((E-E_F)/(kB·T)) + 1)

        对于非简并情形:
            n_2D ≈ g_2D · kB·T · ln(1 + exp(-(E_c-E_F)/(kB·T)))

        参数
        ----
        T : float
            温度 [K]
        carrier_density : float
            面载流子密度 [1/m²]
        m_star : float
            有效质量 [m0]
        band_edge : float
            带边能量 [eV]
        """
        kB_T = 8.617333262e-5 * T  # [eV]
        m_kg = m_star * M0
        g_2d = m_kg / (np.pi * HBAR**2)  # [1/(J·m²)]

        # 解析反解 (非简并近似)
        # n = g_2D * kB_T * ln(1 + exp((E_F-E_c)/kB_T))
        # => E_F = E_c + kB_T * ln(exp(n/(g_2D*kB_T)) - 1)
        arg = carrier_density / (g_2d * kB_T * E0)
        if arg > 0:
            E_F = band_edge + kB_T * np.log(np.exp(arg) - 1 + 1e-30)
        else:
            E_F = band_edge - 10 * kB_T

        return E_F

    def sample_energies(self, n_samples, seed=42):
        """
        从 DOS 分布中采样能量 (融合 542_histogram_pdf_2d_sample).
        使用逆变换采样.
        """
        rng = np.random.RandomState(seed)
        E_min = np.min(self.energies) - 5 * self.sigma
        E_max = np.max(self.energies) + 5 * self.sigma
        E_grid = np.linspace(E_min, E_max, 1000)

        dos = self.histogram_dos(E_grid)
        # 累积分布
        cdf = np.cumsum(dos) * (E_grid[1] - E_grid[0])
        if cdf[-1] > 0:
            cdf /= cdf[-1]
        else:
            cdf = np.linspace(0, 1, len(E_grid))

        # 逆变换
        u = rng.uniform(0, 1, n_samples)
        samples = np.interp(u, cdf, E_grid)
        return samples
