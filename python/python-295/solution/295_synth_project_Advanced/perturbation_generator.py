#!/usr/bin/env python3
"""
perturbation_generator.py
=========================
ICF 内爆初始扰动生成模块，融合种子项目:
  - [042] asa144: 随机列联表 (Boyett算法) → 随机扰动分布
  - [910] prime: 素数筛 → 用于构造准随机扰动模式

物理背景:
  ICF 内爆的初始扰动来源:
    1. 靶丸表面粗糙度 (RMS ~ 50 nm)
    2. 激光辐照不均匀性 (非均匀度 ~ 1%)
    3. 材料密度涨落
    4. 界面粗糙度 (烧蚀面, DT冰层)

  扰动谱通常用功率谱密度 (PSD) 描述:
    PSD(k) = A k^{-α}  (白噪声: α=0; 红噪声: α>0)
  或按球谐模式:
    <a_{lm}²> = σ_l²  (各模式独立)

  RT 不稳定性种子 = 初始扰动 × 增长因子
"""

import numpy as np


class PrimeBasedSeeding:
    """
    基于素数的伪随机扰动生成 (融合 [910] prime)。
    利用素数分布的 quasi-random 特性生成低差异扰动序列。
    """

    @staticmethod
    def prime_sieve(n):
        """
        Eratosthenes 素数筛。
        返回 ≤ n 的所有素数。
        """
        if n < 2:
            return np.array([], dtype=np.int64)
        is_prime = np.ones(n + 1, dtype=bool)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                is_prime[i * i:n + 1:i] = False
        return np.where(is_prime)[0].astype(np.int64)

    @staticmethod
    def nth_prime(n):
        """
        返回第 n 个素数 (1-indexed)。
        使用素数定理估计上界, 然后筛。
        p_n ≈ n ln(n) + n ln(ln(n))
        """
        if n <= 0:
            return 0
        if n <= 5:
            return [2, 2, 3, 5, 5][n]
        # 估计上界
        upper = int(n * (np.log(n) + np.log(np.log(n)) + 2)) + 10
        primes = PrimeBasedSeeding.prime_sieve(upper)
        while len(primes) < n:
            upper *= 2
            primes = PrimeBasedSeeding.prime_sieve(upper)
        return primes[n - 1]

    @staticmethod
    def van_der_corput(n, base=2):
        """
        van der Corput 序列 (低差异序列)。
        用于生成准随机扰动。
        x_k = Σ_i a_i(k) base^{-(i+1)}
        其中 a_i(k) 是 k 在 base 进制下的第 i 位。
        """
        result = np.zeros(n)
        for k in range(n):
            f = 1.0
            r = 0.0
            i = k + 1
            while i > 0:
                f /= base
                r += f * (i % base)
                i = i // base
            result[k] = r
        return result

    @staticmethod
    def halton_sequence(n, dimensions=2):
        """
        Halton 序列 (多维低差异序列)。
        各维度使用不同素数作为 base。
        """
        primes = PrimeBasedSeeding.prime_sieve(100)[:dimensions]
        result = np.zeros((n, dimensions))
        for d in range(dimensions):
            result[:, d] = PrimeBasedSeeding.van_der_corput(n, base=int(primes[d]))
        return result


class RandomContingencyPerturbation:
    """
    基于随机列联表的扰动生成 (融合 [042] asa144, Boyett算法)。
    用于生成满足边际约束的随机扰动分布。

    物理应用:
      将靶丸表面划分为 N 个区域, 每个区域分配固定的扰动量。
      在满足总扰动量约束下, 随机分配各区域的扰动值。
    """

    @staticmethod
    def boyett_rcont(row_totals, col_totals, rng=None):
        """
        Boyett 算法 AS 144: 生成给定边际的随机 R×C 列联表。

        算法:
          1. 创建一个长度为 N_total 的数组
          2. 前 n_1 个标记为行 1, 接下来 n_2 个标记为行 2, ...
          3. 随机排列后, 统计每列中各行的计数
          4. 得到满足边际约束的随机表

        参数:
            row_totals: 行边际, shape=(R,)
            col_totals: 列边际, shape=(C,)
        返回:
            matrix: (R, C) 随机列联表
        """
        if rng is None:
            rng = np.random.default_rng(42)

        row_totals = np.asarray(row_totals, dtype=np.int64)
        col_totals = np.asarray(col_totals, dtype=np.int64)

        R = len(row_totals)
        C = len(col_totals)
        N_total = np.sum(row_totals)

        if N_total != np.sum(col_totals):
            raise ValueError(f"行边际和 ({N_total}) ≠ 列边际和 ({np.sum(col_totals)})")

        # 创建标记数组
        labels = np.zeros(N_total, dtype=np.int64)
        idx = 0
        for r in range(R):
            labels[idx:idx + row_totals[r]] = r
            idx += row_totals[r]

        # 随机排列
        rng.shuffle(labels)

        # 统计列联表
        matrix = np.zeros((R, C), dtype=np.int64)
        # 将位置映射到列
        col_idx = np.zeros(N_total, dtype=np.int64)
        idx = 0
        for c in range(C):
            col_idx[idx:idx + col_totals[c]] = c
            idx += col_totals[c]

        for i in range(N_total):
            matrix[labels[i], col_idx[i]] += 1

        return matrix

    @staticmethod
    def generate_surface_perturbation(n_theta, n_phi, total_amplitude, rng=None):
        """
        生成靶丸表面扰动分布。
        在满足总振幅约束下, 随机分配各 (θ,φ) 区域的扰动量。

        参数:
            n_theta: 极角分辨率
            n_phi: 方位角分辨率
            total_amplitude: 总扰动振幅 (归一化)
        返回:
            perturbation: (n_theta, n_phi) 扰动场
        """
        if rng is None:
            rng = np.random.default_rng(42)

        # 行边际: 按 sin(θ) 加权 (球面面积元)
        theta = np.linspace(0, np.pi, n_theta + 1)
        row_weights = np.diff(-np.cos(theta))  # sin θ Δθ
        row_totals = np.maximum(1, (row_weights / row_weights.sum() * total_amplitude * 100).astype(int))
        # 调整使总和一致
        row_totals[-1] += int(total_amplitude * 100) - row_totals.sum()
        row_totals = np.maximum(1, row_totals)

        col_totals = np.full(n_phi, row_totals.sum() // n_phi, dtype=np.int64)
        col_totals[-1] += row_totals.sum() - col_totals.sum() * n_phi

        matrix = RandomContingencyPerturbation.boyett_rcont(row_totals, col_totals, rng)
        return matrix.astype(np.float64) / max(matrix.sum(), 1)


class PerturbationSpectrum:
    """
    扰动功率谱生成器。
    """

    @staticmethod
    def power_law_spectrum(k, amplitude, exponent=-2.0):
        """
        幂律功率谱:
          P(k) = A k^α
        α=0: 白噪声
        α=-2: 红噪声
        α=-11/3: Kolmogorov 湍流谱
        """
        k = np.maximum(np.asarray(k, dtype=np.float64), 1.0)
        return amplitude * k ** exponent

    @staticmethod
    def generate_mode_amplitudes(max_mode, sigma_l_func, rng=None):
        """
        生成各球谐模式的随机振幅。
        <a_{lm}²> = σ_l²
        a_{lm} ~ N(0, σ_l²)

        参数:
            max_mode: 最大模式阶数
            sigma_l_func: σ_l 关于 l 的函数
        返回:
            modes: dict {(l, m): a_lm}
        """
        if rng is None:
            rng = np.random.default_rng(42)

        modes = {}
        for l in range(max_mode + 1):
            sigma_l = sigma_l_func(l)
            for m in range(-l, l + 1):
                if l == 0:
                    modes[(l, m)] = sigma_l  # l=0 为确定性
                else:
                    a_lm = rng.normal(0, sigma_l) + 1j * rng.normal(0, sigma_l)
                    a_lm /= np.sqrt(2)
                    modes[(l, m)] = a_lm
        return modes

    @staticmethod
    def rms_perturbation(modes):
        """
        计算扰动的 RMS 值。
        RMS = sqrt(Σ_{l,m} |a_{lm}|² / (4π))
        """
        total = sum(abs(a_lm) ** 2 for a_lm in modes.values())
        return np.sqrt(total / (4 * np.pi))


class InitialPerturbationFactory:
    """
    ICF 内爆初始扰动生成工厂。
    """

    @staticmethod
    def surface_roughness_perturbation(n_theta, n_phi, rms_roughness_cm,
                                         spectral_exponent=-2.0, seed=42):
        """
        生成靶丸表面粗糙度扰动。

        参数:
            n_theta, n_phi: 角度分辨率
            rms_roughness_cm: RMS 粗糙度 [cm]
            spectral_exponent: 功率谱指数
            seed: 随机种子
        返回:
            perturbation: (n_theta, n_phi) 扰动场 [cm]
        """
        rng = np.random.default_rng(seed)

        # 生成球谐模式振幅
        sigma_l = lambda l: rms_roughness_cm * (max(l, 1)) ** (spectral_exponent / 2.0)
        ps = PerturbationSpectrum()
        modes = ps.generate_mode_amplitudes(min(n_theta, n_phi) // 2, sigma_l, rng)

        # 合成扰动场
        perturbation = np.zeros((n_theta, n_phi))
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

        from scipy.special import sph_harm
        for (l, m), a_lm in modes.items():
            for i, th in enumerate(theta):
                for j, ph in enumerate(phi):
                    Y_lm = sph_harm(m, l, ph, th)
                    perturbation[i, j] += np.real(a_lm * Y_lm)

        # 归一化到指定 RMS
        current_rms = np.sqrt(np.mean(perturbation ** 2))
        if current_rms > 1e-30:
            perturbation *= rms_roughness_cm / current_rms

        return perturbation
