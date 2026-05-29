"""
lattice_occupation.py
晶格占据与多体构型模块

基于种子项目 1389_variomino 的核心算法：
- variomino_matrix: 构建平铺问题的线性系统 A x = b
- variomino_embed_list/number: 枚举嵌入方式
- variomino_variants: 旋转与反射变体

在离子通道问题中的应用：
将离子在通道离散格点上的占据视为“多格骨牌”（polyomino）平铺问题：
- 区域 R：通道内可容纳离子的离散位点集合
- 离子：具有特定“形状”和电荷的 variomino
- 约束：Pauli 不相容（每个位点至多一个离子）+ 静电排斥（同号离子不能相邻）

KcsA 滤器最多同时容纳 2-3 个 K+，且由于高浓度下相邻 K+ 之间的强静电排斥，
离子通常以“knock-on”方式交替占据，形成 S1-S3 或 S2-S4 的二聚体构型。
"""

import numpy as np
from combinatorial_stats import binomial_coefficient


class LatticeChannel:
    """
    一维/二维离散晶格通道模型。
    """
    def __init__(self, shape, binding_energies=None):
        """
        Parameters
        ----------
        shape : tuple
            晶格尺寸 (nz,) 或 (nx, nz)
        binding_energies : ndarray
            每个格点的结合自由能 (J)
        """
        self.shape = shape
        self.n_sites = np.prod(shape)
        if binding_energies is None:
            # 默认：滤器中间位点能量最低
            self.energies = np.zeros(shape)
            if len(shape) == 1:
                mid = shape[0] // 2
                for i in range(shape[0]):
                    self.energies[i] = -1.0e-20 * np.exp(-0.5 * ((i - mid) / 1.5) ** 2)
            else:
                mid_z = shape[1] // 2
                for j in range(shape[1]):
                    self.energies[:, j] = -1.0e-20 * np.exp(-0.5 * ((j - mid_z) / 1.5) ** 2)
        else:
            self.energies = binding_energies

    def valid_configurations(self, n_ions, min_distance=1):
        """
        枚举所有满足最小间距约束的离子占据构型。

        约束：
            - 恰好 n_ions 个位点被占据
            - 任意两个占据位点的距离 >= min_distance（Pauli + 静电排斥）

        Returns
        -------
        configs : list of ndarray
            每个元素为占据位点的索引列表
        """
        if len(self.shape) == 1:
            return self._valid_1d(self.n_sites, n_ions, min_distance)
        else:
            return self._valid_2d(self.shape, n_ions, min_distance)

    def _valid_1d(self, n, k, d):
        """
        一维晶格上间距至少为 d 的 k 个粒子的所有构型。
        采用递归生成（源自 variomino_embed_list 思想）。
        """
        configs = []

        def backtrack(start, chosen):
            if len(chosen) == k:
                configs.append(np.array(chosen))
                return
            for i in range(start, n):
                if len(chosen) == 0 or i - chosen[-1] >= d:
                    backtrack(i + 1, chosen + [i])

        backtrack(0, [])
        return configs

    def _valid_2d(self, shape, k, d):
        """
        二维晶格的简化版本：展平为一维后处理（忽略对角邻域）。
        """
        n = np.prod(shape)
        return self._valid_1d(n, k, d)

    def configuration_energy(self, config):
        """
        计算给定构型的总能量（包含结合能和离子间排斥）。

        E_total = Σ_i E_bind(i) + Σ_{i<j} V_Coulomb(i,j)

        简化 Coulomb 排斥：
            V_ij = (e^2 / 4π ε_0 ε_r) * (1 / r_ij)
        """
        e_charge = 1.602176634e-19
        eps0 = 8.854187817e-12
        eps_r = 40.0  # 滤器内有效介电常数
        coeff = e_charge ** 2 / (4.0 * np.pi * eps0 * eps_r)

        E = 0.0
        for idx in config:
            E += self.energies.flat[idx]

        for i in range(len(config)):
            for j in range(i + 1, len(config)):
                # 简化的 1D 距离（nm -> m）
                r_ij = abs(config[i] - config[j]) * 0.3e-9  # 每个格点间距 0.3 nm
                if r_ij > 0:
                    E += coeff / r_ij
        return E

    def partition_function(self, n_ions, T=300.0, min_distance=1):
        """
        计算固定离子数的正则配分函数：
            Z(N, V, T) = Σ_{合法构型} exp(-E_conf / k_B T)
        """
        kB = 1.380649e-23
        configs = self.valid_configurations(n_ions, min_distance)
        Z = 0.0
        for conf in configs:
            E = self.configuration_energy(conf)
            Z += np.exp(-E / (kB * T))
        return Z, configs

    def most_probable_configuration(self, n_ions, T=300.0, min_distance=1):
        """
        返回最概然构型及其概率。
        """
        Z, configs = self.partition_function(n_ions, T, min_distance)
        kB = 1.380649e-23
        probs = []
        for conf in configs:
            E = self.configuration_energy(conf)
            probs.append(np.exp(-E / (kB * T)) / Z)
        idx = int(np.argmax(probs))
        return configs[idx], probs[idx]


def knock_on_energy_barrier(dK_K=0.3e-9, dK_Na=0.25e-9):
    """
    计算 knock-on 机制的能垒差异。

    K+ 在滤器中的配位距离约为 0.28 nm，Na+ 为 0.24 nm。
    KcsA 滤器的羰基氧间距为 0.28-0.30 nm，恰好匹配 K+ 但大于 Na+，
    导致 Na+ 的结合能较弱，能垒较高。

    采用简化的 Lennard-Jones 型势：
        V(r) = 4ε [ (σ/r)^12 - (σ/r)^6 ]
    """
    # 简化参数
    epsilon = 1.0e-21  # J
    sigma_K = 0.28e-9
    sigma_Na = 0.24e-9

    def lj(r, sigma):
        x = sigma / r
        return 4.0 * epsilon * (x ** 12 - x ** 6)

    V_K = lj(dK_K, sigma_K)
    V_Na = lj(dK_Na, sigma_Na)
    return V_K, V_Na
