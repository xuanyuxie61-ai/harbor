"""
free_energy.py
自由能计算模块

基于种子项目的核心算法：
- 1103_sparse_grid_cc: 高维稀疏网格积分
- 1124_sphere_monte_carlo: 球面平均积分
- 654_lattice_rule: 格点规则积分

在离子通道问题中的应用：
计算离子跨膜传输的自由能面（Potential of Mean Force, PMF）：

    G(ξ) = -k_B T ln P(ξ) + C

其中 ξ 为反应坐标（如离子沿通道轴的位置）。

多反应坐标自由能面：
    G(ξ1, ξ2) = -k_B T ln P(ξ1, ξ2)

采用稀疏网格和格点规则进行高维积分，计算配分函数：
    Z = ∫ exp(-β U(r)) dr
"""

import numpy as np
from monte_carlo_integrator import integrate_sparse_grid, fibonacci_lattice_2d
from special_functions import log_gamma_lanczos


class FreeEnergySurface:
    """
    自由能面计算与分析。
    """
    def __init__(self, temperature=300.0):
        self.kB = 1.380649e-23
        self.T = temperature
        self.beta = 1.0 / (self.kB * temperature)

    def pmf_1d_from_histogram(self, positions, bins=50, range_z=None):
        """
        从一维位置直方图计算 PMF。

        G(z) = -k_B T ln [ P(z) / P_max ]
        """
        hist, bin_edges = np.histogram(positions, bins=bins, range=range_z, density=True)
        z_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # 避免 log(0)
        hist = np.maximum(hist, 1e-30)
        pmf = -self.kB * self.T * np.log(hist / np.max(hist))
        return z_centers, pmf

    def pmf_2d(self, pos1, pos2, bins=30):
        """
        二维自由能面 G(z1, z2)。
        """
        hist, xedges, yedges = np.histogram2d(pos1, pos2, bins=bins, density=True)
        hist = np.maximum(hist, 1e-30)
        pmf = -self.kB * self.T * np.log(hist / np.max(hist))
        xcenters = 0.5 * (xedges[:-1] + xedges[1:])
        ycenters = 0.5 * (yedges[:-1] + yedges[1:])
        return xcenters, ycenters, pmf

    def barrier_height(self, z, pmf):
        """
        计算自由能垒高度 ΔG‡ = G_TS - G_min。
        """
        g_min = np.min(pmf)
        g_max = np.max(pmf)
        return g_max - g_min

    def integration_factor(self, z, pmf):
        """
        计算传输系数的积分因子：
            κ = ∫ exp(-β G(z)) dz
        """
        dz = z[1] - z[0]
        integral = np.sum(np.exp(-self.beta * pmf)) * dz
        return integral


def partition_function_integral(potential_func, dim, level_max=3):
    """
    使用稀疏网格计算多维配分函数：
        Z = ∫_{[-1,1]^d} exp(-β V(x)) dx

    其中 potential_func 接受 d 维向量，返回势能 (J)。
    """
    kB = 1.380649e-23
    T = 300.0
    beta = 1.0 / (kB * T)

    def integrand(x):
        return np.exp(-beta * potential_func(x))

    result = integrate_sparse_grid(integrand, dim, level_max)
    return result


def sphere_solvation_free_energy(charge, radius, epsilon=78.5, T=300.0):
    """
    计算 Born 溶剂化自由能（球面蒙特卡洛积分解析形式）：

        ΔG_solv = - (1 - 1/ε) * (q^2 / 8π ε_0 r_ion)

    这是将点电荷放入介电连续介质中的经典结果。
    """
    e_charge = 1.602176634e-19
    eps0 = 8.854187817e-12
    kB = 1.380649e-23

    delta_G = -(1.0 - 1.0 / epsilon) * (charge ** 2 * e_charge ** 2) / (8.0 * np.pi * eps0 * radius)
    return delta_G


def debye_huckel_excess_energy(ionic_strength, charge, radius, T=300.0):
    """
    Debye-Hückel 极限定律下的过量自由能：

        ΔG_excess = - (N_A e^2 κ z^2) / (8π ε_0 ε_r) * (1 / (1 + κ a))

    其中 κ 为 Debye 长度倒数，a 为离子有效半径。
    """
    NA = 6.02214076e23
    e_charge = 1.602176634e-19
    eps0 = 8.854187817e-12
    eps_r = 78.5
    kB = 1.380649e-23

    kappa = np.sqrt(2000.0 * NA * e_charge ** 2 * ionic_strength / (eps0 * eps_r * kB * T))
    numerator = NA * e_charge ** 2 * kappa * charge ** 2
    denominator = 8.0 * np.pi * eps0 * eps_r * (1.0 + kappa * radius)
    return -numerator / denominator


def selective_permeability_ratio(dG_k, dG_na, D_k=1.96e-9, D_na=1.33e-9, T=300.0):
    """
    基于 Eyring 过渡态理论的通透选择性比值：

        P_K / P_Na = (D_K / D_Na) * exp( -(ΔG_K‡ - ΔG_Na‡) / k_B T )

    其中 ΔG‡ 为离子通过选择性滤器的自由能垒。
    """
    kB = 1.380649e-23
    ratio = (D_k / D_na) * np.exp(-(dG_k - dG_na) / (kB * T))
    return ratio
