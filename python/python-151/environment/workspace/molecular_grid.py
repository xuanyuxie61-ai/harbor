"""
molecular_grid.py
=================
分子轨道积分网格生成与Centroidal Voronoi Tessellation优化

原项目映射:
- 558_hypercube_grid: 多维超立方体笛卡尔积网格生成
- 263_cvtm_1d: 一维镜像周期CVT（Lloyd算法迭代优化节点分布）

科学功能:
本模块生成用于计算分子哈密顿量双电子积分的数值积分网格。
使用超立方体网格覆盖多维参数空间，并通过CVT优化将网格节点
适配到电子密度分布，提高高维数值积分的精度和效率。
在VQE中，这些网格用于近似计算分子轨道的一体和双体积分。
"""

import numpy as np
from typing import Tuple, List, Optional


def hypercube_grid(m: int, ns: np.ndarray, a: np.ndarray,
                   b: np.ndarray, c: Optional[np.ndarray] = None) -> np.ndarray:
    """
    生成m维超立方体内部网格点，对应 558_hypercube_grid/hypercube_grid。

    总点数 n = product(ns)。
    对每一维度i，在[a[i], b[i]]上生成ns[i]个点，
    中心化模式c[i]控制节点分布:
        1: 包含端点的均匀分布
        2: 内部分布 (1/(n+1), ..., n/(n+1))
        3: 包含左端点
        4: 包含右端点
        5: 中点分布 ((2i-1)/(2n))

    返回:
        x: (m, n) 网格点坐标
    """
    ns = np.asarray(ns, dtype=int)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if c is None:
        c = np.ones(m, dtype=int)
    else:
        c = np.asarray(c, dtype=int)

    if ns.shape[0] != m or a.shape[0] != m or b.shape[0] != m:
        raise ValueError("维度参数长度不一致")
    n = int(np.prod(ns))
    x = np.zeros((m, n))

    for i in range(m):
        s = ns[i]
        xs = np.zeros(s)
        for j in range(s):
            if c[i] == 1:
                if s == 1:
                    xs[j] = 0.5 * (a[i] + b[i])
                else:
                    xs[j] = ((s - 1 - j) * a[i] + j * b[i]) / (s - 1)
            elif c[i] == 2:
                xs[j] = ((s - j) * a[i] + (j + 1) * b[i]) / (s + 1)
            elif c[i] == 3:
                xs[j] = ((s - j) * a[i] + (j - 1) * b[i]) / s
            elif c[i] == 4:
                xs[j] = ((s - 1 - j) * a[i] + (j + 1) * b[i]) / s
            elif c[i] == 5:
                xs[j] = ((2 * s - 2 * j - 1) * a[i] + (2 * j + 1) * b[i]) / (2 * s)
        # 笛卡尔积：将第i维的s个点与已有的维度做张量积
        if i == 0:
            x[i, :] = np.repeat(xs, int(np.prod(ns[i + 1:])))
            if i + 1 < m:
                tile_size = int(np.prod(ns[i + 1:]))
                x[i, :] = np.tile(np.repeat(xs, tile_size), int(np.prod(ns[:i])))
        else:
            repeats = int(np.prod(ns[i + 1:]))
            tiles = int(np.prod(ns[:i]))
            x[i, :] = np.tile(np.repeat(xs, repeats), tiles)
    return x


def cvtm_1d_optimize(g_num: int, it_num: int = 50, s_num: int = 5000,
                     density_func: Optional[callable] = None) -> np.ndarray:
    """
    一维镜像周期CVT优化，对应 263_cvtm_1d/cvtm_1d。

    算法（Lloyd迭代）:
        1. 随机初始化生成元 g[0..g_num-1] 在 [0,1] 内。
        2. 每轮迭代:
           a. 在[0,1]中随机采样s_num个点。
           b. 对每个样本，找到最近的生成元（考虑周期镜像: s, s+1, s-1）。
           c. 更新生成元为其Voronoi单元的质心。
        3. 排序并返回优化后的生成元。

    参数:
        g_num: 生成元数量
        it_num: 迭代次数
        s_num: 每轮采样数
        density_func: 目标密度函数 rho(x)，默认均匀密度
    返回:
        generators: (g_num,) 优化后的节点位置
    """
    if g_num <= 0:
        raise ValueError("生成元数量必须为正")
    if density_func is None:
        density_func = lambda x: 1.0

    # 初始化生成元
    g = np.sort(np.random.rand(g_num))

    for it in range(it_num):
        s = np.random.rand(s_num)
        sa = 0.0 - s
        sb = 2.0 - s

        g_new = np.zeros(g_num)
        w_new = np.zeros(g_num)

        for i in range(s_num):
            # 寻找最近生成元（考虑周期镜像）
            d0 = np.abs(s[i] - g)
            d1 = np.abs(sa[i] - g)
            d2 = np.abs(sb[i] - g)
            d_min = np.minimum(np.minimum(d0, d1), d2)
            idx = int(np.argmin(d_min))

            # 确定实际使用的样本坐标
            if d1[idx] < d0[idx] and d1[idx] < d2[idx]:
                si = sa[i]
            elif d2[idx] < d0[idx] and d2[idx] < d1[idx]:
                si = sb[i]
            else:
                si = s[i]

            # 加权质心
            rho = max(density_func(si), 1e-10)
            g_new[idx] += rho * si
            w_new[idx] += rho

        # 更新生成元
        mask = w_new > 1e-14
        g[mask] = g_new[mask] / w_new[mask]
        # 未分配的生成元保持原位
        g = np.sort(np.mod(g, 1.0))

    return g


class MolecularIntegralGrid:
    """
    分子积分网格管理器，用于VQE中双电子积分的数值计算。
    """
    def __init__(self, n_orbitals: int = 4, grid_level: int = 3):
        self.n_orbitals = n_orbitals
        self.grid_level = grid_level
        # 高斯型原子轨道参数（STO-3G简化）
        self.zeta = np.array([1.0, 1.0, 0.8, 0.8])  # 轨道指数
        self.centers = np.array([[0.0, 0.0, 0.0],
                                  [1.4, 0.0, 0.0],
                                  [0.7, 0.7, 0.0],
                                  [0.7, -0.7, 0.0]])

    def build_3d_grid(self, n_per_dim: int = 8) -> np.ndarray:
        """
        使用超立方体网格生成3D积分网格。
        覆盖分子所在区域: [-2, 3] x [-2, 2] x [-2, 2]。
        """
        m = 3
        ns = np.array([n_per_dim, n_per_dim, n_per_dim])
        a = np.array([-2.0, -2.0, -2.0])
        b = np.array([3.0, 2.0, 2.0])
        c = np.array([5, 5, 5])  # 中点分布，避免边界奇异性
        grid = hypercube_grid(m, ns, a, b, c)
        return grid.T  # 返回 (n_points, 3)

    def slater_orbital(self, r: np.ndarray, center: np.ndarray,
                       zeta: float) -> float:
        """
        计算归一化的Slater型轨道值:
            phi(r) = sqrt(zeta^3/pi) * exp(-zeta * |r - R|)
        """
        dist = np.linalg.norm(r - center)
        norm = np.sqrt(zeta ** 3 / np.pi)
        return norm * np.exp(-zeta * dist)

    def one_electron_integral(self, i: int, j: int,
                               grid_points: np.ndarray,
                               weights: np.ndarray) -> float:
        """
        计算一体积分 h_{ij} = integral phi_i(r) * (-0.5 * nabla^2 - sum_A Z_A/|r-R_A|) * phi_j(r) dr
        使用数值积分近似。
        """
        val = 0.0
        for gp, w in zip(grid_points, weights):
            phi_i = self.slater_orbital(gp, self.centers[i], self.zeta[i])
            phi_j = self.slater_orbital(gp, self.centers[j], self.zeta[j])
            # 简化核吸引项: V = -sum_A Z_A / |r - R_A|
            V = 0.0
            for center in self.centers[:2]:
                dist = np.linalg.norm(gp - center) + 1e-10
                V -= 1.0 / dist
            # 动能项近似（通过轨道曲率）
            T = 0.5 * self.zeta[i] * self.zeta[j] * phi_i * phi_j
            val += w * phi_i * (T + V) * phi_j
        return val

    def two_electron_integral(self, i: int, j: int, k: int, l: int,
                               grid_points: np.ndarray,
                               weights: np.ndarray) -> float:
        """
        计算双电子积分 (ij|kl) 的简化数值近似。
        使用密度拟合: (ij|kl) \approx sum_p w_p * phi_i(r_p) phi_j(r_p) phi_k(r_p) phi_l(r_p)
        """
        val = 0.0
        for gp, w in zip(grid_points, weights):
            phi_i = self.slater_orbital(gp, self.centers[i], self.zeta[i])
            phi_j = self.slater_orbital(gp, self.centers[j], self.zeta[j])
            phi_k = self.slater_orbital(gp, self.centers[k], self.zeta[k])
            phi_l = self.slater_orbital(gp, self.centers[l], self.zeta[l])
            val += w * phi_i * phi_j * phi_k * phi_l
        return val

    def optimize_radial_grid(self, n_points: int = 16) -> np.ndarray:
        """
        使用CVT优化径向积分网格，适配电子密度分布。
        密度函数: rho(r) = exp(-2r)（氢原子基态密度）
        """
        rho_func = lambda r: np.exp(-2.0 * r) if isinstance(r, (int, float)) else np.exp(-2.0 * np.array(r))
        generators = cvtm_1d_optimize(n_points, it_num=30, s_num=2000,
                                      density_func=rho_func)
        # 映射到实际径向坐标 [0, R_max]
        R_max = 10.0
        return generators * R_max
