"""
spin_lattice_geometry.py
========================

三维立方晶格的几何结构、邻居索引与边界拓扑。

物理背景
--------
Edwards-Anderson 自旋玻璃模型定义在 d 维超立方晶格上。
在 d=3 时, 每个格点 i=(i_x, i_y, i_z) 与 6 个近邻
(±x, ±y, ±z) 通过交换相互作用 J_{ij} 耦合。

核心公式
--------
哈密顿量:
    H = -sum_{<ij>} J_{ij} S_i S_j
其中 <ij> 表示仅对最近邻对求和一次。

格点线性索引 (row-major):
    n(i_x, i_y, i_z) = i_x * L^2 + i_y * L + i_z

最近邻偏移向量集合 (d=3, 周期性边界条件):
    delta = {+x_hat, -x_hat, +y_hat, -y_hat, +z_hat, -z_hat}

配位数 z = 2d = 6 (d=3)

键的总数:
    N_bond = d * N = 3 * L^3

本模块核心算法来源于 seed project:
- 1345_triangulation_plot: 三角网格的节点-单元拓扑编码
  (映射为晶格节点-近邻键拓扑)
- 545_house: 折线/顶点序列编码
  (映射为近邻键的有序列表编码)
"""

import numpy as np
from typing import Tuple, List, Dict, Optional


# =====================================================================
#  晶格几何基类
# =====================================================================

class CubicLattice3D:
    """
    三维简单立方晶格 (Simple Cubic Lattice, SC)。

    参数
    ----
    L : int
        每个方向的格点数, L >= 2
    periodic : bool, default True
        是否采用周期性边界条件 (PBC)

    属性
    ----
    N : int
        总格点数 N = L^3
    z : int
        配位数 z = 2d = 6
    n_bond : int
        键总数 N_bond = d*L^3 = 3*L^3
    """

    # 最近邻偏移向量 (d=3)
    NEIGHBOR_OFFSETS = np.array([
        (+1,  0,  0),
        (-1,  0,  0),
        ( 0, +1,  0),
        ( 0, -1,  0),
        ( 0,  0, +1),
        ( 0,  0, -1),
    ], dtype=np.int64)

    def __init__(self, L: int, periodic: bool = True):
        if L < 2:
            raise ValueError(f"L 必须 >= 2, 当前 L={L}")
        self.L = int(L)
        self.periodic = periodic
        self.d = 3
        self.N = self.L ** 3
        self.z = 2 * self.d  # 配位数 = 6
        self.n_bond = self.d * self.N
        self._build_neighbor_table()

    # -----------------------------------------------------------------
    #  坐标映射: 三维 <-> 线性
    # -----------------------------------------------------------------

    def coord_to_index(self, ix: int, iy: int, iz: int) -> int:
        """(i_x, i_y, i_z) -> 线性索引 n"""
        return ix * self.L * self.L + iy * self.L + iz

    def index_to_coord(self, n: int) -> Tuple[int, int, int]:
        """线性索引 n -> (i_x, i_y, i_z)"""
        if n < 0 or n >= self.N:
            raise IndexError(f"索引 {n} 越界 [0, {self.N})")
        ix = n // (self.L * self.L)
        rem = n % (self.L * self.L)
        iy = rem // self.L
        iz = rem % self.L
        return (ix, iy, iz)

    # -----------------------------------------------------------------
    #  边界条件: 坐标折叠 (模 L)
    # -----------------------------------------------------------------

    def wrap_coordinate(self, ix: int, iy: int, iz: int) -> Tuple[int, int, int]:
        """
        将坐标折叠到 [0, L) 范围内。

        若 periodic=True, 采用模运算 (torus 拓扑);
        若 periodic=False, 超出边界的坐标返回 None。
        """
        if self.periodic:
            return (ix % self.L, iy % self.L, iz % self.L)
        else:
            if 0 <= ix < self.L and 0 <= iy < self.L and 0 <= iz < self.L:
                return (ix, iy, iz)
            else:
                return None

    # -----------------------------------------------------------------
    #  邻居表
    # -----------------------------------------------------------------

    def get_neighbors(self, ix: int, iy: int, iz: int) -> List[Tuple[int, int, int]]:
        """
        获取格点 (ix, iy, iz) 的所有最近邻坐标 (经边界处理)。

        返回
        ----
        neighbors : list of (ix, iy, iz)
            长度为 z = 6
        """
        neighbors = []
        for offset in self.NEIGHBOR_OFFSETS:
            nx, ny, nz = ix + offset[0], iy + offset[1], iz + offset[2]
            wrapped = self.wrap_coordinate(nx, ny, nz)
            if wrapped is not None:
                neighbors.append(wrapped)
        return neighbors

    def get_neighbors_linear(self, n: int) -> List[int]:
        """
        获取线性索引为 n 的格点的所有最近邻线性索引。
        """
        ix, iy, iz = self.index_to_coord(n)
        nbr_coords = self.get_neighbors(ix, iy, iz)
        return [self.coord_to_index(*c) for c in nbr_coords]

    def _build_neighbor_table(self):
        """
        构建全局邻居表 neighbor_table[n, :] = [n1, n2, ..., n_{z-1}]

        形状: (N, z) 的 int64 数组
        """
        self.neighbor_table = np.zeros((self.N, self.z), dtype=np.int64)
        for n in range(self.N):
            nbrs = self.get_neighbors_linear(n)
            if len(nbrs) < self.z:
                # 开边界条件下, 用 -1 填充缺失的邻居
                nbrs += [-1] * (self.z - len(nbrs))
            self.neighbor_table[n, :] = nbrs

    # -----------------------------------------------------------------
    #  键列表: 所有最近邻对 (无重复)
    # -----------------------------------------------------------------

    def get_bond_list(self) -> List[Tuple[int, int]]:
        """
        返回所有键 (i, j) 的列表, 其中 i < j, 避免重复计数。

        键总数 = d * N = 3 * L^3 (PBC)
        """
        bonds = []
        seen = set()
        for n in range(self.N):
            for m in self.neighbor_table[n]:
                if m < 0:
                    continue
                pair = (min(n, m), max(n, m))
                if pair not in seen:
                    seen.add(pair)
                    bonds.append(pair)
        return bonds

    # -----------------------------------------------------------------
    #  几何验证
    # -----------------------------------------------------------------

    def verify_consistency(self) -> bool:
        """
        验证晶格拓扑的一致性:
        1) 邻居表对称性: 若 j 是 n 的邻居, 则 n 也是 j 的邻居
        2) 配位数检查
        3) 键数检查
        """
        # 对称性检查
        for n in range(self.N):
            for m in self.neighbor_table[n]:
                if m < 0:
                    continue
                if n not in self.neighbor_table[m]:
                    return False
        # 键数检查
        bonds = self.get_bond_list()
        expected = self.d * self.N if self.periodic else len(bonds)
        return len(bonds) == expected

    # -----------------------------------------------------------------
    #  对偶格 (reciprocal lattice) —— 用于 von Neumann 稳定性分析
    # -----------------------------------------------------------------

    def reciprocal_wavevectors(self) -> np.ndarray:
        """
        计算第一布里渊区的波矢 k = (kx, ky, kz)。

        在周期性边界条件下:
            k_mu = 2*pi*n_mu / L,  n_mu in {-L/2, ..., L/2-1}

        返回
        ----
        kvecs : ndarray, shape (N, 3)
            N 个波矢
        """
        n_modes = np.arange(-self.L // 2, self.L // 2)
        kvecs = []
        for kx in n_modes:
            for ky in n_modes:
                for kz in n_modes:
                    kvecs.append((kx, ky, kz))
        kvecs = 2.0 * np.pi / self.L * np.array(kvecs, dtype=np.float64)
        return kvecs

    def structure_factor_lattice(self) -> np.ndarray:
        """
        计算晶格结构因子 (Lattice Structure Factor):
            gamma(k) = (1/d) * sum_{mu=1}^{d} cos(k_mu * a)

        在 d=3, a=1 时:
            gamma(k) = (1/3) * [cos(kx) + cos(ky) + cos(kz)]

        用于色散关系 omega(k) 的计算。
        """
        kvecs = self.reciprocal_wavevectors()
        gamma = (np.cos(kvecs[:, 0]) + np.cos(kvecs[:, 1]) +
                 np.cos(kvecs[:, 2])) / self.d
        return gamma


# =====================================================================
#  工具函数
# =====================================================================

def lattice_site_energy(spins: np.ndarray, couplings: Dict[Tuple[int, int], float],
                        site: int, lattice: CubicLattice3D) -> float:
    """
    计算单个格点 site 对总能量的贡献:
        E_site = -S_{site} * sum_{j in nbr(site)} J_{site,j} * S_j

    注意这是键能的一半 (每个键被两个格点共享)。
    """
    s_site = spins.flat[site]
    e_site = 0.0
    for nbr in lattice.neighbor_table[site]:
        if nbr < 0:
            continue
        key = (min(site, nbr), max(site, nbr))
        J = couplings.get(key, 0.0)
        e_site += J * spins.flat[nbr]
    return -s_site * e_site


def total_energy(spins: np.ndarray, couplings: Dict[Tuple[int, int], float],
                 lattice: CubicLattice3D) -> float:
    """
    计算总哈密顿量 H = -sum_{<ij>} J_{ij} S_i S_j
    仅对键列表中的键求和一次。
    """
    H = 0.0
    bonds = lattice.get_bond_list()
    for (i, j) in bonds:
        key = (i, j)
        J = couplings.get(key, 0.0)
        H -= J * spins.flat[i] * spins.flat[j]
    return H


def magnetization(spins: np.ndarray) -> float:
    """
    总磁化强度 M = sum_i S_i
    """
    return float(np.sum(spins))


def magnetization_density(spins: np.ndarray) -> float:
    """
    磁化强度密度 m = M / N
    """
    return magnetization(spins) / spins.size
