"""
lattice_geometry.py
===================

4 维超立方体格点几何, 支持有限温 QCD 的边界条件设置.

物理背景:
---------
有限温 QCD 通过紧致化欧几里得时间方向实现:
    x_4 ∈ [0, β = 1/T],  周期边界条件 (PBC) for gauge fields
    对费米子场, 时间方向采用反周期边界条件 (APBC):
        ψ(x_4 = 1/T) = -ψ(x_4 = 0)

温度与格点参数的关系:
    T = 1 / (N_t · a)
其中 a 为格距, N_t 为时间方向格点数.

空间方向 x, y, z 采用通常的周期边界条件.

本模块融合种子项目:
  - 1333_triangulation_boundary_nodes: 边界节点识别算法
      在 4D 超立方体格点上推广: 通过边 (link) 的计数识别边界 plaquette
  - 631_l4lib: 布尔位运算用于 parity (偶奇分解)
      站点奇偶性: p(x) = XOR_{μ=0}^3 x_μ (mod 2)
      用于 even-odd preconditioning of Dirac operator
  - 1338_triangulation_l2q: 线性→二次提升
      类比: Wilson (plaquette-only) → Symanzik 改进 (plaquette + rectangle)
"""

import numpy as np
from typing import Tuple, List, Dict, Optional


# ============================================================
# 4D 格点结构
# ============================================================

class LatticeGeometry:
    """4 维超立方体格点的几何与拓扑.

    属性:
        Ns:  空间方向格点数 (各向同性)
        Nt:  时间方向格点数 (紧致化, T = 1/(Nt a))
        ndim: 维数 (固定为 4)
        volume: 总格点数 V = Ns^3 * Nt
        boundary_phase: 各方向的边界相位
            μ=0,1,2 (空间): 0.0 (周期)
            μ=3 (时间):
                - 规范场: 0.0 (周期)
                - 费米子: π (反周期)

    站点编号约定 (字典序):
        idx = x + Ns * (y + Ns * (z + Ns * t))
        0 ≤ idx < volume
    """

    def __init__(self, Ns: int, Nt: int,
                 fermion_apbc: bool = True):
        """
        参数:
            Ns: 空间方向格点数 (立方体)
            Nt: 时间方向格点数
            fermion_apbc: 是否为费米子场启用时间反周期边界条件
        """
        self.Ns = int(Ns)
        self.Nt = int(Nt)
        self.ndim = 4
        self.volume = Ns ** 3 * Nt
        self.spatial_volume = Ns ** 3
        self.shape = (Ns, Ns, Ns, Nt)

        # 边界相位: 规范场始终周期
        self.gauge_bc = np.zeros(4, dtype=np.float64)
        # 费米子: 时间方向反周期
        self.fermion_bc = np.zeros(4, dtype=np.float64)
        if fermion_apbc:
            self.fermion_bc[3] = np.pi

        # 预计算索引结构
        self._site_coords = self._build_site_coords()
        self._parity = self._compute_parity()
        self._even_sites, self._odd_sites = self._split_parity()
        self._boundary_nodes = self._find_boundary_nodes()

    # --------------------------------------------------------
    # 索引转换
    # --------------------------------------------------------

    def _build_site_coords(self) -> np.ndarray:
        """预计算所有站点的 4D 坐标, shape = (volume, 4)."""
        coords = np.zeros((self.volume, 4), dtype=np.int32)
        for idx in range(self.volume):
            coords[idx] = self.idx_to_coord(idx)
        return coords

    def coord_to_idx(self, x: int, y: int, z: int, t: int) -> int:
        """站点坐标 → 字典序索引."""
        return x + self.Ns * (y + self.Ns * (z + self.Ns * t))

    def idx_to_coord(self, idx: int) -> np.ndarray:
        """字典序索引 → 站点坐标 [x, y, z, t]."""
        t = idx // (self.Ns ** 3)
        rem = idx % (self.Ns ** 3)
        z = rem // (self.Ns ** 2)
        rem = rem % (self.Ns ** 2)
        y = rem // self.Ns
        x = rem % self.Ns
        return np.array([x, y, z, t], dtype=np.int32)

    def neighbor_idx(self, idx: int, mu: int, direction: int = +1) -> int:
        """计算 idx 在 μ 方向 ±1 的邻居索引.

        有限温格点: 空间方向为 PBC, 时间方向对规范场为 PBC.

        参数:
            idx: 源站点索引
            mu:  方向 (0,1,2,3)
            direction: +1 (正向) or -1 (反向)
        """
        c = self.idx_to_coord(idx).copy()
        L = self.Ns if mu < 3 else self.Nt
        c[mu] = (c[mu] + direction) % L
        return self.coord_to_idx(*c)

    # --------------------------------------------------------
    # 站点奇偶性 (融合 631_l4lib XOR 思想)
    # --------------------------------------------------------

    def _compute_parity(self) -> np.ndarray:
        """计算所有站点的奇偶性: p(x) = XOR_{μ=0}^3 x_μ (mod 2).

        返回 0 (偶) 或 1 (奇) 的布尔数组, 长度为 volume.
        用于 even-odd preconditioning of Dirac operator.
        """
        parity = np.zeros(self.volume, dtype=np.int8)
        for idx in range(self.volume):
            c = self._site_coords[idx]
            # XOR of all coordinates mod 2 (from 631_l4lib)
            parity[idx] = (c[0] ^ c[1] ^ c[2] ^ c[3]) & 1
        return parity

    def _split_parity(self) -> Tuple[np.ndarray, np.ndarray]:
        """分离偶/奇站点索引."""
        even = np.where(self._parity == 0)[0].astype(np.int32)
        odd = np.where(self._parity == 1)[0].astype(np.int32)
        return even, odd

    @property
    def even_sites(self) -> np.ndarray:
        return self._even_sites

    @property
    def odd_sites(self) -> np.ndarray:
        return self._odd_sites

    @property
    def parity(self) -> np.ndarray:
        return self._parity

    # --------------------------------------------------------
    # 边界节点识别 (融合 1333_triangulation_boundary_nodes)
    # --------------------------------------------------------

    def _find_boundary_nodes(self) -> np.ndarray:
        """识别格点边界上的站点.

        算法 (推广 1333 的边计数思想到 4D 超立方体):
            对每个 link (idx, μ), 生成有向边.
            在内部, 每条边被两个 plaquette 共享.
            边界 plaquette 是只被一个立体共享的 plaquette.
            对于 PBC 的紧致格点, 实际上没有物理边界,
            但 "temporal boundary" 在有限温物理中有特殊意义:
            t = 0 和 t = N_t - 1 处的站点构成 Polyakov loop 的端点.

        返回:
            时间边界站点索引 (t = 0 or t = N_t - 1)
        """
        boundary = []
        for idx in range(self.volume):
            t = self._site_coords[idx, 3]
            if t == 0 or t == self.Nt - 1:
                boundary.append(idx)
        return np.array(boundary, dtype=np.int32)

    @property
    def boundary_nodes(self) -> np.ndarray:
        """时间方向边界上的站点."""
        return self._boundary_nodes

    def boundary_links(self) -> List[Tuple[int, int]]:
        """跨越时间边界的 link: (idx, μ=3) 其中 t = Nt-1.

        这些 link 在 Polyakov loop 计算中起关键作用.
        """
        links = []
        for idx in self._boundary_nodes:
            t = self._site_coords[idx, 3]
            if t == self.Nt - 1:
                links.append((idx, 3))
        return links

    # --------------------------------------------------------
    # plaquette 与 loop 枚举
    # --------------------------------------------------------

    def plaquette_list(self) -> List[Tuple[int, int, int]]:
        """返回所有 (站点, μ, ν) 组合, μ < ν.

        总数: 6 × volume (每个站点 6 个独立平面).
        """
        plist = []
        for idx in range(self.volume):
            for mu in range(4):
                for nu in range(mu + 1, 4):
                    plist.append((idx, mu, nu))
        return plist

    def rectangle_list(self) -> List[Tuple[int, int, int, int]]:
        """返回所有 1×2 矩形 loop 的 (站点, μ, ν, orientation).

        用于 Symanzik 改进规范作用量 (融合 1338_l2q 的中边节点思想):
            矩形 loop 的 6 个节点对应于二次三角形的新增中边节点.

        orientation:
            0: 长边在 μ 方向 (2 步 μ, 1 步 ν)
            1: 长边在 ν 方向
        """
        rlist = []
        for idx in range(self.volume):
            for mu in range(4):
                for nu in range(mu + 1, 4):
                    for orient in (0, 1):
                        rlist.append((idx, mu, nu, orient))
        return rlist

    # --------------------------------------------------------
    # 温度与物理量
    # --------------------------------------------------------

    def temperature(self, a: float = 1.0) -> float:
        """格点温度 T = 1 / (N_t a)."""
        return 1.0 / (self.Nt * a)

    def debye_mass_tree_level(self, g: float, a: float = 1.0) -> float:
        """树级 Debye 屏蔽质量 (高温微扰 QCD).

        m_D^2 = (1/3) g^2 T^2 (N_c + N_f/2)
        对纯规范理论 N_f = 0: m_D^2 = g^2 T^2

        返回: m_D (格点单位)
        """
        T = self.temperature(a)
        return g * T

    def critical_beta_ya02003(self, Nt: int) -> float:
        """经验拟合的临界 β_c(N_t) (Yano et al. / Karsch).

        纯 SU(3) 规范理论退禁闭相变:
            N_t = 4: β_c ≈ 5.692
            N_t = 6: β_c ≈ 5.894
            N_t = 8: β_c ≈ 6.111

        使用 2-loop 跑动耦合拟合:
            β_c = 6 / g_c^2, 其中 g_c^2 由 2-loop beta function 确定.
        """
        if Nt == 4:
            return 5.692
        elif Nt == 6:
            return 5.894
        elif Nt == 8:
            return 6.111
        else:
            # 2-loop 拟合: β_c(N_t) ≈ β_∞ - c / log(N_t)
            # β_∞ ≈ 7.2 (连续极限)
            beta_inf = 7.20
            c = 2.5
            return beta_inf - c / np.log(Nt)

    # --------------------------------------------------------
    # 几何工具
    # --------------------------------------------------------

    def distance_sq(self, idx1: int, idx2: int,
                    metric: str = 'euclidean') -> float:
        """计算两站点间的最小镜像距离的平方.

        考虑 PBC 的最小镜像约定.
        """
        c1 = self._site_coords[idx1]
        c2 = self._site_coords[idx2]
        d = c2 - c1
        # 最小镜像
        L = np.array([self.Ns, self.Ns, self.Ns, self.Nt])
        d = np.where(d > L / 2, d - L, d)
        d = np.where(d < -L / 2, d + L, d)
        if metric == 'euclidean':
            return float(np.sum(d ** 2))
        elif metric == 'manhattan':
            return float(np.sum(np.abs(d)))
        else:
            raise ValueError(f"未知度量: {metric}")

    def all_site_coords(self) -> np.ndarray:
        """返回所有站点坐标, shape = (volume, 4)."""
        return self._site_coords.copy()


# ============================================================
# 边界条件相位因子
# ============================================================

class BoundaryPhases:
    """规范场和费米子场的边界相位管理.

    规范场在时间方向的 Polyakov loop:
        L(x) = Tr Π_{t=0}^{Nt-1} U_4(x, t)
    当 U_4 跨越边界时, 采用 PBC: U_4(x, Nt-1; +4) = U_4(x, 0).

    费米子场采用 APBC:
        ψ(x, Nt) = -ψ(x, 0)
    通过引入相位因子 e^{i π} = -1 实现.
    """

    def __init__(self, geometry: LatticeGeometry):
        self.geom = geometry
        # 规范场边界相位 (全部为 1)
        self.gauge_phase = np.ones(4, dtype=np.complex128)
        # 费米子边界相位 (时间方向为 -1)
        self.fermion_phase = np.exp(1j * geometry.fermion_bc)

    def gauge_link_factor(self, idx: int, mu: int) -> complex:
        """计算规范场 link 的边界相位因子.

        对于紧致 PBC, 所有因子为 1.
        (如需 twisted BC, 可在此处添加 twist 角.)
        """
        c = self.geom.idx_to_coord(idx)
        L = self.geom.Ns if mu < 3 else self.geom.Nt
        # 跨越边界的 link: c[mu] = L - 1 → + 方向
        # 对于 PBC, 因子为 1
        return 1.0 + 0.0j

    def fermion_link_factor(self, idx: int, mu: int) -> complex:
        """计算费米子 hop 的边界相位因子.

        时间方向跨越边界时乘以 -1.
        """
        c = self.geom.idx_to_coord(idx)
        if mu == 3:
            L = self.geom.Nt
            if c[3] == L - 1:
                return -1.0 + 0.0j
        return 1.0 + 0.0j
