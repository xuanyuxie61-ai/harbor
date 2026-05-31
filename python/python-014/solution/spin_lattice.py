"""
spin_lattice.py
===============
阻挫磁性晶格构造与邻接拓扑模块。
融合来源：lights_out（邻接矩阵构造）、prime_plot（素数周期性阻挫）、components（连通分量标记）。

本模块实现：
- 三维烧绿石(pyrochlore)型晶格及其降维投影
- 基于素数周期调制的几何阻挫晶格
- 邻接关系矩阵（模2或实数域）
- 连通分量分析以识别自旋团簇
"""

import numpy as np
from typing import Tuple, List, Optional
from utils import is_prime, primes_up_to, EPS_MACHINE


class PyrochloreLattice:
    """
    烧绿石晶格上的经典自旋系统。

    烧绿石晶格由_corner-sharing tetrahedra_构成，具有强烈的几何阻挫。
    原胞包含 4 个子格点，晶格常数设为 a=1。

    位置由面心立方(fcc)格点加上基矢决定：
        R1 = (0, 0, 0)
        R2 = (0, 1/2, 1/2)
        R3 = (1/2, 0, 1/2)
        R4 = (1/2, 1/2, 0)
    在常规立方晶胞中，归一化到 [0,1)^3 内。
    """

    def __init__(self, L: int = 4, J1: float = 1.0, J2: float = 0.0, disorder_std: float = 0.15):
        """
        参数
        ----
        L : int
            每个方向的晶胞数，总格点数 N = 4 * L^3。
        J1 : float
            最近邻交换耦合强度（反铁磁为正）。
        J2 : float
            次近邻耦合强度。
        disorder_std : float
            键无序标准差（自旋玻璃无序）。
        """
        if L < 1:
            raise ValueError("L must be >= 1")
        self.L = L
        self.N = 4 * L * L * L
        self.J1 = J1
        self.J2 = J2
        self.disorder_std = disorder_std
        self._build_sites()
        self._build_bonds()

    def _build_sites(self):
        """构造所有格点的笛卡尔坐标。"""
        basis = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ], dtype=float)
        sites = []
        for ix in range(self.L):
            for iy in range(self.L):
                for iz in range(self.L):
                    origin = np.array([ix, iy, iz], dtype=float) / self.L
                    for b in basis:
                        sites.append(origin + b / self.L)
        self.sites = np.array(sites, dtype=float)
        # 保证坐标在 [0,1)
        self.sites %= 1.0

    def _periodic_distance_sq(self, i: int, j: int) -> float:
        """考虑周期性边界条件的最小映像距离平方。"""
        diff = np.abs(self.sites[i] - self.sites[j])
        diff = np.minimum(diff, 1.0 - diff)
        return np.sum(diff ** 2)

    def _build_bonds(self):
        """
        构造键（bond）列表与耦合矩阵 J。
        最近邻判定：在烧绿石晶格中，最近邻距离 d_nn = sqrt(2)/ (2L)。
        """
        self.bonds = []
        self.J = np.zeros((self.N, self.N), dtype=float)
        d_nn_sq = (np.sqrt(2.0) / (2.0 * self.L)) ** 2
        tol = 1e-6
        for i in range(self.N):
            for j in range(i + 1, self.N):
                d2 = self._periodic_distance_sq(i, j)
                if np.abs(d2 - d_nn_sq) < tol:
                    disorder = np.random.normal(0.0, self.disorder_std)
                    jval = self.J1 + disorder
                    self.bonds.append((i, j, jval))
                    self.J[i, j] = jval
                    self.J[j, i] = jval
                elif self.J2 != 0.0:
                    d_nnn_sq = 2.0 * d_nn_sq  # 次近邻近似
                    if np.abs(d2 - d_nnn_sq) < tol:
                        disorder = np.random.normal(0.0, self.disorder_std)
                        jval = self.J2 + disorder
                        self.bonds.append((i, j, jval))
                        self.J[i, j] = jval
                        self.J[j, i] = jval

    def adjacency_matrix(self) -> np.ndarray:
        """返回无权邻接矩阵（用于连通分量分析）。"""
        A = (np.abs(self.J) > EPS_MACHINE).astype(int)
        return A

    def degree_matrix(self) -> np.ndarray:
        """返回度矩阵 D。"""
        degrees = np.sum(np.abs(self.J) > EPS_MACHINE, axis=1)
        return np.diag(degrees)

    def graph_laplacian(self) -> np.ndarray:
        """返回图拉普拉斯 L = D - A。融合来源：laplacian_matrix。"""
        return self.degree_matrix() - self.adjacency_matrix()


class PrimeFrustratedLattice:
    """
    基于素数周期调制的二维阻挫晶格。
    融合来源：prime_plot（素数分布与周期性）。

    构造一个 LxL 的方格，其交换耦合强度随位置按素数序列调制，
    产生非均匀阻挫，模拟化学无序导致的自旋玻璃行为。
    """

    def __init__(self, L: int = 20, J0: float = 1.0, alpha: float = 0.3, seed: int = 42):
        np.random.seed(seed)
        if L < 2:
            raise ValueError("L must be >= 2")
        self.L = L
        self.N = L * L
        self.J0 = J0
        self.alpha = alpha
        self.primes = primes_up_to(max(100, 2 * L))
        self._build_couplings()

    def _build_couplings(self):
        """
        水平与垂直耦合均按素数索引调制：
            J_{i,j;x} = J0 * (1 + alpha * sin(pi * p_i / p_max))
            其中 p_i 为第 i 个素数。
        同时引入随机符号翻转以模拟阻挫。
        """
        self.J_h = np.zeros((self.L, self.L - 1), dtype=float)
        self.J_v = np.zeros((self.L - 1, self.L), dtype=float)
        pmax = self.primes[-1] if self.primes else 1.0
        for i in range(self.L):
            for j in range(self.L - 1):
                idx = (i + j) % max(len(self.primes), 1)
                p = self.primes[idx] if self.primes else 1
                phase = np.sin(np.pi * p / pmax)
                sign = 1.0 if np.random.rand() > 0.3 else -1.0
                self.J_h[i, j] = sign * self.J0 * (1.0 + self.alpha * phase)
        for i in range(self.L - 1):
            for j in range(self.L):
                idx = (i * 3 + j * 7) % max(len(self.primes), 1)
                p = self.primes[idx] if self.primes else 1
                phase = np.cos(np.pi * p / pmax)
                sign = 1.0 if np.random.rand() > 0.3 else -1.0
                self.J_v[i, j] = sign * self.J0 * (1.0 + self.alpha * phase)

    def to_full_matrix(self) -> np.ndarray:
        """将键耦合展开为 N x N 全矩阵。"""
        N = self.N
        J = np.zeros((N, N), dtype=float)
        for i in range(self.L):
            for j in range(self.L - 1):
                idx1 = i * self.L + j
                idx2 = i * self.L + (j + 1)
                J[idx1, idx2] = self.J_h[i, j]
                J[idx2, idx1] = self.J_h[i, j]
        for i in range(self.L - 1):
            for j in range(self.L):
                idx1 = i * self.L + j
                idx2 = (i + 1) * self.L + j
                J[idx1, idx2] = self.J_v[i, j]
                J[idx2, idx1] = self.J_v[i, j]
        return J


def connected_components_2d_spin_map(spin_map: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    对二维自旋投影图进行连通分量标记。
    融合来源：components（2D 连通分量标记算法）。

    将 |spin_z| > threshold 的位置视为 "活跃" 像素，
    使用四邻域连通性标记磁畴团簇。

    参数
    ----
    spin_map : np.ndarray, shape (M, N)
        二维自旋投影（如 z 分量）。
    threshold : float
        活跃判定阈值。

    返回
    ----
    labels : np.ndarray, shape (M, N)
        连通分量标签，0 表示背景。
    """
    if spin_map.ndim != 2:
        raise ValueError("spin_map must be 2D")
    m, n = spin_map.shape
    A = (np.abs(spin_map) > threshold).astype(int)
    C = np.zeros((m, n), dtype=int)
    component_index = 0

    for i2 in range(m):
        for j2 in range(n):
            if A[i2, j2] != 0 and C[i2, j2] == 0:
                plist = [(i2, j2)]
                component_index += 1
                while plist:
                    i, j = plist.pop()
                    if C[i, j] != 0:
                        continue
                    C[i, j] = component_index
                    # 四邻域
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < m and 0 <= nj < n:
                            if A[ni, nj] != 0 and C[ni, nj] == 0:
                                plist.append((ni, nj))
    return C
