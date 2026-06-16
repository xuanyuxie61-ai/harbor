"""
nas_search.py  —  邻居搜索与粒子-网格质量赋值加速
================================================

科学来源种子:
  - 786_nas / nas.m
    直接使用其 N-body Acceleration Search (NAS) 思路:
    在周期域中快速定位每个粒子最近的网格节点,用于 CIC/TSC
    质量赋值与力插值。
    对应算法: 基于哈希的 O(N) 邻居查询。

物理背景:
  Particle-Mesh (PM) 方法核心操作:
    1. 质量赋值 (deposit): ρ_grid[i,j,k] = Σ_p m_p W(x_p - x_grid)
    2. Poisson 求解: ∇²Φ = 4πG a² ρ
    3. 力插值 (interpolate): F_p = -∇Φ(x_p)
  每步复杂度 O(N) + O(N_grid log N_grid)。
  邻居搜索使用周期哈希:
      hash(i, j, k) = ((i mod N) · P1 + (j mod N) · P2 + (k mod N) · P3) mod H
  其中 P1, P2, P3 为大素数, H 为哈希表大小。
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Dict


# ---------------------------------------------------------------------------- #
#                周期哈希邻居搜索 (源自 nas.m)
# ---------------------------------------------------------------------------- #
class PeriodicHashGrid:
    """
    周期盒子中的哈希网格邻居搜索结构。
    """
    def __init__(self, box_length: float, cell_size: float):
        self.L = box_length
        self.h = cell_size
        self.n_cells = max(1, int(box_length / cell_size))
        self.h = box_length / self.n_cells  # 精确调整
        self._cells: Dict[int, list] = {}
        self._primes = (73856003, 19349663, 83492791)

    def _cell_index(self, pos: NDArray) -> Tuple[int, int, int]:
        q = np.floor(pos / self.h).astype(int) % self.n_cells
        return (int(q[0]), int(q[1]), int(q[2]))

    def _hash(self, ijk: Tuple[int, int, int]) -> int:
        i, j, k = ijk
        return ((i * self._primes[0]) ^ (j * self._primes[1])
                ^ (k * self._primes[2])) % (2 ** 31 - 1)

    def build(self, positions: NDArray) -> None:
        """将粒子插入哈希网格。"""
        self._cells.clear()
        for p_idx, pos in enumerate(positions):
            ijk = self._cell_index(pos)
            h = self._hash(ijk)
            if h not in self._cells:
                self._cells[h] = []
            self._cells[h].append((p_idx, ijk))

    def query_radius(self, center: NDArray, radius: float) -> list:
        """返回距离 center ≤ radius 的所有粒子索引。"""
        r_cells = max(1, int(np.ceil(radius / self.h)))
        c0 = self._cell_index(center)
        found = []
        for di in range(-r_cells, r_cells + 1):
            for dj in range(-r_cells, r_cells + 1):
                for dk in range(-r_cells, r_cells + 1):
                    ijk = ((c0[0] + di) % self.n_cells,
                           (c0[1] + dj) % self.n_cells,
                           (c0[2] + dk) % self.n_cells)
                    h = self._hash(ijk)
                    if h in self._cells:
                        for p_idx, p_ijk in self._cells[h]:
                            found.append(p_idx)
        return found


# ---------------------------------------------------------------------------- #
#                    质量赋值 (deposit)
# ---------------------------------------------------------------------------- #
def cic_deposit(positions: NDArray, masses: NDArray,
                n_grid: int, box_length: float) -> NDArray:
    """
    Cloud-In-Cell 质量赋值: 将粒子质量分配到网格。
        ρ[i,j,k] = (1/h³) Σ_p m_p W_CIC(x_p - x_ijk)
    其中 W_CIC 为三线性窗函数。

    Parameters
    ----------
    positions : (N_p, 3)
    masses : (N_p,)
    n_grid : int
    box_length : float

    Returns
    -------
    rho : (n_grid, n_grid, n_grid)
    """
    h = box_length / n_grid
    rho = np.zeros((n_grid, n_grid, n_grid))
    q = positions / h - 0.5
    i0 = np.floor(q).astype(int) % n_grid
    i1 = (i0 + 1) % n_grid
    frac = q - np.floor(q)
    fx, fy, fz = frac[:, 0], frac[:, 1], frac[:, 2]
    # 8 个最近网格节点的贡献:
    weights = np.array([
        (1 - fx) * (1 - fy) * (1 - fz),
        fx * (1 - fy) * (1 - fz),
        (1 - fx) * fy * (1 - fz),
        fx * fy * (1 - fz),
        (1 - fx) * (1 - fy) * fz,
        fx * (1 - fy) * fz,
        (1 - fx) * fy * fz,
        fx * fy * fz,
    ])  # (8, N_p)
    offsets = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
               (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
    for w_idx, (di, dj, dk) in enumerate(offsets):
        ii = (i0[:, 0] + di) % n_grid
        jj = (i0[:, 1] + dj) % n_grid
        kk = (i0[:, 2] + dk) % n_grid
        np.add.at(rho, (ii, jj, kk), masses * weights[w_idx])
    rho /= h ** 3
    return rho


def tsc_deposit(positions: NDArray, masses: NDArray,
                n_grid: int, box_length: float) -> NDArray:
    """
    Triangular-Shaped Cloud 质量赋值: 更高阶,27 点模板。
        W_TSC(x) = Π_{d∈{x,y,z}} (3/4 - (x_d/h)²)
    """
    h = box_length / n_grid
    rho = np.zeros((n_grid, n_grid, n_grid))
    q = positions / h
    i0 = np.floor(q).astype(int) % n_grid
    frac = q - np.floor(q)
    # 1D TSC 权重:
    def _w(f):
        w_m = 0.5 * (0.5 - f) ** 2
        w_0 = 0.75 - f ** 2
        w_p = 0.5 * (0.5 + f) ** 2
        return w_m, w_0, w_p
    wx_m, wx_0, wx_p = _w(frac[:, 0])
    wy_m, wy_0, wy_p = _w(frac[:, 1])
    wz_m, wz_0, wz_p = _w(frac[:, 2])
    weights = np.array([
        wx_m * wy_m * wz_m, wx_0 * wy_m * wz_m, wx_p * wy_m * wz_m,
        wx_m * wy_0 * wz_m, wx_0 * wy_0 * wz_m, wx_p * wy_0 * wz_m,
        wx_m * wy_p * wz_m, wx_0 * wy_p * wz_m, wx_p * wy_p * wz_m,
        wx_m * wy_m * wz_0, wx_0 * wy_m * wz_0, wx_p * wy_m * wz_0,
        wx_m * wy_0 * wz_0, wx_0 * wy_0 * wz_0, wx_p * wy_0 * wz_0,
        wx_m * wy_p * wz_0, wx_0 * wy_p * wz_0, wx_p * wy_p * wz_0,
        wx_m * wy_m * wz_p, wx_0 * wy_m * wz_p, wx_p * wy_m * wz_p,
        wx_m * wy_0 * wz_p, wx_0 * wy_0 * wz_p, wx_p * wy_0 * wz_p,
        wx_m * wy_p * wz_p, wx_0 * wy_p * wz_p, wx_p * wy_p * wz_p,
    ])
    offsets = [(di, dj, dk) for di in (-1, 0, 1)
               for dj in (-1, 0, 1) for dk in (-1, 0, 1)]
    for w_idx, (di, dj, dk) in enumerate(offsets):
        ii = (i0[:, 0] + di) % n_grid
        jj = (i0[:, 1] + dj) % n_grid
        kk = (i0[:, 2] + dk) % n_grid
        np.add.at(rho, (ii, jj, kk), masses * weights[w_idx])
    rho /= h ** 3
    return rho


# ---------------------------------------------------------------------------- #
#                邻居统计 (源自 NAS 的邻居数分布)
# ---------------------------------------------------------------------------- #
def neighbor_count_statistics(positions: NDArray, box_length: float,
                              r_smooth: float,
                              max_samples: int = 1000,
                              seed: int = 42) -> Dict[str, float]:
    """
    在随机采样位置统计 r_smooth 半径内的邻居数分布:
        mean, std, min, max, median
    用于评估密度场的局部涨落。
    """
    rng = np.random.default_rng(seed)
    grid = PeriodicHashGrid(box_length, cell_size=r_smooth)
    grid.build(positions)
    n = min(max_samples, positions.shape[0])
    idx = rng.choice(positions.shape[0], size=n, replace=False)
    counts = []
    for i in idx:
        nbrs = grid.query_radius(positions[i], r_smooth)
        counts.append(len(nbrs) - 1)  # 排除自身
    counts = np.array(counts)
    return {
        "mean": float(counts.mean()),
        "std": float(counts.std()),
        "min": int(counts.min()),
        "max": int(counts.max()),
        "median": float(np.median(counts)),
    }
