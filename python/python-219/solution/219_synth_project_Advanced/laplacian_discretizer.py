"""
laplacian_discretizer.py
========================

二维五点差分 Laplacian 稀疏算子构造模块。

融合种子项目:
  - 269_delsq   : Cleve Moler 的五点差分 Laplacian 稀疏矩阵构造
  - 244_cvt_1d_lumping : 1D Lloyd CVT 算法中的空间离散化思想

在 PDE 约束最优控制中, 该模块用于:
  1. 构建热扩散方程的空间离散 Laplacian  -Delta_h
  2. 为状态-伴随系统的 PDE 约束提供空间算子
  3. 与 CVT 最优空间网格配合, 在非均匀网格上离散

数学公式:
---------
1. 五点差分 Laplacian  (在矩形网格上):
     (-Delta_h u)_{i,j} = (1/h^2) * (4 u_{i,j} - u_{i-1,j}
                                    - u_{i+1,j} - u_{i,j-1} - u_{i,j+1})

2. 网格索引映射 (一维编号):
     节点 (i, j) 对应一维索引 G[i, j], 其中 G 为网格节点编号矩阵

3. 谱半径 (用于显式格式稳定性):
     rho(-Delta_h) = (4 / h^2) * (sin^2(pi/(2m)) + sin^2(pi/(2m)))

4. 与 CVT 网格耦合:
     设 CVT 生成器集合 {z_k}, 每个 Voronoi 单元 Omega_k 内的
     局部 Laplacian 采用单元平均尺寸 h_k.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class SparseEntry:
    """稀疏矩阵三元组 (row, col, value)."""
    row: int
    col: int
    val: float


class LaplacianDiscretizer:
    """五点差分 Laplacian 稀疏算子 (来自 269_delsq).

    在 m x m 网格上构造离散负 Laplacian, 返回稀疏三元组列表.
    内部节点编号由输入 grid G 决定, 非零位置对应于内部节点.

    Parameters
    ----------
    grid : List[List[int]]
        m x m 整数矩阵, 正值项表示内部节点编号 (1-based).
        零或负值项表示边界或外部.
    """

    def __init__(self, grid: List[List[int]]):
        if not grid or not grid[0]:
            raise ValueError("网格不能为空")
        m = len(grid)
        for row in grid:
            if len(row) != m:
                raise ValueError(f"网格非方阵: 期望 {m} 列, 实际 {len(row)} 列")
        self._grid: List[List[int]] = grid
        self._m: int = m
        self._entries: List[SparseEntry] = []
        self._nnz: int = 0
        self._build()

    @property
    def size(self) -> int:
        """矩阵阶数 n = 内部节点数."""
        max_idx = 0
        for row in self._grid:
            for v in row:
                if v > max_idx:
                    max_idx = v
        return max_idx

    @property
    def nnz(self) -> int:
        return self._nnz

    @property
    def entries(self) -> List[SparseEntry]:
        return self._entries

    def _build(self) -> None:
        """核心算法 (来自 269_delsq):

        1. 找到所有内部节点 p = {G[i,j] > 0}
        2. 对角元: 每个内部节点 4 (对应 5 点模板中心)
        3. 非对角元: 对每个邻居方向 k in {-1, +m, +1, -m},
           若 G[p + k] > 0, 则该邻居贡献 -1
        """
        m = self._m
        grid = self._grid

        # 提取内部节点
        interior: List[Tuple[int, int, int]] = []  # (row, col, idx)
        for i in range(m):
            for j in range(m):
                if grid[i][j] > 0:
                    interior.append((i, j, grid[i][j]))

        entries: List[SparseEntry] = []

        # 1. 对角元 (系数 4)
        for _, _, idx in interior:
            entries.append(SparseEntry(idx, idx, 4.0))

        # 2. 邻居方向: north(-1col), east(+1row), south(+1col), west(-1row)
        #    在 2D 网格中, 对应偏移 (-1, 0), (0, 1), (1, 0), (0, -1)
        offsets = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        for di, dj in offsets:
            for i, j, idx in interior:
                ni, nj = i + di, j + dj
                if 0 <= ni < m and 0 <= nj < m:
                    neighbor_idx = grid[ni][nj]
                    if neighbor_idx > 0:
                        entries.append(SparseEntry(idx, neighbor_idx, -1.0))

        self._entries = entries
        self._nnz = len(entries)

    def to_dense(self) -> List[List[float]]:
        """转为稠密矩阵 (用于小规模问题)."""
        n = self.size
        mat = [[0.0] * n for _ in range(n)]
        for e in self._entries:
            mat[e.row - 1][e.col - 1] = e.val
        return mat

    def matvec(self, x: List[float]) -> List[float]:
        """稀疏矩阵-向量乘 y = D x.

        索引约定: x[0] 对应节点编号 1.
        """
        if len(x) != self.size:
            raise ValueError(
                f"向量长度 {len(x)} 与矩阵阶数 {self.size} 不匹配"
            )
        y = [0.0] * self.size
        for e in self._entries:
            y[e.row - 1] += e.val * x[e.col - 1]
        return y

    def spectral_radius_estimate(self) -> float:
        """Laplacian 谱半径的估计 (幂法).

        对 m x m 方形网格上的 5 点差分 Laplacian, 理论谱半径为:
            rho = 4 * (sin^2(pi/(2m)) + sin^2(pi/(2m))) * ...  (简化)
        这里通过幂法 50 次迭代得到数值估计.
        """
        import math
        n = self.size
        if n == 0:
            return 0.0

        # 初始向量 (全部为 1)
        v = [1.0 / math.sqrt(n)] * n
        sigma = 1.0
        for _ in range(50):
            w = self.matvec(v)
            sigma = math.sqrt(sum(wi * wi for wi in w))
            if sigma < 1e-15:
                return 0.0
            v = [wi / sigma for wi in w]
        return sigma

    def apply_inverse_power(self, rhs: List[float], n_iter: int = 20) -> List[float]:
        """简单 Jacobi 迭代求解 D x = rhs (用于小规模演示).

        x^{k+1}_i = (1/4) * (rhs_i + sum_{j != i} x^k_j)
        """
        if len(rhs) != self.size:
            raise ValueError("向量长度不匹配")
        n = self.size
        x = [0.0] * n
        for _ in range(n_iter):
            x_new = list(x)
            for e in self._entries:
                i, j, v = e.row - 1, e.col - 1, e.val
                if i == j:
                    continue
                x_new[i] -= v * x[j]
            for i in range(n):
                x_new[i] /= 4.0
            x = x_new
        return x


# ===========================================================================
# 辅助: 构造矩形网格
# ===========================================================================
def make_rectangular_grid(m: int) -> List[List[int]]:
    """构造 m x m 网格, 所有内部节点编号为 1 .. m^2.

    Parameters
    ----------
    m : int
        每边的节点数.

    Returns
    -------
    List[List[int]]
        m x m 整数矩阵.
    """
    if m <= 0:
        raise ValueError("m 必须为正")
    grid = [[0] * m for _ in range(m)]
    k = 1
    for i in range(m):
        for j in range(m):
            grid[i][j] = k
            k += 1
    return grid


def make_annular_grid(m: int, r_inner_frac: float = 0.3) -> List[List[int]]:
    """构造环形网格 (内部有洞), 模拟 CVT 非均匀离散.

    以 (m-1)/2 为中心, 半径 < r_inner_frac * (m-1)/2 的节点被挖去.
    """
    if m <= 2:
        raise ValueError("m 必须 >= 3")
    center = (m - 1) / 2.0
    r_inner = r_inner_frac * center
    grid = [[0] * m for _ in range(m)]
    k = 1
    for i in range(m):
        for j in range(m):
            r = ((i - center) ** 2 + (j - center) ** 2) ** 0.5
            if r >= r_inner:
                grid[i][j] = k
                k += 1
    return grid


# ===========================================================================
# 自检
# ===========================================================================
def self_check() -> None:
    """Laplacian 模块自检."""
    m = 4
    grid = make_rectangular_grid(m)
    disc = LaplacianDiscretizer(grid)
    print(f"[Laplacian] m={m}, n={disc.size}, nnz={disc.nnz}")
    rho = disc.spectral_radius_estimate()
    print(f"[Laplacian] 谱半径估计 = {rho:.4f}")

    # 环形网格
    grid2 = make_annular_grid(5, 0.3)
    disc2 = LaplacianDiscretizer(grid2)
    print(f"[Laplacian annular] m=5, n={disc2.size}, nnz={disc2.nnz}")


if __name__ == "__main__":
    self_check()
