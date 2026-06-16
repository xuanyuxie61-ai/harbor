"""
block_toeplitz_operator.py
==========================

分块 Toeplitz 矩阵代数模块。

融合种子项目:
  - 971_r8bto : John Burkardt 的 R8BTO 分块 Toeplitz 矩阵工具集

在 Pontryagin 最优控制中, 该模块用于:
  1. 构建时间离散后的状态转移矩阵 (分块 Toeplitz 结构)
  2. 表示线性时不变离散系统的输入-输出映射
  3. 在多阶段最优控制中描述跨阶段的耦合算子
  4. 为打靶法提供高效的矩阵-向量乘法

数学公式:
---------
1. 分块 Toeplitz 矩阵:
     T = [ T_0     T_{-1}   T_{-2}   ...  T_{-(L-1)} ]
         [ T_1     T_0      T_{-1}   ...  T_{-(L-2)} ]
         [ T_2     T_1      T_0      ...  T_{-(L-3)} ]
         ...
     其中 T_k 为 m x m 子块

2. 存储格式 (R8BTO):
     存储 L 个第一行子块 + (L-1) 个第一列子块
     共 (2L - 1) 个 m x m 子块, 存储于 (m, m, 2L-1) 数组

3. 矩阵-向量乘:
     y = T x, 其中 x, y 为分块向量 (L 块, 每块 m 维)
     计算量 ~ O(L^2 m^2), 但利用 Toeplitz 结构可降至 O(L m^2 log L)

4. 在最优控制中的应用:
     离散伴随方程  lambda_{k+1} = A^T lambda_k + Q x_k
     可写成  Lambda = T_Lambda X,  其中 T_Lambda 为分块 Toeplitz
"""

from __future__ import annotations

import math
from typing import List, Tuple


class BlockToeplitzOperator:
    """分块 Toeplitz 矩阵算子.

    Parameters
    ----------
    m : int
        每个子块的阶数.
    blocks : List[List[List[float]]]
        长度为 (2L - 1) 的子块列表, 存储格式:
            blocks[0..L-1]  : 第一行的 L 个子块 T_0, T_{-1}, ..., T_{-(L-1)}
            blocks[L..2L-2] : 第一列 (不含 T_0) 的 L-1 个子块
                              T_1, T_2, ..., T_{L-1}
    """

    def __init__(self, m: int, blocks: List[List[List[float]]]):
        if m <= 0:
            raise ValueError(f"子块阶数 m={m} 必须为正")
        n_blocks = len(blocks)
        if n_blocks % 2 == 0:
            raise ValueError(f"子块数 {n_blocks} 必须为奇数 (= 2L - 1)")
        L = (n_blocks + 1) // 2
        if L <= 0:
            raise ValueError("L 必须 >= 1")

        # 验证每个子块
        for k, blk in enumerate(blocks):
            if len(blk) != m:
                raise ValueError(f"子块 {k} 行数 {len(blk)} != m={m}")
            for row in blk:
                if len(row) != m:
                    raise ValueError(f"子块 {k} 列数 {len(row)} != m={m}")

        self._m = m
        self._L = L
        self._blocks = blocks

    @property
    def m(self) -> int:
        return self._m

    @property
    def L(self) -> int:
        return self._L

    @property
    def n(self) -> int:
        """总阶数 N = m * L."""
        return self._m * self._L

    def _get_block(self, i: int, j: int) -> List[List[float]]:
        """获取 (i, j) 位置的 m x m 子块.

        在分块 Toeplitz 矩阵中, T[i, j] 仅依赖于 (i - j):
            T[i, j] = blocks[L - 1 + (i - j)]
        其中 blocks 索引:
            k = 0 .. L-1  对应  T_0, T_{-1}, ..., T_{-(L-1)}
            k = L .. 2L-2 对应  T_1, T_2, ..., T_{L-1}
        """
        diff = i - j
        if diff <= 0:
            # T_{-diff} = blocks[-diff]  (第一行)
            idx = -diff
        else:
            # T_{diff} = blocks[L - 1 + diff]  (第一列)
            idx = self._L - 1 + diff
        if idx < 0 or idx >= len(self._blocks):
            # 越界返回零矩阵
            return [[0.0] * self._m for _ in range(self._m)]
        return self._blocks[idx]

    def matvec(self, x: List[float]) -> List[float]:
        """分块矩阵-向量乘 y = T x.

        x, y 为 N = m * L 维向量.
        """
        if len(x) != self.n:
            raise ValueError(f"向量长度 {len(x)} != N={self.n}")
        y = [0.0] * self.n
        m, L = self._m, self._L
        for i in range(L):
            for j in range(L):
                blk = self._get_block(i, j)
                # y[i*m : (i+1)*m] += blk * x[j*m : (j+1)*m]
                for r in range(m):
                    s = 0.0
                    for c in range(m):
                        s += blk[r][c] * x[j * m + c]
                    y[i * m + r] += s
        return y

    def to_dense(self) -> List[List[float]]:
        """转为稠密 N x N 矩阵."""
        N = self.n
        mat = [[0.0] * N for _ in range(N)]
        m, L = self._m, self._L
        for i in range(L):
            for j in range(L):
                blk = self._get_block(i, j)
                for r in range(m):
                    for c in range(m):
                        mat[i * m + r][j * m + c] = blk[r][c]
        return mat

    def frobenius_norm(self) -> float:
        """Frobenius 范数 ||T||_F = sqrt(sum T_{ij}^2)."""
        s = 0.0
        N = self.n
        dense = self.to_dense()
        for row in dense:
            for v in row:
                s += v * v
        return math.sqrt(s)

    def spectral_radius_estimate(self, n_iter: int = 30) -> float:
        """谱半径估计 (幂法)."""
        N = self.n
        if N == 0:
            return 0.0
        v = [1.0 / math.sqrt(N)] * N
        sigma = 1.0
        for _ in range(n_iter):
            w = self.matvec(v)
            sigma = math.sqrt(sum(wi * wi for wi in w))
            if sigma < 1e-15:
                return 0.0
            v = [wi / sigma for wi in w]
        return sigma


# ===========================================================================
# 辅助构造器
# ===========================================================================
def make_toeplitz_indicator(m: int, L: int) -> BlockToeplitzOperator:
    """构造 indicator 分块 Toeplitz 矩阵 (来自 971_r8bto/r8bto_indicator).

    子块 T_k 的元素为顺序编号:
        T_k[i, j] = (i * m + j + 1 + k * m * m) mod (m * m) + 1
    """
    if m <= 0 or L <= 0:
        raise ValueError("m, L 必须为正")
    n_blocks = 2 * L - 1
    blocks: List[List[List[float]]] = []
    for k in range(n_blocks):
        blk = [[0.0] * m for _ in range(m)]
        for i in range(m):
            for j in range(m):
                blk[i][j] = float((i * m + j + k * m * m) % (m * m) + 1)
        blocks.append(blk)
    return BlockToeplitzOperator(m, blocks)


def make_discrete_state_transition(
    A: List[List[float]], B: List[List[float]], L: int
) -> BlockToeplitzOperator:
    """构造离散 LTI 系统的状态转移分块 Toeplitz 矩阵.

    系统:  x_{k+1} = A x_k + B u_k
    输入-输出映射 (零初始条件):
        x_k = sum_{j=0}^{k-1} A^{k-1-j} B u_j

    对应的分块 Toeplitz 下三角矩阵:
        T[0, 0] = 0,  T[k, j] = A^{k-1-j} B  (k > j)

    Parameters
    ----------
    A : m x m 矩阵
    B : m x p 矩阵 (此处简化为 m x m)
    L : 时间步数

    Returns
    -------
    BlockToeplitzOperator
    """
    m = len(A)
    if m <= 0 or L <= 0:
        raise ValueError("m, L 必须为正")

    # 矩阵乘法辅助
    def mat_mul(X, Y):
        n = len(X)
        Z = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                s = 0.0
                for k in range(n):
                    s += X[i][k] * Y[k][j]
                Z[i][j] = s
        return Z

    # 计算 A^k for k = 0 .. L-1
    I = [[1.0 if i == j else 0.0 for j in range(m)] for i in range(m)]
    A_powers = [I]
    for _ in range(1, L):
        A_powers.append(mat_mul(A_powers[-1], A))

    # T_0 = 0,  T_{-k} = 0 (上三角部分)
    # T_k = A^{k-1} B  (下三角部分, k >= 1)
    # 存储格式: blocks[0..L-1] = 第一行, blocks[L..2L-2] = 第一列
    Z = [[0.0] * m for _ in range(m)]
    blocks: List[List[List[float]]] = [Z for _ in range(L)]  # 第一行全零
    for k in range(1, L):
        blocks.append(mat_mul(A_powers[k - 1], B))  # 第一列

    return BlockToeplitzOperator(m, blocks)


# ===========================================================================
# 自检
# ===========================================================================
def self_check() -> None:
    """分块 Toeplitz 模块自检."""
    m, L = 2, 3
    op = make_toeplitz_indicator(m, L)
    print(f"[BlockToeplitz] indicator m={m}, L={L}, N={op.n}")
    print(f"  Frobenius norm = {op.frobenius_norm():.4f}")
    print(f"  谱半径估计     = {op.spectral_radius_estimate():.4f}")

    # 离散状态转移
    A = [[0.9, 0.1], [0.0, 0.8]]
    B = [[0.1], [0.2]]
    # 补齐为方阵
    B_sq = [[0.1, 0.0], [0.2, 0.0]]
    op2 = make_discrete_state_transition(A, B_sq, L=4)
    print(f"[StateTransition] m=2, L=4, N={op2.n}")
    print(f"  谱半径估计 = {op2.spectral_radius_estimate():.4f}")


if __name__ == "__main__":
    self_check()
