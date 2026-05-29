"""
banded_solver.py
带状矩阵 LU 分解与求解模块
融合种子项目：
  - 973_r8cb（紧凑带状矩阵无选主元 LU 分解）
  - 979_r8gb（一般带状矩阵 PLU 分解，带选主元）
"""
import numpy as np
from typing import Tuple, Optional


class BandedSolver:
    r"""
    带状线性系统求解器。

    对于 n x n 矩阵 A，下带宽 ml，上带宽 mu，
    紧凑存储为 A_band[ml+mu+1, n]，其中主对角线位于第 mu 行。

    融合 r8cb_np_fa（无选主元紧凑带状分解）与 r8gb_fa（带选主元一般带状分解）。
    """

    def __init__(self, n: int, ml: int, mu: int, compact: bool = True):
        self.n = n
        self.ml = ml
        self.mu = mu
        self.compact = compact
        self.m = mu + 1  # 主对角线在紧凑存储中的行索引
        self._lu: Optional[np.ndarray] = None
        self._pivot: Optional[np.ndarray] = None
        self._info: int = 0

    def full_to_compact(self, A_full: np.ndarray) -> np.ndarray:
        r"""
        将完整矩阵转换为紧凑带状存储。
        存储规则：A_band[k, j] = A_full[i, j], k = i - j + mu
        """
        A_band = np.zeros((self.ml + self.mu + 1, self.n))
        for j in range(self.n):
            i1 = max(0, j - self.mu)
            i2 = min(self.n - 1, j + self.ml)
            for i in range(i1, i2 + 1):
                k = i - j + self.mu
                A_band[k, j] = A_full[i, j]
        return A_band

    def compact_to_full(self, A_band: np.ndarray) -> np.ndarray:
        """将紧凑带状存储还原为完整矩阵。"""
        A_full = np.zeros((self.n, self.n))
        for j in range(self.n):
            i1 = max(0, j - self.mu)
            i2 = min(self.n - 1, j + self.ml)
            for i in range(i1, i2 + 1):
                k = i - j + self.mu
                A_full[i, j] = A_band[k, j]
        return A_full

    def factorize_np(self, A_band: np.ndarray) -> np.ndarray:
        r"""
        无选主元紧凑带状 LU 分解（融合 r8cb_np_fa）。
        适用于对称正定或已知可分解的矩阵。

        算法：对 k = 1, ..., n-1：
          1. 检查主元 A_band[mu, k] != 0
          2. 计算乘子 multipliers = A_band[mu+1:mu+lm, k] / A_band[mu, k]
          3. 更新子矩阵
        """
        # [HOLE 3]: 请实现紧凑带状矩阵的无选主元 LU 分解。
        # 提示：主对角线在紧凑存储中的行索引为 m = mu + 1
        # 对 k = 0, ..., n-2：检查主元、计算乘子、更新子矩阵
        pass

    def factorize_with_pivot(self, A_band: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        带选主元的一般带状 PLU 分解（融合 r8gb_fa）。
        存储需要 2*ml + mu + 1 行，允许 fill-in。

        算法：
        对 k = 1, ..., n-1：
          1. 在当前列的允许行中选主元
          2. 交换行
          3. 计算乘子并消元
        """
        nrow = 2 * self.ml + self.mu + 1
        alu = np.zeros((nrow, self.n))
        # 复制原始数据到下半部分
        for j in range(self.n):
            i1 = max(0, j - self.mu)
            i2 = min(self.n - 1, j + self.ml)
            for i in range(i1, i2 + 1):
                k = i - j + self.ml + self.mu
                alu[k, j] = A_band[i - j + self.mu, j]
        pivot = np.zeros(self.n, dtype=int)
        m = self.ml + self.mu + 1
        info = 0
        jz = self.mu + 1
        j1 = min(self.n, m) - 1
        for jz_idx in range(jz, j1 + 1):
            i0 = m + 1 - jz_idx
            if i0 <= self.ml:
                alu[i0 - 1:self.ml, jz_idx - 1] = 0.0
        jz = j1
        ju = 0
        for k in range(self.n - 1):
            jz += 1
            if jz <= self.n:
                alu[0:self.ml, jz - 1] = 0.0
            lm = min(self.ml, self.n - 1 - k)
            l = m - 1
            for j in range(m, m + lm):
                if abs(alu[l, k]) < abs(alu[j, k]):
                    l = j
            pivot[k] = l + k - m + 1
            if abs(alu[l, k]) < 1e-20:
                info = k + 1
                raise ValueError(f"Zero pivot at step {k} in banded PLU factorization")
            # 交换
            if l != m - 1:
                t = alu[l, k]
                alu[l, k] = alu[m - 1, k]
                alu[m - 1, k] = t
            alu[m:m + lm, k] = -alu[m:m + lm, k] / alu[m - 1, k]
            ju = min(ju + self.mu + pivot[k], self.n)
            ju = min(ju, self.n)
            mm = m
            for j in range(k + 1, ju):
                l -= 1
                mm -= 1
                if l != mm:
                    t = alu[l, j]
                    alu[l, j] = alu[mm, j]
                    alu[mm, j] = t
                alu[mm:mm + lm, j] = alu[mm:mm + lm, j] + alu[mm - 1, j] * alu[m:m + lm, k]
        pivot[self.n - 1] = self.n - 1
        if abs(alu[m - 1, self.n - 1]) < 1e-20:
            info = self.n
            raise ValueError("Zero pivot at final step in banded PLU factorization")
        self._lu = alu
        self._pivot = pivot
        self._info = info
        return alu, pivot

    def solve_np(self, alu: np.ndarray, b: np.ndarray) -> np.ndarray:
        r"""
        使用前向/后向替换求解紧凑带状 LU 系统（融合 r8cb_np_sl）。
        """
        x = b.copy()
        m = self.mu + 1
        # 前向替换 L y = b
        for k in range(self.n - 1):
            lm = min(self.ml, self.n - 1 - k)
            x[k + 1:k + 1 + lm] = x[k + 1:k + 1 + lm] + alu[m:m + lm, k] * x[k]
        # 后向替换 U x = y
        for k in range(self.n - 1, -1, -1):
            x[k] = x[k] / alu[m - 1, k]
            lm = min(self.ml, self.n - 1 - k)
            for i in range(1, lm + 1):
                if k - i >= 0:
                    x[k - i] = x[k - i] - alu[m - 1 + i, k - i] * x[k]
        return x

    def solve_with_pivot(self, alu: np.ndarray, pivot: np.ndarray, b: np.ndarray) -> np.ndarray:
        r"""
        使用 PLU 分解求解带状系统（融合 r8gb_trs）。
        """
        x = b.copy()
        m = self.ml + self.mu + 1
        # 前向替换（含置换）
        for k in range(self.n - 1):
            lm = min(self.ml, self.n - 1 - k)
            l = pivot[k]
            if l != k:
                t = x[l]
                x[l] = x[k]
                x[k] = t
            x[k + 1:k + 1 + lm] = x[k + 1:k + 1 + lm] + alu[m:m + lm, k] * x[k]
        # 后向替换
        for k in range(self.n - 1, -1, -1):
            x[k] = x[k] / alu[m - 1, k]
            lm = min(self.ml, self.n - 1 - k)
            for i in range(1, lm + 1):
                if k - i >= 0:
                    x[k - i] = x[k - i] - alu[m - 1 + i, k - i] * x[k]
        return x

    def solve_system(self, A_band: np.ndarray, b: np.ndarray, use_pivot: bool = False) -> np.ndarray:
        r"""
        封装：分解并求解带状系统。
        """
        if use_pivot:
            alu, pivot = self.factorize_with_pivot(A_band)
            return self.solve_with_pivot(alu, pivot, b)
        else:
            alu = self.factorize_np(A_band)
            return self.solve_np(alu, b)


def extract_banded_submatrix(K: np.ndarray, contact_nodes: np.ndarray,
                              n_nodes: int, ml: int, mu: int) -> np.ndarray:
    r"""
    从全局刚度矩阵中提取与接触自由度相关的带状子矩阵。
    用于快速局部接触修正。
    """
    n_c = len(contact_nodes)
    n_sub = 2 * n_c
    idx = np.zeros(n_sub, dtype=int)
    for i, node in enumerate(contact_nodes):
        idx[2 * i] = 2 * node
        idx[2 * i + 1] = 2 * node + 1
    K_sub = K[np.ix_(idx, idx)]
    # 转为紧凑带状存储
    solver = BandedSolver(n_sub, ml, mu, compact=True)
    return solver.full_to_compact(K_sub)
