"""
toeplitz_solver.py — 下三角Toeplitz矩阵的内存受限运算
=======================================================
融合来源: 985_r8ltt (下三角Toeplitz矩阵运算库)

在高性能外排序中，前缀和变换、累积分布函数构造以及数据块的
负载均衡矩阵均可建模为Toeplitz系统。下三角Toeplitz矩阵仅需
存储第一列的N个元素即可表示完整的N×N矩阵，内存复杂度为O(N)，
相比稠密矩阵的O(N^2)有数量级优势。
"""

import math
from typing import List, Tuple, Optional


class R8LTTSolver:
    """
    下三角Toeplitz矩阵求解器。

    下三角Toeplitz矩阵 T 的第一列为 [t_0, t_1, ..., t_{N-1}]^T，
    即 T_{ij} = t_{i-j}（当 i ≥ j），否则为 0：

        | t_0     0      0    ...  0    |
        | t_1    t_0     0    ...  0    |
        | t_2    t_1    t_0   ...  0    |
        |  ...   ...    ...   ...  ...  |
        | t_{N-1} t_{N-2} ... ... t_0  |

    利用对角线恒定性，矩阵-向量乘法可通过前向递推在 O(N^2) 内完成。
    """

    def __init__(self, first_col: List[float]):
        """
        参数:
            first_col: 矩阵第一列元素 [t_0, t_1, ..., t_{N-1}]
        """
        if not first_col:
            raise ValueError("First column must not be empty.")
        self.n = len(first_col)
        self.t = list(first_col)
        # 数值鲁棒性：检查对角元非零
        if abs(self.t[0]) < 1e-15:
            raise ValueError(
                f"Diagonal element t_0 = {self.t[0]} is too close to zero; "
                "matrix is singular or ill-conditioned."
            )

    def matvec(self, x: List[float]) -> List[float]:
        """
        计算矩阵-向量乘法 y = T · x。

        递推公式：
            y_i = Σ_{j=0}^{i} t_{i-j} · x_j,  i = 0, ..., N-1
        """
        if len(x) != self.n:
            raise ValueError(f"Dimension mismatch: x has {len(x)}, expected {self.n}")
        y = [0.0] * self.n
        for i in range(self.n):
            s = 0.0
            for j in range(i + 1):
                s += self.t[i - j] * x[j]
            y[i] = s
        return y

    def solve(self, b: List[float]) -> List[float]:
        """
        使用前向替换法求解 T · x = b。

        算法：
            x_0 = b_0 / t_0
            x_i = (b_i - Σ_{j=0}^{i-1} t_{i-j} · x_j) / t_0,  i ≥ 1

        时间复杂度 O(N^2)，空间复杂度 O(N)。
        """
        if len(b) != self.n:
            raise ValueError(f"Dimension mismatch: b has {len(b)}, expected {self.n}")
        x = [0.0] * self.n
        x[0] = b[0] / self.t[0]
        for i in range(1, self.n):
            s = 0.0
            for j in range(i):
                s += self.t[i - j] * x[j]
            x[i] = (b[i] - s) / self.t[0]
        return x

    def determinant(self) -> float:
        """
        行列式 det(T) = (t_0)^N。

        下三角矩阵的行列式为对角元乘积，Toeplitz结构保证所有对角元相等。
        """
        return self.t[0] ** self.n

    def inverse(self) -> 'R8LTTSolver':
        """
        计算Toeplitz矩阵的逆矩阵（仍为下三角Toeplitz矩阵）。

        利用递推公式：设逆矩阵第一列为 [s_0, s_1, ..., s_{N-1}]，则
            s_0 = 1 / t_0
            Σ_{k=0}^{i} t_{i-k} · s_k = δ_{i0},  i ≥ 1
        即
            s_i = -(1/t_0) · Σ_{k=0}^{i-1} t_{i-k} · s_k
        """
        s = [0.0] * self.n
        s[0] = 1.0 / self.t[0]
        for i in range(1, self.n):
            accum = 0.0
            for k in range(i):
                accum += self.t[i - k] * s[k]
            s[i] = -accum / self.t[0]
        return R8LTTSolver(s)

    def condition_estimate(self) -> float:
        """
        简单的条件数估计：||T||_∞ · ||T^{-1}||_∞。

        对于下三角Toeplitz矩阵，∞-范数为每行绝对值之和的最大值：
            ||T||_∞ = max_i Σ_{j=0}^{i} |t_{i-j}| = Σ_{k=0}^{N-1} |t_k|
        """
        norm_t = sum(abs(v) for v in self.t)
        inv = self.inverse()
        norm_inv = sum(abs(v) for v in inv.t)
        return norm_t * norm_inv


def build_prefix_toeplitz(n: int, decay: float = 0.5) -> R8LTTSolver:
    """
    构造用于前缀和运算的Toeplitz矩阵。

    第一列为 [1, decay, decay^2, ..., decay^{N-1}]，对应指数衰减权重。
    在外排序的负载均衡中，该矩阵可将历史负载加权累积为当前负载估计：
        L_i = Σ_{j=0}^{i} decay^{i-j} · w_j
    其中 w_j 为第 j 个数据块的权重。
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    first_col = [decay ** i for i in range(n)]
    return R8LTTSolver(first_col)


def toeplitz_transform_keys(keys: List[float], decay: float = 0.9) -> List[float]:
    """
    对键值序列施加Toeplitz变换，增强局部相关性。

    变换公式：
        y_i = Σ_{j=0}^{i} decay^{i-j} · (keys_j - μ)
    其中 μ 为序列均值。该变换将白噪声映射为具有时间相关性的过程，
    用于模拟科学数据流中的时间序列相关性。
    """
    n = len(keys)
    if n == 0:
        return []
    mu = sum(keys) / n
    centered = [k - mu for k in keys]
    solver = build_prefix_toeplitz(n, decay)
    return solver.matvec(centered)
