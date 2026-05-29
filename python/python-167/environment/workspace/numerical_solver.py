"""
numerical_solver.py
大规模数值线性代数求解模块。
融入种子项目：
  - 786_nas（NAS 基准测试：Cholesky 分解、块三对角求解、FFT、矩阵乘法）
  - 1004_r8vm（Vandermonde 矩阵求解）

科学背景：
多足机器人全身动力学涉及大规模稀疏/块结构线性系统：
    H(q)·q̈ + C(q,q̇) = τ
其中 H(q) ∈ R^{n×n} 为对称正定质量矩阵，常用 Cholesky 分解求解。
对于多腿耦合系统，H 呈块三对角结构，可用专用前向-后向消元算法高效求解。
频域分析（FFT）用于步态周期性的功率谱密度估计。
"""

import numpy as np
from typing import Tuple, List
from utils import robust_sqrt, check_numerical_singularity


class CholeskySolver:
    """
    源自 nas.m 中 cholsky() 的 Cholesky 分解与回代求解。

    数学公式：
    对对称正定矩阵 A，存在唯一分解 A = L·L^T，其中 L 为下三角矩阵。
    分解算法（i 从 0 到 n-1）：
        L_{ii} = sqrt( A_{ii} - Σ_{k=0}^{i-1} L_{ik}^2 )
        L_{ji} = ( A_{ji} - Σ_{k=0}^{i-1} L_{jk}·L_{ik} ) / L_{ii}   (j > i)
    求解 Ax = b：
        前代：L·y = b  →  y
        回代：L^T·x = y →  x
    """

    def __init__(self, eps: float = 1e-13):
        self.eps = eps

    def decompose(self, A: np.ndarray) -> np.ndarray:
        """
        返回下三角矩阵 L，满足 A = L·L^T。
        边界处理：若对角元过小，加入 eps 保证正定性。
        """
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("A must be square matrix")
        n = A.shape[0]
        L = np.zeros_like(A)
        for i in range(n):
            diag_sum = sum(L[i, k] ** 2 for k in range(i))
            val = A[i, i] - diag_sum
            # 鲁棒性：确保正对角元
            L[i, i] = robust_sqrt(val, self.eps)
            for j in range(i + 1, n):
                off_sum = sum(L[j, k] * L[i, k] for k in range(i))
                L[j, i] = (A[j, i] - off_sum) / L[i, i]
        return L

    def solve(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        求解 Ax = b。
        """
        L = self.decompose(A)
        n = L.shape[0]
        b = np.asarray(b, dtype=float).flatten()
        # 前代
        y = np.zeros(n)
        for i in range(n):
            y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
        # 回代
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            x[i] = (y[i] - np.dot(L[i + 1:, i], x[i + 1:])) / L[i, i]
        return x


class BlockTridiagonalSolver:
    """
    源自 nas.m 中 btrix() 的块三对角求解器，适配多腿机器人耦合系统。

    数学结构：
    对块三对角系统
        A_i·x_{i-1} + B_i·x_i + C_i·x_{i+1} = d_i    (i = 1..N)
    其中 A_i, B_i, C_i ∈ R^{m×m}，采用块 LU 分解：
        L_i·U_{i-1} = A_i
        L_i = B_i - A_i·U_{i-1}^{-1}·C_{i-1}   (修正的块消元)
    本实现采用简化的前向-后向块消元（Thomas 算法的块版本）。
    """

    def __init__(self, block_size: int = 3):
        self.m = block_size

    def solve(self, lower_blocks: List[np.ndarray], diag_blocks: List[np.ndarray],
              upper_blocks: List[np.ndarray], rhs: List[np.ndarray]) -> List[np.ndarray]:
        """
        lower_blocks: A_2..A_N  (N-1 个 m×m)
        diag_blocks:  B_1..B_N  (N 个 m×m)
        upper_blocks: C_1..C_{N-1} (N-1 个 m×m)
        rhs:          d_1..d_N  (N 个 m 维向量)
        返回 x_1..x_N。
        """
        N = len(diag_blocks)
        if N == 0:
            return []
        # 前向消元
        L = [np.zeros((self.m, self.m)) for _ in range(N)]
        U = [np.zeros((self.m, self.m)) for _ in range(N)]
        y = [np.zeros(self.m) for _ in range(N)]

        L[0] = diag_blocks[0].copy()
        y[0] = rhs[0].copy()
        for i in range(1, N):
            # U_{i-1} = L_{i-1}^{-1} · C_{i-1}
            U[i - 1] = np.linalg.solve(L[i - 1], upper_blocks[i - 1])
            # L_i = B_i - A_i · U_{i-1}
            L[i] = diag_blocks[i] - lower_blocks[i - 1] @ U[i - 1]
            # y_i = d_i - A_i · (L_{i-1}^{-1} · y_{i-1})
            temp = np.linalg.solve(L[i - 1], y[i - 1])
            y[i] = rhs[i] - lower_blocks[i - 1] @ temp

        # 回代
        x = [np.zeros(self.m) for _ in range(N)]
        x[-1] = np.linalg.solve(L[-1], y[-1])
        for i in range(N - 2, -1, -1):
            x[i] = np.linalg.solve(L[i], y[i] - U[i] @ x[i + 1])
        return x


class Radix2FFT:
    """
    源自 nas.m 中 cfft2d1/cfft2d2 的基-2 FFT 实现，适配一维实数序列。

    数学公式（离散傅里叶变换）：
        X_k = Σ_{n=0}^{N-1} x_n · e^{-i·2π·k·n/N},   k = 0..N-1
    逆变换：
        x_n = (1/N) · Σ_{k=0}^{N-1} X_k · e^{i·2π·k·n/N}

    Cooley-Tukey 基-2 算法将 DFT 分解为两个 N/2 点 DFT：
        X_k       = E_k + e^{-i·2π·k/N} · O_k
        X_{k+N/2} = E_k - e^{-i·2π·k/N} · O_k
    其中 E_k 为偶数样本 DFT，O_k 为奇数样本 DFT。
    计算复杂度从 O(N^2) 降至 O(N log N)。
    """

    def __init__(self):
        pass

    def _bit_reverse(self, n: int, bits: int) -> int:
        rev = 0
        for i in range(bits):
            rev = (rev << 1) | ((n >> i) & 1)
        return rev

    def fft(self, x: np.ndarray) -> np.ndarray:
        """
        一维复数 FFT。输入 x 长度必须为 2 的幂。
        """
        N = len(x)
        if N == 0 or (N & (N - 1)) != 0:
            raise ValueError("FFT input length must be power of 2")
        bits = int(np.log2(N))
        X = np.array(x, dtype=complex)
        # 位反转重排
        for i in range(N):
            j = self._bit_reverse(i, bits)
            if i < j:
                X[i], X[j] = X[j], X[i]
        # 蝶形运算
        length = 2
        while length <= N:
            half = length // 2
            for start in range(0, N, length):
                for k in range(half):
                    twiddle = np.exp(-2j * np.pi * k / length)
                    even = X[start + k]
                    odd = twiddle * X[start + k + half]
                    X[start + k] = even + odd
                    X[start + k + half] = even - odd
            length *= 2
        return X

    def ifft(self, X: np.ndarray) -> np.ndarray:
        """
        一维逆 FFT。
        """
        x_conj = self.fft(np.conj(X))
        return np.conj(x_conj) / len(X)

    def power_spectral_density(self, signal: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算功率谱密度 S_{xx}(f) = |X(f)|^2 / (N·Δt)。
        返回频率数组 f 与 PSD 数组。
        """
        N = len(signal)
        X = self.fft(signal)
        psd = np.abs(X) ** 2 / (N * dt)
        freqs = np.fft.fftfreq(N, dt)
        return freqs, psd


class VandermondeSolver:
    """
    源自 r8vm_sl.m 的 Vandermonde 矩阵求解。

    数学背景：
    Vandermonde 矩阵 V_{ij} = x_i^{j-1}，i,j = 1..n。
    用于多项式插值：给定节点 x_i 与值 y_i，求系数 c 使得
        p(x) = Σ_{j=0}^{n-1} c_j · x^j
    满足 p(x_i) = y_i，即 V·c = y。

    求解算法（Neville/Aitken 型递推，O(n^2)）：
    1. 前向差分：对 j = 1..n-1, i = n..j+1:
           y_i = y_i - x_j · y_{i-1}
    2. 回代：对 j = n-1..1:
           y_i = y_i / (x_i - x_{i-j})    (i = j+1..n)
           y_i = y_i - y_{i+1}            (i = j..n-1)
    """

    def __init__(self):
        pass

    def solve(self, x_nodes: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        求解 V(x_nodes)·c = b。
        返回 (c, info)，info=0 表示成功，info=1 表示有重复节点（奇异）。
        """
        x = np.asarray(x_nodes, dtype=float).flatten()
        b_arr = np.asarray(b, dtype=float).flatten()
        n = x.size
        if b_arr.size != n:
            raise ValueError("b length must match x_nodes length")
        # 显式检测重复节点
        for j in range(n - 1):
            for i in range(j + 1, n):
                if np.isclose(x[i], x[j]):
                    return np.zeros(n), 1
        y = b_arr.copy()
        # 前向差分
        for j in range(n - 1):
            for i in range(n - 1, j, -1):
                y[i] = y[i] - x[j] * y[i - 1]
        # 回代
        for j in range(n - 2, -1, -1):
            for i in range(j + 1, n):
                y[i] = y[i] / (x[i] - x[i - j - 1])
            for i in range(j, n - 1):
                y[i] = y[i] - y[i + 1]
        return y, 0

    def evaluate(self, x_nodes: np.ndarray, coeffs: np.ndarray, x_query: np.ndarray) -> np.ndarray:
        """
        用 Horner 法则求多项式值：
            p(x) = c_0 + x·(c_1 + x·(c_2 + ... ))
        """
        coeffs = np.asarray(coeffs, dtype=float)
        x_query = np.asarray(x_query, dtype=float)
        result = np.zeros_like(x_query)
        for c in reversed(coeffs):
            result = result * x_query + c
        return result


class MatrixMultiplyBenchmark:
    """
    源自 nas.m 中 mxm() 的矩阵乘法，提供基础线性代数运算。
    """

    @staticmethod
    def multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        标准矩阵乘法 C = A·B，含维度检查。
        """
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)
        if A.ndim != 2 or B.ndim != 2:
            raise ValueError("A and B must be 2D arrays")
        if A.shape[1] != B.shape[0]:
            raise ValueError("Incompatible shapes for matrix multiplication")
        return A @ B
