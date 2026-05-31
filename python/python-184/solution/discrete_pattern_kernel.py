"""
Discrete Convolution Kernels over GF(2) for Binary Anomaly Pattern Detection
============================================================================
源自种子项目 673_lights_out_game (GF(2) linear algebra, discrete convolution kernel)。

Lights Out 的数学本质是 GF(2) 上的线性代数：
- 棋盘状态向量 v ∈ GF(2)^{MN}
- 操作矩阵 A ∈ GF(2)^{MN x MN}，其中 A_{ij}=1 表示操作 j 影响格子 i
- 求解 A x = v (mod 2)

在时间序列异常检测中，将其转化为：
- 二值化时间序列：v_t = 1 表示时刻 t 异常，0 表示正常
- 离散卷积核 K 表示异常传播模式（如相邻时刻的级联效应）
- GF(2) 上的卷积检测特定异常模式（如连续异常 burst）

核心数学：
1. GF(2) 运算：加法 = XOR，乘法 = AND
2. 离散卷积（一维）：
    (K * v)_i = sum_j K_j v_{i-j}  (mod 2)
3. 模式检测：设计核 K 使得仅当特定局部模式出现时输出 1
   例如检测 "连续 3 个异常"：K = [1,1,1]，要求 (K*v)_i = 1 (mod 2) 且 ...

4. 状态转移矩阵：
   对于滑动窗口模型，转移矩阵 T 描述异常状态的概率演化。
"""

import numpy as np
from typing import List


class DiscretePatternKernel:
    """
    基于 GF(2) 离散卷积的二值时间序列异常模式检测。
    """

    def __init__(self, window_size: int = 5):
        self.w = window_size

    def binarize(self, series: np.ndarray, threshold: float | None = None) -> np.ndarray:
        """
        将连续时间序列二值化。
        若 threshold 为 None，使用均值 + 1.5 标准差。
        """
        if threshold is None:
            threshold = np.mean(series) + 1.5 * np.std(series)
        return (np.abs(series) > threshold).astype(int)

    def gf2_convolve(self, signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        GF(2) 一维卷积：结果对 2 取模。
        signal, kernel ∈ {0,1}
        """
        n = len(signal)
        k = len(kernel)
        result = np.zeros(n, dtype=int)
        for i in range(n):
            acc = 0
            for j in range(k):
                idx = i - j
                if 0 <= idx < n:
                    acc += signal[idx] * kernel[j]
            result[i] = acc % 2
        return result

    def detect_burst_pattern(self, binary_series: np.ndarray, burst_length: int = 3) -> np.ndarray:
        """
        检测连续 burst_length 个 1 的模式。
        使用滑动窗口 AND 运算。
        """
        n = len(binary_series)
        result = np.zeros(n, dtype=int)
        for i in range(n - burst_length + 1):
            if np.all(binary_series[i:i + burst_length] == 1):
                result[i + burst_length // 2] = 1
        return result

    def detect_alternating_pattern(self, binary_series: np.ndarray) -> np.ndarray:
        """
        检测交替模式 101010...（可能表示振荡异常）。
        使用核 K = [1,0,1,0,1] 进行 GF(2) 卷积。
        """
        kernel = np.array([1, 0, 1, 0, 1])
        conv = self.gf2_convolve(binary_series, kernel)
        # 额外检查中心对称性
        result = np.zeros(len(binary_series), dtype=int)
        for i in range(2, len(binary_series) - 2):
            if conv[i] == 1 and binary_series[i] == 1:
                result[i] = 1
        return result

    def transition_matrix_analysis(self, binary_series: np.ndarray) -> np.ndarray:
        """
        估计 2 状态（0/1）Markov 转移矩阵：
            P(s_{t+1} = j | s_t = i) = N_{ij} / N_{i·}
        转移矩阵可用于异常状态演化预测。
        """
        n = len(binary_series)
        counts = np.zeros((2, 2))
        for i in range(n - 1):
            curr = int(binary_series[i])
            next_ = int(binary_series[i + 1])
            counts[curr, next_] += 1

        # 平滑处理避免零概率
        counts += 0.5
        P = counts / counts.sum(axis=1, keepdims=True)
        return P

    def solve_state_transition(self, target_state: np.ndarray, kernel_matrix: np.ndarray) -> np.ndarray | None:
        """
        在 GF(2) 上求解线性系统 A x = b，找到达到目标状态的操作序列。
        使用高斯消元（模 2）。
        """
        A = kernel_matrix.copy() % 2
        b = target_state.copy() % 2
        n = A.shape[0]
        m = A.shape[1] if A.ndim > 1 else n

        if A.ndim == 1:
            A = A.reshape(-1, 1)

        # 高斯消元（模 2）
        aug = np.concatenate([A, b.reshape(-1, 1)], axis=1)
        rank = 0
        for col in range(m):
            # 找主元
            pivot = -1
            for row in range(rank, n):
                if aug[row, col] == 1:
                    pivot = row
                    break
            if pivot == -1:
                continue
            # 交换行
            aug[[rank, pivot]] = aug[[pivot, rank]]
            # 消元
            for row in range(n):
                if row != rank and aug[row, col] == 1:
                    aug[row] = (aug[row] + aug[rank]) % 2
            rank += 1

        # 检查一致性
        for row in range(rank, n):
            if aug[row, -1] == 1 and np.all(aug[row, :-1] == 0):
                return None  # 无解

        # 回代提取解
        x = np.zeros(m, dtype=int)
        for row in range(rank - 1, -1, -1):
            lead = np.where(aug[row, :-1] == 1)[0]
            if len(lead) == 0:
                continue
            first = lead[0]
            x[first] = aug[row, -1]
            for j in lead[1:]:
                x[first] = (x[first] + x[j]) % 2

        return x

    def entropy_rate(self, binary_series: np.ndarray) -> float:
        """
        计算二值序列的 Shannon 熵率（每比特的不确定性）。
        H = -sum_{i,j} π_i P_{ij} log P_{ij}
        高熵率表示随机性强（可能是噪声），低熵率表示有规律性（可能是结构性异常）。
        """
        P = self.transition_matrix_analysis(binary_series)
        # 稳态分布
        eigvals, eigvecs = np.linalg.eig(P.T)
        stationary = np.real(eigvecs[:, np.argmin(np.abs(eigvals - 1.0))])
        stationary = stationary / stationary.sum()

        H = 0.0
        for i in range(2):
            for j in range(2):
                if P[i, j] > 1e-12:
                    H -= stationary[i] * P[i, j] * np.log2(P[i, j])
        return float(H)
