"""
cs_detector.py
==============
压缩感知稀疏检测与 L1 最小化重建模块

科学背景：
---------
压缩感知理论（Candès, Romberg, Tao; Donoho）指出：若信号 x \in \mathbb{R}^N
在某个基 \Psi 下是 s-稀疏的，即 \|c\|_0 = s \ll N，则可通过远少于奈奎斯特采样
定理要求的测量数 m = O(s \log(N/s)) 精确重建信号。

测量模型：
    y = \Phi x + \eta = \Phi \Psi c + \eta
其中：
    - \Phi \in \mathbb{R}^{m \times N}：测量矩阵（sensing matrix）
    - \Psi \in \mathbb{R}^{N \times N}：稀疏表示基
    - c \in \mathbb{R}^N：稀疏系数
    - \eta \in \mathbb{R}^m：测量噪声

重建问题转化为 Basis Pursuit Denoising（BPDN）：
    \min_c \frac{1}{2} \|\Phi \Psi c - y\|_2^2 + \lambda \|c\|_1

或约束形式（Basis Pursuit）：
    \min_c \|c\|_1 \quad \text{s.t.} \quad \|\Phi \Psi c - y\|_2 \leq \epsilon

本模块实现基于迭代软阈值算法（Iterative Soft Thresholding Algorithm, ISTA）
和快速 ISTA（FISTA）的求解器，替代原始 MATLAB 的 linprog 方法，更适合大规模问题。

来自项目 223_counterfeit_detection 的核心思想。
"""

import numpy as np
from typing import Optional, Tuple


def soft_thresholding(x: np.ndarray, lambda_: float) -> np.ndarray:
    """
    软阈值算子（proximal operator of L1 norm）。

    数学定义：
        S_\lambda(x) = \text{sign}(x) \cdot \max(|x| - \lambda, 0)

    这是 L1 范数次梯度的近端映射：
        \text{prox}_{\lambda \|\cdot\|_1}(x) = \arg\min_z \frac{1}{2}\|z - x\|_2^2 + \lambda \|z\|_1

    参数:
        x: 输入向量或数组
        lambda_: 阈值参数（必须非负）
    返回:
        软阈值后的结果
    """
    if lambda_ < 0:
        raise ValueError("阈值 lambda 必须非负")
    return np.sign(x) * np.maximum(np.abs(x) - lambda_, 0.0)


def ista_reconstruction(A: np.ndarray, y: np.ndarray,
                        lambda_: float,
                        max_iter: int = 1000,
                        tol: float = 1e-6,
                        x0: Optional[np.ndarray] = None) -> np.ndarray:
    """
    迭代软阈值算法（ISTA）求解 Basis Pursuit Denoising。

    算法迭代格式：
        x^{k+1} = S_{\lambda/L}\left(x^k - \frac{1}{L} A^T(A x^k - y)\right)
    其中 L = \|A^T A\|_2 为 Lipschitz 常数。

    收敛性：
        当步长 \alpha = 1/L 时，目标函数值以 O(1/k) 速率收敛。

    参数:
        A: 感知矩阵，形状为 (m, N)
        y: 测量向量，形状为 (m,)
        lambda_: L1 正则化参数
        max_iter: 最大迭代次数
        tol: 收敛容差
        x0: 初始估计（默认零向量）
    返回:
        重建的稀疏系数向量 x，形状为 (N,)
    """
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    m, N = A.shape

    if len(y) != m:
        raise ValueError(f"测量向量维度 {len(y)} 与矩阵行数 {m} 不匹配")

    # 计算 Lipschitz 常数上界
    L = np.linalg.norm(A.T @ A, 2)
    if L < 1e-14:
        raise ValueError("感知矩阵 A 的奇异值过小，问题病态")

    step = 1.0 / L

    if x0 is None:
        x = np.zeros(N, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).ravel().copy()
        if len(x) != N:
            raise ValueError("初始向量维度与 A 的列数不匹配")

    for k in range(max_iter):
        x_old = x.copy()
        # 梯度步
        grad = A.T @ (A @ x - y)
        # 软阈值
        x = soft_thresholding(x - step * grad, lambda_ * step)

        # 收敛判断
        if np.linalg.norm(x - x_old) < tol * max(1.0, np.linalg.norm(x_old)):
            break

    return x


def fista_reconstruction(A: np.ndarray, y: np.ndarray,
                         lambda_: float,
                         max_iter: int = 1000,
                         tol: float = 1e-6,
                         x0: Optional[np.ndarray] = None) -> np.ndarray:
    """
    快速迭代软阈值算法（FISTA）求解 BPDN。

    算法（Beck & Teboulle, 2009）：
        t_1 = 1, z^1 = x^0
        x^k = S_{\lambda/L}(z^k - \frac{1}{L} A^T(A z^k - y))
        t_{k+1} = \frac{1 + \sqrt{1 + 4 t_k^2}}{2}
        z^{k+1} = x^k + \frac{t_k - 1}{t_{k+1}}(x^k - x^{k-1})

    收敛速率：O(1/k^2)，优于 ISTA 的 O(1/k)。

    参数:
        A: 感知矩阵
        y: 测量向量
        lambda_: L1 正则化参数
        max_iter: 最大迭代次数
        tol: 收敛容差
        x0: 初始估计
    返回:
        重建的稀疏系数向量
    """
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    m, N = A.shape

    if len(y) != m:
        raise ValueError("测量向量维度与矩阵行数不匹配")

    # TODO [Hole_2]: 实现 FISTA 快速迭代软阈值算法
    # 科学知识点：
    #   FISTA (Beck & Teboulle, 2009) 求解 Basis Pursuit Denoising：
    #     min_c 0.5 * ||A c - y||_2^2 + lambda * ||c||_1
    #   算法步骤：
    #     1. 计算 Lipschitz 常数上界 L = ||A^T A||_2，步长 step = 1/L
    #     2. 初始化：t_1 = 1, z^1 = x^0 (或零向量)
    #     3. 迭代：
    #          x^k = S_{lambda/L}(z^k - (1/L) * A^T(A z^k - y))
    #          t_{k+1} = (1 + sqrt(1 + 4 t_k^2)) / 2
    #          z^{k+1} = x^k + ((t_k - 1) / t_{k+1}) * (x^k - x^{k-1})
    #     4. 收敛判据：||x^k - x^{k-1}|| < tol * max(1, ||x^{k-1}||)
    #   其中 S_{lambda} 为软阈值算子（已在 soft_thresholding 中实现）
    # ========================================
    # 请在下方实现完整的 FISTA 迭代逻辑，最终返回解向量 x
    # ========================================
    raise NotImplementedError("Hole_2: FISTA 算法待实现")


def orthogonal_matching_pursuit(A: np.ndarray, y: np.ndarray,
                                sparsity: int,
                                max_iter: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    正交匹配追踪（OMP）算法求解稀疏恢复问题。

    数学模型：
        \min \|x\|_0 \quad \text{s.t.} \quad A x = y

    算法步骤：
        1. 初始化残差 r = y，支持集 \Omega = \emptyset
        2. 迭代：
           a. 寻找与残差最相关的列：j = \arg\max_j |\langle a_j, r \rangle|
           b. 更新支持集：\Omega = \Omega \cup \{j\}
           c. 最小二乘：x_\Omega = \arg\min \|A_\Omega x - y\|_2
           d. 更新残差：r = y - A_\Omega x_\Omega
        3. 直到 |\Omega| = sparsity 或残差足够小

    参数:
        A: 感知矩阵，形状为 (m, N)
        y: 测量向量，形状为 (m,)
        sparsity: 目标稀疏度
        max_iter: 最大迭代次数（默认 sparsity）
    返回:
        (x, support): 重建向量和支持集索引
    """
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    m, N = A.shape

    if max_iter is None:
        max_iter = sparsity

    if sparsity <= 0 or sparsity > N:
        raise ValueError(f"稀疏度必须在 [1, {N}] 范围内")

    residual = y.copy()
    support = []
    x = np.zeros(N, dtype=float)

    for _ in range(max_iter):
        # 寻找最相关列
        correlations = np.abs(A.T @ residual)
        # 排除已选列
        for idx in support:
            correlations[idx] = -1.0

        j = np.argmax(correlations)
        if correlations[j] < 1e-14:
            break

        support.append(j)

        # 支持集上的最小二乘
        A_omega = A[:, support]
        try:
            x_omega = np.linalg.lstsq(A_omega, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            break

        # 更新解和残差
        x.fill(0.0)
        x[support] = x_omega
        residual = y - A_omega @ x_omega

        if np.linalg.norm(residual) < 1e-12 * np.linalg.norm(y):
            break

    return x, np.array(support, dtype=int)


def build_sensing_matrix_gaussian(m: int, N: int, normalize: bool = True) -> np.ndarray:
    """
    构造高斯随机测量矩阵。

    数学性质：
        对于元素独立服从 \mathcal{N}(0, 1/m) 的矩阵 \Phi，
        以高概率满足 (2s, \delta)-RIP，其中 m \geq C \delta^{-2} s \log(N/s)。

    参数:
        m: 测量数
        N: 信号维度
        normalize: 是否归一化列
    返回:
        高斯随机测量矩阵，形状为 (m, N)
    """
    if m <= 0 or N <= 0:
        raise ValueError("m 和 N 必须为正整数")

    Phi = np.random.randn(m, N) / np.sqrt(m)
    if normalize:
        col_norms = np.linalg.norm(Phi, axis=0)
        col_norms[col_norms == 0] = 1.0
        Phi = Phi / col_norms
    return Phi
