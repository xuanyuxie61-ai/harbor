"""
covariance_pfaffian.py

基于 1280_toms923 核心算法的 Pfaffian 协方差分析模块。

原项目 pfaffian_LTL 使用 Parlett-Reid 算法计算斜对称矩阵的 Pfaffian。

Pfaffian 在统计物理和量子场论中至关重要，定义为：
    pf(A)^2 = det(A)
对于斜对称矩阵 A（A^T = -A）。

在本气候归因框架中，我们将 Pfaffian 用于：
1. 高斯随机场（GRF）的配分函数计算
2. 基于 Pfaffian 点过程的极端事件空间分布建模
3. 协方差矩阵特征结构的快速判定（det > 0 等价于 pf ≠ 0）

核心公式：
- 斜对称矩阵 Pfaffian（Parlett-Reid LTL 分解）：
    A = P^T L T L^T P
    pf(A) = (-1)^{swaps} * Π_{k=1}^{N/2} T_{2k-1, 2k}
- 对于 N×N 斜对称矩阵（N 偶数）：
    pf(A) = 1/(2^{N/2} (N/2)!) * Σ_{σ∈S_N} sgn(σ) Π_{i=1}^{N/2} A_{σ(2i-1), σ(2i)}
- 高斯随机场配分函数：
    Z = pf(K)^{1/2}
    其中 K 为斜对称协方差核矩阵。
"""

import numpy as np


def pfaffian_LTL(A_in):
    """
    使用 Parlett-Reid LTL 算法计算斜对称矩阵的 Pfaffian。

    基于 1280_toms923 的 pfaffian_LTL 实现。

    Parameters
    ----------
    A_in : ndarray, shape (N, N)
        斜对称矩阵（A^T = -A）。

    Returns
    -------
    pf : float
        Pfaffian 值。
    """
    A = np.array(A_in, dtype=np.float64, copy=True)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("输入必须是方阵")
    N = A.shape[0]

    # 检查斜对称性
    if np.linalg.norm(A + A.T) > 1e-12 * N:
        raise ValueError("输入矩阵不是斜对称的")

    if N % 2 == 1:
        return 0.0

    pf = 1.0
    for k in range(0, N - 1, 2):
        # 在 A[k+1:N, k] 中寻找最大元并主元交换到 A[k+1, k]
        sub = A[k + 1:N, k]
        kp_rel = np.argmax(np.abs(sub))
        kp = kp_rel + k + 1

        if kp != k + 1:
            # 交换行 k+1 和 kp
            temp = A[k + 1, k:N].copy()
            A[k + 1, k:N] = A[kp, k:N]
            A[kp, k:N] = temp
            # 交换列 k+1 和 kp
            temp = A[k:N, k + 1].copy()
            A[k:N, k + 1] = A[k:N, kp]
            A[k:N, kp] = temp
            pf = -pf

        pf *= A[k, k + 1]

        if A[k + 1, k] != 0.0:
            tau = A[k + 2:N, k] / A[k + 1, k]
            # 更新 A[k+2:, k+2:]
            if k + 2 < N:
                A[k + 2:N, k + 2:N] += np.outer(tau, A[k + 2:N, k + 1]) \
                                       - np.outer(A[k + 2:N, k + 1], tau)

    pf *= A[N - 1, N - 2] if N >= 2 else 1.0
    return float(pf)


def build_skew_covariance_from_kernel(nodes, kernel_func):
    """
    从核函数构建斜对称协方差矩阵。

    构造反对称核：K(x,y) = f(x,y) - f(y,x)
    用于建模具有手征对称性的气候场。
    """
    n = nodes.shape[1]
    K = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            val = kernel_func(nodes[:, i], nodes[:, j])
            K[i, j] = val
            K[j, i] = -val
    return K


def gaussian_random_field_log_partition(K):
    """
    计算高斯随机场配分函数的对数。

    对于斜对称协方差矩阵 K：
        Z = sqrt(pf(K))
        log Z = 0.5 * log(|pf(K)|)
    """
    pf = pfaffian_LTL(K)
    if abs(pf) < 1e-14:
        return -np.inf
    return 0.5 * np.log(abs(pf))


def extreme_event_pfaffian_correlation(nodes, correlation_length=1.0):
    """
    基于 Pfaffian 的极端事件空间相关模型。

    使用反对称指数核：
        K_{ij} = exp(-|x_i - x_j| / ξ) * sgn(x_i - x_j)

    该模型的 Pfaffian 给出了具有特定空间尺度的极端事件构型的概率权重。
    """
    def kernel(xi, xj):
        dx = np.linalg.norm(xi - xj)
        if dx < 1e-14:
            return 0.0
        s = np.sign(np.sum(xi - xj))
        if s == 0:
            s = 1.0
        return s * np.exp(-dx / correlation_length)

    K = build_skew_covariance_from_kernel(nodes, kernel)
    return K, pfaffian_LTL(K)


def test_pfaffian():
    # 测试：4x4 标准斜对称矩阵
    A = np.array([
        [0, 1, 0, 0],
        [-1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, -1, 0],
    ], dtype=np.float64)
    pf = pfaffian_LTL(A)
    assert abs(pf - 1.0) < 1e-10

    # 更大矩阵测试
    N = 6
    B = np.random.randn(N, N)
    B = B - B.T
    pf2 = pfaffian_LTL(B)
    det_val = np.linalg.det(B)
    assert abs(pf2 ** 2 - det_val) < 1e-8 * max(abs(det_val), 1.0)
    print("covariance_pfaffian 自测试通过")


if __name__ == "__main__":
    test_pfaffian()
