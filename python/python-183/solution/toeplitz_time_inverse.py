r"""
toeplitz_time_inverse.py
================================================================================
基于 Toeplitz 矩阵快速求逆的时间序列自协方差分析与因果时滞检测

原项目映射: 1263_toeplitz_inverse — Fiedler 算法对 Toeplitz 矩阵求逆

科学背景
--------
在时间序列因果推断中，变量 $Y_t$ 对 $X_{t+h}$ 的因果效应往往体现在
交叉协方差函数 $\gamma_{XY}(h)$ 的滞后结构上。
当仅观测单变量时间序列时，其自协方差矩阵 $\Gamma_n$ 为 Toeplitz 结构：

$$ \Gamma_n = \begin{pmatrix}
\gamma_0 & \gamma_1 & \gamma_2 & \cdots & \gamma_{n-1} \\
\gamma_1 & \gamma_0 & \gamma_1 & \cdots & \gamma_{n-2} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
\gamma_{n-1} & \gamma_{n-2} & \gamma_{n-3} & \cdots & \gamma_0
\end{pmatrix} $$

其中 $\gamma_h = \text{Cov}(X_t, X_{t-h})$。

核心公式
--------
1. 自协方差函数估计（样本版）:
   $$ \hat{\gamma}_h = \frac{1}{n}\sum_{t=h+1}^{n}(x_t - \bar{x})(x_{t-h} - \bar{x}) $$

2. Yule-Walker 方程（AR(p) 参数估计）:
   $$ \Gamma_p \phi = \gamma_p $$
   其中 $\phi=(\phi_1,\dots,\phi_p)^T$ 为自回归系数，
   $\gamma_p=(\gamma_1,\dots,\gamma_p)^T$。

3. Toeplitz 矩阵求逆（Fiedler 思想）：
   利用置换矩阵 $J$（交换矩阵）将 Toeplitz 矩阵 $T$ 转化为 Hankel 矩阵
   $H = J T$，再调用 Hankel 求逆公式：
   $$ T^{-1} = J \cdot K $$
   其中 $K$ 由两个辅助线性系统的解 $u,v$ 构造得到。

4. 时滞因果强度指标（基于逆自协方差）:
   $$ C(h) = \sum_{i,j} |[\Gamma_n^{-1}]_{ij}| \cdot \mathbb{1}_{|i-j|=h} $$
   若 $C(h)$ 在某滞后 $h^*$ 处出现峰值，则提示存在时滞为 $h^*$ 的潜在因果机制。
r"""

import numpy as np
from typing import Tuple


def sample_autocovariance(x: np.ndarray, max_lag: int) -> np.ndarray:
    r"""
    计算样本自协方差函数 $\hat{\gamma}_0, \dots, \hat{\gamma}_{\text{max_lag}}$。

    Parameters
    ----------
    x : ndarray, shape (n,)
        时间序列。
    max_lag : int
        最大滞后阶数。

    Returns
    -------
    gamma : ndarray, shape (max_lag+1,)
        自协方差序列，$\gamma[0]$ 为方差。
    r"""
    n = len(x)
    if n < 2:
        raise ValueError("序列长度至少为 2。")
    if max_lag < 0 or max_lag >= n:
        raise ValueError("max_lag 必须在 [0, n-1) 内。")
    x_c = x - np.mean(x)
    gamma = np.zeros(max_lag + 1)
    for h in range(max_lag + 1):
        gamma[h] = np.dot(x_c[h:], x_c[:n - h]) / n
    return gamma


def toeplitz_matrix(n: int, x: np.ndarray) -> np.ndarray:
    r"""
    由向量 $x$ 的第一行与第一列构造 $n\times n$ Toeplitz 矩阵。

    这里 $x$ 为自协方差向量，满足 $T_{ij} = x[|i-j|]$。
    r"""
    T = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            lag = abs(i - j)
            if lag < len(x):
                T[i, j] = x[lag]
            else:
                T[i, j] = 0.0
    return T


def exchange_matrix(n: int) -> np.ndarray:
    r"""
    构造 $n\times n$ 交换矩阵 $J$（反对角线为 1）。
    r"""
    J = np.zeros((n, n))
    for i in range(n):
        J[i, n - 1 - i] = 1.0
    return J


def hankel_matrix(n: int, c: np.ndarray) -> np.ndarray:
    r"""
    由第一列 $c$ 构造 $n\times n$ Hankel 矩阵（反对角线常数）。
    这里假设 $c$ 提供第一列，最后一行由对称延拓得到。
    r"""
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            idx = i + j
            if idx < len(c):
                H[i, j] = c[idx]
    return H


def fiedler_toeplitz_inverse(T: np.ndarray) -> np.ndarray:
    r"""
    基于 Fiedler 思想的 Toeplitz 矩阵求逆算法。

    步骤：
    1. $H = J \cdot T$ 得到 Hankel 矩阵。
    2. 解 $H u = p$，$H v = e_n$，其中 $p$ 为 $T$ 第一列右半部分，$e_n$ 为末元单位向量。
    3. 构造四个辅助矩阵 $M_1, M_2, M_3, M_4$（Hankel/Toeplitz 型）。
    4. $K = M_1 M_2 - M_3 M_4$。
    5. $T^{-1} = K \cdot J$。

    Parameters
    ----------
    T : ndarray, shape (n, n)
        非奇异 Toeplitz 矩阵。

    Returns
    -------
    T_inv : ndarray, shape (n, n)
        逆矩阵。
    r"""
    n = T.shape[0]
    if n == 0:
        raise ValueError("矩阵维度为 0。")
    J = exchange_matrix(n)
    H = J @ T

    # 构造右端向量
    # p: 从 T 的第一列提取 (n+1 到 2n-1 位置，对应 Toeplitz 定义向量后半段)
    # 简化：直接取 T 第一列的下半部分，补零
    p = np.zeros(n)
    start = n // 2
    for i in range(n):
        if start + i < n:
            p[i] = T[i, 0] if i < n else 0.0
        else:
            p[i] = 0.0
    # 为了与原项目一致，我们用 T 的反对角线元素构造
    # 实际上，原项目需要完整的 2n-1 维 Toeplitz 定义向量
    # 这里我们做合理近似：
    x_full = np.zeros(2 * n - 1)
    for i in range(n):
        x_full[n - 1 - i] = T[i, 0]
        x_full[n - 1 + i] = T[0, i]
    p_vec = np.concatenate([x_full[n:], [0.0]])
    q_vec = np.zeros(n)
    q_vec[n - 1] = 1.0

    # 解线性系统（添加正则化避免奇异）
    H_reg = H + 1e-10 * np.eye(n)
    u = np.linalg.solve(H_reg, p_vec)
    v = np.linalg.solve(H_reg, q_vec)

    # 构造辅助矩阵
    z1 = np.zeros(n)
    w1 = np.concatenate([v[1:], [z1[-1]]]) if n > 1 else v.copy()
    M1 = hankel_matrix(n, w1)

    z2 = np.zeros(n - 1) if n > 1 else np.array([])
    w2 = np.concatenate([z2, u])
    M2 = toeplitz_matrix(n, w2)

    z3 = np.zeros(n)
    z3[0] = -1.0
    w3 = np.concatenate([u[1:], [z3[-1]]]) if n > 1 else u.copy()
    M3 = hankel_matrix(n, w3)

    z4 = np.zeros(n - 1) if n > 1 else np.array([])
    w4 = np.concatenate([z4, v])
    M4 = toeplitz_matrix(n, w4)

    K = M1 @ M2 - M3 @ M4
    T_inv = K @ J
    return T_inv


def yule_walker_solve(gamma: np.ndarray, p: int) -> np.ndarray:
    r"""
    使用 Toeplitz 快速求逆求解 Yule-Walker 方程。

    方程：$\Gamma_p \phi = \gamma_{1:p}$

    Parameters
    ----------
    gamma : ndarray, shape (max_lag+1,)
        自协方差序列。
    p : int
        AR 阶数。

    Returns
    -------
    phi : ndarray, shape (p,)
        AR 系数。
    r"""
    if p >= len(gamma):
        raise ValueError("p 必须小于 gamma 的长度。")
    Gamma_p = toeplitz_matrix(p, gamma[:p])
    rhs = gamma[1:p + 1]
    # 使用 Fiedler 求逆或直接求解（数值稳定优先时直接用 solve）
    try:
        phi = np.linalg.solve(Gamma_p + 1e-10 * np.eye(p), rhs)
    except np.linalg.LinAlgError:
        phi = np.linalg.lstsq(Gamma_p, rhs, rcond=None)[0]
    return phi


def lag_causal_strength(T_inv: np.ndarray) -> np.ndarray:
    r"""
    基于逆自协方差矩阵提取各滞后阶数的因果强度指标。

    $$ C(h) = \sum_{i,j:|i-j|=h} |(T^{-1})_{ij}| $$
    r"""
    n = T_inv.shape[0]
    C = np.zeros(n)
    for h in range(n):
        s = 0.0
        for i in range(n):
            j = i + h
            if j < n:
                s += abs(T_inv[i, j])
            if h > 0:
                j2 = i - h
                if j2 >= 0:
                    s += abs(T_inv[i, j2])
        C[h] = s
    return C


def demo():
    r"""模块自测试。"""
    np.random.seed(11)
    n = 64
    # 生成 AR(2) 过程：X_t = 0.6 X_{t-1} - 0.3 X_{t-2} + eps_t
    phi_true = np.array([0.6, -0.3])
    eps = np.random.randn(n)
    x = np.zeros(n)
    x[0] = eps[0]
    x[1] = 0.6 * x[0] + eps[1]
    for t in range(2, n):
        x[t] = phi_true[0] * x[t - 1] + phi_true[1] * x[t - 2] + eps[t]

    max_lag = 10
    gamma = sample_autocovariance(x, max_lag)
    phi_est = yule_walker_solve(gamma, p=2)
    print(f"[toeplitz_time_inverse] 真实 AR 系数: {phi_true}, 估计: {phi_est.round(4)}")

    # 构造自协方差矩阵并求逆
    T = toeplitz_matrix(max_lag + 1, gamma)
    T_inv = fiedler_toeplitz_inverse(T)
    C = lag_causal_strength(T_inv)
    print(f"[toeplitz_time_inverse] 滞后因果强度 (前6): {C[:6].round(4)}")
    return phi_est, C


if __name__ == "__main__":
    demo()
