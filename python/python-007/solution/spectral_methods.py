"""
谱方法模块
整合自：895_polynomial_multiply（多项式乘法）

在吸积盘模拟中，谱方法用于角向（φ方向）的高精度微分。
采用切比雪夫/勒让德基函数的乘积运算实现谱卷积。
"""
import numpy as np


def polynomial_multiply(p, q):
    """
    两个多项式的系数向量相乘（离散卷积）。

    给定多项式:
        P(x) = p[0] + p[1]x + ... + p[n-1]x^(n-1)
        Q(x) = q[0] + q[1]x + ... + q[m-1]x^(m-1)

    乘积 R(x) = P(x)·Q(x) 的系数由 Cauchy 积给出:
        r[k] = Σ_{i=0}^{k} p[i] · q[k-i]

    参数:
        p, q: 系数向量（从低次到高次）

    返回:
        r: 乘积多项式的系数向量
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    # 去除尾部零，确定真实次数
    pn = len(p)
    qn = len(q)
    while pn > 1 and abs(p[pn - 1]) < 1e-15:
        pn -= 1
    while qn > 1 and abs(q[qn - 1]) < 1e-15:
        qn -= 1

    if pn == 0 or qn == 0:
        return np.array([0.0])

    p = p[:pn]
    q = q[:qn]

    # 离散卷积
    r = np.zeros(pn + qn - 1, dtype=np.float64)
    for i in range(pn):
        for j in range(qn):
            r[i + j] += p[i] * q[j]

    return r


def chebyshev_polynomial(n):
    """
    构造第 n 阶切比雪夫多项式 T_n(x) 的系数。
    递推关系:
        T_0(x) = 1
        T_1(x) = x
        T_{n+1}(x) = 2x·T_n(x) - T_{n-1}(x)

    参数:
        n: 阶数

    返回:
        系数向量（从低次到高次）
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return np.array([1.0])
    if n == 1:
        return np.array([0.0, 1.0])

    T_prev2 = np.array([1.0])       # T_0
    T_prev1 = np.array([0.0, 1.0])  # T_1

    for k in range(1, n):
        # 2x * T_prev1
        T_cur = np.zeros(len(T_prev1) + 1, dtype=np.float64)
        T_cur[1:] += 2.0 * T_prev1
        # - T_prev2
        T_cur[:len(T_prev2)] -= T_prev2
        T_prev2, T_prev1 = T_prev1, T_cur

    return T_prev1


def legendre_polynomial(n):
    """
    构造第 n 阶勒让德多项式 P_n(x) 的系数。
    递推关系（Bonnet 公式）:
        (n+1) P_{n+1}(x) = (2n+1) x P_n(x) - n P_{n-1}(x)

    参数:
        n: 阶数

    返回:
        系数向量
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return np.array([1.0])
    if n == 1:
        return np.array([0.0, 1.0])

    P_prev2 = np.array([1.0])
    P_prev1 = np.array([0.0, 1.0])

    for k in range(1, n):
        # (2k+1) * x * P_prev1
        temp = np.zeros(len(P_prev1) + 1, dtype=np.float64)
        temp[1:] = (2 * k + 1) * P_prev1
        # - k * P_prev2
        temp[:len(P_prev2)] -= k * P_prev2
        # divide by (k+1)
        P_cur = temp / (k + 1)
        P_prev2, P_prev1 = P_prev1, P_cur

    return P_prev1


def spectral_differentiation_matrix(N):
    """
    构造切比雪夫谱微分矩阵 D，使得 f'(x) ≈ D·f(x)。

    在吸积盘的角向方向，使用谱微分可高精度计算 ∂/∂φ。
    切比雪夫-高斯-洛巴托节点:
        x_j = cos(πj/N), j = 0, ..., N

    微分矩阵元素（Fornberg 公式）:
        D_{ij} = (c_i / c_j) · (-1)^(i+j) / (x_i - x_j),   i ≠ j
        D_{ii} = -x_i / (2(1-x_i²)),                        0 < i < N
        D_{00} = (2N² + 1) / 6
        D_{NN} = -(2N² + 1) / 6

    其中 c_0 = c_N = 2, c_j = 1 (其他)。

    参数:
        N: 节点数减一（即使用 N+1 个节点）

    返回:
        (N+1, N+1) 微分矩阵
    """
    if N < 1:
        raise ValueError("N must be at least 1")

    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.ones(N + 1, dtype=np.float64)
    c[0] = 2.0
    c[N] = 2.0

    D = np.zeros((N + 1, N + 1), dtype=np.float64)

    for i in range(N + 1):
        for j in range(N + 1):
            if i != j:
                D[i, j] = (c[i] / c[j]) * ((-1) ** (i + j)) / (x[i] - x[j])
            else:
                if i == 0:
                    D[i, i] = (2.0 * N * N + 1.0) / 6.0
                elif i == N:
                    D[i, i] = -(2.0 * N * N + 1.0) / 6.0
                else:
                    D[i, i] = -x[i] / (2.0 * (1.0 - x[i] ** 2))

    return D, x


def apply_angular_spectral_derivative(f_phi, N_modes=16):
    """
    对周期角向函数 f(φ) 应用谱微分。
    采用傅里叶-切比雪夫混合方法：
        ∂f/∂φ ≈ Σ_{k=-N}^{N} i·k·f̂_k · e^{ikφ}

    参数:
        f_phi: 在等间距角向网格上的函数值
        N_modes: 保留的傅里叶模数

    返回:
        df_dphi: 角向导数
    """
    f_phi = np.asarray(f_phi, dtype=np.complex128)
    n = len(f_phi)
    if n < 2:
        return np.zeros_like(f_phi)

    # FFT
    f_hat = np.fft.fft(f_phi)

    # 构造波数
    k = np.fft.fftfreq(n, d=1.0 / n) * (2.0 * np.pi)

    # 限制模数
    k_max = N_modes
    mask = np.abs(k) > k_max * (2.0 * np.pi / n) * (n // 2)
    # 更精确的截断
    idx = np.arange(n)
    idx = np.where(idx > n // 2, idx - n, idx)
    mask = np.abs(idx) > N_modes
    f_hat[mask] = 0.0

    # 频域微分: i·k
    df_hat = 1j * k * f_hat
    df_dphi = np.fft.ifft(df_hat).real

    return df_dphi
