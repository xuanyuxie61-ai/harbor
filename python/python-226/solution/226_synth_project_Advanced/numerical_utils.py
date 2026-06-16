"""
数值工具与正则化模块
==================
实现对比度增强正则化、特殊函数计算、数值稳定性工具等。

核心功能：
1. 局部对比度增强（锐化/模糊）：
   p_new = s × p + (1-s) × mean(neighbors)
   用作参数反演的正则化

2. 特殊数学函数：
   - Gamma函数
   - 贝塞尔函数
   - 误差函数

3. 数值稳定性工具：
   - log-sum-exp技巧
   - 条件数估计
   - 矩阵病态诊断

数据来源：
- 574_image_contrast: 局部对比度增强算法
"""

import numpy as np
from typing import Tuple, Optional
from scipy import special


def local_contrast_enhancement(data: np.ndarray, sharpness: float = 1.5,
                                boundary_mode: str = 'nearest') -> np.ndarray:
    """
    局部对比度增强（锐化/平滑）。

    算法：
    对每个点计算邻域均值，然后混合：
    p_new = s × p + (1-s) × mean(neighbors)

    当 s > 1 时增强对比度（锐化）
    当 s < 1 时降低对比度（平滑）
    当 s = 1 时无变化

    在参数反演中的应用：
    对参数估计进行空间正则化，抑制噪声引起的剧烈振荡。

    参数：
        data: 输入数据 (可以是1D/2D/3D)
        sharpness: 锐度参数 s
        boundary_mode: 边界处理模式 ('nearest', 'wrap', 'zero')

    返回：
        enhanced: 增强后的数据
    """
    if data.ndim == 1:
        return _contrast_1d(data, sharpness, boundary_mode)
    elif data.ndim == 2:
        return _contrast_2d(data, sharpness, boundary_mode)
    else:
        # 对高维数据逐轴处理
        result = data.copy()
        for axis in range(data.ndim):
            result = np.apply_along_axis(
                lambda x: _contrast_1d(x, sharpness, boundary_mode),
                axis, result
            )
        return result


def _contrast_1d(data: np.ndarray, s: float, mode: str) -> np.ndarray:
    """1D对比度增强"""
    N = len(data)
    if N < 3:
        return data.copy()

    enhanced = np.zeros_like(data, dtype=float)

    for i in range(N):
        # 获取邻居
        if mode == 'wrap':
            left = data[(i - 1) % N]
            right = data[(i + 1) % N]
        elif mode == 'zero':
            left = data[i - 1] if i > 0 else 0.0
            right = data[i + 1] if i < N - 1 else 0.0
        else:  # nearest
            left = data[max(0, i - 1)]
            right = data[min(N - 1, i + 1)]

        neighbor_mean = (left + right) / 2.0

        # 混合
        enhanced[i] = s * data[i] + (1 - s) * neighbor_mean

    return enhanced


def _contrast_2d(data: np.ndarray, s: float, mode: str) -> np.ndarray:
    """2D对比度增强"""
    M, N = data.shape
    enhanced = np.zeros_like(data, dtype=float)

    for i in range(M):
        for j in range(N):
            # 收集8个邻居
            neighbors = []
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue

                    if mode == 'wrap':
                        ni = (i + di) % M
                        nj = (j + dj) % N
                        neighbors.append(data[ni, nj])
                    elif mode == 'zero':
                        ni, nj = i + di, j + dj
                        if 0 <= ni < M and 0 <= nj < N:
                            neighbors.append(data[ni, nj])
                        # 否则不加（相当于0）
                    else:  # nearest
                        ni = np.clip(i + di, 0, M - 1)
                        nj = np.clip(j + dj, 0, N - 1)
                        neighbors.append(data[ni, nj])

            if neighbors:
                neighbor_mean = np.mean(neighbors)
                enhanced[i, j] = s * data[i, j] + (1 - s) * neighbor_mean
            else:
                enhanced[i, j] = data[i, j]

    return enhanced


def log_sum_exp(x: np.ndarray, axis: int = None) -> float:
    """
    数值稳定的 log-sum-exp 计算。

    log(Σ exp(x_i)) = max(x) + log(Σ exp(x_i - max(x)))

    防止溢出和下溢。

    参数：
        x: 输入数组
        axis: 求和轴

    返回：
        result: log-sum-exp值
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    result = x_max + np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True))

    if axis is not None:
        result = result.squeeze(axis=axis)

    # 如果输入是1D且没有指定axis，返回标量
    if axis is None and x.ndim == 1:
        return float(result.item())

    return result


def condition_number(A: np.ndarray) -> float:
    """
    计算矩阵条件数。

    κ(A) = ||A|| × ||A⁻¹|| = σ_max / σ_min

    参数：
        A: 输入矩阵

    返回：
        kappa: 条件数
    """
    try:
        return np.linalg.cond(A)
    except np.linalg.LinAlgError:
        return np.inf


def regularize_matrix(A: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    矩阵正则化（Tikhonov正则化）。

    A_reg = A + εI

    用于改善病态矩阵的条件数。

    参数：
        A: 输入矩阵
        epsilon: 正则化参数

    返回：
        A_reg: 正则化后的矩阵
    """
    return A + epsilon * np.eye(A.shape[0])


def compute_gamma_function(z: complex) -> complex:
    """
    计算复Gamma函数。

    Γ(z) = ∫₀^∞ t^{z-1} e^{-t} dt

    使用scipy的实现。

    参数：
        z: 复数参数

    返回：
        Gamma(z): Gamma函数值
    """
    return special.gamma(z)


def compute_bessel_j(n: int, x: float) -> float:
    """
    计算第一类贝塞尔函数 J_n(x)。

    微分方程：
    x²y'' + xy' + (x² - n²)y = 0

    级数展开：
    J_n(x) = Σ_{k=0}^∞ (-1)^k (x/2)^{n+2k} / (k! Γ(n+k+1))

    参数：
        n: 阶数
        x: 自变量

    返回：
        J_n(x): 贝塞尔函数值
    """
    return special.jv(n, x)


def compute_error_function(x: float) -> float:
    """
    计算误差函数 erf(x)。

    erf(x) = (2/√π) ∫₀^x exp(-t²) dt

    参数：
        x: 自变量

    返回：
        erf(x): 误差函数值
    """
    return special.erf(x)


def compute_legendre_polynomial(n: int, x: float) -> float:
    """
    计算Legendre多项式 P_n(x)。

    递推关系：
    (n+1)P_{n+1}(x) = (2n+1)x P_n(x) - n P_{n-1}(x)

    参数：
        n: 阶数
        x: 自变量 [-1, 1]

    返回：
        P_n(x): Legendre多项式值
    """
    return special.legendre(n)(x)


def compute_spherical_harmonic(l: int, m: int, theta: float, phi: float) -> complex:
    """
    计算球谐函数 Y_l^m(θ, φ)。

    Y_l^m(θ,φ) = N_l^m P_l^m(cos θ) e^{imφ}
    N_l^m = √[(2l+1)/(4π) × (l-m)!/(l+m)!]

    参数：
        l: 角量子数
        m: 磁量子数
        theta: 极角
        phi: 方位角

    返回：
        Y_l^m: 球谐函数值
    """
    return special.sph_harm(m, l, phi, theta)


def numerical_gradient(f: callable, x: np.ndarray, h: float = 1e-7) -> np.ndarray:
    """
    中心差分数值梯度。

    ∂f/∂x_i ≈ [f(x + h e_i) - f(x - h e_i)] / (2h)

    参数：
        f: 目标函数
        x: 求梯度点
        h: 差分步长

    返回：
        grad: 梯度向量
    """
    n = len(x)
    grad = np.zeros(n)

    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h

        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)

    return grad


def numerical_hessian(f: callable, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """
    中心差分数值Hessian矩阵。

    ∂²f/∂x_i∂x_j ≈ [f(x+h_i+h_j) - f(x+h_i-h_j) - f(x-h_i+h_j) + f(x-h_i-h_j)] / (4h²)

    参数：
        f: 目标函数
        x: 求Hessian点
        h: 差分步长

    返回：
        H: Hessian矩阵
    """
    n = len(x)
    H = np.zeros((n, n))

    f0 = f(x)

    for i in range(n):
        for j in range(i, n):
            x_pp = x.copy()
            x_pm = x.copy()
            x_mp = x.copy()
            x_mm = x.copy()

            x_pp[i] += h; x_pp[j] += h
            x_pm[i] += h; x_pm[j] -= h
            x_mp[i] -= h; x_mp[j] += h
            x_mm[i] -= h; x_mm[j] -= h

            H[i, j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * h**2)
            H[j, i] = H[i, j]

    return H


def safe_divide(a: np.ndarray, b: np.ndarray, epsilon: float = 1e-15) -> np.ndarray:
    """
    安全除法，防止除零。

    参数：
        a, b: 输入数组
        epsilon: 小量

    返回：
        result: a / max(b, epsilon)
    """
    return a / np.maximum(np.abs(b), epsilon) * np.sign(b + 1e-30)


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    数值稳定的softmax函数。

    softmax(x_i) = exp(x_i/T) / Σ_j exp(x_j/T)

    参数：
        x: 输入数组
        temperature: 温度参数

    返回：
        probs: 概率分布
    """
    x_scaled = x / temperature
    x_max = np.max(x_scaled)
    exp_x = np.exp(x_scaled - x_max)
    return exp_x / np.sum(exp_x)


def kullback_leibler_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    KL散度。

    D_KL(P||Q) = Σ_i p_i log(p_i/q_i)

    参数：
        p, q: 概率分布

    返回：
        kl: KL散度
    """
    # 防止log(0)
    p_safe = np.maximum(p, 1e-30)
    q_safe = np.maximum(q, 1e-30)

    return np.sum(p_safe * np.log(p_safe / q_safe))


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    JS散度（对称化的KL散度）。

    D_JS(P||Q) = ½ D_KL(P||M) + ½ D_KL(Q||M)
    其中 M = ½(P + Q)

    参数：
        p, q: 概率分布

    返回：
        js: JS散度
    """
    m = 0.5 * (p + q)
    return 0.5 * kullback_leibler_divergence(p, m) + 0.5 * kullback_leibler_divergence(q, m)
