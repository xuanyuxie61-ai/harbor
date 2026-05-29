"""
spectral_expansion.py
基于种子项目 666_legendre_shifted_polynomial 的移位 Legendre 多项式展开

在 r 过程核合成中，温度依赖的反应率可以表示为：
    R(T) = Σ_{k=0}^{K} c_k · P̃_k(τ)
其中 τ = (T - T_min)/(T_max - T_min) ∈ [0,1] 为归一化温度，
P̃_k(τ) 为 [0,1] 区间上的移位 Legendre 多项式，满足正交性：
    ∫_0^1 P̃_m(τ) P̃_n(τ) dτ = δ_{mn} / (2n+1)

三项递推关系：
    P̃_0(τ) = 1
    P̃_1(τ) = 2τ - 1
    (n+1) P̃_{n+1}(τ) = (2n+1)(2τ-1) P̃_n(τ) - n P̃_{n-1}(τ)
"""

import numpy as np


def shifted_legendre_polynomial_value(m, n, x):
    """
    计算移位 Legendre 多项式 P̃_0(x) 到 P̃_n(x) 在 m 个点上的值。

    参数:
        m : int, 计算点的数量
        n : int, 最高阶数
        x : array-like, shape (m,), 点坐标，必须在 [0,1] 区间内

    返回:
        v : ndarray, shape (m, n+1), v[i,j] = P̃_j(x[i])
    """
    x = np.asarray(x, dtype=float)
    if x.shape[0] != m:
        raise ValueError("x 的长度必须与 m 一致")
    if np.any(x < 0.0) or np.any(x > 1.0):
        # 鲁棒处理：将越界值截断到 [0,1]
        x = np.clip(x, 0.0, 1.0)

    v = np.zeros((m, n + 1), dtype=float)
    v[:, 0] = 1.0
    if n >= 1:
        v[:, 1] = 2.0 * x - 1.0
    for j in range(1, n):
        v[:, j + 1] = (
            (2 * j + 1) * (2.0 * x - 1.0) * v[:, j]
            - j * v[:, j - 1]
        ) / (j + 1)
    return v


def gauss_legendre_shifted_nodes_weights(n):
    """
    计算 [0,1] 区间上的 Gauss-Legendre 积分节点和权重。
    通过标准 [-1,1] 节点 xi 变换：x_i = (xi + 1)/2，w_i = w̃_i / 2。

    参数:
        n : int, 节点数

    返回:
        x, w : ndarray, 节点和权重
    """
    xi, wi = np.polynomial.legendre.leggauss(n)
    x = 0.5 * (xi + 1.0)
    w = 0.5 * wi
    return x, w


def spectral_expand_reaction_rate(temperatures, rates, degree=8):
    """
    将反应率 R(T) 用移位 Legendre 多项式展开到指定阶数。

    利用正交性，展开系数为：
        c_k = (2k+1) ∫_0^1 R(T(τ)) P̃_k(τ) dτ

    参数:
        temperatures : ndarray, 温度数组 (K)
        rates : ndarray, 对应反应率 (cm^3/mol/s)
        degree : int, 展开阶数

    返回:
        coeffs : ndarray, 展开系数 c_0, ..., c_degree
        t_min, t_max : float, 温度归一化参数
    """
    temperatures = np.asarray(temperatures, dtype=float)
    rates = np.asarray(rates, dtype=float)
    t_min, t_max = np.min(temperatures), np.max(temperatures)
    if t_max <= t_min:
        raise ValueError("温度范围必须为正")
    # 归一化到 [0,1]
    tau = (temperatures - t_min) / (t_max - t_min)
    tau = np.clip(tau, 0.0, 1.0)

    # Gauss-Legendre 数值积分
    x_quad, w_quad = gauss_legendre_shifted_nodes_weights(degree + 4)
    v_quad = shifted_legendre_polynomial_value(len(x_quad), degree, x_quad)

    # 插值反应率到积分节点
    rates_quad = np.interp(x_quad, tau, rates)

    coeffs = np.zeros(degree + 1, dtype=float)
    for k in range(degree + 1):
        # c_k = (2k+1) * Σ w_i * R(τ_i) * P̃_k(τ_i)
        coeffs[k] = (2 * k + 1) * np.sum(w_quad * rates_quad * v_quad[:, k])
    return coeffs, t_min, t_max


def spectral_evaluate_reaction_rate(tau, coeffs, t_min, t_max):
    """
    用展开系数重构反应率。

    参数:
        tau : ndarray, 归一化温度 [0,1]
        coeffs : ndarray, 展开系数
        t_min, t_max : float, 归一化参数

    返回:
        rates : ndarray, 重构的反应率
    """
    tau = np.asarray(tau, dtype=float)
    tau = np.clip(tau, 0.0, 1.0)
    degree = len(coeffs) - 1
    m = tau.shape[0]
    v = shifted_legendre_polynomial_value(m, degree, tau)
    rates = v @ coeffs
    return rates


def test_spectral_expansion():
    """自包含测试"""
    T = np.linspace(1e9, 10e9, 200)
    # 模拟一个温度依赖的反应率：R(T) ∝ T^ν * exp(-Q/kT)
    kB = 1.380649e-16  # erg/K
    Q = 2.5e6  # erg/mol
    nu = 0.5
    R = (T ** nu) * np.exp(-Q / (kB * T))
    coeffs, t_min, t_max = spectral_expand_reaction_rate(T, R, degree=10)
    tau_test = np.linspace(0, 1, 100)
    R_recon = spectral_evaluate_reaction_rate(tau_test, coeffs, t_min, t_max)
    T_test = tau_test * (t_max - t_min) + t_min
    R_exact = (T_test ** nu) * np.exp(-Q / (kB * T_test))
    rel_err = np.abs(R_recon - R_exact) / (np.abs(R_exact) + 1e-30)
    print(f"[spectral_expansion] Max relative reconstruction error: {np.max(rel_err):.3e}")
    assert np.max(rel_err) < 0.05, "Spectral expansion accuracy too low"


if __name__ == "__main__":
    test_spectral_expansion()
