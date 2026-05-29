"""
jacobi_spectral.py
基于Jacobi多项式的谱方法模块（对应种子项目 607_jacobi_polynomial）

在光纤非线性脉冲传输中，Jacobi多项式被用作频域上的正交基函数，
用于对脉冲包络进行谱展开，以高精度计算色散算子作用。

核心公式：
  Jacobi多项式 P_n^{(α,β)}(x) 满足正交性：
    ∫_{-1}^{1} (1-x)^α (1+x)^β P_m^{(α,β)}(x) P_n^{(α,β)}(x) dx
      = δ_{mn} * 2^{α+β+1} / (2n+α+β+1) * Γ(n+α+1)Γ(n+β+1) / (n! Γ(n+α+β+1))

  利用Gauss-Jacobi求积规则：
    ∫_{-1}^{1} (1-x)^α (1+x)^β f(x) dx ≈ Σ_{i=1}^{N} w_i f(x_i)
  其中 x_i 为 P_N^{(α,β)}(x) 的零点，w_i 为对应权重。
"""

import numpy as np
from math import gamma
from scipy.special import gammaln


def jacobi_polynomial(m, n, alpha, beta, x):
    """
    计算前n+1阶Jacobi多项式在点x处的值。

    参数:
        m: int, 求值点数量
        n: int, 最高阶数
        alpha, beta: float, Jacobi参数, 必须 > -1
        x: ndarray shape (m,), 求值点

    返回:
        v: ndarray shape (m, n+1), 各阶多项式值

    递推公式:
      P_0 = 1
      P_1 = [(2+α+β)x + (α-β)] / 2
      P_n = [(2n+α+β-1)((α²-β²)+(2n+α+β)(2n+α+β-2)x) P_{n-1}
            - 2(n-1+α)(n-1+β)(2n+α+β) P_{n-2}] / [2n(n+α+β)(2n-2+α+β)]
    """
    if alpha <= -1.0 or beta <= -1.0:
        raise ValueError("jacobi_polynomial: alpha and beta must be > -1")
    if n < 0:
        return np.empty((m, 0))

    x = np.asarray(x, dtype=float)
    v = np.ones((m, n + 1), dtype=float)

    if n == 0:
        return v

    v[:, 1] = (1.0 + 0.5 * (alpha + beta)) * x + 0.5 * (alpha - beta)

    for i in range(2, n + 1):
        c1 = 2.0 * i * (i + alpha + beta) * (2.0 * i - 2.0 + alpha + beta)
        c2 = (2.0 * i - 1.0 + alpha + beta) * (2.0 * i + alpha + beta) * (2.0 * i - 2.0 + alpha + beta)
        c3 = (2.0 * i - 1.0 + alpha + beta) * (alpha + beta) * (alpha - beta)
        c4 = -2.0 * (i - 1.0 + alpha) * (i - 1.0 + beta) * (2.0 * i + alpha + beta)

        v[:, i] = ((c3 + c2 * x) * v[:, i - 1] + c4 * v[:, i - 2]) / c1

    return v


def jacobi_quadrature_rule(n, alpha, beta):
    """
    Gauss-Jacobi求积规则：返回节点x和权重w。

    通过构造对称三对角Jacobi矩阵并求其特征值/特征向量得到。
    节点x_i为矩阵特征值，权重w_i = μ_0 * (v_{i,1})^2，
    其中v_{i,1}为归一化特征向量的第一个分量。

    Jacobi矩阵元素:
      a_i = (β²-α²) / [(2i+α+β)(2i+α+β-2)]   (i >= 2)
      a_1 = (β-α)/(α+β+2)
      b_i = 4i(i+α)(i+β)(i+α+beta) / [(2i+α+β)²-1] / (2i+α+β)²  (i >= 1)
    """
    if n < 1:
        return np.array([]), np.array([])

    ab = alpha + beta
    abi = 2.0 + ab

    # 零阶矩 μ_0 = 2^{α+β+1} Γ(α+1)Γ(β+1) / Γ(α+β+2)
    zemu = (2.0 ** (ab + 1.0)) * gamma(alpha + 1.0) * gamma(beta + 1.0) / gamma(abi)

    # 构造Jacobi矩阵对角线和次对角线
    diag = np.zeros(n)
    offd = np.zeros(n - 1)

    diag[0] = (beta - alpha) / abi
    a2b2 = beta * beta - alpha * alpha

    for i in range(1, n):
        abi_val = 2.0 * (i + 1) + ab
        diag[i] = a2b2 / ((abi_val - 2.0) * abi_val)
        abi_sq = abi_val ** 2
        offd[i - 1] = np.sqrt(
            4.0 * (i + 1) * (i + 1 + alpha) * (i + 1 + beta) * (i + 1 + ab)
            / ((abi_sq - 1.0) * abi_sq)
        )

    # 使用隐式QL算法（简化版：用numpy的eigh）
    # 在科研代码中保持数值鲁棒性
    jacobi_mat = np.diag(diag) + np.diag(offd, k=1) + np.diag(offd, k=-1)
    eigvals, eigvecs = np.linalg.eigh(jacobi_mat)

    x = eigvals
    w = zemu * (eigvecs[0, :] ** 2)

    return x, w


def spectral_expand_pulse(t_grid, pulse, alpha_jac=-0.5, beta_jac=-0.5, n_modes=32):
    """
    将时域脉冲包络利用Jacobi多项式进行谱展开。

    由于光纤脉冲通常在有限时间窗口内，我们将时间映射到[-1,1]区间：
      τ = 2(t - t_c) / T_window

    展开式:
      A(t) ≈ Σ_{n=0}^{N-1} c_n P_n^{(α,β)}(τ)

    系数由正交性确定:
      c_n = 1/h_n * ∫_{-1}^{1} w(τ) A(τ) P_n(τ) dτ

    返回系数c_n和重构脉冲。
    """
    if pulse.size != t_grid.size:
        raise ValueError("spectral_expand_pulse: t_grid and pulse must have same size")
    if t_grid.size < 2:
        raise ValueError("spectral_expand_pulse: grid too small")

    t_min, t_max = np.min(t_grid), np.max(t_grid)
    if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
        raise ValueError("spectral_expand_pulse: invalid time grid")

    # 映射到[-1,1]
    tau = 2.0 * (t_grid - t_min) / (t_max - t_min) - 1.0
    tau = np.clip(tau, -1.0, 1.0)

    # Gauss-Jacobi求积节点（需要插值到均匀网格）
    x_q, w_q = jacobi_quadrature_rule(n_modes, alpha_jac, beta_jac)

    # 将求积节点映射回时间域并插值获取脉冲值
    t_q = t_min + (x_q + 1.0) * 0.5 * (t_max - t_min)
    pulse_q = np.interp(t_q, t_grid, np.real(pulse)) + 1j * np.interp(t_q, t_grid, np.imag(pulse))

    # 计算Jacobi多项式在求积节点处的值
    v = jacobi_polynomial(n_modes, n_modes - 1, alpha_jac, beta_jac, x_q)

    # 计算归一化常数 h_n (使用gammaln避免溢出和奇点)
    hn = np.zeros(n_modes)
    for n in range(n_modes):
        log_num = (alpha_jac + beta_jac + 1.0) * np.log(2.0) + gammaln(n + alpha_jac + 1.0) + gammaln(n + beta_jac + 1.0)
        denom_arg = n + alpha_jac + beta_jac + 1.0
        if abs(denom_arg) < 1e-14:
            if abs(alpha_jac + 0.5) < 1e-10 and abs(beta_jac + 0.5) < 1e-10:
                hn[n] = np.pi if n == 0 else np.pi / 2.0
            else:
                hn[n] = 1.0
            continue
        log_den = np.log(abs(2.0 * n + alpha_jac + beta_jac + 1.0)) + gammaln(n + 1.0) + gammaln(denom_arg)
        hn[n] = np.exp(log_num - log_den)

    # 计算展开系数
    coeffs = np.zeros(n_modes, dtype=complex)
    for n in range(n_modes):
        integrand = pulse_q * v[:, n]
        coeffs[n] = np.sum(w_q * integrand) / hn[n]

    # 在原始网格上重构
    v_orig = jacobi_polynomial(t_grid.size, n_modes - 1, alpha_jac, beta_jac, tau)
    reconstructed = v_orig @ coeffs

    return coeffs, reconstructed


def dispersion_operator_spectral(coeffs, alpha_jac, beta_jac, n_modes, beta2, beta3, L):
    """
    在Jacobi谱空间中施加色散算子。

    频域色散算子: D(ω) = i(β₂/2)ω² - i(β₃/6)ω³
    在有限时间窗口近似下，离散频谱算子等效于在谱系数上的微分矩阵作用。

    利用Jacobi多项式的微分性质:
      d/dx P_n^{(α,β)}(x) = (n+α+β+1)/2 * P_{n-1}^{(α+1,β+1)}(x) + ...

    此处采用简化但鲁棒的实现：通过重构-微分-再展开的方式施加色散。
    """
    if coeffs.size != n_modes:
        raise ValueError("dispersion_operator_spectral: coefficient size mismatch")

    # 为数值稳定性，使用低阶展开
    # 二阶色散等价于对时间域的二阶导数
    # 通过构造微分矩阵 D_{ij} = ∫ w(x) P_i(x) d²/dx² P_j(x) dx / h_i

    x_q, w_q = jacobi_quadrature_rule(n_modes + 2, alpha_jac, beta_jac)

    v0 = jacobi_polynomial(n_modes + 2, n_modes - 1, alpha_jac, beta_jac, x_q)
    # 一阶导数对应的Jacobi参数为 (α+1, β+1)
    v1 = jacobi_polynomial(n_modes + 2, n_modes - 1, alpha_jac + 1.0, beta_jac + 1.0, x_q)

    # Jacobi多项式导数关系:
    # d/dx P_n^{(α,β)}(x) = (n+α+β+1)/2 * P_{n-1}^{(α+1,β+1)}(x)
    # 这里仅利用一阶导数构造二阶导数的近似

    # 计算归一化常数
    hn = np.zeros(n_modes)
    for n in range(n_modes):
        # 使用对数gamma避免溢出，并处理奇点
        log_num = (alpha_jac + beta_jac + 1.0) * np.log(2.0) + gammaln(n + alpha_jac + 1.0) + gammaln(n + beta_jac + 1.0)
        # 处理分母中可能出现的 gamma(0) 或负整数
        denom_arg = n + alpha_jac + beta_jac + 1.0
        if abs(denom_arg) < 1e-14:
            # 极限情况: gamma(z)/z ~ gamma(z+1)/z^2 当 z->0
            # 直接利用 beta 函数关系
            # 对于 alpha+beta+1=0 (即 alpha=beta=-0.5), 使用已知公式
            if abs(alpha_jac + 0.5) < 1e-10 and abs(beta_jac + 0.5) < 1e-10:
                # Chebyshev 第一类多项式: h_0 = pi, h_n = pi/2 for n>=1
                hn[n] = np.pi if n == 0 else np.pi / 2.0
            else:
                hn[n] = 1.0
            continue
        log_den = np.log(abs(2.0 * n + alpha_jac + beta_jac + 1.0)) + gammaln(n + 1.0) + gammaln(denom_arg)
        hn[n] = np.exp(log_num - log_den)

    # 构建一阶微分矩阵
    D1 = np.zeros((n_modes, n_modes))
    for i in range(n_modes):
        for j in range(1, n_modes):
            # d/dx P_j ≈ (j+α+β+1)/2 * P_{j-1}^{(α+1,β+1)}
            # 近似展开回原始基
            deriv_vals = 0.5 * (j + alpha_jac + beta_jac + 1.0) * v1[:, j - 1]
            D1[i, j] = np.sum(w_q * v0[:, i] * deriv_vals) / hn[i]

    # 二阶微分矩阵
    D2 = D1 @ D1

    # 色散算子: (i β₂/2) ∂²/∂t² - (i β₃/6) ∂³/∂t³
    # 在无量纲坐标τ下，∂/∂t = (2/T_window) ∂/∂τ
    # 这里使用归一化因子 1.0 作为简化（在重构步骤中统一处理）
    # 实际上在谱空间中直接作用:

    disp_coeffs = 1j * (beta2 / 2.0) * (D2 @ coeffs) - 1j * (beta3 / 6.0) * (D2 @ D1 @ coeffs)

    return disp_coeffs
