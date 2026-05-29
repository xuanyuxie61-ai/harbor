"""
参数优化与模型校准模块
======================
融合种子项目:
  - 476_golden_section : 黄金分割搜索
  - 210_continuation   : 延拓法/路径跟踪

在金融工程中，模型校准是将理论模型参数调整至与市场观测数据
（如隐含波动率微笑/偏斜）最佳匹配的过程。

本模块实现：
1. 黄金分割搜索 —— 一维模型参数（如相关性ρ）的局部优化
2. 延拓法（Continuation/Homotopy） —— 跟踪波动率微笑曲线随
   参数变化的解分支，实现全局鲁棒校准

数学背景:
---------
黄金分割搜索:
    对于区间[a,b]上的单峰函数f(x)，取内点:
        x1 = φ·a + (1-φ)·b
        x2 = (1-φ)·a + φ·b
    其中 φ = (√5 - 1)/2 ≈ 0.618 为黄金分割比。
    通过比较f(x1)与f(x2)缩小区间，收敛率 ≈ 0.618。

延拓法（参数同伦）:
    给定非线性系统 F(x; λ) = 0，从已知解 (x0, λ0) 出发，
    通过预测-校正步跟踪解曲线:
        预测: x1 = x0 + h · t
        校正: Newton迭代求解 G(x) = [F(x;λ); x_p - x1_p] = 0
    其中 t 为切向量，满足 J·t = 0, ||t||=1。

校准目标泛函（最小二乘）:
    min_Θ  Σ_{i,j} w_{ij} [ σ_{model}(K_i, T_j; Θ) - σ_{market}(K_i, T_j) ]²
    其中 Θ = (κ, θ, σ, ρ, v0) 为Heston模型参数。
"""

import numpy as np
from math import sqrt


# ========================================================================
# 476_golden_section : 黄金分割搜索
# ========================================================================

def golden_section_search(f, a, b, n_max=100, x_tol=1e-12):
    """
    黄金分割搜索求解一维单峰函数在[a,b]上的极小值点。

    算法:
    ------
    φ = (√5 - 1)/2 ≈ 0.6180339887
    区间缩小比例恒为 φ，每次迭代只需计算一个新函数值。

    参数:
    ------
    f      : callable, 目标函数
    a, b   : float, 初始区间端点
    n_max  : int, 最大迭代次数
    x_tol  : float, 区间宽度容限

    返回:
    ------
    a_out, b_out : float, 包含极小值的最终区间
    it           : int, 迭代次数
    nf           : int, 函数求值次数
    """
    a = float(a)
    b = float(b)
    if a >= b:
        raise ValueError("必须满足 a < b")
    phi = (sqrt(5.0) - 1.0) / 2.0
    nf = 0

    x1 = phi * a + (1.0 - phi) * b
    x2 = (1.0 - phi) * a + phi * b
    f1 = f(x1)
    f2 = f(x2)
    nf += 2

    for it in range(1, n_max + 1):
        if abs(b - a) <= x_tol:
            return a, b, it, nf

        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = phi * a + (1.0 - phi) * b
            f1 = f(x1)
            nf += 1
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = (1.0 - phi) * a + phi * b
            f2 = f(x2)
            nf += 1

    return a, b, n_max, nf


# ========================================================================
# 210_continuation : 延拓法/路径跟踪
# ========================================================================

def compute_tangent(J, n, p):
    """
    计算解曲线的单位切向量。

    方程组:
        F(x; λ) = 0   (n-1个方程)
    其中第n个变量为延拓参数λ。

    雅可比矩阵 J = ∂F/∂x ∈ R^{(n-1)×n} 为长方阵。
    切向量 t 满足 J · t = 0，即 t 为 J 的零空间向量。
    通过QR分解或SVD求得。
    """
    J = np.asarray(J, dtype=np.float64)
    # 使用SVD求零空间
    u, s, vh = np.linalg.svd(J)
    t = vh[-1, :].copy()
    # 保证连续性：与前一方向同向
    norm_t = np.linalg.norm(t)
    if norm_t < 1e-15:
        raise RuntimeError("切向量范数为零，可能遇到分歧点")
    t = t / norm_t
    return t


def continuation_step(f, fp, x0, p0, h, tol=1e-10, it_max=10):
    """
    执行一个延拓步：从已知解 x0 沿切线方向前进 h，然后用Newton校正。

    参数:
    ------
    f      : callable, f(n, x) 返回 (n-1,) 的函数值
    fp     : callable, fp(n, x) 返回 (n-1, n) 的雅可比矩阵
    x0     : ndarray, 当前解点 (n,)
    p0     : int, 当前延拓参数索引
    h      : float, 步长
    tol    : float, Newton收敛容限
    it_max : int, Newton最大迭代次数

    返回:
    ------
    status : int, 0=成功, 1=Newton不收敛
    x2     : ndarray, 新解点
    t2     : ndarray, 新切向量
    p2     : int, 建议的下一步延拓参数索引
    """
    n = len(x0)
    x0 = np.asarray(x0, dtype=np.float64)

    # 计算切向量
    J0 = fp(n, x0)
    t2 = compute_tangent(J0, n, p0)
    # 选择下一步延拓参数：取切向量绝对值最大分量的索引
    p2 = int(np.argmax(np.abs(t2)))

    # 预测步
    x1 = x0 + h * t2
    x1[p0] = x0[p0] + h * t2[p0]  # 保持参数方向

    # Newton校正：固定第p0个分量
    x = x1.copy()
    alpha = x1[p0]

    for it in range(it_max):
        fx = f(n, x)
        fx = np.append(fx, x[p0] - alpha)
        fx_norm = np.max(np.abs(fx))
        if fx_norm <= tol:
            return 0, x, t2, p2

        J = fp(n, x)
        J_aug = np.zeros((n, n), dtype=np.float64)
        J_aug[:n-1, :] = J
        J_aug[n-1, :] = 0.0
        J_aug[n-1, p0] = 1.0

        try:
            dx = np.linalg.solve(J_aug, -fx)
        except np.linalg.LinAlgError:
            return 1, x, t2, p2
        x = x + dx

    return 1, x, t2, p2


def continuation_trace(f, fp, x_start, p_start, h_init, target_param_index,
                       target_value, max_steps=100, tol=1e-8):
    """
    跟踪解曲线直到指定参数达到目标值。

    参数:
    ------
    f                  : callable, 非线性方程组
    fp                 : callable, 雅可比矩阵
    x_start            : ndarray, 初始解
    p_start            : int, 初始延拓参数索引
    h_init             : float, 初始步长
    target_param_index : int, 目标参数索引
    target_value       : float, 目标参数值
    max_steps          : int, 最大步数
    tol                : float, 收敛容限

    返回:
    ------
    list of ndarray, 解曲线上的点序列
    """
    path = [x_start.copy()]
    x = x_start.copy()
    p = p_start
    h = h_init
    h_min = 1e-8
    h_max = 0.5

    for step in range(max_steps):
        status, x_new, t_new, p_new = continuation_step(f, fp, x, p, h, tol=tol)
        if status != 0:
            # 步长减半重试
            h *= 0.5
            if h < h_min:
                break
            continue

        path.append(x_new.copy())
        x = x_new
        p = p_new

        # 检查是否到达目标
        if abs(x[target_param_index] - target_value) < tol:
            break

        # 自适应步长调整
        if status == 0:
            h = min(h * 1.2, h_max)

    return path


# ========================================================================
# Heston模型校准目标函数
# ========================================================================

def heston_calibration_objective(market_iv, strikes, maturities, params_to_opt,
                                 fixed_params, param_index):
    """
    构建Heston模型校准的一维目标函数（用于黄金分割搜索或延拓法）。

    参数:
    ------
    market_iv   : ndarray, 市场隐含波动率
    strikes     : ndarray, 行权价
    maturities  : ndarray, 到期日
    params_to_opt: list of str, 待优化参数名
    fixed_params : dict, 固定参数值
    param_index  : int, 当前一维优化的参数在params_to_opt中的索引

    返回:
    ------
    callable, f(param_value) -> 校准误差平方和
    """
    from heston_pde_engine import heston_european_call_price

    def objective(param_value):
        params = fixed_params.copy()
        params[params_to_opt[param_index]] = param_value
        error = 0.0
        count = 0
        for i, K in enumerate(strikes):
            for j, T in enumerate(maturities):
                if i >= market_iv.shape[0] or j >= market_iv.shape[1]:
                    continue
                try:
                    model_price = heston_european_call_price(
                        S0=params['S0'], K=K, T=T, r=params['r'],
                        kappa=params['kappa'], theta=params['theta'],
                        sigma=params['sigma'], rho=params['rho'], v0=params['v0']
                    )
                    # 由模型价格反推隐含波动率（简化：使用Brent法或直接比较价格）
                    market_price = black_scholes_call_price(
                        params['S0'], K, T, params['r'], market_iv[i, j]
                    )
                    diff = model_price - market_price
                    error += diff * diff
                    count += 1
                except Exception:
                    continue
        if count == 0:
            return 1e10
        return error / count

    return objective


def black_scholes_call_price(S0, K, T, r, sigma):
    """
    Black-Scholes欧式看涨期权定价公式。

        C = S0 N(d1) - K e^{-rT} N(d2)
        d1 = [ln(S0/K) + (r + σ²/2)T] / (σ√T)
        d2 = d1 - σ√T
    """
    from math import log, sqrt, exp, erf
    if T <= 0 or sigma <= 0:
        return max(S0 - K, 0.0)
    d1 = (log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    # 标准正态CDF
    nd1 = 0.5 * (1.0 + erf(d1 / sqrt(2.0)))
    nd2 = 0.5 * (1.0 + erf(d2 / sqrt(2.0)))
    return S0 * nd1 - K * exp(-r * T) * nd2


def calibrate_rho_golden_section(market_iv, strikes, maturities, fixed_params):
    """
    使用黄金分割搜索校准Heston模型相关性参数ρ。

    ρ的物理约束: -1 ≤ ρ ≤ 1，通常实际范围 [-0.9, -0.1]。
    """
    obj = heston_calibration_objective(market_iv, strikes, maturities,
                                       ['rho'], fixed_params, 0)
    a, b, it, nf = golden_section_search(obj, -0.95, -0.05, n_max=50, x_tol=1e-4)
    best_rho = (a + b) / 2.0
    best_err = obj(best_rho)
    return best_rho, best_err, it, nf
