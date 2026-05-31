"""
math_utils.py
数学工具函数集合

基于以下种子项目合成：
- 527_hexagon_integrals: 多边形矩计算
- 661_legendre_polynomial: 连带Legendre函数、导数等
- 144_cc_project: 数值积分辅助函数

提供通用的数学工具，包括数值积分、特殊函数、矩阵操作等。
"""

import numpy as np
from typing import Tuple


# =============================================================================
# 数值积分工具
# =============================================================================

def simpson_integration(f: callable, a: float, b: float, n: int = 1000) -> float:
    """
    Simpson数值积分：∫_a^b f(x) dx
    
    公式（n为偶数）：
        ∫ ≈ h/3 * [f_0 + 4*Σf_{odd} + 2*Σf_{even} + f_n]
    
    其中 h = (b-a)/n
    
    参数:
        f: 被积函数
        a, b: 积分区间
        n: 区间分割数（必须为偶数）
    
    返回:
        integral: 积分值
    """
    if n % 2 == 1:
        n += 1
    
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = np.array([f(xi) for xi in x], dtype=np.float64)
    
    integral = y[0] + y[-1]
    integral += 4.0 * np.sum(y[1:-1:2])
    integral += 2.0 * np.sum(y[2:-1:2])
    integral *= h / 3.0
    
    return float(integral)


def trapezoidal_integration(x: np.ndarray, y: np.ndarray) -> float:
    """
    梯形法数值积分。
    
    ∫ y(x) dx ≈ Σ_i (x_{i+1}-x_i) * (y_i + y_{i+1}) / 2
    
    参数:
        x: (n,) 自变量数组
        y: (n,) 因变量数组
    
    返回:
        integral: 积分值
    """
    dx = np.diff(x)
    avg_y = (y[:-1] + y[1:]) / 2.0
    return float(np.sum(dx * avg_y))


# =============================================================================
# 连带Legendre函数（基于661_legendre_polynomial）
# =============================================================================

def associated_legendre_value(n: int, m: int, x: float) -> float:
    """
    计算连带Legendre函数 P_n^m(x)（包含Condon-Shortley相位）。
    
    递推关系：
        P_m^m(x) = (-1)^m * (2m-1)!! * (1-x^2)^{m/2}
        P_{m+1}^m(x) = x * (2m+1) * P_m^m(x)
        P_l^m(x) = [x*(2l-1)*P_{l-1}^m(x) - (l+m-1)*P_{l-2}^m(x)] / (l-m)   for l > m+1
    
    其中 (2m-1)!! = 1·3·5·...·(2m-1)
    
    参数:
        n: 阶数
        m: 次数（0 ≤ m ≤ n）
        x: 评估点（|x| ≤ 1）
    
    返回:
        value: P_n^m(x)
    """
    x = np.clip(float(x), -1.0, 1.0)
    
    if m > n or m < 0:
        return 0.0
    
    # 计算 P_m^m
    pmm = 1.0
    if m > 0:
        somx2 = np.sqrt(max(0.0, 1.0 - x * x))
        fact = 1.0
        for i in range(1, m + 1):
            pmm *= -fact * somx2
            fact += 2.0
    
    if n == m:
        return float(pmm)
    
    # 计算 P_{m+1}^m
    pmmp1 = x * (2.0 * m + 1.0) * pmm
    
    if n == m + 1:
        return float(pmmp1)
    
    # 递推到 P_n^m
    pll = 0.0
    for l in range(m + 2, n + 1):
        pll = (x * (2.0 * l - 1.0) * pmmp1 - (l + m - 1.0) * pmm) / (l - m)
        pmm = pmmp1
        pmmp1 = pll
    
    return float(pll)


# =============================================================================
# 球谐函数（视网膜曲面上的模式展开）
# =============================================================================

def spherical_harmonic_y(l: int, m: int, theta: float, phi: float) -> complex:
    """
    计算球谐函数 Y_l^m(θ, φ)。
    
    Y_l^m(θ, φ) = sqrt((2l+1)/(4π) * (l-|m|)!/(l+|m|)!) * P_l^{|m|}(cosθ) * exp(i*m*φ)
    
    参数:
        l: 球谐阶数
        m: 次数（-l ≤ m ≤ l）
        theta: 极角 [0, π]
        phi: 方位角 [0, 2π]
    
    返回:
        Y: 复数值
    """
    if abs(m) > l:
        return 0.0 + 0.0j
    
    # 归一化常数
    # N_l^m = sqrt((2l+1)/(4π) * (l-|m|)!/(l+|m|)!)
    import math
    
    # 计算阶乘比
    fact_ratio = 1.0
    for i in range(l - abs(m) + 1, l + abs(m) + 1):
        fact_ratio /= i
    
    N = np.sqrt((2.0 * l + 1.0) / (4.0 * np.pi) * fact_ratio)
    
    # 连带Legendre（使用|m|）
    Plm = associated_legendre_value(l, abs(m), np.cos(theta))
    
    # 相位因子
    if m >= 0:
        Y = N * Plm * np.exp(1.0j * m * phi)
    else:
        # Y_l^{-m} = (-1)^m * conjugate(Y_l^m)
        phase = (-1) ** abs(m)
        Y = phase * N * Plm * np.exp(1.0j * m * phi)
    
    return complex(Y)


# =============================================================================
# 矩阵条件数与稳定性分析
# =============================================================================

def matrix_condition_number_estimate(A: np.ndarray) -> float:
    """
    估计矩阵的条件数（使用1-范数）。
    
    cond(A) = ||A||_1 * ||A^{-1}||_1
    
    参数:
        A: (n,n) 方阵
    
    返回:
        cond_est: 条件数估计
    """
    n = A.shape[0]
    
    # ||A||_1
    norm_A = np.max(np.sum(np.abs(A), axis=0))
    
    # 使用numpy的svd估计
    try:
        s = np.linalg.svd(A, compute_uv=False)
        if s[-1] < 1e-14:
            return 1e16
        return float(s[0] / s[-1])
    except Exception:
        return float(norm_A * 1e6)


def check_diagonal_dominance(A: np.ndarray) -> bool:
    """
    检查矩阵是否严格对角占优。
    
    严格对角占优：|A_ii| > Σ_{j≠i} |A_ij| 对所有i成立
    
    参数:
        A: (n,n) 方阵
    
    返回:
        is_dominant: 是否严格对角占优
    """
    n = A.shape[0]
    for i in range(n):
        diag = abs(A[i, i])
        off_diag = np.sum(np.abs(A[i, :])) - diag
        if diag <= off_diag:
            return False
    return True


# =============================================================================
# 误差分析与收敛判断
# =============================================================================

def relative_error(approx: float, exact: float) -> float:
    """
    计算相对误差：|approx - exact| / |exact|
    """
    if abs(exact) < 1e-14:
        return abs(approx - exact)
    return abs(approx - exact) / abs(exact)


def convergence_order(errors: np.ndarray, ratios: np.ndarray) -> np.ndarray:
    """
    根据误差序列估计收敛阶数。
    
    若步长减半，误差按 ratio = error_{k+1} / error_k 衰减，
    则收敛阶数 p = log(ratio) / log(1/2) = -log(ratio) / log(2)
    
    参数:
        errors: 误差序列
        ratios: 步长比序列
    
    返回:
        orders: 估计的收敛阶数
    """
    orders = np.zeros(len(errors) - 1)
    for i in range(len(errors) - 1):
        if errors[i] > 1e-14 and errors[i + 1] > 1e-14 and ratios[i] > 1e-14:
            orders[i] = np.log(errors[i + 1] / errors[i]) / np.log(ratios[i])
        else:
            orders[i] = 0.0
    return orders


# =============================================================================
# Gamma函数和对数Gamma
# =============================================================================

def log_gamma_lanczos(x: float) -> float:
    """
    使用Lanczos近似计算ln(Γ(x))。
    
    Lanczos公式（g=7, n=9）：
        Γ(z) ≈ sqrt(2π) * (z+g-0.5)^{z-0.5} * exp(-(z+g-0.5)) * A_g(z)
    
    其中 A_g(z) = c_0 + Σ_{k=1}^{n-1} c_k / (z + k - 1)
    
    参数:
        x: 正实数
    
    返回:
        ln(Γ(x))
    """
    if x <= 0:
        return np.inf
    
    # Lanczos系数（g=7）
    c = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ]
    
    g = 7.0
    z = x - 1.0
    
    # A_g(z)
    Ag = c[0]
    for k in range(1, len(c)):
        Ag += c[k] / (z + k)
    
    t = z + g + 0.5
    return 0.5 * np.log(2.0 * np.pi) + np.log(Ag) + (z + 0.5) * np.log(t) - t


# =============================================================================
# 特殊函数：误差函数近似
# =============================================================================

def erf_approx(x: float) -> float:
    """
    使用Abramowitz-Stegun近似计算误差函数erf(x)。
    
    erf(x) ≈ 1 - (a1*t + a2*t² + a3*t³ + a4*t⁴ + a5*t⁵) * exp(-x²)
    其中 t = 1 / (1 + p*x)，p=0.3275911
    
    参数:
        x: 实数
    
    返回:
        erf(x)近似值
    """
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    
    sign = 1.0 if x >= 0 else -1.0
    x = abs(x)
    
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    
    return sign * y
