"""
high_order_fd.py — 高阶有限差分算子与精度分析模块

融合种子项目:
  - 1356_trig_interp : 三角插值 (周期性函数的高精度插值)
  - 961_r8_scale : 浮点邻域 (nextafter) 用于精度控制
  - 964_r83p : 点集代数运算

物理背景:
  在喷注 substructure 分析中, 需要对以下量计算导数:
  1. 能量密度场 ε(η, φ) 的梯度 → 喷注轴方向修正
  2. N-subjettiness τ_N 对轴位置的导数 → 轴优化
  3. 横动量分布 dσ/dpT 的高阶矩 → PDF 约束
  4. 喷注质量分布的高阶矩 → 非微扰修正

高阶有限差分格式:
  一阶导数:
    D⁺_h f(x) = [f(x+h) - f(x)] / h                    (向前, O(h))
    D⁰_h f(x) = [f(x+h) - f(x-h)] / (2h)               (中心, O(h²))
    5点中心差分: [-f(x+2h)+8f(x+h)-8f(x-h)+f(x-2h)]/(12h)  (O(h⁴))

  二阶导数:
    D²_h f(x) = [f(x+h) - 2f(x) + f(x-h)] / h²        (O(h²))
    5点: [-f(x+2h)+16f(x+h)-30f(x)+16f(x-h)-f(x-2h)]/(12h²) (O(h⁴))

  三角插值 (源自 1356_trig_interp):
    对周期性函数 (如 φ 方向), 使用 Dirichlet 核:
    D_N(x) = sin((N+1/2)x) / sin(x/2)
    插值: f(x) = Σ_k y_k · D_N(x - x_k) / N
"""

import numpy as np
import math
from typing import Callable, List, Tuple, Optional
from jet_fourvector import r8_next, r8_previous


# ─────────────────────────────────────────────────────────────────────────────
# 标准有限差分系数表
# ─────────────────────────────────────────────────────────────────────────────

def fd_coefficients(derivative: int, accuracy: int,
                    scheme: str = 'central') -> Tuple[List[int], List[float]]:
    """计算有限差分系数.

    使用 Fornberg 算法生成任意阶导数、任意精度的 FD 系数.

    Args:
        derivative: 导数阶数 (1, 2, 3, ...)
        accuracy: 截断精度阶数 (2, 4, 6, 8)
        scheme: 'forward', 'backward', 'central'

    Returns:
        (stencils, coefficients): 模板偏移和系数
    """
    # 需要的点数
    if scheme == 'central':
        n_points = derivative + accuracy
        if n_points % 2 == 0:
            n_points += 1
        half = n_points // 2
        stencils = list(range(-half, half + 1))
    elif scheme == 'forward':
        n_points = derivative + accuracy
        stencils = list(range(n_points))
    else:  # backward
        n_points = derivative + accuracy
        stencils = list(range(-n_points + 1, 1))

    n = len(stencils)

    # Fornberg 算法: 计算 FD 系数
    # c[i] 使得 f^(m)(x) ≈ Σ c[i] f(x + stencils[i]*h) / h^m
    c = np.zeros((n, derivative + 1))
    c[0, 0] = 1.0

    c1 = 1.0
    for i in range(1, n):
        c2 = 1.0
        for j in range(i):
            c3 = stencils[i] - stencils[j]
            c2 *= c3

            if i <= derivative:
                pass

            for k in range(min(i, derivative), 0, -1):
                c[i, k] = c1 * (k * c[i - 1, k - 1] -
                                stencils[j] * c[i - 1, k]) / c2
            c[i, 0] = -c1 * stencils[j] * c[i - 1, 0] / c2

            for k in range(min(i, derivative), 0, -1):
                c[j, k] = (stencils[i] * c[j, k] -
                           k * c[j, k - 1]) / c3
            c[j, 0] = stencils[i] * c[j, 0] / c3

        c1 = c2

    coefficients = [float(c[i, derivative]) for i in range(n)]
    return stencils, coefficients


# ─────────────────────────────────────────────────────────────────────────────
# 一维高阶有限差分
# ─────────────────────────────────────────────────────────────────────────────

def fd_derivative_1d(values: np.ndarray, h: float,
                     derivative: int = 1,
                     accuracy: int = 4,
                     periodic: bool = False) -> np.ndarray:
    """计算一维数组的高阶有限差分离散导数.

    Args:
        values: 函数值数组
        h: 网格间距
        derivative: 导数阶数
        accuracy: 精度阶数 (2, 4, 6, 8)
        periodic: 是否为周期性边界

    Returns:
        导数值数组 (与输入同长)

    边界处理:
      - periodic: 使用周期延拓
      - 非周期: 使用单侧差分或降低精度
    """
    n = len(values)
    if n < 3:
        return np.zeros_like(values)

    stencils, coeffs = fd_coefficients(derivative, accuracy, 'central')
    half = max(abs(s) for s in stencils)

    result = np.zeros(n)

    for i in range(n):
        s = 0.0
        valid = True
        for offset, coeff in zip(stencils, coeffs):
            j = i + offset
            if periodic:
                j = j % n
            elif j < 0 or j >= n:
                # 边界: 使用更低阶的单侧差分
                valid = False
                break
            s += coeff * values[j]

        if valid:
            result[i] = s / (h ** derivative)
        else:
            # 边界处理: 向前或向后差分
            if i < half:
                # 向前差分
                fd_fwd = _one_sided_fd(values, i, h, derivative, 'forward',
                                       min(accuracy, n - i))
                result[i] = fd_fwd
            else:
                # 向后差分
                fd_bwd = _one_sided_fd(values, i, h, derivative, 'backward',
                                       min(accuracy, i + 1))
                result[i] = fd_bwd

    return result


def _one_sided_fd(values: np.ndarray, i: int, h: float,
                  derivative: int, side: str,
                  n_points: int) -> float:
    """单侧有限差分 (边界使用)."""
    n = len(values)
    n_points = max(n_points, derivative + 1)

    if side == 'forward':
        indices = list(range(i, min(i + n_points, n)))
    else:
        indices = list(range(max(i - n_points + 1, 0), i + 1))

    if len(indices) < derivative + 1:
        return 0.0

    # 简单前向/后向差分 (低阶)
    if derivative == 1 and len(indices) >= 2:
        if side == 'forward':
            return (values[indices[1]] - values[indices[0]]) / h
        else:
            return (values[indices[-1]] - values[indices[-2]]) / h
    elif derivative == 2 and len(indices) >= 3:
        return (values[indices[0]] - 2 * values[indices[1]] +
                values[indices[2]]) / (h * h)
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 二维有限差分 (用于 η-φ 能量密度场)
# ─────────────────────────────────────────────────────────────────────────────

def fd_gradient_2d(field: np.ndarray,
                   d_eta: float, d_phi: float,
                   periodic_phi: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """计算二维标量场的梯度 (4阶精度).

    ∇f = (∂f/∂η, ∂f/∂φ)

    使用 5 点中心差分 (O(h⁴)):
      ∂f/∂x ≈ [-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)] / (12h)

    Args:
        field: shape (n_eta, n_phi) 的二维数组
        d_eta, d_phi: 网格间距
        periodic_phi: φ 方向是否周期

    Returns:
        (df_deta, df_dphi)
    """
    n_eta, n_phi = field.shape
    df_deta = np.zeros_like(field)
    df_dphi = np.zeros_like(field)

    # η 方向 (非周期)
    for i in range(n_eta):
        for j in range(n_phi):
            if 2 <= i <= n_eta - 3:
                df_deta[i, j] = (
                    -field[i + 2, j] + 8 * field[i + 1, j]
                    - 8 * field[i - 1, j] + field[i - 2, j]
                ) / (12.0 * d_eta)
            elif i == 0 and n_eta > 2:
                df_deta[i, j] = (field[1, j] - field[0, j]) / d_eta
            elif i == 1 and n_eta > 3:
                df_deta[i, j] = (field[2, j] - field[0, j]) / (2 * d_eta)
            elif i == n_eta - 1 and n_eta > 2:
                df_deta[i, j] = (field[-1, j] - field[-2, j]) / d_eta
            elif i == n_eta - 2 and n_eta > 3:
                df_deta[i, j] = (field[-1, j] - field[-3, j]) / (2 * d_eta)

    # φ 方向 (可能周期)
    for i in range(n_eta):
        for j in range(n_phi):
            if periodic_phi:
                jm2 = (j - 2) % n_phi
                jm1 = (j - 1) % n_phi
                jp1 = (j + 1) % n_phi
                jp2 = (j + 2) % n_phi
                df_dphi[i, j] = (
                    -field[i, jp2] + 8 * field[i, jp1]
                    - 8 * field[i, jm1] + field[i, jm2]
                ) / (12.0 * d_phi)
            else:
                if 2 <= j <= n_phi - 3:
                    df_dphi[i, j] = (
                        -field[i, j + 2] + 8 * field[i, j + 1]
                        - 8 * field[i, j - 1] + field[i, j - 2]
                    ) / (12.0 * d_phi)
                elif j == 0 and n_phi > 2:
                    df_dphi[i, j] = (field[i, 1] - field[i, 0]) / d_phi
                elif j == n_phi - 1 and n_phi > 2:
                    df_dphi[i, j] = (field[i, -1] - field[i, -2]) / d_phi
                else:
                    df_dphi[i, j] = 0.0

    return df_deta, df_dphi


def fd_laplacian_2d(field: np.ndarray,
                    d_eta: float, d_phi: float,
                    periodic_phi: bool = True) -> np.ndarray:
    """计算二维标量场的 Laplacian (4阶精度).

    ∇²f = ∂²f/∂η² + ∂²f/∂φ²

    5点中心差分:
      ∂²f/∂x² ≈ [-f(x+2h) + 16f(x+h) - 30f(x) + 16f(x-h) - f(x-2h)] / (12h²)
    """
    n_eta, n_phi = field.shape
    lap = np.zeros_like(field)

    for i in range(n_eta):
        for j in range(n_phi):
            # η 方向
            if 2 <= i <= n_eta - 3:
                d2_eta = (
                    -field[i + 2, j] + 16 * field[i + 1, j]
                    - 30 * field[i, j] + 16 * field[i - 1, j]
                    - field[i - 2, j]
                ) / (12.0 * d_eta ** 2)
            else:
                # 低阶边界
                if 1 <= i <= n_eta - 2:
                    d2_eta = (field[i + 1, j] - 2 * field[i, j] +
                              field[i - 1, j]) / (d_eta ** 2)
                else:
                    d2_eta = 0.0

            # φ 方向
            if periodic_phi:
                jm2 = (j - 2) % n_phi
                jm1 = (j - 1) % n_phi
                jp1 = (j + 1) % n_phi
                jp2 = (j + 2) % n_phi
                d2_phi = (
                    -field[i, jp2] + 16 * field[i, jp1]
                    - 30 * field[i, j] + 16 * field[i, jm1]
                    - field[i, jm2]
                ) / (12.0 * d_phi ** 2)
            elif 2 <= j <= n_phi - 3:
                d2_phi = (
                    -field[i, j + 2] + 16 * field[i, j + 1]
                    - 30 * field[i, j] + 16 * field[i, j - 1]
                    - field[i, j - 2]
                ) / (12.0 * d_phi ** 2)
            elif 1 <= j <= n_phi - 2:
                d2_phi = (field[i, j + 1] - 2 * field[i, j] +
                          field[i, j - 1]) / (d_phi ** 2)
            else:
                d2_phi = 0.0

            lap[i, j] = d2_eta + d2_phi

    return lap


# ─────────────────────────────────────────────────────────────────────────────
# 三角插值 (源自 1356_trig_interp)
# ─────────────────────────────────────────────────────────────────────────────

def trig_interpolation(x_data: np.ndarray, y_data: np.ndarray,
                       x_eval: np.ndarray) -> np.ndarray:
    """三角插值 (用于 φ 方向的周期性插值).

    对 N 个等距节点, 使用 Dirichlet 核进行插值:

    对偶数 N:
      f(x) = (1/N) Σ_k y_k · [sin(N(x-x_k)/2) / tan((x-x_k)/2)]

    对奇数 N:
      f(x) = (1/N) Σ_k y_k · [sin(N(x-x_k)/2) / sin((x-x_k)/2)]

    此方法保证:
      - 精确通过所有数据点
      - 周期性: f(x + 2π) = f(x)
      - 对带限函数是最佳插值
    """
    N = len(x_data)
    if N == 0:
        return np.zeros_like(x_eval)
    if N == 1:
        return np.full_like(x_eval, y_data[0])

    y_eval = np.zeros_like(x_eval)

    for idx, x in enumerate(x_eval):
        s = 0.0
        for k in range(N):
            dx = x - x_data[k]
            # 折叠到 [-π, π]
            dx = dx - 2 * math.pi * round(dx / (2 * math.pi))

            if abs(dx) < 1e-14:
                s += y_data[k]
            else:
                if N % 2 == 0:
                    # 偶数 N: Dirichlet 核的变体
                    num = math.sin(N * dx / 2)
                    den = math.tan(dx / 2)
                    if abs(den) < 1e-30:
                        s += y_data[k]
                    else:
                        s += y_data[k] * num / den
                else:
                    # 奇数 N
                    num = math.sin(N * dx / 2)
                    den = math.sin(dx / 2)
                    if abs(den) < 1e-30:
                        s += y_data[k]
                    else:
                        s += y_data[k] * num / den
        y_eval[idx] = s / N

    return y_eval


def trig_interpolation_cardinal(xd: np.ndarray, yd: np.ndarray,
                                xi: np.ndarray) -> np.ndarray:
    """三角 cardinal 插值 (源自 1356_trig_interp 的核心函数).

    Cardinal 基函数:
      C_k(x) = (1/N) sin(N(x-x_k)/2) / sin((x-x_k)/2)
    满足 C_k(x_j) = δ_{kj}.
    """
    return trig_interpolation(xd, yd, xi)


# ─────────────────────────────────────────────────────────────────────────────
# Richardson 外推 (提高精度)
# ─────────────────────────────────────────────────────────────────────────────

def richardson_extrapolation(f: Callable, x: float, h: float,
                             derivative: int = 1,
                             n_levels: int = 3) -> Tuple[float, float]:
    """Richardson 外推法计算导数.

    使用不同步长的中心差分, 通过外推消除低阶误差项.

    对 2 阶精度格式:
      D(h) = f'(x) + c₂h² + c₄h⁴ + ...
      D_extrap = (4D(h/2) - D(h)) / 3  → O(h⁴)

    Returns:
        (extrapolated_value, error_estimate)
    """
    # 计算不同步长的差分
    h_values = [h / (2 ** i) for i in range(n_levels)]
    d_values = []

    for h_i in h_values:
        if derivative == 1:
            d = (f(x + h_i) - f(x - h_i)) / (2 * h_i)
        elif derivative == 2:
            d = (f(x + h_i) - 2 * f(x) + f(x - h_i)) / (h_i ** 2)
        else:
            d = 0.0
        d_values.append(d)

    # Richardson 外推表
    table = np.array(d_values, dtype=float)
    for k in range(1, n_levels):
        factor = 4 ** k  # 对 2 阶格式
        new_table = np.zeros(n_levels - k)
        for i in range(n_levels - k):
            new_table[i] = (factor * table[i + 1] - table[i]) / (factor - 1)
        table = new_table

    result = table[0]
    error_est = abs(table[0] - d_values[0]) if len(d_values) > 0 else 0.0

    return float(result), float(error_est)


# ─────────────────────────────────────────────────────────────────────────────
# 数值精度敏感性分析 (源自 961_r8_scale)
# ─────────────────────────────────────────────────────────────────────────────

def fd_precision_scan(f: Callable, x: float, h_range: np.ndarray = None,
                      derivative: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """扫描步长 h 对有限差分精度的影响.

    展示典型的 "最佳步长" 现象:
      - h 太大: 截断误差主导 (O(h^p))
      - h 太小: 舍入误差主导 (O(ε/h))
      - 最佳 h: 两者平衡点

    Returns:
        (h_values, errors): 步长和对应的绝对误差
    """
    if h_range is None:
        h_range = np.logspace(-16, -1, 30)

    # 精确导数 (用 Richardson 外推)
    exact, _ = richardson_extrapolation(f, x, 1e-4, derivative, n_levels=5)

    errors = np.zeros_like(h_range)
    for i, h in enumerate(h_range):
        if derivative == 1:
            approx = (f(x + h) - f(x - h)) / (2 * h)
        elif derivative == 2:
            approx = (f(x + h) - 2 * f(x) + f(x - h)) / (h ** 2)
        else:
            approx = 0.0
        errors[i] = abs(approx - exact)

    return h_range, errors


def optimal_step_size(epsilon: float = None,
                      derivative: int = 1,
                      accuracy_order: int = 2) -> float:
    """计算最佳步长 h_opt.

    对 p 阶精度格式, 总误差:
      E(h) = C₁ h^p + C₂ ε/h

    最优: h_opt = (C₂ ε / (p C₁))^{1/(p+1)}
    近似: h_opt ≈ ε^{1/(p+1)}

    Args:
        epsilon: 机器精度 (默认 float64)
        derivative: 导数阶数
        accuracy_order: 差分格式阶数
    """
    if epsilon is None:
        epsilon = np.finfo(float).eps

    p = accuracy_order
    h_opt = epsilon ** (1.0 / (p + 1))
    return h_opt
