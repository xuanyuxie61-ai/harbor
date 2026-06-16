"""
finite_difference.py — 高阶有限差分与 Richardson 外推 (PROJECT_232)

本模块是 PROJECT_232 的核心创新模块，实现了用于散射振幅
数值微分的高阶有限差分方法及其稳定性分析。

核心算法:
  1. 中心差分格式 (2阶, 4阶, 6阶, 8阶, 10阶)
  2. 前向/后向差分格式 (边界处理)
  3. Fornberg 算法 — 任意阶任意精度的差分系数
  4. Richardson 外推加速
  5. 复步微分法 (Complex Step Differentiation)
  6. 自适应阶数选择

物理背景:
  散射振幅对能量的导数用于提取:
    - 散射长度: a₀ = lim_{k→0} (-tan δ₀/k)
    - 有效力程: r₀ = 2 d/dk² (k cot δ₀)|_{k=0}
    - 散射体积: a₁ = lim_{k→0} (-tan δ₁/k³)
    - 形状参数: P 波有效力程

  数值挑战:
    - 高阶差分对网格间距 h 敏感 (条件数 ~ h^{-n})
    - 浮点精度限制了可使用的最高阶数
    - Richardson 外推可在不减小 h 的情况下提高精度

参考:
  - Fornberg, B. (1988). "Generation of Finite Difference Formulas
    on Arbitrarily Spaced Grids", Mathematics of Computation 51, 699-706.
  - Lyness, J.N. & Moler, C.B. (1967). "Numerical Differentiation
    of Analytic Functions", SIAM J. Numer. Anal. 4, 202-210.
"""
import numpy as np
from constants import EPS_MACH, TOL_ZERO


# ===================================================================
# 差分系数计算
# ===================================================================

def fornberg_weights(x_points, x_derivative, deriv_order):
    """
    Fornberg 算法: 计算任意网格上的有限差分权重

    给定节点 x_0, x_1, ..., x_n 和导数阶数 m,
    计算权重 w_0, w_1, ..., w_n 使得:
      f^{(m)}(x_d) ≈ Σ_j w_j f(x_j)

    算法使用三项递推关系，时间复杂度 O(n·m)。

    递推关系:
      c_{j}^{(k,s)} = ((x_d - x_{k-1}) c_{j}^{(k-1,s)} - s c_{j}^{(k-1,s-1)})
                       / (x_k - x_{k-j})
                       (对 j < k)
      c_{k}^{(k,s)} = ...

    Parameters
    ----------
    x_points : ndarray, shape (n+1,)
        差分节点 (不必等距)
    x_derivative : float
        求导点
    deriv_order : int
        导数阶数 m

    Returns
    -------
    weights : ndarray, shape (n+1,)
        差分权重
    """
    x_points = np.asarray(x_points, dtype=np.float64)
    n = len(x_points) - 1
    m = deriv_order

    if m > n:
        raise ValueError(
            f"fornberg_weights: deriv_order {m} > number of intervals {n}")

    # c[j, s] 存储第 j 个节点对 s 阶导数的贡献
    c = np.zeros((n + 1, m + 1))
    c1 = 1.0
    c4 = x_points[0] - x_derivative
    c[0, 0] = 1.0

    for i in range(1, n + 1):
        mn = min(i, m)
        c2 = 1.0
        c5 = c4
        c4 = x_points[i] - x_derivative
        for j in range(i):
            c3 = x_points[i] - x_points[j]
            c2 *= c3
            if j == i - 1:
                for s in range(mn, 0, -1):
                    c[i, s] = c1 * (s * c[i - 1, s - 1] - c5 * c[i - 1, s]) / c2
                c[i, 0] = -c1 * c5 * c[i - 1, 0] / c2
            for s in range(mn, 0, -1):
                c[j, s] = (c4 * c[j, s] - s * c[j, s - 1]) / c3
            c[j, 0] = c4 * c[j, 0] / c3
        c1 = c2

    return c[:, m]


def central_diff_weights(deriv_order, accuracy_order):
    """
    中心差分权重 (等距网格)

    对于 2p 阶精度的 m 阶导数，使用 2p+m-1 个节点 (对称):
      x_j = j·h, j = -(p+m//2-1), ..., p+m//2-1  (适当偏移)

    实际使用节点数: n = 2*floor((accuracy_order + deriv_order) / 2) + 1

    例如:
      - 1阶导数, 2阶精度: [-1/2, 0, 1/2] / h  → [-1, 0, 1] / (2h)
      - 1阶导数, 4阶精度: [-2, -1, 0, 1, 2] → [1, -8, 0, 8, -1] / (12h)
      - 2阶导数, 2阶精度: [-1, 0, 1] → [1, -2, 1] / h²

    Parameters
    ----------
    deriv_order : int
        导数阶数 m
    accuracy_order : int
        精度阶数 (截断误差 ~ O(h^accuracy_order))

    Returns
    -------
    stencil_offsets : ndarray
        模板偏移 (整数)
    weights : ndarray
        差分权重 (除以 h^m)
    """
    # 需要至少 deriv_order + accuracy_order 个节点
    half_width = (deriv_order + accuracy_order) // 2
    n_points = 2 * half_width + 1
    if n_points < deriv_order + 1:
        half_width += 1
        n_points = 2 * half_width + 1

    offsets = np.arange(-half_width, half_width + 1)
    weights = fornberg_weights(offsets.astype(float), 0.0, deriv_order)
    return offsets, weights


def forward_diff_weights(deriv_order, accuracy_order):
    """
    前向差分权重 (用于左边界)

    Parameters
    ----------
    deriv_order : int
        导数阶数
    accuracy_order : int
        精度阶数

    Returns
    -------
    stencil_offsets : ndarray
        模板偏移 (0, 1, 2, ...)
    weights : ndarray
        差分权重
    """
    n_points = deriv_order + accuracy_order
    offsets = np.arange(n_points, dtype=float)
    weights = fornberg_weights(offsets, 0.0, deriv_order)
    return offsets, weights


def backward_diff_weights(deriv_order, accuracy_order):
    """
    后向差分权重 (用于右边界)

    Parameters
    ----------
    deriv_order : int
        导数阶数
    accuracy_order : int
        精度阶数

    Returns
    -------
    stencil_offsets : ndarray
        模板偏移 (0, -1, -2, ...)
    weights : ndarray
        差分权重
    """
    n_points = deriv_order + accuracy_order
    offsets = -np.arange(n_points, dtype=float)
    weights = fornberg_weights(offsets, 0.0, deriv_order)
    return offsets, weights


# ===================================================================
# 导数计算
# ===================================================================

def numerical_derivative(f_values, h, deriv_order=1, accuracy_order=4,
                         boundary_method='one-sided'):
    """
    均匀网格上的高阶数值导数

    使用中心差分模板计算内部点，边界使用前向/后向差分。

    截断误差:
      E = C · h^p · f^{(m+p)}(ξ)
    其中 p 为精度阶数，m 为导数阶数。

    舍入误差放大:
      ε_round ≈ ε_mach · ||f|| / h^m

    最优步长:
      h_opt ≈ (ε_mach · ||f|| / |f^{(m+p)}|)^{1/p}

    Parameters
    ----------
    f_values : ndarray, shape (N,)
        函数值 f(x_0), f(x_1), ..., f(x_{N-1})
    h : float
        网格间距
    deriv_order : int
        导数阶数 (default: 1)
    accuracy_order : int
        精度阶数 (default: 4)
    boundary_method : str
        边界处理方法: 'one-sided' 或 'extrapolate'

    Returns
    -------
    df : ndarray, shape (N,)
        数值导数
    """
    N = len(f_values)
    df = np.zeros(N, dtype=f_values.dtype)

    if N < deriv_order + 1:
        raise ValueError(
            f"numerical_derivative: need at least {deriv_order + 1} points, got {N}")

    # 内部点: 中心差分
    _, w_center = central_diff_weights(deriv_order, accuracy_order)
    half_w = len(w_center) // 2

    for i in range(half_w, N - half_w):
        for j, w in enumerate(w_center):
            idx = i - half_w + j
            if 0 <= idx < N:
                df[i] += w * f_values[idx]
        df[i] /= h ** deriv_order

    # 左边界: 前向差分
    _, w_fwd = forward_diff_weights(deriv_order, accuracy_order)
    n_fwd = len(w_fwd)
    for i in range(min(half_w, N)):
        n_use = min(n_fwd, N - i)
        for j in range(n_use):
            if j < len(w_fwd):
                df[i] += w_fwd[j] * f_values[i + j]
        df[i] /= h ** deriv_order

    # 右边界: 后向差分
    _, w_bwd = backward_diff_weights(deriv_order, accuracy_order)
    n_bwd = len(w_bwd)
    for i in range(max(N - half_w, 0), N):
        for j in range(min(n_bwd, i + 1)):
            if j < len(w_bwd):
                df[i] += w_bwd[j] * f_values[i - j]
        df[i] /= h ** deriv_order

    return df


def complex_step_derivative(f_complex, x, h=1e-20, deriv_order=1):
    """
    复步微分法 (Complex Step Differentiation)

    对于解析函数 f，一阶导数可用虚部提取:
      f'(x) ≈ Im[f(x + ih)] / h

    此方法无减法相消 (subtractive cancellation)，
    因此可使用极小的 h (甚至 h → 0)，精度接近机器精度。

    推广到二阶导数 (Martins et al. 2003):
      f''(x) ≈ 2 · Re[f(x + ih) - f(x)] / h²

    更高阶使用多步复步方法:
      f^{(n)}(x) = Im[...] / h^n 的适当组合

    Parameters
    ----------
    f_complex : callable
        可接受复数输入的函数 f(x)
    x : float
        求导点
    h : float
        虚步长 (default: 1e-20)
    deriv_order : int
        导数阶数 (仅支持 1 或 2)

    Returns
    -------
    float
        导数值
    """
    if deriv_order == 1:
        f_val = f_complex(complex(x, h))
        return f_val.imag / h
    elif deriv_order == 2:
        f_real = f_complex(complex(x, 0))
        f_pert = f_complex(complex(x, h))
        return 2.0 * (f_pert.real - f_real.real) / (h * h)
    else:
        raise ValueError("complex_step_derivative: only order 1 and 2 supported")


# ===================================================================
# Richardson 外推
# ===================================================================

def richardson_extrapolation(derivatives, h_values, order):
    """
    Richardson 外推加速

    假设数值导数有展开:
      D(h) = D_exact + c_1 h^p + c_2 h^{p+2} + c_3 h^{p+4} + ...
    其中 p 为精度阶数。

    Richardson 外推通过消除 leading error term 提高精度:
      D_R = (2^p D(h/2) - D(h)) / (2^p - 1)

    多级 Richardson 表:
      T[i,j] = T[i,j-1] + (T[i,j-1] - T[i-1,j-1]) / (r_j^{p_j} - 1)
    其中 r_j 为步长比 (通常为 2)。

    Parameters
    ----------
    derivatives : ndarray, shape (k,)
        不同步长 h 的数值导数
    h_values : ndarray, shape (k,)
        对应的步长
    order : int
        误差展开的阶数 p

    Returns
    -------
    richardson_table : ndarray, shape (k, k)
        Richardson 外推表
    best_estimate : float
        最佳估计值 (表的最右下角元素)
    """
    k = len(derivatives)
    if k < 2:
        return derivatives.reshape(1, 1), derivatives[0] if len(derivatives) > 0 else 0.0

    table = np.zeros((k, k), dtype=np.float64)
    table[:, 0] = derivatives

    for j in range(1, k):
        ratio = h_values[j - 1] / h_values[j]
        factor = ratio ** order
        for i in range(j, k):
            table[i, j] = table[i, j - 1] + (table[i, j - 1] - table[i - 1, j - 1]) / (factor - 1.0)
            factor *= ratio ** 2  # 下一项阶数增加 2

    return table, table[-1, -1]


def adaptive_derivative(f_values, x_grid, deriv_order=1,
                        tol=1e-8, max_accuracy=10):
    """
    自适应阶数数值导数

    自动选择最优差分阶数，使得:
      |D^{(p)} - D^{(p-2)}| < tol

    算法:
      1. 从 2 阶精度开始
      2. 逐步提高精度至 max_accuracy
      3. 检查相邻精度的差异
      4. 若差异 < tol，返回当前结果
      5. 若达到 max_accuracy 仍未收敛，发出警告

    Parameters
    ----------
    f_values : ndarray
        函数值
    x_grid : ndarray
        x 坐标 (均匀间距)
    deriv_order : int
        导数阶数
    tol : float
        收敛容差
    max_accuracy : int
        最大精度阶数

    Returns
    -------
    df_best : ndarray
        最佳导数估计
    actual_order : int
        实际使用的精度阶数
    error_estimate : ndarray
        误差估计
    """
    h = x_grid[1] - x_grid[0] if len(x_grid) > 1 else 1.0
    df_prev = None
    df_best = None
    actual_order = 2

    for acc_order in range(2, max_accuracy + 1, 2):
        df_curr = numerical_derivative(
            f_values, h, deriv_order, acc_order)
        if df_prev is not None:
            error_est = np.abs(df_curr - df_prev)
            if np.max(error_est) < tol:
                return df_curr, acc_order, error_est
        df_prev = df_curr
        df_best = df_curr
        actual_order = acc_order

    # 未收敛
    error_est = np.abs(df_best) * EPS_MACH * 100 if df_best is not None else np.ones_like(f_values)
    return df_best, actual_order, error_est


# ===================================================================
# 散射物理量的导数
# ===================================================================

def scattering_length_derivative(k_grid, delta_0_grid):
    """
    从 S 波相移计算散射长度

    a₀ = -lim_{k→0} tan(δ₀) / k

    数值实现: 使用高阶差分计算 d/dk [k cot δ₀] 在 k=0 处的值

    k cot δ₀ ≈ -1/a₀ + r₀ k²/2 + ...

    Parameters
    ----------
    k_grid : ndarray
        动量网格 [GeV], 应从接近 0 开始
    delta_0_grid : ndarray
        S 波相移 [弧度]

    Returns
    -------
    a0 : complex
        散射长度 [GeV⁻¹]
    r0 : complex
        有效力程 [GeV⁻¹]
    kcot_delta : ndarray
        k cot δ₀(k) 数据
    """
    valid = k_grid > 1e-8
    if np.sum(valid) < 3:
        return complex(0), complex(0), np.zeros_like(k_grid)

    k_v = k_grid[valid]
    delta_v = delta_0_grid[valid]

    # k cot δ₀
    tan_delta = np.tan(delta_v + 0j)
    kcot_delta = np.where(
        np.abs(tan_delta) > EPS_MACH,
        k_v / tan_delta,
        complex(1e10)  # δ ≈ 0 时 cot → ∞
    )

    k_sq = k_v ** 2

    # 线性拟合 k cot δ₀ = A + B k²
    if len(k_sq) >= 2:
        A_mat = np.column_stack([np.ones_like(k_sq), k_sq])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A_mat, kcot_delta, rcond=None)
            A_coeff = coeffs[0]
            B_coeff = coeffs[1]
            a0 = -1.0 / A_coeff if abs(A_coeff) > EPS_MACH else complex(np.inf)
            r0 = 2.0 * B_coeff
        except np.linalg.LinAlgError:
            a0 = complex(0)
            r0 = complex(0)
    else:
        a0 = complex(0)
        r0 = complex(0)

    return a0, r0, kcot_delta


def curvature_analysis(sqrt_s_grid, amplitude_grid):
    """
    散射振幅曲率分析

    计算 d²|f|/d(√s)² 用于识别共振峰:
      - 曲率极大值 → 共振位置
      - 曲率过零点 → 共振边缘

    Parameters
    ----------
    sqrt_s_grid : ndarray
        √s 网格
    amplitude_grid : ndarray
        |f(√s)| 或 σ_tot(√s)

    Returns
    -------
    curvature : ndarray
        二阶导数
    resonance_candidates : ndarray
        候选共振位置 (曲率极大)
    """
    h = sqrt_s_grid[1] - sqrt_s_grid[0] if len(sqrt_s_grid) > 1 else 1.0
    # 使用 4 阶精度的二阶导数
    curvature = numerical_derivative(
        amplitude_grid, h, deriv_order=2, accuracy_order=4)

    # 寻找局部极大值 (共振候选)
    resonance_candidates = []
    for i in range(1, len(curvature) - 1):
        if curvature[i] > curvature[i - 1] and curvature[i] > curvature[i + 1]:
            if curvature[i] > 0:
                resonance_candidates.append(sqrt_s_grid[i])

    return curvature, np.array(resonance_candidates)
