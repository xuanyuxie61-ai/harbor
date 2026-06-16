"""
finite_difference.py - 高阶有限差分模板与虚时导数近似
=====================================================

科学背景 (Scientific Background):
    在行列式量子蒙特卡洛 (DQMC) 中, 我们需要处理虚时 (imaginary time)
    方向上的离散化. Trotter-Suzuki 分解将虚时区间 [0, β] 分为 L 段:
        Δτ = β / L

    虚时格林函数 G(τ) 的导数通过有限差分近似:
        G'(τ_n) ≈ Σ_k c_k G(τ_{n+k}) / Δτ

    高阶模板 (4阶, 6阶, 8阶) 显著降低离散化误差.
    本模块实现各类有限差分模板, 用于:
      (1) 虚时格林函数导数 → 动能估计
      (2) 虚时流方程 (flow equation) 的数值积分
      (3) 自能 (self-energy) Σ(τ) 的提取

融合种子项目:
    - 362_fd1d_heat_steady: 一维稳态热传导 FD → Dyson 方程离散化
    - 764_midpoint:          隐式中点法 → 虚时流方程积分器

核心公式 (Key Formulas):
    中心差分模板 (2阶):
        f'(x) ≈ [-f(x+h) + f(x-h)] / (2h) + O(h²)

    中心差分模板 (4阶):
        f'(x) ≈ [f(x-2h) - 8f(x-h) + 8f(x+h) - f(x+2h)] / (12h) + O(h⁴)

    中心差分模板 (6阶):
        f'(x) ≈ [-f(x-3h) + 9f(x-2h) - 45f(x-h) + 45f(x+h)
                 - 9f(x+2h) + f(x+3h)] / (60h) + O(h⁶)

    中心差分模板 (8阶):
        f'(x) ≈ [f(x-4h) - (32/3)f(x-3h) + 56f(x-2h) - 224f(x-h)
                 + 224f(x+h) - 56f(x+2h) + (32/3)f(x+3h) - f(x+4h)] / (280h)
                + O(h⁸)

    二阶导数模板 (4阶):
        f''(x) ≈ [-f(x-2h) + 16f(x-h) - 30f(x) + 16f(x+h) - f(x+2h)] / (12h²)
"""

import numpy as np
from typing import Tuple, List, Optional, Callable


# ==========================================================================
#  有限差分模板系数生成
# ==========================================================================

def fd_coefficients_first_derivative(order: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    生成一阶导数的中心差分模板系数.

    参数:
        order: 精度阶数, 必须是偶数 (2, 4, 6, 8)

    返回:
        (stencil_indices, coefficients, half_width)
        stencil_indices: 相对格点偏移 [-half_width, ..., +half_width]
        coefficients: 各偏移处的权重
        half_width: 模板半宽度

    推导:
        由 Taylor 展开:
            f(x+kh) = Σ_{n=0}^{∞} (kh)^n f^{(n)}(x) / n!
        要求:
            Σ_k c_k k^n = δ_{n,1}  for n = 0, 1, ..., 2*half_width
        解线性方程组得到 c_k.
    """
    if order not in (2, 4, 6, 8):
        raise ValueError(f"仅支持 2/4/6/8 阶精度, 收到 order={order}")

    # 预先计算好的精确系数 (有理数)
    if order == 2:
        # f'(x) ≈ [-f(x-h) + f(x+h)] / (2h)
        indices = np.array([-1, 1])
        coeffs = np.array([-1.0 / 2.0, 1.0 / 2.0])
        hw = 1
    elif order == 4:
        # 4阶中心差分
        indices = np.array([-2, -1, 1, 2])
        coeffs = np.array([1.0 / 12.0, -8.0 / 12.0,
                           8.0 / 12.0, -1.0 / 12.0])
        hw = 2
    elif order == 6:
        # 6阶中心差分
        indices = np.array([-3, -2, -1, 1, 2, 3])
        coeffs = np.array([-1.0 / 60.0, 9.0 / 60.0, -45.0 / 60.0,
                           45.0 / 60.0, -9.0 / 60.0, 1.0 / 60.0])
        hw = 3
    elif order == 8:
        # 8阶中心差分
        indices = np.array([-4, -3, -2, -1, 1, 2, 3, 4])
        coeffs = np.array([
            1.0 / 280.0,
            -32.0 / (3.0 * 280.0),
            56.0 / 280.0,
            -224.0 / 280.0,
            224.0 / 280.0,
            -56.0 / 280.0,
            32.0 / (3.0 * 280.0),
            -1.0 / 280.0,
        ])
        hw = 4
    return indices, coeffs, hw


def fd_coefficients_second_derivative(order: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    生成二阶导数的中心差分模板系数.

    2阶: f''(x) ≈ [f(x-h) - 2f(x) + f(x+h)] / h²
    4阶: f''(x) ≈ [-f(x-2h) + 16f(x-h) - 30f(x) + 16f(x+h) - f(x+2h)] / (12h²)
    """
    if order == 2:
        indices = np.array([-1, 0, 1])
        coeffs = np.array([1.0, -2.0, 1.0])
        hw = 1
    elif order == 4:
        indices = np.array([-2, -1, 0, 1, 2])
        coeffs = np.array([-1.0 / 12.0, 16.0 / 12.0, -30.0 / 12.0,
                           16.0 / 12.0, -1.0 / 12.0])
        hw = 2
    elif order == 6:
        indices = np.array([-3, -2, -1, 0, 1, 2, 3])
        coeffs = np.array([
            1.0 / 90.0, -3.0 / 20.0, 3.0 / 2.0,
            -49.0 / 18.0,
            3.0 / 2.0, -3.0 / 20.0, 1.0 / 90.0
        ])
        hw = 3
    elif order == 8:
        indices = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4])
        coeffs = np.array([
            -1.0 / 560.0, 8.0 / 315.0, -1.0 / 5.0, 8.0 / 5.0,
            -205.0 / 72.0,
            8.0 / 5.0, -1.0 / 5.0, 8.0 / 315.0, -1.0 / 560.0
        ])
        hw = 4
    else:
        raise ValueError(f"仅支持 2/4/6/8 阶精度, 收到 order={order}")
    return indices, coeffs, hw


# ==========================================================================
#  有限差分微分算子
# ==========================================================================

def apply_first_derivative(f: np.ndarray, dx: float, order: int = 4,
                           boundary: str = 'periodic') -> np.ndarray:
    """
    对数组 f 应用一阶导数有限差分模板.

    参数:
        f: 一维数组, 函数在各格点的值
        dx: 格点间距
        order: 精度阶数
        boundary: 'periodic' (周期边界) 或 'extrapolate' (外推边界)

    返回:
        df: 一阶导数的离散近似

    物理应用:
        在 DQMC 中, 虚时格林函数 G(τ) 满足反周期性:
            G(τ + β) = -G(τ)
        因此使用周期边界条件时, 需带符号翻转.
    """
    indices, coeffs, hw = fd_coefficients_first_derivative(order)
    N = len(f)
    df = np.zeros_like(f)

    if boundary == 'periodic':
        for k, c in zip(indices, coeffs):
            df += c * np.roll(f, -k) / dx
    elif boundary == 'fermionic_antiperiodic':
        # 费米子反周期边界: G(τ+β) = -G(τ)
        for k, c in zip(indices, coeffs):
            shifted = np.roll(f, -k)
            # 修正环绕部分
            if k > 0:
                shifted[-k:] = -f[:k]
            elif k < 0:
                shifted[:-k] = -f[N + k:]
            df += c * shifted / dx
    elif boundary == 'extrapolate':
        for i in range(hw, N - hw):
            for k, c in zip(indices, coeffs):
                df[i] += c * f[i + k] / dx
        # 边界用低阶模板
        for i in range(hw):
            df[i] = (f[min(i + 1, N - 1)] - f[max(i - 1, 0)]) / (2.0 * dx)
        for i in range(N - hw, N):
            df[i] = (f[min(i + 1, N - 1)] - f[max(i - 1, 0)]) / (2.0 * dx)
    else:
        raise ValueError(f"未知边界类型: {boundary}")
    return df


def apply_second_derivative(f: np.ndarray, dx: float, order: int = 4,
                            boundary: str = 'periodic') -> np.ndarray:
    """
    对数组 f 应用二阶导数有限差分模板.
    """
    indices, coeffs, hw = fd_coefficients_second_derivative(order)
    N = len(f)
    ddf = np.zeros_like(f)

    if boundary == 'periodic':
        for k, c in zip(indices, coeffs):
            ddf += c * np.roll(f, -k) / (dx * dx)
    elif boundary == 'extrapolate':
        for i in range(hw, N - hw):
            for k, c in zip(indices, coeffs):
                ddf[i] += c * f[i + k] / (dx * dx)
        # 边界用二阶模板
        for i in range(hw):
            if 0 < i < N - 1:
                ddf[i] = (f[i - 1] - 2.0 * f[i] + f[i + 1]) / (dx * dx)
            else:
                ddf[i] = 0.0
        for i in range(N - hw, N):
            if 0 < i < N - 1:
                ddf[i] = (f[i - 1] - 2.0 * f[i] + f[i + 1]) / (dx * dx)
            else:
                ddf[i] = 0.0
    return ddf


def build_fd_derivative_matrix(N: int, dx: float, order: int = 4,
                               derivative: int = 1,
                               boundary: str = 'periodic') -> np.ndarray:
    """
    构造有限差分微分算子矩阵 D.

    使得 D @ f ≈ f^{(derivative)} 的离散近似.

    在 Dyson 方程 iω_n G(iω_n) = 1 + (ε_k - μ + Σ(iω_n)) G(iω_n) 中,
    虚频导数对应于乘以 iω_n, 其离散版本即为 D 矩阵.
    """
    if derivative == 1:
        indices, coeffs, hw = fd_coefficients_first_derivative(order)
        scale = 1.0 / dx
    elif derivative == 2:
        indices, coeffs, hw = fd_coefficients_second_derivative(order)
        scale = 1.0 / (dx * dx)
    else:
        raise ValueError(f"仅支持 1 阶或 2 阶导数矩阵")

    D = np.zeros((N, N))
    for i in range(N):
        for k, c in zip(indices, coeffs):
            j = i + k
            if boundary == 'periodic':
                j_mod = j % N
                D[i, j_mod] += c * scale
            elif boundary == 'extrapolate':
                if 0 <= j < N:
                    D[i, j] += c * scale
    return D


# ==========================================================================
#  一维稳态热传导 FD 求解器 (融合 362_fd1d_heat_steady)
# ==========================================================================

def solve_steady_state_diffusion_1d(N: int, a: float, b: float,
                                    u_left: float, u_right: float,
                                    kappa_func: Callable[[np.ndarray], np.ndarray],
                                    source_func: Callable[[np.ndarray], np.ndarray]
                                    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    用有限差分法求解一维稳态扩散方程:

        -d/dx [ κ(x) du/dx ] = f(x)    x ∈ [a, b]
        u(a) = u_left, u(b) = u_right

    融合种子项目 362_fd1d_heat_steady.

    在 DQMC 语境下, 此方程类比 Dyson 方程的实空间版本:
        -∇² G(r, r') + Σ(r) G(r, r') = δ(r - r')
    其中 κ(x) → 1 (均匀传播), f(x) → δ 源项.

    离散化 (二阶中心差分):
        -κ_{i+1/2}(u_{i+1} - u_i)/h² + κ_{i-1/2}(u_i - u_{i-1})/h² = f_i

    返回:
        x: (N,) 格点位置
        u: (N,) 数值解
    """
    h = (b - a) / (N - 1)
    x = np.linspace(a, b, N)
    kappa = kappa_func(x)
    f_rhs = source_func(x)

    # 构造三对角系统
    # 半格点热导率 (算术平均)
    kappa_half = np.zeros(N + 1)
    kappa_half[1:-1] = 0.5 * (kappa[:-1] + kappa[1:])
    kappa_half[0] = kappa[0]
    kappa_half[-1] = kappa[-1]

    # 组装稀疏矩阵 (三对角)
    lower = np.zeros(N)
    diag = np.zeros(N)
    upper = np.zeros(N)
    rhs = np.zeros(N)

    for i in range(1, N - 1):
        lower[i] = -kappa_half[i] / (h * h)
        upper[i] = -kappa_half[i + 1] / (h * h)
        diag[i] = (kappa_half[i] + kappa_half[i + 1]) / (h * h)
        rhs[i] = f_rhs[i]

    # 边界条件
    diag[0] = 1.0
    rhs[0] = u_left
    diag[-1] = 1.0
    rhs[-1] = u_right

    # Thomas 算法 (追赶法) 求解三对角系统
    u = _thomas_algorithm(lower, diag, upper, rhs)
    return x, u


def _thomas_algorithm(lower: np.ndarray, diag: np.ndarray,
                      upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """
    Thomas 算法: O(N) 求解三对角线性系统.

    算法复杂度: O(N), 远优于一般 LU 分解的 O(N³).

    步骤:
        1. 前消: 修改对角和右端项
        2. 回代: 从最后一个未知量向前求解

    数值稳定性条件:
        对角占优 (diagonal dominance):
            |d_i| ≥ |l_i| + |u_i|  for all i
    """
    N = len(diag)
    d = diag.copy()
    r = rhs.copy()
    u = upper.copy()
    l = lower.copy()

    # 前消 (forward elimination)
    for i in range(1, N):
        if abs(d[i - 1]) < 1e-300:
            raise ValueError(f"Thomas 算法在第 {i-1} 步遇到零主元, "
                             f"系统可能奇异或需要对角占优条件")
        w = l[i] / d[i - 1]
        d[i] -= w * u[i - 1]
        r[i] -= w * r[i - 1]

    # 回代 (back substitution)
    x = np.zeros(N)
    if abs(d[-1]) < 1e-300:
        raise ValueError("Thomas 算法: 最后一行对角元为零")
    x[-1] = r[-1] / d[-1]
    for i in range(N - 2, -1, -1):
        x[i] = (r[i] - u[i] * x[i + 1]) / d[i]
    return x


# ==========================================================================
#  隐式中点法积分器 (融合 764_midpoint)
# ==========================================================================

def midpoint_implicit_integrator(
    rhs_func: Callable[[float, np.ndarray], np.ndarray],
    tspan: Tuple[float, float],
    y0: np.ndarray,
    n_steps: int,
    fsolve_tol: float = 1e-10,
    fsolve_maxiter: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    隐式中点法 (implicit midpoint rule) 求解 ODE:

        dy/dt = f(t, y)

    融合种子项目 764_midpoint.

    中点法公式 (单步):
        y_{n+1} = y_n + h * f(t_n + h/2, (y_n + y_{n+1})/2)

    这是一个隐式方法, 需要用 Newton 迭代求解 y_{n+1}.

    在 DQMC 中, 此积分器用于:
      - 虚时流方程 (Wegner flow / SRG flow):
            dH_λ/dλ = [[ω, H_λ], H_λ]
        其中 λ 为流参数, ω 为生成元.
      - 绝热消除快变量时的有效哈密顿量演化

    中点法是辛积分器 (symplectic), 保哈密顿结构.
    收敛阶: O(h²), 但对振荡问题有优秀的长期稳定性.
    """
    t0, tf = tspan
    h = (tf - t0) / n_steps
    t_values = np.linspace(t0, tf, n_steps + 1)
    m = len(y0)
    y_values = np.zeros((n_steps + 1, m))
    y_values[0] = y0

    for step in range(n_steps):
        t_n = t_values[step]
        y_n = y_values[step]

        # Newton 迭代求解隐式中点方程
        # 定义: G(y_new) = y_new - y_n - h * f(t_n + h/2, (y_n + y_new)/2) = 0
        y_new = y_n + h * rhs_func(t_n, y_n)  # 显式 Euler 作为初值

        for newton_iter in range(fsolve_maxiter):
            y_mid = 0.5 * (y_n + y_new)
            t_mid = t_n + 0.5 * h
            f_mid = rhs_func(t_mid, y_mid)
            residual = y_new - y_n - h * f_mid

            if np.linalg.norm(residual) < fsolve_tol:
                break

            # 近似 Jacobian: J ≈ I - (h/2) * ∂f/∂y
            # 用有限差分近似 Jacobian
            eps_fd = 1e-7
            J = np.eye(m)
            for j in range(m):
                y_mid_pert = y_mid.copy()
                y_mid_pert[j] += eps_fd
                f_pert = rhs_func(t_mid, y_mid_pert)
                J[:, j] -= (h / 2.0) * (f_pert - f_mid) / eps_fd

            # Newton 步
            try:
                delta = np.linalg.solve(J, -residual)
            except np.linalg.LinAlgError:
                # 奇异 Jacobian, 使用伪逆
                delta = np.linalg.lstsq(J, -residual, rcond=None)[0]

            y_new = y_new + delta
        else:
            # 未收敛但不中断, 使用当前最佳估计
            pass

        y_values[step + 1] = y_new

    return t_values, y_values


# ==========================================================================
#  有限差分误差分析与 Richardson 外推
# ==========================================================================

def richardson_extrapolation(f_h: float, f_h2: float,
                             order: int) -> float:
    """
    Richardson 外推: 利用两个不同步长的结果提高精度.

    若 f_h 为步长 h 的 p 阶近似, f_{h/2} 为步长 h/2 的近似, 则:
        f_extrap = (2^p * f_{h/2} - f_h) / (2^p - 1)
    的精度提升到 p+1 阶.

    物理应用:
        在 DQMC 中, 我们比较不同 Δτ 的结果来外推连续虚时极限:
            O(Δτ→0) = (2^p O(Δτ/2) - O(Δτ)) / (2^p - 1)
    """
    factor = 2.0 ** order
    return (factor * f_h2 - f_h) / (factor - 1.0)


def fd_truncation_error_estimate(f: Callable[[float], float],
                                 x0: float, dx: float,
                                 order: int = 4) -> float:
    """
    估计有限差分的截断误差.

    对于 p 阶中心差分, 截断误差:
        E = C_p * h^p * f^{(p+1)}(ξ)
    其中 C_p 为依赖于模板的常数.

    通过比较 p 阶和 p-2 阶模板的结果来估计:
        E ≈ |f'_p - f'_{p-2}|
    """
    N_test = 101
    x_test = np.linspace(x0 - 5 * dx, x0 + 5 * dx, N_test)
    f_vals = np.array([f(x) for x in x_test])

    df_p = apply_first_derivative(f_vals, dx, order=order, boundary='extrapolate')
    df_p2 = apply_first_derivative(f_vals, dx, order=max(order - 2, 2),
                                   boundary='extrapolate')
    error = np.max(np.abs(df_p - df_p2))
    return error
