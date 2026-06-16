"""
fp_solver.py — Fokker-Planck 方程时间推进求解器
================================================

核心数值方法:
  1. 4 阶有限差分离散 (速度空间)
  2. Crank-Nicolson 隐式时间推进
  3. 三对角/五对角系统求解
  4. Newton 迭代 (非线性碰撞算子)

物理方程 (无量纲):
  ∂f/∂τ = C[f] = (1/x²) ∂/∂x { x² [D(x) ∂f/∂x + 2x G(x) M(x)/x² · f] }

  展开为:
  ∂f/∂τ = D(x) ∂²f/∂x² + [D'(x) + 2D(x)/x + A(x)] ∂f/∂x
          + [A'(x) + 2A(x)/x] f

  其中 A(x) = 2 G(x) M(x)/x², D(x) = G(x) M(x)/x

边界条件:
  v = 0 (x → 0):  ∂f/∂x = 0 (对称性)
  v = v_max (x → ∞): f → 0 (无粒子逃逸)

守恒律:
  粒子数: ∫ f 4π x² dx = const
  能量:   ∫ f x² 4π x² dx = const
"""

import numpy as np

from physical_constants import FOUR_PI, PI, maxwellian_1d
from special_functions import chandrasekhar_G
from fp_collision import (
    compute_collision_operator,
    compute_cumulative_M,
    compute_collision_coefficients,
)
from banded_solver import solve_tridiagonal


# ===========================================================================
#  §1  4 阶有限差分算子
# ===========================================================================
def fd_first_derivative_4th(f, dx):
    """4 阶中心差分一阶导数.

    f'(x_j) = (-f_{j+2} + 8f_{j+1} - 8f_{j-1} + f_{j-2}) / (12 Δx)
    """
    N = len(f)
    df = np.zeros(N)
    for j in range(2, N - 2):
        df[j] = (-f[j+2] + 8.0*f[j+1] - 8.0*f[j-1] + f[j-2]) / (12.0 * dx)
    # 边界: 2 阶单侧
    if N > 3:
        df[0] = (-3.0*f[0] + 4.0*f[1] - f[2]) / (2.0 * dx)
        df[1] = (-3.0*f[1] + 4.0*f[2] - f[3]) / (2.0 * dx)
        df[N-1] = (3.0*f[N-1] - 4.0*f[N-2] + f[N-3]) / (2.0 * dx)
        df[N-2] = (3.0*f[N-2] - 4.0*f[N-3] + f[N-4]) / (2.0 * dx)
    return df


def fd_second_derivative_4th(f, dx):
    """4 阶中心差分二阶导数.

    f''(x_j) = (-f_{j+2} + 16f_{j+1} - 30f_j + 16f_{j-1} - f_{j-2}) / (12 Δx²)
    """
    N = len(f)
    d2f = np.zeros(N)
    for j in range(2, N - 2):
        d2f[j] = (-f[j+2] + 16.0*f[j+1] - 30.0*f[j]
                    + 16.0*f[j-1] - f[j-2]) / (12.0 * dx**2)
    # 边界
    if N > 4:
        d2f[0] = (2.0*f[0] - 5.0*f[1] + 4.0*f[2] - f[3]) / dx**2
        d2f[1] = (f[0] - 2.0*f[1] + f[2]) / dx**2
        d2f[N-1] = (2.0*f[N-1] - 5.0*f[N-2] + 4.0*f[N-3] - f[N-4]) / dx**2
        d2f[N-2] = (f[N-1] - 2.0*f[N-2] + f[N-3]) / dx**2
    return d2f


# ===========================================================================
#  §2  碰撞算子的矩阵形式
# ===========================================================================
def build_collision_matrix(x, f, dx):
    """将碰撞算子 C[f] 表示为矩阵作用: C ≈ L · f.

    对于给定背景 f_background, 线性化碰撞算子:
    C[f] ≈ L(f_background) · f

    L 是一个五对角矩阵 (4 阶差分模板宽度 5).

    Parameters
    ----------
    x : ndarray  速度网格
    f : ndarray  背景分布 (用于计算系数)
    dx : float  网格间距

    Returns
    -------
    L : ndarray(N, N)  碰撞矩阵
    """
    N = len(x)
    A_coeff, D_coeff, M, G = compute_collision_coefficients(x, f)

    # 构造 L 矩阵
    L = np.zeros((N, N))

    for j in range(2, N - 2):
        xj = x[j]
        Dj = D_coeff[j]
        Aj = A_coeff[j]

        if xj < 1e-10:
            continue

        # C[f]_j = (2/x_j) J_j + dJ/dx |_j
        # J = D f' + A f
        # J' = D' f' + D f'' + A' f + A f'
        #    = D f'' + (D' + A) f' + A' f
        # C = D f'' + (2D/x + D' + A) f' + (2A/x + A') f

        # D' ≈ (D_{j+1} - D_{j-1}) / (2dx)
        Dp = (D_coeff[j+1] - D_coeff[j-1]) / (2.0 * dx) if j > 0 and j < N-1 else 0.0
        Ap = (A_coeff[j+1] - A_coeff[j-1]) / (2.0 * dx) if j > 0 and j < N-1 else 0.0

        # 系数
        c_d2 = Dj                                          # f'' 系数
        c_d1 = 2.0 * Dj / xj + Dp + Aj                    # f' 系数
        c_d0 = 2.0 * Aj / xj + Ap                          # f 系数

        # 4 阶差分模板
        # f'': [-1, 16, -30, 16, -1] / (12 dx²)
        d2_stencil = np.array([-1.0, 16.0, -30.0, 16.0, -1.0]) / (12.0 * dx**2)
        # f':  [-1, 8, 0, -8, 1] / (12 dx)
        d1_stencil = np.array([-1.0, 8.0, 0.0, -8.0, 1.0]) / (12.0 * dx)

        for s in range(5):
            k = j - 2 + s
            if 0 <= k < N:
                L[j, k] = (c_d2 * d2_stencil[s]
                            + c_d1 * d1_stencil[s])
        # f 的对角贡献
        L[j, j] += c_d0

    # 边界处理
    L[0, 0] = -1.0
    L[0, 1] = 1.0
    L[N-1, N-1] = -1.0
    L[N-1, N-2] = 1.0

    return L


# ===========================================================================
#  §3  时间推进求解器
# ===========================================================================
def solve_fokker_planck(x, f0, dt, n_steps, method="semi_implicit"):
    """求解 Fokker-Planck 方程的时间推进.

    Parameters
    ----------
    x : ndarray  速度网格
    f0 : ndarray  初始分布函数
    dt : float  时间步长
    n_steps : int  总步数
    method : str  "explicit", "semi_implicit", "crank_nicolson"

    Returns
    -------
    f_final : ndarray  最终分布函数
    history : dict  历史记录
    """
    N = len(x)
    dx = x[1] - x[0] if N > 1 else 1.0

    f = f0.copy()
    f_initial = f0.copy()

    # 历史记录
    history = {
        "f_snapshots": [],
        "moments": [],
        "max_values": [],
    }

    snapshot_interval = max(1, n_steps // 10)

    for step in range(n_steps):
        # 计算碰撞算子
        C, A, D, J = compute_collision_operator(x, f)

        # 时间推进
        if method == "explicit":
            # 显式 Euler: f^{n+1} = f^n + dt · C[f^n]
            f_new = f + dt * C

        elif method == "semi_implicit":
            # 半隐式: 扩散项隐式, 对流项显式
            # f^{n+1} - dt · (D ∂²f/∂x²)^{n+1} = f^n + dt · (A ∂f/∂x + ...)^n
            # 构造三对角系统

            # 显式部分
            rhs = f + dt * C  # 先用完全显式作为近似

            # 隐式修正 (只修正扩散项)
            # 构造扩散矩阵
            a_sub = np.zeros(N)
            a_diag = np.ones(N)
            a_sup = np.zeros(N)

            for j in range(1, N - 1):
                if x[j] > 1e-10:
                    theta = 0.5  # Crank-Nicolson 参数
                    coeff = -theta * dt * D[j] / dx**2
                    a_sub[j] = coeff
                    a_diag[j] = 1.0 - 2.0 * coeff
                    a_sup[j] = coeff

            # 三对角求解
            f_new = solve_tridiagonal(a_sub, a_diag, a_sup, rhs)

        elif method == "crank_nicolson":
            # 完全 Crank-Nicolson
            # (I - dt/2 · L) f^{n+1} = (I + dt/2 · L) f^n
            # 使用矩阵构建
            L = build_collision_matrix(x, f, dx)
            I = np.eye(N)
            A_lhs = I - 0.5 * dt * L
            b_rhs = (I + 0.5 * dt * L) @ f

            # 求解 (用 numpy, 因为 L 是稠密的)
            try:
                f_new = np.linalg.solve(A_lhs, b_rhs)
            except np.linalg.LinAlgError:
                f_new = f + dt * C  # 回退到显式

        else:
            raise ValueError(f"未知方法: {method}")

        # 边界条件
        f_new[0] = f_new[1]  # ∂f/∂x = 0 at v=0 (对称性)
        f_new[-1] = 0.0      # f → 0 at v=v_max

        # 保持非负
        f_new = np.maximum(f_new, 0.0)

        # 粒子数守恒修正
        n_current = np.trapz(FOUR_PI * x**2 * f_new, x)
        n_target = np.trapz(FOUR_PI * x**2 * f_initial, x)
        if n_current > 1e-30 and n_target > 1e-30:
            f_new *= n_target / n_current

        f = f_new

        # 记录
        history["max_values"].append(float(np.max(f)))
        if step % snapshot_interval == 0 or step == n_steps - 1:
            history["f_snapshots"].append(f.copy())
            # 矩
            from pwl_velocity import compute_moments
            moments = compute_moments(x, f)
            history["moments"].append(moments)

    return f, history


# ===========================================================================
#  §4  CFL 条件检查
# ===========================================================================
def check_cfl_condition(x, f, dt):
    """检查 CFL 稳定性条件.

    对于显式格式:
      Δt < 2 Δx² / (π² D_max)  (扩散 CFL)
      Δt < Δx / A_max           (对流 CFL)

    Returns
    -------
    info : dict
    """
    dx = x[1] - x[0] if len(x) > 1 else 1.0
    A, D, M, G = compute_collision_coefficients(x, f)
    D_max = np.max(D)
    A_max = np.max(A)

    dt_diffusion = 0.5 * dx**2 / max(D_max, 1e-30)
    dt_advection = dx / max(A_max, 1e-30)
    dt_max = min(dt_diffusion, dt_advection)

    return {
        "D_max": D_max,
        "A_max": A_max,
        "dx": dx,
        "dt": dt,
        "dt_max_diffusion": dt_diffusion,
        "dt_max_advection": dt_advection,
        "dt_max": dt_max,
        "cfl_diffusion": dt / max(dt_diffusion, 1e-30),
        "cfl_advection": dt / max(dt_advection, 1e-30),
        "stable_explicit": dt < dt_max,
    }
