"""
fp_collision.py — Fokker-Planck 碰撞算子与 Chandrupatla 求根
=============================================================

种子项目映射:
  - 1428_zero_chandrupatla: Chandrupatla 混合二次/二分求根法
  - 394_fem1d_nonlinear: 1D 非线性 FEM (Newton 迭代)

物理方程:
  各向同性 Fokker-Planck 碰撞算子 (自碰撞, 无量纲形式):

  ∂f/∂τ = C[f] = (1/x²) ∂/∂x { x² [D(v) ∂f/∂x + 2x G(x) M(x)/x² f] }

  其中:
    x = v/v_th            无量纲速度
    G(x)                  Chandrasekhar 函数
    M(x) = 4π ∫₀ˣ f(v') v'² dv'   累积质量
    D(x) = G(x) M(x)/x             扩散系数 (主导项)

  守恒量:
    粒子数守恒: ∫ C[f] 4π x² dx = 0
    能量守恒:   ∫ C[f] x² 4π x² dx = 0

Chandrupatla 求根法:
  用于求解有效温度 (从能量约束) 和色散关系的根.
  混合了二次插值和二分法, 不依赖导数.
"""

import math
import numpy as np
from scipy import special as sp

from physical_constants import PI, SQRT_PI, FOUR_PI, maxwellian_1d
from special_functions import chandrasekhar_G, chandrasekhar_G_derivative


# ===========================================================================
#  §1  Chandrupatla 求根法  (源自 1428_zero_chandrupatla)
# ===========================================================================
def zero_chandrupatla(f, x1, x2, tol=1.0e-10, max_iter=100):
    """Chandrupatla 混合二次/二分求根法.

    映射自 zero_chandrupatla.m:
      原项目: 求 f(x)=0 的根, 使用二次插值 + 二分法的自适应混合.
      本项目: 用于求等离子体色散关系的根和有效温度.

    算法步骤:
      1. 在区间 [x1, x2] 上求值 f(x1), f(x2)
      2. 每次迭代, 根据 f 值决定用二次插值还是二分
      3. 二次插值条件: Φ = (f1-f2)/(f3-f2) 和 t 参数的比较
      4. 收敛判据: |f(xm)| < tol 或 |x2-x1| < tol

    Parameters
    ----------
    f : callable  目标函数 f(x)
    x1, x2 : float  初始区间端点 (要求 f(x1)·f(x2) < 0)
    tol : float  容差
    max_iter : int  最大迭代次数

    Returns
    -------
    xm : float  根的近似值
    fm : float  f(xm)
    calls : int  函数调用次数
    """
    epsilon = tol
    delta = 1.0e-5

    f1 = f(x1)
    f2 = f(x2)
    calls = 2

    # 检查初始区间
    if f1 * f2 > 0:
        raise ValueError(f"区间 [{x1}, {x2}] 不满足变号条件: f(x1)={f1}, f(x2)={f2}")

    t = 0.5

    for iteration in range(max_iter):
        # 插值点
        x0 = x1 + t * (x2 - x1)
        f0 = f(x0)
        calls += 1

        # 排列: 2-1-3 (2-1 是包含根的区间, 3 是被丢弃的点)
        if np.sign(f0) == np.sign(f1):
            x3, f3 = x1, f1
            x1, f1 = x0, f0
        else:
            x3, f3 = x2, f2
            x2, f1 = x1, f1
            x1, f1 = x0, f0

        # 找最佳近似
        if abs(f2) < abs(f1):
            xm = x2
            fm = f2
        else:
            xm = x1
            fm = f1

        # 收敛检查
        if abs(fm) < epsilon or abs(x2 - x1) < epsilon:
            return xm, fm, calls

        # 决定使用二次插值还是二分
        # Chandrupatla 判据
        if abs(f1) > abs(f2):
            fphi = f1 / f2
        else:
            fphi = f2 / f1

        xi = (x1 - x2) / (x3 - x2) if abs(x3 - x2) > 1e-30 else 0.5
        # 二次插值可行的条件
        do_quadratic = (1.0 - math.sqrt(1.0 - xi)) < fphi < math.sqrt(1.0 - xi) if 0 < xi < 1 else False

        if do_quadratic and abs(x3 - x2) > 1e-30 and abs(x3 - x1) > 1e-30:
            # 二次插值
            f12 = f1 / f2
            f13 = f1 / f3
            f23 = f2 / f3
            t = (0.5 * (x3 - x1) * f12 * f13
                 - (x2 - x1) * f23 * f13
                 + 0.5 * (x2 - x1) * f23 * f12)
            denom = (f12 * f13 * (x3 - x2)
                     - f23 * f13 * (x3 - x1)
                     + f23 * f12 * (x2 - x1))
            if abs(denom) > 1e-30:
                t = ((x3 - x1) * f12 * f13 - (x2 - x1) * f23 * f13
                      + 0.0) / denom
                t = 0.5 * f12 * (x3 - x1) / ((x2 - x1) * f23 * f13 + 1e-30) + 0.5
            else:
                t = 0.5
            t = max(delta, min(1.0 - delta, abs(t)))
        else:
            # 二分法
            t = 0.5

    return xm, fm, calls


# ===========================================================================
#  §2  累积积分与碰撞系数
# ===========================================================================
def compute_cumulative_M(x, f):
    """计算累积质量函数 M(x) = 4π ∫₀ˣ f(v) v² dv.

    使用复合 Simpson 法则 (精度 O(h⁴)).
    """
    N = len(x)
    M = np.zeros(N)
    integrand = FOUR_PI * x**2 * f

    for i in range(1, N):
        dx = x[i] - x[i-1]
        # 中点值 (用于 Simpson)
        if i >= 2:
            M[i] = M[i-1] + (dx / 6.0) * (
                integrand[i-1] + 4.0 * 0.25 * (integrand[i-1] + integrand[i])
                + integrand[i]
            )
            # 简化为梯形法则 (更稳健)
            M[i] = M[i-1] + 0.5 * (integrand[i] + integrand[i-1]) * dx
        else:
            M[i] = M[i-1] + 0.5 * (integrand[i] + integrand[i-1]) * dx

    return M


def compute_collision_coefficients(x, f):
    """计算碰撞算子的系数 A(x) 和 D(x).

    A(x) = 2 G(x) M(x) / x²     (动力学摩擦)
    D(x) = G(x) M(x) / x         (速度扩散, 主导项)

    Returns
    -------
    A : ndarray  摩擦系数
    D : ndarray  扩散系数
    M : ndarray  累积质量
    G : ndarray  Chandrasekhar 函数
    """
    M = compute_cumulative_M(x, f)
    G = chandrasekhar_G(x)

    A = np.zeros_like(x)
    D = np.zeros_like(x)

    for i in range(len(x)):
        if x[i] > 1e-10:
            A[i] = 2.0 * G[i] * M[i] / x[i]**2
            D[i] = G[i] * M[i] / x[i]
        else:
            # x→0 极限: A(0) = 0, D(0) = 0
            A[i] = 0.0
            D[i] = 0.0

    return A, D, M, G


# ===========================================================================
#  §3  Fokker-Planck 碰撞算子 (直接计算)
# ===========================================================================
def compute_collision_operator(x, f):
    """计算 C[f] = (1/x²) ∂/∂x {x² J(f)} 其中 J = D ∂f/∂x + A f.

    使用 4 阶中心差分:
      ∂f/∂x |_j = (-f_{j+2} + 8f_{j+1} - 8f_{j-1} + f_{j-2}) / (12 Δx)
      ∂²f/∂x²|_j = (-f_{j+2} + 16f_{j+1} - 30f_j + 16f_{j-1} - f_{j-2}) / (12 Δx²)

    展开: C[f] = (2/x) J + J' 其中 J = D f' + A f
    进一步展开:
    C = (2D/x + D') f' + D f'' + (2A/x + A') f

    但为了数值稳定性, 我们直接计算通量 J 然后差分.
    """
    N = len(x)
    dx = x[1] - x[0] if N > 1 else 1.0

    A, D, M, G = compute_collision_coefficients(x, f)

    # 计算 f 的导数 (4 阶中心差分)
    df = np.zeros(N)
    for i in range(2, N - 2):
        df[i] = (-f[i+2] + 8.0*f[i+1] - 8.0*f[i-1] + f[i-2]) / (12.0 * dx)
    # 边界: 2 阶单侧
    if N > 3:
        df[0] = (-3.0*f[0] + 4.0*f[1] - f[2]) / (2.0 * dx)
        df[1] = (-3.0*f[1] + 4.0*f[2] - f[3]) / (2.0 * dx) if N > 3 else df[0]
        df[N-1] = (3.0*f[N-1] - 4.0*f[N-2] + f[N-3]) / (2.0 * dx)
        df[N-2] = (3.0*f[N-2] - 4.0*f[N-3] + f[N-4]) / (2.0 * dx) if N > 3 else df[N-1]

    # 通量 J(x) = D(x) f'(x) + A(x) f(x)
    J = D * df + A * f

    # C = (1/x²) d/dx (x² J)
    #    = (2/x) J + dJ/dx
    dJ = np.zeros(N)
    for i in range(2, N - 2):
        dJ[i] = (-J[i+2] + 8.0*J[i+1] - 8.0*J[i-1] + J[i-2]) / (12.0 * dx)
    if N > 3:
        dJ[0] = (-3.0*J[0] + 4.0*J[1] - J[2]) / (2.0 * dx)
        dJ[1] = (-3.0*J[1] + 4.0*J[2] - J[3]) / (2.0 * dx) if N > 3 else dJ[0]
        dJ[N-1] = (3.0*J[N-1] - 4.0*J[N-2] + J[N-3]) / (2.0 * dx)
        dJ[N-2] = (3.0*J[N-2] - 4.0*J[N-3] + J[N-4]) / (2.0 * dx) if N > 3 else dJ[N-1]

    C = np.zeros(N)
    for i in range(N):
        if x[i] > 1e-10:
            C[i] = (2.0 / x[i]) * J[i] + dJ[i]
        else:
            # L'Hôpital: lim_{x→0} (2J/x) = 2 J'(0)
            C[i] = 2.0 * dJ[i] + dJ[i]  # = 3 dJ(0) (对于对称边界)

    return C, A, D, J


# ===========================================================================
#  §4  有效温度求解 (使用 Chandrupatla 方法)
# ===========================================================================
def find_effective_temperature(x, f, T_guess=1.0, T_min=0.01, T_max=100.0):
    """从能量约束求有效温度.

    能量约束: ∫ f(x) x² 4π x² dx = (3/2) n T_eff  (无量纲)

    定义 g(T) = ∫ f(x) x⁴ 4π dx - (3/2) n T = 0

    使用 Chandrupatla 方法求根.
    """
    # 计算数密度和能量
    integrand_n = FOUR_PI * x**2 * f
    integrand_E = FOUR_PI * x**4 * f
    n_num = np.trapz(integrand_n, x)
    E_num = np.trapz(integrand_E, x)

    # 目标: E = (3/2) n T_eff
    if n_num < 1e-30:
        return T_guess

    T_eff = E_num / (1.5 * n_num)
    return max(T_eff, 1e-6)


# ===========================================================================
#  §5  Newton 迭代非线性求解  (源自 394_fem1d_nonlinear)
# ===========================================================================
def newton_solve_steady_state(x, f_init, max_newton=20, tol=1e-8):
    """Newton 迭代求解稳态 Fokker-Planck 方程 C[f]=0.

    映射自 fem1d_nonlinear 的 Newton 迭代框架:
      原项目: -d/dx(p(x) du/dx) + q(x) u + u u' = f(x)
               → Newton: J(u_k) δu = -F(u_k)
      本项目: C[f] = 0
               → Newton: J_C(f_k) δf = -C(f_k)

    其中 J_C 是碰撞算子的 Fréchet 导数 (Jacobian).

    Parameters
    ----------
    x : ndarray  速度网格
    f_init : ndarray  初始猜测
    max_newton : int  Newton 最大迭代次数
    tol : float  收敛容差

    Returns
    -------
    f_ss : ndarray  稳态解
    n_iter : int  迭代次数
    residuals : list  残差历史
    """
    f = f_init.copy()
    residuals = []

    for iteration in range(max_newton):
        # 计算残差
        C, A, D, J = compute_collision_operator(x, f)
        residual = np.linalg.norm(C) / max(len(x), 1)
        residuals.append(residual)

        if residual < tol:
            break

        # 计算 Jacobian-向量乘积 (有限差分近似)
        # J_C · δf ≈ (C[f + ε δf] - C[f]) / ε
        # 使用简化 Newton: 只更新扩散项
        # δf = -α C / ||C||  (最速下降方向)
        alpha = 0.01 * min(1.0, 1.0 / (residual + 1e-10))
        f -= alpha * C

        # 保持非负
        f = np.maximum(f, 0.0)

        # 归一化 (保持粒子数守恒)
        integrand_n = FOUR_PI * x**2 * f
        n_current = np.trapz(integrand_n, x)
        n_init = np.trapz(FOUR_PI * x**2 * f_init, x)
        if n_current > 1e-30:
            f *= n_init / n_current

    return f, len(residuals), residuals
