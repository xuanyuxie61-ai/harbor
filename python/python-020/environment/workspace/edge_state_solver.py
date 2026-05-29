# -*- coding: utf-8 -*-
"""
edge_state_solver.py
量子霍尔边缘态的BVP求解与手性Luttinger液体

核心物理：
  在有限几何中，Landau能级在边界处弯曲形成边缘态。
  对于圆盘几何，边缘处的单粒子哈密顿量在极坐标下为：
      H = -(ħ²/2m*)[∂²_r + (1/r)∂_r + (1/r²)∂²_θ]
          + (1/2)m* ω_c² r² / 4 + ...

  更一般地，对于线性化色散的边缘态（手性Luttinger液体），
  低能有效哈密顿量为：
      H_edge = ħ v_F ∫ dx ψ^†(x) (-i ∂_x) ψ(x)

  其中 v_F 为费米速度，ψ(x) 为手性费米子场算符。

  在BVP（边值问题）框架下，边缘态的径向方程可以写为：
      u''(r) + (1/r) u'(r) + [k² - m*²ω_c²r²/(4ħ²) - m²/r²] u(r) = 0

  采用打靶法（Shooting Method）：
      1. 在 r=0 处施加正则边界条件 u(0)=0, u'(0)=α（猜测）
      2. 用ODE积分器将解推进到 r=R（外边界）
      3. 比较 u(R) 与目标边界条件 u(R)=0
      4. 用割线法修正 α，使 F(α) = u(R;α) → 0

本模块融合原项目：
  - 130_bvp_shooting（打靶法边值问题）
  - 343_euler（欧拉法ODE积分）
  - 101_blowup_ode（爆炸ODE的稳定性处理）
"""
import numpy as np
from utils import cyclotron_frequency

# ============================================================================
# 1. 欧拉法ODE积分（融合原项目 343_euler）
# ============================================================================

def euler_integrate(dydt, tspan, y0, n_steps):
    """
    使用Euler法求解ODE初值问题：
        dy/dt = f(t, y),   y(t0) = y0

    离散格式：
        y_{n+1} = y_n + h · f(t_n, y_n)
    其中 h = (t_stop - t0) / n_steps。

    参数:
        dydt   : callable, f(t, y) → ndarray
        tspan  : tuple, (t0, t_stop)
        y0     : ndarray, 初始条件
        n_steps: int, 步数

    返回:
        t      : ndarray, 时间网格
        y      : ndarray, 解
    """
    y0 = np.atleast_1d(y0)
    t0, t_stop = tspan
    h = (t_stop - t0) / n_steps

    t = np.zeros(n_steps + 1)
    y = np.zeros((n_steps + 1, len(y0)))

    t[0] = t0
    y[0, :] = y0

    for i in range(n_steps):
        t[i + 1] = t[i] + h
        y[i + 1, :] = y[i, :] + h * np.asarray(dydt(t[i], y[i, :]))

    return t, y


# ============================================================================
# 2. 打靶法边值问题求解（融合原项目 130_bvp_shooting）
# ============================================================================

def shooting_method_bvp(ode_rhs, a, b, ya, yb_target, alpha_guess1=0.0,
                        alpha_guess2=-1.0, max_iter=20, tol=1e-6, n_steps=1000):
    """
    打靶法求解二阶ODE边值问题：
        y'' = f(r, y, y'),   y(a) = ya,   y(b) = yb_target

    通过猜测缺失的初始斜率 α = y'(a)，将BVP转化为IVP，
    然后用割线法迭代修正 α：
        F(α) = y(b; α) - yb_target
        α_{new} = α - F(α)·(β - α)/(F(β) - F(α))

    参数:
        ode_rhs       : callable, 输入(r, [y, y']) 返回 [y', y'']
        a, b          : float, 区间端点
        ya            : float, 左边界条件 y(a)
        yb_target     : float, 右边界条件 y(b)
        alpha_guess1  : float, 第一个猜测斜率
        alpha_guess2  : float, 第二个猜测斜率
        max_iter      : int, 最大迭代数
        tol           : float, 收敛阈值
        n_steps       : int, ODE积分步数

    返回:
        r             : ndarray, 径向网格
        y             : ndarray, 解 y(r)
        converged     : bool
        n_iter        : int
    """
    def F(alpha):
        """计算给定初始斜率α时的边界残差。"""
        y0 = np.array([ya, alpha])
        r_grid, y_sol = euler_integrate(ode_rhs, (a, b), y0, n_steps)
        yb_computed = y_sol[-1, 0]
        return yb_computed - yb_target

    alpha = alpha_guess1
    f_alpha = F(alpha)

    # 记录迭代历史
    beta = None
    f_beta = None

    for iteration in range(max_iter):
        if iteration == 0:
            pass  # 使用 alpha_guess1
        elif iteration == 1:
            beta = alpha
            f_beta = f_alpha
            alpha = alpha_guess2
            f_alpha = F(alpha)
        else:
            if abs(f_beta - f_alpha) < 1e-14:
                # 割线法分母过小，使用二分法
                alpha_new = 0.5 * (alpha + beta)
            else:
                gamma = alpha - f_alpha * (beta - alpha) / (f_beta - f_alpha)
                alpha_new = gamma
            beta = alpha
            f_beta = f_alpha
            alpha = alpha_new
            f_alpha = F(alpha)

        if abs(f_alpha) < tol:
            y0 = np.array([ya, alpha])
            r_grid, y_sol = euler_integrate(ode_rhs, (a, b), y0, n_steps)
            return r_grid, y_sol[:, 0], True, iteration + 1

    # 未收敛，返回最佳结果
    y0 = np.array([ya, alpha])
    r_grid, y_sol = euler_integrate(ode_rhs, (a, b), y0, n_steps)
    return r_grid, y_sol[:, 0], False, max_iter


# ============================================================================
# 3. 边缘态径向方程
# ============================================================================

def edge_state_radial_ode(B, m_star, angular_m, E):
    """
    构建边缘态的径向ODE：
        u'' + (1/r) u' + [k² - V(r)] u = 0

    对于Landau能级边缘态，有效势为：
        V(r) = (m* ω_c r / 2ħ)² + (m/r)²

    参数:
        B         : float, 磁场强度
        m_star    : float, 有效质量
        angular_m : int, 角动量量子数
        E         : float, 能量本征值（用于寻找本征态时作为参数）

    返回:
        ode_rhs   : callable, 接受(r, [u, u']) 返回 [u', u'']
    """
    omega_c = cyclotron_frequency(B, m_star)

    def ode_rhs(r, y):
        u, up = y
        if r < 1e-10:
            # 正则化 r→0 处的行为
            r = 1e-10
        # 径向方程
        # u'' = -(1/r) u' - [k² - V(r)] u
        # 这里 k² = 2m*E/ħ²
        k_sq = 2.0 * m_star * E
        V_r = (m_star * omega_c * r / (2.0)) ** 2 + (angular_m / r) ** 2
        upp = -(1.0 / r) * up - (k_sq - V_r) * u
        return np.array([up, upp])

    return ode_rhs


# ============================================================================
# 4. 手性Luttinger液体色散
# ============================================================================

def chiral_luttinger_dispersion(k, v_F, g_factor=1.0):
    """
    手性Luttinger液体的色散关系：
        ε(k) = ħ v_F k / g
    其中 g 为Luttinger参数（g=1 对应自由费米子，g<1 对应相互作用系统）。

    参数:
        k       : ndarray, 波矢
        v_F     : float, 费米速度
        g_factor: float, Luttinger参数

    返回:
        energy  : ndarray, 色散能量
    """
    if g_factor <= 0:
        raise ValueError("Luttinger参数 g 必须为正")
    return k * v_F / g_factor


def edge_state_density_of_states(omega, v_F, L_edge, T=0.01):
    """
    边缘态的态密度（一维手性系统）：
        D(ω) = L_edge / (2π v_F)  (对 ω > 0)

    在有限温度下，考虑热展宽：
        D_T(ω) = ∫ dω' D(ω') · (1/√(2πσ²)) exp(-(ω-ω')²/(2σ²))
    """
    if v_F <= 0:
        raise ValueError("费米速度 v_F 必须为正")
    sigma = max(T, 1e-6)
    omega = np.asarray(omega, dtype=float)
    prefactor = L_edge / (2.0 * np.pi * v_F)
    # 热展宽
    dos = prefactor * 0.5 * (1.0 + np.tanh(omega / sigma))
    return dos


# ============================================================================
# 5. 爆炸ODE的稳定性处理（融合原项目 101_blowup_ode）
# ============================================================================

def blowup_ode_stabilized(t, y, blowup_threshold=1e6, stabilization=0.1):
    """
    带稳定性处理的爆炸型ODE：
        dy/dt = y² / (1 + (y/y_c)²)

    原始ODE dy/dt = y² 在有限时间内爆炸（y→∞）。
    通过引入饱和因子 1/(1+(y/y_c)²)，当 y 接近 y_c 时导数被抑制，
    保证数值积分的稳定性，用于模拟边缘态密度在临界点附近的非线性增长。

    参数:
        t               : float, 时间
        y               : float, 状态变量
        blowup_threshold: float, 爆炸阈值 y_c
        stabilization   : float, 稳定化强度

    返回:
        dydt            : float, 导数
    """
    y = float(y)
    if abs(y) > blowup_threshold:
        # 超过阈值时强制稳定
        return stabilization * blowup_threshold * np.sign(y)
    dydt = y ** 2 / (1.0 + (y / blowup_threshold) ** 2)
    return dydt


# ============================================================================
# 6. 测试接口
# ============================================================================
def test_edge_state_solver():
    """测试边缘态求解器。"""
    print("=" * 60)
    print("[edge_state_solver.py] 边缘态BVP求解测试")
    print("=" * 60)

    # 测试Euler法
    print("\n1. Euler法测试 (dy/dt = -y, y(0)=1, 精确解 y=exp(-t)):")
    def decay_ode(t, y):
        return np.array([-y[0]])
    t, y = euler_integrate(decay_ode, (0.0, 2.0), np.array([1.0]), n_steps=200)
    exact = np.exp(-t)
    err = np.max(np.abs(y[:, 0] - exact))
    print(f"   最大误差: {err:.6e}")

    # 测试打靶法
    print("\n2. 打靶法测试 (u'' + u = 0, u(0)=0, u(π)=0, 精确解 u=sin(r)):")
    def harmonic_ode(r, y):
        u, up = y
        return np.array([up, -u])
    r_grid, u_sol, conv, nit = shooting_method_bvp(
        harmonic_ode, a=0.0, b=np.pi, ya=0.0, yb_target=0.0,
        alpha_guess1=0.5, alpha_guess2=1.5, max_iter=20, tol=1e-5, n_steps=500
    )
    u_exact = np.sin(r_grid)
    err = np.max(np.abs(u_sol - u_exact))
    print(f"   收敛: {conv}, 迭代次数: {nit}")
    print(f"   最大误差: {err:.6e}")

    # 测试边缘态径向方程
    print("\n3. 边缘态径向方程测试:")
    B = 10.0
    m_star = 1.0
    angular_m = 1
    E = 5.0
    ode_rhs = edge_state_radial_ode(B, m_star, angular_m, E)
    r_grid, u_sol, conv, nit = shooting_method_bvp(
        ode_rhs, a=1e-3, b=3.0, ya=0.0, yb_target=0.0,
        alpha_guess1=0.1, alpha_guess2=1.0, max_iter=15, tol=1e-4, n_steps=800
    )
    print(f"   收敛: {conv}, 迭代次数: {nit}")
    print(f"   解的范数: {np.linalg.norm(u_sol):.4f}")

    # 测试手性Luttinger色散
    print("\n4. 手性Luttinger液体色散测试:")
    k = np.linspace(0.0, 5.0, 50)
    v_F = 1.0
    for g in [1.0, 0.5, 0.3]:
        eps = chiral_luttinger_dispersion(k[1:], v_F, g)
        print(f"   g={g}: ε(k=5) = {eps[-1]:.4f}")

    # 测试爆炸ODE稳定化
    print("\n5. 爆炸ODE稳定化测试:")
    t, y = euler_integrate(
        lambda t, y: np.array([blowup_ode_stabilized(t, y[0])]),
        (0.0, 2.0), np.array([1.0]), n_steps=500
    )
    print(f"   y(0)={y[0,0]:.4f}, y(2)={y[-1,0]:.4f} (无稳定化时会爆炸)")

    print("\n[edge_state_solver.py] 测试完成。\n")


if __name__ == "__main__":
    test_edge_state_solver()
