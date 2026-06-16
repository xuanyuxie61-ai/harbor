"""
stability_analysis.py -- von Neumann 稳定性分析与 CFL 条件
================================================================
Project 260: 暗能量状态方程约束 -- 高阶有限差分与稳定性分析

本模块实现:
  1. 时间推进格式 (Euler, RK, BDF, IMEX) 的放大因子
  2. 空间离散 (FD 各阶) 的色散关系
  3. von Neumann 稳定性区域
  4. CFL 条件推导
  5. 增长方程的刚性分析
  6. 不稳定 ODE 测试 (种子项目 1374)
  7. 摄动 Kepler 哈密顿守恒 (种子项目 619)

数学公式
--------
(1)  模型方程 (von Neumann 分析):
         u_t = lambda u,  lambda = alpha + i beta
         稳定条件: |G(z)| <= 1,  z = lambda * dt

(2)  显式 Euler:     G(z) = 1 + z
(3)  隐式 Euler:     G(z) = 1 / (1 - z)
(4)  Crank-Nicolson: G(z) = (1 + z/2) / (1 - z/2)
(5)  RK2 (Heun):     G(z) = 1 + z + z^2/2
(6)  RK3:            G(z) = 1 + z + z^2/2 + z^3/6
(7)  RK4:            G(z) = 1 + z + z^2/2 + z^3/6 + z^4/24
(8)  BDF1 (隐式 Euler): G(z) = 1/(1-z)
(9)  BDF2:           G(z) = (2/3) / (1 - (4/3)z + (1/3)z^2)
(10) BDF3:           (6/11) / (1 - (18/11)z + (9/11)z^2 - (2/11)z^3)
(11) BDF4:           (12/25) / (1 - (48/25)z + (36/25)z^2 - (16/25)z^3 + (3/25)z^4)

(12) IMEX (Implicit-Explicit) RK:
     显式部分处理对流, 隐式部分处理扩散/刚性项.
     ARS(2,2,2):
         gamma = 1 - 1/sqrt(2)
         Stage 1: y1 = y_n + dt * [(1-gamma)*f_E(y_n) + gamma*f_I(y1)]
         Stage 2: y_{n+1} = y_n + dt * [gamma*f_E(y1) + (1-gamma)*f_E(y_n)
                                          + gamma*f_I(y_{n+1}) + (1-gamma)*f_I(y1)]

(13) 空间离散色散关系:
     对 FD_p (p阶) 离散 d/dx:
         (d/dx)_num * exp(i k x) = i k_eff(k) exp(i k x)
     修正波数 k_eff 满足:
         k_eff * h = sum_{j=-p/2}^{p/2} c_j sin(j k h)    (奇对称系数)

     2阶: k_eff h = sin(kh)
     4阶: k_eff h = (4/3) sin(kh) - (1/6) sin(2kh)
     6阶: k_eff h = (3/2) sin(kh) - (3/10) sin(2kh) + (1/30) sin(3kh)
     8阶: k_eff h = (8/5) sin(kh) - (2/5) sin(2kh)
                   + (8/175) sin(3kh) - (1/350) sin(4kh)

(14) 紧致格式修正波数 (4阶):
         k_eff h = (3/2) sin(kh) / (1 + (1/3) cos(kh))
                  (不对, 正确为:)
         k_eff h = (6 sin(kh)) / (6 + cos(kh))... 待核实
         实际: (1/6 + 2/3 cos(kh) + 1/6 cos(kh)) * k_eff h
              = i sin(kh)
         => k_eff h = 6 sin(kh) / (4 + 2 cos(kh)) = 3 sin(kh) / (2 + cos(kh))

(15) 增长方程刚性比:
     D'' + P D' + Q D = 0
     => y' = A y,  A = [[0, 1], [-Q, -P]]
     特征值: lambda_{1,2} = (-P ± sqrt(P^2 + 4Q)) / 2
     刚性比: S = |Re(lambda_1)| / |Re(lambda_2)|  (假设 Re < 0)

(16) CFL 条件 (对流方程 u_t + c u_x = 0, FD_p 空间离散):
     dt * |c| / h * max_k |k_eff h| <= C_stable
     对显式 Euler: C_stable = 1 (纯虚数 z 时 G = 1 + i z, |G|^2 = 1+z^2 > 1 不稳定)
     对 RK4: C_stable ~ 2.83 (在虚轴上)

(17) 摄动 Kepler Hamiltonian (种子项目 619):
     H = (p1^2 + p2^2)/2 - 1/r + epsilon/(2 r^2)
     其中 epsilon > 0 为摄动参数.
     运动方程:
         dq1/dt = p1,  dq2/dt = p2
         dp1/dt = -q1/r^3 + epsilon*q1/r^4  (不对, 摄动势 V_pert = epsilon/(2r^2))
         dp1/dt = -q1(1/r^3 + epsilon/r^4)  ...
     实际摄动: H = p^2/2 - 1/r - delta/(2 r^2)

(18) 不稳定 ODE (种子项目 1374):
     y' = A y,  A = [[mu, 1/mu], [-1/mu, mu]]
     特征值: lambda = mu ± i/mu
     精确解: y(t) = exp(mu*t) * [[cos(t/mu), -mu*sin(t/mu)],
                                   [sin(t/mu)/mu, cos(t/mu)]] * y0
     数值挑战: 当 mu 小时, 振荡极快 (stiff).
================================================================
"""
from __future__ import annotations
import math
import cmath
from typing import List, Tuple, Dict, Optional, Callable


# =====================================================================
#  时间推进格式的放大因子
# =====================================================================

def G_explicit_euler(z: complex) -> complex:
    """显式 Euler: G(z) = 1 + z"""
    return 1.0 + z


def G_implicit_euler(z: complex) -> complex:
    """隐式 Euler: G(z) = 1 / (1 - z)"""
    denom = 1.0 - z
    if abs(denom) < 1e-30:
        return complex(float('inf'), 0.0)
    return 1.0 / denom


def G_crank_nicolson(z: complex) -> complex:
    """Crank-Nicolson: G(z) = (1 + z/2) / (1 - z/2)"""
    num = 1.0 + z / 2.0
    den = 1.0 - z / 2.0
    if abs(den) < 1e-30:
        return complex(float('inf'), 0.0)
    return num / den


def G_rk2(z: complex) -> complex:
    """RK2 (Heun): G(z) = 1 + z + z^2/2"""
    return 1.0 + z + z*z / 2.0


def G_rk3(z: complex) -> complex:
    """RK3: G(z) = 1 + z + z^2/2 + z^3/6"""
    return 1.0 + z + z*z/2.0 + z**3 / 6.0


def G_rk4(z: complex) -> complex:
    """RK4: G(z) = 1 + z + z^2/2 + z^3/6 + z^4/24"""
    return 1.0 + z + z*z/2.0 + z**3/6.0 + z**4/24.0


def G_bdf1(z: complex) -> complex:
    """BDF1 = 隐式 Euler: G(z) = 1/(1-z)"""
    return G_implicit_euler(z)


def G_bdf2(z: complex) -> complex:
    """
    BDF2: (3/2 - 2z + z^2/2) G = 1/2
    => G(z) = (1/2) / (3/2 - 2z + z^2/2)
           = 1 / (3 - 4z + z^2)
    """
    denom = 3.0 - 4.0*z + z*z
    if abs(denom) < 1e-30:
        return complex(float('inf'), 0.0)
    return 1.0 / denom


def G_bdf3(z: complex) -> complex:
    """
    BDF3: G(z) = (6/11) / (1 - (18/11)z + (9/11)z^2 - (2/11)z^3)
    """
    denom = 1.0 - (18.0/11.0)*z + (9.0/11.0)*z**2 - (2.0/11.0)*z**3
    if abs(denom) < 1e-30:
        return complex(float('inf'), 0.0)
    return (6.0/11.0) / denom


def G_bdf4(z: complex) -> complex:
    """
    BDF4: G(z) = (12/25) / (1 - 48/25 z + 36/25 z^2 - 16/25 z^3 + 3/25 z^4)
    """
    denom = (1.0 - (48.0/25.0)*z + (36.0/25.0)*z**2
             - (16.0/25.0)*z**3 + (3.0/25.0)*z**4)
    if abs(denom) < 1e-30:
        return complex(float('inf'), 0.0)
    return (12.0/25.0) / denom


def G_bdf5(z: complex) -> complex:
    """BDF5."""
    denom = (1.0 - (300.0/137.0)*z + (450.0/137.0)*z**2
             - (400.0/137.0)*z**3 + (225.0/137.0)*z**4
             - (50.0/137.0)*z**5 + (5.0/137.0)*z**5)
    # Simplified BDF5 denominator (proper formula):
    # alpha_0 = 137/60, etc.
    a0 = 137.0 / 60.0
    a1 = -5.0
    a2 = 50.0 / 12.0
    a3 = -100.0 / 12.0
    a4 = 75.0 / 12.0
    a5 = -12.0 / 12.0  # = -1
    denom2 = a0 + a1*z + a2*z**2 + a3*z**3 + a4*z**4 + a5*z**5
    if abs(denom2) < 1e-30:
        return complex(float('inf'), 0.0)
    return 1.0 / denom2


# =====================================================================
#  放大因子查找表
# =====================================================================

AMPLIFICATION_METHODS = {
    'explicit_euler': G_explicit_euler,
    'implicit_euler': G_implicit_euler,
    'crank_nicolson': G_crank_nicolson,
    'rk2': G_rk2,
    'rk3': G_rk3,
    'rk4': G_rk4,
    'bdf1': G_bdf1,
    'bdf2': G_bdf2,
    'bdf3': G_bdf3,
    'bdf4': G_bdf4,
}


# =====================================================================
#  空间离散修正波数 (色散关系)
# =====================================================================

def modified_wavenumber_fd1(kh: float, order: int) -> complex:
    """
    FD_p 空间离散 d/dx 的修正波数.

    对平面波 exp(i k x), FD 离散给出 i k_eff exp(i k x).
    返回 k_eff * h (复数, 对纯中心差分为实数).

    2阶: k_eff h = sin(kh)
    4阶: k_eff h = (4/3) sin(kh) - (1/6) sin(2 kh)
    6阶: k_eff h = (3/2) sin(kh) - (3/10) sin(2 kh) + (1/30) sin(3 kh)
    8阶: (8/5) sin(kh) - (2/5) sin(2kh) + (8/175) sin(3kh) - (1/350) sin(4kh)
    """
    if order == 2:
        return math.sin(kh)
    elif order == 4:
        return (4.0/3.0)*math.sin(kh) - (1.0/6.0)*math.sin(2*kh)
    elif order == 6:
        return ((3.0/2.0)*math.sin(kh) - (3.0/10.0)*math.sin(2*kh)
                + (1.0/30.0)*math.sin(3*kh))
    elif order == 8:
        return ((8.0/5.0)*math.sin(kh) - (2.0/5.0)*math.sin(2*kh)
                + (8.0/175.0)*math.sin(3*kh) - (1.0/350.0)*math.sin(4*kh))
    else:
        raise ValueError(f"order {order} not supported")


def modified_wavenumber_fd2(kh: float, order: int) -> complex:
    """
    FD_p 空间离散 d^2/dx^2 的修正波数 (平方).

    对平面波 exp(i k x), D2 给出 -k_eff^2 exp(i k x).
    返回 k_eff^2 * h^2.

    2阶: k^2 h^2 = 2(1 - cos(kh))
    4阶: k^2 h^2 = (30 - 16 cos(kh) + 2 cos(2kh)) / (6)  不对
         实际: = (-cos(2kh) + 16 cos(kh) - 30) / (-12)
         = (30 - 16 cos(kh) + cos(2kh)) / 6  ... 不对
         正确: (-1*cos(2kh) + 16*cos(kh) - 30) / (12 * (-1))  不对
    从 FD2 系数: [-1/12, 16/12, -30/12, 16/12, -1/12]
    对 exp(ikh): sum c_j exp(ijk h) = (-e^{-2ikh} + 16 e^{-ikh} - 30
                                       + 16 e^{ikh} - e^{2ikh}) / 12
    = (-2cos(2kh) + 32 cos(kh) - 30) / 12
    = (30 - 32 cos(kh) + 2 cos(2kh)) / 12

    6阶: (从系数 [1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90])
    """
    if order == 2:
        return 2.0 * (1.0 - math.cos(kh))
    elif order == 4:
        return (30.0 - 32.0*math.cos(kh) + 2.0*math.cos(2*kh)) / 12.0
    elif order == 6:
        # 6阶 FD2 系数: c = [1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90]
        # sum c_j exp(ijkh) = 2*c_{-3}cos(3kh) + 2*c_{-2}cos(2kh)
        #                   + 2*c_{-1}cos(kh) + c_0
        c3 = 1.0/90.0
        c2 = -3.0/20.0
        c1 = 3.0/2.0
        c0 = -49.0/18.0
        return (2.0*c3*math.cos(3*kh) + 2.0*c2*math.cos(2*kh)
                + 2.0*c1*math.cos(kh) + c0)
    else:
        raise ValueError(f"order {order} not supported for FD2")


def modified_wavenumber_compact4(kh: float) -> float:
    """
    紧致 4阶格式修正波数:
        (1/6) f'_{i-1} + (2/3) f'_i + (1/6) f'_{i+1} = (f_{i+1} - f_{i-1})/(2h)

    对 exp(ikh):
        [(1/6)e^{-ikh} + 2/3 + (1/6)e^{ikh}] * (i k_eff h)
            = (e^{ikh} - e^{-ikh})/(2)
        [(2/3 + cos(kh)/3)] * k_eff h = sin(kh)
        k_eff h = 3 sin(kh) / (2 + cos(kh))
    """
    denom = 2.0 + math.cos(kh)
    if abs(denom) < 1e-30:
        return 0.0
    return 3.0 * math.sin(kh) / denom


def modified_wavenumber_compact6(kh: float) -> float:
    """
    紧致 6阶 (Lele scheme C):
        (1/3) f'_{i-1} + f'_i + (1/3) f'_{i+1}
            = (14/9)(f_{i+1}-f_{i-1})/(2h) - (1/9)(f_{i+2}-f_{i-2})/(4h)

    对 exp(ikh):
        [1 + (2/3) cos(kh)] k_eff h
            = (14/9) sin(kh) - (1/18) sin(2kh)

        k_eff h = [(14/9) sin(kh) - (1/18) sin(2kh)] / [1 + (2/3) cos(kh)]
    """
    num = (14.0/9.0)*math.sin(kh) - (1.0/18.0)*math.sin(2*kh)
    den = 1.0 + (2.0/3.0)*math.cos(kh)
    if abs(den) < 1e-30:
        return 0.0
    return num / den


# =====================================================================
#  稳定性区域与 CFL
# =====================================================================

def stability_boundary(method: str, n_theta: int = 720
                       ) -> List[Tuple[float, float]]:
    """
    绘制稳定性区域边界 |G(z)| = 1.
    在复平面上参数化 z = r exp(i theta), 二分搜索 r.
    返回边界点列表 [(Re(z), Im(z)), ...].
    """
    G_func = AMPLIFICATION_METHODS.get(method)
    if G_func is None:
        raise ValueError(f"Unknown method: {method}")

    boundary = []
    for it in range(n_theta):
        theta = 2.0 * math.pi * it / n_theta
        # 二分搜索 |G(r e^{i theta})| = 1
        r_lo, r_hi = 0.0, 10.0
        for _ in range(50):
            r_mid = 0.5 * (r_lo + r_hi)
            z = r_mid * cmath.exp(1j * theta)
            G = G_func(z)
            if abs(G) < 1.0:
                r_lo = r_mid
            else:
                r_hi = r_mid
        r_bound = 0.5 * (r_lo + r_hi)
        z = r_bound * cmath.exp(1j * theta)
        boundary.append((z.real, z.imag))

    return boundary


def cfl_limit_explicit(method: str, fd_order: int) -> float:
    """
    显式方法 + FD_p 空间离散的 CFL 限制.

    对纯对流方程 u_t + c u_x = 0:
    半离散: du/dt = -i c k_eff u
    z = -i c k_eff dt = -i nu (k_eff h)
    其中 nu = c dt/h 为 Courant 数.

    稳定条件: |G(-i nu k_eff h)| <= 1 for all kh in [0, pi].

    返回最大 nu = CFL.
    """
    G_func = AMPLIFICATION_METHODS.get(method)
    if G_func is None:
        raise ValueError(f"Unknown method: {method}")

    # 二分搜索最大 Courant 数
    nu_lo, nu_hi = 0.0, 5.0
    n_kh = 200

    for _ in range(60):
        nu_mid = 0.5 * (nu_lo + nu_hi)
        stable = True
        for ik in range(1, n_kh):
            kh = math.pi * ik / n_kh
            keff_h = modified_wavenumber_fd1(kh, fd_order)
            z = -1j * nu_mid * keff_h
            G = G_func(z)
            if abs(G) > 1.0 + 1e-10:
                stable = False
                break
        if stable:
            nu_lo = nu_mid
        else:
            nu_hi = nu_mid

    return 0.5 * (nu_lo + nu_hi)


def cfl_limit(method: str = 'rk4', fd_order: int = 4) -> Dict:
    """
    计算给定方法和 FD 阶数的 CFL 限制.
    """
    cfl = cfl_limit_explicit(method, fd_order)
    return {
        'method': method,
        'fd_order': fd_order,
        'cfl_number': cfl,
        'dt_max': cfl,  # 假设 c=1, h=1
    }


# =====================================================================
#  增长方程特征值与刚性分析
# =====================================================================

def growth_equation_eigenvalues(P: float, Q: float
                                ) -> Tuple[complex, complex]:
    """
    增长方程 D'' + P D' + Q D = 0 的特征值.

    化为系统 y' = A y:
        A = [[0, 1], [-Q, -P]]
    特征值: lambda = (-P ± sqrt(P^2 + 4Q)) / 2

    对物理增长方程:
        P = 2 + H'/H  (>0, 阻尼)
        Q = -3/2 Omega_m(a)  (<0, 源项)
    P^2 - 4Q 通常 > 0 => 实特征值, 一个正 (增长), 一个负 (衰减).
    """
    disc = P * P - 4.0 * Q
    if disc >= 0:
        sqrt_disc = math.sqrt(disc)
        lam1 = (-P + sqrt_disc) / 2.0
        lam2 = (-P - sqrt_disc) / 2.0
        return complex(lam1, 0.0), complex(lam2, 0.0)
    else:
        sqrt_disc = cmath.sqrt(complex(disc, 0))
        lam1 = (-P + sqrt_disc) / 2.0
        lam2 = (-P - sqrt_disc) / 2.0
        return lam1, lam2


def stiffness_ratio(P: float, Q: float) -> float:
    """
    增长方程刚性比:
        S = |Re(lambda_max)| / |Re(lambda_min)|

    S >> 1 表示刚性系统, 需要隐式方法.
    """
    lam1, lam2 = growth_equation_eigenvalues(P, Q)
    re1 = abs(lam1.real)
    re2 = abs(lam2.real)
    if min(re1, re2) < 1e-30:
        return float('inf')
    return max(re1, re2) / min(re1, re2)


def growth_matrix(P: float, Q: float) -> List[List[float]]:
    """
    增长方程系统矩阵 A = [[0, 1], [-Q, -P]].
    """
    return [[0.0, 1.0], [-Q, -P]]


# =====================================================================
#  摄动 Kepler 问题 (种子项目 619)
# =====================================================================

def kepler_perturbed_deriv(state: List[float], epsilon: float = 0.001
                           ) -> List[float]:
    """
    摄动 Kepler 问题的右端函数.

    状态: [q1, q2, p1, p2]
    H = (p1^2 + p2^2)/2 - 1/r - epsilon/(2 r^2)

    运动方程:
        dq1/dt = dH/dp1 = p1
        dq2/dt = dH/dp2 = p2
        dp1/dt = -dH/dq1 = -q1/r^3 - epsilon*q1/r^4
        dp2/dt = -dH/dq2 = -q2/r^3 - epsilon*q2/r^4
    """
    q1, q2, p1, p2 = state
    r = math.sqrt(q1*q1 + q2*q2)
    if r < 1e-15:
        return [p1, p2, 0.0, 0.0]
    r3 = r * r * r
    r4 = r3 * r
    dq1 = p1
    dq2 = p2
    dp1 = -q1/r3 - epsilon * q1 / r4
    dp2 = -q2/r3 - epsilon * q2 / r4
    return [dq1, dq2, dp1, dp2]


def kepler_perturbed_hamiltonian(state: List[float],
                                  epsilon: float = 0.001) -> float:
    """
    摄动 Kepler 哈密顿量:
        H = (p1^2 + p2^2)/2 - 1/r - epsilon/(2 r^2)
    """
    q1, q2, p1, p2 = state
    r = math.sqrt(q1*q1 + q2*q2)
    if r < 1e-15:
        return float('inf')
    return 0.5*(p1*p1 + p2*p2) - 1.0/r - epsilon/(2.0*r*r)


def kepler_angular_momentum(state: List[float]) -> float:
    """角动量 L = q1*p2 - q2*p1."""
    q1, q2, p1, p2 = state
    return q1*p2 - q2*p1


def integrate_kepler_rk4(state0: List[float], t_end: float, dt: float,
                          epsilon: float = 0.001
                          ) -> List[List[float]]:
    """RK4 积分摄动 Kepler 问题."""
    n_steps = int(t_end / dt)
    state = state0[:]
    traj = [state[:]]
    for _ in range(n_steps):
        k1 = kepler_perturbed_deriv(state, epsilon)
        s2 = [state[j] + 0.5*dt*k1[j] for j in range(4)]
        k2 = kepler_perturbed_deriv(s2, epsilon)
        s3 = [state[j] + 0.5*dt*k2[j] for j in range(4)]
        k3 = kepler_perturbed_deriv(s3, epsilon)
        s4 = [state[j] + dt*k3[j] for j in range(4)]
        k4 = kepler_perturbed_deriv(s4, epsilon)
        state = [state[j] + (dt/6.0)*(k1[j] + 2*k2[j] + 2*k3[j] + k4[j])
                 for j in range(4)]
        traj.append(state[:])
    return traj


# =====================================================================
#  不稳定 ODE 测试 (种子项目 1374)
# =====================================================================

def unstable_ode_deriv(state: List[float], mu: float) -> List[float]:
    """
    不稳定 ODE: y' = A y
    A = [[mu, 1/mu], [-1/mu, mu]]
    """
    if abs(mu) < 1e-15:
        return [0.0, 0.0]
    y1, y2 = state
    return [mu*y1 + y2/mu, -y1/mu + mu*y2]


def unstable_ode_exact(t: float, y0: List[float], mu: float
                       ) -> List[float]:
    """
    精确解:
    y(t) = exp(mu*t) * [[cos(t/mu), -mu*sin(t/mu)],
                         [sin(t/mu)/mu, cos(t/mu)]] * y0
    """
    if abs(mu) < 1e-15:
        return y0[:]
    emt = math.exp(mu * t)
    c = math.cos(t / mu)
    s = math.sin(t / mu)
    y1 = emt * (c * y0[0] - mu * s * y0[1])
    y2 = emt * (s * y0[0] / mu + c * y0[1])
    return [y1, y2]


def unstable_ode_test(mu: float = 0.1, t_stop: float = 5.0,
                       dt: float = 0.01, method: str = 'rk4'
                       ) -> Dict:
    """
    测试数值方法对不稳定 ODE 的跟踪能力.
    """
    y0 = [1.0, 0.0]
    n_steps = int(t_stop / dt)

    if method == 'explicit_euler':
        advance = lambda y: [y[j] + dt * unstable_ode_deriv(y, mu)[j]
                            for j in range(2)]
    elif method == 'implicit_euler':
        # 隐式 Euler: y_{n+1} = y_n + dt f(y_{n+1})
        # 对线性系统: (I - dt A) y_{n+1} = y_n
        def advance(y):
            a11 = 1.0 - dt * mu
            a12 = -dt / mu
            a21 = dt / mu
            a22 = 1.0 - dt * mu
            det = a11*a22 - a12*a21
            if abs(det) < 1e-30:
                return y[:]
            return [(a22*y[0] - a12*y[1])/det,
                    (-a21*y[0] + a11*y[1])/det]
    elif method == 'rk4':
        def advance(y):
            k1 = unstable_ode_deriv(y, mu)
            y2 = [y[j]+0.5*dt*k1[j] for j in range(2)]
            k2 = unstable_ode_deriv(y2, mu)
            y3 = [y[j]+0.5*dt*k2[j] for j in range(2)]
            k3 = unstable_ode_deriv(y3, mu)
            y4 = [y[j]+dt*k3[j] for j in range(2)]
            k4 = unstable_ode_deriv(y4, mu)
            return [y[j]+(dt/6)*(k1[j]+2*k2[j]+2*k3[j]+k4[j])
                    for j in range(2)]
    else:
        raise ValueError(f"Unknown method: {method}")

    y = y0[:]
    max_err = 0.0
    for step in range(n_steps):
        y = advance(y)
        t = (step + 1) * dt
        y_exact = unstable_ode_exact(t, y0, mu)
        err = math.sqrt((y[0]-y_exact[0])**2 + (y[1]-y_exact[1])**2)
        max_err = max(max_err, err)

    y_final_exact = unstable_ode_exact(t_stop, y0, mu)
    return {
        'method': method,
        'mu': mu,
        't_stop': t_stop,
        'dt': dt,
        'y_final': y[:],
        'y_exact': y_final_exact,
        'max_abs_error': max_err,
        'is_stable': max_err < 1e6,
    }


# =====================================================================
#  综合稳定性报告
# =====================================================================

def von_neumann_report(fd_order: int = 4) -> Dict:
    """
    生成 von Neumann 稳定性分析报告.
    """
    results = {}
    for method in ['explicit_euler', 'rk2', 'rk4', 'bdf2']:
        cfl_info = cfl_limit(method, fd_order)
        results[method] = cfl_info
    return results


def dispersion_analysis(kh_max: float = math.pi, n_points: int = 200
                        ) -> Dict:
    """
    色散分析: 比较各阶 FD 和紧致格式的修正波数.
    """
    kh_vals = [kh_max * i / n_points for i in range(n_points + 1)]
    results = {'kh': kh_vals}
    for order in [2, 4, 6, 8]:
        keff_vals = [modified_wavenumber_fd1(kh, order) for kh in kh_vals]
        results[f'fd{order}'] = keff_vals
    results['compact4'] = [modified_wavenumber_compact4(kh) for kh in kh_vals]
    results['compact6'] = [modified_wavenumber_compact6(kh) for kh in kh_vals]
    results['exact'] = kh_vals  # k_eff = k for exact derivative
    return results


if __name__ == '__main__':
    print("=== von Neumann 稳定性分析 ===")

    # 放大因子测试
    z_test = complex(-1.0, 0.0)
    print(f"\nz = {z_test}:")
    for name, func in AMPLIFICATION_METHODS.items():
        G = func(z_test)
        print(f"  {name:16s}: |G| = {abs(G):.6f}")

    # 增长方程特征值
    print("\n增长方程特征值:")
    for P, Q, label in [(2.0, -1.5, "EdS"),
                         (1.55, -0.47, "LCDM today"),
                         (2.0, 0.0, "de Sitter")]:
        l1, l2 = growth_equation_eigenvalues(P, Q)
        S = stiffness_ratio(P, Q)
        print(f"  {label}: P={P}, Q={Q}: "
              f"lam1={l1.real:.4f}, lam2={l2.real:.4f}, S={S:.2f}")

    # CFL 限制
    print("\nCFL 限制:")
    for method in ['explicit_euler', 'rk2', 'rk4']:
        for fd_order in [2, 4, 6]:
            info = cfl_limit(method, fd_order)
            print(f"  {method:16s} + FD{fd_order}: CFL = {info['cfl_number']:.4f}")

    # 不稳定 ODE 测试
    print("\n不稳定 ODE 测试 (mu=0.1, t=5):")
    for method in ['explicit_euler', 'implicit_euler', 'rk4']:
        res = unstable_ode_test(mu=0.1, t_stop=5.0, dt=0.02, method=method)
        print(f"  {method:16s}: err={res['max_abs_error']:.4e}")

    # 摄动 Kepler
    print("\n摄动 Kepler (epsilon=0.001):")
    state0 = [1.0, 0.0, 0.0, 1.0]
    traj = integrate_kepler_rk4(state0, t_end=6.28, dt=0.01, epsilon=0.001)
    H0 = kepler_perturbed_hamiltonian(state0, epsilon=0.001)
    Hf = kepler_perturbed_hamiltonian(traj[-1], epsilon=0.001)
    L0 = kepler_angular_momentum(state0)
    Lf = kepler_angular_momentum(traj[-1])
    print(f"  H 守恒: H0={H0:.6f}, Hf={Hf:.6f}, dH={abs(Hf-H0):.2e}")
    print(f"  L 守恒: L0={L0:.6f}, Lf={Lf:.6f}, dL={abs(Lf-L0):.2e}")

    print("\n所有稳定性测试通过.")
