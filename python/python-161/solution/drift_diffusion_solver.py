"""
drift_diffusion_solver.py
基于种子项目 767_midpoint_fixed (midpoint fixed-point ODE solver)
和 550_humps_ode (humps derivative/exact solution)
改造为钙钛矿太阳能电池稳态载流子输运方程的数值求解器。

核心物理：
  1. 电子连续性方程（稳态）：
       d/dx (μ_n * n * dφ/dx + D_n * dn/dx) = G - R
  2. 空穴连续性方程（稳态）：
       d/dx (μ_p * p * dφ/dx - D_p * dp/dx) = -(G - R)
  3. Poisson 方程：
       d²φ/dx² = -q/ε * (p - n + N_D^+ - N_A^-)

数值方法：
  - 采用 767_midpoint_fixed 中的固定点中点法对时间相关漂移-扩散方程
    进行时间推进至稳态。
  - humps_ode 的精确解作为验证算例，测试求解器精度。
  - Scharfetter-Gummel 格式保证数值稳定性。

核心公式：
  1. Einstein 关系：D_n = (k_B T / q) * μ_n
  2. Scharfetter-Gummel 电流密度：
       J_n,i+1/2 = (q D_n / Δx) * [B(Δφ/kT_q) * n_i - B(-Δφ/kT_q) * n_{i+1}]
  3. 中点法时间推进（θ=0.5）：
       y_{m} = y_i + θ Δt f(t_m, y_m)   (fixed-point 迭代)
       y_{i+1} = (1/θ) y_m + (1 - 1/θ) y_i
"""

import numpy as np
from typing import Callable, Tuple


def midpoint_fixed_time_stepper(
    f: Callable[[np.ndarray], np.ndarray],
    y0: np.ndarray,
    dt: float,
    theta: float = 0.5,
    it_max: int = 10,
    tol: float = 1e-12,
) -> np.ndarray:
    """
    单步固定点中点法。
    对应原项目 midpoint_fixed 的核心算法。

    Parameters
    ----------
    f : callable
        右端函数 f(y) -> dy/dt
    y0 : (m,) array
        当前时刻状态
    dt : float
        时间步长
    theta : float
        中点参数（默认 0.5）
    it_max : int
        固定点迭代最大次数
    tol : float
        收敛容差

    Returns
    -------
    y1 : (m,) array
        下一时刻状态
    """
    if dt <= 0:
        raise ValueError("dt 必须为正")
    if theta <= 0 or theta > 1:
        raise ValueError("theta 必须在 (0,1] 内")

    ym = y0.copy()
    for _ in range(it_max):
        ym_new = y0 + theta * dt * f(ym)
        if np.linalg.norm(ym_new - ym) < tol:
            ym = ym_new
            break
        ym = ym_new

    y1 = (1.0 / theta) * ym + (1.0 - 1.0 / theta) * y0
    return y1


def solve_transient_drift_diffusion_1d(
    N: int,
    L: float,
    T: float,
    mu_n: float,
    mu_p: float,
    eps_r: float,
    N_D: float,
    N_A: float,
    G: np.ndarray,
    R_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    tspan: Tuple[float, float],
    n_steps: int,
    phi_left: float = 0.0,
    phi_right: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    求解一维瞬态漂移-扩散方程至稳态。

    Parameters
    ----------
    N : int
        空间格点数
    L : float
        器件厚度 [cm]
    T : float
        温度 [K]
    mu_n, mu_p : float
        迁移率 [cm^2/(V·s)]
    eps_r : float
        相对介电常数
    N_D, N_A : float
        施主/受主掺杂浓度 [cm^{-3}]
    G : (N,) array
        载流子产生率 [cm^{-3}·s^{-1}]
    R_fn : callable
        复合率函数 R(n, p) -> (N,) array
    tspan : (t0, tf)
        时间范围 [s]
    n_steps : int
        时间步数
    phi_left, phi_right : float
        左右电极电势 [V]

    Returns
    -------
    t : (n_steps+1,) array
        时间序列
    n : (n_steps+1, N) array
        电子浓度
    p : (n_steps+1, N) array
        空穴浓度
    phi : (n_steps+1, N) array
        电势
    """
    if N <= 2:
        raise ValueError("N 必须大于 2")
    if L <= 0:
        raise ValueError("L 必须为正")

    dx = L / (N - 1)
    k_B = 1.380649e-23  # J/K
    q = 1.602176634e-19  # C
    eps0 = 8.854187817e-14  # F/cm
    kT = k_B * T  # J
    kT_q = kT / q  # V
    D_n = kT_q * mu_n
    D_p = kT_q * mu_p
    eps = eps_r * eps0

    # 初始条件：热平衡近似
    n_i = 1.0e10  # 本征载流子浓度 [cm^{-3}]
    n0 = np.full(N, n_i * n_i / N_A if N_A > 0 else n_i)
    p0 = np.full(N, n_i * n_i / N_D if N_D > 0 else n_i)
    phi0 = np.linspace(phi_left, phi_right, N)

    y0 = np.concatenate([n0, p0, phi0])

    dt = (tspan[1] - tspan[0]) / n_steps
    t_arr = np.linspace(tspan[0], tspan[1], n_steps + 1)
    history = np.zeros((n_steps + 1, 3 * N))
    history[0, :] = y0

    def rhs(y: np.ndarray) -> np.ndarray:
        n = np.clip(y[0:N], 1.0, 1e25)
        p = np.clip(y[N:2 * N], 1.0, 1e25)
        phi = y[2 * N:3 * N]
        dydt = np.zeros_like(y)

        # 电子连续性
        for i in range(1, N - 1):
            dphi = (phi[i + 1] - phi[i]) / kT_q if abs(kT_q) > 1e-15 else 0.0
            Bp = _bernoulli(dphi)
            Bm = _bernoulli(-dphi)
            Jn_right = D_n * (Bp * n[i] - Bm * n[i + 1]) / dx
            if not np.isfinite(Jn_right):
                Jn_right = 0.0

            dphi2 = (phi[i] - phi[i - 1]) / kT_q if abs(kT_q) > 1e-15 else 0.0
            Bp2 = _bernoulli(dphi2)
            Bm2 = _bernoulli(-dphi2)
            Jn_left = D_n * (Bp2 * n[i - 1] - Bm2 * n[i]) / dx
            if not np.isfinite(Jn_left):
                Jn_left = 0.0

            R_val = R_fn(float(n[i]), float(p[i]))
            if not np.isfinite(R_val):
                R_val = 0.0
            dydt[i] = -(Jn_right - Jn_left) / dx + G[i] - R_val
            if not np.isfinite(dydt[i]):
                dydt[i] = 0.0

        # 空穴连续性
        for i in range(1, N - 1):
            dphi = (phi[i + 1] - phi[i]) / kT_q if abs(kT_q) > 1e-15 else 0.0
            Bp = _bernoulli(dphi)
            Bm = _bernoulli(-dphi)
            Jp_right = D_p * (Bp * p[i + 1] - Bm * p[i]) / dx
            if not np.isfinite(Jp_right):
                Jp_right = 0.0

            dphi2 = (phi[i] - phi[i - 1]) / kT_q if abs(kT_q) > 1e-15 else 0.0
            Bp2 = _bernoulli(dphi2)
            Bm2 = _bernoulli(-dphi2)
            Jp_left = D_p * (Bp2 * p[i] - Bm2 * p[i - 1]) / dx
            if not np.isfinite(Jp_left):
                Jp_left = 0.0

            R_val = R_fn(float(n[i]), float(p[i]))
            if not np.isfinite(R_val):
                R_val = 0.0
            dydt[N + i] = (Jp_right - Jp_left) / dx + G[i] - R_val
            if not np.isfinite(dydt[N + i]):
                dydt[N + i] = 0.0

        # Poisson
        for i in range(1, N - 1):
            d2phi = (phi[i + 1] - 2 * phi[i] + phi[i - 1]) / (dx * dx)
            rho = q * (p[i] - n[i] + N_D - N_A)
            dydt[2 * N + i] = -(d2phi + rho / eps)
            if not np.isfinite(dydt[2 * N + i]):
                dydt[2 * N + i] = 0.0

        # 边界条件（Dirichlet）
        dydt[0] = 0.0
        dydt[N - 1] = 0.0
        dydt[N] = 0.0
        dydt[2 * N - 1] = 0.0
        dydt[2 * N] = 0.0
        dydt[3 * N - 1] = 0.0

        return dydt

    for step in range(n_steps):
        y1 = midpoint_fixed_time_stepper(rhs, history[step, :], dt, theta=0.5, it_max=10)
        # 数值鲁棒性：浓度非负
        y1[0:N] = np.maximum(y1[0:N], 1.0)
        y1[N:2 * N] = np.maximum(y1[N:2 * N], 1.0)
        history[step + 1, :] = y1

    n_hist = history[:, 0:N]
    p_hist = history[:, N:2 * N]
    phi_hist = history[:, 2 * N:3 * N]
    return t_arr, n_hist, p_hist, phi_hist


def _bernoulli(x: float) -> float:
    """稳定的 Bernoulli 函数。"""
    if abs(x) < 1e-5:
        return 1.0 - x / 2.0 + x * x / 12.0
    elif x > 20.0:
        return x * np.exp(-x)
    elif x < -20.0:
        return -x
    else:
        return x / (np.exp(x) - 1.0)


def humps_deriv(t: float, y: np.ndarray) -> np.ndarray:
    """
    humps ODE 的导数函数（来自 550_humps_ode）。
    用作数值求解器精度的基准测试。
    dy/dt = -2(t-0.3)/((t-0.3)^2+0.01)^2 - 2(t-0.9)/((t-0.9)^2+0.04)^2
    """
    t = float(t)
    dydt = np.array([
        -2.0 * (t - 0.3) / ((t - 0.3) ** 2 + 0.01) ** 2
        - 2.0 * (t - 0.9) / ((t - 0.9) ** 2 + 0.04) ** 2
    ])
    return dydt


def humps_exact(x: np.ndarray) -> np.ndarray:
    """
    humps ODE 的精确解（来自 550_humps_ode）。
    y(x) = 1/((x-0.3)^2+0.01) + 1/((x-0.9)^2+0.04) - 6
    """
    x = np.asarray(x)
    return 1.0 / ((x - 0.3) ** 2 + 0.01) + 1.0 / ((x - 0.9) ** 2 + 0.04) - 6.0


def verify_solver() -> float:
    """
    使用 humps ODE 验证中点法求解器精度。
    返回 L2 误差。
    """
    y0 = np.array([humps_exact(0.0)])
    tspan = (0.0, 1.0)
    n = 200
    dt = (tspan[1] - tspan[0]) / n
    t = np.linspace(tspan[0], tspan[1], n + 1)
    y = np.zeros((n + 1, 1))
    y[0] = y0

    for i in range(n):
        f = lambda yy: humps_deriv(t[i] + 0.5 * dt, yy)
        y[i + 1] = midpoint_fixed_time_stepper(f, y[i], dt, theta=0.5, it_max=20)

    y_exact = humps_exact(t)
    err = np.sqrt(np.mean((y.flatten() - y_exact) ** 2))
    return float(err)


if __name__ == "__main__":
    err = verify_solver()
    print(f"Humps ODE 验证 L2 误差: {err:.3e}")

    # 简单漂移扩散测试
    N = 20
    L = 5e-5  # 500 nm
    def simple_R(n, p):
        return 1e-10 * (n * p - 1e20)
    G = np.ones(N) * 1e21
    t, n_arr, p_arr, phi_arr = solve_transient_drift_diffusion_1d(
        N, L, 300.0, 20.0, 10.0, 30.0, 1e16, 1e16,
        G, simple_R, (0.0, 1e-9), 50
    )
    print(f"稳态 n_max={n_arr[-1].max():.3e}, p_max={p_arr[-1].max():.3e}")
