"""
spin_dynamics.py
================
自旋动力学与弛豫过程数值积分模块。
融合来源：
- brusselator_ode（非线性ODE系统，化学反应动力学类比自旋非线性弛豫）
- ode_euler_system（显式欧拉系统求解）
- trapezoidal（隐式梯形法 + Newton 迭代）
- fisher_exact（行波解/孤子传播思想，应用于磁畴壁动力学）

物理模型：
    Landau-Lifshitz-Gilbert (LLG) 方程描述经典自旋在有效场中的进动与阻尼：
        dS_i/dt = -γ S_i × H_i + α S_i × (S_i × H_i)
    其中：
        γ 为旋磁比（gyromagnetic ratio），
        α 为 Gilbert 阻尼系数，
        H_i = -δE/δS_i = Σ_j J_{ij} S_j + H_ext + H_anis。

    在阻挫磁体中，有效场 H_i 强烈依赖于周围自旋构型，
    导致高度非线性、刚性的 ODE 系统。

    磁畴壁动力学可类比 Fisher-KPP 行波：
        在易轴各向异性铁磁体中，180° 畴壁以速度 v = sqrt(A K_u) / (M_s sinθ) 传播，
        其中 A 为交换刚度，K_u 为各向异性能量密度。
"""

import numpy as np
from typing import Tuple, Callable, Optional
from utils import EPS_MACHINE, clip_spin_norm
from spin_quaternion import q_rotate_vector, axis_angle_to_q


def effective_field(
    J: np.ndarray,
    spins: np.ndarray,
    H_ext: np.ndarray = None,
    anisotropy_axis: np.ndarray = None,
    K_anis: float = 0.0,
) -> np.ndarray:
    """
    计算有效场：
        H_eff,i = Σ_j J_{ij} S_j + H_ext + 2 K_anis (S_i · n̂) n̂
    """
    N = spins.shape[0]
    H = J @ spins
    if H_ext is not None:
        H = H + H_ext
    if K_anis > EPS_MACHINE and anisotropy_axis is not None:
        n = np.array(anisotropy_axis, dtype=float)
        n_norm = np.linalg.norm(n)
        if n_norm > EPS_MACHINE:
            n = n / n_norm
        proj = np.sum(spins * n, axis=1, keepdims=True)
        H = H + 2.0 * K_anis * proj * n
    return H


def llg_rhs(
    spins: np.ndarray,
    J: np.ndarray,
    gamma: float = 1.0,
    alpha: float = 0.1,
    H_ext: np.ndarray = None,
    anisotropy_axis: np.ndarray = None,
    K_anis: float = 0.0,
) -> np.ndarray:
    """
    LLG 方程右端项：
        dS_i/dt = -γ S_i × H_i + α S_i × (S_i × H_i)
    使用向量三重积恒等式：
        S × (S × H) = S (S·H) - H (S·S) = S (S·H) - H   (因为 |S|=1)
    """
    H = effective_field(J, spins, H_ext, anisotropy_axis, K_anis)
    # 叉积 S × H
    cross = np.cross(spins, H)
    # 阻尼项 S × (S × H) = S (S·H) - H
    dot = np.sum(spins * H, axis=1, keepdims=True)
    damping = spins * dot - H
    dSdt = -gamma * cross + alpha * damping
    # 投影到切平面，保证 |S| 守恒（一阶修正）
    tangent_proj = dSdt - spins * np.sum(dSdt * spins, axis=1, keepdims=True)
    return tangent_proj


def llg_rhs_flat(t: float, y_flat: np.ndarray, J: np.ndarray, **kwargs) -> np.ndarray:
    """将展平自旋向量还原为 (N,3) 后求 LLG 右端，再展平。"""
    N = J.shape[0]
    spins = y_flat.reshape((N, 3))
    # 强制归一化，防止数值漂移
    spins = np.array([clip_spin_norm(s) for s in spins])
    dSdt = llg_rhs(spins, J, **kwargs)
    return dSdt.ravel()


def euler_integrate_llg(
    J: np.ndarray,
    spins0: np.ndarray,
    t_span: Tuple[float, float],
    n_steps: int,
    gamma: float = 1.0,
    alpha: float = 0.1,
    H_ext: np.ndarray = None,
    anisotropy_axis: np.ndarray = None,
    K_anis: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    显式欧拉法积分 LLG 方程。
    融合来源：ode_euler_system（Euler method for ODE system）。

    时间步进：
        S^{n+1} = S^n + h * f(t_n, S^n)
    每一步后重新归一化自旋。

    稳定性限制：h < h_max ~ 1 / (γ |H_max|)。
    """
    t0, tf = t_span
    h = (tf - t0) / n_steps
    N = spins0.shape[0]
    t_arr = np.linspace(t0, tf, n_steps + 1)
    spins_traj = np.zeros((n_steps + 1, N, 3), dtype=float)
    spins = spins0.copy()
    spins_traj[0] = spins

    for i in range(n_steps):
        dSdt = llg_rhs(spins, J, gamma, alpha, H_ext, anisotropy_axis, K_anis)
        spins = spins + h * dSdt
        # 归一化
        for j in range(N):
            spins[j] = clip_spin_norm(spins[j])
        spins_traj[i + 1] = spins

    return t_arr, spins_traj


def trapezoidal_integrate_llg(
    J: np.ndarray,
    spins0: np.ndarray,
    t_span: Tuple[float, float],
    n_steps: int,
    gamma: float = 1.0,
    alpha: float = 0.1,
    H_ext: np.ndarray = None,
    anisotropy_axis: np.ndarray = None,
    K_anis: float = 0.0,
    newton_tol: float = 1e-10,
    newton_max_iter: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    隐式梯形法积分 LLG 方程，配合简化的 Newton-Raphson 迭代。
    融合来源：trapezoidal（隐式梯形 + fsolve 思想）。

    离散格式：
        S^{n+1} = S^n + (h/2) [ f(t_n, S^n) + f(t_{n+1}, S^{n+1}) ]

    由于 LLG 的强非线性，显式欧拉需极小的步长；
    梯形法具有 A-稳定性，允许更大步长，但每步需解非线性方程。
    """
    t0, tf = t_span
    h = (tf - t0) / n_steps
    N = spins0.shape[0]
    t_arr = np.linspace(t0, tf, n_steps + 1)
    spins_traj = np.zeros((n_steps + 1, N, 3), dtype=float)
    spins = spins0.copy()
    spins_traj[0] = spins

    kwargs = {
        "gamma": gamma,
        "alpha": alpha,
        "H_ext": H_ext,
        "anisotropy_axis": anisotropy_axis,
        "K_anis": K_anis,
    }

    for i in range(n_steps):
        f_old = llg_rhs(spins, J, **kwargs)
        # 初始猜测：显式 Euler 一步
        spins_new = spins + h * f_old
        for j in range(N):
            spins_new[j] = clip_spin_norm(spins_new[j])

        # 简化的 Newton 迭代：固定 Jacobian 近似为单位矩阵的修正
        for _newton in range(newton_max_iter):
            f_new = llg_rhs(spins_new, J, **kwargs)
            residual = spins_new - spins - 0.5 * h * (f_old + f_new)
            # 线性修正（阻尼 Newton）
            delta = -residual
            spins_new = spins_new + 0.5 * delta
            for j in range(N):
                spins_new[j] = clip_spin_norm(spins_new[j])
            if np.linalg.norm(delta) < newton_tol:
                break

        spins = spins_new
        spins_traj[i + 1] = spins

    return t_arr, spins_traj


def brusselator_like_spin_pump(
    t: float,
    y: np.ndarray,
    a: float = 1.0,
    b: float = 3.0,
) -> np.ndarray:
    """
    Brusselator 型非线性自旋泵模型。
    融合来源：brusselator_ode（Brusselator 化学反应动力学）。

    将自旋密度 u, v 类比于 Brusselator 中的两种化学组分，
    引入非线性泵浦项模拟自旋波 parametric pumping：
        du/dt = a + u^2 v - (b+1) u
        dv/dt = b u - u^2 v

    在磁性系统中，u 可视为磁化强度纵向分量，v 为横向泵浦幅度。
    """
    u, v = y[0], y[1]
    dudt = a + u * u * v - (b + 1.0) * u
    dvdt = b * u - u * u * v
    return np.array([dudt, dvdt])


def integrate_brusselator_pump(
    a: float = 1.0,
    b: float = 3.0,
    y0: np.ndarray = None,
    t_span: Tuple[float, float] = (0.0, 20.0),
    n_steps: int = 2000,
) -> Tuple[np.ndarray, np.ndarray]:
    """积分 Brusselator 型自旋泵方程（显式欧拉）。"""
    if y0 is None:
        y0 = np.array([0.5, 1.0])
    t0, tf = t_span
    h = (tf - t0) / n_steps
    t_arr = np.linspace(t0, tf, n_steps + 1)
    y_traj = np.zeros((n_steps + 1, 2), dtype=float)
    y = y0.copy()
    y_traj[0] = y
    for i in range(n_steps):
        dydt = brusselator_like_spin_pump(t_arr[i], y, a, b)
        y = y + h * dydt
        y_traj[i + 1] = y
    return t_arr, y_traj


def fisher_kpp_domain_wall_exact(
    t: float,
    x: np.ndarray,
    a: float = 1.0,
    c: float = 5.0 / np.sqrt(6.0),
    k: float = np.sqrt(6.0) / 6.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fisher-KPP 方程的行波精确解，融合来源：fisher_exact。

    方程：
        ∂u/∂t = D ∂²u/∂x² + r u (1 - u)
    对于特定波速 c = 5/√6 * sqrt(D r)，存在解析解：
        u(x,t) = 1 / [1 + a exp(k (x - c t))]²

    在磁学中，此解可描述一维易轴各向异性铁磁体中
    180° 磁畴壁的孤子型传播：
        M_z(x,t) = M_s [ 2 u(x,t) - 1 ]
    畴壁中心以速度 c 运动，宽度 ~ 1/k。

    返回
    ----
    u, ut, ux, uxx : np.ndarray
        解及其偏导数。
    """
    z = x - c * t
    exp_kz = np.exp(k * z)
    denom = 1.0 + a * exp_kz
    u = 1.0 / (denom ** 2)
    ut = 2.0 * c * a * k * exp_kz / (denom ** 3)
    ux = -2.0 * a * k * exp_kz / (denom ** 3)
    uxx = 6.0 * (a ** 2) * (k ** 2) * np.exp(2.0 * k * z) / (denom ** 4) - \
          2.0 * a * (k ** 2) * exp_kz / (denom ** 3)
    return u, ut, ux, uxx


def domain_wall_magnetization(
    t: float,
    x: np.ndarray,
    Ms: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    由 Fisher-KPP 行波解构造磁化强度分布 M_z(x,t)。
    """
    u, ut, ux, uxx = fisher_kpp_domain_wall_exact(t, x)
    Mz = Ms * (2.0 * u - 1.0)
    dMz_dt = 2.0 * Ms * ut
    dMz_dx = 2.0 * Ms * ux
    d2Mz_dx2 = 2.0 * Ms * uxx
    return Mz, dMz_dt, dMz_dx, d2Mz_dx2


def compute_magnetization_trajectory(spins_traj: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从自旋轨迹计算总磁化强度 M(t) = (1/N) Σ_i S_i(t)。
    返回 Mx, My, Mz 时间序列。
    """
    N = spins_traj.shape[1]
    M = np.mean(spins_traj, axis=1)
    return M[:, 0], M[:, 1], M[:, 2]
