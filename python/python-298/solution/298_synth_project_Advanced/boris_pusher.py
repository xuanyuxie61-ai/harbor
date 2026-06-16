"""
Boris 粒子推进器

实现相对论性 Boris 推进算法, 是 PIC 方法的核心:
    dv/dt = (q/m) * (E + v × B)

Boris 算法:
    1. 半步电场加速:  v^- = v^n + (q*dt)/(2m) * E^n
    2. 磁场旋转:       v' = v^- + v^- × t
                       v^+ = v^- + v' × s
    3. 半步电场加速:  v^{n+1} = v^+ + (q*dt)/(2m) * E^n
    其中 t = (q*dt)/(2m) * B,  s = 2t/(1+t^2)

位置更新:
    x^{n+1} = x^n + dt * v^{n+1/2}
"""

import numpy as np
from typing import Tuple
try:
    from plasma_constants import ELECTRON_CHARGE, ELECTRON_MASS, SPEED_OF_LIGHT
except ImportError:
    from plasma_constants import ELECTRON_CHARGE, ELECTRON_MASS, SPEED_OF_LIGHT


def boris_push_nonrelativistic(v: np.ndarray, E: np.ndarray, B: np.ndarray,
                                 q_over_m: float, dt: float) -> np.ndarray:
    """
    非相对论 Boris 推进 (单粒子)

    Parameters:
        v       : 速度 [3]
        E       : 电场 [3]
        B       : 磁场 [3]
        q_over_m: 荷质比 q/m
        dt      : 时间步长

    Returns:
        v_new   : 新速度 [3]
    """
    # Step 1: 半步电场加速
    v_minus = v + 0.5 * q_over_m * dt * E

    # Step 2: 磁场旋转
    t_vec = 0.5 * q_over_m * dt * B
    t_mag_sq = np.dot(t_vec, t_vec)
    s_vec = 2.0 * t_vec / (1.0 + t_mag_sq)

    v_prime = v_minus + np.cross(v_minus, t_vec)
    v_plus = v_minus + np.cross(v_prime, s_vec)

    # Step 3: 半步电场加速
    v_new = v_plus + 0.5 * q_over_m * dt * E

    return v_new


def boris_push_relativistic(v: np.ndarray, E: np.ndarray, B: np.ndarray,
                              q_over_m: float, dt: float) -> np.ndarray:
    """
    相对论 Boris 推进

    使用 Lorentz 因子 gamma = 1/sqrt(1 - v^2/c^2)
    """
    c = SPEED_OF_LIGHT
    beta = v / c
    beta_sq = np.dot(beta, beta)
    if beta_sq >= 1.0:
        raise ValueError("粒子速度超光速, 物理非法")
    gamma = 1.0 / np.sqrt(1.0 - beta_sq)

    # 相对论修正: 有效质量 = gamma * m
    gamma_m = gamma

    # Step 1: 半步电场加速
    v_minus = v + 0.5 * (q_over_m / gamma_m) * dt * E

    # gamma update for rotation
    beta_minus = v_minus / c
    gamma_minus = 1.0 / np.sqrt(max(1.0 - np.dot(beta_minus, beta_minus), 1e-15))

    # Step 2: 磁场旋转 (相对论 t 矢量)
    t_vec = 0.5 * (q_over_m / gamma_minus) * dt * B
    t_mag_sq = np.dot(t_vec, t_vec)
    s_vec = 2.0 * t_vec / (1.0 + t_mag_sq)

    v_prime = v_minus + np.cross(v_minus, t_vec)
    v_plus = v_minus + np.cross(v_prime, s_vec)

    # Step 3: 半步电场加速
    v_new = v_plus + 0.5 * (q_over_m / gamma_m) * dt * E

    return v_new


def push_particles_1d(x: np.ndarray, v: np.ndarray, E_field: np.ndarray,
                        q_over_m: float, dt: float, dx: float,
                        L: float, relativistic: bool = False,
                        interp_order: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    1D 静电 PIC 中的粒子推进 (所有粒子)

    Parameters:
        x           : 粒子位置 [Np]
        v           : 粒子速度 [Np]  (1D)
        E_field     : 网格上的电场 [Ng]
        q_over_m    : 荷质比
        dt          : 时间步
        dx          : 网格间距
        L           : 域长度
        relativistic: 是否相对论
        interp_order: 插值阶数 (0: NGP, 1: CIC, 2: TSC)

    Returns:
        (x_new, v_new)
    """
    Np = len(x)
    Ng = len(E_field)

    # 粒子处插值电场
    E_at_particle = interpolate_field_to_particles(x, E_field, dx, L, interp_order)

    # 1D: B = 0, 纯电场加速
    v_new = np.zeros(Np)
    for i in range(Np):
        v_vec = np.array([v[i], 0.0, 0.0])
        E_vec = np.array([E_at_particle[i], 0.0, 0.0])
        B_vec = np.array([0.0, 0.0, 0.0])

        if relativistic:
            v_new_vec = boris_push_relativistic(v_vec, E_vec, B_vec, q_over_m, dt)
        else:
            v_new_vec = boris_push_nonrelativistic(v_vec, E_vec, B_vec, q_over_m, dt)
        v_new[i] = v_new_vec[0]

    # 位置更新 (leapfrog: 半步)
    x_new = x + v_new * dt

    # 周期边界
    x_new = x_new % L

    return x_new, v_new


def interpolate_field_to_particles(x: np.ndarray, field: np.ndarray,
                                     dx: float, L: float,
                                     order: int = 1) -> np.ndarray:
    """
    将网格场插值到粒子位置

    order=0: NGP (Nearest Grid Point)
    order=1: CIC (Cloud In Cell)
    order=2: TSC (Triangular Shaped Cloud)

    CIC 公式 (1D):
        E(x_p) = E_{i} * (1 - d) + E_{i+1} * d
        其中 d = (x_p - x_i) / dx
    """
    Np = len(x)
    Ng = len(field)
    E_p = np.zeros(Np)

    for p in range(Np):
        xp = x[p] % L  # 周期化
        i = int(xp / dx)
        d = (xp - i * dx) / dx

        if order == 0:  # NGP
            i = i % Ng
            E_p[p] = field[i]
        elif order == 1:  # CIC
            i0 = i % Ng
            i1 = (i + 1) % Ng
            E_p[p] = field[i0] * (1.0 - d) + field[i1] * d
        elif order == 2:  # TSC
            i_minus = (i - 1) % Ng
            i_zero = i % Ng
            i_plus = (i + 1) % Ng
            E_p[p] = (field[i_minus] * 0.5 * (0.5 - d)**2 +
                      field[i_zero] * (0.75 - d**2) +
                      field[i_plus] * 0.5 * (0.5 + d)**2)
        else:
            raise ValueError(f"不支持插值阶数 {order}")

    return E_p


def deposit_charge_1d(x: np.ndarray, q: float, weight: float,
                        Ng: int, dx: float, L: float,
                        order: int = 1) -> np.ndarray:
    """
    将粒子电荷沉积到网格 (形状因子)

    rho(x_grid) = sum_p q_p * S(x_grid - x_p)

    CIC:
        S(x) = 1 - |x|/dx,  |x| < dx
        S(x) = 0,            otherwise
    """
    Np = len(x)
    rho = np.zeros(Ng)

    for p in range(Np):
        xp = x[p] % L
        i = int(xp / dx)
        d = (xp - i * dx) / dx

        if order == 0:  # NGP
            rho[i % Ng] += q * weight
        elif order == 1:  # CIC
            rho[i % Ng] += q * weight * (1.0 - d)
            rho[(i + 1) % Ng] += q * weight * d
        elif order == 2:  # TSC
            rho[(i - 1) % Ng] += q * weight * 0.5 * (0.5 - d)**2
            rho[i % Ng] += q * weight * (0.75 - d**2)
            rho[(i + 1) % Ng] += q * weight * 0.5 * (0.5 + d)**2

    rho /= dx  # 归一化
    return rho


def leapfrog_position(x: np.ndarray, v: np.ndarray, dt: float,
                        L: float) -> np.ndarray:
    """
    Leapfrog 位置更新:
        x^{n+1} = x^n + dt * v^{n+1/2}
    """
    x_new = x + v * dt
    return x_new % L


def leapfrog_velocity(v: np.ndarray, accel: np.ndarray, dt: float) -> np.ndarray:
    """
    Leapfrog 速度更新:
        v^{n+1} = v^n + dt * a^{n+1/2}
    """
    return v + accel * dt


def energy_conservation_check(x: np.ndarray, v: np.ndarray,
                                E_field: np.ndarray, dx: float,
                                m: float, q: float, epsilon_0: float) -> float:
    """
    总能量守恒检查:
        E_total = sum_p 0.5*m*v_p^2 + 0.5*epsilon_0*sum_g E_g^2 * dx
    """
    from poisson_solver import energy_from_field

    W_kinetic = 0.5 * m * np.sum(v**2)
    W_field = energy_from_field(E_field, dx, epsilon_0)
    return W_kinetic + W_field
