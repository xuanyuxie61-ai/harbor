"""
neutrino_diffusion.py
中子星内部中微子输运方程求解模块

中微子在致密物质中的扩散-对流-反应过程由以下方程组描述:
    ∂Y_e/∂t = D ∇^2 Y_e + v · ∇Y_e + S(Y_e, T)
    ∂T/∂t   = (K_thermal / c_v) ∇^2 T + Q_ν(T)

其中 Y_e 为电子丰度，T 为温度，D 为扩散系数，
v 为物质流速，S 为弱反应源项，Q_ν 为中微子冷却率。

原项目映射:
- 1368_tumor_pde  -> 反应-扩散-对流 PDE 的系数函数与源项结构
"""

import numpy as np
import math
from typing import Callable, Tuple


# =============================================================================
# 物理参数与系数函数
# =============================================================================
def neutrino_parameters() -> dict:
    """
    返回中微子输运的物理参数。

    参考值:
        D      ~ 10^4 cm^2/s (扩散系数)
        v      ~ 10^7 cm/s   (对流速度)
        K_th   ~ 10^{23} erg/(cm s K) (热导率)
        c_v    ~ 10^{20} erg/(cm^3 K) (热容)
        lambda ~ 1e-2 s^-1   (弱反应率)
    """
    return {
        'diffusion_coeff': 1.0e4,       # cm^2/s
        'convection_velocity': 1.0e7,   # cm/s
        'thermal_conductivity': 1.0e23, # erg/(cm s K)
        'heat_capacity': 1.0e20,        # erg/(cm^3 K)
        'weak_rate': 1.0e-2,            # s^-1
        'neutrino_luminosity_coeff': 1.0e25,  # erg/(cm^3 s)
        't0': 0.0,
        'tstop': 10.0,                  # s
        'xmin': 0.0,
        'xmax': 1.0e5,                  # cm (约1km)
    }


def neutrino_coefficients(x: float, t: float, state: np.ndarray,
                          dstate_dx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    定义中微子输运PDE的系数函数。

    PDE形式（参照 tumor_pde 的 c, f, s 结构）:
        c(x,t,u,DuDx) * ∂u/∂t = x^{-m} ∂/∂x [ x^m f(x,t,u,DuDx) ] + s(x,t,u,DuDx)

    u = [Y_e, T]  (电子丰度, 温度)

    Parameters
    ----------
    x : float
        空间坐标 (cm)。
    t : float
        时间坐标 (s)。
    state : np.ndarray, shape (2,)
        [Y_e, T]。
    dstate_dx : np.ndarray, shape (2,)
        空间导数 [dY_e/dx, dT/dx]。

    Returns
    -------
    c : np.ndarray, shape (2,)
        时间导数系数。
    f : np.ndarray, shape (2,)
        通量项。
    s : np.ndarray, shape (2,)
        源项。
    """
    params = neutrino_parameters()
    D = params['diffusion_coeff']
    v = params['convection_velocity']
    K_th = params['thermal_conductivity']
    c_v = params['heat_capacity']
    weak_rate = params['weak_rate']
    lum_coeff = params['neutrino_luminosity_coeff']

    Y_e, T = state
    dYedx, dTdx = dstate_dx

    # 边界保护
    Y_e = np.clip(Y_e, 0.0, 1.0)
    T = max(T, 1.0e3)  # 最低温度 1000 K

    c_vec = np.array([1.0, 1.0])

    # 扩散通量 + 对流通量
    f_vec = np.array([
        D * dYedx + v * Y_e,
        (K_th / c_v) * dTdx + v * T
    ])

    # 弱反应源项 (电子俘获/β衰变平衡)
    # S_Y = -λ (Y_e - Y_eq(T))
    Y_eq = 0.05 + 0.1 * math.exp(-T / 1.0e9)  # 近似平衡丰度
    source_Y = -weak_rate * (Y_e - Y_eq)

    # 中微子冷却率
    # Q_ν ∝ T^6 (URCA过程)
    Q_nu = -lum_coeff * (T / 1.0e9)**6

    s_vec = np.array([source_Y, Q_nu])

    return c_vec, f_vec, s_vec


# =============================================================================
# 有限差分求解
# =============================================================================
def solve_neutrino_diffusion_1d(
    nx: int = 200,
    nt: int = 5000,
    t_final: float = 10.0,
    L: float = 1.0e5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    用显式有限差分法求解1D中微子扩散-对流方程。

    空间离散: x_i = i * Δx, i = 0, ..., nx
    时间离散: t_n = n * Δt, n = 0, ..., nt

    稳定性条件（CFL条件）:
        Δt <= min( Δx^2 / (2D), Δx / |v| )

    Parameters
    ----------
    nx : int
        空间网格数。
    nt : int
        时间步数。
    t_final : float
        终止时间 (s)。
    L : float
        空间区域长度 (cm)。

    Returns
    -------
    x : np.ndarray
        空间网格。
    t : np.ndarray
        时间网格。
    solution : np.ndarray, shape (nt+1, nx+1, 2)
        [Y_e, T] 在每个时空点的值。
    """
    params = neutrino_parameters()
    D = params['diffusion_coeff']
    v = params['convection_velocity']
    K_th = params['thermal_conductivity']
    c_v = params['heat_capacity']
    alpha_th = K_th / c_v

    dx = L / nx
    # 满足稳定性条件的最大时间步
    dt_diff = dx**2 / (2.0 * max(D, alpha_th))
    dt_conv = dx / max(abs(v), 1.0e-10)
    dt = min(dt_diff, dt_conv, t_final / nt)
    nt_actual = int(t_final / dt) + 1

    x = np.linspace(0.0, L, nx + 1)
    t = np.linspace(0.0, t_final, nt_actual)

    # 初始条件
    Y_e = np.ones(nx + 1) * 0.3  # 初始电子丰度
    T = np.ones(nx + 1) * 1.0e9  # 初始温度 10^9 K

    solution = np.zeros((nt_actual, nx + 1, 2))
    solution[0, :, 0] = Y_e
    solution[0, :, 1] = T

    for n in range(nt_actual - 1):
        Y_new = Y_e.copy()
        T_new = T.copy()

        for i in range(1, nx):
            # 扩散项 (中心差分)
            diff_Y = D * (Y_e[i + 1] - 2.0 * Y_e[i] + Y_e[i - 1]) / dx**2
            diff_T = alpha_th * (T[i + 1] - 2.0 * T[i] + T[i - 1]) / dx**2

            # 对流项 (迎风差分)
            if v > 0.0:
                conv_Y = v * (Y_e[i] - Y_e[i - 1]) / dx
                conv_T = v * (T[i] - T[i - 1]) / dx
            else:
                conv_Y = v * (Y_e[i + 1] - Y_e[i]) / dx
                conv_T = v * (T[i + 1] - T[i]) / dx

            # 源项
            Y_eq = 0.05 + 0.1 * math.exp(-T[i] / 1.0e9)
            source_Y = -params['weak_rate'] * (Y_e[i] - Y_eq)
            source_T = -params['neutrino_luminosity_coeff'] * (T[i] / 1.0e9)**6

            Y_new[i] = Y_e[i] + dt * (diff_Y - conv_Y + source_Y)
            T_new[i] = T[i] + dt * (diff_T - conv_T + source_T)

        # 边界条件
        # x=0: 对称边界 (Neumann)
        Y_new[0] = Y_new[1]
        T_new[0] = T_new[1]
        # x=L: 固定边界 (Dirichlet)
        Y_new[nx] = 0.1
        T_new[nx] = 5.0e8

        # 物理约束
        Y_new = np.clip(Y_new, 0.0, 1.0)
        T_new = np.clip(T_new, 1.0e3, 1.0e12)

        Y_e = Y_new
        T = T_new
        solution[n + 1, :, 0] = Y_e
        solution[n + 1, :, 1] = T

    return x, t, solution


def compute_neutrino_luminosity(
    T_profile: np.ndarray,
    dx: float,
    R_star: float = 1.0e6
) -> float:
    """
    计算中子星总中微子光度。

    公式:
        L_ν = 4π R^2 ∫_0^R Q_ν(r) dr

    其中 Q_ν ∝ T^6 为体积冷却率。
    """
    params = neutrino_parameters()
    lum_coeff = params['neutrino_luminosity_coeff']

    Q_vol = lum_coeff * (T_profile / 1.0e9)**6
    integral = np.trapz(Q_vol, dx=dx)

    L_nu = 4.0 * math.pi * R_star**2 * integral
    return L_nu


def compute_deleptonization_timescale(
    Y_e_initial: float,
    Y_e_final: float,
    weak_rate: float = 1.0e-2
) -> float:
    """
    估算电子丰度弛豫的特征时标。

    公式:
        τ ~ |Y_e,final - Y_e,initial| / (λ Y_e,initial)
    """
    if Y_e_initial <= 0.0 or weak_rate <= 0.0:
        return float('inf')
    return abs(Y_e_final - Y_e_initial) / (weak_rate * max(Y_e_initial, 1e-10))
