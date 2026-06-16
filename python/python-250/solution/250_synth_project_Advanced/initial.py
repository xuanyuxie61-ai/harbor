"""
初始条件：前身星剖面的简化参数化.

真实超新星前身星来自 MESA/KEPLER 演化计算. 这里采用
解析近似 (Arnett 1980, Kifonidis 2003)：

密度剖面 (分三段):
  1. 铁核 (r < r_Fe = 1.5e8 cm):
       ρ(r) = ρ_c (1 - (r/r_Fe)^2)^{3/2}  (多方 n=3 多)
  2. 硅层 (r_Fe < r < r_Si = 5e8 cm):
       ρ(r) = ρ_Fe (r_Fe/r)^{2.5}
  3. 包层 (r > r_Si):
       ρ(r) = ρ_Si (r_Si/r)^{n_env},  n_env = 2.0 - 3.5

温度剖面：
  T(r) = T_c (P(r)/P_c)^{∇_ad}
  简化: T ∝ ρ^{1/3} (理想气体近似)

速度初值：
  坍缩阶段: v(r) = - H_0 r (均匀坍缩, H_0 ≈ 100 /s)
  反弹后: 初始静止 v(r) = 0.

激波初值：
  在 r = r_shock 处放置 Rankine-Hugoniot 跳跃.
"""
from __future__ import annotations
import math
import numpy as np

import constants as C
import eos as eos_mod


def progenitor_profile(r: np.ndarray, M_proto: float = 1.4 * C.M_SUN) -> dict:
    """构造前身星剖面 (解析近似).

  返回 {rho, T, v, P, Ye}.
  """
    r_Fe = 1.5e8    # cm
    r_Si = 5.0e8    # cm
    r_env = 1.0e11  # cm (整个恒星)
    rho_c = 1.0e10  # g/cm^3 (铁核中心)
    T_c = 1.0e10    # K
    # 密度分段
    rho = np.empty_like(r, dtype=np.float64)
    # 铁核
    mask_fe = r < r_Fe
    rho[mask_fe] = rho_c * (1.0 - (r[mask_fe] / r_Fe) ** 2) ** 1.5
    # 硅层
    mask_si = (r >= r_Fe) & (r < r_Si)
    rho_fe = rho_c * (1.0 - 1.0) ** 1.5  # 边界 ρ=0 不物理，修正
    rho_fe = max(rho_c * 0.01, 1.0e7)  # 铁核边界密度
    rho[mask_si] = rho_fe * (r_Fe / r[mask_si]) ** 2.5
    # 包层
    mask_env = r >= r_Si
    rho_si = rho_fe * (r_Fe / r_Si) ** 2.5
    rho[mask_env] = rho_si * (r_Si / r[mask_env]) ** 3.0
    # 密度下限
    rho = np.maximum(rho, 1.0e-5)
    # 温度
    T = T_c * (rho / rho_c) ** (1.0 / 3.0)
    T = np.maximum(T, 1.0e6)
    # 电子丰度
    Ye = np.where(rho > 1.0e9, 0.42, 0.5)  # 高密区电子俘获降 Ye
    # 物态
    eos = eos_mod.EquationOfState(y_e=0.42)
    _, _, _, P = eos.pressures(rho, T)
    # 初始速度 (坍缩)
    v = -1.0e8 * (r / r_Fe)  # cm/s, 向内
    v = np.where(r > r_Fe, 0.0, v)
    return {'rho': rho, 'T': T, 'v': v, 'P': P, 'Ye': Ye}


def initial_conservative(mesh, rho, v, P, T, eos):
    """从原始变量构造守恒变量."""
    return eos if False else _convert(mesh, rho, v, P, T, eos)


def _convert(mesh, rho, v, P, T, eos):
    """内部转换函数."""
    e_int = eos.specific_internal_energy(rho, T)
    e_kin = 0.5 * rho * v ** 2
    E = rho * (e_int + e_kin)
    U = np.stack([rho, rho * v, E], axis=-1)
    return U


def insert_shock(U: np.ndarray, mesh, r_shock: float, mach: float = 5.0,
                 gamma: float = 5.0 / 3.0) -> np.ndarray:
    """在 r_shock 处插入 Rankine-Hugoniot 激波跳跃.

  对激波上游 (r > r_shock): 未扰动
  对激波下游 (r < r_shock): 压强密度跳跃
    ρ_2/ρ_1 = (γ+1) M^2 / ((γ-1) M^2 + 2)
    P_2/P_1 = (2 γ M^2 - (γ-1)) / (γ+1)
    v_2 = v_1 (1 - ρ_1/ρ_2)  (质量守恒)
  """
    U_new = U.copy()
    idx_shock = np.searchsorted(mesh.r_centers, r_shock)
    idx_shock = min(idx_shock, mesh.n_cells - 1)
    # 下游
    for i in range(idx_shock + 1):
        rho1 = U[i, 0]
        v1 = U[i, 1] / max(rho1, 1.0e-30)
        E1 = U[i, 2]
        P1 = (gamma - 1.0) * (E1 - 0.5 * rho1 * v1 ** 2)
        rho2 = rho1 * (gamma + 1.0) * mach ** 2 / (
            (gamma - 1.0) * mach ** 2 + 2.0)
        P2 = P1 * (2.0 * gamma * mach ** 2 - (gamma - 1.0)) / (gamma + 1.0)
        v2 = v1 * (1.0 - rho1 / max(rho2, 1.0e-30))
        # 更新守恒变量
        e_int1 = (E1 - 0.5 * rho1 * v1 ** 2) / max(rho1, 1.0e-30)
        e_int2 = P2 / ((gamma - 1.0) * max(rho2, 1.0e-30))
        E2 = rho2 * (e_int2 + 0.5 * v2 ** 2)
        U_new[i, 0] = rho2
        U_new[i, 1] = rho2 * v2
        U_new[i, 2] = E2
    return U_new
