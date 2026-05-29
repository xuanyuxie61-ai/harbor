"""
tov_solver.py
Tolman-Oppenheimer-Volkoff (TOV) 方程数值求解器

该模块实现中子星静力学结构方程的积分，并通过参数扫描
获取不同中心密度下的质量-半径关系。

原项目映射:
- 832_ode_sweep_parfor  -> ODE参数扫描思想，用于扫描中心密度网格

核心物理方程（几何单位 G = c = 1，以 km 为长度单位）:
    dP/dr = - (ε + P)(m + 4π r^3 P) / (r (r - 2m))
    dm/dr = 4π r^2 ε

单位转换:
    1 M_sun = 1.47667 km  (几何质量)
    1 MeV/fm^3 = 1.3234e-6 km^{-2} (几何能量密度/压强)
"""

import numpy as np
import math
from utils_physics import G_NEWTON, C_LIGHT, M_SUN, safe_divide


# =============================================================================
# 单位转换常量 (以 km 为基准)
# =============================================================================
KM = 1.0e3  # m
# 1 M_sun 对应的几何长度 (km)
M_SUN_KM = G_NEWTON * M_SUN / C_LIGHT**2 / KM  # ~1.47667 km

# 1 MeV/fm^3 转换为几何单位 km^{-2}
MEV_FM3_TO_GEOM_KM = 1.602176634e32 * (G_NEWTON / C_LIGHT**4) * (KM**2)


def tov_equations(r: float, y: np.ndarray, eps_of_P: callable) -> np.ndarray:
    """
    TOV 方程右端项（几何单位 G = c = 1，长度单位为 km）。

    y = [P_geom, m_geom]
    dy/dr = [dP/dr, dm/dr]
    """
    # TODO: 修复此处被挖空的 TOV 方程实现
    # 需要计算:
    #   1. 由 P_geom 通过 eps_of_P 得到 eps_geom
    #   2. dm/dr = 4π r^2 ε
    #   3. dP/dr = - (ε + P)(m + 4π r^3 P) / (r(r - 2m))
    raise NotImplementedError("Hole 2: 请补全 TOV 方程右端项的实现")


def build_eps_of_P_simplified(Gamma: float = 2.5, eps0_MeV: float = 50.0):
    """
    构建简化的物态方程 ε(P) = ε0 + P / (Γ - 1)。

    ε0 对应零压能量密度（ crust-core 边界附近约 50 MeV/fm^3），
    Γ 为多方指数，控制物态方程的硬度。
    """
    eps0_geom = eps0_MeV * MEV_FM3_TO_GEOM_KM

    def eps_of_P(P_geom: float) -> float:
        if P_geom <= 0.0:
            return eps0_geom
        return eps0_geom + P_geom / (Gamma - 1.0)

    return eps_of_P


class TOVIntegrator:
    """
    TOV方程数值积分器（几何单位，长度单位为 km）。
    """

    def __init__(self, eps_of_P: callable, max_radius_km: float = 30.0,
                 n_steps: int = 200000, dr_init_km: float = 1.0e-3):
        self.eps_of_P = eps_of_P
        self.max_radius = max_radius_km
        self.n_steps = n_steps
        self.dr_init = dr_init_km

    def _rk4_step(self, r: float, y: np.ndarray, h: float) -> np.ndarray:
        k1 = tov_equations(r, y, self.eps_of_P)
        k2 = tov_equations(r + 0.5 * h, y + 0.5 * h * k1, self.eps_of_P)
        k3 = tov_equations(r + 0.5 * h, y + 0.5 * h * k2, self.eps_of_P)
        k4 = tov_equations(r + h, y + h * k3, self.eps_of_P)
        return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def integrate(self, Pc_MeV_fm3: float) -> dict:
        """
        从中心压强 Pc (MeV/fm^3) 开始积分TOV方程。

        Returns
        -------
        dict
            {'radius_km': R, 'mass_Msun': M, ...}
        """
        if Pc_MeV_fm3 <= 0.0:
            raise ValueError("Central pressure must be positive.")

        Pc_geom = Pc_MeV_fm3 * MEV_FM3_TO_GEOM_KM
        eps_c_geom = self.eps_of_P(Pc_geom)

        r = self.dr_init
        m = (4.0 / 3.0) * math.pi * r**3 * eps_c_geom
        y = np.array([Pc_geom, m])

        r_list = [r]
        P_list = [Pc_geom]
        m_list = [m]

        h = self.dr_init
        for step in range(self.n_steps):
            if y[0] <= 0.0 or r > self.max_radius:
                break

            # 自适应步长
            P_ratio = y[0] / Pc_geom
            if P_ratio < 1e-5:
                h = min(h, r * 0.001)
            elif P_ratio < 0.01:
                h = min(h, r * 0.01)
            else:
                h = min(h, r * 0.05)
            h = max(h, 1e-7)

            y_new = self._rk4_step(r, y, h)
            r_new = r + h

            # 检查Schwarzschild半径
            if r_new <= 2.0 * y_new[1] and step > 100:
                break

            if y_new[0] <= 0.0:
                frac = safe_divide(y[0], y[0] - y_new[0], default=0.0)
                R_surf = r + frac * h
                M_surf = y[1] + frac * (y_new[1] - y[1])
                r_list.append(R_surf)
                P_list.append(0.0)
                m_list.append(M_surf)

                return {
                    'radius_km': R_surf,
                    'mass_Msun': M_surf / M_SUN_KM,
                    'radius_m': R_surf * KM,
                    'mass_kg': M_surf / M_SUN_KM * M_SUN,
                    'profiles': (
                        np.array(r_list),
                        np.array(P_list) / MEV_FM3_TO_GEOM_KM,
                        np.array(m_list) / M_SUN_KM
                    )
                }

            y = y_new
            r = r_new
            r_list.append(r)
            P_list.append(y[0])
            m_list.append(y[1])

        R_surf = r
        M_surf = y[1]
        return {
            'radius_km': R_surf,
            'mass_Msun': M_surf / M_SUN_KM,
            'radius_m': R_surf * KM,
            'mass_kg': M_surf / M_SUN_KM * M_SUN,
            'profiles': (
                np.array(r_list),
                np.array(P_list) / MEV_FM3_TO_GEOM_KM,
                np.array(m_list) / M_SUN_KM
            )
        }


def compute_mass_radius_relation(
    eps_of_P: callable,
    Pc_min_MeV: float = 10.0,
    Pc_max_MeV: float = 1000.0,
    n_points: int = 15
) -> dict:
    """
    参数扫描计算质量-半径关系。
    """
    Pc_vals = np.logspace(math.log10(Pc_min_MeV), math.log10(Pc_max_MeV), n_points)
    R_vals = np.zeros(n_points)
    M_vals = np.zeros(n_points)

    integrator = TOVIntegrator(eps_of_P)

    for i, Pc in enumerate(Pc_vals):
        try:
            result = integrator.integrate(Pc)
            R_vals[i] = result['radius_km']
            M_vals[i] = result['mass_Msun']
        except Exception:
            R_vals[i] = np.nan
            M_vals[i] = np.nan

    M_over_R = np.zeros(n_points)
    for i in range(n_points):
        if not (np.isnan(R_vals[i]) or np.isnan(M_vals[i])) and R_vals[i] > 0.0:
            M_over_R[i] = M_vals[i] * M_SUN_KM / R_vals[i]
        else:
            M_over_R[i] = np.nan

    return {
        'Pc': Pc_vals,
        'R_km': R_vals,
        'M_sun': M_vals,
        'M_over_R': M_over_R
    }


def compute_tidal_deformability(eps_of_P: callable, Pc_MeV: float) -> float:
    """
    计算潮汐形变参数 Λ。
    """
    integrator = TOVIntegrator(eps_of_P)
    result = integrator.integrate(Pc_MeV)
    R_km = result['radius_km']
    M_sun = result['mass_Msun']

    if R_km <= 0.0 or M_sun <= 0.0:
        return np.nan

    C = M_sun * M_SUN_KM / R_km
    if C >= 0.5 or C <= 0.0:
        return 0.0

    k2_approx = (8.0 / 5.0) * (1.0 - 2.0 * C)**2 * C**5 * (
        1.0 + 1.75 * C - 2.8 * C**2
    ) / (1.0 - 2.0 * C + 4.0 * C**2)

    Lambda = (2.0 / 3.0) * k2_approx / C**5
    return Lambda
