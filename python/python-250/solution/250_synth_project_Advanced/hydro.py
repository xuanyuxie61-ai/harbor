"""
球对称辐射流体求解器 (from 744_md 速度 Verlet 积分思想).

将分子动力学中的 velocity-Verlet 辛积分推广到辐射流体：
  显式部分 (流体力学 + 源项) 用 WENO5 + HLL 通量 + RK3.
  隐式部分 (辐射扩散 + 中微子冷却) 用隐式 GMRES 迭代.

守恒变量 U = (ρ, ρv, E)^T 在球坐标：
  ∂U/∂t + (1/r^2) ∂(r^2 F)/∂r = S(U)

通量 F = (ρv, ρv^2 + P, v(E + P))^T.
源项 S 包括：
  - 几何源项 (球坐标曲率): S_geom = (2P/r, 0, 0)^T  [仅在动量方程]
  - 引力源项: S_grav = (0, -ρ GM/r^2, 0)^T
  - 辐射源项 (动量+能量交换): S_rad = (-χ_F F_rad/c, ..., -χ_F c E_rad + χ_E a T^4)

时间积分 (SSP-RK3, Gottlieb-Shu 1998):
  U^(1) = U^n + Δt L(U^n)
  U^(2) = (3/4) U^n + (1/4) (U^(1) + Δt L(U^(1)))
  U^(n+1) = (1/3) U^n + (2/3) (U^(2) + Δt L(U^(2)))
每步后做 positivity-preserving 限幅 (Zhang-Shu 2010)：
  ρ_{i}^{new} = max(ρ_i^{new}, ε_ρ)
  P_i^{new} = max(P_i^{new}, ε_P)

Von Neumann 稳定性分析：对线性化方程 U_t = A U_x,
  放大矩阵 G(k) = I + Δt A (ik) + (Δt A)^2 (ik)^2 / 2 + ...
  稳定性条件：谱半径 ρ(G) ≤ 1 对所有波数 k.
  Courant 条件 C = max |λ(A)| Δt/Δx ≤ C_CFL ≈ 0.4.
"""
from __future__ import annotations
import math
import numpy as np

import constants as C
from reconstruction import WENO5Reconstructor


class RadiationHydroSolver:
    """球对称辐射流体显式求解器."""

    def __init__(self, mesh, eos, opacity_table, quadrature,
                 gravity_M: float = 1.4 * C.M_SUN,
                 cfl: float = 0.4,
                 epsilon_rho: float = 1.0e-10,
                 epsilon_p: float = 1.0e-10):
        self.mesh = mesh
        self.eos = eos
        self.opacity = opacity_table
        self.quad = quadrature
        self.gravity_M = float(gravity_M)
        self.cfl = float(cfl)
        self.eps_rho = float(epsilon_rho)
        self.eps_p = float(epsilon_p)
        self.weno = WENO5Reconstructor()

    def conservative_to_primitive(self, U: np.ndarray):
        """从守恒变量 U=(ρ, ρv, E) 提取原始变量 (ρ, v, P, T)."""
        rho = U[:, 0]
        rho = np.maximum(rho, self.eps_rho)
        v = U[:, 1] / rho
        E = U[:, 2]
        # 动能密度
        e_kin = 0.5 * rho * v ** 2
        # 比内能 = (E - e_kin) / ρ
        e_int = (E - e_kin) / rho
        e_int = np.maximum(e_int, self.eps_p)
        # 温度 (由物态反解)
        T = self.eos.temperature_from_energy(rho, e_int)
        # 压强
        _, _, _, P = self.eos.pressures(rho, T)
        # 正性保护
        rho = np.maximum(rho, self.eps_rho)
        P = np.maximum(P, self.eps_p)
        return rho, v, P, T

    def primitive_to_conservative(self, rho, v, P, T):
        """从原始变量构造守恒变量."""
        e_int = self.eos.specific_internal_energy(rho, T)
        e_kin = 0.5 * rho * v ** 2
        E = rho * (e_int + e_kin)
        U = np.stack([rho, rho * v, E], axis=-1)
        return U

    def flux_physical(self, rho, v, P, E):
        """物理通量 F(U) = (ρv, ρv^2+P, v(E+P))."""
        f_rho = rho * v
        f_mom = rho * v ** 2 + P
        f_E = v * (E + P)
        return f_rho, f_mom, f_E

    def wave_speeds(self, rho, v, cs):
        """HLL 波速 (Davis 1988)."""
        SL = float(np.min(v - cs))
        SR = float(np.max(v + cs))
        return SL, SR

    def source_terms(self, rho, v, P, r):
        """源项 S(U) = S_geom + S_grav + S_rad.

        几何项 (球坐标): S_geom = (2ρv^2/r + 2P/r, -ρ v^2 / r, 2Pv/r)
          仅当采用几何守恒形式时显式处理。
        引力项: S_grav = (0, -ρ GM/r^2, -ρ GM v / r^2)
        """
        S = np.zeros((len(rho), 3), dtype=np.float64)
        r_safe = np.maximum(r, 1.0e10)  # cm
        # 引力
        g_r = -C.G_GRAV * self.gravity_M / (r_safe ** 2)
        S[:, 1] += rho * g_r
        S[:, 2] += rho * v * g_r
        # 几何曲率源 (仅动量方程)
        S[:, 1] += 2.0 * P / r_safe
        return S

    def rhs_explicit(self, U: np.ndarray) -> np.ndarray:
        """显式空间离散右端项 - (1/r^2) ∂(r^2 F)/∂r + S.

        采用几何守恒形式 (Ritter 2003):
          d/dt (V_i U_i) = - (r_{i+1/2}^2 F_{i+1/2} - r_{i-1/2}^2 F_{i-1/2}) + V_i S_i
          V_i = (4π/3) (r_{i+1}^3 - r_i^3)
        """
        N = self.mesh.n_cells
        rho, v, P, T = self.conservative_to_primitive(U)
        cs = self.eos.sound_speed(rho, T, P)
        # 守恒变量
        E = U[:, 2]
        # 重构 ρ, v, P 到界面
        rho_L, rho_R = self.weno.reconstruct_interface(rho)
        v_L, v_R = self.weno.reconstruct_interface(v)
        P_L, P_R = self.weno.reconstruct_interface(P)
        # 界面能量密度
        e_int_L = self.eos.specific_internal_energy(rho_L, T[:-1] + 0.0 * rho_L)
        e_int_R = self.eos.specific_internal_energy(rho_R, T[1:] + 0.0 * rho_R)
        E_L = rho_L * (e_int_L + 0.5 * v_L ** 2)
        E_R = rho_R * (e_int_R + 0.5 * v_R ** 2)
        # 界面物理通量
        fL_rho, fL_mom, fL_E = self.flux_physical(rho_L, v_L, P_L, E_L)
        fR_rho, fR_mom, fR_E = self.flux_physical(rho_R, v_R, P_R, E_R)
        # HLL 通量
        SL, SR = self.wave_speeds(rho, v, cs)
        denom = SR - SL
        if abs(denom) < 1.0e-30:
            F_rho = 0.5 * (fL_rho + fR_rho)
            F_mom = 0.5 * (fL_mom + fR_mom)
            F_E = 0.5 * (fL_E + fR_E)
        else:
            F_rho = (SR * fL_rho - SL * fR_rho + SL * SR * (rho_R - rho_L)) / denom
            F_mom = (SR * fL_mom - SL * fR_mom + SL * SR * (rho_R * v_R - rho_L * v_L)) / denom
            F_E = (SR * fL_E - SL * fR_E + SL * SR * (E_R - E_L)) / denom
        # 界面半径 (单元边界): N+1 个
        r_iface = self.mesh.r_nodes
        r2 = r_iface ** 2
        # 边界条件：内边界通量 = 0, 外边界通量 = 0 (零通量边界)
        F_rho_full = np.zeros(N + 1, dtype=np.float64)
        F_mom_full = np.zeros(N + 1, dtype=np.float64)
        F_E_full = np.zeros(N + 1, dtype=np.float64)
        F_rho_full[1:-1] = F_rho
        F_mom_full[1:-1] = F_mom
        F_E_full[1:-1] = F_E
        # 通量散度 (几何守恒形式)
        dU = np.zeros_like(U)
        for k, Fk in enumerate([F_rho_full, F_mom_full, F_E_full]):
            # 通量差: r_{i+1}^2 F_{i+1/2} - r_i^2 F_{i-1/2}
            flux_term = r2[1:] * Fk[1:] - r2[:-1] * Fk[:-1]
            # 体积 (4π/3) (r_{i+1}^3 - r_i^3)
            V = self.mesh.volumes
            dU[:, k] = -3.0 * flux_term / (4.0 * math.pi * V + 1.0e-30)
        # 源项 (使用单元中心值)
        S = self.source_terms(rho, v, P, self.mesh.r_centers)
        dU += S
        return dU

    def positivity_limiter(self, U: np.ndarray) -> np.ndarray:
        """Zhang-Shu (2010) 正性保护限幅."""
        U_new = U.copy()
        U_new[:, 0] = np.maximum(U_new[:, 0], self.eps_rho)
        # 压强检查
        rho = U_new[:, 0]
        v = U_new[:, 1] / np.maximum(rho, self.eps_rho)
        E = U_new[:, 2]
        e_int = (E - 0.5 * rho * v ** 2) / rho
        e_int = np.maximum(e_int, self.eps_p)
        U_new[:, 2] = rho * (e_int + 0.5 * v ** 2)
        return U_new

    def timestep_explicit(self, rho, v, cs):
        """显式部分时间步."""
        return self.mesh.courant_dt(v, cs)
