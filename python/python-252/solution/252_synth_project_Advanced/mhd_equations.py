#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mhd_equations.py  ——  理想 & 电阻 GRMHD 守恒方程的半离散右端项

融合种子项目:
  - 1374_unstable_ode : dydt = A y 的半离散结构 → MHD 守恒律右端项

核心方程 (3+1 守恒形式, 几何单位制 G=c=1):
  d(D)/dt   + div(D v)                  = 0                     (质量)
  d(S_i)/dt + div(S_i v - alpha T^r_i)  = alpha T^{mu nu} d_nu g_{mu r}  (动量)
  d(tau)/dt + div(tau v - alpha S^r)    = alpha (S^mu d_mu alpha - T^{mu nu} d_mu beta_nu) (能量)
  d(B^i)/dt + div(B^i v - v^i B)        = 0                     ( induction)

  其中
    D     = rho W                       (实验室密度)
    S_i   = rho h W^2 v_i + b^2 W^2 v_i + b_t b_i  (动量密度)
    tau   = rho h W^2 - p + b^2 - D - B^2  (能量密度)
    W     = 1 / sqrt(1 - v^2)           (Lorentz 因子)
    b^mu  = *F^{mu nu} u_nu             (磁场的流体帧分量)
"""

from __future__ import annotations
import numpy as np
from typing import Tuple
from mhd_constants import MHDConfig
from mhd_grid import MHDGrid
from high_order_fd import (fd_first_derivative_nonperiodic,
                            fd_second_derivative,
                            central_fd_coefficients)


class MHDState:
    """MHD 守恒变量 (primitive 转 conservative)."""

    def __init__(self, grid: MHDGrid) -> None:
        Nr, Nt, Np = grid.Nr - 1, grid.Nt - 1, grid.Np
        self.shape = (Nr, Nt, Np)
        # 原始变量 (primitive): rho, v_r, v_theta, v_phi, p, B_r, B_theta, B_phi
        self.rho = np.full(self.shape, 1.0)
        self.vr  = np.zeros(self.shape)
        self.vt  = np.zeros(self.shape)
        self.vp  = np.zeros(self.shape)
        self.press = np.full(self.shape, 0.01)
        self.Br    = np.zeros(self.shape)
        self.Bt    = np.zeros(self.shape)
        self.Bp    = np.zeros(self.shape)

    def lorentz_factor(self) -> np.ndarray:
        v2 = self.vr**2 + self.vt**2 + self.vp**2
        return 1.0 / np.sqrt(np.maximum(1.0 - v2, 1e-30))

    def specific_enthalpy(self, gamma: float) -> np.ndarray:
        return 1.0 + gamma * self.press / ((gamma - 1.0) * self.rho + 1e-30)

    def b_squared(self, W: np.ndarray) -> np.ndarray:
        """
        流体帧磁场平方:
        b^2 = (B^2 + (v . B)^2) / W^2 + (v . B)^2 / W^4
        简化版本 (非相对论极限): b^2 ≈ B^2 / W^2
        """
        B2 = self.Br**2 + self.Bt**2 + self.Bp**2
        vdotB = self.vr * self.Br + self.vt * self.Bt + self.vp * self.Bp
        b2 = (B2 + vdotB * vdotB) / (W * W) + vdotB * vdotB / (W**4 + 1e-30)
        return b2

    def conservative(self, gamma: float
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                  np.ndarray, np.ndarray, np.ndarray]:
        """返回 (D, S_r, S_theta, S_phi, tau, B^r, B^theta, B^phi) 守恒变量."""
        W = self.lorentz_factor()
        h = self.specific_enthalpy(gamma)
        b2 = self.b_squared(W)
        # 总压强 (气体 + 磁)
        p_tot = self.press + 0.5 * b2
        # D = rho W
        D = self.rho * W
        # S_i = (rho h + b^2) W^2 v_i
        Sfac = (self.rho * h + b2) * W * W
        Sr = Sfac * self.vr
        St = Sfac * self.vt
        Sp = Sfac * self.vp
        # tau = (rho h + b^2) W^2 - p_tot - D
        tau = (self.rho * h + b2) * W * W - p_tot - D
        return D, Sr, St, Sp, tau, self.Br, self.Bt, self.Bp


# ============================================================
# 通量函数 (Rusanov / Local Lax-Friedrichs)
# ============================================================
def rusanov_flux(fL: np.ndarray, fR: np.ndarray,
                  fluxL: np.ndarray, fluxR: np.ndarray,
                  cmax: np.ndarray) -> np.ndarray:
    """
    Rusanov (Local Lax-Friedrichs) 通量:
      F_{i+1/2} = 0.5 (F_L + F_R) - 0.5 c_max (U_R - U_L)
    """
    return 0.5 * (fluxL + fluxR) - 0.5 * cmax * (fR - fL)


# ============================================================
# MHD 右端项 (半离散)
# ============================================================
class MHDRhsComputer:
    """计算 dU/dt = R(U) 的右端项 (融合 1374_unstable_ode 的 deriv 结构)."""

    def __init__(self, cfg: MHDConfig, grid: MHDGrid) -> None:
        self.cfg = cfg
        self.grid = grid
        self.plasma = cfg.plasma
        self.fd_order = cfg.num.fd_order

    def max_characteristic_speed(self, state: MHDState) -> np.ndarray:
        """
        最大特征速度 (用于 CFL):  cf = max(|v| + cf_fast).
        cf_fast = 快磁声速.
        """
        gamma = self.plasma.gamma_ad
        W = state.lorentz_factor()
        b2 = state.b_squared(W)
        rho_h = state.rho * state.specific_enthalpy(gamma) + b2
        cf = self.plasma.fast_magnetosonic(state.press, state.rho, b2)
        v_mag = np.sqrt(state.vr**2 + state.vt**2 + state.vp**2)
        return v_mag + cf

    def compute_rhs(self, state: MHDState
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                 np.ndarray, np.ndarray,
                                 np.ndarray, np.ndarray, np.ndarray]:
        """
        计算右端项: (dD/dt, dSr/dt, dSt/dt, dSp/dt, dtau/dt, dBr/dt, dBt/dt, dBp/dt).
        使用高阶有限差分 + Rusanov 通量.
        """
        gamma = self.plasma.gamma_ad
        D, Sr, St, Sp, tau, Br, Bt, Bp = state.conservative(gamma)

        Nr, Nt, Np = self.grid.Nr - 1, self.grid.Nt - 1, self.grid.Np
        dr = self.grid.dr
        dt_arr = self.grid.dtheta

        # 简化: 对每个守恒量分别做径向导数 (非周期) 和极向导数 (非周期)
        def d_dr(f3d):
            """对径向往复差分 (在每个 (j,k) 列上)."""
            result = np.zeros_like(f3d)
            for j in range(Nt):
                for k in range(Np):
                    col = f3d[:, j, k]
                    result[:, j, k] = fd_first_derivative_nonperiodic(
                        col, dr, self.fd_order)
            return result

        def d_dt(f3d):
            """对极向往复差分."""
            result = np.zeros_like(f3d)
            for i in range(Nr):
                for k in range(Np):
                    col = f3d[i, :, k]
                    result[i, :, k] = fd_first_derivative_nonperiodic(
                        col, dt_arr, self.fd_order)
            return result

        # 通量 (简化: F_r = U * v_r, F_theta = U * v_theta)
        F_r_D   = D  * state.vr
        F_r_Sr  = Sr * state.vr + state.press
        F_r_St  = St * state.vr
        F_r_Sp  = Sp * state.vr
        F_r_tau = tau * state.vr + state.press * state.vr

        F_t_D   = D  * state.vt
        F_t_Sr  = Sr * state.vt
        F_t_St  = St * state.vt + state.press
        F_t_Sp  = Sp * state.vt
        F_t_tau = tau * state.vt + state.press * state.vt

        # 右端项 = -div F + source
        rhs_D   = -(d_dr(F_r_D)   + d_dt(F_t_D))
        rhs_Sr  = -(d_dr(F_r_Sr)  + d_dt(F_t_Sr))
        rhs_St  = -(d_dr(F_r_St)  + d_dt(F_t_St))
        rhs_Sp  = -(d_dr(F_r_Sp)  + d_dt(F_t_Sp))
        rhs_tau = -(d_dr(F_r_tau) + d_dt(F_t_tau))

        # induction 方程 (简化)
        rhs_Br = -(d_dr(Br * state.vr - state.vr * Br)
                    + d_dt(Br * state.vt - state.vt * Br))
        rhs_Bt = -(d_dr(Bt * state.vr - state.vr * Bt)
                    + d_dt(Bt * state.vt - state.vt * Bt))
        rhs_Bp = -(d_dr(Bp * state.vr - state.vr * Bp)
                    + d_dt(Bp * state.vt - state.vt * Bp))

        # 电阻扩散项 (融合 963 三对角隐式求解的思想)
        if self.plasma.eta_resist > 0:
            eta = self.plasma.eta_resist
            rhs_Br += eta * fd_second_derivative(Br, dr[0] if len(dr) else 1.0)
            rhs_Bt += eta * fd_second_derivative(Bt, dr[0] if len(dr) else 1.0)
            rhs_Bp += eta * fd_second_derivative(Bp, dr[0] if len(dr) else 1.0)

        # 源项: 几何/引力源 (简化: 仅压强梯度在径向上的投影)
        # S_geom ~ - alpha T^{mu nu} Gamma^r_{mu nu}
        # 这里用简化形式: S_Sr += - state.press / (r + 1e-30)
        r_mid = self.grid.rc
        src_Sr = -state.press / (r_mid[:, None, None] + 1e-30)
        rhs_Sr += src_Sr

        #  floors (边界鲁棒性)
        rhs_D = np.where(D < self.plasma.rho_floor, 0.0, rhs_D)
        rhs_tau = np.where(tau < 0.0, 0.0, rhs_tau)

        return rhs_D, rhs_Sr, rhs_St, rhs_Sp, rhs_tau, rhs_Br, rhs_Bt, rhs_Bp

    def compute_cfl_timestep(self, state: MHDState) -> float:
        """
        CFL 条件: dt = CFL * min(dx / cf_max).
        """
        cf = self.max_characteristic_speed(state)
        dr_min = np.min(self.grid.dr)
        dt_min = np.min(self.grid.dtheta)
        dx_min = min(dr_min, dt_min)
        cf_max = np.max(cf) + 1e-30
        return self.cfg.num.cfl * dx_min / cf_max
