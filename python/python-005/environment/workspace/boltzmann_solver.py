# -*- coding: utf-8 -*-
"""
boltzmann_solver.py
线性化爱因斯坦-玻尔兹曼方程求解器

核心物理：
在共形时间 η 内，求解光子-重子流体微扰：
    Δ_0' = - (k/3) Δ_1 - Φ'
    Δ_1' = k (Δ_0 + Ψ) - τ' (Δ_1 - v_b)
    Δ_2' = (2k/3) Δ_1 - (3/10) τ' Δ_2
    v_b' = - (a'/a) v_b + k Ψ + (τ'/R) (Δ_1 - v_b)
其中 R = 3ρ_b / (4ρ_γ)，τ' 为汤姆孙散射不透明度。

引力势满足泊松型约束：
    k^2 Φ = - (3/2) (a'/a)^2 [Δ_0 + (ρ_b/ρ) (3 v_b / k)]
    Ψ = - Φ - (3/2) (a'/a)^2 (ρ_b/ρ) (v_b/k)

本模块融合种子项目 1060_schroedinger_linear_pde（Method of Lines）
与 345_exm/waterwave（Lax-Wendroff 型 PDE 求解思想）。
"""

import numpy as np
from typing import Tuple
from utils import robust_divide, ensure_positive


class CosmologyParams:
    """ΛCDM 宇宙学参数（普朗克 2018 最佳拟合值）。"""
    def __init__(self):
        self.Omega_b = 0.022383  # 重子密度参数 × h^2
        self.Omega_c = 0.12011   # CDM 密度参数 × h^2
        self.Omega_L = 0.684     # 暗能量密度参数
        self.h = 0.6732          # Hubble 常数 H0/(100 km/s/Mpc)
        self.Tcmb = 2.7255       # CMB 温度 [K]
        self.YHe = 0.2454        # 氦质量丰度
        self.Neff = 3.046        # 有效相对论性中微子种类


class BoltzmannSolver:
    """
    一阶线性化爱因斯坦-玻尔兹曼方程数值求解器。
    使用有限差分法对共形时间 η 进行离散，
    在紧耦合极限下解析处理光子-重子滑移。
    """

    def __init__(self, params: CosmologyParams, k_mode: float,
                 n_eta: int = 2000, eta_max: float = 14000.0):
        """
        Parameters
        ----------
        params : CosmologyParams
            宇宙学参数。
        k_mode : float
            共动波数 [Mpc^{-1}]。
        n_eta : int
            共形时间网格点数。
        eta_max : float
            最大共形时间 [Mpc]。
        """
        self.params = params
        self.k = ensure_positive(k_mode, "k_mode")
        self.n_eta = n_eta
        self.eta_max = eta_max
        # 共形时间网格
        self.eta = np.linspace(0.0, eta_max, n_eta)
        self.deta = self.eta[1] - self.eta[0]
        # 初始化尺度因子和背景量
        self._init_background()

    # ------------------------------------------------------------------
    # 背景宇宙学演化
    # ------------------------------------------------------------------
    def _scale_factor(self, eta: float) -> float:
        """
        尺度因子 a(η) 的近似解析解（辐射-物质-Λ 纪元）。
        对辐射主导 a ∝ η；物质主导 a ∝ η^2；Λ 主导 a ∝ exp(HΛ η)。
        这里采用简化分段形式：
            a_eq = Ω_r / Ω_m  （辐射-物质相等时刻）
        """
        Omega_r = 2.47e-5 / (self.params.h ** 2)
        Omega_m = (self.params.Omega_b + self.params.Omega_c) / (self.params.h ** 2)
        a_eq = Omega_r / Omega_m
        # 简化：辐射+物质主导期解析近似
        # a(η) = a_eq * [(η/η_eq)^2 + 2 η/η_eq]
        # 其中 η_eq ≈ 1 / (a_eq H0 sqrt(Ω_m))
        H0 = 100.0 * self.params.h  # km/s/Mpc
        # 转换为 Mpc^{-1} 的量纲，这里采用简化数值近似
        eta_eq = 14.0 / np.sqrt(Omega_m * (self.params.h ** 2))  # 近似
        if eta <= 0.0:
            return 1e-10
        a = a_eq * ((eta / eta_eq) ** 2 + 2.0 * (eta / eta_eq))
        return max(a, 1e-10)

    def _hub_conformal(self, eta: float) -> float:
        """
        共形 Hubble 参数 H = a'/a = (1/a) da/dη。
        对 a ∝ η^α，有 H = α/η。
        """
        a = self._scale_factor(eta)
        # 数值微分（中心差分）
        dh = 1e-4
        a_p = self._scale_factor(eta + dh)
        a_m = self._scale_factor(max(eta - dh, 1e-6))
        dadeta = (a_p - a_m) / (2.0 * dh)
        return dadeta / a

    def _thomson_opacity(self, eta: float) -> float:
        """
        汤姆孙散射不透明度 τ'(η) [Mpc^{-1}]。
        τ' = n_e σ_T a，其中 n_e 为自由电子数密度。
        在复合时期采用简单的解析拟合（Jones & Wyse 1985 型）。
        """
        a = self._scale_factor(eta)
        z = 1.0 / a - 1.0
        if z < 50.0:
            return 0.0
        # 简化拟合：τ' ∝ a^{-2} (1+z)^2，在 z_rec ~ 1100 处峰值
        tau_dot = 7.0e-4 * ((1.0 + z) / 1100.0) ** 2.5
        return tau_dot

    def _init_background(self):
        """预计算背景数组。"""
        self.a_arr = np.array([self._scale_factor(e) for e in self.eta])
        self.H_arr = np.array([self._hub_conformal(e) for e in self.eta])
        self.tau_dot_arr = np.array([self._thomson_opacity(e) for e in self.eta])

    # ------------------------------------------------------------------
    # 微扰方程右端项
    # ------------------------------------------------------------------
    def _rhs(self, y: np.ndarray, ieta: int) -> np.ndarray:
        """
        返回 dy/dη，其中 y = [Δ_0, Δ_1, Δ_2, v_b, Φ]。
        采用紧耦合极限解析修正以提高数值稳定性。
        """
        eta = self.eta[ieta]
        k = self.k
        a = self.a_arr[ieta]
        H = self.H_arr[ieta]
        tau_dot = self.tau_dot_arr[ieta]

        Delta0, Delta1, Delta2, vb, Phi = y

        # 重子-光子比 R = 3ρ_b / (4ρ_γ)
        rho_b_over_rho = self.params.Omega_b / (self.params.Omega_b + self.params.Omega_c)
        R = 3.0 * rho_b_over_rho / (4.0 * (1.0 - rho_b_over_rho))

        # 紧耦合极限：滑移 Δ_1 - v_b 被抑制
        slip = Delta1 - vb
        if tau_dot > 1e-3:
            # 在紧耦合期，滑移近似为零，但保留数值项
            slip = 0.0

        # 引力势 Ψ（采用简化关系 Ψ ≈ -Φ，忽略各向异性应力）
        Psi = -Phi

        # 光子单极方程
        dDelta0 = - (k / 3.0) * Delta1 - self._dPhi_deta(ieta)

        # 光子偶极方程
        dDelta1 = k * (Delta0 + Psi)
        if tau_dot > 1e-3:
            dDelta1 -= tau_dot * slip

        # 光子四极方程（紧耦合时 Δ_2 被抑制 ~ k/τ' Δ_1）
        dDelta2 = (2.0 * k / 3.0) * Delta1
        if tau_dot > 1e-3:
            dDelta2 -= (9.0 / 10.0) * tau_dot * Delta2

        # 重子速度方程
        dvb = -H * vb + k * Psi
        if tau_dot > 1e-3:
            dvb += (tau_dot / R) * slip

        # 引力势演化（简化，来自泊松方程的时间导数）
        dPhi = self._dPhi_deta(ieta)

        return np.array([dDelta0, dDelta1, dDelta2, dvb, dPhi])

    def _dPhi_deta(self, ieta: int) -> float:
        """
        通过泊松约束数值计算 Φ'。
        泊松方程：k^2 Φ = - (3/2) H^2 [Δ_0 + 3 (ρ_b/ρ) (v_b / k)]
        """
        if ieta <= 0 or ieta >= self.n_eta - 1:
            return 0.0
        # 这里采用简化近似：在超视界外 Φ' ≈ 0，在子视界内由物质主导
        H = self.H_arr[ieta]
        return -H * self._Phi_from_constraint(ieta)

    def _Phi_from_constraint(self, ieta: int) -> float:
        """由泊松约束反解 Φ。"""
        k = self.k
        if k < 1e-12:
            return 1.0
        H = self.H_arr[ieta]
        rho_b_over_rho = self.params.Omega_b / (self.params.Omega_b + self.params.Omega_c)
        # 这里用简化初值
        Delta0_init = 1.0  # 归一化曲率扰动
        Phi = -1.5 * (H ** 2) / (k ** 2) * Delta0_init
        return Phi

    # ------------------------------------------------------------------
    # 时间积分（改进 Euler + 紧耦合修正）
    # ------------------------------------------------------------------
    def solve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        从初始条件出发，沿 η 积分微扰方程。

        Returns
        -------
        eta, Delta0, Delta1, Delta2, vb 数组
        """
        n = self.n_eta
        Delta0 = np.zeros(n)
        Delta1 = np.zeros(n)
        Delta2 = np.zeros(n)
        vb = np.zeros(n)
        Phi = np.zeros(n)

        # 初始条件（曲率规范，超视界近似）
        # Δ_0(0) = - (2/3) Φ(0), Δ_1(0) = 0, v_b(0) = 0
        Phi0 = 1.0  # 归一化
        Delta0[0] = -2.0 / 3.0 * Phi0
        Delta1[0] = 0.0
        Delta2[0] = 0.0
        vb[0] = 0.0
        Phi[0] = Phi0

        # 改进 Euler 积分
        for i in range(n - 1):
            y = np.array([Delta0[i], Delta1[i], Delta2[i], vb[i], Phi[i]])
            k1 = self._rhs(y, i)
            y_pred = y + self.deta * k1
            # 边界保护
            y_pred = np.clip(y_pred, -1e6, 1e6)
            k2 = self._rhs(y_pred, i + 1)
            y_new = y + 0.5 * self.deta * (k1 + k2)
            Delta0[i + 1], Delta1[i + 1], Delta2[i + 1], vb[i + 1], Phi[i + 1] = y_new

        return self.eta, Delta0, Delta1, Delta2, vb

    def transfer_function_today(self) -> float:
        """
        提取今天的转移函数 T(k)。
        在复合之后，T(k) ∝ Δ_0(η_rec) + 2 Φ(η_rec) + ...
        这里采用 Sachs-Wolfe 近似：
            T(k) ≈ (1/3) ζ(k) cos(k r_s) - 3 Φ(k) sin(k r_s)/(k r_s)
        其中 ζ 为曲率扰动，r_s 为声视界。
        """
        # TODO: 请补全转移函数提取的核心公式
        # 提示：需要基于 solve() 的结果，结合 Sachs-Wolfe 近似、
        #       声学振荡和 Silk 阻尼计算 T(k)
        raise NotImplementedError("Hole_1: 请补全 transfer_function_today 的实现")
