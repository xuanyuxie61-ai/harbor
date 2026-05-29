"""
微反应器稳态稳定性特征值分析 (基于 Chladni 图与双调和算子思想)
===============================================================
对微反应器内的稳态浓度-温度场进行线性稳定性分析，判断热失控风险。

核心物理模型：
    在稳态解 (C_s(x), T_s(x)) 附近引入小扰动 (c', θ')，
    线性化后得到扰动方程组：

        ∂c'/∂t = D_m ∇²c' - u·∇c' - (∂r/∂C)|_s c' - (∂r/∂T)|_s θ'
        ∂θ'/∂t = α ∇²θ' - u·∇θ' + (-ΔH)/(ρ c_p) [(∂r/∂C)|_s c' + (∂r/∂T)|_s θ']
                                  - (4 h_w)/(ρ c_p d_h) θ'

    其中 α = λ/(ρ c_p) 为热扩散系数。

    离散后写成矩阵形式：
        d/dt [c'; θ'] = J · [c'; θ']

    Jacobian 矩阵 J 的特征值 λ_k 决定稳定性：
        - 若所有 Re(λ_k) < 0：稳态渐近稳定
        - 若存在 Re(λ_k) > 0：稳态不稳定，可能发生热失控

    临界 Damköhler 数 Da_c 由最大实部特征值过零点确定。
"""

import numpy as np
from typing import Tuple, Optional


class ReactorStabilityAnalyzer:
    """
    微反应器稳态线性稳定性分析器。
    """

    def __init__(
        self,
        Nx: int = 100,
        L: float = 0.1,
        D_m: float = 1e-9,
        alpha: float = 1.43e-7,  # 热扩散系数 [m²/s]
        u: float = 0.01,
        dH: float = -8.0e4,
        rho: float = 1000.0,
        cp: float = 4180.0,
        h_wall: float = 500.0,
        d_h: float = 5.0e-4,
        R_gas: float = 8.314,
    ):
        if Nx < 4:
            raise ValueError("Nx 至少为 4")
        self.Nx = Nx
        self.L = L
        self.dx = L / (Nx - 1)
        self.D_m = D_m
        self.alpha = alpha
        self.u = u
        self.dH = dH
        self.rho = rho
        self.cp = cp
        self.h_wall = h_wall
        self.d_h = d_h
        self.R_gas = R_gas

    def _build_advection_diffusion_operator(
        self, diff_coeff: float
    ) -> np.ndarray:
        """
        构建一维对流-扩散算子矩阵 A，采用迎风格式+中心差分：
            A_ij 使得 (A·φ)_i ≈ -u dφ/dx + diff_coeff d²φ/dx²
        Dirichlet 入口，Neumann 出口。
        """
        n = self.Nx
        A = np.zeros((n, n))
        dx = self.dx
        Pe_local = self.u * dx / diff_coeff

        for i in range(1, n - 1):
            # 对流: 迎风格式 (假设 u>0)
            adv = -self.u / dx
            A[i, i] += adv
            A[i, i - 1] += -adv

            # 扩散: 中心差分
            diff = diff_coeff / (dx ** 2)
            A[i, i - 1] += diff
            A[i, i] += -2.0 * diff
            A[i, i + 1] += diff

        # 入口 Dirichlet: 扰动 c'(0)=0, θ'(0)=0
        A[0, :] = 0.0
        A[0, 0] = -1.0  # 固定扰动为 0 (特征值为 -1，不影响稳定性判据)

        # 出口 Neumann (零梯度): 简单外推
        A[n - 1, :] = 0.0
        A[n - 1, n - 1] = -1.0
        A[n - 1, n - 2] = 1.0

        return A

    def compute_jacobian(
        self,
        C_steady: np.ndarray,
        T_steady: np.ndarray,
        A_arr: float,
        Ea: float,
        n_order: float,
    ) -> np.ndarray:
        """
        构建稳态下的 Jacobian 矩阵 J (2Nx × 2Nx)。

        J = [  A_c - diag(dr/dC)          ,   -diag(dr/dT)           ]
            [  (-dH)/(rho*cp) diag(dr/dC) ,   A_T + (-dH)/(rho*cp) diag(dr/dT) - diag(beta_wall) ]

        其中 A_c, A_T 为对流-扩散算子。
        """
        if len(C_steady) != self.Nx or len(T_steady) != self.Nx:
            raise ValueError("稳态场维度与 Nx 不匹配")

        n = self.Nx
        A_c = self._build_advection_diffusion_operator(self.D_m)
        A_t = self._build_advection_diffusion_operator(self.alpha)

        # 反应速率导数
        C_safe = np.maximum(C_steady, 1.0e-12)
        T_safe = np.maximum(T_steady, 200.0)
        # TODO(Hole_2): 计算反应速率对 C 和 T 的偏导数
        # 基于 Arrhenius 模型 r = A·exp(-Ea/(R·T))·C^n 的导数:
        #   dr/dC = r · n / C
        #   dr/dT = r · Ea / (R·T²)
        r_base = np.zeros(n)
        dr_dC = np.zeros(n)
        dr_dT = np.zeros(n)

        # 壁面热损失系数
        beta_wall = 4.0 * self.h_wall / (self.rho * self.cp * self.d_h)

        # 组装 Jacobian
        J = np.zeros((2 * n, 2 * n))
        # 左上: A_c - diag(dr_dC)
        J[:n, :n] = A_c - np.diag(dr_dC)
        # 右上: -diag(dr_dT)
        J[:n, n:] = -np.diag(dr_dT)
        # 左下: (-dH)/(rho*cp) * diag(dr_dC)
        factor = -self.dH / (self.rho * self.cp)
        J[n:, :n] = factor * np.diag(dr_dC)
        # 右下: A_t + factor * diag(dr_dT) - diag(beta_wall)
        J[n:, n:] = A_t + factor * np.diag(dr_dT) - np.diag(np.full(n, beta_wall))

        return J

    def analyze_stability(
        self,
        C_steady: np.ndarray,
        T_steady: np.ndarray,
        A_arr: float,
        Ea: float,
        n_order: float,
    ) -> Tuple[np.ndarray, float, bool]:
        """
        计算 Jacobian 特征值并判断稳定性。

        返回:
            eigenvalues: 全部特征值 (复数)
            max_real:    最大实部
            is_stable:   是否稳定 (max_real < 0)
        """
        J = self.compute_jacobian(C_steady, T_steady, A_arr, Ea, n_order)
        eigenvalues = np.linalg.eigvals(J)
        max_real = np.max(np.real(eigenvalues))
        is_stable = max_real < -1.0e-10
        return eigenvalues, max_real, is_stable

    def compute_critical_damkohler_bracket(
        self,
        C_steady_base: np.ndarray,
        T_steady_base: np.ndarray,
        A_arr_base: float,
        Ea: float,
        n_order: float,
        da_min: float = 0.01,
        da_max: float = 10.0,
        n_scan: int = 20,
    ) -> Tuple[float, float, float]:
        """
        扫描 Damköhler 数范围，定位稳定性转变区间。

        Da = A_arr · L / u，通过等比缩放 A_arr 实现。

        返回:
            (Da_stable, Da_unstable, max_real_at_transition)
        """
        if da_min >= da_max or da_min <= 0.0:
            raise ValueError("da_min 必须为正且小于 da_max")

        Da_vals = np.logspace(np.log10(da_min), np.log10(da_max), n_scan)
        max_reals = np.zeros(n_scan)

        for i, Da in enumerate(Da_vals):
            A_scaled = Da * A_arr_base / (A_arr_base * self.L / self.u)
            A_scaled = A_arr_base * Da / (A_arr_base * self.L / self.u)
            # 实际上 Da = k_ref * L / u，这里简单缩放 A_arr
            A_scaled = A_arr_base * Da
            _, max_reals[i], _ = self.analyze_stability(
                C_steady_base, T_steady_base, A_scaled, Ea, n_order
            )

        # 找 sign change
        stable_mask = max_reals < 0
        if np.all(stable_mask):
            return Da_vals[-1], Da_vals[-1] * 10.0, max_reals[-1]
        if not np.any(stable_mask):
            return Da_vals[0] / 10.0, Da_vals[0], max_reals[0]

        idx = np.where(~stable_mask)[0][0]
        if idx == 0:
            return Da_vals[0], Da_vals[1], max_reals[0]
        return Da_vals[idx - 1], Da_vals[idx], max_reals[idx]

    def compute_thermal_explosion_index(self, max_real: float) -> float:
        """
        热爆炸风险指数：
            I_TE = tanh( max(0, Re(λ_max)) · τ_res )
        其中 τ_res = L / u 为停留时间。指数越接近 1 风险越高。
        """
        tau_res = self.L / max(self.u, 1.0e-12)
        risk = np.tanh(max(0.0, max_real) * tau_res)
        return risk
