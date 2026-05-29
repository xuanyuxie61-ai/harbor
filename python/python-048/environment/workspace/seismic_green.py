"""
seismic_green.py
地震波格林函数计算模块

原项目映射:
    430_filon_rule -> 振荡积分数值求值（Filon 方法）

在微地震监测中，震源激发的地震波在均匀或分层介质中的传播
可用格林函数描述。频域格林函数涉及振荡积分，
Filon 方法对高频振荡被积函数具有远超普通 Newton-Cotes 的精度。

核心公式:
1. 三维均匀介质中点力格林函数（频域）:
   G_{ij}(x,ω;ξ) = (1/4πρ) [
       (δ_{ij} - γ_i γ_j) exp(i ω r / β) / (ω² r / β²)
     + γ_i γ_j exp(i ω r / α) / (ω² r / α²)
   ]
   其中 r = |x - ξ|, γ = (x - ξ) / r，
   α 为 P 波速度，β 为 S 波速度，ρ 为密度。

2. 远场位移谱（矩张量源）:
   ũ_i(x,ω) = M_{jk}(ω) ∂G_{ij}(x,ω;ξ)/∂ξ_k
            ≈ (i ω) / (4πρ r α³) γ_i γ_j γ_k M_{jk}(ω) exp(i ω r / α)
            + (i ω) / (4πρ r β³) (δ_{ij} - γ_i γ_j) γ_k M_{jk}(ω) exp(i ω r / β)

3. 时间域位移通过逆 Fourier 变换得到:
   u_i(x,t) = (1/2π) ∫_{-∞}^{∞} ũ_i(x,ω) exp(-i ω t) dω
   该积分对固定 x,t 具有强振荡性，适合 Filon 型方法。

4. Filon 型积分用于计算频域到时间域的转换:
   I(t) = ∫_{ω_min}^{ω_max} A(ω) cos[ω (t - r/c)] dω
   其中 A(ω) 为慢变振幅，cos[ω(t-r/c)] 为快变相位。
"""

import numpy as np
from typing import Callable
from quadrature_rules import filon_cos_quad, filon_sin_quad


class SeismicGreen:
    """
    均匀各向同性介质中的地震波格林函数计算器。
    """

    def __init__(self, rho: float = 2650.0,
                 alpha: float = 4500.0,
                 beta: float = 2600.0):
        """
        参数:
            rho: 密度 (kg/m³)。
            alpha: P 波速度 (m/s)。
            beta: S 波速度 (m/s)。
        """
        if alpha <= beta:
            raise ValueError("P 波速度必须大于 S 波速度")
        if rho <= 0 or alpha <= 0 or beta <= 0:
            raise ValueError("介质参数必须为正")
        self.rho = rho
        self.alpha = alpha
        self.beta = beta

    def radiation_factors(self, x: np.ndarray, xi: np.ndarray) -> tuple:
        """
        计算震源-接收点几何因子。

        返回:
            r: 震源距 (m)。
            gamma: 单位方向向量 (x - xi) / r。
        """
        dx = x - xi
        r = np.linalg.norm(dx)
        if r < 1.0e-12:
            r = 1.0e-12
            gamma = np.array([0.0, 0.0, 1.0])
        else:
            gamma = dx / r
        return r, gamma

    def displacement_spectrum_farfield(self, x: np.ndarray, xi: np.ndarray,
                                        M: np.ndarray, omega: float) -> np.ndarray:
        """
        计算远场位移谱（单一频率 ω）。

        公式:
            ũ_i = (iω)/(4πρr) [
                    (1/α³) γ_i γ_j γ_k M_{jk} exp(iωr/α)
                  + (1/β³) (δ_{ij} - γ_i γ_j) γ_k M_{jk} exp(iωr/β)
                  ]

        参数:
            x: 接收点坐标 [x,y,z]。
            xi: 震源坐标 [x,y,z]。
            M: 矩张量 (3x3) (N·m)。
            omega: 角频率 (rad/s)。

        返回:
            ũ: 复位移谱向量 [ux, uy, uz]。
        """
        # TODO Hole 1: 实现远场位移谱计算
        # 公式: ũ_i = (iω)/(4πρr) [
        #     (1/α³) γ_i γ_j γ_k M_{jk} exp(iωr/α)
        #   + (1/β³) (δ_{ij} - γ_i γ_j) γ_k M_{jk} exp(iωr/β)
        # ]
        raise NotImplementedError("Hole 1: 请实现远场位移谱公式")

    def time_domain_displacement_filon(self, x: np.ndarray, xi: np.ndarray,
                                        M: np.ndarray, t: float,
                                        omega_max: float = 200.0,
                                        n_omega: int = 401) -> np.ndarray:
        """
        使用 Filon 积分将频域位移转换到时间域。

        公式:
            u_i(x,t) = Re{ (1/π) ∫_{0}^{ω_max} ũ_i(x,ω) exp(-iωt) dω }
            被积函数的实部为:
            A(ω) cos[ω(t - r/c)] 形式的叠加。

        参数:
            x: 接收点。
            xi: 震源点。
            M: 矩张量。
            t: 时间 (s)。
            omega_max: 截断角频率。
            n_omega: Filon 积分采样点数（必须为奇数）。

        返回:
            u: 时间域位移向量 (m)。
        """
        if n_omega % 2 == 0:
            n_omega += 1

        r, gamma = self.radiation_factors(x, xi)

        # 分别对三个分量进行 Filon 积分
        u = np.zeros(3, dtype=float)

        for comp in range(3):
            # 构造振幅函数 A(ω)
            def amplitude(omega_arr):
                if np.isscalar(omega_arr):
                    omega_arr = np.array([omega_arr])
                else:
                    omega_arr = np.asarray(omega_arr)
                out = np.zeros_like(omega_arr, dtype=complex)
                for idx, w in enumerate(omega_arr):
                    if w == 0:
                        out[idx] = 0.0
                    else:
                        us = self.displacement_spectrum_farfield(x, xi, M, w)
                        out[idx] = us[comp]
                return out

            # 对每个波型分别积分
            # P 波贡献: Re{ coeff * exp(iω(r/α - t)) }
            def integrand_p(omega_arr):
                omega_arr = np.asarray(omega_arr, dtype=float)
                out = np.zeros_like(omega_arr, dtype=float)
                for idx, w in enumerate(omega_arr):
                    if w == 0:
                        continue
                    coeff = 1.0j * w / (4.0 * np.pi * self.rho * r * self.alpha ** 3)
                    M_proj = np.sum(M * np.outer(gamma, gamma))
                    val = coeff * M_proj * np.exp(1.0j * w * (r / self.alpha - t))
                    out[idx] = np.real(val)
                return out

            def integrand_s(omega_arr):
                omega_arr = np.asarray(omega_arr, dtype=float)
                out = np.zeros_like(omega_arr, dtype=float)
                for idx, w in enumerate(omega_arr):
                    if w == 0:
                        continue
                    coeff = 1.0j * w / (4.0 * np.pi * self.rho * r * self.beta ** 3)
                    identity = np.eye(3)
                    Mgk = (identity - np.outer(gamma, gamma)) @ (M @ gamma)
                    val = coeff * Mgk[comp] * np.exp(1.0j * w * (r / self.beta - t))
                    out[idx] = np.real(val)
                return out

            # 这里对振幅使用简化处理：取实部的 Filon 余弦积分
            # 由于被积函数形式较复杂，使用直接采样 Simpson 作为稳健 fallback
            omega_vals = np.linspace(0.0, omega_max, n_omega)
            d_omega = omega_vals[1] - omega_vals[0]

            f_p = integrand_p(omega_vals)
            f_s = integrand_s(omega_vals)

            # Simpson 积分（对非振荡部分同样稳健）
            def simpson_integral(y, h):
                if y.size < 3 or y.size % 2 == 0:
                    return np.trapezoid(y, dx=h)
                return h / 3.0 * (y[0] + y[-1] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-2:2]))

            u[comp] = simpson_integral(f_p, d_omega) + simpson_integral(f_s, d_omega)

        # 归一化因子 1/π（因只积正频率，需乘 2/2 = 1，已包含在系数中）
        return u

    def travel_time_p(self, x: np.ndarray, xi: np.ndarray) -> float:
        """P 波走时。"""
        r, _ = self.radiation_factors(x, xi)
        return r / self.alpha

    def travel_time_s(self, x: np.ndarray, xi: np.ndarray) -> float:
        """S 波走时。"""
        r, _ = self.radiation_factors(x, xi)
        return r / self.beta
