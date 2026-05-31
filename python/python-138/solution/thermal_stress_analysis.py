"""
微反应器薄板热应力分析 (基于双调和方程精确解)
=============================================
微反应器的薄壁结构在高温梯度下产生热应力，需要满足双调和控制方程。

控制方程：
    对于薄板挠度 w(x,y)，双调和方程为：

        ∇⁴ w = ∂⁴w/∂x⁴ + 2 ∂⁴w/∂x²∂y² + ∂⁴w/∂y⁴ = q(x,y)/D

    其中 D = E h³ / [12(1-ν²)] 为弯曲刚度，q 为等效热载荷。

    热应力与挠度的关系：
        σ_x = -E z / (1-ν²) [ ∂²w/∂x² + ν ∂²w/∂y² ] - E α_T ΔT / (1-ν)
        σ_y = -E z / (1-ν²) [ ∂²w/∂y² + ν ∂²w/∂x² ] - E α_T ΔT / (1-ν)
        τ_xy = -E z / (1+ν) ∂²w/∂x∂y

    最大 von Mises 等效应力：
        σ_vm = √(σ_x² + σ_y² - σ_x σ_y + 3 τ_xy²)

本模块构造满足双调和方程的解析试函数，用于验证有限元代码或评估
微反应器板在典型温度场下的最大应力。
"""

import numpy as np
from typing import Tuple, Optional


class ThermalStressAnalyzer:
    """
    微反应器薄板热应力分析器（基于双调和方程解析解）。
    """

    def __init__(
        self,
        E: float = 200.0e9,      # 弹性模量 [Pa]
        nu: float = 0.3,         # 泊松比
        h: float = 1.0e-3,       # 板厚 [m]
        alpha_T: float = 1.2e-5,  # 热膨胀系数 [1/K]
        z_eval: float = 0.5e-3,  # 评估面位置 (z = h/2 为最大应力面)
    ):
        if E <= 0.0 or h <= 0.0:
            raise ValueError("材料参数必须为正")
        if not (0.0 < nu < 0.5):
            raise ValueError("泊松比必须在 (0, 0.5) 之间")
        self.E = E
        self.nu = nu
        self.h = h
        self.alpha_T = alpha_T
        self.z_eval = z_eval
        self.D_flexural = E * h ** 3 / (12.0 * (1.0 - nu ** 2))

    def biharmonic_solution_w(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        a: float = 1.0,
        b: float = 0.0,
        c: float = 0.0,
        d: float = 0.0,
        e: float = 1.0,
        f: float = 0.0,
        g: float = 1.0,
    ) -> np.ndarray:
        """
        构造双调和方程的一个精确解：

            w(x,y) = [a cosh(gx) + b sinh(gx) + c x cosh(gx) + d x sinh(gx)]
                     × [e cos(gy) + f sin(gy)]

        验证：∇⁴ w = 0（无体力情况）。
        """
        term_x = (
            a * np.cosh(g * X)
            + b * np.sinh(g * X)
            + c * X * np.cosh(g * X)
            + d * X * np.sinh(g * X)
        )
        term_y = e * np.cos(g * Y) + f * np.sin(g * Y)
        W = term_x * term_y
        return W

    def biharmonic_residual(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        a: float = 1.0,
        b: float = 0.0,
        c: float = 0.0,
        d: float = 0.0,
        e: float = 1.0,
        f: float = 0.0,
        g: float = 1.0,
    ) -> np.ndarray:
        """
        计算双调和残差 R = w_xxxx + 2 w_xxyy + w_yyyy。
        对于精确解，R 理论上为 0。
        """
        # 解析计算各阶导数
        # w_xxxx
        wxxxx = (
            g ** 3
            * (e * np.cos(g * Y) + f * np.sin(g * Y))
            * (
                a * g * np.cosh(g * X)
                + b * g * np.sinh(g * X)
                + c * g * X * np.cosh(g * X)
                + 4.0 * c * np.sinh(g * X)
                + d * g * X * np.sinh(g * X)
                + 4.0 * d * np.cosh(g * X)
            )
        )
        # w_xxyy
        wxxyy = (
            -g ** 3
            * (e * np.cos(g * Y) + f * np.sin(g * Y))
            * (
                a * g * np.cosh(g * X)
                + b * g * np.sinh(g * X)
                + c * g * X * np.cosh(g * X)
                + 2.0 * c * np.sinh(g * X)
                + d * g * X * np.sinh(g * X)
                + 2.0 * d * np.cosh(g * X)
            )
        )
        # w_yyyy
        wyyyy = (
            g ** 4
            * (e * np.cos(g * Y) + f * np.sin(g * Y))
            * (
                a * np.cosh(g * X)
                + b * np.sinh(g * X)
                + c * X * np.cosh(g * X)
                + d * X * np.sinh(g * X)
            )
        )
        R = wxxxx + 2.0 * wxxyy + wyyyy
        return R

    def compute_curvatures(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        a: float = 1.0,
        b: float = 0.0,
        c: float = 0.0,
        d: float = 0.0,
        e: float = 1.0,
        f: float = 0.0,
        g: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算挠度曲率：
            κ_x = ∂²w/∂x²
            κ_y = ∂²w/∂y²
            κ_xy = ∂²w/∂x∂y
        """
        term_y = e * np.cos(g * Y) + f * np.sin(g * Y)
        d2_term_x = (
            a * g ** 2 * np.cosh(g * X)
            + b * g ** 2 * np.sinh(g * X)
            + c * (2.0 * g * np.sinh(g * X) + g ** 2 * X * np.cosh(g * X))
            + d * (2.0 * g * np.cosh(g * X) + g ** 2 * X * np.sinh(g * X))
        )
        d2_term_y = -g ** 2 * (e * np.cos(g * Y) + f * np.sin(g * Y))
        term_x = (
            a * np.cosh(g * X)
            + b * np.sinh(g * X)
            + c * X * np.cosh(g * X)
            + d * X * np.sinh(g * X)
        )
        d_term_y = -g * (e * np.sin(g * Y) - f * np.cos(g * Y))
        d_term_x = (
            a * g * np.sinh(g * X)
            + b * g * np.cosh(g * X)
            + c * (np.cosh(g * X) + g * X * np.sinh(g * X))
            + d * (np.sinh(g * X) + g * X * np.cosh(g * X))
        )

        kappa_x = d2_term_x * term_y
        kappa_y = term_x * d2_term_y
        kappa_xy = d_term_x * d_term_y
        return kappa_x, kappa_y, kappa_xy

    def compute_thermal_stresses(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        delta_T: np.ndarray,
        a: float = 1.0,
        b: float = 0.0,
        c: float = 0.0,
        d: float = 0.0,
        e: float = 1.0,
        f: float = 0.0,
        g: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        计算热应力场：
            σ_x, σ_y, τ_xy, σ_vm
        """
        kappa_x, kappa_y, kappa_xy = self.compute_curvatures(X, Y, a, b, c, d, e, f, g)
        z = self.z_eval
        E = self.E
        nu = self.nu
        alpha_T = self.alpha_T

        sigma_x = -E * z / (1.0 - nu ** 2) * (kappa_x + nu * kappa_y) - E * alpha_T * delta_T / (1.0 - nu)
        sigma_y = -E * z / (1.0 - nu ** 2) * (kappa_y + nu * kappa_x) - E * alpha_T * delta_T / (1.0 - nu)
        tau_xy = -E * z / (1.0 + nu) * kappa_xy

        sigma_vm = np.sqrt(
            sigma_x ** 2 + sigma_y ** 2 - sigma_x * sigma_y + 3.0 * tau_xy ** 2
        )
        return sigma_x, sigma_y, tau_xy, sigma_vm

    def safety_factor(
        self,
        sigma_vm: np.ndarray,
        yield_strength: float = 250.0e6,
    ) -> float:
        """
        计算安全系数：
            SF = σ_yield / max(σ_vm)
        """
        max_stress = np.max(sigma_vm)
        if max_stress < 1.0e-12:
            return float("inf")
        return yield_strength / max_stress

    def thermal_shock_parameter(
        self,
        delta_T_max: float,
        thermal_diffusivity: float = 1.2e-5,
        char_length: float = 1.0e-3,
    ) -> float:
        """
        热冲击参数（Biot 数相关）：
            Bi = h_conv * L / λ
        这里使用无量纲温度梯度参数：
            Θ = α_T · ΔT_max · E / σ_yield
        """
        theta = self.alpha_T * delta_T_max * self.E / 250.0e6
        return theta
