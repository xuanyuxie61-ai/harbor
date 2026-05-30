
import numpy as np
from typing import Tuple, Optional


class ThermalStressAnalyzer:

    def __init__(
        self,
        E: float = 200.0e9,
        nu: float = 0.3,
        h: float = 1.0e-3,
        alpha_T: float = 1.2e-5,
        z_eval: float = 0.5e-3,
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
        theta = self.alpha_T * delta_T_max * self.E / 250.0e6
        return theta
