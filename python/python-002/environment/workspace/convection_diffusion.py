# -*- coding: utf-8 -*-

import numpy as np
from typing import Tuple, Optional, Callable


class ConvectionDiffusion:

    def __init__(self, n_points: int = 256, r_min: float = 1e8, r_max: float = 7e10):
        self.n_points = n_points
        self.r_min = r_min
        self.r_max = r_max

        self.r = np.logspace(np.log10(r_min), np.log10(r_max), n_points)
        self.dr = np.diff(self.r)
        self.dr = np.append(self.dr, self.dr[-1])

    def laplacian_spherical_1d(self, f: np.ndarray, r: np.ndarray) -> np.ndarray:
        f = np.asarray(f, dtype=np.float64)
        n = len(f)
        lap = np.zeros(n, dtype=np.float64)


        for i in range(1, n - 1):
            dr = 0.5 * (r[i + 1] - r[i - 1])
            if dr == 0:
                continue
            d2f = (f[i + 1] - 2.0 * f[i] + f[i - 1]) / (dr ** 2)
            df = (f[i + 1] - f[i - 1]) / (2.0 * dr)
            lap[i] = d2f + (2.0 / r[i]) * df


        lap[0] = lap[1]
        lap[-1] = lap[-2]
        return lap

    def convective_diffusivity(self, r: np.ndarray, rho: np.ndarray, T: np.ndarray,
                               P: np.ndarray, nabla: float, nabla_ad: float,
                               alpha_mlt: float = 1.5) -> np.ndarray:
        G = 6.67430e-8

        rho_avg = np.maximum(rho, 1e-10)
        m_r = (4.0 / 3.0) * np.pi * r ** 3 * rho_avg
        g = G * m_r / r ** 2
        g = np.maximum(g, 1e-5)

        H_P = P / (rho * g)
        H_P = np.clip(H_P, 1e6, 1e14)

        l_mix = alpha_mlt * H_P
        Gamma1 = 5.0 / 3.0
        delta_nabla = max(nabla - nabla_ad, 0.0)
        v_conv_sq = g * l_mix * delta_nabla / (8.0 * Gamma1)
        v_conv_sq = np.maximum(v_conv_sq, 0.0)
        v_conv = np.sqrt(v_conv_sq)

        D_mix = (1.0 / 3.0) * v_conv * l_mix
        D_mix = np.clip(D_mix, 0.0, 1e18)
        return D_mix

    def solve_diffusion_step(self, X: np.ndarray, D: np.ndarray,
                             R: np.ndarray, dt: float, r: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        D_arr = np.asarray(D, dtype=np.float64)
        R_arr = np.asarray(R, dtype=np.float64)
        r_arr = np.asarray(r, dtype=np.float64)

        lap_X = self.laplacian_spherical_1d(X, r_arr)


        dXdt = D_arr * lap_X + R_arr


        dr_min = np.min(np.diff(r_arr))
        D_max = np.max(D_arr)
        if D_max > 0:
            dt_cfl = 0.5 * dr_min ** 2 / D_max
            if dt > dt_cfl:
                dt = dt_cfl

        X_new = X + dt * dXdt
        X_new = np.clip(X_new, 1e-15, 1.0)
        return X_new

    def solve_gray_scott_like(self, U: np.ndarray, V: np.ndarray,
                              Du: float, Dv: float, F: float, k: float,
                              dt: float, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        U = np.asarray(U, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        r_arr = np.asarray(r, dtype=np.float64)

        lap_U = self.laplacian_spherical_1d(U, r_arr)
        lap_V = self.laplacian_spherical_1d(V, r_arr)

        UV2 = U * V ** 2
        dUdt = Du * lap_U - UV2 + F * (1.0 - U)
        dVdt = Dv * lap_V + UV2 - (F + k) * V


        dr_min = np.min(np.diff(r_arr))
        dt_cfl = 0.5 * dr_min ** 2 / max(Du, Dv, 1e-10)
        if dt > dt_cfl:
            dt = dt_cfl

        U_new = U + dt * dUdt
        V_new = V + dt * dVdt
        U_new = np.clip(U_new, 0.0, 1.0)
        V_new = np.clip(V_new, 0.0, 1.0)
        return U_new, V_new

    def mixing_timescale(self, r: np.ndarray, D: np.ndarray) -> float:
        r = np.asarray(r, dtype=np.float64)
        D = np.asarray(D, dtype=np.float64)
        H_P = np.max(r) - np.min(r)
        D_avg = np.mean(D[D > 0]) if np.any(D > 0) else 1e10
        return H_P ** 2 / D_avg if D_avg > 0 else 1e20
