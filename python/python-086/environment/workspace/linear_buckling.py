# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import jv, yv, jvp
from scipy.optimize import newton


def bessel_zero_halley(n: float, k: int, kind: int = 1, tol: float = 1e-14, max_iter: int = 100) -> float:
    if kind not in (1, 2):
        raise ValueError("kind 必须为 1 或 2")

    if k == 1:
        x0 = 2.0 * np.abs(n) + 1.857 * np.abs(n) ** 0.333 + 1.0
    else:

        x0 = (k + 0.5 * np.abs(n) - 0.25) * np.pi
    x = x0
    for _ in range(max_iter):
        if kind == 1:
            f = jv(n, x)
            fp = jvp(n, x, 1)
            fpp = jvp(n, x, 2)
        else:
            f = yv(n, x)
            fp = jvp(n, x, 1)

            fpp = jvp(n, x, 2)
        denom = 2.0 * fp * fp - f * fpp
        if abs(denom) < 1e-20:
            break
        dx = 2.0 * f * fp / denom
        x_new = x - dx
        if abs(dx) < tol:
            return float(x_new)
        x = x_new
    return float(x)


def bessel_zeros_vector(n: float, k_max: int, kind: int = 1) -> np.ndarray:
    zeros = []
    for k in range(1, k_max + 1):
        z = bessel_zero_halley(n, k, kind)
        zeros.append(z)
    return np.array(zeros)


class LinearBucklingAnalyzer:

    def __init__(self, geometry, material):
        self.geom = geometry
        self.mat = material
        self.D = self.mat.bending_rigidity(self.geom.t)
        self.C = self.mat.extensional_rigidity(self.geom.t)

    def analytical_buckling_load(self) -> float:
        E, nu = self.mat.E, self.mat.nu
        t, R = self.geom.t, self.geom.R
        Ncr = E * t ** 2 / (R * np.sqrt(3.0 * (1.0 - nu ** 2)))
        return float(Ncr)

    def buckling_modes_discrete(self, m_max: int = 10, n_max: int = 10) -> tuple:
        R, L, t = self.geom.R, self.geom.L, self.geom.t
        D = self.D
        E = self.mat.E
        nu = self.mat.nu
        N_min = float('inf')
        m_opt = 1
        n_opt = 0
        modes = []
        for m in range(1, m_max + 1):
            alpha = m * np.pi * R / L
            for n in range(0, n_max + 1):
                beta = float(n)
                if alpha == 0:
                    continue
                term1 = (alpha ** 2 + beta ** 2) ** 2 / alpha ** 2
                term2 = alpha ** 2 / (alpha ** 2 + beta ** 2) ** 2

                Nx = (D / (R ** 2 * t)) * term1 + (E * t / (1.0 - nu ** 2)) * term2
                modes.append((m, n, Nx))
                if Nx < N_min:
                    N_min = Nx
                    m_opt = m
                    n_opt = n
        return float(N_min), m_opt, n_opt, modes

    def bessel_verification(self, n_circumferential: int, n_zeros: int = 5) -> np.ndarray:
        n = float(n_circumferential)
        zeros = bessel_zeros_vector(n, n_zeros, kind=1)
        return zeros

    def imperfection_sensitivity_koiter(self, imperfection_amplitude: float,
                                       imperfection_mode: int) -> float:
        nu = self.mat.nu
        delta = imperfection_amplitude
        t = self.geom.t

        a = 1.5 * np.sqrt(3.0 * (1.0 - nu ** 2))
        ratio = max(0.0, 1.0 - a * (delta / t))
        return float(ratio)
