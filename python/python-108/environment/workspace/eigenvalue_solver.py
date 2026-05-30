# -*- coding: utf-8 -*-

import numpy as np
from typing import Callable, Tuple, Optional


class BisectionSolver:

    def __init__(self, max_iter: int = 200, tol: float = 1e-12):
        self.max_iter = max_iter
        self.tol = tol

    def solve(self, f: Callable[[float], float], a: float, b: float) -> Tuple[float, int]:
        fa = f(a)
        fb = f(b)
        if fa * fb > 0:
            raise ValueError(f"区间 [{a}, {b}] 两端同号，无法保证根存在")
        if np.isnan(fa) or np.isnan(fb):
            raise ValueError("边界函数值为 NaN")

        for it in range(1, self.max_iter + 1):
            c = 0.5 * (a + b)
            fc = f(c)
            if abs(fc) < 1e-30 or abs(b - a) < self.tol:
                return c, it
            if fa * fc <= 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
        c = 0.5 * (a + b)
        return c, self.max_iter


class CollatzPolynomial:

    def __init__(self, coeffs: np.ndarray):
        self.coeffs = np.array(coeffs, dtype=int) % 2
        self.coeffs = self._trim(self.coeffs)

    @staticmethod
    def _trim(c: np.ndarray) -> np.ndarray:
        if len(c) == 0:
            return np.array([0], dtype=int)
        idx = len(c) - 1
        while idx > 0 and c[idx] == 0:
            idx -= 1
        return c[:idx + 1]

    def degree(self) -> int:
        return len(self.coeffs) - 1

    def next_poly(self) -> "CollatzPolynomial":
        if self.coeffs[0] == 0:

            new_coeffs = self.coeffs[1:] if len(self.coeffs) > 1 else np.array([0], dtype=int)
        else:

            conv = np.convolve(self.coeffs, [1, 1]) % 2
            new_coeffs = (conv + np.array([1] + [0] * (len(conv) - 1))) % 2
        return CollatzPolynomial(new_coeffs)

    def sequence(self, max_steps: int = 100) -> list:
        seq = [self.coeffs.copy()]
        current = self
        for _ in range(max_steps):
            if current.degree() == 0:
                break
            current = current.next_poly()
            seq.append(current.coeffs.copy())
        return seq

    @staticmethod
    def smooth_analog(x: float, max_iter: int = 50, threshold: float = 1e-12) -> float:
        for _ in range(max_iter):
            if abs(x) < threshold:
                return x
            if abs(x) < 1.0:
                x = x / 2.0
            else:
                x = (x * x + x + 1.0) / 3.0
        return x


class ResonanceEigenSolver:

    def __init__(self, R_major: float, n_nominal: float):
        self.R_major = R_major
        self.n_nominal = n_nominal
        self.c = 2.99792458e8

    def resonance_condition(self, m: int, lambda_nm: float, n_eff: float) -> float:
        lambda_m = lambda_nm * 1e-9
        return 2.0 * np.pi * self.R_major * n_eff - m * lambda_m

    def find_resonance_wavelength(self, m: int,
                                   n_eff_func: Callable[[float], float],
                                   lambda_min_nm: float = 1500.0,
                                   lambda_max_nm: float = 1600.0) -> Tuple[float, int]:
        def f(lam):
            return self.resonance_condition(m, lam, n_eff_func(lam))

        bisect = BisectionSolver(max_iter=300, tol=1e-6)
        root, iters = bisect.solve(f, lambda_min_nm, lambda_max_nm)
        return root, iters

    def compute_mode_spacing(self, m: int, lambda_nm: float, n_eff: float,
                             ng: float) -> float:
        lambda_m = lambda_nm * 1e-9
        fsr = lambda_m ** 2 / (2.0 * np.pi * self.R_major * ng)
        return fsr

    def power_iteration_eigenvalue(self, A: np.ndarray, max_iter: int = 500,
                                    tol: float = 1e-10) -> Tuple[float, np.ndarray, int]:
        n = A.shape[0]
        x = np.random.default_rng(42).random(n)
        x = x / np.linalg.norm(x)
        lam = 0.0
        for it in range(1, max_iter + 1):
            y = A @ x
            norm_y = np.linalg.norm(y)
            if norm_y < 1e-30:
                raise ValueError("迭代向量收敛到零")
            y = y / norm_y
            lam_new = float(np.dot(y.conj(), A @ y))
            if abs(lam_new - lam) < tol:
                return lam_new, y, it
            lam = lam_new
            x = y
        return lam, x, max_iter

    def sensitivity_dlambda_dn(self, m: int, lambda_nm: float, n_eff: float) -> float:
        return lambda_nm / n_eff
