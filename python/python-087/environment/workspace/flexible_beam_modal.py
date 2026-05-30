
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import CubicSpline
from typing import List, Tuple, Optional


class EulerBernoulliBeam:
    def __init__(self, L: float, E: float, I: float, rho: float, A: float,
                 boundary: str = "cantilever"):
        if L <= 0 or E <= 0 or I <= 0 or rho <= 0 or A <= 0:
            raise ValueError("所有几何与材料参数必须为正")
        self.L = float(L)
        self.E = float(E)
        self.I = float(I)
        self.rho = float(rho)
        self.A = float(A)
        self.boundary = boundary
        self._beta_roots: Optional[np.ndarray] = None
        self._modes_cached: Optional[List] = None

    @property
    def c0(self) -> float:
        return np.sqrt(self.E * self.I / (self.rho * self.A))

    def frequency_equation(self, beta_L: float) -> float:
        b = float(beta_L)
        if self.boundary == "cantilever":
            return np.cos(b) * np.cosh(b) + 1.0
        elif self.boundary == "pinned_pinned":
            return np.sin(b)
        elif self.boundary == "free_free":
            return np.cos(b) * np.cosh(b) - 1.0
        elif self.boundary == "clamped_clamped":
            return np.cos(b) * np.cosh(b) - 1.0
        else:
            raise ValueError(f"未知边界条件: {self.boundary}")

    def find_beta_roots(self, n_modes: int = 6) -> np.ndarray:
        if n_modes < 1:
            raise ValueError("n_modes 必须 ≥ 1")
        roots = []
        upper = (n_modes + 2) * np.pi

        xs = np.linspace(0.1, upper, 20 * n_modes + 50)
        fs = np.array([self.frequency_equation(x) for x in xs])
        for i in range(len(xs) - 1):
            if fs[i] == 0.0:
                roots.append(xs[i])
            elif fs[i] * fs[i + 1] < 0:
                try:
                    r = brentq(self.frequency_equation, xs[i], xs[i + 1],
                               xtol=1e-14, maxiter=100)
                    if r > 1e-6:
                        roots.append(r)
                except ValueError:
                    pass
        roots = np.array(sorted(set(np.round(roots, 12))))
        if len(roots) < n_modes:

            if self.boundary == "pinned_pinned":
                roots = np.arange(1, n_modes + 1) * np.pi
            else:
                raise RuntimeError(f"仅找到 {len(roots)} 个根，需要 {n_modes} 个")
        self._beta_roots = roots[:n_modes]
        return self._beta_roots

    def modal_shape(self, x: np.ndarray, n: int) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if np.any(x < -1e-12) or np.any(x > self.L + 1e-12):
            raise ValueError("x 必须在 [0, L] 区间内")
        if self._beta_roots is None or n >= len(self._beta_roots):
            self.find_beta_roots(n_modes=max(n + 1, 6))
        b = self._beta_roots[n] / self.L
        bL = self._beta_roots[n]

        if self.boundary == "cantilever":
            sigma = (np.sin(bL) - np.sinh(bL)) / (np.cos(bL) + np.cosh(bL))
            W = (np.cosh(b * x) - np.cos(b * x)
                 - sigma * (np.sinh(b * x) - np.sin(b * x)))
        elif self.boundary == "pinned_pinned":
            W = np.sin(b * x)
        elif self.boundary in ("free_free", "clamped_clamped"):

            sigma = (np.sinh(bL) - np.sin(bL)) / (np.cosh(bL) - np.cos(bL))
            W = (np.cosh(b * x) + np.cos(b * x)
                 - sigma * (np.sinh(b * x) + np.sin(b * x)))
        else:
            raise ValueError(f"未知边界: {self.boundary}")

        norm2 = np.trapezoid(W ** 2, x)
        if norm2 > 0:
            W /= np.sqrt(norm2)
        return W

    def natural_frequencies(self, n_modes: int = 6) -> np.ndarray:
        bL = self.find_beta_roots(n_modes)
        beta = bL / self.L
        omega = beta ** 2 * self.c0
        return omega

    def modal_stiffness_mass(self, n_modes: int = 6) -> Tuple[np.ndarray, np.ndarray]:
        bL = self.find_beta_roots(n_modes)
        beta = bL / self.L

        raise NotImplementedError("Hole 1: modal_stiffness_mass 待实现")

    def displacement_field_spline(self, nodal_coords: np.ndarray,
                                   nodal_displacements: np.ndarray,
                                   num_eval: int = 101) -> Tuple[np.ndarray, np.ndarray]:
        nodal_coords = np.asarray(nodal_coords, dtype=np.float64)
        nodal_displacements = np.asarray(nodal_displacements, dtype=np.float64)
        if len(nodal_coords) < 2:
            raise ValueError("至少需要两个节点才能插值")
        if not np.all(np.diff(nodal_coords) > 0):
            raise ValueError("节点坐标必须严格递增")
        if len(nodal_coords) != len(nodal_displacements):
            raise ValueError("坐标与位移长度不一致")
        cs = CubicSpline(nodal_coords, nodal_displacements, bc_type="not-a-knot")
        x_eval = np.linspace(nodal_coords[0], nodal_coords[-1], num_eval)
        w_eval = cs(x_eval)
        return x_eval, w_eval

    def exact_static_deflection(self, x: np.ndarray, load_type: str = "tip_force",
                                P: float = 1.0) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if np.any(x < 0) or np.any(x > self.L):
            raise ValueError("x 必须在 [0, L] 内")
        if load_type == "tip_force":
            w = (P / (6.0 * self.E * self.I)) * (3.0 * self.L * x ** 2 - x ** 3)
        elif load_type == "uniform":
            q0 = P / self.L
            w = (q0 / (24.0 * self.E * self.I)) * (x ** 4
                  - 4.0 * self.L * x ** 3 + 6.0 * self.L ** 2 * x ** 2)
        else:
            raise ValueError(f"未知载荷类型: {load_type}")
        return w
