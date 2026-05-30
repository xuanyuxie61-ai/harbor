
import numpy as np
from typing import Tuple, Optional


class GramSchmidt:
    @staticmethod
    def classical(A: np.ndarray) -> np.ndarray:
        A = np.asarray(A, dtype=float)
        m, n = A.shape
        U = np.zeros((m, n), dtype=float)
        for j in range(n):
            v = A[:, j].copy()
            for j2 in range(j):
                vu = float(np.dot(v, U[:, j2]))
                v = v - vu * U[:, j2]
            v_norm = float(np.linalg.norm(v))
            if v_norm > 1e-15:
                U[:, j] = v / v_norm
        return U

    @staticmethod
    def modified(A: np.ndarray) -> np.ndarray:
        A = np.asarray(A, dtype=float)
        m, n = A.shape
        U = A.copy()
        for j in range(n):
            v = U[:, j].copy()
            for j2 in range(j + 1, n):
                v2 = U[:, j2].copy()
                denom = float(np.dot(v, v))
                if denom > 1e-15:
                    p = float(np.dot(v, v2)) / denom
                    v2 = v2 - p * v
                    U[:, j2] = v2
        return U


class TrigonometricBasis:
    @staticmethod
    def basis(x: np.ndarray, k: int) -> np.ndarray:
        if k < 1:
            raise ValueError("k must be >= 1")
        x = np.asarray(x, dtype=float)
        eps = np.finfo(float).eps

        safe_x = np.where(np.abs(x) < eps, eps, x)
        if k % 2 == 1:
            numerator = np.sin(k * np.pi * safe_x / 2.0)
            denominator = k * np.sin(np.pi * safe_x / 2.0)
        else:
            numerator = np.sin(k * np.pi * safe_x / 2.0)
            denominator = k * np.tan(np.pi * safe_x / 2.0)
        denominator = np.where(np.abs(denominator) < eps, eps, denominator)
        value = numerator / denominator

        value[np.abs(x) < eps] = 1.0
        return value

    @staticmethod
    def interpolate(x_nodes: np.ndarray, y_values: np.ndarray,
                    x_eval: np.ndarray) -> np.ndarray:
        x_nodes = np.asarray(x_nodes, dtype=float)
        y_values = np.asarray(y_values, dtype=float)
        x_eval = np.asarray(x_eval, dtype=float)
        N = len(x_nodes)
        if N < 1:
            raise ValueError("Need at least one node")
        result = np.zeros_like(x_eval, dtype=float)
        for k in range(1, N + 1):


            dx = x_eval - x_nodes[k - 1]

            if N > 1:
                h = x_nodes[1] - x_nodes[0]
                if abs(h) < 1e-15:
                    h = 1.0
                dx_norm = dx / h
            else:
                dx_norm = dx

            dx_norm = np.clip(dx_norm, -1e3, 1e3)
            result += y_values[k - 1] * TrigonometricBasis.basis(dx_norm, k)
        return result


class LagrangeInterpolation:
    @staticmethod
    def basis(nd: int, xd: np.ndarray, ni: int, xi: np.ndarray) -> np.ndarray:
        xd = np.asarray(xd, dtype=float).ravel()
        xi = np.asarray(xi, dtype=float).ravel()
        if nd < 1:
            raise ValueError("nd must be >= 1")
        lb = np.ones((ni, nd), dtype=float)
        for i in range(ni):
            for j in range(nd):
                for k in range(nd):
                    if j != k:
                        diff = xd[j] - xd[k]
                        if abs(diff) < 1e-15:
                            diff = 1e-15
                        lb[i, j] *= (xi[i] - xd[k]) / diff
        return lb

    @staticmethod
    def interpolate(xd: np.ndarray, yd: np.ndarray, xi: np.ndarray) -> np.ndarray:
        xd = np.asarray(xd, dtype=float).ravel()
        yd = np.asarray(yd, dtype=float).ravel()
        xi = np.asarray(xi, dtype=float).ravel()
        nd = len(xd)
        ni = len(xi)
        if nd != len(yd):
            raise ValueError("xd and yd must have same length")
        lb = LagrangeInterpolation.basis(nd, xd, ni, xi)
        return lb @ yd


class SpectralExpansion:
    def __init__(self, n_radial: int = 8, n_angular: int = 16):
        self.n_radial = n_radial
        self.n_angular = n_angular
        self._radial_basis = None
        self._angular_basis = None

    def build_radial_basis(self, r_nodes: np.ndarray, r_eval: np.ndarray) -> np.ndarray:
        r_nodes = np.asarray(r_nodes, dtype=float)
        r_eval = np.asarray(r_eval, dtype=float)
        n = self.n_radial

        V = np.zeros((len(r_eval), n), dtype=float)
        for j in range(n):
            V[:, j] = r_eval ** j

        U = GramSchmidt.classical(V)
        return U

    def build_angular_basis(self, theta_eval: np.ndarray) -> np.ndarray:
        theta_eval = np.asarray(theta_eval, dtype=float)
        n = self.n_angular
        B = np.zeros((len(theta_eval), n), dtype=float)
        for k in range(1, n + 1):

            x = (theta_eval / np.pi) - 1.0
            x = np.clip(x, -1.0, 1.0)
            B[:, k - 1] = TrigonometricBasis.basis(x, k)
        return B
