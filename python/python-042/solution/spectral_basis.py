"""
spectral_basis.py

Spectral and interpolation basis construction for mantle convection solver.

Core seed mappings:
- 480_gram_schmidt     -> Gram-Schmidt and modified Gram-Schmidt for orthonormal basis
- 1357_trig_interp_basis-> trigonometric cardinal basis functions for angular discretization
- 635_lagrange_interp_1d-> 1D Lagrange interpolation for radial profiles

Scientific formulas:
- Modified Gram-Schmidt orthogonalization:
    u_j = a_j − Σ_{k=1}^{j−1} proj_{u_k}(a_j)
    e_j = u_j / ‖u_j‖
- Trigonometric cardinal basis (Dirichlet kernel family):
    For odd k:  D_k(x) = sin(kπx/2) / [k sin(πx/2)]
    For even k: D_k(x) = sin(kπx/2) / [k tan(πx/2)]
- Lagrange interpolation polynomial:
    L(x) = Σ_{i=1}^{N} y_i * ℓ_i(x)
    ℓ_i(x) = Π_{j≠i} (x − x_j) / (x_i − x_j)
- Spherical harmonic-like modal expansion for temperature:
    T(r, θ) = Σ_{l=0}^{L_max} Σ_{m=−l}^{l} T_{lm}(r) Y_{lm}(θ, φ)
"""

import numpy as np
from typing import Tuple, Optional


class GramSchmidt:
    """
    Gram-Schmidt orthogonalization for constructing orthonormal bases.
    Adapted from seed 480_gram_schmidt.
    """
    @staticmethod
    def classical(A: np.ndarray) -> np.ndarray:
        """
        Classical Gram-Schmidt orthonormalization of matrix columns.
        Input: A (m, n). Output: U (m, n) with orthonormal columns.
        """
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
        """
        Modified Gram-Schmidt (more numerically stable).
        Returns orthogonal (not necessarily orthonormal) columns.
        """
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
    """
    Trigonometric cardinal basis functions for periodic angular coordinate.
    Adapted from seed 1357_trig_interp_basis.

    These basis functions form a partition of unity on the periodic domain
    and are used for spectral interpolation in the azimuthal direction.
    """
    @staticmethod
    def basis(x: np.ndarray, k: int) -> np.ndarray:
        """
        Evaluate the k-th trigonometric cardinal basis at points x.

        Parameters
        ----------
        x : np.ndarray
            Evaluation points (can be any shape, should be in [-1, 1] for standard mapping).
        k : int
            Order of cardinal function, k >= 1.

        Returns
        -------
        value : np.ndarray
            Basis values, with safe handling at x = 0.
        """
        if k < 1:
            raise ValueError("k must be >= 1")
        x = np.asarray(x, dtype=float)
        eps = np.finfo(float).eps
        # Avoid division by zero
        safe_x = np.where(np.abs(x) < eps, eps, x)
        if k % 2 == 1:
            numerator = np.sin(k * np.pi * safe_x / 2.0)
            denominator = k * np.sin(np.pi * safe_x / 2.0)
        else:
            numerator = np.sin(k * np.pi * safe_x / 2.0)
            denominator = k * np.tan(np.pi * safe_x / 2.0)
        denominator = np.where(np.abs(denominator) < eps, eps, denominator)
        value = numerator / denominator
        # Fix exact zeros
        value[np.abs(x) < eps] = 1.0
        return value

    @staticmethod
    def interpolate(x_nodes: np.ndarray, y_values: np.ndarray,
                    x_eval: np.ndarray) -> np.ndarray:
        """
        Trigonometric interpolation on equispaced nodes using cardinal basis.

        For N equispaced nodes on [-1, 1], the interpolant is:
            S(x) = Σ_{k=1}^{N} y_k * B_k(x)
        where B_k are the trigonometric cardinal functions.
        """
        x_nodes = np.asarray(x_nodes, dtype=float)
        y_values = np.asarray(y_values, dtype=float)
        x_eval = np.asarray(x_eval, dtype=float)
        N = len(x_nodes)
        if N < 1:
            raise ValueError("Need at least one node")
        result = np.zeros_like(x_eval, dtype=float)
        for k in range(1, N + 1):
            # Map to standardized coordinate relative to node k
            # For equispaced nodes, shift so that node k is at 0
            dx = x_eval - x_nodes[k - 1]
            # Normalize by node spacing
            if N > 1:
                h = x_nodes[1] - x_nodes[0]
                if abs(h) < 1e-15:
                    h = 1.0
                dx_norm = dx / h
            else:
                dx_norm = dx
            # Clamp to avoid overflow
            dx_norm = np.clip(dx_norm, -1e3, 1e3)
            result += y_values[k - 1] * TrigonometricBasis.basis(dx_norm, k)
        return result


class LagrangeInterpolation:
    """
    1D Lagrange interpolation for radial mantle property profiles.
    Adapted from seed 635_lagrange_interp_1d.
    """
    @staticmethod
    def basis(nd: int, xd: np.ndarray, ni: int, xi: np.ndarray) -> np.ndarray:
        """
        Compute Lagrange basis matrix LB of shape (ni, nd):
            LB[i, j] = ℓ_j(xi_i)
        """
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
        """
        Evaluate Lagrange interpolant at points xi.
        L(x) = Σ_j yd_j * ℓ_j(x)
        """
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
    """
    Combined spectral expansion using Gram-Schmidt orthonormalized
    radial basis and trigonometric angular basis.
    """
    def __init__(self, n_radial: int = 8, n_angular: int = 16):
        self.n_radial = n_radial
        self.n_angular = n_angular
        self._radial_basis = None
        self._angular_basis = None

    def build_radial_basis(self, r_nodes: np.ndarray, r_eval: np.ndarray) -> np.ndarray:
        """
        Build orthonormal radial basis functions via modified Gram-Schmidt
        applied to Legendre-like polynomials on radial nodes.
        """
        r_nodes = np.asarray(r_nodes, dtype=float)
        r_eval = np.asarray(r_eval, dtype=float)
        n = self.n_radial
        # Construct monomial-like Vandermonde matrix
        V = np.zeros((len(r_eval), n), dtype=float)
        for j in range(n):
            V[:, j] = r_eval ** j
        # Orthonormalize columns
        U = GramSchmidt.classical(V)
        return U

    def build_angular_basis(self, theta_eval: np.ndarray) -> np.ndarray:
        """
        Build trigonometric basis matrix for angular coordinate θ ∈ [0, 2π).
        Returns matrix of shape (len(theta_eval), n_angular).
        """
        theta_eval = np.asarray(theta_eval, dtype=float)
        n = self.n_angular
        B = np.zeros((len(theta_eval), n), dtype=float)
        for k in range(1, n + 1):
            # Map θ to standardized x ∈ [-1, 1]
            x = (theta_eval / np.pi) - 1.0
            x = np.clip(x, -1.0, 1.0)
            B[:, k - 1] = TrigonometricBasis.basis(x, k)
        return B
