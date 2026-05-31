"""
dg_basis.py
===========
High-order polynomial basis functions for Discontinuous Galerkin methods.
Synthesized from lagrange_approx_1d (Lagrange interpolation on Chebyshev nodes)
and gram_polynomial (discrete orthogonal polynomials).
Provides modal and nodal bases on the reference tetrahedron for DG-SEM.
"""

import numpy as np
from typing import List, Tuple
from integer_utils import i4_choose, multi_index_to_dof, dof_count


# ---------------------------------------------------------------------------
# 1D Lagrange basis on Chebyshev nodes
# ---------------------------------------------------------------------------

def chebyshev_extreme_nodes(n: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    """
    Generate n Chebyshev extreme points on [a, b].
    x_i = ((1 - cos(theta_i)) * a + (1 + cos(theta_i)) * b) / 2
    theta_i = (n - 1 - i) * pi / (n - 1)
    """
    if n < 2:
        return np.array([(a + b) / 2.0], dtype=np.float64)
    theta = (n - 1 - np.arange(n, dtype=np.float64)) * np.pi / (n - 1)
    x = ((1.0 - np.cos(theta)) * a + (1.0 + np.cos(theta)) * b) / 2.0
    return x


def lagrange_basis_1d(xd: np.ndarray, x: float) -> np.ndarray:
    """
    Evaluate all Lagrange basis functions at point x for nodes xd.
    L_j(x) = prod_{k!=j} (x - x_k) / (x_j - x_k)
    """
    n = len(xd)
    L = np.ones(n, dtype=np.float64)
    for j in range(n):
        denom = 1.0
        for k in range(n):
            if k != j:
                L[j] *= (x - xd[k])
                denom *= (xd[j] - xd[k])
        if abs(denom) < 1e-30:
            raise ValueError("Duplicate nodes in Lagrange basis.")
        L[j] /= denom
    return L


def lagrange_interpolate_1d(xd: np.ndarray, yd: np.ndarray, x: float) -> float:
    """Lagrange interpolation of data (xd, yd) at point x."""
    L = lagrange_basis_1d(xd, x)
    return float(np.dot(L, yd))


# ---------------------------------------------------------------------------
# Gram (discrete orthogonal) polynomials
# ---------------------------------------------------------------------------

def gram_abscissas(m: int) -> np.ndarray:
    """Generate m equally-spaced abscissas in [-1, 1]."""
    if m < 1:
        raise ValueError("m must be positive.")
    return -1.0 + (2.0 * np.arange(1, m + 1, dtype=np.float64) - 1.0) / m


def gram_beta(n: int, m: int) -> float:
    """
    Three-term recurrence coefficient for Gram polynomials:
    beta_{n,m} = (m+n)(m-n)n^2 / (m^2 (4n^2 - 1))
    """
    if n < 1:
        return 0.0
    return (m + n) * (m - n) * n * n / (m * m * (4.0 * n * n - 1.0))


def gram_polynomial_coefficients(n: int, m: int) -> np.ndarray:
    """
    Compute monomial coefficients of the n-th Gram polynomial
    on m equally-spaced abscissas in [-1, 1].
    Returns coefficients [c0, c1, ..., cn] for p_n(x) = sum c_k x^k.
    """
    if n < 0:
        raise ValueError("n must be non-negative.")
    if n == 0:
        return np.array([1.0 / np.sqrt(m)], dtype=np.float64)
    # p0 and p1 coefficients
    p_prev2 = np.zeros(n + 1, dtype=np.float64)
    p_prev2[0] = 1.0 / np.sqrt(m)  # normalized
    if n == 0:
        return p_prev2
    p_prev1 = np.zeros(n + 1, dtype=np.float64)
    p_prev1[0] = 0.0
    p_prev1[1] = np.sqrt(3.0 * m / (m * m - 1.0))
    if n == 1:
        return p_prev1
    for k in range(2, n + 1):
        beta_k = gram_beta(k - 1, m)
        p_curr = np.zeros(n + 1, dtype=np.float64)
        # x * p_{k-1}(x)
        for j in range(k):
            p_curr[j + 1] += p_prev1[j]
        # - beta * p_{k-2}(x)
        p_curr[:k] -= beta_k * p_prev2[:k]
        # Normalize: discrete L2 norm should be 1
        x = gram_abscissas(m)
        vals = np.polyval(p_curr[::-1], x)
        norm = np.sqrt(np.mean(vals * vals))
        if norm > 1e-30:
            p_curr /= norm
        p_prev2, p_prev1 = p_prev1, p_curr
    return p_prev1


def gram_polynomial_evaluate(n: int, m: int, x: np.ndarray) -> np.ndarray:
    """
    Evaluate Gram polynomials 0..n at points x using stable three-term recurrence.
    Returns array of shape (len(x), n+1).
    """
    x = np.asarray(x, dtype=np.float64)
    nx = x.size
    result = np.zeros((nx, n + 1), dtype=np.float64)
    if n >= 0:
        result[:, 0] = 1.0 / np.sqrt(m)
    if n >= 1:
        result[:, 1] = x * np.sqrt(3.0 * m / (m * m - 1.0))
    for k in range(2, n + 1):
        beta = gram_beta(k - 1, m)
        result[:, k] = x * result[:, k - 1] - beta * result[:, k - 2]
        # Renormalize to unit discrete L2 norm (approximate)
        norm_est = np.sqrt(1.0 / (2.0 * k + 1.0))  # approximate continuous norm
        if norm_est > 1e-30:
            result[:, k] /= norm_est
    return result


def gram_projection(f_vals: np.ndarray, m: int, n: int) -> np.ndarray:
    """
    Project function values f_vals (on m abscissas) onto Gram polynomial basis
    up to degree n. Returns coefficients c_j.
    c_j = <f, g_j> / <g_j, g_j>
    """
    x = gram_abscissas(m)
    G = gram_polynomial_evaluate(n, m, x)
    coeffs = np.zeros(n + 1, dtype=np.float64)
    for j in range(n + 1):
        gj = G[:, j]
        inner_gj = np.mean(gj * gj)
        inner_fg = np.mean(f_vals * gj)
        if abs(inner_gj) > 1e-30:
            coeffs[j] = inner_fg / inner_gj
    return coeffs


# ---------------------------------------------------------------------------
# Multivariate modal basis on reference tetrahedron
# ---------------------------------------------------------------------------

class ModalBasisTet:
    """
    Modal orthonormal polynomial basis on the 3D reference tetrahedron
    with vertices (0,0,0), (1,0,0), (0,1,0), (0,0,1).
    Uses monomial basis orthonormalized via modified Gram-Schmidt with
    numerical integration on the reference simplex.
    """
    def __init__(self, p: int):
        if p < 0:
            raise ValueError("Polynomial degree must be non-negative.")
        self.p = p
        self.dof = dof_count(p, 3)
        self.multi_indices = multi_index_to_dof(p, 3)
        self._orthonormalize()

    def _orthonormalize(self):
        """Numerically orthonormalize monomial basis using 4th-order tetrahedron quadrature."""
        from quadrature_rules import TETRAHEDRON_QUADRATURE_RULES
        rule = TETRAHEDRON_QUADRATURE_RULES[4]
        pts = rule['points']
        wts = rule['weights']
        nq = len(wts)
        # Evaluate monomials at quadrature points
        mono_vals = np.zeros((nq, self.dof), dtype=np.float64)
        for idx, (i, j, k) in enumerate(self.multi_indices):
            mono_vals[:, idx] = (pts[:, 0] ** i) * (pts[:, 1] ** j) * (pts[:, 2] ** k)
        # Modified Gram-Schmidt
        self._basis_vals = np.zeros((nq, self.dof), dtype=np.float64)
        self._grad_vals = np.zeros((nq, self.dof, 3), dtype=np.float64)
        for idx in range(self.dof):
            v = mono_vals[:, idx].copy()
            for j in range(idx):
                # inner product
                ip = np.sum(wts * v * self._basis_vals[:, j])
                v -= ip * self._basis_vals[:, j]
            # Normalize
            norm = np.sqrt(np.sum(wts * v * v))
            if norm > 1e-30:
                v /= norm
            self._basis_vals[:, idx] = v
        # Precompute gradients via finite differences
        h = 1e-8
        for q in range(nq):
            xi, eta, zeta = pts[q]
            v0 = self.evaluate(np.array([xi]), np.array([eta]), np.array([zeta]))[0]
            vxi = self.evaluate(np.array([xi + h]), np.array([eta]), np.array([zeta]))[0]
            veta = self.evaluate(np.array([xi]), np.array([eta + h]), np.array([zeta]))[0]
            vzeta = self.evaluate(np.array([xi]), np.array([eta]), np.array([zeta + h]))[0]
            self._grad_vals[q, :, 0] = (vxi - v0) / h
            self._grad_vals[q, :, 1] = (veta - v0) / h
            self._grad_vals[q, :, 2] = (vzeta - v0) / h

    def evaluate(self, xi: np.ndarray, eta: np.ndarray, zeta: np.ndarray) -> np.ndarray:
        """
        Evaluate all basis functions at points (xi, eta, zeta).
        Uses monomial expansion with stored orthonormal coefficients.
        """
        xi = np.asarray(xi, dtype=np.float64).ravel()
        eta = np.asarray(eta, dtype=np.float64).ravel()
        zeta = np.asarray(zeta, dtype=np.float64).ravel()
        # Clip to reference tetrahedron to avoid numerical roundoff issues
        xi = np.clip(xi, 0.0, 1.0)
        eta = np.clip(eta, 0.0, 1.0)
        zeta = np.clip(zeta, 0.0, 1.0)
        mask = xi + eta + zeta > 1.0
        if np.any(mask):
            s = xi[mask] + eta[mask] + zeta[mask]
            xi[mask] /= s
            eta[mask] /= s
            zeta[mask] /= s
        npts = len(xi)
        mono_vals = np.zeros((npts, self.dof), dtype=np.float64)
        for idx, (i, j, k) in enumerate(self.multi_indices):
            mono_vals[:, idx] = (xi ** i) * (eta ** j) * (zeta ** k)
        # mono_vals @ C gives orthonormal basis, where C is the transformation matrix
        # We can compute C from the orthonormalization: basis = mono @ C
        # C can be precomputed:
        if not hasattr(self, '_C'):
            from quadrature_rules import TETRAHEDRON_QUADRATURE_RULES
            rule = TETRAHEDRON_QUADRATURE_RULES[4]
            pts = rule['points']
            nq = len(pts)
            mono_ref = np.zeros((nq, self.dof), dtype=np.float64)
            for idx, (i, j, k) in enumerate(self.multi_indices):
                mono_ref[:, idx] = (pts[:, 0] ** i) * (pts[:, 1] ** j) * (pts[:, 2] ** k)
            # basis = mono_ref @ C  =>  C = pinv(mono_ref) @ basis_vals
            try:
                self._C = np.linalg.lstsq(mono_ref, self._basis_vals, rcond=None)[0]
            except Exception:
                self._C = np.eye(self.dof)
        return mono_vals @ self._C

    def gradient(self, xi: float, eta: float, zeta: float) -> np.ndarray:
        """
        Evaluate gradients of all basis functions at a single point.
        Returns array of shape (dof, 3).
        """
        h = 1e-8
        # Clip point to reference tetrahedron
        xi = max(0.0, min(1.0, xi))
        eta = max(0.0, min(1.0, eta))
        zeta = max(0.0, min(1.0, zeta))
        s = xi + eta + zeta
        if s > 1.0:
            xi /= s
            eta /= s
            zeta /= s
        v0 = self.evaluate(np.array([xi]), np.array([eta]), np.array([zeta]))[0]
        vxi = self.evaluate(np.array([xi + h]), np.array([eta]), np.array([zeta]))[0]
        veta = self.evaluate(np.array([xi]), np.array([eta + h]), np.array([zeta]))[0]
        vzeta = self.evaluate(np.array([xi]), np.array([eta]), np.array([zeta + h]))[0]
        grad = np.zeros((self.dof, 3), dtype=np.float64)
        grad[:, 0] = (vxi - v0) / h
        grad[:, 1] = (veta - v0) / h
        grad[:, 2] = (vzeta - v0) / h
        return grad


# ---------------------------------------------------------------------------
# Nodal basis on tetrahedron (Lagrange on internal nodes)
# ---------------------------------------------------------------------------

class NodalBasisTet:
    """
    Nodal Lagrange basis on the reference tetrahedron using
    Warp & Blend nodes (approximate) or uniformly spaced nodes.
    """
    def __init__(self, p: int):
        if p < 0:
            raise ValueError("Degree must be non-negative.")
        self.p = p
        self.dof = dof_count(p, 3)
        self.nodes = self._generate_nodes()

    def _generate_nodes(self) -> np.ndarray:
        """Generate nodal points in reference tetrahedron using simple lattice."""
        nodes = []
        for i in range(self.p + 1):
            for j in range(self.p + 1 - i):
                for k in range(self.p + 1 - i - j):
                    xi = i / self.p if self.p > 0 else 0.0
                    eta = j / self.p if self.p > 0 else 0.0
                    zeta = k / self.p if self.p > 0 else 0.0
                    nodes.append([xi, eta, zeta])
        return np.array(nodes, dtype=np.float64)

    def vandermonde(self) -> np.ndarray:
        """Vandermonde matrix V[i,j] = phi_j(node_i) using modal basis."""
        modal = ModalBasisTet(self.p)
        xi = self.nodes[:, 0]
        eta = self.nodes[:, 1]
        zeta = self.nodes[:, 2]
        return modal.evaluate(xi, eta, zeta)

    def evaluate(self, xi: np.ndarray, eta: np.ndarray, zeta: np.ndarray) -> np.ndarray:
        """
        Evaluate nodal basis at arbitrary points using modal expansion.
        Returns shape (n_points, dof).
        """
        modal = ModalBasisTet(self.p)
        V = self.vandermonde()
        try:
            Vinv = np.linalg.inv(V)
        except np.linalg.LinAlgError:
            Vinv = np.linalg.pinv(V)
        modal_vals = modal.evaluate(xi, eta, zeta)
        return modal_vals @ Vinv.T
