"""
Spectral Methods and Quadrature for Reaction-Diffusion Problems
Integrates concepts from:
- chebyshev2_rule (Gauss-Chebyshev quadrature)
- biharmonic_cheby1d (Chebyshev spectral differentiation)

Applications:
- High-precision integration of reaction rate profiles across film thickness
- Spectral solution of diffusion-reaction boundary value problems
- Polynomial approximation of equilibrium and kinetic data
"""

import numpy as np
from utils import validate_positive, chebyshev_differentiation_matrix, chebyshev_nodes


class ChebyshevQuadrature:
    """
    Gauss-Chebyshev Type 2 quadrature for integrals of the form:
        integral_{a}^{b} f(x) * sqrt((x-a)*(b-x)) dx
    Based on chebyshev2_rule.m (IQPACK algorithm by Elhay & Kautsky).
    """

    def __init__(self, n_points, a=-1.0, b=1.0):
        validate_positive(n_points, "n_points")
        self.n = n_points
        self.a = a
        self.b = b
        self._compute_rule()

    def _compute_rule(self):
        """
        Compute nodes and weights using Jacobi matrix eigenvalue method.
        For Chebyshev Type 2 (weight sqrt(1-x^2)), the recurrence is:
            alpha_j = 0
            beta_j = 1/4  (for j >= 1), beta_0 = pi/2
        """
        n = self.n
        # Jacobi matrix diagonal and subdiagonal
        alpha = np.zeros(n)
        beta = np.zeros(n)
        beta[0] = np.pi / 2.0
        if n > 1:
            beta[1:] = 0.5

        # Build symmetric tridiagonal Jacobi matrix
        J = np.diag(alpha) + np.diag(beta[1:], k=1) + np.diag(beta[1:], k=-1)

        # Eigenvalues = nodes, eigenvectors give weights
        eigenvalues, eigenvectors = np.linalg.eigh(J)
        self.nodes = eigenvalues
        self.weights = beta[0] * eigenvectors[0, :] ** 2

        # Sort nodes in ascending order
        idx = np.argsort(self.nodes)
        self.nodes = self.nodes[idx]
        self.weights = self.weights[idx]

        # Scale to [a, b]
        self.nodes = 0.5 * (self.a + self.b) + 0.5 * (self.b - self.a) * self.nodes
        self.weights = self.weights * ((self.b - self.a) / 2.0) ** 2

    def integrate(self, f):
        """Integrate function f over [a, b] with weight sqrt((x-a)*(b-x))."""
        return np.sum(self.weights * f(self.nodes))


def integrate_reaction_rate_profile(rate_func, z_a, z_b, n=64):
    """
    Integrate reaction rate across a film thickness using high-order quadrature.
    Useful for computing total CO2 absorption flux.
    """
    quad = ChebyshevQuadrature(n, z_a, z_b)
    return quad.integrate(rate_func)


class SpectralDiffusionSolver:
    """
    Solve 1D diffusion-reaction problems using Chebyshev spectral methods.
    Based on biharmonic_cheby1d.m (Trefethen's spectral methods).

    Equation: D * d2c/dz2 - k*c = 0  (film model with first-order reaction)
    Boundary conditions: c(0) = c_i, c(delta) = c_b
    """

    def __init__(self, n_cheb=32):
        validate_positive(n_cheb, "n_cheb")
        self.n = n_cheb + 1
        self.D, self.z = chebyshev_differentiation_matrix(self.n, 0.0, 1.0)
        self.D2 = self.D @ self.D

    def solve_film_diffusion_reaction(self, D_diff, k_rxn, delta, c_interface, c_bulk):
        """
        Solve D*d2c/dz2 - k*c = 0 on [0, delta].
        Returns concentration profile and flux at interface.
        """
        validate_positive(D_diff, "Diffusion coefficient")
        validate_positive(delta, "Film thickness")

        # Scale differentiation matrix to [0, delta]
        scale = 2.0 / delta
        D2_scaled = self.D2 * (scale ** 2)
        z_scaled = 0.5 * delta * (self.z + 1.0)

        # Operator: D * d2c/dz2 - k*c
        L = D_diff * D2_scaled - k_rxn * np.eye(self.n)

        # Boundary conditions: c(0)=c_i, c(delta)=c_b
        L[0, :] = 0.0
        L[0, 0] = 1.0
        L[-1, :] = 0.0
        L[-1, -1] = 1.0

        rhs = np.zeros(self.n)
        rhs[0] = c_interface
        rhs[-1] = c_bulk

        c = np.linalg.solve(L, rhs)

        # Flux at interface: J = -D * dc/dz|_{z=0}
        D_scaled = self.D * scale
        dc_dz = D_scaled[0, :] @ c
        flux = -D_diff * dc_dz

        return z_scaled, c, flux

    def solve_channel_flow(self, mu, delta_p, L, R=0.01):
        """
        Solve Poiseuille flow in a channel: mu * d2u/dy2 = -dp/dx
        Boundary conditions: u(0)=u(R)=0 (no-slip)
        Analytical solution: u(y) = (1/(2*mu)) * (-dp/dx) * y * (R - y)
        """
        n = self.n
        scale = 2.0 / R
        D2_scaled = self.D2 * (scale ** 2)

        L_op = mu * D2_scaled
        rhs = -delta_p / L * np.ones(n)

        # Apply boundary conditions
        L_op[0, :] = 0.0
        L_op[0, 0] = 1.0
        rhs[0] = 0.0
        L_op[-1, :] = 0.0
        L_op[-1, -1] = 1.0
        rhs[-1] = 0.0

        u = np.linalg.solve(L_op, rhs)
        y = 0.5 * R * (self.z + 1.0)
        return y, u


def polynomial_fit_2d_vandermonde(x, y, z, degree=3):
    """
    2D polynomial least-squares fit using Vandermonde matrix.
    Based on vandermonde_approx_2d_coef.m.

    Fits: p(x,y) = sum_{i+j<=degree} c_{ij} * x^i * y^j

    Parameters:
        x, y: coordinates (N,)
        z: values to fit (N,)
        degree: total polynomial degree
    Returns:
        coeffs: polynomial coefficients
        condition_number: Vandermonde matrix condition number
    """
    # TODO: Implement 2D Vandermonde matrix construction and least-squares solve.
    # The coefficient ordering must match evaluate_2d_polynomial.
    raise NotImplementedError("Hole 1: polynomial_fit_2d_vandermonde needs implementation")


def evaluate_2d_polynomial(coeffs, degree, x, y):
    """Evaluate 2D polynomial at points (x, y)."""
    # TODO: Implement 2D polynomial evaluation.
    # The coefficient ordering must match polynomial_fit_2d_vandermonde.
    raise NotImplementedError("Hole 1: evaluate_2d_polynomial needs implementation")
