
import numpy as np
from utils import validate_positive, chebyshev_differentiation_matrix, chebyshev_nodes


class ChebyshevQuadrature:

    def __init__(self, n_points, a=-1.0, b=1.0):
        validate_positive(n_points, "n_points")
        self.n = n_points
        self.a = a
        self.b = b
        self._compute_rule()

    def _compute_rule(self):
        n = self.n

        alpha = np.zeros(n)
        beta = np.zeros(n)
        beta[0] = np.pi / 2.0
        if n > 1:
            beta[1:] = 0.5


        J = np.diag(alpha) + np.diag(beta[1:], k=1) + np.diag(beta[1:], k=-1)


        eigenvalues, eigenvectors = np.linalg.eigh(J)
        self.nodes = eigenvalues
        self.weights = beta[0] * eigenvectors[0, :] ** 2


        idx = np.argsort(self.nodes)
        self.nodes = self.nodes[idx]
        self.weights = self.weights[idx]


        self.nodes = 0.5 * (self.a + self.b) + 0.5 * (self.b - self.a) * self.nodes
        self.weights = self.weights * ((self.b - self.a) / 2.0) ** 2

    def integrate(self, f):
        return np.sum(self.weights * f(self.nodes))


def integrate_reaction_rate_profile(rate_func, z_a, z_b, n=64):
    quad = ChebyshevQuadrature(n, z_a, z_b)
    return quad.integrate(rate_func)


class SpectralDiffusionSolver:

    def __init__(self, n_cheb=32):
        validate_positive(n_cheb, "n_cheb")
        self.n = n_cheb + 1
        self.D, self.z = chebyshev_differentiation_matrix(self.n, 0.0, 1.0)
        self.D2 = self.D @ self.D

    def solve_film_diffusion_reaction(self, D_diff, k_rxn, delta, c_interface, c_bulk):
        validate_positive(D_diff, "Diffusion coefficient")
        validate_positive(delta, "Film thickness")


        scale = 2.0 / delta
        D2_scaled = self.D2 * (scale ** 2)
        z_scaled = 0.5 * delta * (self.z + 1.0)


        L = D_diff * D2_scaled - k_rxn * np.eye(self.n)


        L[0, :] = 0.0
        L[0, 0] = 1.0
        L[-1, :] = 0.0
        L[-1, -1] = 1.0

        rhs = np.zeros(self.n)
        rhs[0] = c_interface
        rhs[-1] = c_bulk

        c = np.linalg.solve(L, rhs)


        D_scaled = self.D * scale
        dc_dz = D_scaled[0, :] @ c
        flux = -D_diff * dc_dz

        return z_scaled, c, flux

    def solve_channel_flow(self, mu, delta_p, L, R=0.01):
        n = self.n
        scale = 2.0 / R
        D2_scaled = self.D2 * (scale ** 2)

        L_op = mu * D2_scaled
        rhs = -delta_p / L * np.ones(n)


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
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    z = np.asarray(z).flatten()
    N = len(x)


    n_coeffs = (degree + 1) * (degree + 2) // 2

    A = np.zeros((N, n_coeffs))
    col = 0
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            A[:, col] = (x ** i) * (y ** j)
            col += 1


    coeffs, residuals, rank, s = np.linalg.lstsq(A, z, rcond=None)
    cond_num = np.linalg.cond(A)

    return coeffs, cond_num


def evaluate_2d_polynomial(coeffs, degree, x, y):
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    N = len(x)
    z = np.zeros(N)
    col = 0
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            z += coeffs[col] * (x ** i) * (y ** j)
            col += 1
    return z
