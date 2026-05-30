
import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.polynomial.chebyshev import chebgauss


class VandermondeSolver:

    def __init__(self, nodes):
        self.nodes = np.asarray(nodes, dtype=float).flatten()
        self.n = self.nodes.size
        if self.n < 2:
            raise ValueError("Vandermonde matrix dimension must be at least 2.")

        if len(np.unique(np.round(self.nodes, 12))) < self.n:
            raise ValueError("Vandermonde nodes contain duplicates (near-singular).")

    def solve(self, b):
        b = np.asarray(b, dtype=float).flatten()
        if b.size != self.n:
            raise ValueError("b size must match nodes size.")

        a = self.nodes.copy()
        x = b.copy()


        for j in range(self.n - 1):
            for i in range(j + 1, self.n):
                if np.isclose(a[i], a[j]):
                    raise ValueError("Vandermonde matrix is singular: duplicate nodes.")


        for j in range(self.n - 1):
            for i in range(self.n - 1, j, -1):
                x[i] -= a[j] * x[i - 1]

        for j in range(self.n - 2, -1, -1):
            for i in range(j + 1, self.n):
                x[i] /= (a[i] - a[i - j - 1])
            for i in range(j, self.n - 1):
                x[i] -= x[i + 1]

        return x

    def determinant(self):
        det_val = 1.0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                det_val *= (self.nodes[j] - self.nodes[i])
        return det_val

    def to_dense(self):
        V = np.vander(self.nodes, N=self.n, increasing=True)
        return V

    def apply_mv(self, v):
        v = np.asarray(v, dtype=float).flatten()
        if v.size != self.n:
            raise ValueError("v size must match nodes size.")

        y = np.zeros(self.n, dtype=float)
        for i in range(self.n):

            p = v[self.n - 1]
            for j in range(self.n - 2, -1, -1):
                p = p * self.nodes[i] + v[j]
            y[i] = p
        return y


class SpectralDifferentiator:

    def __init__(self, n, node_type='legendre_gauss_lobatto'):
        self.n = int(n)
        if self.n < 2:
            raise ValueError("n must be at least 2.")
        self.node_type = node_type
        self.nodes = self._compute_nodes()
        self.differentiation_matrix = self._compute_dm()

    def _compute_nodes(self):
        if self.node_type == 'legendre_gauss':
            x, _ = leggauss(self.n)
            return x
        elif self.node_type == 'legendre_gauss_lobatto':

            if self.n == 2:
                return np.array([-1.0, 1.0])

            x, _ = leggauss(self.n - 2)
            nodes = np.concatenate([[-1.0], x, [1.0]])

            nodes = self._refine_lgl_nodes(nodes)
            return np.sort(nodes)
        elif self.node_type == 'chebyshev_gauss_lobatto':

            j = np.arange(self.n)
            return np.cos(np.pi * j / (self.n - 1))
        else:
            raise ValueError(f"Unknown node_type: {self.node_type}")

    def _refine_lgl_nodes(self, nodes, max_iter=10, tol=1e-14):
        from numpy.polynomial.legendre import legval, legder
        for _ in range(max_iter):

            c = np.zeros(self.n)
            c[-1] = 1.0
            P = legval(nodes, c)
            dP = legval(nodes, legder(c))


            update = np.zeros_like(nodes)
            mask = np.abs(nodes) < 1.0 - 1e-12
            update[mask] = P[mask] / dP[mask]
            nodes = nodes - update
            if np.max(np.abs(update[mask])) < tol:
                break
        return np.clip(nodes, -1.0, 1.0)

    def _compute_dm(self):
        n = self.n
        x = self.nodes
        D = np.zeros((n, n), dtype=float)

        if self.node_type.startswith('chebyshev'):

            c = np.ones(n)
            c[0] = 2.0
            c[-1] = 2.0
            for i in range(n):
                for j in range(n):
                    if i != j:
                        D[i, j] = (c[i] / c[j]) * ((-1) ** (i + j)) / (x[i] - x[j])
            for i in range(1, n - 1):
                D[i, i] = -x[i] / (2.0 * (1.0 - x[i] ** 2))
            D[0, 0] = (2.0 * (n - 1) ** 2 + 1.0) / 6.0
            D[-1, -1] = -D[0, 0]
        else:


            w = np.ones(n)
            if self.node_type == 'legendre_gauss_lobatto':

                from numpy.polynomial.legendre import legval
                c = np.zeros(n)
                c[-1] = 1.0
                Pn_1 = legval(x, c)
                w = 1.0 / (np.ones(n) - x ** 2)
                w[0] = 0.5 * n * (n - 1)
                w[-1] = 0.5 * n * (n - 1)
                mask = (np.abs(x) < 1.0 - 1e-12)
                w[mask] = 1.0 / (Pn_1[mask] ** 2)
            else:

                for j in range(n):
                    prod = 1.0
                    for k in range(n):
                        if k != j:
                            prod *= (x[j] - x[k])
                    w[j] = 1.0 / prod

            for i in range(n):
                for j in range(n):
                    if i != j:
                        D[i, j] = (w[j] / w[i]) / (x[i] - x[j])

            for i in range(n):
                D[i, i] = -np.sum(D[i, :]) + D[i, i]

        return D

    def differentiate(self, u):
        u = np.asarray(u, dtype=float)
        if u.size != self.n:
            raise ValueError("u size must match n.")
        return self.differentiation_matrix @ u

    def second_derivative_matrix(self):
        return self.differentiation_matrix @ self.differentiation_matrix


def map_nodes_to_interval(nodes, a, b):
    if b <= a:
        raise ValueError("b must be greater than a.")
    x_mapped = 0.5 * (b - a) * nodes + 0.5 * (a + b)

    jacobian = 1.0
    return x_mapped, jacobian


def solve_burgers_spectral_1d(u0_func, x_a, x_b, N, t_span, nu,
                               n_time_steps=1000, node_type='legendre_gauss_lobatto'):
    spec = SpectralDifferentiator(N, node_type=node_type)
    xi = spec.nodes
    x, jac = map_nodes_to_interval(xi, x_a, x_b)

    D = spec.differentiation_matrix / jac
    D2 = spec.second_derivative_matrix() / (jac ** 2)

    u = u0_func(x)
    if np.any(~np.isfinite(u)):
        raise ValueError("Initial condition produced non-finite values.")

    t0, tf = t_span
    dt = (tf - t0) / n_time_steps
    t_vec = np.linspace(t0, tf, n_time_steps + 1)


    U = np.zeros((n_time_steps + 1, N), dtype=float)
    U[0, :] = u


    for n_step in range(n_time_steps):

        def rhs(v):
            v = np.asarray(v, dtype=float)
            v_x = D @ v
            v_xx = D2 @ v
            return -v * v_x + nu * v_xx

        k1 = rhs(u)
        k2 = rhs(u + 0.5 * dt * k1)
        k3 = rhs(u + 0.5 * dt * k2)
        k4 = rhs(u + dt * k3)

        u = u + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


        u[0] = 0.0
        u[-1] = 0.0


        u_max = 10.0 * np.max(np.abs(U[0, :]))
        u = np.clip(u, -u_max, u_max)

        U[n_step + 1, :] = u

    return U, x, t_vec
