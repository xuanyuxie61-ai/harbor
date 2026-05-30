
import numpy as np


class LatticeIntegrator:

    def __init__(self, dim=2):
        self.dim = dim

    def fibonacci(self, k):
        if k < 1:
            raise ValueError("k must be >= 1")
        a, b = 1, 1
        for _ in range(k - 1):
            a, b = b, a + b
        return a

    def fibonacci_lattice_rule(self, k, func, bounds=None):
        if k < 3:
            raise ValueError("k must be >= 3")
        m = self.fibonacci(k)
        n = self.fibonacci(k - 1)
        z = np.array([1, n])

        if bounds is None:
            bounds = [(0.0, 1.0)] * self.dim

        quad = 0.0
        for j in range(m):
            x_unit = (j * z / m) % 1.0
            x = np.array([
                bounds[d][0] + x_unit[d] * (bounds[d][1] - bounds[d][0])
                for d in range(self.dim)
            ])
            quad += func(x)

        quad /= m


        jac = 1.0
        for d in range(self.dim):
            jac *= (bounds[d][1] - bounds[d][0])
        quad *= jac

        return quad

    def standard_lattice_rule(self, m, z, func, bounds=None):
        if bounds is None:
            bounds = [(0.0, 1.0)] * self.dim

        quad = 0.0
        for j in range(m):
            x_unit = (j * z / m) % 1.0
            x = np.array([
                bounds[d][0] + x_unit[d] * (bounds[d][1] - bounds[d][0])
                for d in range(self.dim)
            ])
            quad += func(x)

        quad /= m
        jac = 1.0
        for d in range(self.dim):
            jac *= (bounds[d][1] - bounds[d][0])
        quad *= jac
        return quad


class JacobiQuadrature:

    def __init__(self):
        pass

    def gamma_ln(self, x):
        from math import log, exp, sqrt, pi, sin
        if x <= 0:
            return float('inf')

        g = 7
        p = [
            0.99999999999980993, 676.5203681218851, -1259.1392167224028,
            771.32342877765313, -176.61502916214059, 12.507343278686905,
            -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7
        ]
        z = x - 1.0
        a = p[0]
        for i in range(1, g + 2):
            a += p[i] / (z + i)
        t = z + g + 0.5
        return log(sqrt(2.0 * pi)) + (z + 0.5) * log(t) - t + log(a)

    def jacobi_quadrature(self, n, alpha, beta):
        if alpha <= -1.0 or beta <= -1.0:
            raise ValueError("alpha and beta must be > -1")

        ab = alpha + beta
        abi = 2.0 + ab


        zemu = np.exp((ab + 1.0) * np.log(2.0)
                      + self.gamma_ln(alpha + 1.0)
                      + self.gamma_ln(beta + 1.0)
                      - self.gamma_ln(abi))


        diag = np.zeros(n)
        offdiag = np.zeros(n)

        diag[0] = (beta - alpha) / abi
        offdiag[0] = np.sqrt(4.0 * (1.0 + alpha) * (1.0 + beta)
                             / ((abi + 1.0) * abi * abi))
        a2b2 = beta * beta - alpha * alpha

        for i in range(1, n):
            abi_i = 2.0 * (i + 1) + ab
            diag[i] = a2b2 / ((abi_i - 2.0) * abi_i)
            abi_sq = abi_i * abi_i
            offdiag[i] = np.sqrt(
                4.0 * (i + 1) * (i + 1 + alpha) * (i + 1 + beta) * (i + 1 + ab)
                / ((abi_sq - 1.0) * abi_sq)
            )




        T = np.diag(diag) + np.diag(offdiag[:-1], k=1) + np.diag(offdiag[:-1], k=-1)
        eigvals, eigvecs = np.linalg.eigh(T)

        x = eigvals
        w = zemu * eigvecs[0, :] ** 2
        return x, w

    def integrate(self, func, n=64, alpha=0.0, beta=0.0, a=-1.0, b=1.0):
        xj, wj = self.jacobi_quadrature(n, alpha, beta)

        t = 0.5 * (b - a) * xj + 0.5 * (b + a)
        jac = 0.5 * (b - a)
        ft = np.array([func(ti) for ti in t])
        result = np.sum(wj * ft) * jac
        return result


class MonteCarloIntegrator:

    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    def integrate_circle(self, func, radius=1.0, n_samples=10000):

        u = np.random.rand(n_samples)
        v = np.random.rand(n_samples)
        r = radius * np.sqrt(u)
        theta = 2.0 * np.pi * v
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        samples = np.array([func(xi, yi) for xi, yi in zip(x, y)])
        area = np.pi * radius ** 2
        mean = np.mean(samples)
        std = np.std(samples, ddof=1)
        result = area * mean
        std_err = area * std / np.sqrt(n_samples)
        return result, std_err

    def integrate_hypersphere_positive(self, func, dim=3, n_samples=10000):
        samples = np.random.randn(dim, n_samples)
        norms = np.linalg.norm(samples, axis=0)
        samples = np.abs(samples) / norms



        from math import gamma, pi
        A = (0.5 ** dim) * 2.0 * (pi ** (dim / 2.0)) / gamma(dim / 2.0)

        f_vals = np.array([func(samples[:, i]) for i in range(n_samples)])
        mean = np.mean(f_vals)
        std = np.std(f_vals, ddof=1)
        result = A * mean
        std_err = A * std / np.sqrt(n_samples)
        return result, std_err

    def integrate_2d_brillouin_zone(self, func, k_max=1e10, n_samples=50000):
        kx = np.random.uniform(-k_max, k_max, n_samples)
        ky = np.random.uniform(-k_max, k_max, n_samples)
        f_vals = np.array([func(kx[i], ky[i]) for i in range(n_samples)])
        area = (2.0 * k_max) ** 2
        mean = np.mean(f_vals)
        std = np.std(f_vals, ddof=1)
        result = area * mean
        std_err = area * std / np.sqrt(n_samples)
        return result, std_err


def test_integrators():

    lat = LatticeIntegrator(dim=2)
    f1 = lambda x: x[0] * x[1]
    q1 = lat.fibonacci_lattice_rule(10, f1)
    print(f"Fibonacci lattice: int x*y = {q1:.8f} (exact=0.25)")


    jq = JacobiQuadrature()
    f2 = lambda x: x ** 2
    q2 = jq.integrate(f2, n=32, alpha=0.5, beta=0.5)

    print(f"Gauss-Jacobi: int x^2 w(x) = {q2:.8f} (exact={np.pi/8:.8f})")


    mc = MonteCarloIntegrator(seed=42)
    f3 = lambda x, y: x ** 2 + y ** 2
    q3, err3 = mc.integrate_circle(f3, radius=1.0, n_samples=20000)

    print(f"MC circle: int (x^2+y^2) = {q3:.8f} ± {err3:.6f} (exact={np.pi/2:.8f})")


if __name__ == "__main__":
    test_integrators()
