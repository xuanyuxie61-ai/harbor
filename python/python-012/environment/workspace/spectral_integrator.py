"""
Spectral Integration Methods for Brillouin Zone Integrals
=========================================================
Implements high-precision numerical integration over the 2D Brillouin zone
using:
1. Lattice rules (Fibonacci lattice, project 654)
2. Gauss-Jacobi quadrature (project 607)
3. Monte Carlo sampling with importance sampling

These methods are essential for computing transport coefficients,
Chern numbers, and DOS integrals over the BZ.

For a periodic function f(k) on the torus [0,1)^2, the lattice rule:
    I ≈ (1/M) sum_{j=0}^{M-1} f( (j * z / M) mod 1 )

where z is the generator vector. For 2D, optimal Fibonacci lattices
are used.

For integrals with endpoint singularities (e.g., DOS near band edges),
Gauss-Jacobi quadrature with weight (1-x)^alpha (1+x)^beta is used.
"""

import numpy as np


class LatticeIntegrator:
    """
    Brillouin zone integration using lattice rules and related methods.
    Based on lattice_rule and fibonacci_lattice from project 654.
    """

    def __init__(self, dim=2):
        self.dim = dim

    def fibonacci(self, k):
        """
        Compute the k-th Fibonacci number.
        F(1) = 1, F(2) = 1, F(n) = F(n-1) + F(n-2).
        """
        if k < 1:
            raise ValueError("k must be >= 1")
        a, b = 1, 1
        for _ in range(k - 1):
            a, b = b, a + b
        return a

    def fibonacci_lattice_rule(self, k, func, bounds=None):
        """
        Apply a 2D Fibonacci lattice integration rule.

        M = F(k), z = [1, F(k-1)].
        Quad = (1/M) sum_{j=0}^{M-1} f( mod(j * z / M, 1) ).

        Parameters
        ----------
        k : int
            Fibonacci index, k >= 3.
        func : callable
            Function f(x) where x is a 1D array of length dim.
        bounds : list of tuples, optional
            [(xmin, xmax), (ymin, ymax)] to map [0,1]^2 to the domain.

        Returns
        -------
        quad : float
            Estimated integral.
        """
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

        # Jacobian factor
        jac = 1.0
        for d in range(self.dim):
            jac *= (bounds[d][1] - bounds[d][0])
        quad *= jac

        return quad

    def standard_lattice_rule(self, m, z, func, bounds=None):
        """
        Apply a general lattice integration rule.

        Quad = (1/M) sum_{j=0}^{M-1} f( mod(j * z / M, 1) ).

        Parameters
        ----------
        m : int
            Number of points.
        z : ndarray
            Generator vector of length dim.
        func : callable
        bounds : list of tuples, optional

        Returns
        -------
        quad : float
        """
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
    """
    Gauss-Jacobi quadrature for integrals with algebraic singularities.
    Based on j_quadrature_rule from project 607.
    """

    def __init__(self):
        pass

    def gamma_ln(self, x):
        """Logarithm of gamma function using Lanczos approximation."""
        from math import log, exp, sqrt, pi, sin
        if x <= 0:
            return float('inf')
        # Lanczos coefficients
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
        """
        Compute Gauss-Jacobi quadrature nodes and weights.

        Integral_{-1}^{1} (1-x)^alpha (1+x)^beta f(x) dx ≈ sum_i w_i f(x_i)

        Parameters
        ----------
        n : int
            Number of quadrature points.
        alpha, beta : float
            Jacobi parameters, alpha, beta > -1.

        Returns
        -------
        x, w : ndarray
            Nodes and weights.
        """
        if alpha <= -1.0 or beta <= -1.0:
            raise ValueError("alpha and beta must be > -1")

        ab = alpha + beta
        abi = 2.0 + ab

        # Zero-th moment
        zemu = np.exp((ab + 1.0) * np.log(2.0)
                      + self.gamma_ln(alpha + 1.0)
                      + self.gamma_ln(beta + 1.0)
                      - self.gamma_ln(abi))

        # Jacobi matrix (tridiagonal)
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

        # Eigenvalue problem for symmetric tridiagonal matrix
        # T = diag(diag) + offdiag on sub/super-diagonal
        # Use numpy's eigh on the full symmetric matrix
        T = np.diag(diag) + np.diag(offdiag[:-1], k=1) + np.diag(offdiag[:-1], k=-1)
        eigvals, eigvecs = np.linalg.eigh(T)

        x = eigvals
        w = zemu * eigvecs[0, :] ** 2
        return x, w

    def integrate(self, func, n=64, alpha=0.0, beta=0.0, a=-1.0, b=1.0):
        """
        Integrate func over [a, b] using Gauss-Jacobi quadrature.

        Parameters
        ----------
        func : callable
        n : int
        alpha, beta : float
            Singularity exponents at a and b.
        a, b : float
            Integration limits.

        Returns
        -------
        result : float
        """
        xj, wj = self.jacobi_quadrature(n, alpha, beta)
        # Map from [-1, 1] to [a, b]
        t = 0.5 * (b - a) * xj + 0.5 * (b + a)
        jac = 0.5 * (b - a)
        ft = np.array([func(ti) for ti in t])
        result = np.sum(wj * ft) * jac
        return result


class MonteCarloIntegrator:
    """
    Monte Carlo integration with importance sampling for BZ integrals.
    Inspired by hypersphere_positive_sample and circle_unit_sample
    from projects 567 and 178.
    """

    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    def integrate_circle(self, func, radius=1.0, n_samples=10000):
        """
        Integrate over a circular domain using uniform sampling on the disk.

        Inspired by circle_unit_sample (project 178).

        Parameters
        ----------
        func : callable
            f(x, y) where (x, y) is a point in the disk.
        radius : float
        n_samples : int

        Returns
        -------
        result : float
        std_err : float
        """
        # Sample uniformly in disk: r = R*sqrt(u), theta = 2*pi*v
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
        """
        Integrate over the positive octant of a unit hypersphere.

        Inspired by hypersphere_positive_sample (project 567).
        Sampling: generate normal random vector, normalize, take absolute value.

        Parameters
        ----------
        func : callable
            f(x) where x is a point on the positive hypersphere.
        dim : int
        n_samples : int

        Returns
        -------
        result : float
        std_err : float
        """
        samples = np.random.randn(dim, n_samples)
        norms = np.linalg.norm(samples, axis=0)
        samples = np.abs(samples) / norms

        # Surface area of positive octant of unit (dim-1)-sphere
        # A = (1/2^dim) * 2 * pi^(dim/2) / Gamma(dim/2)
        from math import gamma, pi
        A = (0.5 ** dim) * 2.0 * (pi ** (dim / 2.0)) / gamma(dim / 2.0)

        f_vals = np.array([func(samples[:, i]) for i in range(n_samples)])
        mean = np.mean(f_vals)
        std = np.std(f_vals, ddof=1)
        result = A * mean
        std_err = A * std / np.sqrt(n_samples)
        return result, std_err

    def integrate_2d_brillouin_zone(self, func, k_max=1e10, n_samples=50000):
        """
        Monte Carlo integration over a 2D square BZ [-k_max, k_max]^2.

        Parameters
        ----------
        func : callable
            f(kx, ky).
        k_max : float
        n_samples : int

        Returns
        -------
        result : float
        std_err : float
        """
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
    """Test integration methods on known integrals."""
    # Test 1: Fibonacci lattice rule for int_0^1 int_0^1 x*y dx dy = 1/4
    lat = LatticeIntegrator(dim=2)
    f1 = lambda x: x[0] * x[1]
    q1 = lat.fibonacci_lattice_rule(10, f1)
    print(f"Fibonacci lattice: int x*y = {q1:.8f} (exact=0.25)")

    # Test 2: Gauss-Jacobi for int_{-1}^1 (1-x)^0.5 (1+x)^0.5 x^2 dx
    jq = JacobiQuadrature()
    f2 = lambda x: x ** 2
    q2 = jq.integrate(f2, n=32, alpha=0.5, beta=0.5)
    # Exact: pi/8
    print(f"Gauss-Jacobi: int x^2 w(x) = {q2:.8f} (exact={np.pi/8:.8f})")

    # Test 3: Monte Carlo over unit circle
    mc = MonteCarloIntegrator(seed=42)
    f3 = lambda x, y: x ** 2 + y ** 2
    q3, err3 = mc.integrate_circle(f3, radius=1.0, n_samples=20000)
    # Exact: pi/2
    print(f"MC circle: int (x^2+y^2) = {q3:.8f} ± {err3:.6f} (exact={np.pi/2:.8f})")


if __name__ == "__main__":
    test_integrators()
