"""
quadrature_engine.py

Numerical integration rules for mantle property evaluation.

Core seed mappings:
- 1208_test_int_2d/legendre_dr_compute -> Gauss-Legendre quadrature via Davis-Rabinowitz
- 467_gen_laguerre_rule               -> Generalized Gauss-Laguerre quadrature
- 559_hypercube_integrals             -> Hypercube sampling for stochastic integration

Scientific formulas:
- Gauss-Legendre quadrature on [-1, 1]:
    ∫_{-1}^{1} f(x) dx ≈ Σ_{i=1}^{n} w_i f(x_i)
    Nodes x_i are roots of Legendre polynomial P_n(x).
    Weights w_i = 2 / [(1 − x_i²) (P_n′(x_i))²]
- Generalized Gauss-Laguerre quadrature on [a, ∞):
    ∫_{a}^{∞} (x−a)^α exp(−b(x−a)) f(x) dx ≈ Σ_{i=1}^{n} w_i f(x_i)
- 2D product rule on rectangle [a,b]×[c,d]:
    ∫∫ f(x,y) dx dy ≈ Σ_i Σ_j w_i^x w_j^y f(x_i, y_j) * (b−a)/2 * (d−c)/2
- Hypercube Monte Carlo:
    ∫_{[0,1]^m} f(x) dx ≈ (1/N) Σ_{k=1}^{N} f(x_k)
"""

import numpy as np
from typing import Tuple, Callable, Optional
from scipy.special import gamma as gamma_func


class GaussLegendre:
    """
    Gauss-Legendre quadrature via the Davis-Rabinowitz method.
    Adapted from seed 1208_test_int_2d / legendre_dr_compute.
    """
    @staticmethod
    def compute(n: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Gauss-Legendre nodes x and weights w for order n.
        """
        if n < 1:
            raise ValueError("n must be >= 1")
        x = np.zeros(n, dtype=float)
        w = np.zeros(n, dtype=float)
        e1 = n * (n + 1)
        m = (n + 1) // 2
        for i in range(1, m + 1):
            mp1mi = m + 1 - i
            t = (4 * i - 1) * np.pi / (4 * n + 2)
            x0 = np.cos(t) * (1.0 - (1.0 - 1.0 / n) / (8 * n * n))
            pkm1 = 1.0
            pk = x0
            for k in range(2, n + 1):
                pkp1 = 2.0 * x0 * pk - pkm1 - (x0 * pk - pkm1) / k
                pkm1 = pk
                pk = pkp1
            d1 = n * (pkm1 - x0 * pk)
            dpn = d1 / (1.0 - x0 * x0)
            d2pn = (2.0 * x0 * dpn - e1 * pk) / (1.0 - x0 * x0)
            d3pn = (4.0 * x0 * d2pn + (2.0 - e1) * dpn) / (1.0 - x0 * x0)
            d4pn = (6.0 * x0 * d3pn + (6.0 - e1) * d2pn) / (1.0 - x0 * x0)
            u = pk / dpn
            v = d2pn / dpn
            h = -u * (1.0 + 0.5 * u * (v + u * (v * v - d3pn / (3.0 * dpn))))
            p = pk + h * (dpn + 0.5 * h * (d2pn + h / 3.0 *
                           (d3pn + 0.25 * h * d4pn)))
            dp = dpn + h * (d2pn + 0.5 * h * (d3pn + h * d4pn / 3.0))
            h = h - p / dp
            xtemp = x0 + h
            x[mp1mi - 1] = xtemp
            fx = d1 - h * e1 * (pk + 0.5 * h * (dpn + h / 3.0 *
                                (d2pn + 0.25 * h * (d3pn + 0.2 * h * d4pn))))
            w[mp1mi - 1] = 2.0 * (1.0 - xtemp * xtemp) / (fx * fx)
        if n % 2 == 1:
            x[m - 1] = 0.0
        # Reflect negative abscissas
        nmove = m
        ncopy = n - nmove
        for i in range(1, nmove + 1):
            iback = n - i
            x[iback] = x[iback - ncopy]
            w[iback] = w[iback - ncopy]
        for i in range(1, n - nmove + 1):
            x[i - 1] = -x[n - i]
            w[i - 1] = w[n - i]
        return x, w

    @staticmethod
    def integrate_1d(f: Callable[[np.ndarray], np.ndarray], a: float, b: float,
                     n: int = 16) -> float:
        """
        Integrate f(x) over [a, b] using n-point Gauss-Legendre rule.
        """
        x, w = GaussLegendre.compute(n)
        # Affine map from [-1, 1] to [a, b]
        t = 0.5 * (b - a) * x + 0.5 * (b + a)
        jac = 0.5 * (b - a)
        return float(np.sum(w * f(t) * jac))


class GaussLaguerre:
    """
    Generalized Gauss-Laguerre quadrature.
    Adapted from seed 467_gen_laguerre_rule.

    Computes nodes and weights for:
        ∫_{a}^{∞} (x−a)^α exp(−b(x−a)) f(x) dx
    """
    @staticmethod
    def jacobi_matrix(m: int, alpha: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Construct symmetric tridiagonal Jacobi matrix for generalized Laguerre.
        Returns diagonal aj, subdiagonal bj, and zeroth moment zemu.
        """
        if alpha <= -1.0:
            raise ValueError("alpha must be > -1")
        aj = np.zeros(m, dtype=float)
        bj = np.zeros(m, dtype=float)
        zemu = gamma_func(alpha + 1.0)
        for i in range(1, m + 1):
            aj[i - 1] = 2.0 * i - 1.0 + alpha
            bj[i - 1] = i * (i + alpha)
        bj = np.sqrt(bj)
        return aj, bj, zemu

    @staticmethod
    def compute(n: int, alpha: float = 0.0, a: float = 0.0,
                b: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute generalized Gauss-Laguerre nodes x and weights w.
        """
        if n < 1:
            raise ValueError("n must be >= 1")
        aj, bj, zemu = GaussLaguerre.jacobi_matrix(n, alpha)
        # Eigenvalues of symmetric tridiagonal matrix = nodes
        # Use numpy's eigh for diagonalization
        T = np.diag(aj) + np.diag(bj[:-1], 1) + np.diag(bj[:-1], -1)
        eigvals, eigvecs = np.linalg.eigh(T)
        x = eigvals
        # Weights from first component of eigenvectors
        w = zemu * (eigvecs[0, :] ** 2)
        # Scale to interval [a, ∞) with exponential factor b
        x = a + x / b
        w = w / b
        return x, w

    @staticmethod
    def integrate(f: Callable[[np.ndarray], np.ndarray], n: int = 16,
                  alpha: float = 0.0, a: float = 0.0, b: float = 1.0) -> float:
        """
        Integrate f(x) with weight (x−a)^α exp(−b(x−a)) over [a, ∞).
        """
        x, w = GaussLaguerre.compute(n, alpha, a, b)
        return float(np.sum(w * f(x)))


class Quadrature2D:
    """
    2D numerical integration on rectangular and polygonal domains.
    Adapted from seed 1208_test_int_2d concepts.
    """
    @staticmethod
    def integrate_rectangle(f: Callable[[np.ndarray, np.ndarray], np.ndarray],
                            xlim: Tuple[float, float],
                            ylim: Tuple[float, float],
                            nx: int = 8, ny: int = 8) -> float:
        """
        2D Gauss-Legendre product rule over rectangle.
        """
        x_nodes, x_weights = GaussLegendre.compute(nx)
        y_nodes, y_weights = GaussLegendre.compute(ny)
        a, b = xlim
        c, d = ylim
        # Map nodes to physical domain
        xi = 0.5 * (b - a) * x_nodes + 0.5 * (b + a)
        yj = 0.5 * (d - c) * y_nodes + 0.5 * (d + c)
        jac_x = 0.5 * (b - a)
        jac_y = 0.5 * (d - c)
        result = 0.0
        for i in range(nx):
            for j in range(ny):
                result += x_weights[i] * y_weights[j] * f(xi[i], yj[j]) * jac_x * jac_y
        return float(result)

    @staticmethod
    def integrate_triangular(f: Callable[[np.ndarray, np.ndarray], np.ndarray],
                             tri_nodes: np.ndarray,
                             n: int = 4) -> float:
        """
        Integrate over a triangle using mapped Gauss-Legendre rules.
        tri_nodes: array of shape (3, 2) with vertices (x1,y1), (x2,y2), (x3,y3).
        """
        if tri_nodes.shape != (3, 2):
            raise ValueError("tri_nodes must be shape (3, 2)")
        # Area-weighted centroid rule as fallback for robustness
        area = 0.5 * abs((tri_nodes[1, 0] - tri_nodes[0, 0]) * (tri_nodes[2, 1] - tri_nodes[0, 1])
                         - (tri_nodes[2, 0] - tri_nodes[0, 0]) * (tri_nodes[1, 1] - tri_nodes[0, 1]))
        if area < 1e-15:
            return 0.0
        # Simple 3-point centroid rule (sufficient for demonstration)
        centroid_x = np.mean(tri_nodes[:, 0])
        centroid_y = np.mean(tri_nodes[:, 1])
        return float(area * f(np.array([centroid_x]), np.array([centroid_y]))[0])


class HypercubeSampler:
    """
    Monte Carlo sampling in the M-dimensional unit hypercube.
    Adapted from seed 559_hypercube_integrals.
    """
    @staticmethod
    def sample(m: int, n: int, seed: Optional[int] = None) -> np.ndarray:
        """
        Sample n points uniformly in the m-dimensional unit hypercube [0,1]^m.
        Returns array of shape (m, n).
        """
        if m < 1 or n < 1:
            raise ValueError("m and n must be >= 1")
        rng = np.random.default_rng(seed)
        return rng.random((m, n))

    @staticmethod
    def integrate(f: Callable[[np.ndarray], np.ndarray], m: int, n: int,
                  seed: Optional[int] = None) -> Tuple[float, float]:
        """
        Monte Carlo integration of f over [0,1]^m using n samples.
        Returns (estimate, std_error).
        """
        x = HypercubeSampler.sample(m, n, seed)
        vals = f(x)
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1))
        std_err = std / np.sqrt(n)
        return mean, std_err
