
import numpy as np
from typing import Tuple, Callable, Optional
from scipy.special import gamma as gamma_func


class GaussLegendre:
    @staticmethod
    def compute(n: int) -> Tuple[np.ndarray, np.ndarray]:
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
        x, w = GaussLegendre.compute(n)

        t = 0.5 * (b - a) * x + 0.5 * (b + a)
        jac = 0.5 * (b - a)
        return float(np.sum(w * f(t) * jac))


class GaussLaguerre:
    @staticmethod
    def jacobi_matrix(m: int, alpha: float) -> Tuple[np.ndarray, np.ndarray, float]:
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
        if n < 1:
            raise ValueError("n must be >= 1")
        aj, bj, zemu = GaussLaguerre.jacobi_matrix(n, alpha)


        T = np.diag(aj) + np.diag(bj[:-1], 1) + np.diag(bj[:-1], -1)
        eigvals, eigvecs = np.linalg.eigh(T)
        x = eigvals

        w = zemu * (eigvecs[0, :] ** 2)

        x = a + x / b
        w = w / b
        return x, w

    @staticmethod
    def integrate(f: Callable[[np.ndarray], np.ndarray], n: int = 16,
                  alpha: float = 0.0, a: float = 0.0, b: float = 1.0) -> float:
        x, w = GaussLaguerre.compute(n, alpha, a, b)
        return float(np.sum(w * f(x)))


class Quadrature2D:
    @staticmethod
    def integrate_rectangle(f: Callable[[np.ndarray, np.ndarray], np.ndarray],
                            xlim: Tuple[float, float],
                            ylim: Tuple[float, float],
                            nx: int = 8, ny: int = 8) -> float:
        x_nodes, x_weights = GaussLegendre.compute(nx)
        y_nodes, y_weights = GaussLegendre.compute(ny)
        a, b = xlim
        c, d = ylim

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
        if tri_nodes.shape != (3, 2):
            raise ValueError("tri_nodes must be shape (3, 2)")

        area = 0.5 * abs((tri_nodes[1, 0] - tri_nodes[0, 0]) * (tri_nodes[2, 1] - tri_nodes[0, 1])
                         - (tri_nodes[2, 0] - tri_nodes[0, 0]) * (tri_nodes[1, 1] - tri_nodes[0, 1]))
        if area < 1e-15:
            return 0.0

        centroid_x = np.mean(tri_nodes[:, 0])
        centroid_y = np.mean(tri_nodes[:, 1])
        return float(area * f(np.array([centroid_x]), np.array([centroid_y]))[0])


class HypercubeSampler:
    @staticmethod
    def sample(m: int, n: int, seed: Optional[int] = None) -> np.ndarray:
        if m < 1 or n < 1:
            raise ValueError("m and n must be >= 1")
        rng = np.random.default_rng(seed)
        return rng.random((m, n))

    @staticmethod
    def integrate(f: Callable[[np.ndarray], np.ndarray], m: int, n: int,
                  seed: Optional[int] = None) -> Tuple[float, float]:
        x = HypercubeSampler.sample(m, n, seed)
        vals = f(x)
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1))
        std_err = std / np.sqrt(n)
        return mean, std_err
