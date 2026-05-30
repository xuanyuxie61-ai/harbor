
import numpy as np


class GenHermiteQuadrature:

    def __init__(self, alpha: float = 0.0, a: float = 0.0, b: float = 1.0, n: int = 20):
        self.alpha = alpha
        self.a = a
        self.b = b
        self.n = n
        self.nodes: np.ndarray | None = None
        self.weights: np.ndarray | None = None

    def _jacobi_matrix(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = self.n
        aj = np.zeros(n)
        bj = np.zeros(n)

        if self.alpha == 0.0 and self.a == 0.0 and self.b == 1.0:

            aj[:] = 0.0
            bj[0] = np.sqrt(np.pi)
            for i in range(1, n):
                bj[i] = i / 2.0
            bj_sqrt = np.sqrt(bj[1:])
            return aj, bj_sqrt, bj





        aj[:] = self.a
        bj[0] = np.sqrt(np.pi / self.b)
        scale = 1.0 / (2.0 * self.b)
        for i in range(1, n):
            bj[i] = i * scale
        bj_sqrt = np.sqrt(bj[1:])
        return aj, bj_sqrt, bj

    def _imtqlx(self, d: np.ndarray, e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = len(d)
        if n == 1:
            return d.copy(), np.array([[1.0]])

        eigvals, eigvecs = np.linalg.eigh(np.diag(d) + np.diag(e, k=1) + np.diag(e, k=-1))
        return eigvals, eigvecs

    def compute_rule(self) -> tuple[np.ndarray, np.ndarray]:
        aj, bj_sqrt, bj_full = self._jacobi_matrix()
        eigvals, eigvecs = self._imtqlx(aj.copy(), bj_sqrt.copy())


        w = bj_full[0] * eigvecs[0, :] ** 2

        self.nodes = eigvals
        self.weights = w
        return self.nodes, self.weights

    def integrate(self, f: callable) -> float:
        if self.nodes is None:
            self.compute_rule()
        return float(np.sum(self.weights * f(self.nodes)))

    def predictive_moments(self, mean: float, std: float,
                           predictive_func: callable) -> tuple[float, float]:
        if self.nodes is None:
            self.compute_rule()
        theta = mean + np.sqrt(2.0) * std * self.nodes
        vals = predictive_func(theta)

        norm_w = self.weights / np.sqrt(np.pi)
        m1 = float(np.sum(norm_w * vals))
        m2 = float(np.sum(norm_w * vals ** 2))
        return m1, m2
