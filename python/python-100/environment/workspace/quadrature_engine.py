
import numpy as np
from numpy.linalg import eigvalsh
from scipy.linalg import eigh_tridiagonal


class GaussLegendreQuadrature:

    def __init__(self, n_points: int):
        if n_points < 1:
            raise ValueError("求积点数必须 ≥ 1")
        self.n_points = n_points
        self._build_rule()

    def _build_rule(self):
        n = self.n_points

        alpha = np.zeros(n)
        beta = np.zeros(n - 1)
        for k in range(1, n):
            beta[k - 1] = k / np.sqrt(4.0 * k * k - 1.0)


        eigvals, eigvecs = eigh_tridiagonal(alpha, beta)
        self.nodes = eigvals

        self.weights = 2.0 * eigvecs[0, :] ** 2

    def integrate_1d(self, f, a: float = -1.0, b: float = 1.0) -> float:
        if a >= b:
            raise ValueError("积分上限必须大于下限")
        scale = 0.5 * (b - a)
        shift = 0.5 * (b + a)
        x_phys = shift + scale * self.nodes
        return scale * np.sum(self.weights * f(x_phys))

    def get_nodes_weights_1d(self, a: float = -1.0, b: float = 1.0):
        scale = 0.5 * (b - a)
        shift = 0.5 * (b + a)
        return shift + scale * self.nodes, scale * self.weights


class TriangleGaussianQuadrature:

    def __init__(self, order: int = 7):
        self.order = order
        self._build_dunavant_7()

    def _build_dunavant_7(self):

        a1 = (6.0 + np.sqrt(15.0)) / 21.0
        a2 = (6.0 - np.sqrt(15.0)) / 21.0
        w1 = (155.0 - np.sqrt(15.0)) / 1200.0
        w2 = (155.0 + np.sqrt(15.0)) / 1200.0
        w0 = 9.0 / 40.0

        self.ref_nodes = np.array([
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [a1, a1, 1.0 - 2.0 * a1],
            [a1, 1.0 - 2.0 * a1, a1],
            [1.0 - 2.0 * a1, a1, a1],
            [a2, a2, 1.0 - 2.0 * a2],
            [a2, 1.0 - 2.0 * a2, a2],
            [1.0 - 2.0 * a2, a2, a2],
        ])
        self.ref_weights = np.array([w0, w1, w1, w1, w2, w2, w2])

    def integrate_triangle(self, f, vertices: np.ndarray) -> float:
        vertices = np.asarray(vertices, dtype=float)
        if vertices.shape != (3, 2):
            raise ValueError("顶点数组形状必须为 (3, 2)")


        x1, x2, x3 = vertices[0, 0], vertices[1, 0], vertices[2, 0]
        y1, y2, y3 = vertices[0, 1], vertices[1, 1], vertices[2, 1]
        jac = np.array([[x2 - x1, x3 - x1], [y2 - y1, y3 - y1]])
        det_j = jac[0, 0] * jac[1, 1] - jac[0, 1] * jac[1, 0]
        if abs(det_j) < 1.0e-14:
            raise ValueError("退化三角形: Jacobian行列式接近零")



        xi = self.ref_nodes[:, 1]
        eta = self.ref_nodes[:, 2]
        x_phys = x1 + jac[0, 0] * xi + jac[0, 1] * eta
        y_phys = y1 + jac[1, 0] * xi + jac[1, 1] * eta
        pts = np.column_stack((x_phys, y_phys))


        return 0.5 * abs(det_j) * np.sum(self.ref_weights * f(pts))

    def get_physical_nodes_weights(self, vertices: np.ndarray):
        vertices = np.asarray(vertices, dtype=float)
        x1, x2, x3 = vertices[0, 0], vertices[1, 0], vertices[2, 0]
        y1, y2, y3 = vertices[0, 1], vertices[1, 1], vertices[2, 1]
        jac = np.array([[x2 - x1, x3 - x1], [y2 - y1, y3 - y1]])
        det_j = jac[0, 0] * jac[1, 1] - jac[0, 1] * jac[1, 0]

        xi = self.ref_nodes[:, 1]
        eta = self.ref_nodes[:, 2]
        x_phys = x1 + jac[0, 0] * xi + jac[0, 1] * eta
        y_phys = y1 + jac[1, 0] * xi + jac[1, 1] * eta
        pts = np.column_stack((x_phys, y_phys))
        w_phys = 0.5 * abs(det_j) * self.ref_weights
        return pts, w_phys


class TriangleMonteCarlo:

    def __init__(self, seed: int = None):
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def triangle_area(vertices: np.ndarray) -> float:
        vertices = np.asarray(vertices, dtype=float)
        if vertices.shape != (3, 2):
            raise ValueError("顶点数组形状必须为 (3, 2)")
        x1, y1 = vertices[0]
        x2, y2 = vertices[1]
        x3, y3 = vertices[2]
        return 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

    def _sample_unit_triangle(self, n: int) -> np.ndarray:
        u1 = self.rng.random(n)
        u2 = self.rng.random(n)
        xi = 1.0 - np.sqrt(u1)
        eta = np.sqrt(u1) * (1.0 - u2)
        return np.column_stack((xi, eta))

    def _reference_to_physical(self, vertices: np.ndarray, ref_pts: np.ndarray) -> np.ndarray:
        vertices = np.asarray(vertices, dtype=float)
        ref_pts = np.asarray(ref_pts, dtype=float)
        v1 = vertices[0]
        v2 = vertices[1]
        v3 = vertices[2]
        xi = ref_pts[:, 0]
        eta = ref_pts[:, 1]
        x = v1[0] + (v2[0] - v1[0]) * xi + (v3[0] - v1[0]) * eta
        y = v1[1] + (v2[1] - v1[1]) * xi + (v3[1] - v1[1]) * eta
        return np.column_stack((x, y))

    def integrate(self, f, vertices: np.ndarray, n_samples: int = 10000) -> dict:
        area = self.triangle_area(vertices)
        if area < 1.0e-14:
            raise ValueError("退化三角形面积接近零")

        ref_pts = self._sample_unit_triangle(n_samples)
        phys_pts = self._reference_to_physical(vertices, ref_pts)
        values = f(phys_pts)
        values = np.asarray(values, dtype=float).flatten()

        mean_val = np.mean(values)
        estimate = area * mean_val

        if n_samples > 1:
            var_val = np.var(values, ddof=1)
            std_error = area * np.sqrt(var_val / n_samples)
        else:
            std_error = np.inf

        ci_lower = estimate - 1.96 * std_error
        ci_upper = estimate + 1.96 * std_error

        return {
            "estimate": float(estimate),
            "std_error": float(std_error),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "area": float(area),
        }


class MomentMethodQuadrature:

    def __init__(self, moments: np.ndarray, check_positive: bool = True):
        moments = np.asarray(moments, dtype=float)
        if len(moments) % 2 == 0:
            raise ValueError("矩序列长度必须为奇数 (2n+1)")
        self.n = (len(moments) - 1) // 2
        self.moments = moments
        if check_positive:
            if moments[0] <= 0.0:
                raise ValueError("0阶矩必须为正")
        self._build_from_moments()

    def _build_from_moments(self):
        n = self.n

        H = np.zeros((n + 1, n + 1))
        for i in range(n + 1):
            for j in range(n + 1):
                H[i, j] = self.moments[i + j]


        try:
            R = np.linalg.cholesky(H).T
        except np.linalg.LinAlgError as exc:
            raise ValueError("Hankel矩阵不正定，矩序列可能不一致") from exc


        alpha = np.zeros(n)
        alpha[0] = R[0, 1] / R[0, 0]
        for i in range(1, n):
            alpha[i] = R[i, i + 1] / R[i, i] - R[i - 1, i] / R[i - 1, i - 1]

        beta = np.zeros(n - 1)
        for i in range(n - 1):
            beta[i] = R[i + 1, i + 1] / R[i, i]


        if n == 1:
            self.nodes = np.array([alpha[0]])
            self.weights = np.array([self.moments[0]])
        else:
            eigvals, eigvecs = eigh_tridiagonal(alpha, beta)
            self.nodes = eigvals
            self.weights = self.moments[0] * eigvecs[0, :] ** 2

    def integrate(self, f) -> float:
        return np.sum(self.weights * f(self.nodes))
