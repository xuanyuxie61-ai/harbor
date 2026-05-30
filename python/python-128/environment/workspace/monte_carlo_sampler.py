
import numpy as np


def uniform_in_sphere3d(n: int):
    n = max(1, int(n))
    g = np.random.normal(0.0, 1.0, size=(n, 3))
    norms = np.linalg.norm(g, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    u = g / norms
    r = np.random.uniform(0.0, 1.0, size=(n, 1)) ** (1.0 / 3.0)
    return r * u


def cholesky_upper(A: np.ndarray):
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("cholesky_upper: A 必须为方阵")
    n = A.shape[0]
    U = np.zeros((n, n), dtype=float)
    for i in range(n):
        s = A[i, i] - np.sum(U[:i, i] ** 2)
        if s <= 1e-15:
            raise ValueError("cholesky_upper: 矩阵非正定或接近奇异 (i=%d, s=%g)" % (i, s))
        U[i, i] = np.sqrt(s)
        for j in range(i + 1, n):
            U[i, j] = (A[i, j] - np.dot(U[:i, i], U[:i, j])) / U[i, i]
    return U


def solve_upper_triangular(U: np.ndarray, b: np.ndarray):
    n = U.shape[0]
    x = b.copy().astype(float)
    for i in range(n - 1, -1, -1):
        if abs(U[i, i]) < 1e-15:
            raise ValueError("solve_upper_triangular: 零对角元")
        x[i] = (x[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
    return x


def ellipsoid_sample(m: int, n: int, A: np.ndarray, v: np.ndarray, r: float):
    A = np.asarray(A, dtype=float)
    v = np.asarray(v, dtype=float)
    if A.shape != (m, m):
        raise ValueError("ellipsoid_sample: A 维度不匹配")
    if v.size != m:
        raise ValueError("ellipsoid_sample: v 维度不匹配")

    U = cholesky_upper(A)

    if m == 3:
        Y = uniform_in_sphere3d(n).T
    else:

        Y = np.zeros((m, n), dtype=float)
        accepted = 0
        batch = max(n, 1000)
        while accepted < n:
            g = np.random.normal(0.0, 1.0, size=(m, batch))
            norms = np.linalg.norm(g, axis=0)
            g = g / np.maximum(norms, 1e-15)
            rad = np.random.uniform(0.0, 1.0, size=batch) ** (1.0 / m)
            candidates = g * rad
            for j in range(batch):
                if accepted >= n:
                    break
                Y[:, accepted] = candidates[:, j]
                accepted += 1

    Y *= r
    X = np.zeros((m, n), dtype=float)
    for j in range(n):
        X[:, j] = solve_upper_triangular(U, Y[:, j]) + v
    return X


def ellipsoid_volume_mc(A: np.ndarray, r: float, m: int = 3):
    A = np.asarray(A, dtype=float)
    U = cholesky_upper(A)
    sqrt_det = np.prod(np.diag(U))
    if m == 3:
        V_unit = 4.0 * np.pi / 3.0
    else:

        from math import gamma
        V_unit = np.pi ** (m / 2.0) / gamma(m / 2.0 + 1.0)
    return (r ** m) * V_unit / sqrt_det


class CellMonteCarloSampler:

    def __init__(self, cell_agent, n_samples: int = 500):
        from cell_dynamics import CellAgent
        if not isinstance(cell_agent, CellAgent):
            raise TypeError("CellMonteCarloSampler: 需要 CellAgent 对象")
        self.cell = cell_agent
        self.n_samples = max(10, int(n_samples))

    def sample_cell_body(self):
        a, b, c = self.cell.shape

        A = np.diag([1.0 / (a * a), 1.0 / (b * b), 1.0 / (c * c)])
        v = self.cell.position
        return ellipsoid_sample(3, self.n_samples, A, v, 1.0)

    def estimate_receptor_binding(self, concentration_func):
        pts = self.sample_cell_body()
        Kd = 0.1
        vals = np.zeros(self.n_samples)
        for j in range(self.n_samples):
            c_loc = concentration_func(pts[:, j])
            vals[j] = c_loc / (Kd + c_loc)
        return float(np.mean(vals))

    def estimate_local_volume(self):
        a, b, c = self.cell.shape
        return ellipsoid_volume_mc(np.diag([1.0 / (a * a), 1.0 / (b * b), 1.0 / (c * c)]), 1.0, 3)
