
import numpy as np
from typing import Tuple, Optional, Callable


class PDEParameters:
    _defaults = {
        'alpha': 10.0,
        'beta': 4.0,
        'cstar': 0.2,
        'delta': 1.0,
        'epsilon': 0.001,
        'gamma': 1.0,
        'k': 0.75,
        'lambda_': 1.0,
        'mu': 100.0,
        't0': 0.0,
        'tstop': 0.7,
        'xmin': 0.0,
        'xmax': 1.0,
    }

    @classmethod
    def get_defaults(cls) -> dict:
        return cls._defaults.copy()

    @classmethod
    def update_defaults(cls, **kwargs):
        for k, v in kwargs.items():
            if k in cls._defaults:
                cls._defaults[k] = float(v)


def pde_coefficients(x: float, t: float, u: np.ndarray, dudx: np.ndarray,
                     params: Optional[dict] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if params is None:
        params = PDEParameters.get_defaults()

    alpha = params.get('alpha', 10.0)
    beta = params.get('beta', 4.0)
    cstar = params.get('cstar', 0.2)
    delta = params.get('delta', 1.0)
    epsilon = params.get('epsilon', 0.001)
    gamma = params.get('gamma', 1.0)
    k = params.get('k', 0.75)
    lambda_ = params.get('lambda_', 1.0)
    mu = params.get('mu', 100.0)

    c = np.ones(2)
    f = np.array([delta * dudx[0],
                  epsilon * dudx[1] - k * u[1] * dudx[0]])
    g = max(u[0] - cstar, 0.0)
    s = np.array([
        -alpha * u[0] * u[1] / (gamma + u[0] + 1e-12) - lambda_ * u[1],
        mu * u[1] * (1.0 - u[1]) * g - beta * u[1]
    ])
    return c, f, s


def pde_boundary_conditions(xl: float, ul: np.ndarray,
                            xr: float, ur: np.ndarray,
                            t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pl = np.array([ul[0] - 1.0, 0.0])
    ql = np.array([0.0, 1.0])
    pr = np.array([ur[0], ur[1] - 1.0])
    qr = np.array([0.0, 0.0])
    return pl, ql, pr, qr


def pde_initial_condition(x: float) -> np.ndarray:
    c0 = np.cos(0.5 * np.pi * x)
    n0 = 0.0 if x < 1.0 else 1.0
    return np.array([c0, n0])


class BandedMatrix:
    def __init__(self, n: int, ml: int, data: Optional[np.ndarray] = None):
        if n <= 0:
            raise ValueError("矩阵阶数 n 必须为正整数")
        if ml < 0 or ml > n - 1:
            raise ValueError("半带宽 ml 必须在 [0, n-1] 范围内")
        self.n = n
        self.ml = ml
        if data is None:
            self.a = np.zeros((ml + 1, n))
        else:
            if data.shape != (ml + 1, n):
                raise ValueError(f"data形状应为({ml+1},{n})，实际为{data.shape}")
            self.a = data.astype(float).copy()

    @classmethod
    def dif2(cls, n: int, ml: int = 1) -> 'BandedMatrix':
        bm = cls(n, ml)
        bm.a[0, :] = 2.0
        if ml >= 1 and n > 1:
            bm.a[1, :n - 1] = -1.0
        return bm

    def mv(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.shape[0] != self.n:
            raise ValueError(f"向量维度{x.shape[0]}不等于矩阵阶数{self.n}")
        b = self.a[0, :] * x
        for k in range(1, self.ml + 1):
            for j in range(self.n - k):
                i = j + k
                aij = self.a[k, j]
                b[i] += aij * x[j]
                b[j] += aij * x[i]
        return b

    def to_dense(self) -> np.ndarray:
        A = np.zeros((self.n, self.n))
        for i in range(self.n):
            A[i, i] = self.a[0, i]
        for k in range(1, self.ml + 1):
            for j in range(self.n - k):
                i = j + k
                A[i, j] = self.a[k, j]
                A[j, i] = self.a[k, j]
        return A

    def cholesky_band(self) -> np.ndarray:
        L = np.zeros((self.ml + 1, self.n))
        for j in range(self.n):

            sum_sq = self.a[0, j]
            for k in range(1, self.ml + 1):
                if j - k >= 0:
                    sum_sq -= L[k, j - k] ** 2
            if sum_sq <= 0:
                raise ValueError("矩阵非正定，Cholesky分解失败")
            L[0, j] = np.sqrt(sum_sq)

            for i in range(j + 1, min(j + self.ml + 1, self.n)):
                k = i - j
                s = self.a[k, j]
                for m in range(1, self.ml + 1):
                    if j - m >= 0 and k - m >= 0:
                        s -= L[m, j - m] * L[k - m, j - m + k - m]

                if k == 1 and j + 1 < self.n:
                    s = self.a[1, j]
                    if j > 0:
                        s -= L[1, j - 1] * L[0, j - 1]
                L[k, j] = s / L[0, j]
        return L

    def solve_cholesky(self, b: np.ndarray) -> np.ndarray:
        L = self.cholesky_band()
        y = np.zeros(self.n)

        for i in range(self.n):
            y[i] = b[i]
            for k in range(1, self.ml + 1):
                if i - k >= 0:
                    y[i] -= L[k, i - k] * y[i - k]
            y[i] /= L[0, i]

        x = np.zeros(self.n)
        for i in range(self.n - 1, -1, -1):
            x[i] = y[i]
            for k in range(1, self.ml + 1):
                if i + k < self.n:
                    x[i] -= L[k, i] * x[i + k]
            x[i] /= L[0, i]
        return x

    def eigenvalues_dif2(self) -> np.ndarray:
        i = np.arange(1, self.n + 1)
        return 4.0 * np.sin(i * np.pi / (2.0 * (self.n + 1))) ** 2

    def eigenvector_dif2(self, idx: int) -> np.ndarray:
        j = np.arange(1, self.n + 1)
        return np.sqrt(2.0 / (self.n + 1)) * np.sin(idx * j * np.pi / (self.n + 1))


def finite_difference_discretize(n: int, xl: float, xr: float,
                                  pde_func: Optional[Callable] = None,
                                  params: Optional[dict] = None) -> Tuple[BandedMatrix, np.ndarray]:
    if n <= 2:
        raise ValueError("离散化点数n必须大于2")
    h = (xr - xl) / (n + 1)
    if h <= 0:
        raise ValueError("区间长度必须为正")

    A = BandedMatrix.dif2(n, ml=1)
    delta = 1.0 if params is None else params.get('delta', 1.0)
    coeff = delta / (h ** 2)
    A.a[0, :] *= coeff
    A.a[1, :] *= coeff


    x_grid = np.linspace(xl + h, xr - h, n)
    if pde_func is not None:
        for i, xi in enumerate(x_grid):
            _, _, s = pde_func(xi, 0.0, np.zeros(2), np.zeros(2), params)

            V = abs(s[0]) * 0.01
            A.a[0, i] += V
    else:

        V = 0.5 * x_grid ** 2
        A.a[0, :] += V


    f = np.ones(n) * h
    return A, f


def solve_steady_pde(n: int = 128, xl: float = 0.0, xr: float = 1.0) -> np.ndarray:
    params = PDEParameters.get_defaults()
    A, f = finite_difference_discretize(n, xl, xr, pde_coefficients, params)

    pl, ql, pr, qr = pde_boundary_conditions(xl, np.array([1.0, 0.0]),
                                             xr, np.array([0.0, 1.0]), 0.0)

    h = (xr - xl) / (n + 1)
    coeff = params['delta'] / (h ** 2)
    f[0] -= coeff * 1.0
    f[-1] -= coeff * 0.0
    u = A.solve_cholesky(f)
    return u
