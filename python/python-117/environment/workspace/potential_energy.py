
import numpy as np
from typing import List, Tuple, Callable






def gauss_legendre_1d(n: int) -> Tuple[np.ndarray, np.ndarray]:
    from numpy.polynomial.legendre import leggauss
    x, w = leggauss(n)
    return x.astype(np.float64), w.astype(np.float64)


def product_rule_1d_to_nd(rules_1d: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    d = len(rules_1d)
    orders = [len(r[0]) for r in rules_1d]
    N = int(np.prod(orders))
    x_nd = np.zeros((N, d), dtype=np.float64)
    w_nd = np.ones(N, dtype=np.float64)

    for j in range(N):
        tmp = j
        for k in range(d):
            n_k = orders[k]
            idx = tmp % n_k
            tmp //= n_k
            x_nd[j, k] = rules_1d[k][0][idx]
            w_nd[j] *= rules_1d[k][1][idx]
    return x_nd, w_nd


def integrate_nd(f: Callable[[np.ndarray], np.ndarray],
                 a: np.ndarray,
                 b: np.ndarray,
                 n_per_dim: int = 5) -> float:
    d = len(a)
    rules = []
    jac = 1.0
    for k in range(d):
        x, w = gauss_legendre_1d(n_per_dim)

        scale = (b[k] - a[k]) / 2.0
        shift = (a[k] + b[k]) / 2.0
        rules.append((scale * x + shift, scale * w))
        jac *= scale
    x_nd, w_nd = product_rule_1d_to_nd(rules)
    f_vals = f(x_nd)
    value = np.dot(w_nd, f_vals) * (2.0 ** d) / (2.0 ** d)

    value = np.dot(w_nd, f_vals)
    return float(value)






class CubicSplineInterpolator:

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = np.asarray(x, dtype=np.float64).copy()
        self.y = np.asarray(y, dtype=np.float64).copy()
        self.n = len(self.x)
        if self.n < 4:
            raise ValueError("样条插值至少需要 4 个节点。")
        if np.any(np.diff(self.x) <= 0):
            raise ValueError("节点 x 必须严格递增。")
        self._compute_coefficients()

    def _compute_coefficients(self):
        n = self.n
        h = np.diff(self.x)

        alpha = np.zeros(n, dtype=np.float64)
        for i in range(1, n - 1):
            alpha[i] = (3.0 / h[i]) * (self.y[i + 1] - self.y[i]) - (3.0 / h[i - 1]) * (self.y[i] - self.y[i - 1])



        l = np.ones(n, dtype=np.float64)
        mu = np.zeros(n, dtype=np.float64)
        z = np.zeros(n, dtype=np.float64)

        l[0] = 1.0
        mu[0] = 0.0
        z[0] = 0.0
        for i in range(1, n - 1):
            l[i] = 2.0 * (self.x[i + 1] - self.x[i - 1]) - h[i - 1] * mu[i - 1]
            if abs(l[i]) < 1e-30:
                l[i] = 1e-30
            mu[i] = h[i] / l[i]
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]
        l[n - 1] = 1.0
        z[n - 1] = 0.0

        self.M = np.zeros(n, dtype=np.float64)
        for j in range(n - 2, -1, -1):
            self.M[j] = z[j] - mu[j] * self.M[j + 1]

        self.a = self.y[:-1]
        self.b = np.zeros(n - 1, dtype=np.float64)
        self.c = np.zeros(n - 1, dtype=np.float64)
        self.d = np.zeros(n - 1, dtype=np.float64)
        for j in range(n - 1):
            self.b[j] = (self.y[j + 1] - self.y[j]) / h[j] - h[j] * (2.0 * self.M[j] + self.M[j + 1]) / 6.0
            self.c[j] = self.M[j] / 2.0
            self.d[j] = (self.M[j + 1] - self.M[j]) / (6.0 * h[j])
        self.h = h

    def evaluate(self, xi: np.ndarray) -> np.ndarray:
        xi = np.asarray(xi, dtype=np.float64)
        yi = np.zeros_like(xi)

        for k in range(len(xi)):
            xk = xi[k]

            if xk <= self.x[0]:
                yi[k] = self.y[0]
                continue
            if xk >= self.x[-1]:
                yi[k] = self.y[-1]
                continue

            lo, hi = 0, self.n - 1
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if self.x[mid] <= xk:
                    lo = mid
                else:
                    hi = mid
            j = lo
            dx = xk - self.x[j]
            yi[k] = self.a[j] + self.b[j] * dx + self.c[j] * dx ** 2 + self.d[j] * dx ** 3
        return yi

    def derivative(self, xi: np.ndarray) -> np.ndarray:
        xi = np.asarray(xi, dtype=np.float64)
        dy = np.zeros_like(xi)
        for k in range(len(xi)):
            xk = xi[k]
            if xk <= self.x[0]:
                j = 0
                dx = xk - self.x[0]
            elif xk >= self.x[-1]:
                j = self.n - 2
                dx = xk - self.x[j]
            else:
                lo, hi = 0, self.n - 1
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    if self.x[mid] <= xk:
                        lo = mid
                    else:
                        hi = mid
                j = lo
                dx = xk - self.x[j]
            dy[k] = self.b[j] + 2.0 * self.c[j] * dx + 3.0 * self.d[j] * dx ** 2
        return dy






def membrane_binding_energy_integral(R_np: float = 2.5,
                                     kappa: float = 20.0,
                                     sigma: float = 1.0,
                                     n_quad: int = 5) -> float:
    def f_polar(x):

        r = x[:, 0]
        theta = x[:, 1]


        vals = kappa * np.exp(-r ** 2 / (2.0 * sigma ** 2)) * r
        return vals

    a = np.array([0.0, 0.0])
    b = np.array([R_np, 2.0 * np.pi])
    E_num = integrate_nd(f_polar, a, b, n_quad)

    E_exact = 2.0 * np.pi * kappa * sigma ** 2 * (1.0 - np.exp(-R_np ** 2 / (2.0 * sigma ** 2)))
    return float(E_num), float(E_exact)
