
import numpy as np
from utils import safe_log


def cheby_nodes(a, b, n):
    if n < 1:
        raise ValueError("n must be >= 1")
    k = np.arange(n)
    theta = (2.0 * k + 1.0) * np.pi / (2.0 * n)
    c = np.cos(theta)
    return 0.5 * ((1.0 - c) * a + (1.0 + c) * b)


def divided_differences(xd, yd):
    xd = np.array(xd, dtype=float)
    yd = np.array(yd, dtype=float)
    n = xd.size
    dd = yd.copy()
    for i in range(1, n):
        for j in range(n - 1, i - 1, -1):
            dd[j] = (dd[j] - dd[j - 1]) / (xd[j] - xd[j - i])
    return dd


def newton_interp_eval(xd, dd, x):
    xd = np.array(xd, dtype=float)
    dd = np.array(dd, dtype=float)
    x = np.array(x, dtype=float)
    n = dd.size
    y = dd[-1] * np.ones_like(x)
    for i in range(n - 2, -1, -1):
        y = dd[i] + (x - xd[i]) * y
    return y


def chebyshev_approximate_1d(func, a, b, n, ne=10001):
    xd = cheby_nodes(a, b, n)
    yd = func(xd)
    dd = divided_differences(xd, yd)

    xe = np.linspace(a, b, ne)
    ye = newton_interp_eval(xd, dd, xe)
    fe = func(xe)
    maxerr = float(np.max(np.abs(ye - fe)))
    return xd, dd, maxerr


def least_squares_approximant_matrix(nd, xd, m):
    xd = np.array(xd, dtype=float)
    A = np.zeros((nd, m), dtype=float)
    A[:, 0] = 1.0
    for j in range(1, m):
        A[:, j] = A[:, j - 1] * xd
    return A


def least_squares_fit(xd, yd, m):
    xd = np.array(xd, dtype=float)
    yd = np.array(yd, dtype=float)
    nd = xd.size
    if nd < m:
        A = least_squares_approximant_matrix(nd, xd, nd)
        c1 = np.linalg.lstsq(A, yd, rcond=None)[0]

        U, s, Vh = np.linalg.svd(A, full_matrices=False)
        s_inv = np.where(s < np.sqrt(np.finfo(float).eps), 0.0, 1.0 / s)
        c2 = Vh.T @ np.diag(s_inv) @ U.T @ yd
        c = np.zeros(m, dtype=float)
        c[:nd] = c1
        residual = float(np.linalg.norm(A @ c1 - yd))
    else:
        A = least_squares_approximant_matrix(nd, xd, m)
        c1 = np.linalg.lstsq(A, yd, rcond=None)[0]
        U, s, Vh = np.linalg.svd(A, full_matrices=False)
        s_inv = np.where(s < np.sqrt(np.finfo(float).eps), 0.0, 1.0 / s)
        c2 = Vh.T @ np.diag(s_inv) @ U.T @ yd
        c = c1
        residual = float(np.linalg.norm(A @ c - yd))
    return c, residual


def poly_value(c, x):
    x = np.array(x, dtype=float)
    c = np.array(c, dtype=float)
    y = c[-1] * np.ones_like(x)
    for i in range(len(c) - 2, -1, -1):
        y = y * x + c[i]
    return y


class PerformanceSurrogate:
    def __init__(self, model_type='chebyshev'):
        self.model_type = model_type
        self.xd = None
        self.dd = None
        self.coeffs = None
        self.maxerr = None
        self.residual = None
        self.a = None
        self.b = None

    def train(self, feature_range, func, n_nodes, m_poly=None):
        a, b = feature_range
        self.a = a
        self.b = b
        if self.model_type == 'chebyshev':
            self.xd, self.dd, self.maxerr = chebyshev_approximate_1d(
                func, a, b, n_nodes, ne=5001
            )
        elif self.model_type == 'least_squares':
            if m_poly is None:
                m_poly = n_nodes
            xd = np.linspace(a, b, n_nodes)
            yd = func(xd)
            self.coeffs, self.residual = least_squares_fit(xd, yd, m_poly)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def predict(self, x):
        x = np.clip(x, self.a, self.b)
        if self.model_type == 'chebyshev':
            return newton_interp_eval(self.xd, self.dd, x)
        elif self.model_type == 'least_squares':
            return poly_value(self.coeffs, x)
        else:
            raise ValueError("Model not trained")
