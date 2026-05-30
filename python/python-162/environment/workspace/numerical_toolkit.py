
import numpy as np
from typing import Tuple, List, Callable






def cubic_spline_coeffs(x: np.ndarray, y: np.ndarray, bc_type: str = "natural") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(x)
    if len(y) != n or n < 2:
        raise ValueError("x and y must have same length >= 2")
    h = np.diff(x)
    if np.any(h <= 0):
        raise ValueError("x must be strictly increasing")
    a = y.copy()
    alpha = np.zeros(n, dtype=float)
    for i in range(1, n - 1):
        alpha[i] = 3.0 / h[i] * (a[i + 1] - a[i]) - 3.0 / h[i - 1] * (a[i] - a[i - 1])
    c = np.zeros(n, dtype=float)
    l = np.ones(n, dtype=float)
    mu = np.zeros(n, dtype=float)
    z = np.zeros(n, dtype=float)
    if bc_type == "natural":
        l[0] = 1.0
        mu[0] = 0.0
        z[0] = 0.0
    for i in range(1, n - 1):
        l[i] = 2.0 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
        if abs(l[i]) < 1e-14:
            l[i] = 1e-14
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]
    if bc_type == "natural":
        l[n - 1] = 1.0
        z[n - 1] = 0.0
        c[n - 1] = 0.0
    for j in range(n - 2, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
    b = np.zeros(n - 1, dtype=float)
    d = np.zeros(n - 1, dtype=float)
    for i in range(n - 1):
        b[i] = (a[i + 1] - a[i]) / h[i] - h[i] * (c[i + 1] + 2.0 * c[i]) / 3.0
        d[i] = (c[i + 1] - c[i]) / (3.0 * h[i])
    return a, b, c, d


def cubic_spline_eval(x_nodes: np.ndarray, a: np.ndarray, b: np.ndarray,
                      c: np.ndarray, d: np.ndarray, xq: float) -> float:
    n = len(x_nodes)
    if xq <= x_nodes[0]:
        i = 0
    elif xq >= x_nodes[n - 2]:
        i = n - 2
    else:

        lo, hi = 0, n - 2
        while lo < hi:
            mid = (lo + hi) // 2
            if xq < x_nodes[mid]:
                hi = mid
            elif xq > x_nodes[mid + 1]:
                lo = mid + 1
            else:
                lo = hi = mid
        i = lo
    dx = xq - x_nodes[i]
    return a[i] + dx * (b[i] + dx * (c[i] + dx * d[i]))


class TemperatureDependentProperty:

    def __init__(self, temperatures: np.ndarray, values: np.ndarray):
        self.temps = np.asarray(temperatures, dtype=float)
        self.vals = np.asarray(values, dtype=float)
        self.a, self.b, self.c, self.d = cubic_spline_coeffs(self.temps, self.vals)

    def eval(self, T: float) -> float:
        if T <= self.temps[0]:
            return self.vals[0]
        if T >= self.temps[-1]:
            return self.vals[-1]
        return cubic_spline_eval(self.temps, self.a, self.b, self.c, self.d, T)






def lu_factor_dense(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square")
    U = A.copy().astype(float)
    P = np.eye(n, dtype=float)
    for k in range(n - 1):
        pivot = np.argmax(np.abs(U[k:, k])) + k
        if abs(U[pivot, k]) < 1e-15:
            continue
        if pivot != k:
            U[[k, pivot], :] = U[[pivot, k], :]
            P[[k, pivot], :] = P[[pivot, k], :]
        for i in range(k + 1, n):
            U[i, k] /= U[k, k]
            U[i, k + 1:] -= U[i, k] * U[k, k + 1:]
    return P, U


def lu_solve_dense(P: np.ndarray, LU: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = LU.shape[0]
    pb = P @ b
    y = np.zeros(n, dtype=float)
    for i in range(n):
        y[i] = pb[i] - np.dot(LU[i, :i], y[:i])
    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(LU[i, i + 1:], x[i + 1:])) / (LU[i, i] + 1e-18)
    return x






def muller_root(f: Callable[[float], float], x0: float, x1: float, x2: float,
                tol: float = 1e-10, max_iter: int = 50) -> float:
    p0, p1, p2 = float(x0), float(x1), float(x2)
    f0, f1, f2 = f(p0), f(p1), f(p2)
    for _ in range(max_iter):
        h0, h1 = p1 - p0, p2 - p1
        if abs(h0) < 1e-15 or abs(h1) < 1e-15:
            break
        delta0 = (f1 - f0) / h0
        delta1 = (f2 - f1) / h1
        a = (delta1 - delta0) / (h1 + h0)
        b = a * h1 + delta1
        c = f2
        disc = b * b - 4.0 * a * c
        if disc < 0:
            disc = 0.0
        sqrt_disc = np.sqrt(disc)
        if abs(b + sqrt_disc) >= abs(b - sqrt_disc):
            den = b + sqrt_disc
        else:
            den = b - sqrt_disc
        if abs(den) < 1e-15:
            break
        dp = -2.0 * c / den
        p3 = p2 + dp
        if abs(dp) < tol:
            return p3
        p0, p1, p2 = p1, p2, p3
        f0, f1, f2 = f1, f2, f(p3)
    return p2






def rk2_step(f: Callable[[float, np.ndarray], np.ndarray], t: float, y: np.ndarray,
             h: float) -> np.ndarray:
    k1 = f(t, y)
    k2 = f(t + h, y + h * k1)
    return y + 0.5 * h * (k1 + k2)


def rk2_integrate(f: Callable[[float, np.ndarray], np.ndarray], y0: np.ndarray,
                  t_span: Tuple[float, float], n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    t0, tf = t_span
    h = (tf - t0) / n_steps
    t_arr = np.linspace(t0, tf, n_steps + 1)
    y_arr = np.zeros((n_steps + 1, len(y0)), dtype=float)
    y_arr[0] = y0
    for i in range(n_steps):
        y_arr[i + 1] = rk2_step(f, t_arr[i], y_arr[i], h)
    return t_arr, y_arr






def cooley_tukey_fft(x: np.ndarray) -> np.ndarray:
    n = len(x)
    if n == 1:
        return x.copy().astype(complex)
    if n % 2 != 0:

        n2 = 2 ** int(np.ceil(np.log2(n)))
        x_pad = np.zeros(n2, dtype=complex)
        x_pad[:n] = x
        return cooley_tukey_fft(x_pad)[:n]
    even = cooley_tukey_fft(x[0::2])
    odd = cooley_tukey_fft(x[1::2])
    factor = np.exp(-2j * np.pi * np.arange(n // 2) / n)
    return np.concatenate([even + factor * odd, even - factor * odd])


def compute_impedance_spectrum(current_signal: np.ndarray, voltage_signal: np.ndarray,
                               dt: float) -> Tuple[np.ndarray, np.ndarray]:
    n = len(current_signal)
    I_fft = cooley_tukey_fft(current_signal)
    V_fft = cooley_tukey_fft(voltage_signal)
    freqs = np.fft.fftfreq(n, dt)

    mask = np.abs(I_fft) > 1e-12
    Z = np.zeros(n, dtype=complex)
    Z[mask] = V_fft[mask] / I_fft[mask]

    pos = freqs >= 0
    return freqs[pos], Z[pos]
