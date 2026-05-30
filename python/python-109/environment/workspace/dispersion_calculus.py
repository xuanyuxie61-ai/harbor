
import numpy as np
from typing import Callable, Tuple, Optional


def chebyshev_zeros(n: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    if n < 1:
        raise ValueError("chebyshev_zeros: n must be >= 1")
    angles = (2.0 * np.arange(1, n + 1) - 1.0) * np.pi / (2.0 * n)
    x = 0.5 * (a + b) + 0.5 * (b - a) * np.cos(angles)
    return x


def chebyshev_coefficients(a: float, b: float, n: int, f: Callable) -> np.ndarray:
    if n < 1:
        raise ValueError("chebyshev_coefficients: n must be >= 1")
    if b <= a:
        raise ValueError("chebyshev_coefficients: must have b > a")
    x = chebyshev_zeros(n, a, b)
    fx = f(x)
    c = np.zeros(n)
    for j in range(n):
        s = 0.0
        for k in range(n):
            s += fx[k] * np.cos(np.pi * j * (2.0 * k + 1.0) / (2.0 * n))
        c[j] = 2.0 * s / n
    return c


def chebyshev_interpolant(c: np.ndarray, a: float, b: float, x: np.ndarray) -> np.ndarray:
    if b <= a:
        raise ValueError("chebyshev_interpolant: must have b > a")
    n = len(c)
    t = (2.0 * x - a - b) / (b - a)

    t = np.clip(t, -1.0, 1.0)

    if n == 1:
        return np.full_like(x, c[0])
    T0 = np.ones_like(x)
    T1 = t.copy()
    y = c[0] * T0 + c[1] * T1
    for j in range(2, n):
        T2 = 2.0 * t * T1 - T0
        y += c[j] * T2
        T0, T1 = T1, T2
    return y


def cubic_spline_coefficients(x: np.ndarray, y: np.ndarray,
                              derivative: int = 1,
                              muL: Optional[float] = None,
                              muR: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float).flatten()
    y = np.asarray(y, dtype=float).flatten()
    n = len(x)
    if n < 4:
        raise ValueError("cubic_spline_coefficients: need at least 4 points")
    if len(y) != n:
        raise ValueError("cubic_spline_coefficients: x and y must have same length")
    if np.any(np.diff(x) <= 0):
        raise ValueError("cubic_spline_coefficients: x must be strictly increasing")
    Dx = np.diff(x)
    yp = np.diff(y) / Dx

    T = np.zeros((n - 2, n - 2))
    r = np.zeros(n - 2)
    for i in range(1, n - 3):
        T[i, i] = 2.0 * (Dx[i] + Dx[i + 1])
        T[i, i - 1] = Dx[i + 1]
        T[i, i + 1] = Dx[i]
        r[i] = 3.0 * (Dx[i + 1] * yp[i] + Dx[i] * yp[i + 1])
    if muL is not None and muR is not None:
        if derivative == 1:
            T[0, 0] = 2.0 * (Dx[0] + Dx[1])
            T[0, 1] = Dx[0]
            r[0] = 3.0 * (Dx[1] * yp[0] + Dx[0] * yp[1]) - Dx[1] * muL
            T[n - 3, n - 3] = 2.0 * (Dx[n - 3] + Dx[n - 2])
            T[n - 3, n - 4] = Dx[n - 2]
            r[n - 3] = 3.0 * (Dx[n - 2] * yp[n - 3] + Dx[n - 3] * yp[n - 2]) - Dx[n - 3] * muR
            s = np.concatenate([[muL], np.linalg.solve(T, r), [muR]])
        elif derivative == 2:
            T[0, 0] = 2.0 * Dx[0] + 1.5 * Dx[1]
            T[0, 1] = Dx[0]
            r[0] = 1.5 * Dx[1] * yp[0] + 3.0 * Dx[0] * yp[1] + Dx[0] * Dx[1] * muL / 4.0
            T[n - 3, n - 3] = 1.5 * Dx[n - 3] + 2.0 * Dx[n - 2]
            T[n - 3, n - 4] = Dx[n - 2]
            r[n - 3] = (3.0 * Dx[n - 2] * yp[n - 3] + 1.5 * Dx[n - 3] * yp[n - 2]
                        - Dx[n - 2] * Dx[n - 3] * muR / 4.0)
            stilde = np.linalg.solve(T, r)
            s1 = (3.0 * yp[0] - stilde[0] - muL * Dx[0] / 2.0) / 2.0
            sn = (3.0 * yp[n - 2] - stilde[n - 3] + muR * Dx[n - 2] / 2.0) / 2.0
            s = np.concatenate([[s1], stilde, [sn]])
        else:
            raise ValueError("cubic_spline_coefficients: derivative must be 1 or 2")
    else:

        q = Dx[0] * Dx[0] / Dx[1]
        T[0, 0] = 2.0 * Dx[0] + Dx[1] + q
        T[0, 1] = Dx[0] + q
        r[0] = Dx[1] * yp[0] + Dx[0] * yp[1] + 2.0 * yp[1] * (q + Dx[0])
        q = Dx[n - 2] * Dx[n - 2] / Dx[n - 3]
        T[n - 3, n - 3] = 2.0 * Dx[n - 2] + Dx[n - 3] + q
        T[n - 3, n - 4] = Dx[n - 2] + q
        r[n - 3] = (Dx[n - 2] * yp[n - 3] + Dx[n - 3] * yp[n - 2]
                    + 2.0 * yp[n - 2] * (Dx[n - 2] + q))
        stilde = np.linalg.solve(T, r)
        s1 = -stilde[0] + 2.0 * yp[0]
        s1 = s1 + ((Dx[0] / Dx[1]) ** 2) * (stilde[0] + stilde[1] - 2.0 * yp[1])
        sn = -stilde[n - 3] + 2.0 * yp[n - 2]
        sn = sn + ((Dx[n - 2] / Dx[n - 3]) ** 2) * (stilde[n - 4] + stilde[n - 3] - 2.0 * yp[n - 3])
        s = np.concatenate([[s1], stilde, [sn]])
    a = y[:-1].copy()
    b = s[:-1].copy()
    c = (yp - s[:-1]) / Dx
    d = (s[1:] + s[:-1] - 2.0 * yp) / (Dx * Dx)
    return a, b, c, d


def cubic_spline_eval(x: np.ndarray, a: np.ndarray, b: np.ndarray,
                      c: np.ndarray, d: np.ndarray, xk: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = len(xk)
    m = len(a)
    if m != n - 1:
        raise ValueError("cubic_spline_eval: coefficient length mismatch")
    y = np.zeros_like(x)
    for i in range(len(x)):
        xi = x[i]

        if xi <= xk[0]:
            idx = 0
        elif xi >= xk[n - 2]:
            idx = n - 2
        else:
            idx = int(np.searchsorted(xk[1:-1], xi))
        dx = xi - xk[idx]
        y[i] = a[idx] + b[idx] * dx + c[idx] * dx * dx + d[idx] * dx * dx * dx
    return y


def lebesgue_function(n: int, x_nodes: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    if len(x_nodes) != n:
        raise ValueError("lebesgue_function: node count mismatch")
    m = len(x_eval)
    lfun = np.zeros(m)
    for j in range(n):
        lj = np.ones(m)
        for k in range(n):
            if k != j:
                denom = x_nodes[j] - x_nodes[k]
                if abs(denom) < 1e-15:
                    continue
                lj *= (x_eval - x_nodes[k]) / denom
        lfun += np.abs(lj)
    return lfun


def lebesgue_constant(n: int, x_nodes: np.ndarray, x_eval: np.ndarray) -> float:
    lfun = lebesgue_function(n, x_nodes, x_eval)
    return float(np.max(lfun))


def dispersion_taylor_coefficients(omega: np.ndarray, beta: np.ndarray,
                                    omega0: float, order: int = 6) -> np.ndarray:
    if order < 2:
        raise ValueError("dispersion_taylor_coefficients: order must be >= 2")
    omega = np.asarray(omega, dtype=float)
    beta = np.asarray(beta, dtype=float)

    delta = np.max(np.abs(omega - omega0)) * 0.3
    mask = np.abs(omega - omega0) <= delta
    if np.sum(mask) < order + 2:

        mask = np.abs(omega - omega0) <= delta * 2.0
    x_local = omega[mask] - omega0
    y_local = beta[mask]


    x_scale = np.max(np.abs(x_local))
    if x_scale < 1e-30:
        x_scale = 1.0
    x_scaled = x_local / x_scale


    V = np.vander(x_scaled, order + 1, increasing=True)
    coeffs_scaled, _, _, _ = np.linalg.lstsq(V, y_local, rcond=None)

    coeffs = np.zeros_like(coeffs_scaled)
    for m in range(len(coeffs_scaled)):
        coeffs[m] = coeffs_scaled[m] / (x_scale ** m)
    return coeffs


def sellmeier_equation_silica(wavelength_um: np.ndarray) -> np.ndarray:
    lam2 = wavelength_um ** 2
    B = np.array([0.6961663, 0.4079426, 0.8974794])
    C = np.array([0.004679148, 0.01351206, 98.96161])
    n2 = np.ones_like(wavelength_um)
    for Bi, Ci in zip(B, C):
        n2 += Bi * lam2 / (lam2 - Ci)
    return np.sqrt(n2)


def beta_from_sellmeier(wavelength_um: np.ndarray) -> np.ndarray:
    n = sellmeier_equation_silica(wavelength_um)

    beta = 2.0 * np.pi * n / (wavelength_um * 1e-6)
    return beta
