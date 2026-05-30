
import numpy as np
from numpy.linalg import norm


def clenshaw_chebyshev_eval(x, coeffs):
    a = np.asarray(coeffs, dtype=float)
    n = len(a) - 1
    if n < 0:
        return 0.0
    if n == 0:
        return float(a[0])

    b2 = 0.0
    b1 = 0.0
    for k in range(n, 0, -1):
        b0 = a[k] + 2.0 * x * b1 - b2
        b2 = b1
        b1 = b0

    return float(a[0] + x * b1 - b2)


def chebyshev_coefficients_from_function(f, n, a=-1.0, b=1.0):

    j = np.arange(n + 1)
    x_tilde = np.cos(np.pi * j / n)

    x = 0.5 * (b - a) * x_tilde + 0.5 * (b + a)
    fx = np.array([f(xi) for xi in x], dtype=float)
    coeffs = np.zeros(n + 1, dtype=float)
    for k in range(n + 1):
        Tk = np.cos(k * np.arccos(x_tilde))
        if k == 0:
            coeffs[k] = np.sum(fx * Tk) / n
        else:
            coeffs[k] = 2.0 * np.sum(fx * Tk) / n

    coeffs[0] *= 0.5
    coeffs[n] *= 0.5
    return coeffs


def sawtooth_wave(t, omega, amplitude=1.0):
    period = 2.0 * np.pi / omega
    return amplitude * (np.mod(t + np.pi / omega, period) - np.pi / omega)


def sigmoid_activation(x, theta=0.0, sigma=1.0):
    x = np.asarray(x, dtype=float)
    z = -(x - theta) / sigma

    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(z))


def softplus(x):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    mask = x > 100
    out[mask] = x[mask]
    out[~mask] = np.log1p(np.exp(x[~mask]))
    return out


def rk4_step(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
    k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
    k4 = f(t + h, y + h * k3)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def gauss_legendre_nodes_weights(n):
    from numpy.polynomial.legendre import leggauss
    x, w = leggauss(n)
    return x, w


def barycentric_lagrange_interpolate(x_nodes, y_nodes, x_eval):
    x_nodes = np.asarray(x_nodes, dtype=float)
    y_nodes = np.asarray(y_nodes, dtype=float)
    x_eval = np.asarray(x_eval, dtype=float)
    n = len(x_nodes)


    b = np.ones(n)
    b[0] = 0.5
    b[-1] = 0.5
    b[1::2] *= -1.0

    diff = x_eval[:, None] - x_nodes[None, :]

    exact = np.isclose(diff, 0.0)
    if np.any(exact):

        result = np.zeros(len(x_eval), dtype=float)
        for i, xe in enumerate(x_eval):
            mask = np.isclose(x_nodes, xe)
            if np.any(mask):
                result[i] = y_nodes[np.argmax(mask)]
            else:
                num = np.sum(b * y_nodes / (xe - x_nodes))
                den = np.sum(b / (xe - x_nodes))
                result[i] = num / den
        return result
    num = np.sum(b * y_nodes / diff, axis=1)
    den = np.sum(b / diff, axis=1)
    return num / den


def sparse_adjacency_to_laplacian(A):
    A = np.asarray(A, dtype=float)
    degrees = np.sum(A, axis=1)
    D = np.diag(degrees)
    return D - A


def matrix_exponential_power_series(A, t, terms=20):
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    I = np.eye(n)
    result = I.copy()
    term = I.copy()
    for k in range(1, terms):
        term = term @ (A * t) / k
        result += term
        if norm(term, ord='fro') < 1e-15:
            break
    return result


def safe_log1p_exp(x):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x > 20
    neg = x < -20
    mid = ~(pos | neg)
    out[pos] = x[pos]
    out[neg] = np.exp(x[neg])
    out[mid] = np.log1p(np.exp(x[mid]))
    return out


def softmax_stable(x):
    x = np.asarray(x, dtype=float)
    m = np.max(x)
    ex = np.exp(x - m)
    return ex / np.sum(ex)
