
import numpy as np


def r8_csevl(x: float, a: np.ndarray) -> float:
    if x < -1.0 or x > 1.0:
        raise ValueError("Chebyshev 求值点 x 必须在 [-1, 1] 内。")
    n = len(a)
    if n == 0:
        return 0.0
    if n == 1:
        return float(a[0])

    b0 = float(a[-1])
    b1 = 0.0
    b2 = 0.0
    for i in range(n - 2, -1, -1):
        b2 = b1
        b1 = b0
        b0 = 2.0 * x * b1 - b2 + float(a[i])
    return 0.5 * (b0 - b2)


def clausen(x: float) -> float:

    twopi = 2.0 * np.pi
    x_red = x
    while x_red < -np.pi:
        x_red += twopi
    while x_red > np.pi:
        x_red -= twopi

    eps = np.finfo(float).eps
    if abs(x_red) < eps:
        return 0.0


    k = np.arange(1, 20001)
    value = float(np.sum(np.sin(k * x_red) / (k ** 2)))
    return value


def clausen_array(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.vectorize(clausen)(x)


def clausen_activation(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    scale = 6.0 / (np.pi ** 2)
    return scale * clausen_array(np.pi * x)


def clausen_activation_derivative(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    s = np.sin(np.pi * x * 0.5)
    s = np.where(np.abs(s) < 1e-12, 1e-12, s)

    deriv = -np.log(2.0 * np.abs(s))

    deriv = np.clip(deriv, -50.0, 50.0)
    scale = 6.0 / np.pi
    return scale * deriv


def special_function_spectral_basis(x: np.ndarray, n_modes: int = 8) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    basis = np.zeros((N, n_modes))
    for k in range(1, n_modes + 1):
        basis[:, k - 1] = clausen_activation(k * x / n_modes)
    return basis
