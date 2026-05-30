
import numpy as np
from typing import Tuple, List






def hexagon01_area() -> float:
    return 3.0 * np.sqrt(3.0) / 2.0


def _rotate_60(points: np.ndarray) -> np.ndarray:
    c = np.cos(np.pi / 3.0)
    s = np.sin(np.pi / 3.0)
    R = np.array([[c, -s], [s, c]])
    return points @ R.T


def hexagon_lyness_rule03() -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, int]:

    xc = np.array([0.0])
    yc = np.array([0.0])
    wc = np.array([0.5])


    r = 0.830015503296728
    w_rot = np.array([1.0 / 12.0])

    x_rot = np.array([r])
    y_rot = np.array([0.0])


    for _ in range(5):
        new_pts = _rotate_60(np.column_stack([x_rot[-1:], y_rot[-1:]]))
        x_rot = np.append(x_rot, new_pts[0, 0])
        y_rot = np.append(y_rot, new_pts[0, 1])


    x = np.concatenate([xc, x_rot])
    y = np.concatenate([yc, y_rot])
    w = np.concatenate([wc, np.full(6, w_rot[0])])


    area = hexagon01_area()
    w = w / w.sum() * area

    return len(x), x, y, w, 5


def hexagon_lyness_rule07() -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, int]:

    xc, yc, wc = np.array([0.0]), np.array([0.0]), np.array([0.30])


    r1 = 0.520
    w1 = 0.12
    x1 = np.array([r1])
    y1 = np.array([0.0])
    for _ in range(5):
        new = _rotate_60(np.column_stack([x1[-1:], y1[-1:]]))
        x1 = np.append(x1, new[0, 0])
        y1 = np.append(y1, new[0, 1])


    r2 = 0.850
    w2 = 0.08
    x2 = np.array([r2])
    y2 = np.array([0.0])
    for _ in range(5):
        new = _rotate_60(np.column_stack([x2[-1:], y2[-1:]]))
        x2 = np.append(x2, new[0, 0])
        y2 = np.append(y2, new[0, 1])


    r3 = 0.680
    w3 = 0.05
    x3 = np.array([r3])
    y3 = np.array([0.0])
    for _ in range(5):
        new = _rotate_60(np.column_stack([x3[-1:], y3[-1:]]))
        x3 = np.append(x3, new[0, 0])
        y3 = np.append(y3, new[0, 1])

    x = np.concatenate([xc, x1, x2, x3])
    y = np.concatenate([yc, y1, y2, y3])
    w = np.concatenate([wc, np.full(6, w1), np.full(6, w2), np.full(6, w3)])

    area = hexagon01_area()
    w = w / w.sum() * area
    return len(x), x, y, w, 9


def integrate_on_hexagon(f, rule_id: int = 7) -> float:
    if rule_id == 3:
        n, x, y, w, s = hexagon_lyness_rule03()
    elif rule_id == 7:
        n, x, y, w, s = hexagon_lyness_rule07()
    else:
        raise ValueError(f"Unsupported rule_id {rule_id}")

    vals = f(x, y)
    return float(np.dot(w, vals))






def _legendre_polynomial_and_derivative(n: int, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.atleast_1d(x)
    p0 = np.ones_like(x)
    p1 = x.copy()

    if n == 0:
        return p0, np.zeros_like(x)
    if n == 1:
        return p1, np.ones_like(x)

    for k in range(1, n):
        p2 = ((2.0 * k + 1.0) * x * p1 - k * p0) / (k + 1.0)
        p0, p1 = p1, p2


    dp = n * (p0 - x * p1) / (1.0 - x * x + 1e-15)
    return p1, dp


def legendre_gauss_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
    if n < 1:
        raise ValueError("n must be positive")


    i = np.arange(1, n + 1)
    theta = (4.0 * i - 1.0) * np.pi / (4.0 * n + 2.0)
    x = np.cos(theta)


    for _ in range(20):
        p, dp = _legendre_polynomial_and_derivative(n, x)
        dx = p / (dp + 1e-15)
        x_new = x - dx
        if np.max(np.abs(dx)) < 1e-14:
            x = x_new
            break
        x = x_new

    _, dp = _legendre_polynomial_and_derivative(n, x)
    w = 2.0 / ((1.0 - x * x) * dp * dp + 1e-15)

    return x, w


def gauss_legendre_quadrature(f, a: float, b: float, n: int = 64) -> float:
    if a >= b:
        raise ValueError("Interval [a,b] must satisfy a < b")
    t, w = legendre_gauss_nodes_weights(n)
    x = 0.5 * (b + a) + 0.5 * (b - a) * t
    fx = f(x)
    return 0.5 * (b - a) * float(np.dot(w, fx))






def runge_fun(x: np.ndarray) -> np.ndarray:
    x = np.atleast_1d(x)
    return 1.0 / (1.0 + 25.0 * x * x)


def runge_deriv(x: np.ndarray) -> np.ndarray:
    x = np.atleast_1d(x)
    denom = (1.0 + 25.0 * x * x) ** 2
    return -50.0 * x / denom


def runge_deriv2(x: np.ndarray) -> np.ndarray:
    x = np.atleast_1d(x)
    x2 = x * x
    num = 50.0 * (75.0 * x2 - 1.0)
    denom = (1.0 + 25.0 * x2) ** 3
    return num / denom


def runge_antideriv(x: np.ndarray) -> np.ndarray:
    x = np.atleast_1d(x)
    return np.arctan(5.0 * x) / 5.0


def runge_power_series(x: np.ndarray, n_terms: int) -> np.ndarray:
    x = np.atleast_1d(x)
    result = np.zeros_like(x)
    for k in range(n_terms):
        result += ((-1.0) ** k) * ((5.0 * x) ** (2 * k))
    return result


def wss_distribution_analog(x: np.ndarray, peak_wss: float = 7.0,
                            center: float = 0.0, width: float = 0.3) -> np.ndarray:
    x = np.atleast_1d(x)
    xx = (x - center) / width
    return peak_wss * runge_fun(xx)


def lagrange_interpolation_error(nodes: np.ndarray, f, x_test: np.ndarray) -> np.ndarray:
    y_nodes = f(nodes)
    n = len(nodes)
    x_test = np.atleast_1d(x_test)
    result = np.zeros_like(x_test)

    for xi in x_test:

        p = 0.0
        for j in range(n):
            Lj = 1.0
            for k in range(n):
                if k != j:
                    Lj *= (xi - nodes[k]) / (nodes[j] - nodes[k] + 1e-15)
            p += y_nodes[j] * Lj
        result[np.isclose(x_test, xi)] = p

    return np.abs(result - f(x_test))
