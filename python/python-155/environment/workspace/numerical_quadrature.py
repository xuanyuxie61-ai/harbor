import numpy as np
from typing import Callable, Optional, Tuple





def hexagon_area() -> float:
    return 3.0 * np.sqrt(3.0) / 2.0


def hexagon_stroud_rule1() -> Tuple[np.ndarray, np.ndarray]:
    pts = np.array([[0.0, 0.0]])
    w = np.array([hexagon_area()])
    return pts, w


def hexagon_stroud_rule2() -> Tuple[np.ndarray, np.ndarray]:
    r = np.sqrt(3.0) / 3.0
    pts = np.array([[r, r], [-r, r], [-r, -r], [r, -r]])
    w = np.full(4, hexagon_area() / 4.0)
    return pts, w


def hexagon_stroud_rule3() -> Tuple[np.ndarray, np.ndarray]:
    r = np.sqrt(6.0) / 3.0
    pts = np.array([
        [0.0, 0.0],
        [r, 0.0],
        [-r / 2.0, r * np.sqrt(3.0) / 2.0],
        [-r / 2.0, -r * np.sqrt(3.0) / 2.0],
        [r / 2.0, r * np.sqrt(3.0) / 2.0],
        [r / 2.0, -r * np.sqrt(3.0) / 2.0],
        [-r, 0.0]
    ])
    w = np.full(7, hexagon_area() / 7.0)
    return pts, w


def hexagon_stroud_rule4() -> Tuple[np.ndarray, np.ndarray]:
    r1 = np.sqrt(14.0) / 5.0
    r2 = np.sqrt(42.0) / 10.0
    pts = np.array([
        [0.0, 0.0],
        [r1, 0.0],
        [-r1 / 2.0, r1 * np.sqrt(3.0) / 2.0],
        [-r1 / 2.0, -r1 * np.sqrt(3.0) / 2.0],
        [r2, 0.0],
        [-r2 / 2.0, r2 * np.sqrt(3.0) / 2.0],
        [-r2 / 2.0, -r2 * np.sqrt(3.0) / 2.0]
    ])
    w = np.array([0.5, 0.125, 0.125, 0.125, 0.125, 0.0, 0.0]) * hexagon_area()
    w[5] = w[6] = (hexagon_area() - np.sum(w[:5])) / 2.0
    return pts, w


def integrate_hexagon(f: Callable[[np.ndarray, np.ndarray], np.ndarray],
                      rule: int = 4) -> float:
    if rule == 1:
        pts, w = hexagon_stroud_rule1()
    elif rule == 2:
        pts, w = hexagon_stroud_rule2()
    elif rule == 3:
        pts, w = hexagon_stroud_rule3()
    else:
        pts, w = hexagon_stroud_rule4()
    x = pts[:, 0]
    y = pts[:, 1]
    vals = f(x, y)
    return float(np.dot(w, vals))


def hexagon_monomial_integral(p: int, q: int) -> float:
    from geometry_mesh import hexagon_monomial_integral as hmi
    return hmi(p, q)





def integrate_trapezoidal(y: np.ndarray, x: Optional[np.ndarray] = None) -> float:
    if x is None:
        x = np.arange(len(y))
    if len(y) != len(x):
        raise ValueError("x and y must have same length")
    if len(y) < 2:
        return 0.0
    integral = 0.0
    for i in range(len(y) - 1):
        dx = x[i + 1] - x[i]
        if np.isclose(dx, 0.0):
            continue
        integral += 0.5 * (y[i] + y[i + 1]) * dx
    return float(integral)


def integrate_simpson(y: np.ndarray, x: Optional[np.ndarray] = None) -> float:
    n = len(y)
    if n < 3:
        return integrate_trapezoidal(y, x)
    if x is None:
        h = 1.0
    else:
        if len(x) != n:
            raise ValueError("x and y must have same length")
        h = x[1] - x[0]
        if not np.allclose(np.diff(x), h):

            return integrate_simpson_nonuniform(y, x)
    if n % 2 == 0:

        return integrate_simpson(y[:-1], x[:-1] if x is not None else None) + \
               0.5 * h * (y[-2] + y[-1])
    integral = y[0] + y[-1]
    integral += 4.0 * np.sum(y[1:-1:2])
    integral += 2.0 * np.sum(y[2:-2:2])
    return float(integral * h / 3.0)


def integrate_simpson_nonuniform(y: np.ndarray, x: np.ndarray) -> float:
    n = len(y)
    integral = 0.0
    for i in range(0, n - 2, 2):
        h1 = x[i + 1] - x[i]
        h2 = x[i + 2] - x[i + 1]
        if np.isclose(h1 + h2, 0.0):
            continue

        alpha = (2.0 * h1 ** 2 + h1 * h2 - h2 ** 2) / (6.0 * h1 * (h1 + h2))
        beta = (h1 + h2) ** 3 / (6.0 * h1 * h2 * (h1 + h2))
        gamma = (-h1 ** 2 + h1 * h2 + 2.0 * h2 ** 2) / (6.0 * h2 * (h1 + h2))
        integral += alpha * y[i] + beta * y[i + 1] + gamma * y[i + 2]
    return float(integral)


def integrate_quantum_probability(prob: np.ndarray, times: np.ndarray) -> float:
    return integrate_trapezoidal(prob, times)





def steinerberger_function(n: int, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    result = np.zeros_like(x, dtype=float)
    for k in range(1, n + 1):
        result += np.abs(np.sin(np.pi * k * x)) / k
    return result


def harmonic_number(n: int) -> float:
    if n <= 0:
        return 0.0
    return float(np.sum(1.0 / np.arange(1, n + 1)))


def steinerberger_integral01_exact(n: int) -> float:
    return (2.0 / np.pi) * harmonic_number(n)


def test_quadrature_accuracy(quadrature_func: Callable, n_max: int = 10) -> dict:
    results = []
    for n in range(1, n_max + 1):
        exact = steinerberger_integral01_exact(n)

        x = np.linspace(0.0, 1.0, 2001)
        y = steinerberger_function(n, x)
        approx = integrate_simpson(y, x)
        err = abs(approx - exact)
        results.append({
            "n": n,
            "exact": exact,
            "approximate": approx,
            "error": err,
            "relative_error": err / exact if exact > 1e-12 else 0.0
        })
    return {"tests": results}





def integrate_observable_over_grid(observable: Callable,
                                   grid_pts: np.ndarray,
                                   weights: Optional[np.ndarray] = None) -> float:
    if grid_pts.ndim == 1:
        vals = observable(grid_pts)
    else:
        vals = observable(grid_pts[:, 0], grid_pts[:, 1])
    if weights is None:
        weights = np.ones(len(vals)) / len(vals)
    return float(np.dot(weights, vals))


def compute_fidelity_integral(psi1: np.ndarray, psi2: np.ndarray,
                              times: np.ndarray) -> float:
    if len(times) < 2:
        return 0.0
    fidelities = np.abs(np.vdot(psi1, psi2)) ** 2 * np.ones_like(times)
    integral = integrate_trapezoidal(fidelities, times)
    T = times[-1] - times[0]
    if np.isclose(T, 0.0):
        return 0.0
    return float(integral / T)
