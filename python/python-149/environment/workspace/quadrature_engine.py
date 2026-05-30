
import numpy as np
from typing import Callable, Optional, Tuple






def pyramid_jaskowiec_rule(p: int) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if p < 0 or p > 6:
        raise ValueError("精度阶数p必须在[0,6]范围内")


    if p == 0:
        n = 1
        x = np.array([0.0])
        y = np.array([0.0])
        z = np.array([2.0 / 3.0])
        w = np.array([4.0 / 3.0])
    elif p == 1:
        n = 1
        x = np.array([0.0])
        y = np.array([0.0])
        z = np.array([0.5])
        w = np.array([4.0 / 3.0])
    elif p == 2:
        n = 5
        a = 2.0 / 5.0
        b = 3.0 / 5.0
        x = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        y = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        z = np.array([a, a, a, a, b])
        w0 = 4.0 / 15.0
        w1 = 4.0 / 15.0
        w = np.array([w0, w0, w0, w0, w1])
    elif p == 3:
        n = 6
        a = (6.0 - np.sqrt(6.0)) / 10.0
        b = (6.0 + np.sqrt(6.0)) / 10.0
        wa = (3.0 * a - 1.0) / (6.0 * (a - b))
        wb = (3.0 * b - 1.0) / (6.0 * (b - a))
        x = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        y = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        z = np.array([a, a, a, a, b, b])
        w = np.array([wa, wa, wa, wa, wb, wb])
    elif p == 4:
        n = 8
        a = 0.25
        b = 0.75
        c = np.sqrt(2.0 / 3.0)
        x = np.array([c, -c, 0.0, 0.0, c, -c, 0.0, 0.0])
        y = np.array([0.0, 0.0, c, -c, 0.0, 0.0, c, -c])
        z = np.array([a, a, a, a, b, b, b, b])
        w = np.full(n, 1.0 / 6.0)
    elif p == 5:
        n = 8
        a = (5.0 - np.sqrt(5.0)) / 10.0
        b = (5.0 + np.sqrt(5.0)) / 10.0
        wa = 1.0 / 12.0
        wb = 1.0 / 12.0
        x = np.zeros(n)
        y = np.zeros(n)
        z = np.array([a, a, a, a, b, b, b, b])
        w = np.array([wa, wa, wa, wa, wb, wb, wb, wb])
    else:
        n = 14
        a1 = 0.2
        a2 = 0.6
        a3 = 0.9
        c = np.sqrt(2.0 / 3.0)
        x = np.array([
            0.0, 0.0, 0.0, 0.0,
            c, -c, 0.0, 0.0, c, -c, 0.0, 0.0,
            0.0, 0.0,
        ])
        y = np.array([
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, c, -c, 0.0, 0.0, c, -c,
            0.0, 0.0,
        ])
        z = np.array([
            a1, a1, a1, a1,
            a2, a2, a2, a2, a2, a2, a2, a2,
            a3, a3,
        ])
        w = np.array([
            0.05, 0.05, 0.05, 0.05,
            0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025,
            0.1, 0.1,
        ])


    vol = 4.0 / 3.0
    w = w * vol / np.sum(w)

    return n, x, y, z, w


def integrate_over_pyramid(
    f: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    p: int = 4,
) -> float:
    n, x, y, z, w = pyramid_jaskowiec_rule(p)
    vals = f(x, y, z)
    vals = np.atleast_1d(vals)
    if len(vals) != n:

        vals = np.array([f(x[k], y[k], z[k]) for k in range(n)])
    return float(np.sum(w * vals))


def gauss_hermite_quad_1d(
    n_points: int,
    f: Callable[[np.ndarray], np.ndarray],
    sigma: float = 1.0,
) -> float:
    from numpy.polynomial.hermite import hermgauss

    if n_points < 1:
        raise ValueError("n_points必须≥1")


    y_nodes, w_phys = hermgauss(n_points)






    x_nodes = np.sqrt(2.0) * sigma * y_nodes
    w_prob = w_phys / np.sqrt(np.pi)

    vals = np.atleast_1d(f(x_nodes))
    return float(np.sum(w_prob * vals))


def monte_carlo_expectation(
    f: Callable[[np.ndarray], np.ndarray],
    sampler: Callable[[int, np.random.Generator], np.ndarray],
    n_samples: int = 10000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    if rng is None:
        rng = np.random.default_rng(seed=42)

    samples = sampler(n_samples, rng)
    vals = np.array([f(samples[i, :]) for i in range(n_samples)])


    valid = np.isfinite(vals)
    if np.sum(valid) < n_samples * 0.5:
        raise ValueError("超过50%的样本产生非有限值")

    vals = vals[valid]
    mean = float(np.mean(vals))
    std_err = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
    return mean, std_err
