import math
import numpy as np


def line01_sample_random(n: int, rng):
    return rng.uniforms(n)


def line01_sample_ergodic(n: int, shift: float):
    golden = (1.0 + math.sqrt(5.0)) / 2.0
    x = np.empty(n, dtype=float)
    s = shift % 1.0
    for j in range(n):
        x[j] = s
        s = (s + golden) % 1.0
    return x


def line01_monomial_integral(e: int) -> float:
    return 1.0 / (e + 1.0)


def square01_sample(n: int, rng):
    return rng.uniforms(2 * n).reshape(2, n)


def square01_monomial_integral(a: int, b: int) -> float:
    return 1.0 / ((a + 1.0) * (b + 1.0))


def triangle01_sample(n: int, rng):
    u = rng.uniforms(2 * n)
    e1 = -np.log(u[:n])
    e2 = -np.log(u[n:])
    s = e1 + e2

    s = np.where(s < 1e-15, 1e-15, s)
    x = e1 / s
    y = e2 / s
    return np.vstack((x, y))


def triangle01_monomial_integral(a: int, b: int) -> float:
    from math import factorial
    return factorial(a) * factorial(b) / factorial(a + b + 2)


def integrate_1d(f, n: int, method: str = "random", rng=None, shift: float = 0.0):
    if method == "random":
        if rng is None:
            raise ValueError("integrate_1d random method requires rng")
        x = line01_sample_random(n, rng)
    elif method == "ergodic":
        x = line01_sample_ergodic(n, shift)
    else:
        raise ValueError("integrate_1d: method must be 'random' or 'ergodic'")
    fx = np.asarray(f(x))
    mu = float(np.mean(fx))
    if n > 1:
        se = float(np.std(fx, ddof=1) / math.sqrt(n))
    else:
        se = 0.0
    return mu, se


def integrate_square(f, n: int, method: str = "random", rng=None):
    if method != "random":
        raise ValueError("integrate_square currently supports only 'random'")
    if rng is None:
        raise ValueError("integrate_square requires rng")
    pts = square01_sample(n, rng)
    fx = np.asarray(f(pts))
    mu = float(np.mean(fx))
    if n > 1:
        se = float(np.std(fx, ddof=1) / math.sqrt(n))
    else:
        se = 0.0
    return mu, se


def integrate_triangle(f, n: int, method: str = "random", rng=None):
    if method != "random":
        raise ValueError("integrate_triangle currently supports only 'random'")
    if rng is None:
        raise ValueError("integrate_triangle requires rng")
    pts = triangle01_sample(n, rng)
    fx = np.asarray(f(pts))
    mu = 0.5 * float(np.mean(fx))
    if n > 1:
        se = 0.5 * float(np.std(fx, ddof=1) / math.sqrt(n))
    else:
        se = 0.0
    return mu, se
