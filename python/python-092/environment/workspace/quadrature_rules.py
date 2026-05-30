
import numpy as np
from math import gamma as math_gamma


def factorial(n):
    if n <= 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def triangle_unit_monomial_integral(expon):
    m, n = expon
    return factorial(m) * factorial(n) / factorial(m + n + 2)


def triangle_unit_area():
    return 0.5


def triangle_symq_rule(precision):
    if precision <= 1:

        a = np.array([1.0 / 3.0])
        b = np.array([1.0 / 3.0])
        c = np.array([1.0 / 3.0])
        w = np.array([0.5])
        return 1, a, b, c, w
    elif precision <= 2:

        a = np.array([2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0])
        b = np.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0])
        c = np.array([1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0])
        w = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        return 3, a, b, c, w
    elif precision <= 3:

        a = np.array([1.0 / 3.0, 0.6, 0.2, 0.2])
        b = np.array([1.0 / 3.0, 0.2, 0.6, 0.2])
        c = np.array([1.0 / 3.0, 0.2, 0.2, 0.6])
        w = np.array([-27.0 / 96.0, 25.0 / 96.0, 25.0 / 96.0, 25.0 / 96.0])
        return 4, a, b, c, w
    elif precision <= 5:

        a = np.array([
            1.0 / 3.0,
            0.059715871789770, 0.470142064105115, 0.470142064105115,
            0.797426985353087, 0.101286507323456, 0.101286507323456
        ])
        b = np.array([
            1.0 / 3.0,
            0.470142064105115, 0.059715871789770, 0.470142064105115,
            0.101286507323456, 0.797426985353087, 0.101286507323456
        ])
        c = np.array([
            1.0 / 3.0,
            0.470142064105115, 0.470142064105115, 0.059715871789770,
            0.101286507323456, 0.101286507323456, 0.797426985353087
        ])
        w = np.array([
            0.225000000000000,
            0.132394152788506, 0.132394152788506, 0.132394152788506,
            0.125939180544827, 0.125939180544827, 0.125939180544827
        ]) * 0.5
        return 7, a, b, c, w
    else:

        return triangle_symq_rule(5)


def integrate_over_triangle(func, v0, v1, v2, precision=5):
    n, a, b, c, w = triangle_symq_rule(precision)


    area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
    result = 0.0
    for i in range(n):
        p = a[i] * v0 + b[i] * v1 + c[i] * v2
        result += w[i] * func(p)
    return result * (area / 0.5)


def ball01_volume():
    return 4.0 * np.pi / 3.0


def ball01_monomial_integral(e):
    e = np.asarray(e, dtype=int)
    if np.any(e < 0):
        return 0.0
    if np.any(e % 2 == 1):
        return 0.0
    if np.all(e == 0):
        integral = 2.0 * np.sqrt(np.pi ** 3) / math_gamma(1.5)
    elif np.any(e % 2 == 1):
        return 0.0
    else:
        integral = 2.0
        for i in range(3):
            integral = integral * math_gamma(0.5 * (e[i] + 1))
        integral = integral / math_gamma(0.5 * (e[0] + e[1] + e[2] + 3))

    r = 1.0
    s = e[0] + e[1] + e[2] + 3
    integral = integral * (r ** s) / s
    return integral


def ball01_sample(n):

    xyz = np.random.randn(n, 3)

    norms = np.linalg.norm(xyz, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-14)
    xyz = xyz / norms

    u = np.random.rand(n, 1)
    r = u ** (1.0 / 3.0)
    return xyz * r


def monomial_value(n_points, e, x):
    e = np.asarray(e, dtype=int)
    x = np.asarray(x, dtype=float)
    val = np.ones(n_points, dtype=float)
    for dim in range(len(e)):
        if e[dim] > 0:
            val *= x[:, dim] ** e[dim]
    return val


def line01_monomial_integral(e):
    return 1.0 / (e + 1.0)


def line01_sample_random(n):
    return np.random.rand(n)


def line01_sample_ergodic(n, shift=0.0):
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    x = np.zeros(n, dtype=float)
    x[0] = shift % 1.0
    for j in range(1, n):
        x[j] = (x[j - 1] + phi) % 1.0
    return x


def integrate_over_ball_monte_carlo(func, n_samples=10000):
    samples = ball01_sample(n_samples)
    vals = np.array([func(s) for s in samples])
    return ball01_volume() * np.mean(vals)
