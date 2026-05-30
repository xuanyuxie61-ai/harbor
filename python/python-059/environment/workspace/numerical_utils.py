
import numpy as np
from math import lgamma, exp, sqrt, cos, sin, log, pi


class NumericalError(Exception):
    pass


def bisection(f, a, b, tol=1e-12, max_iter=100):
    fa = f(a)
    fb = f(b)

    if fa == 0.0:
        return float(a), 0
    if fb == 0.0:
        return float(b), 0

    if fa * fb > 0.0:
        raise NumericalError("bisection: 区间端点函数值同号，无法保证根存在。")

    it = 0
    while abs(b - a) > tol:
        if it >= max_iter:
            raise NumericalError(f"bisection: 超过最大迭代次数 {max_iter}")
        c = (a + b) / 2.0
        fc = f(c)
        it += 1
        if fc == 0.0:
            return float(c), it
        if np.sign(fc) == np.sign(fa):
            a = c
            fa = fc
        else:
            b = c
            fb = fc
    return float((a + b) / 2.0), it


def binomial_coefficient(n, k):
    if k < 0 or k > n or n < 0:
        return 0.0
    if k == 0 or k == n:
        return 1.0

    k = min(k, n - k)
    val = exp(lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1))
    return val


def comb_lexicographic(n, p, l):
    if p <= 0 or p > n or n <= 0:
        raise NumericalError("comb_lexicographic: 参数非法")
    total = binomial_coefficient(n, p)
    if l < 1 or l > total:
        raise NumericalError(f"comb_lexicographic: 索引 l={l} 超出范围 [1, {total}]")

    c = [0] * p
    if p == 1:
        c[0] = l
        return c

    k = 0
    p1 = p - 1
    c[0] = 0

    for i in range(p1):
        if i > 0:
            c[i] = c[i - 1]
        while True:
            c[i] += 1
            r = binomial_coefficient(n - c[i], p - i - 1)
            k += r
            if l <= k:
                break
        k -= r

    c[p - 1] = c[p1 - 1] + l - k
    return c


def rnorm():
    u1 = np.random.rand()
    u2 = np.random.rand()

    if u1 < 1e-15:
        u1 = 1e-15
    mag = sqrt(-2.0 * log(u1))
    z1 = mag * cos(2.0 * pi * u2)
    z2 = mag * sin(2.0 * pi * u2)
    return z1, z2


def gamma_log_values(n):
    return np.array([lgamma(i + 1) for i in range(n + 1)])


def wilson_hilferty_chi_square(df, z):
    if df <= 0:
        raise NumericalError("wilson_hilferty_chi_square: df 必须为正")
    u1 = 2.0 / (9.0 * df)
    u2 = 1.0 - u1
    u1_sqrt = sqrt(u1)
    val = df * abs(u2 + z * u1_sqrt) ** 3
    return sqrt(val)


def safe_acos(x):
    x = max(-1.0, min(1.0, x))
    return np.arccos(x)
