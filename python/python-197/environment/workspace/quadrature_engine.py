
import math
import numpy as np





def imtqlx(n: int, d: np.ndarray, e: np.ndarray, z: np.ndarray):
    d = d.copy()
    e = e.copy()
    z = z.copy()
    e[n - 1] = 0.0
    for l in range(1, n + 1):
        j = 0
        while True:
            for m in range(l, n + 1):
                if m == n:
                    break
                if abs(e[m - 1]) <= 1.0e-14 * (abs(d[m - 1]) + abs(d[m])):
                    break
            if m == l:
                break
            if j >= 60:
                raise RuntimeError("IMTQLX did not converge")
            j += 1
            g = (d[l] - d[l - 1]) / (2.0 * e[l - 1])
            r = math.hypot(g, 1.0)
            g = d[m - 1] - d[l - 1] + e[l - 1] / (g + math.copysign(r, g))
            s = 1.0
            c = 1.0
            p = 0.0
            for i in range(m - 1, l - 1, -1):
                f = s * e[i - 1]
                b = c * e[i - 1]
                r = math.hypot(f, g)
                e[i] = r
                if r == 0.0:
                    d[i] = d[i] - p
                    e[m - 1] = 0.0
                    break
                s = f / r
                c = g / r
                g = d[i] - p
                r = (d[i - 1] - g) * s + 2.0 * c * b
                p = s * r
                d[i] = g + p
                g = c * r - b
                f = z[i]
                z[i] = s * z[i - 1] + c * f
                z[i - 1] = c * z[i - 1] - s * f
            if r == 0.0 and i >= l:
                continue
            d[l - 1] = d[l - 1] - p
            e[l - 1] = g
            e[m - 1] = 0.0
    return d, z





def legendre_rule(n: int, a: float = -1.0, b: float = 1.0):
    if n < 1:
        raise ValueError("n must be >= 1")
    d = np.zeros(n)
    e = np.zeros(n)
    z = np.zeros(n)
    z[0] = 1.0
    for i in range(1, n + 1):
        d[i - 1] = 0.0
        if i < n:
            e[i - 1] = i / math.sqrt(4.0 * i * i - 1.0)
    d, z = imtqlx(n, d, e, z)
    w = np.zeros(n)
    for i in range(n):
        w[i] = 2.0 * z[i] * z[i]

    x = 0.5 * (b - a) * d + 0.5 * (a + b)
    w = 0.5 * (b - a) * w
    return x, w





def laguerre_rule(n: int, alpha: float = 0.0):
    if n < 1:
        raise ValueError("n must be >= 1")
    if alpha <= -1.0:
        raise ValueError("alpha must be > -1")
    d = np.zeros(n)
    e = np.zeros(n)
    z = np.zeros(n)
    z[0] = 1.0
    for i in range(1, n + 1):
        d[i - 1] = 2.0 * i - 1.0 + alpha
        if i < n:
            e[i - 1] = math.sqrt(i * (i + alpha))
    d, z = imtqlx(n, d, e, z)
    w = np.zeros(n)
    for i in range(n):
        w[i] = math.exp(math.lgamma(alpha + 1.0)) * z[i] * z[i]
    return d, w





def tetrahedron_unit_volume() -> float:
    return 1.0 / 6.0


def tetrahedron_unit_monomial(expon: tuple) -> float:
    l, m, n = expon
    if l < 0 or m < 0 or n < 0:
        return 0.0
    return (math.gamma(l + 1.0) * math.gamma(m + 1.0) * math.gamma(n + 1.0)
            / math.gamma(l + m + n + 4.0))


def tetrahedron_unit_o04():
    w = np.array([1.0, 1.0, 1.0, 1.0]) / 24.0
    xyz = np.array([
        [0.58541020, 0.13819660, 0.13819660],
        [0.13819660, 0.58541020, 0.13819660],
        [0.13819660, 0.13819660, 0.58541020],
        [0.13819660, 0.13819660, 0.13819660],
    ])
    return w, xyz


def tetrahedron_unit_o14():
    a = 0.1005267652252045
    b = 0.314372873493192
    c = 0.8850566000690581
    d = 0.0931745731195340
    e = 0.3108859192633005
    w = np.array([
        0.1328387466855907,
        0.1328387466855907,
        0.1328387466855907,
        0.1328387466855907,
        0.0882236613785888,
        0.0882236613785888,
        0.0882236613785888,
        0.0882236613785888,
        0.0882236613785888,
        0.0882236613785888,
        0.0190475587642109,
        0.0190475587642109,
        0.0190475587642109,
        0.0190475587642109,
    ])
    xyz = np.array([
        [a, a, a],
        [a, a, c],
        [a, c, a],
        [c, a, a],
        [d, d, e],
        [d, e, d],
        [e, d, d],
        [d, e, e],
        [e, d, e],
        [e, e, d],
        [b, b, b],
        [b, b, e],
        [b, e, b],
        [e, b, b],
    ])
    return w, xyz


def integrate_tetrahedron(f, order: int = 4):
    if order <= 4:
        w, xyz = tetrahedron_unit_o04()
    else:
        w, xyz = tetrahedron_unit_o14()
    vol = tetrahedron_unit_volume()

    w = w / np.sum(w) * vol
    total = 0.0
    for i in range(len(w)):
        total += w[i] * f(xyz[i])
    return total
