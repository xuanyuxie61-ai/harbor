
import numpy as np
from typing import Callable, Tuple, List


def filon_cos_quad(f: Callable, a: float, b: float, n: int, t: float) -> float:
    if a == b:
        return 0.0
    if n <= 1:
        raise ValueError("n 必须大于 1")
    if n % 2 == 0:
        raise ValueError("n 必须为奇数")

    x = np.linspace(a, b, n)
    h = (b - a) / (n - 1)
    theta = t * h
    sint = np.sin(theta)
    cost = np.cos(theta)

    if 6.0 * abs(theta) <= 1.0:
        alpha = (2.0 * theta**3 / 45.0
                 - 2.0 * theta**5 / 315.0
                 + 2.0 * theta**7 / 4725.0)
        beta = (2.0 / 3.0
                + 2.0 * theta**2 / 15.0
                - 4.0 * theta**4 / 105.0
                + 2.0 * theta**6 / 567.0
                - 4.0 * theta**8 / 22275.0)
        gamma = (4.0 / 3.0
                 - 2.0 * theta**2 / 15.0
                 + theta**4 / 210.0
                 - theta**6 / 11340.0)
    else:
        alpha = (theta**2 + theta * sint * cost - 2.0 * sint**2) / (theta**3)
        beta = (2.0 * theta + 2.0 * theta * cost**2
                - 4.0 * sint * cost) / (theta**3)
        gamma = 4.0 * (sint - theta * cost) / (theta**3)

    ftab = np.asarray(f(x), dtype=float)

    c2n = np.sum(ftab[0:n:2] * np.cos(t * x[0:n:2])) \
          - 0.5 * (ftab[-1] * np.cos(t * x[-1]) + ftab[0] * np.cos(t * x[0]))

    c2nm1 = np.sum(ftab[1:n-1:2] * np.cos(t * x[1:n-1:2]))

    value = h * (
        alpha * (ftab[-1] * np.sin(t * x[-1]) - ftab[0] * np.sin(t * x[0]))
        + beta * c2n
        + gamma * c2nm1
    )
    return float(value)


def filon_sin_quad(f: Callable, a: float, b: float, n: int, t: float) -> float:
    if a == b:
        return 0.0
    if n <= 1 or n % 2 == 0:
        raise ValueError("n 必须为奇数且大于 1")

    x = np.linspace(a, b, n)
    h = (b - a) / (n - 1)
    theta = t * h
    sint = np.sin(theta)
    cost = np.cos(theta)

    if 6.0 * abs(theta) <= 1.0:
        alpha = (2.0 * theta**3 / 45.0
                 - 2.0 * theta**5 / 315.0
                 + 2.0 * theta**7 / 4725.0)
        beta = (2.0 / 3.0
                + 2.0 * theta**2 / 15.0
                - 4.0 * theta**4 / 105.0
                + 2.0 * theta**6 / 567.0
                - 4.0 * theta**8 / 22275.0)
        gamma = (4.0 / 3.0
                 - 2.0 * theta**2 / 15.0
                 + theta**4 / 210.0
                 - theta**6 / 11340.0)
    else:
        alpha = (theta**2 + theta * sint * cost - 2.0 * sint**2) / (theta**3)
        beta = (2.0 * theta + 2.0 * theta * cost**2
                - 4.0 * sint * cost) / (theta**3)
        gamma = 4.0 * (sint - theta * cost) / (theta**3)

    ftab = np.asarray(f(x), dtype=float)

    s2n = np.sum(ftab[0:n:2] * np.sin(t * x[0:n:2])) \
          - 0.5 * (ftab[-1] * np.sin(t * x[-1]) + ftab[0] * np.sin(t * x[0]))

    s2nm1 = np.sum(ftab[1:n-1:2] * np.sin(t * x[1:n-1:2]))

    value = h * (
        alpha * (ftab[0] * np.cos(t * x[0]) - ftab[-1] * np.cos(t * x[-1]))
        + beta * s2n
        + gamma * s2nm1
    )
    return float(value)


def hexagon_stroud_rule(p: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if p not in (1, 2, 3, 4):
        raise ValueError("仅支持精度 p ∈ {1,2,3,4}")

    area = 3.0 * np.sqrt(3.0) / 2.0

    if p == 1:

        x = np.array([0.0])
        y = np.array([0.0])
        w = np.array([area])
    elif p == 2:

        r = np.sqrt(2.0 / 3.0)
        angles = np.array([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0])
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        w = np.full(3, area / 3.0)
    elif p == 3:

        r = np.sqrt(10.0 / 9.0)
        angles = np.array([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0])
        x = np.concatenate(([0.0], r * np.cos(angles)))
        y = np.concatenate(([0.0], r * np.sin(angles)))
        w = np.concatenate(([area * 9.0 / 20.0], np.full(3, area * 11.0 / 60.0)))
    else:

        r1 = np.sqrt((6.0 + np.sqrt(6.0)) / 10.0)
        r2 = np.sqrt((6.0 - np.sqrt(6.0)) / 10.0)
        angles = np.array([0.0, np.pi / 3.0, 2.0 * np.pi / 3.0,
                           np.pi, 4.0 * np.pi / 3.0, 5.0 * np.pi / 3.0])
        x = np.concatenate((r1 * np.cos(angles[0::2]), r2 * np.cos(angles[1::2])))
        y = np.concatenate((r1 * np.sin(angles[0::2]), r2 * np.sin(angles[1::2])))
        w1 = area * (16.0 + np.sqrt(6.0)) / 72.0
        w2 = area * (16.0 - np.sqrt(6.0)) / 72.0
        w = np.concatenate((np.full(3, w1), np.full(3, w2)))

    return x, y, w


def prism_jaskowiec_rule(p: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not (0 <= p <= 6):
        raise ValueError("本简化实现仅支持 0 <= p <= 6")

    vol = 0.5

    if p == 0:
        x = np.array([1.0 / 3.0])
        y = np.array([1.0 / 3.0])
        z = np.array([0.5])
        w = np.array([vol])
    elif p == 1:

        x = np.array([1.0 / 3.0, 1.0 / 3.0])
        y = np.array([1.0 / 3.0, 1.0 / 3.0])
        z = np.array([0.5 - np.sqrt(3.0) / 6.0, 0.5 + np.sqrt(3.0) / 6.0])
        w = np.array([vol / 2.0, vol / 2.0])
    elif p == 2:

        a_tri = 1.0 / 6.0
        b_tri = 2.0 / 3.0
        z_lo = 0.5 - np.sqrt(3.0) / 6.0
        z_hi = 0.5 + np.sqrt(3.0) / 6.0
        x = np.array([a_tri, b_tri, a_tri, a_tri, b_tri, a_tri])
        y = np.array([a_tri, a_tri, b_tri, a_tri, a_tri, b_tri])
        z = np.array([z_lo, z_lo, z_lo, z_hi, z_hi, z_hi])
        w = np.full(6, vol / 6.0)
    elif p == 3:

        a_tri = 1.0 / 3.0
        r_tri = np.sqrt(15.0) / 15.0
        z_nodes = np.array([0.5 - np.sqrt(3.0) / 6.0, 0.5 + np.sqrt(3.0) / 6.0])
        x_base = np.array([a_tri - r_tri, a_tri + r_tri, a_tri, a_tri])
        y_base = np.array([a_tri, a_tri, a_tri - r_tri, a_tri + r_tri])
        x = np.tile(x_base, 2)
        y = np.tile(y_base, 2)
        z = np.repeat(z_nodes, 4)
        w_tri = vol / 8.0
        w = np.full(8, w_tri)
    elif p == 4:

        a_tri = 1.0 / 3.0
        r1 = np.sqrt(15.0 + 3.0 * np.sqrt(15.0)) / 15.0
        r2 = np.sqrt(15.0 - 3.0 * np.sqrt(15.0)) / 15.0
        z_nodes = np.array([0.5 - np.sqrt(3.0) / 6.0, 0.5 + np.sqrt(3.0) / 6.0])

        x_base = np.array([a_tri - r1, a_tri + r1, a_tri - r2, a_tri + r2,
                           a_tri, a_tri, a_tri, a_tri])
        y_base = np.array([a_tri, a_tri, a_tri, a_tri,
                           a_tri - r1, a_tri + r1, a_tri - r2, a_tri + r2])


        x = np.tile(x_base, 2)
        y = np.tile(y_base, 2)
        z = np.repeat(z_nodes, 7)
        w = np.full(14, vol / 14.0)
    elif p == 5:

        pts = 9
        a_tri = 1.0 / 3.0
        r1 = 0.4
        r2 = 0.2
        x_base = np.array([a_tri - r1, a_tri + r1, a_tri - r2, a_tri + r2,
                           a_tri, a_tri, a_tri, a_tri, a_tri])
        y_base = np.array([a_tri, a_tri, a_tri, a_tri,
                           a_tri - r1, a_tri + r1, a_tri - r2, a_tri + r2, a_tri])
        z_nodes = np.array([0.5 - np.sqrt(3.0) / 6.0, 0.5 + np.sqrt(3.0) / 6.0])
        x = np.tile(x_base, 2)
        y = np.tile(y_base, 2)
        z = np.repeat(z_nodes, pts)
        w = np.full(2 * pts, vol / (2 * pts))
    else:
        pts = 12
        a_tri = 1.0 / 3.0
        r = 0.35
        x_base = np.array([a_tri - r, a_tri + r, a_tri, a_tri,
                           a_tri - r * 0.5, a_tri + r * 0.5, a_tri - r * 0.5, a_tri + r * 0.5,
                           a_tri, a_tri, a_tri, a_tri])
        y_base = np.array([a_tri, a_tri, a_tri - r, a_tri + r,
                           a_tri - r * 0.5, a_tri - r * 0.5, a_tri + r * 0.5, a_tri + r * 0.5,
                           a_tri, a_tri, a_tri, a_tri])
        z_nodes = np.array([0.5 - np.sqrt(3.0) / 6.0, 0.5 + np.sqrt(3.0) / 6.0])
        x = np.tile(x_base, 2)
        y = np.tile(y_base, 2)
        z = np.repeat(z_nodes, pts)
        w = np.full(2 * pts, vol / (2 * pts))

    return x, y, z, w


def product_rule_1d(rules_x: List[np.ndarray], rules_w: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    D = len(rules_x)
    if len(rules_w) != D:
        raise ValueError("rules_x 与 rules_w 长度不一致")

    orders = [len(xi) for xi in rules_x]
    N = 1
    for od in orders:
        N *= od

    X = np.zeros((D, N))
    W = np.ones(N)



    stride = 1
    for d in range(D):
        od = orders[d]
        rep = N // (stride * od)
        for j in range(od):
            idx_start = j * stride
            for k in range(rep):
                start = idx_start + k * stride * od
                X[d, start:start + stride] = rules_x[d][j]
                W[start:start + stride] *= rules_w[d][j]
        stride *= od

    return X, W
