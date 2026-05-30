# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import gamma as GammaFunc
from scipy.special import jacobi
from numpy.polynomial.legendre import leggauss






def wedge01_volume():
    return 1.0


def wedge01_monomial_integral(e):
    e = np.asarray(e, dtype=int).flatten()
    if e.size != 3:
        raise ValueError("e 必须是长度为 3 的整数向量。")
    e1, e2, e3 = e[0], e[1], e[2]


    if e1 < 0 or e2 < 0:
        return 0.0
    value_xy = 1.0
    for i in range(1, e2 + 1):
        value_xy *= float(i) / float(e1 + i)
    denom = float((e1 + e2 + 1) * (e1 + e2 + 2))
    value_xy /= denom


    if e3 % 2 == 1:
        value_z = 0.0
    else:
        value_z = 2.0 / float(e3 + 1)

    return value_xy * value_z


def wedge01_sample(n):
    if n < 1:
        return np.zeros((3, 0))

    E = -np.log(np.random.rand(3, n))
    S = np.sum(E, axis=0)
    xy = E[:2, :] / S[np.newaxis, :]
    z = 2.0 * np.random.rand(1, n) - 1.0
    return np.vstack([xy, z])


def monomial_value(m, n_pts, e, x):
    e = np.asarray(e, dtype=int)
    x = np.asarray(x, dtype=float)
    if x.shape != (m, n_pts):
        raise ValueError("x 形状不匹配。")
    v = np.ones(n_pts, dtype=float)
    for d in range(m):
        if e[d] == 0:
            continue
        v *= np.power(x[d, :], e[d])
    return v






def tetrahedron_arbq_size(degree):
    sizes = [1, 1, 4, 6, 11, 14, 23, 31, 44, 57, 74, 95, 122, 146, 177, 214]
    d = int(degree)
    if d < 0 or d >= len(sizes):
        raise ValueError("degree 必须在 0..15 范围内。")
    return sizes[d]


def tetrahedron_ref():
    return np.array([
        [-1.0, -1.0, -1.0],
        [-1.0,  1.0, -1.0],
        [ 1.0, -1.0, -1.0],
        [-1.0, -1.0,  1.0]
    ], dtype=float)


def ref_to_koorn(r):
    r = np.asarray(r, dtype=float)
    if r.ndim == 1:

        x, y, z = r[0], r[1], r[2]
        return np.array([x, y, z], dtype=float)
    return r.copy()


def ortho3eva(degree, xyz):
    xyz = np.asarray(xyz, dtype=float)
    single = xyz.ndim == 1
    if single:
        xyz = xyz.reshape(1, 3)
    npts = xyz.shape[0]


    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    eps = 1e-15


    xi1 = np.where(np.abs(1.0 - y - z) > eps,
                   (2.0 * x + 2.0 + y + z) / (-y - z + eps), 0.0)
    xi2 = np.where(np.abs(1.0 - z) > eps,
                   (2.0 * y + 1.0 + z) / (1.0 - z + eps), 0.0)
    zeta = z.copy()


    max_m = degree
    fvals = []
    for m in range(max_m + 1):
        for n in range(max_m + 1 - m):
            for k in range(max_m + 1 - m - n):

                Pm = np.ones(npts)
                if m > 0:

                    leg_m = np.polynomial.legendre.legval(xi1, [0] * m + [1])
                    Pm = leg_m
                factor1 = Pm * np.power(np.maximum((-y - z + 1.0) * 0.5, eps), m)


                Pn = np.ones(npts)
                if n > 0:
                    jac = jacobi(n, 2 * m + 1, 0)
                    Pn = np.polyval(jac, xi2)
                factor2 = Pn * np.power(np.maximum((1.0 - zeta) * 0.5, eps), n)


                Pk = np.ones(npts)
                if k > 0:
                    jac = jacobi(k, 2 * m + 2 * n + 2, 0)
                    Pk = np.polyval(jac, zeta)
                factor3 = Pk

                fvals.append(factor1 * factor2 * factor3)
    res = np.column_stack(fvals) if len(fvals) > 0 else np.ones((npts, 1))
    if single:
        return res[0, :]
    return res


def _gauss_tetrahedron_nodes_weights(degree):
    if degree <= 1:

        nodes = np.array([[0.25, 0.25, 0.25]])
        weights = np.array([1.0 / 6.0])
        return nodes, weights
    elif degree <= 2:

        a = 0.58541020
        b = 0.13819660
        nodes = np.array([
            [a, b, b],
            [b, a, b],
            [b, b, a],
            [b, b, b]
        ])
        weights = np.ones(4) / 24.0
        return nodes, weights
    elif degree <= 3:

        nodes = np.array([
            [0.25, 0.25, 0.25],
            [0.5, 1.0 / 6.0, 1.0 / 6.0],
            [1.0 / 6.0, 0.5, 1.0 / 6.0],
            [1.0 / 6.0, 1.0 / 6.0, 0.5],
            [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0]
        ])
        weights = np.array([-4.0 / 30.0, 3.0 / 40.0, 3.0 / 40.0, 3.0 / 40.0, 3.0 / 40.0])
        return nodes, weights
    else:

        n_mc = tetrahedron_arbq_size(min(degree, 15))
        nodes = np.random.rand(n_mc, 3)

        nodes = -np.log(np.maximum(nodes, 1e-15))
        s = np.sum(nodes, axis=1, keepdims=True)
        nodes = nodes / s
        weights = np.ones(n_mc) / (6.0 * n_mc)
        return nodes, weights


def integrate_tetrahedron(f, degree=3):
    nodes, weights = _gauss_tetrahedron_nodes_weights(degree)
    vals = f(nodes)
    return np.dot(weights, vals)






def jacobi_polynomial_eval(m, n, alpha, beta, x):
    x = np.asarray(x, dtype=float).flatten()
    m_pts = x.size
    if n < 0:
        return np.zeros((m_pts, 0))
    v = np.ones((m_pts, n + 1), dtype=float)
    if n >= 1:
        v[:, 1] = (alpha - beta) * 0.5 + (alpha + beta + 2.0) * 0.5 * x
    for k in range(1, n):
        a1 = 2.0 * (k + 1.0) * (k + alpha + beta + 1.0) * (2.0 * k + alpha + beta)
        a2 = (2.0 * k + alpha + beta + 1.0) * (alpha ** 2 - beta ** 2)
        a3 = (2.0 * k + alpha + beta) * (2.0 * k + alpha + beta + 1.0) * (2.0 * k + alpha + beta + 2.0)
        a4 = 2.0 * (k + alpha) * (k + beta) * (2.0 * k + alpha + beta + 2.0)
        if abs(a1) < 1e-15:
            break
        v[:, k + 1] = ((a2 + a3 * x) * v[:, k] - a4 * v[:, k - 1]) / a1
    return v


def jacobi_polynomial_zeros(n, alpha, beta):
    if n < 1:
        return np.array([])

    d = np.zeros(n)
    e = np.zeros(n - 1)
    for i in range(n):
        d[i] = (beta ** 2 - alpha ** 2) / ((2.0 * i + alpha + beta) * (2.0 * i + alpha + beta + 2.0)) if (2.0 * i + alpha + beta) > 0 else 0.0
    for i in range(1, n):
        ab = alpha + beta
        denom = (2.0 * i + ab - 1.0) * (2.0 * i + ab + 1.0)
        if denom > 1e-15:
            num = i * (i + ab) * (i + alpha) * (i + beta)
            e[i - 1] = 2.0 / (2.0 * i + ab) * np.sqrt(num / denom)

    J = np.diag(d) + np.diag(e, k=1) + np.diag(e, k=-1)
    zeros = np.sort(np.linalg.eigvalsh(J))
    return zeros


def gauss_jacobi_quadrature(n, alpha, beta):
    if n < 1:
        return np.array([]), np.array([])

    d = np.zeros(n)
    e = np.zeros(n - 1)
    for i in range(n):
        ab = alpha + beta
        if (2.0 * i + ab) > 1e-15:
            d[i] = (beta ** 2 - alpha ** 2) / ((2.0 * i + ab) * (2.0 * i + ab + 2.0))
    for i in range(1, n):
        ab = alpha + beta
        denom = (2.0 * i + ab - 1.0) * (2.0 * i + ab + 1.0)
        if denom > 1e-15:
            num = i * (i + alpha + beta) * (i + alpha) * (i + beta)
            e[i - 1] = 2.0 / (2.0 * i + ab) * np.sqrt(num / denom)
    J = np.diag(d) + np.diag(e, k=1) + np.diag(e, k=-1)
    w, v = np.linalg.eigh(J)
    zeros = np.sort(w)

    idx = np.argsort(w)
    v = v[:, idx]

    mu0 = (2.0 ** (alpha + beta + 1.0) *
           GammaFunc(alpha + 1.0) * GammaFunc(beta + 1.0) /
           GammaFunc(alpha + beta + 2.0))
    weights = mu0 * v[0, :] ** 2
    return zeros, weights


def jacobi_double_product_integral(i, j, alpha, beta):
    if i != j:
        return 0.0
    if i < 0:
        return 0.0
    val = (2.0 ** (alpha + beta + 1.0) / (2.0 * i + alpha + beta + 1.0) *
           GammaFunc(i + alpha + 1.0) * GammaFunc(i + beta + 1.0) /
           (GammaFunc(i + 1.0) * GammaFunc(i + alpha + beta + 1.0)))
    return float(val)






def integrate_irreducible_wedge(func, n_sample=20000):
    if n_sample < 1:
        return 0.0

    wedge_area = np.pi ** 2 / 8.0


    samples = []
    batch = min(n_sample * 5, 200000)
    while len(samples) < n_sample:
        s = np.random.rand(batch, 2) * np.pi
        mask = s[:, 0] >= 0
        mask &= s[:, 1] >= 0
        mask &= (s[:, 0] + s[:, 1]) <= np.pi
        valid = s[mask]
        needed = n_sample - len(samples)
        samples.extend(valid[:needed].tolist())
    samples = np.array(samples[:n_sample])
    vals = func(samples)

    return 8.0 * wedge_area * np.mean(vals)


def integrate_bz_gauss_legendre_2d(func, n_per_dim=40):

    raise NotImplementedError("Hole_3: implement integrate_bz_gauss_legendre_2d with Gauss-Legendre quadrature")
