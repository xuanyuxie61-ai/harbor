
import numpy as np
from parameters import N_GAUSS






def legendre_gauss_nodes_weights(n):
    if n < 1:
        raise ValueError("求积阶数 n 必须 ≥ 1")
    if n == 1:
        return np.array([0.0]), np.array([2.0])


    i = np.arange(1.0, n, dtype=float)
    beta = i / np.sqrt(4.0 * i ** 2 - 1.0)
    J = np.diag(beta, 1) + np.diag(beta, -1)

    eigenvalues, eigenvectors = np.linalg.eigh(J)
    x = eigenvalues
    w = 2.0 * eigenvectors[0, :] ** 2

    return x, w


def gauss_quadrature(f, a, b, n=N_GAUSS):
    x, w = legendre_gauss_nodes_weights(n)

    t = 0.5 * (b - a) * x + 0.5 * (b + a)
    jac = 0.5 * (b - a)
    ft = np.asarray([f(ti) for ti in t], dtype=float)
    return float(jac * np.sum(w * ft))








_TRIANGLE_RULES = {
    3: {
        "n": 1,
        "bary": np.array([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]]),
        "w": np.array([0.5]),
    },
    5: {
        "n": 4,
        "bary": np.array([
            [0.33333333333333, 0.33333333333333, 0.33333333333334],
            [0.6, 0.2, 0.2],
            [0.2, 0.6, 0.2],
            [0.2, 0.2, 0.6],
        ]),
        "w": np.array([-0.28125, 0.26041666666667, 0.26041666666667, 0.26041666666667]),
    },
    7: {
        "n": 7,
        "bary": np.array([
            [0.33333333333333, 0.33333333333333, 0.33333333333334],
            [0.79742698535309, 0.10128650732346, 0.10128650732345],
            [0.10128650732346, 0.79742698535309, 0.10128650732345],
            [0.10128650732346, 0.10128650732346, 0.79742698535308],
            [0.05971587178977, 0.47014206410512, 0.47014206410511],
            [0.47014206410512, 0.05971587178977, 0.47014206410511],
            [0.47014206410512, 0.47014206410512, 0.05971587178976],
        ]),
        "w": np.array([
            0.1125,
            0.06296959027241,
            0.06296959027241,
            0.06296959027242,
            0.06619707639425,
            0.06619707639425,
            0.06619707639426,
        ]),
    },
}


def triangle_quadrature(f, vert1, vert2, vert3, precision=7):
    if precision not in _TRIANGLE_RULES:
        raise ValueError(f"不支持精度 {precision}，请选择 3, 5, 7")
    rule = _TRIANGLE_RULES[precision]
    bary = rule["bary"]
    w = rule["w"]



    verts = np.array([vert1, vert2, vert3])
    xy = bary @ verts


    jac = abs(np.linalg.det(np.array([
        [vert2[0] - vert1[0], vert3[0] - vert1[0]],
        [vert2[1] - vert1[1], vert3[1] - vert1[1]],
    ])))

    vals = np.array([f(xy[k, 0], xy[k, 1]) for k in range(rule["n"])], dtype=float)
    return float(jac * np.sum(w * vals))


def assemble_stiffness_triangle(vert1, vert2, vert3):
    v1, v2, v3 = np.asarray(vert1), np.asarray(vert2), np.asarray(vert3)
    area = 0.5 * abs((v2[0] - v1[0]) * (v3[1] - v1[1]) -
                     (v3[0] - v1[0]) * (v2[1] - v1[1]))
    if area < 1e-15:
        return np.zeros((3, 3))


    b = np.array([v2[1] - v3[1], v3[1] - v1[1], v1[1] - v2[1]])
    c = np.array([v3[0] - v2[0], v1[0] - v3[0], v2[0] - v1[0]])

    K = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            K[i, j] = (b[i] * b[j] + c[i] * c[j]) / (4.0 * area)
    return K






def chebyshev_vandermonde(m, a, b, x_nodes):
    x_nodes = np.asarray(x_nodes, dtype=float).flatten()
    n = len(x_nodes)
    xi = (-(b - x_nodes) + (x_nodes - a)) / (b - a)

    V = np.zeros((m, n))
    V[0, :] = 1.0
    if m > 1:
        V[1, :] = xi
    for i in range(2, m):
        V[i, :] = 2.0 * xi * V[i - 1, :] - V[i - 2, :]
    return V


def line_fekete_points(m, a, b, n_sample=200):
    if n_sample < m:
        raise ValueError("样本点数必须不少于基函数数")

    x = np.linspace(a, b, n_sample)
    V = chebyshev_vandermonde(m, a, b, x)


    mom = np.zeros(m)
    mom[0] = np.pi * (b - a) / 2.0

    for k in range(1, m):
        mom[k] = gauss_quadrature(lambda xi: np.cos(k * np.arccos(
            np.clip((-(b - xi) + (xi - a)) / (b - a), -1.0, 1.0))),
            a, b, n=min(32, N_GAUSS))


    w, _, _, _ = np.linalg.lstsq(V, mom, rcond=None)

    ind = np.where(np.abs(w) > 1e-12 * np.max(np.abs(w)))[0]
    if len(ind) < m:

        ind = np.argsort(np.abs(w))[-m:]

    xf = x[ind]
    wf = w[ind]
    Vf = V[:, ind]
    return xf, wf, Vf






def toroidal_volume_integral(f_radial, R0, a, kappa, n_radial=64, n_theta=64):
    r_nodes, r_weights = legendre_gauss_nodes_weights(n_radial)
    r = 0.5 * a * (r_nodes + 1.0)
    r_w = 0.5 * a * r_weights

    theta_nodes, theta_weights = legendre_gauss_nodes_weights(n_theta)
    theta = np.pi * (theta_nodes + 1.0)
    theta_w = np.pi * theta_weights

    total = 0.0
    for i in range(n_radial):
        for j in range(n_theta):
            R_loc = R0 + r[i] * np.cos(theta[j])
            jac = R_loc * r[i]
            total += r_w[i] * theta_w[j] * jac * f_radial(r[i], theta[j])

    return 2.0 * np.pi * total


def magnetic_surface_average(B_p, R, Z, psi, psi_target):
    mask = np.abs(psi - psi_target) < 0.05 * (psi.max() - psi.min())
    if not np.any(mask):
        return 0.0

    dl = np.sqrt(np.gradient(R[mask]) ** 2 + np.gradient(Z[mask]) ** 2)
    if np.sum(dl) < 1e-15:
        return 0.0
    return float(np.sum(B_p[mask] * dl) / np.sum(dl))
