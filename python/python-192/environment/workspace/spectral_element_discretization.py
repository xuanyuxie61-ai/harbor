
import numpy as np
from utils_numerical import safe_divide


def discrete_cosine_transform_1d(d: np.ndarray) -> np.ndarray:
    n = len(d)
    c = np.zeros(n)

    for i in range(n):
        for j in range(n):
            c[i] += np.cos(np.pi * (2 * j + 1) * i / (2.0 * n)) * d[j]

    c *= np.sqrt(2.0 / n)
    return c


def inverse_discrete_cosine_transform_1d(c: np.ndarray) -> np.ndarray:
    n = len(c)
    d = np.zeros(n)

    for j in range(n):
        d[j] = c[0] / 2.0
        for k in range(1, n):
            d[j] += c[k] * np.cos(np.pi * (2 * j + 1) * k / (2.0 * n))
        d[j] *= np.sqrt(2.0 / n)

    return d


def dct_poisson_solver_2d(f: np.ndarray, dx: float, dy: float) -> np.ndarray:
    ny, nx = f.shape


    f_hat = np.zeros_like(f)
    for j in range(ny):
        f_hat[j, :] = discrete_cosine_transform_1d(f[j, :])
    for i in range(nx):
        f_hat[:, i] = discrete_cosine_transform_1d(f_hat[:, i])


    p_hat = np.zeros_like(f_hat)
    for j in range(ny):
        for i in range(nx):
            if i == 0 and j == 0:
                p_hat[j, i] = 0.0
            else:



                raise NotImplementedError("TODO: implement spectral domain Poisson solve")


    p = np.zeros_like(p_hat)
    for i in range(nx):
        p[:, i] = inverse_discrete_cosine_transform_1d(p_hat[:, i])
    for j in range(ny):
        p[j, :] = inverse_discrete_cosine_transform_1d(p[j, :])


    p -= np.mean(p)
    return p


def hermite_interpolant_coeffs(n: int, x: np.ndarray, y: np.ndarray, yp: np.ndarray) -> tuple:
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    yp = np.asarray(yp).flatten()

    nd = 2 * n
    xd = np.zeros(nd)
    xd[0::2] = x
    xd[1::2] = x


    yd = np.zeros(nd)
    yd[0] = y[0]
    if n > 1:
        yd[2::2] = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    yd[1::2] = yp


    for i in range(2, nd):
        for j in range(nd - 1, i - 1, -1):
            denom = xd[j] - xd[j + 1 - i]
            if abs(denom) < 1e-14:


                yd[j] = 0.0
            else:
                yd[j] = (yd[j] - yd[j - 1]) / denom

    return xd, yd


def hermite_interpolant_eval(xd: np.ndarray, yd: np.ndarray, x_eval: float) -> float:
    nd = len(xd)
    result = yd[-1]
    for i in range(nd - 2, -1, -1):
        result = result * (x_eval - xd[i]) + yd[i]
    return float(result)


def hermite_interpolant_derivative(xd: np.ndarray, yd: np.ndarray) -> tuple:
    nd = len(xd)
    xdp = xd[1:].copy()
    ydp = np.zeros(nd - 1)
    for i in range(nd - 1):
        ydp[i] = (i + 1) * yd[i + 1]
    return xdp, ydp


def spectral_derivative_1d(u: np.ndarray, x: np.ndarray) -> np.ndarray:
    n = len(u)
    D = np.zeros((n, n))
    c = np.ones(n)
    c[0] = 2.0
    c[-1] = 2.0

    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = (c[i] / c[j]) * ((-1) ** (i + j)) / (x[i] - x[j])
            else:
                if i == 0:
                    D[i, i] = (2.0 * (n - 1) ** 2 + 1.0) / 6.0
                elif i == n - 1:
                    D[i, i] = -(2.0 * (n - 1) ** 2 + 1.0) / 6.0
                else:
                    D[i, i] = -x[i] / (2.0 * (1.0 - x[i] ** 2))

    du = D @ u
    return du


def assemble_fem_mass_matrix_2d(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    n_nodes = nodes.shape[0]
    n_elements = elements.shape[0]
    M = np.zeros((n_nodes, n_nodes))

    for e in range(n_elements):
        idx = elements[e, :]
        x = nodes[idx, 0]
        y = nodes[idx, 1]


        area = 0.5 * abs((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))
        area = max(area, 1e-14)


        Me = (area / 12.0) * np.array([
            [2.0, 1.0, 1.0],
            [1.0, 2.0, 1.0],
            [1.0, 1.0, 2.0]
        ])

        for i in range(3):
            for j in range(3):
                M[idx[i], idx[j]] += Me[i, j]

    return M


def assemble_fem_stiffness_matrix_2d(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    n_nodes = nodes.shape[0]
    n_elements = elements.shape[0]
    K = np.zeros((n_nodes, n_nodes))

    for e in range(n_elements):
        idx = elements[e, :]
        x = nodes[idx, 0]
        y = nodes[idx, 1]


        area = 0.5 * abs((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))
        area = max(area, 1e-14)


        dN_dx = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]]) / (2.0 * area)
        dN_dy = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]]) / (2.0 * area)


        Ke = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                Ke[i, j] = area * (dN_dx[i] * dN_dx[j] + dN_dy[i] * dN_dy[j])

        for i in range(3):
            for j in range(3):
                K[idx[i], idx[j]] += Ke[i, j]

    return K


def apply_boundary_conditions_matrix(A: np.ndarray, b: np.ndarray, bc_nodes: np.ndarray, bc_values: np.ndarray) -> tuple:
    A_mod = A.copy()
    b_mod = b.copy()

    for i, node in enumerate(bc_nodes):
        A_mod[node, :] = 0.0
        A_mod[node, node] = 1.0
        b_mod[node] = bc_values[i]

    return A_mod, b_mod
