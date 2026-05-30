
import numpy as np
from utils import ensure_positive






def shepard_interp_1d(nd, xd, yd, p, ni, xi):
    xd = np.asarray(xd, dtype=float).reshape(-1)
    yd = np.asarray(yd, dtype=float).reshape(-1)
    xi = np.asarray(xi, dtype=float).reshape(-1)

    if xd.size != nd or yd.size != nd:
        nd = min(xd.size, yd.size)
        xd = xd[:nd]
        yd = yd[:nd]
    if xi.size != ni:
        ni = xi.size

    yi = np.zeros(ni, dtype=float)

    for i in range(ni):
        if p == 0.0:
            w = np.ones(nd, dtype=float) / nd
        else:
            dist = np.abs(xi[i] - xd)
            exact = np.where(dist < 1e-15)[0]
            if exact.size > 0:
                w = np.zeros(nd, dtype=float)
                w[exact[0]] = 1.0
            else:
                w = 1.0 / (dist ** p)
                s = np.sum(w)
                if s > 1e-15:
                    w = w / s
                else:
                    w = np.ones(nd, dtype=float) / nd
        yi[i] = np.dot(w, yd)

    return yi






def quad_trapezoid(f_func, a, b, n):
    if n < 1:
        n = 1
    a = float(a)
    b = float(b)
    x = np.linspace(a, b, n + 1)
    fx = np.asarray([f_func(xi) for xi in x], dtype=float)
    h = (b - a) / n
    q = (h / 2.0) * (fx[0] + 2.0 * np.sum(fx[1:n]) + fx[n])
    return float(q)






def interpolate_vle_data(z_data, T_data, x_data, y_data, z_query, p=2.0):
    nd = len(z_data)
    nc = x_data.shape[1] if x_data.ndim > 1 else 1
    nq = len(z_query)

    T_interp = shepard_interp_1d(nd, z_data, T_data, p, nq, z_query)

    x_interp = np.zeros((nq, nc), dtype=float)
    y_interp = np.zeros((nq, nc), dtype=float)

    for j in range(nc):
        x_interp[:, j] = shepard_interp_1d(nd, z_data, x_data[:, j], p, nq, z_query)
        y_interp[:, j] = shepard_interp_1d(nd, z_data, y_data[:, j], p, nq, z_query)


    for i in range(nq):
        xs = np.sum(x_interp[i, :])
        ys = np.sum(y_interp[i, :])
        if xs > 1e-12:
            x_interp[i, :] /= xs
        if ys > 1e-12:
            y_interp[i, :] /= ys

    return T_interp, x_interp, y_interp


def integrate_mass_transfer_flux(z_nodes, N_A_func):
    a = float(z_nodes[0])
    b = float(z_nodes[-1])
    n = len(z_nodes) - 1
    if n < 1:
        n = 1
    return quad_trapezoid(N_A_func, a, b, n)
