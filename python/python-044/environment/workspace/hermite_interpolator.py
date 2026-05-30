
import numpy as np


def hermite_interpolant(n, x, y, yp):
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    yp = np.asarray(yp).flatten()

    if len(np.unique(x)) != n:
        raise ValueError("Abscissas must be distinct.")

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
            denom = xd[j] - xd[j - i]
            if abs(denom) < 1e-14:
                yd[j] = 0.0
            else:
                yd[j] = (yd[j] - yd[j - 1]) / denom


    ndp = nd - 1
    xdp = np.zeros(ndp)
    ydp = np.zeros(ndp)
    for i in range(ndp):
        xdp[i] = xd[i]
        ydp[i] = yd[i] * (ndp - i)

    for i in range(1, ndp):
        for j in range(ndp - 1, i - 1, -1):
            denom = xdp[j] - xdp[j - i]
            if abs(denom) < 1e-14:
                ydp[j] = 0.0
            else:
                ydp[j] = (ydp[j] - ydp[j - 1]) / denom

    return xd, yd, xdp, ydp


def hermite_interpolant_value(xd, yd, xdp, ydp, xv):
    xv = np.atleast_1d(xv)
    nd = len(yd)
    ndp = len(ydp)
    nv = len(xv)

    yv = np.zeros(nv)
    yvp = np.zeros(nv)

    for j in range(nv):

        yv[j] = yd[nd - 1]
        for i in range(nd - 2, -1, -1):
            yv[j] = yd[i] + (xv[j] - xd[i]) * yv[j]


        yvp[j] = ydp[ndp - 1]
        for i in range(ndp - 2, -1, -1):
            yvp[j] = ydp[i] + (xv[j] - xdp[i]) * yvp[j]

    return yv, yvp


def reconstruct_wave_field_1d(nodes, values, derivatives, eval_points):
    nodes = np.asarray(nodes)
    values = np.asarray(values)
    derivatives = np.asarray(derivatives)
    eval_points = np.asarray(eval_points)

    n = len(nodes)
    vals = np.zeros(len(eval_points))
    ders = np.zeros(len(eval_points))


    for i in range(n - 1):
        mask = (eval_points >= nodes[i]) & (eval_points <= nodes[i + 1])
        if i == 0:
            mask = mask | (eval_points < nodes[0])
        if i == n - 2:
            mask = mask | (eval_points > nodes[-1])

        if np.any(mask):
            x_local = nodes[i:i + 2]
            y_local = values[i:i + 2]
            yp_local = derivatives[i:i + 2]
            xd, yd, xdp, ydp = hermite_interpolant(2, x_local, y_local, yp_local)
            v, d = hermite_interpolant_value(xd, yd, xdp, ydp, eval_points[mask])
            vals[mask] = v
            ders[mask] = d

    return vals, ders
