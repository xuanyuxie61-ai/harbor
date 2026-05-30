
import numpy as np
from math import gamma as math_gamma


def legendre_gauss_nodes_weights(n):
    if n < 1:
        raise ValueError("Quadrature order must be >= 1")
    if n == 1:
        return np.array([0.0]), np.array([2.0])


    j = np.arange(1.0, n)
    beta = j / np.sqrt(4.0 * j**2 - 1.0)
    J = np.diag(beta, 1) + np.diag(beta, -1)

    eigvals, eigvecs = np.linalg.eigh(J)
    x = eigvals
    w = 2.0 * eigvecs[0, :]**2
    return x, w


def jacobi_gauss_nodes_weights(n, alpha, beta_param):
    if n < 1:
        raise ValueError("Quadrature order must be >= 1")
    if alpha <= -1.0 or beta_param <= -1.0:
        raise ValueError("Alpha and beta must be > -1")

    if n == 1:
        x0 = (beta_param - alpha) / (alpha + beta_param + 2.0)

        w0 = (2.0**(alpha + beta_param + 1.0)
              * math_gamma(alpha + 1.0) * math_gamma(beta_param + 1.0)
              / math_gamma(alpha + beta_param + 2.0))
        return np.array([x0]), np.array([w0])


    i = np.arange(1.0, n)
    ab = (beta_param**2 - alpha**2) / ((2.0 * i + alpha + beta_param)
                                        * (2.0 * i + alpha + beta_param + 2.0))
    bb = np.zeros(n - 1)
    for idx in range(len(i)):
        ii = i[idx]
        num = 4.0 * ii * (ii + alpha + beta_param) * (ii + alpha) * (ii + beta_param)
        den = ((2.0 * ii + alpha + beta_param)**2
               * (2.0 * ii + alpha + beta_param - 1.0)
               * (2.0 * ii + alpha + beta_param + 1.0))
        bb[idx] = np.sqrt(num / den)

    J = np.diag(ab) + np.diag(bb, 1) + np.diag(bb, -1)
    eigvals, eigvecs = np.linalg.eigh(J)
    x = eigvals


    mu0 = (2.0**(alpha + beta_param + 1.0)
           * math_gamma(alpha + 1.0) * math_gamma(beta_param + 1.0)
           / math_gamma(alpha + beta_param + 2.0))
    w = mu0 * eigvecs[0, :]**2
    return x, w


def pyramid_tensor_product_quadrature(legendre_order, jacobi_order):
    leg_x, leg_w = legendre_gauss_nodes_weights(legendre_order)
    jac_x, jac_w = jacobi_gauss_nodes_weights(jacobi_order, 2.0, 0.0)

    npts = legendre_order * legendre_order * jacobi_order
    xpts = np.zeros((3, npts))
    w = np.zeros(npts)

    volume = 4.0 / 3.0
    l = 0
    for k in range(jacobi_order):
        zk = (jac_x[k] + 1.0) / 2.0
        wk = jac_w[k] / 2.0
        for j in range(legendre_order):
            xj = leg_x[j]
            wj = leg_w[j]
            for i in range(legendre_order):
                xi = leg_x[i]
                wi = leg_w[i]
                xpts[0, l] = xi * (1.0 - zk)
                xpts[1, l] = xj * (1.0 - zk)
                xpts[2, l] = zk
                w[l] = wi * wj * wk / (4.0 * volume)
                l += 1

    return xpts, w


def integrate_reaction_rate_over_pdf(omega_func, pdf_func, legendre_order=4,
                                     jacobi_order=4, pdf_params=None):
    xpts, w = pyramid_tensor_product_quadrature(legendre_order, jacobi_order)







    mean_rate = 0.0
    for i in range(len(w)):
        Y_h2 = max(0.0, min(1.0, (xpts[0, i] + 1.0) / 2.0))
        Y_o2 = max(0.0, min(1.0, (xpts[1, i] + 1.0) / 2.0))
        T_norm = max(0.0, min(1.0, xpts[2, i]))
        T = 300.0 + T_norm * 2200.0


        total = Y_h2 + Y_o2
        if total > 0:
            Y_h2 = Y_h2 / total * 0.3
            Y_o2 = Y_o2 / total * 0.7
        else:
            Y_h2 = 0.0
            Y_o2 = 0.233
        Y_h2o = 0.0
        Y_n2 = 1.0 - Y_h2 - Y_o2 - Y_h2o
        Y = np.array([Y_h2, Y_o2, Y_h2o, Y_n2])
        Y = Y / Y.sum()

        omega_val = omega_func(Y, T)
        pdf_val = pdf_func(Y, T, 101325.0, pdf_params)
        mean_rate += w[i] * omega_val * pdf_val

    return mean_rate


def gauss_legendre_2d_tensor(nx, ny, ax, bx, ay, by):
    x1d, wx = legendre_gauss_nodes_weights(nx)
    y1d, wy = legendre_gauss_nodes_weights(ny)


    x1d = 0.5 * (bx - ax) * x1d + 0.5 * (bx + ax)
    y1d = 0.5 * (by - ay) * y1d + 0.5 * (by + ay)
    wx = 0.5 * (bx - ax) * wx
    wy = 0.5 * (by - ay) * wy

    npts = nx * ny
    x = np.zeros(npts)
    y = np.zeros(npts)
    w = np.zeros(npts)
    idx = 0
    for j in range(ny):
        for i in range(nx):
            x[idx] = x1d[i]
            y[idx] = y1d[j]
            w[idx] = wx[i] * wy[j]
            idx += 1
    return x, y, w
