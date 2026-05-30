import numpy as np


def quadrilateral_witherden_rule(p):
    p = int(p)
    if p <= 1:
        n = 1
        x = np.array([0.0])
        y = np.array([0.0])
        w = np.array([4.0])
    elif p <= 3:
        n = 4
        a = 1.0 / np.sqrt(3.0)
        x = np.array([-a, a, -a, a])
        y = np.array([-a, -a, a, a])
        w = np.array([1.0, 1.0, 1.0, 1.0])
    elif p <= 5:
        n = 9
        a = np.sqrt(3.0 / 5.0)
        x = np.array([-a, 0.0, a, -a, 0.0, a, -a, 0.0, a])
        y = np.array([-a, -a, -a, 0.0, 0.0, 0.0, a, a, a])
        wg = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
        w = np.array([wg[i] * wg[j] for i in range(3) for j in range(3)])
        w = np.array(w) * (25.0 / 9.0) / np.sum(w) * 4.0
    else:

        n1d = 7
        xi, wi = np.polynomial.legendre.leggauss(n1d)
        n = n1d * n1d
        x = np.zeros(n)
        y = np.zeros(n)
        w = np.zeros(n)
        idx = 0
        for i in range(n1d):
            for j in range(n1d):
                x[idx] = xi[i]
                y[idx] = xi[j]
                w[idx] = wi[i] * wi[j]
                idx += 1
    return n, x, y, w


def map_quad_to_physical(xi, eta, corners):
    N1 = (1.0 - xi) * (1.0 - eta) / 4.0
    N2 = (1.0 + xi) * (1.0 - eta) / 4.0
    N3 = (1.0 + xi) * (1.0 + eta) / 4.0
    N4 = (1.0 - xi) * (1.0 + eta) / 4.0
    x = N1 * corners[0, 0] + N2 * corners[1, 0] + N3 * corners[2, 0] + N4 * corners[3, 0]
    y = N1 * corners[0, 1] + N2 * corners[1, 1] + N3 * corners[2, 1] + N4 * corners[3, 1]
    return x, y


def jacobian_quad(xi, eta, corners):
    dN_dxi = np.array([-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)]) / 4.0
    dN_deta = np.array([-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)]) / 4.0
    dx_dxi = np.sum(dN_dxi * corners[:, 0])
    dx_deta = np.sum(dN_deta * corners[:, 0])
    dy_dxi = np.sum(dN_dxi * corners[:, 1])
    dy_deta = np.sum(dN_deta * corners[:, 1])
    jac = abs(dx_dxi * dy_deta - dx_deta * dy_dxi)
    return max(jac, 1e-14)


def integrate_canopy_respiration(corners, n, xq, yq, wq,
                                 lai_func, rd_func):
    total = 0.0
    for i in range(n):
        x, y = map_quad_to_physical(xq[i], yq[i], corners)
        jac = jacobian_quad(xq[i], yq[i], corners)
        lai = lai_func(x, y)
        rd = rd_func(x, y)
        total += wq[i] * jac * rd * lai
    return total


def lloyd_taylor_soil_respiration(t_soil_c, r10=2.0, e0=308.56):





    raise NotImplementedError("Hole 2: 请补全 Lloyd-Taylor 土壤呼吸公式")


def compute_nee(gpp, canopy_resp, soil_resp):
    return canopy_resp + soil_resp - gpp
