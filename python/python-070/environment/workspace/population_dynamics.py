
import numpy as np
from utils import NumericalConfig, gauss_legendre_3point, solve_tridiagonal


def fem1d_bvp_quadratic(n, a_func, c_func, f_func, x_nodes):
    if n < 3:
        raise ValueError("n must be at least 3")
    if n % 2 != 1:
        raise ValueError("n must be odd for quadratic elements")
    if len(x_nodes) != n:
        raise ValueError("x_nodes length must equal n")

    abscissa, weight = gauss_legendre_3point()

    A_mat = np.zeros((n, n), dtype=float)
    b_vec = np.zeros(n, dtype=float)

    e_num = (n - 1) // 2

    for e in range(e_num):
        l = 2 * e
        m = 2 * e + 1
        r = 2 * e + 2

        xl = x_nodes[l]
        xm = x_nodes[m]
        xr = x_nodes[r]

        for q in range(3):

            xq = 0.5 * ((1.0 - abscissa[q]) * xl + (1.0 + abscissa[q]) * xr)
            wq = weight[q] * 0.5 * (xr - xl)

            axq = a_func(xq)
            cxq = c_func(xq)
            fxq = f_func(xq)


            vl = ((xq - xm) / (xl - xm)) * ((xq - xr) / (xl - xr))
            vm = ((xq - xl) / (xm - xl)) * ((xq - xr) / (xm - xr))
            vr = ((xq - xl) / (xr - xl)) * ((xq - xm) / (xr - xm))

            vlp = (1.0 / (xl - xm)) * ((xq - xr) / (xl - xr)) + \
                  ((xq - xm) / (xl - xm)) * (1.0 / (xl - xr))
            vmp = (1.0 / (xm - xl)) * ((xq - xr) / (xm - xr)) + \
                  ((xq - xl) / (xm - xl)) * (1.0 / (xm - xr))
            vrp = (1.0 / (xr - xl)) * ((xq - xm) / (xr - xm)) + \
                  ((xq - xl) / (xr - xl)) * (1.0 / (xr - xm))


            A_mat[l, l] += wq * (vlp * axq * vlp + vl * cxq * vl)
            A_mat[l, m] += wq * (vlp * axq * vmp + vl * cxq * vm)
            A_mat[l, r] += wq * (vlp * axq * vrp + vl * cxq * vr)
            b_vec[l] += wq * (vl * fxq)

            A_mat[m, l] += wq * (vmp * axq * vlp + vm * cxq * vl)
            A_mat[m, m] += wq * (vmp * axq * vmp + vm * cxq * vm)
            A_mat[m, r] += wq * (vmp * axq * vrp + vm * cxq * vr)
            b_vec[m] += wq * (vm * fxq)

            A_mat[r, l] += wq * (vrp * axq * vlp + vr * cxq * vl)
            A_mat[r, m] += wq * (vrp * axq * vmp + vr * cxq * vm)
            A_mat[r, r] += wq * (vrp * axq * vrp + vr * cxq * vr)
            b_vec[r] += wq * (vr * fxq)


    A_mat[0, :] = 0.0
    A_mat[0, 0] = 1.0
    b_vec[0] = 0.0

    A_mat[n - 1, :] = 0.0
    A_mat[n - 1, n - 1] = 1.0
    b_vec[n - 1] = 0.0


    u = np.linalg.solve(A_mat, b_vec)
    return u


def fem1d_nonlinear_picard_newton(n, x_nodes, p_func, q_func, f_func,
                                   nonlinear_coeff=1.0, max_iter=50, tol=1e-10):
    if n < 3 or n % 2 != 1:
        raise ValueError("n must be odd and >= 3")

    abscissa, weight = gauss_legendre_3point()
    e_num = (n - 1) // 2


    u = np.zeros(n, dtype=float)
    u_old = u.copy()

    for it in range(max_iter):
        A_mat = np.zeros((n, n), dtype=float)
        b_vec = np.zeros(n, dtype=float)

        for e in range(e_num):
            l = 2 * e
            m = 2 * e + 1
            r = 2 * e + 2

            xl = x_nodes[l]
            xm = x_nodes[m]
            xr = x_nodes[r]

            for q_idx in range(3):
                xq = 0.5 * ((1.0 - abscissa[q_idx]) * xl + (1.0 + abscissa[q_idx]) * xr)
                wq = weight[q_idx] * 0.5 * (xr - xl)

                pxq = p_func(xq)
                qxq = q_func(xq)
                fxq = f_func(xq)

                vl = ((xq - xm) / (xl - xm)) * ((xq - xr) / (xl - xr))
                vm = ((xq - xl) / (xm - xl)) * ((xq - xr) / (xm - xr))
                vr = ((xq - xl) / (xr - xl)) * ((xq - xm) / (xr - xm))

                vlp = (1.0 / (xl - xm)) * ((xq - xr) / (xl - xr)) + \
                      ((xq - xm) / (xl - xm)) * (1.0 / (xl - xr))
                vmp = (1.0 / (xm - xl)) * ((xq - xr) / (xm - xr)) + \
                      ((xq - xl) / (xm - xl)) * (1.0 / (xm - xr))
                vrp = (1.0 / (xr - xl)) * ((xq - xm) / (xr - xm)) + \
                      ((xq - xl) / (xr - xl)) * (1.0 / (xr - xm))


                if it < 5:

                    u_old_q = u_old[l] * vl + u_old[m] * vm + u_old[r] * vr
                    du_old_q = u_old[l] * vlp + u_old[m] * vmp + u_old[r] * vrp

                    b_vec[l] += wq * vl * (fxq - nonlinear_coeff * u_old_q * du_old_q)
                    b_vec[m] += wq * vm * (fxq - nonlinear_coeff * u_old_q * du_old_q)
                    b_vec[r] += wq * vr * (fxq - nonlinear_coeff * u_old_q * du_old_q)


                    for i_idx, vi, vip in [(l, vl, vlp), (m, vm, vmp), (r, vr, vrp)]:
                        for j_idx, vj, vjp in [(l, vl, vlp), (m, vm, vmp), (r, vr, vrp)]:
                            A_mat[i_idx, j_idx] += wq * (vip * pxq * vjp + vi * qxq * vj)
                else:

                    u_old_q = u_old[l] * vl + u_old[m] * vm + u_old[r] * vr
                    du_old_q = u_old[l] * vlp + u_old[m] * vmp + u_old[r] * vrp

                    for i_idx, vi, vip in [(l, vl, vlp), (m, vm, vmp), (r, vr, vrp)]:
                        for j_idx, vj, vjp in [(l, vl, vlp), (m, vm, vmp), (r, vr, vrp)]:

                            A_mat[i_idx, j_idx] += wq * (vip * pxq * vjp + vi * qxq * vj)

                            A_mat[i_idx, j_idx] += wq * nonlinear_coeff * vi * (du_old_q * vj + u_old_q * vjp)

                        b_vec[i_idx] += wq * vi * (fxq + nonlinear_coeff * u_old_q * du_old_q)


        A_mat[0, :] = 0.0
        A_mat[0, 0] = 1.0
        b_vec[0] = 0.0
        A_mat[n - 1, :] = 0.0
        A_mat[n - 1, n - 1] = 1.0
        b_vec[n - 1] = 0.0

        u = np.linalg.solve(A_mat, b_vec)

        residual = np.linalg.norm(u - u_old)
        u_old = u.copy()

        if residual < tol:
            return u, it + 1, residual

    return u, max_iter, residual


def solve_age_structured_steady_state(L_age, n_nodes, mortality_func,
                                      recruitment_rate, diffusion_age=0.1):
    if n_nodes % 2 == 0:
        n_nodes += 1

    a_nodes = np.linspace(0.0, L_age, n_nodes)

    def a_func(a):
        return diffusion_age

    def c_func(a):
        return 1.0 + mortality_func(a)

    def f_func(a):

        if a < L_age * 0.05:
            return recruitment_rate / (L_age * 0.05)
        return 0.0

    N = fem1d_bvp_quadratic(n_nodes, a_func, c_func, f_func, a_nodes)
    return a_nodes, N


def l2_error_quadratic(u_exact_func, u_fem, x_nodes):
    n = len(x_nodes)
    if n % 2 != 1 or n < 3:
        raise ValueError("n must be odd and >= 3")

    abscissa, weight = gauss_legendre_3point()
    e_num = (n - 1) // 2
    error_sq = 0.0

    for e in range(e_num):
        l = 2 * e
        m = 2 * e + 1
        r = 2 * e + 2
        xl, xm, xr = x_nodes[l], x_nodes[m], x_nodes[r]

        for q in range(3):
            xq = 0.5 * ((1.0 - abscissa[q]) * xl + (1.0 + abscissa[q]) * xr)
            wq = weight[q] * 0.5 * (xr - xl)

            vl = ((xq - xm) / (xl - xm)) * ((xq - xr) / (xl - xr))
            vm = ((xq - xl) / (xm - xl)) * ((xq - xr) / (xm - xr))
            vr = ((xq - xl) / (xr - xl)) * ((xq - xm) / (xr - xm))

            u_h = u_fem[l] * vl + u_fem[m] * vm + u_fem[r] * vr
            u_ex = u_exact_func(xq)
            error_sq += wq * (u_ex - u_h) ** 2

    return np.sqrt(error_sq)
