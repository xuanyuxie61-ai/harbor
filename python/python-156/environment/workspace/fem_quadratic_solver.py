
import numpy as np
from flamelet_core import (
    scalar_dissipation_rate,
    density_mixture,
    reaction_rate_one_step,
    flamelet_boundary_conditions,
    thermal_diffusivity_ref,
)


def solve_fem_quadratic_species(n, Z_nodes, species_type, T_field,
                                chi_st, tol=1.0e-10, max_iter=50):
    if n < 3 or n % 2 == 0:
        raise ValueError("二次有限元要求节点数 n 为奇数且 >= 3")
    if not np.all(np.diff(Z_nodes) > 0):
        raise ValueError("Z_nodes 必须严格单调递增")

    bc = flamelet_boundary_conditions()
    e_num = (n - 1) // 2


    abscissa = np.array([
        -0.7745966692414834,
        0.0,
        0.7745966692414834
    ])
    weight = np.array([
        0.5555555555555556,
        0.8888888888888889,
        0.5555555555555556
    ])
    quad_num = 3


    if species_type == 'fuel':
        Y = np.linspace(bc['Y_F_left'], bc['Y_F_right'], n)
        left_bc = bc['Y_F_left']
        right_bc = bc['Y_F_right']
    elif species_type == 'oxidizer':
        Y = np.linspace(bc['Y_O_left'], bc['Y_O_right'], n)
        left_bc = bc['Y_O_left']
        right_bc = bc['Y_O_right']
    else:
        raise ValueError("species_type 必须是 'fuel' 或 'oxidizer'")

    D_ref = thermal_diffusivity_ref()

    for iteration in range(max_iter):
        Y_old = Y.copy()
        A = np.zeros((n, n))
        b = np.zeros(n)

        for e in range(e_num):
            l = 2 * e
            m = 2 * e + 1
            r = 2 * e + 2

            xl = Z_nodes[l]
            xm = Z_nodes[m]
            xr = Z_nodes[r]

            for q in range(quad_num):
                xq = ((1.0 - abscissa[q]) * xl + (1.0 + abscissa[q]) * xr) / 2.0
                wq = weight[q] * (xr - xl) / 2.0


                vl = ((xq - xm) / (xl - xm)) * ((xq - xr) / (xl - xr))
                vm = ((xq - xl) / (xm - xl)) * ((xq - xr) / (xm - xr))
                vr = ((xq - xl) / (xr - xl)) * ((xq - xm) / (xr - xm))

                vlp = (1.0 / (xl - xm)) * ((xq - xr) / (xl - xr)) + \
                      ((xq - xm) / (xl - xm)) * (1.0 / (xl - xr))
                vmp = (1.0 / (xm - xl)) * ((xq - xr) / (xm - xr)) + \
                      ((xq - xl) / (xm - xl)) * (1.0 / (xm - xr))
                vrp = (1.0 / (xr - xl)) * ((xq - xm) / (xr - xm)) + \
                      ((xq - xl) / (xr - xl)) * (1.0 / (xr - xm))


                Tq = np.interp(xq, Z_nodes, T_field)
                Yq = np.interp(xq, Z_nodes, Y_old)


                rho = density_mixture(xq, Tq)
                rho_ox = density_mixture(0.0, bc['T_left'])
                D_coeff = D_ref * rho_ox / rho
                D_coeff = max(D_coeff, 1.0e-12)






                raise NotImplementedError("Hole 3: 请实现反应源项线性化处理")


                A[l, l] += wq * (vlp * D_coeff * vlp + vl * 0.0 * vl)
                A[l, m] += wq * (vlp * D_coeff * vmp + vl * 0.0 * vm)
                A[l, r] += wq * (vlp * D_coeff * vrp + vl * 0.0 * vr)
                b[l] += wq * (vl * f_source)

                A[m, l] += wq * (vmp * D_coeff * vlp + vm * 0.0 * vl)
                A[m, m] += wq * (vmp * D_coeff * vmp + vm * 0.0 * vm)
                A[m, r] += wq * (vmp * D_coeff * vrp + vm * 0.0 * vr)
                b[m] += wq * (vm * f_source)

                A[r, l] += wq * (vrp * D_coeff * vlp + vr * 0.0 * vl)
                A[r, m] += wq * (vrp * D_coeff * vmp + vr * 0.0 * vm)
                A[r, r] += wq * (vrp * D_coeff * vrp + vr * 0.0 * vr)
                b[r] += wq * (vr * f_source)


        A[0, :] = 0.0
        A[0, 0] = 1.0
        b[0] = left_bc

        A[n - 1, :] = 0.0
        A[n - 1, n - 1] = 1.0
        b[n - 1] = right_bc


        Y = np.linalg.solve(A, b)


        Y = np.clip(Y, 0.0, 1.0)

        max_change = np.max(np.abs(Y - Y_old))
        if max_change < tol:
            return Y, iteration + 1

    return Y, max_iter
