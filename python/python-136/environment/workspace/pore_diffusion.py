
import numpy as np
from linear_solvers import solve_tridiagonal, solve_sparse_system


class PoreDiffusionError(Exception):
    pass


def _symmetry_bc_fd(C0, C1, dr):
    return 6.0 * (C1 - C0) / (dr ** 2)


def solve_diffusion_reaction_fd(r_nodes, D_e, reaction_func,
                                C_surface, max_iter=100, tol=1e-10):
    n = r_nodes.size
    if n < 3:
        raise PoreDiffusionError("节点数至少为 3")
    if r_nodes[0] != 0.0:
        raise PoreDiffusionError("第一个节点必须为 r=0")

    C = np.linspace(C_surface * 0.5, C_surface, n)
    C[-1] = C_surface

    for it in range(max_iter):

        a_diag = np.zeros(n)
        b_sub = np.zeros(n - 1)
        c_sup = np.zeros(n - 1)
        rhs = np.zeros(n)

















        raise NotImplementedError("Hole 1: 请实现 FDM 内部节点离散与边界处理")


        a_diag[-1] = 1.0
        b_sub[-1] = 0.0
        rhs[-1] = C_surface


        C_new = solve_tridiagonal(a_diag, b_sub, c_sup, rhs)


        relax = 0.7
        C = relax * C_new + (1.0 - relax) * C
        C[-1] = C_surface


        change = np.linalg.norm(C_new - C) / max(np.linalg.norm(C), 1e-12)
        if change < tol:
            return C, {"iter": it + 1, "resid": change}

    return C, {"iter": max_iter, "resid": change}


def solve_diffusion_reaction_fem(r_nodes, D_e, reaction_func, C_surface):
    n = r_nodes.size
    if n < 3:
        raise PoreDiffusionError("节点数至少为 3")


    xi_q = np.array([-1.0, 1.0]) / np.sqrt(3.0)
    w_q = np.array([1.0, 1.0])

    A = np.zeros((n, n))
    b_vec = np.zeros(n)

    e_num = n - 1
    for e in range(e_num):
        l = e
        r = e + 1
        xl = r_nodes[l]
        xr = r_nodes[r]
        h = xr - xl

        for q in range(2):
            xi = xi_q[q]
            rq = 0.5 * ((1.0 - xi) * xl + (1.0 + xi) * xr)
            w = w_q[q] * h / 2.0


            Nl = 0.5 * (1.0 - xi)
            Nr = 0.5 * (1.0 + xi)
            dNldr = -1.0 / h
            dNrdr = 1.0 / h


            A[l, l] += w * D_e * dNldr * dNldr * (rq ** 2)
            A[l, r] += w * D_e * dNldr * dNrdr * (rq ** 2)
            A[r, l] += w * D_e * dNrdr * dNldr * (rq ** 2)
            A[r, r] += w * D_e * dNrdr * dNrdr * (rq ** 2)


            Cq = Nl * C_surface * 0.5 + Nr * C_surface * 0.5
            Rq = reaction_func(Cq, rq)

            b_vec[l] -= w * Rq * Nl * (rq ** 2)
            b_vec[r] -= w * Rq * Nr * (rq ** 2)





    A[0, :] = 0.0
    A[0, 0] = 1.0
    A[0, 1] = -1.0
    b_vec[0] = 0.0

    A[-1, :] = 0.0
    A[-1, -1] = 1.0
    b_vec[-1] = C_surface

    C = solve_sparse_system(A, b_vec)
    return C, {"method": "FEM"}


def diffusion_flux_at_surface(C, r_nodes, D_e):
    n = r_nodes.size
    dr = r_nodes[-1] - r_nodes[-2]
    dCdr = (C[-1] - C[-2]) / dr
    flux = -D_e * dCdr
    return flux


def effectiveness_factor_from_profile(C, r_nodes, R, reaction_func):
    rates = np.array([reaction_func(Ci, ri) for Ci, ri in zip(C, r_nodes)])
    vol_int = np.trapezoid(rates * 4.0 * np.pi * r_nodes ** 2, r_nodes)
    bulk_rate = reaction_func(C[-1], R)
    if abs(bulk_rate) < np.finfo(float).eps:
        return 1.0
    denom = bulk_rate * (4.0 / 3.0) * np.pi * R ** 3
    eta = vol_int / denom
    return float(max(eta, 0.0))
