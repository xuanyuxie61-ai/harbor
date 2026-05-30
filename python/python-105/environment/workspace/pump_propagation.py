
import numpy as np
from linear_solver import gauss_elimination_partial_pivot


def solve_pump_envelope_fem(n_nodes: int, z_domain: tuple,
                            k_p: float, alpha_p: float,
                            gamma_eff: callable,
                            source_spdc: callable,
                            nonlinear_tol: float = 1e-9,
                            max_iter: int = 50) -> np.ndarray:
    if n_nodes < 3 or n_nodes % 2 == 0:
        raise ValueError("n_nodes 必须为奇数且至少为 3。")
    if k_p <= 0.0:
        raise ValueError("k_p 必须为正。")
    if alpha_p < 0.0:
        raise ValueError("alpha_p 必须非负。")

    z_min, z_max = z_domain
    z = np.linspace(z_min, z_max, n_nodes)
    n_elements = (n_nodes - 1) // 2


    xi_q = np.array([-0.7745966692414834, 0.0, 0.7745966692414834])
    w_q = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])


    A_p = np.ones(n_nodes, dtype=np.complex128) * 1.0e3


    A0 = 1.0e4
    A_p[0] = A0

    for it in range(max_iter):
        A_old = A_p.copy()
        K = np.zeros((n_nodes, n_nodes), dtype=np.complex128)
        F = np.zeros(n_nodes, dtype=np.complex128)

        for e in range(n_elements):
            l = 2 * e
            m = 2 * e + 1
            r = 2 * e + 2
            zl, zm, zr = z[l], z[m], z[r]
            h_e = zr - zl

            for q in range(3):
                xi = xi_q[q]
                zq = 0.5 * ((1.0 - xi) * zl + (1.0 + xi) * zr)
                wq = w_q[q] * h_e * 0.5


                phi = np.array([
                    0.5 * xi * (xi - 1.0),
                    1.0 - xi ** 2,
                    0.5 * xi * (xi + 1.0)
                ], dtype=np.float64)
                dphi_dxi = np.array([
                    xi - 0.5,
                    -2.0 * xi,
                    xi + 0.5
                ], dtype=np.float64)
                dz_dxi = h_e / 2.0
                dphi_dz = dphi_dxi / dz_dxi


                Aq = A_old[l] * phi[0] + A_old[m] * phi[1] + A_old[r] * phi[2]

                gamma_val = gamma_eff(zq, Aq)
                source_val = source_spdc(zq)


                coeffs = np.array([l, m, r], dtype=int)
                for i_loc in range(3):
                    i = coeffs[i_loc]
                    F[i] += wq * source_val * phi[i_loc]
                    for j_loc in range(3):
                        j = coeffs[j_loc]

                        conv = phi[i_loc] * dphi_dz[j_loc]

                        diff = (1j / (2.0 * k_p)) * dphi_dz[i_loc] * dphi_dz[j_loc]

                        reac = phi[i_loc] * phi[j_loc] * (0.5 * alpha_p + 1j * gamma_val * abs(Aq) ** 2)
                        K[i, j] += wq * (conv + diff + reac)


        K[0, :] = 0.0
        K[0, 0] = 1.0
        F[0] = A0



        K[-1, :] = 0.0
        K[-1, -1] = 1.0
        F[-1] = A_old[-1] * 0.5


        n = n_nodes
        K_real = np.zeros((2 * n, 2 * n), dtype=np.float64)
        F_real = np.zeros(2 * n, dtype=np.float64)
        K_real[:n, :n] = K.real
        K_real[:n, n:] = -K.imag
        K_real[n:, :n] = K.imag
        K_real[n:, n:] = K.real
        F_real[:n] = F.real
        F_real[n:] = F.imag

        try:
            x_sol = gauss_elimination_partial_pivot(K_real, F_real)
        except ValueError as e:
            raise RuntimeError(f"FEM 线性求解失败: {e}")

        A_p = x_sol[:n] + 1j * x_sol[n:]
        err = np.linalg.norm(A_p - A_old) / max(np.linalg.norm(A_p), 1.0)
        if err < nonlinear_tol:
            break
    else:

        pass

    return A_p


def burgers_like_pump_solution(nu_eff: float, z_grid: np.ndarray,
                                t_grid: np.ndarray) -> np.ndarray:
    if nu_eff <= 0.0:
        raise ValueError("nu_eff 必须为正。")
    z = np.atleast_1d(z_grid)
    t = np.atleast_1d(t_grid)
    nz = z.size
    nt = t.size
    U = np.zeros((nz, nt), dtype=np.float64)



    x_h, w_h = _hermite_ek_compute(8)

    for ti in range(nt):
        tv = t[ti]
        if tv <= 1e-12:
            U[:, ti] = -np.sin(np.pi * z)
            continue

        c = 2.0 * np.sqrt(nu_eff * tv)
        for zi in range(nz):
            zv = z[zi]
            top = 0.0
            bot = 0.0
            for qi in range(8):
                eta = zv - c * x_h[qi]
                arg = -np.cos(np.pi * eta) / (2.0 * np.pi * nu_eff)
                w_exp = w_h[qi] * c * np.exp(arg)
                top += -(eta - zv) / tv * w_exp
                bot += w_exp
            if abs(bot) > 1e-20:
                U[zi, ti] = top / bot
            else:
                U[zi, ti] = 0.0

    return U


def _hermite_ek_compute(n: int):
    from numpy.polynomial.hermite import hermgauss
    x, w = hermgauss(n)

    return x, w
