
import numpy as np
from typing import Callable, Tuple





def fem1d_bvp_quadratic(
    n: int,
    a_func: Callable[[np.ndarray], np.ndarray],
    c_func: Callable[[np.ndarray], np.ndarray],
    f_func: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    left_bc: Tuple[str, float] = ("dirichlet", 0.0),
    right_bc: Tuple[str, float] = ("dirichlet", 0.0),
) -> np.ndarray:
    if n < 3:
        raise ValueError("fem1d_bvp_quadratic: n must be >= 3.")
    if n % 2 == 0:
        raise ValueError("fem1d_bvp_quadratic: n must be odd for quadratic elements.")
    if x.shape[0] != n:
        raise ValueError("fem1d_bvp_quadratic: x length must equal n.")
    if np.any(np.diff(x) <= 0):
        raise ValueError("fem1d_bvp_quadratic: x must be strictly increasing.")


    abscissa = np.array([-0.7745966692414834, 0.0, 0.7745966692414834], dtype=float)
    weight = np.array([0.5555555555555556, 0.8888888888888889, 0.5555555555555556], dtype=float)
    quad_num = 3

    A_mat = np.zeros((n, n), dtype=float)
    b_vec = np.zeros(n, dtype=float)

    e_num = (n - 1) // 2

    for e in range(e_num):
        l = 2 * e
        m = 2 * e + 1
        r = 2 * e + 2

        xl, xm, xr = x[l], x[m], x[r]
        h = xr - xl
        if h <= 0:
            raise ValueError("fem1d_bvp_quadratic: element length must be positive.")

        for q in range(quad_num):

            xi = abscissa[q]
            xq = 0.5 * ((1.0 - xi) * xl + (1.0 + xi) * xr)
            wq = weight[q] * h * 0.5

            axq = float(a_func(np.array([xq]))[0])
            cxq = float(c_func(np.array([xq]))[0])
            fxq = float(f_func(np.array([xq]))[0])





            N = np.array([0.5 * xi * (xi - 1.0), 1.0 - xi ** 2, 0.5 * xi * (xi + 1.0)], dtype=float)
            dN_dxi = np.array([xi - 0.5, -2.0 * xi, xi + 0.5], dtype=float)
            dN_dx = dN_dxi * (2.0 / h)


            for i_local, i_global in enumerate([l, m, r]):
                for j_local, j_global in enumerate([l, m, r]):
                    A_mat[i_global, j_global] += wq * (
                        dN_dx[i_local] * axq * dN_dx[j_local]
                        + N[i_local] * cxq * N[j_local]
                    )
                b_vec[i_global] += wq * N[i_local] * fxq


    if left_bc[0] == "dirichlet":
        A_mat[0, :] = 0.0
        A_mat[0, 0] = 1.0
        b_vec[0] = left_bc[1]
    elif left_bc[0] == "neumann":

        pass
    else:
        raise ValueError("fem1d_bvp_quadratic: left_bc type must be 'dirichlet' or 'neumann'.")

    if right_bc[0] == "dirichlet":
        A_mat[-1, :] = 0.0
        A_mat[-1, -1] = 1.0
        b_vec[-1] = right_bc[1]
    elif right_bc[0] == "neumann":
        pass
    else:
        raise ValueError("fem1d_bvp_quadratic: right_bc type must be 'dirichlet' or 'neumann'.")


    u = np.linalg.solve(A_mat, b_vec)
    return u





def assemble_mass_stiffness_1d(n: int, L: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    if n < 1:
        raise ValueError("assemble_mass_stiffness_1d: n must be >= 1.")
    if L <= 0:
        raise ValueError("assemble_mass_stiffness_1d: L must be > 0.")

    h = L / n


    M = np.zeros((n + 1, n + 1), dtype=float)
    main_diag_m = np.full(n + 1, 2.0 * h / 3.0)
    main_diag_m[0] = h / 3.0
    main_diag_m[-1] = h / 3.0
    off_diag_m = np.full(n, h / 6.0)
    M = np.diag(main_diag_m) + np.diag(off_diag_m, k=1) + np.diag(off_diag_m, k=-1)


    K = np.zeros((n + 1, n + 1), dtype=float)
    main_diag_k = np.full(n + 1, 2.0 / h)
    main_diag_k[0] = 1.0 / h
    main_diag_k[-1] = 1.0 / h
    off_diag_k = np.full(n, -1.0 / h)
    K = np.diag(main_diag_k) + np.diag(off_diag_k, k=1) + np.diag(off_diag_k, k=-1)

    return M, K


def reaction_diffusion_nonlinear(w: np.ndarray, c_array: np.ndarray,
                                  n: int, M: np.ndarray) -> np.ndarray:
    if c_array.shape[0] < 4:
        raise ValueError("reaction_diffusion_nonlinear: c_array must have at least 4 elements.")
    if w.shape[0] != n + 1:
        raise ValueError("reaction_diffusion_nonlinear: w length must be n+1.")
    if M.shape != (n + 1, n + 1):
        raise ValueError("reaction_diffusion_nonlinear: M shape mismatch.")

    ones_vec = np.ones(n + 1, dtype=float)
    ones_vec[0] = 0.5
    ones_vec[-1] = 0.5
    ones_vec /= n

    Nq_val = _nonlinear_quadratic(w, n)
    Nc_val = _nonlinear_cubic(w, n)

    val = (c_array[0] * ones_vec
           + c_array[1] * (M @ w)
           + c_array[2] * Nq_val
           + c_array[3] * Nc_val)
    return val


def _nonlinear_quadratic(w: np.ndarray, n: int) -> np.ndarray:
    w2 = w ** 2
    wx = (w[:-1] + w[1:]) ** 2
    val = np.zeros_like(w)
    val[0] = 2.0 * w2[0] + wx[0]
    val[1:-1] = wx[:-1] + 4.0 * w2[1:-1] + wx[1:]
    val[-1] = wx[-1] + 2.0 * w2[-1]
    val /= (12.0 * n)
    return val


def _nonlinear_cubic(w: np.ndarray, n: int) -> np.ndarray:
    w2 = w ** 2
    w3 = w * w2
    wx = (w[:-1] + w[1:]) ** 3
    val = np.zeros_like(w)
    val[0] = 3.0 * w3[0] + wx[0] - w[0] * w[1] ** 2
    val[1:-1] = (wx[:-1] + 6.0 * w3[1:-1] + wx[1:]
                 - w[1:-1] * (w2[:-2] + w2[2:]))
    val[-1] = wx[-1] + 3.0 * w3[-1] - w[-1] * w[-2] ** 2
    val /= (20.0 * n)
    return val





def solve_poisson_boltzmann_membrane(
    n: int = 65,
    z_min: float = -30.0,
    z_max: float = 30.0,
    epsilon_water: float = 80.0,
    epsilon_protein: float = 4.0,
    epsilon_membrane: float = 2.0,
    kappa_water: float = 0.1,
    protein_z_range: Tuple[float, float] = (-10.0, 10.0),
    membrane_z_range: Tuple[float, float] = (-15.0, -10.0),
    charge_density: Callable[[np.ndarray], np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if n % 2 == 0:
        n += 1
    if n < 3:
        n = 3

    z = np.linspace(z_min, z_max, n)

    def eps_profile(zz: np.ndarray) -> np.ndarray:
        eps = np.full_like(zz, epsilon_water)
        mask_mem = (zz >= membrane_z_range[0]) & (zz <= membrane_z_range[1])
        mask_prot = (zz > membrane_z_range[1]) & (zz < protein_z_range[1])
        eps[mask_mem] = epsilon_membrane
        eps[mask_prot] = epsilon_protein
        return eps

    def kappa_profile(zz: np.ndarray) -> np.ndarray:
        kap = np.full_like(zz, kappa_water)
        mask_mem = (zz >= membrane_z_range[0]) & (zz <= membrane_z_range[1])
        mask_prot = (zz > membrane_z_range[1]) & (zz < protein_z_range[1])
        kap[mask_mem] = 0.0
        kap[mask_prot] = 0.0
        return kap

    def rho_profile(zz: np.ndarray) -> np.ndarray:
        if charge_density is not None:
            return charge_density(zz)

        return np.exp(-zz ** 2 / 10.0) / np.sqrt(10.0 * np.pi)

    phi = fem1d_bvp_quadratic(
        n=n,
        a_func=eps_profile,
        c_func=kappa_profile,
        f_func=rho_profile,
        x=z,
        left_bc=("neumann", 0.0),
        right_bc=("neumann", 0.0),
    )

    return z, phi, eps_profile(z), kappa_profile(z)
