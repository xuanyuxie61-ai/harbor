import numpy as np
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve


def build_laplacian_2d(J, h):
    n = J * J
    mu = 1.0 / (h ** 2)

    main = np.full(n, -4.0)

    off1 = np.ones(n - 1)
    for i in range(1, J):
        off1[i * J - 1] = 0.0

    offJ = np.ones(n - J)


    px = np.zeros(n)
    for i in range(J):
        px[i * J] = 1.0
        px[i * J + J - 1] = 1.0

    py = np.zeros(n)
    py[:J] = 1.0
    py[n - J:] = 1.0

    diagonals = [main, off1, off1, offJ, offJ, px, py]
    offsets = [0, -1, 1, -J, J, -(J - 1), J - 1]
    L = diags(diagonals, offsets, shape=(n, n), format='lil')

    for i in range(J):
        L[i * J, i * J + J - 1] = 1.0
        L[i * J + J - 1, i * J] = 1.0
    for j in range(J):
        L[j, n - J + j] = 1.0
        L[n - J + j, j] = 1.0
    return mu * L.tocsr()


def co2_diffusion_solver(J, h, D, dt, n_steps, C0, R_soil, V_max, K_m, LAI_grid):
    n = J * J
    C = np.full(n, C0, dtype=float)
    L = build_laplacian_2d(J, h)
    I_mat = diags([np.ones(n)], [0], shape=(n, n), format='csr')
    B = I_mat - dt * D * L

    LAI_vec = LAI_grid.ravel()
    results = [C.copy()]

    for _ in range(n_steps):

        C_old = C.copy()
        for _ in range(3):
            absorption = V_max * C_old / (K_m + np.maximum(C_old, 1e-3)) * LAI_vec
            rhs = C + dt * (R_soil - absorption)
            C_new = spsolve(B, rhs)
            C_new = np.clip(C_new, 380.0, 2000.0)
            if np.max(np.abs(C_new - C_old)) < 1e-3:
                break
            C_old = C_new
        C = C_new
        results.append(C.copy())
    return results
