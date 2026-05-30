
import numpy as np


_WATHEN_EM = np.array([
    [6.0, -6.0, 2.0, -8.0, 3.0, -8.0, 2.0, -6.0],
    [-6.0, 32.0, -6.0, 20.0, -8.0, 16.0, -8.0, 20.0],
    [2.0, -6.0, 6.0, -6.0, 2.0, -8.0, 3.0, -8.0],
    [-8.0, 20.0, -6.0, 32.0, -6.0, 20.0, -8.0, 16.0],
    [3.0, -8.0, 2.0, -6.0, 6.0, -6.0, 2.0, -8.0],
    [-8.0, 16.0, -8.0, 20.0, -6.0, 32.0, -6.0, 20.0],
    [2.0, -8.0, 3.0, -8.0, 2.0, -6.0, 6.0, -6.0],
    [-6.0, 20.0, -8.0, 16.0, -8.0, 20.0, -6.0, 32.0]
], dtype=float)


def wathen_st(nx, ny, nz_num=None):
    if nz_num is None:
        nz_num = wathen_st_size(nx, ny)

    row = np.zeros(nz_num, dtype=int)
    col = np.zeros(nz_num, dtype=int)
    a = np.zeros(nz_num, dtype=float)

    em = _WATHEN_EM.T / 180.0
    k = 0

    for j in range(1, ny + 1):
        for i in range(1, nx + 1):
            node = np.zeros(8, dtype=int)
            node[0] = 3 * j * nx + 2 * i + 2 * j + 1
            node[1] = node[0] - 1
            node[2] = node[1] - 1
            node[3] = (3 * j - 1) * nx + 2 * j + i - 1
            node[4] = 3 * (j - 1) * nx + 2 * i + 2 * j - 3
            node[5] = node[4] + 1
            node[6] = node[5] + 1
            node[7] = node[3] + 1


            rho = 50.0 * np.random.rand()

            for krow in range(8):
                for kcol in range(8):
                    row[k] = node[krow]
                    col[k] = node[kcol]
                    a[k] = rho * em[krow, kcol]
                    k += 1


    row -= 1
    col -= 1
    return row, col, a


def wathen_st_size(nx, ny):
    return nx * ny * 64


def wathen_order(nx, ny):
    return 3 * nx * ny + 2 * nx + 2 * ny + 1


def cg_sparse(n, A, b, x0=None, tol=1e-10, max_iter=None):








    pass


def solve_wathen_system(nx=4, ny=4, rhs_func=None):
    n = wathen_order(nx, ny)
    nz_num = wathen_st_size(nx, ny)
    row, col, a = wathen_st(nx, ny, nz_num)


    A_dense = np.zeros((n, n), dtype=float)
    for k in range(nz_num):
        A_dense[row[k], col[k]] += a[k]


    A_dense += 1e-6 * np.eye(n)

    if rhs_func is None:
        b = np.ones(n, dtype=float)
    else:
        b = np.zeros(n, dtype=float)
        for j in range(1, ny + 1):
            for i in range(1, nx + 1):
                node = 3 * j * nx + 2 * i + 2 * j + 1
                b[node - 1] = rhs_func(i, j)

    x = cg_sparse(n, A_dense, b)
    return x, A_dense, b
