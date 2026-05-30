
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve_triangular


class CGSolverRC:

    def __init__(self, n, tol=1e-10, max_iter=None):
        self.n = n
        self.tol = tol
        self.max_iter = max_iter if max_iter is not None else 10 * n
        self.iter = 0
        self.rho = 0.0
        self.rho_old = 0.0
        self.state = 0

    def solve(self, b, x0, matvec_func, precond_func=None):
        if precond_func is None:
            precond_func = lambda v: v.copy()

        x = x0.copy()
        r = b - matvec_func(x)
        z = precond_func(r)
        p = z.copy()
        rho = np.dot(r, z)

        for k in range(self.max_iter):
            q = matvec_func(p)
            pdotq = np.dot(p, q)
            if abs(pdotq) < 1e-30:
                break
            alpha = rho / pdotq
            x += alpha * p
            r -= alpha * q

            norm_r = np.linalg.norm(r)
            if norm_r < self.tol * np.linalg.norm(b):
                break

            z = precond_func(r)
            rho_old = rho
            rho = np.dot(r, z)
            if abs(rho_old) < 1e-30:
                break
            beta = rho / rho_old
            p = z + beta * p

        return x, k + 1, norm_r


def jacobi_preconditioner_diagonal(A_sparse):
    diag = A_sparse.diagonal().copy()
    diag = np.where(np.abs(diag) < 1e-15, 1.0, diag)
    return 1.0 / diag


def apply_jacobi_precond(d_inv, r):
    return d_inv * r


def incomplete_cholesky_preconditioner(A_sparse, drop_tol=1e-12):
    try:
        from sksparse.cholmod import cholesky
        factor = cholesky(A_sparse, beta=drop_tol)
        L = factor.L()
        return L
    except Exception:

        from scipy.sparse.linalg import spilu
        ilu = spilu(A_sparse, drop_tol=drop_tol)
        return ilu


def build_wathen_matrix(nx, ny):
    em = np.array([
        [6.0, -6.0, 2.0, -8.0, 3.0, -8.0, 2.0, -6.0],
        [-6.0, 32.0, -6.0, 20.0, -8.0, 16.0, -8.0, 20.0],
        [2.0, -6.0, 6.0, -6.0, 2.0, -8.0, 3.0, -8.0],
        [-8.0, 20.0, -6.0, 32.0, -6.0, 20.0, -8.0, 16.0],
        [3.0, -8.0, 2.0, -6.0, 6.0, -6.0, 2.0, -8.0],
        [-8.0, 16.0, -8.0, 20.0, -6.0, 32.0, -6.0, 20.0],
        [2.0, -8.0, 3.0, -8.0, 2.0, -6.0, 6.0, -6.0],
        [-6.0, 20.0, -8.0, 16.0, -8.0, 20.0, -6.0, 32.0]
    ])

    n = 3 * nx * ny + 2 * nx + 2 * ny + 1
    A = np.zeros((n, n))

    for j in range(1, ny + 1):
        for i in range(1, nx + 1):
            node = np.zeros(8, dtype=int)
            node[0] = 3 * j * nx + 2 * j + 2 * i + 1
            node[1] = node[0] - 1
            node[2] = node[0] - 2
            node[3] = (3 * j - 1) * nx + 2 * j + i - 1
            node[7] = node[3] + 1
            node[4] = (3 * j - 3) * nx + 2 * j + 2 * i - 3
            node[5] = node[4] + 1
            node[6] = node[4] + 2

            for krow in range(8):
                for kcol in range(8):
                    if 1 <= node[krow] <= n and 1 <= node[kcol] <= n:
                        A[node[krow] - 1, node[kcol] - 1] += 20.0 * em[krow, kcol] / 9.0

    return A


def build_laplacian_spherical_shell(nodes, elements, r_icb, r_cmb):
    n_nodes = len(nodes)
    row_ind = []
    col_ind = []
    data = []

    if elements.size == 0:

        for i in range(n_nodes):
            neighbors = []
            for j in range(n_nodes):
                if i == j:
                    continue
                dist = np.linalg.norm(nodes[i] - nodes[j])
                if dist < 0.3:
                    neighbors.append((j, dist))
            sum_w = 0.0
            for j, dist in neighbors:
                w = np.exp(-dist * dist / 0.02)
                row_ind.append(i)
                col_ind.append(j)
                data.append(-w)
                sum_w += w
            row_ind.append(i)
            col_ind.append(i)
            data.append(sum_w)
        return csr_matrix((data, (row_ind, col_ind)), shape=(n_nodes, n_nodes))


    for elem in elements:
        pts = nodes[elem]

        v0 = pts[1] - pts[0]
        v1 = pts[2] - pts[0]
        v2 = pts[3] - pts[0]
        vol = abs(np.dot(v0, np.cross(v1, v2))) / 6.0
        if vol < 1e-15:
            continue


        for i_idx in range(4):
            for j_idx in range(i_idx + 1, 4):
                i = elem[i_idx]
                j = elem[j_idx]
                edge_len = np.linalg.norm(nodes[i] - nodes[j])
                w = vol / (edge_len ** 2 + 1e-15)
                row_ind.append(i)
                col_ind.append(j)
                data.append(-w)
                row_ind.append(j)
                col_ind.append(i)
                data.append(-w)
                row_ind.append(i)
                col_ind.append(i)
                data.append(w)
                row_ind.append(j)
                col_ind.append(j)
                data.append(w)

    L = csr_matrix((data, (row_ind, col_ind)), shape=(n_nodes, n_nodes))
    return L


def solve_poisson_spherical_shell(rhs, nodes, elements, r_icb, r_cmb, tol=1e-10):
    L = build_laplacian_spherical_shell(nodes, elements, r_icb, r_cmb)
    n = len(nodes)


    boundary = np.zeros(n, dtype=bool)
    for i, node in enumerate(nodes):
        r = np.linalg.norm(node)
        if abs(r - r_icb) < 0.05 or abs(r - r_cmb) < 0.05:
            boundary[i] = True


    rhs_mod = rhs.copy()
    for i in np.where(boundary)[0]:
        L.data[L.indptr[i]:L.indptr[i + 1]] = 0.0
        L[i, i] = 1.0
        rhs_mod[i] = 0.0

    d_inv = jacobi_preconditioner_diagonal(L)
    cg = CGSolverRC(n, tol=tol, max_iter=min(5000, 10 * n))

    def matvec(v):
        return L.dot(v)

    def precond(v):
        return apply_jacobi_precond(d_inv, v)

    x0 = np.zeros(n)
    x, iters, resid = cg.solve(rhs_mod, x0, matvec, precond)
    return x, iters, resid
