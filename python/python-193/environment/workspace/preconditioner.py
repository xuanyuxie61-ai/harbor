
import numpy as np
from utils import r8vec_bracket


def jacobi_preconditioner(A):
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    m = np.zeros(n)
    for i in range(n):
        diag = A[i, i]
        if abs(diag) < 1e-15:
            m[i] = 1.0
        else:
            m[i] = 1.0 / diag
    return m


def ssor_preconditioner(A, omega=1.0):
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    L = np.tril(A, -1)
    U = np.triu(A, 1)
    D = np.diag(np.diag(A))
    D_inv = np.diag(1.0 / np.maximum(np.diag(A), 1e-15))

    M1 = D + omega * L
    M2 = D_inv
    M3 = D + omega * U

    def solve(r):

        y = np.linalg.solve(M1, r)

        z = M2 @ y

        x = np.linalg.solve(M3, z)
        return x

    return solve


def construct_prolongation_1d(fine_nodes, coarse_nodes):
    fine_nodes = np.asarray(fine_nodes, dtype=float)
    coarse_nodes = np.asarray(coarse_nodes, dtype=float)
    nf = len(fine_nodes)
    nc = len(coarse_nodes)

    P = np.zeros((nf, nc))
    for f in range(nf):
        xf = fine_nodes[f]
        left = r8vec_bracket(coarse_nodes, xf)
        if left < 0:

            if nc >= 1:
                P[f, 0] = 1.0
        elif left >= nc - 1:

            if nc >= 1:
                P[f, nc - 1] = 1.0
        else:
            x0 = coarse_nodes[left]
            x1 = coarse_nodes[left + 1]
            denom = x1 - x0
            if abs(denom) < 1e-15:
                t = 0.0
            else:
                t = (xf - x0) / denom
            P[f, left] = 1.0 - t
            P[f, left + 1] = t
    return P


def geometric_coarsening(nodes, elements, coarsening_ratio=0.5):
    nodes = np.asarray(nodes, dtype=float)
    elements = np.asarray(elements, dtype=int)
    n_nodes = nodes.shape[0]


    order = np.lexsort((nodes[:, 1], nodes[:, 0]))
    n_coarse = max(1, int(np.ceil(n_nodes * coarsening_ratio)))
    coarse_set = set(order[::max(1, len(order) // n_coarse)][:n_coarse])
    coarse_idx = sorted(coarse_set)
    fine_idx = [i for i in range(n_nodes) if i not in coarse_set]

    coarse_nodes = nodes[coarse_idx, :]
    fine_nodes = nodes[fine_idx, :]


    nc = len(coarse_idx)
    nf = len(fine_idx)
    P = np.zeros((n_nodes, nc))
    for i, global_i in enumerate(coarse_idx):
        P[global_i, i] = 1.0

    for f, global_f in enumerate(fine_idx):
        xf, yf = fine_nodes[f]

        dists = np.sum((coarse_nodes - np.array([xf, yf])) ** 2, axis=1)
        nearest = np.argsort(dists)[:min(3, nc)]

        w = np.zeros(len(nearest))
        for k, idx in enumerate(nearest):
            d = np.sqrt(dists[idx])
            w[k] = 1.0 / max(d, 1e-10)
        w /= np.sum(w)
        for k, idx in enumerate(nearest):
            P[global_f, idx] = w[k]

    return coarse_idx, fine_idx, P


def multigrid_v_cycle(A, b, x0, P, smoother_sweeps=2, omega=0.8, max_levels=3, level=0):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    x = np.asarray(x0, dtype=float)
    n = A.shape[0]


    D_inv = 1.0 / np.maximum(np.diag(A), 1e-15)

    def smooth(y, rhs, sweeps):
        for _ in range(sweeps):
            y = y + omega * D_inv * (rhs - A @ y)
        return y


    x = smooth(x, b, smoother_sweeps)


    r = b - A @ x


    if level < max_levels - 1 and P is not None and P.shape[1] < n:
        P_mat = np.asarray(P, dtype=float)
        A_c = P_mat.T @ A @ P_mat
        r_c = P_mat.T @ r

        e_c = multigrid_v_cycle(A_c, r_c, np.zeros(len(r_c)), None,
                                smoother_sweeps, omega, max_levels, level + 1)
        x = x + P_mat @ e_c
    else:

        try:
            e = np.linalg.solve(A, r)
            x = x + e
        except np.linalg.LinAlgError:
            x = x + D_inv * r


    x = smooth(x, b, smoother_sweeps)
    return x


def multigrid_preconditioner(A, P=None, smoother_sweeps=2, omega=0.8, max_levels=3):
    def solve(r):
        return multigrid_v_cycle(A, r, np.zeros_like(r), P,
                                 smoother_sweeps, omega, max_levels, 0)
    return solve
