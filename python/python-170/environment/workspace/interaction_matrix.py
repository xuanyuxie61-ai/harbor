
import numpy as np
from scipy import sparse


def build_sparse_laplacian(positions: np.ndarray, sensing_radius: float,
                           weight_func=None, block_size: int = 64):
    N = positions.shape[0]
    if N == 0:
        return sparse.csr_matrix((0, 0)), sparse.csr_matrix((0, 0))

    if weight_func is None:
        def weight_func(d):
            return np.square(np.maximum(0.0, 1.0 - d / sensing_radius))

    rows = []
    cols = []
    vals = []

    n_blocks = int(np.ceil(N / block_size))
    for bi in range(n_blocks):
        i_start = bi * block_size
        i_end = min((bi + 1) * block_size, N)
        for bj in range(n_blocks):
            j_start = bj * block_size
            j_end = min((bj + 1) * block_size, N)

            diff = positions[i_start:i_end, np.newaxis, :] - positions[np.newaxis, j_start:j_end, :]
            dists = np.linalg.norm(diff, axis=2)
            mask = (dists > 1e-12) & (dists <= sensing_radius)
            ii, jj = np.where(mask)
            if ii.size == 0:
                continue
            global_i = i_start + ii
            global_j = j_start + jj
            wvals = weight_func(dists[mask])
            rows.append(global_i)
            cols.append(global_j)
            vals.append(wvals)

    if len(rows) == 0:
        W = sparse.csr_matrix((N, N))
    else:
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        vals = np.concatenate(vals)
        W = sparse.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()


    W = W.maximum(W.T)

    degrees = np.array(W.sum(axis=1)).ravel()
    D = sparse.diags(degrees)
    L = D - W
    return L, W


def fiedler_value(L: sparse.csr_matrix):
    from scipy.sparse.linalg import eigsh
    N = L.shape[0]
    if N <= 1:
        return 0.0
    try:

        w = eigsh(L, k=2, which='SM', return_eigenvectors=False)
        w = np.sort(w)
        lambda2 = float(w[1]) if w.size >= 2 else 0.0
    except Exception:

        lambda2 = 0.0
    return max(lambda2, 0.0)


def consensus_dynamics_step(x: np.ndarray, L: sparse.csr_matrix, dt: float):
    x = np.asarray(x, dtype=float)
    x_next = x - dt * (L.dot(x))
    return x_next
