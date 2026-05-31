"""
interaction_matrix.py
=====================
Sparse interaction matrix assembly for large-scale swarm communication.

Incorporates:
  - sparse_parfor / create_block (from 1111_sparse_parfor)

Scientific role:
  In a swarm of N robots, the full N x N adjacency / Laplacian matrix is
  dense in theory but sparse in practice because each robot only communicates
  with neighbors within a sensing radius R_s. We assemble the weighted
  adjacency matrix W and the graph Laplacian L = D - W in a block-parallel
  sparse format to enable O(N) memory scaling and fast spectral operations.

  The Laplacian spectrum determines consensus convergence rate:
      lambda_2 = algebraic connectivity (Fiedler value)
  which is a key emergent order parameter.
"""

import numpy as np
from scipy import sparse


def build_sparse_laplacian(positions: np.ndarray, sensing_radius: float,
                           weight_func=None, block_size: int = 64):
    """
    Build the sparse graph Laplacian for a geometric proximity graph.

    For positions p_i in R^d, define edge (i,j) if ||p_i - p_j|| <= R_s.
    The weighted adjacency is
        W_{ij} = w(||p_i - p_j||)  for i != j
        W_{ii} = 0
    The degree matrix is D_{ii} = sum_j W_{ij}.
    The Laplacian is L = D - W.

    Parameters
    ----------
    positions : ndarray, shape (N, d)
        Robot positions.
    sensing_radius : float
        Communication / sensing range.
    weight_func : callable or None
        W(dist) mapping. If None, uses w(d) = max(0, 1 - d/R_s)^2.
    block_size : int
        Block size for chunked assembly to reduce memory spikes.

    Returns
    -------
    L : scipy.sparse.csr_matrix
        Sparse graph Laplacian.
    W : scipy.sparse.csr_matrix
        Sparse weighted adjacency.
    """
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

    # symmetrize
    W = W.maximum(W.T)

    degrees = np.array(W.sum(axis=1)).ravel()
    D = sparse.diags(degrees)
    L = D - W
    return L, W


def fiedler_value(L: sparse.csr_matrix):
    """
    Compute the algebraic connectivity (Fiedler value) of the graph.

    lambda_2 is the second-smallest eigenvalue of L. It controls the
    exponential convergence rate of the linear consensus protocol:
        dot(x) = -L x   =>   ||x(t) - x_bar|| <= C exp(-lambda_2 t)

    Parameters
    ----------
    L : scipy.sparse.csr_matrix
        Graph Laplacian.

    Returns
    -------
    lambda2 : float
        Fiedler value (>= 0). Returns 0.0 if graph is disconnected.
    """
    from scipy.sparse.linalg import eigsh
    N = L.shape[0]
    if N <= 1:
        return 0.0
    try:
        # compute two smallest eigenvalues
        w = eigsh(L, k=2, which='SM', return_eigenvectors=False)
        w = np.sort(w)
        lambda2 = float(w[1]) if w.size >= 2 else 0.0
    except Exception:
        # fallback for disconnected or very small graphs
        lambda2 = 0.0
    return max(lambda2, 0.0)


def consensus_dynamics_step(x: np.ndarray, L: sparse.csr_matrix, dt: float):
    """
    Explicit Euler step for linear consensus dynamics.

        x^{k+1} = x^k - dt * L * x^k

    Parameters
    ----------
    x : ndarray, shape (N,)
        Current state vector.
    L : scipy.sparse.csr_matrix
        Laplacian.
    dt : float
        Time step (must satisfy stability condition dt < 2 / lambda_max).

    Returns
    -------
    x_next : ndarray
        Updated state.
    """
    x = np.asarray(x, dtype=float)
    x_next = x - dt * (L.dot(x))
    return x_next
