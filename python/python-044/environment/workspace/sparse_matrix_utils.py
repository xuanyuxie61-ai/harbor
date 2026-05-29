"""
sparse_matrix_utils.py
======================
Sparse matrix utilities and graph connectivity for poroelastic FEM systems.

Incorporates:
  - Adjacency matrix construction from element connectivity (from risk_matrix)
  - Sparse matrix format conversion utilities (from msm_to_hb)
  - Harwell-Boeing format output (from msm_to_hb)
"""

import numpy as np


def build_adjacency_matrix(elements, n_nodes=None):
    """
    Build symmetric adjacency matrix from element connectivity.

    Two nodes are adjacent if they share at least one element.

    Parameters
    ----------
    elements : ndarray, shape (n_elements, nodes_per_element)
        Element connectivity (0-based).
    n_nodes : int, optional
        Total number of nodes. Inferred if not given.

    Returns
    -------
    adj : ndarray, shape (n_nodes, n_nodes)
        Symmetric adjacency matrix (0/1).
    """
    if n_nodes is None:
        n_nodes = int(elements.max()) + 1

    adj = np.zeros((n_nodes, n_nodes), dtype=int)
    n_elem = elements.shape[0]
    npe = elements.shape[1]

    for e in range(n_elem):
        for i in range(npe):
            ni = elements[e, i]
            for j in range(i + 1, npe):
                nj = elements[e, j]
                adj[ni, nj] = 1
                adj[nj, ni] = 1

    return adj


def build_sparsity_pattern(elements_u, n_nodes):
    """
    Build the sparsity pattern for the global stiffness matrix
    based on element connectivity.

    Returns a set of (i, j) pairs where nonzero entries may exist.
    """
    pattern = set()
    n_elements = elements_u.shape[0]
    npe = elements_u.shape[1]

    for e in range(n_elements):
        for i in range(npe):
            ni = elements_u[e, i]
            for j in range(npe):
                nj = elements_u[e, j]
                # For 2D displacement, each node has x and y DoFs
                pattern.add((2 * ni, 2 * nj))
                pattern.add((2 * ni, 2 * nj + 1))
                pattern.add((2 * ni + 1, 2 * nj))
                pattern.add((2 * ni + 1, 2 * nj + 1))

    return pattern


def dense_to_csr(A, tol=1e-15):
    """
    Convert dense matrix to simple CSR-like dictionaries.

    Returns
    -------
    data : list
    row_idx : list
    col_ptr : list
    """
    m, n = A.shape
    data = []
    row_idx = []
    col_ptr = [0]

    for j in range(n):
        count = 0
        for i in range(m):
            if abs(A[i, j]) > tol:
                data.append(A[i, j])
                row_idx.append(i)
                count += 1
        col_ptr.append(col_ptr[-1] + count)

    return data, row_idx, col_ptr


def write_hb_format(filename, A, rhs=None, title="PoroelasticMatrix", key="BIOT01",
                    ifmt=8, job=2):
    """
    Write a sparse matrix in simplified Harwell-Boeing format.

    This is a minimal implementation that writes the header, column pointers,
    row indices, and numerical values.

    Parameters
    ----------
    filename : str
        Output file path.
    A : ndarray
        Dense matrix (small enough to store).
    rhs : ndarray, optional
        Right-hand side vector(s).
    title : str
        Matrix title (up to 72 chars).
    key : str
        Matrix key (up to 8 chars).
    ifmt : int
        Numerical format precision.
    job : int
        1=structure only, 2=with values, 3=with RHS.
    """
    m, n = A.shape
    data, row_idx, col_ptr = dense_to_csr(A)
    nnzeros = len(data)
    nrhs = 0
    if rhs is not None and job >= 3:
        rhs = np.atleast_2d(rhs)
        nrhs = rhs.shape[0]

    # Format widths
    ptr_len = max(1, int(np.ceil(np.log10(max(nnzeros + 1, 2))))) + 1
    ind_len = ptr_len
    ptr_nperline = min(80 // ptr_len, n + 1)
    ind_nperline = min(80 // ind_len, nnzeros)
    ptrcrd = (n + 1 + ptr_nperline - 1) // ptr_nperline
    indcrd = (nnzeros + ind_nperline - 1) // ind_nperline

    valcrd = 0
    rhscrd = 0
    if job > 1:
        valcrd = nnzeros
        if nrhs > 0:
            rhscrd = nrhs * m

    totcrd = ptrcrd + indcrd + valcrd + rhscrd

    with open(filename, "w") as fid:
        # Line 1: title and key
        fid.write(f"{title:<72s}{key:<8s}\n")
        # Line 2: counts
        fid.write(f"{totcrd:14d}{ptrcrd:14d}{indcrd:14d}{valcrd:14d}{rhscrd:14d}\n")
        # Line 3: type and dimensions
        mtype = "RUA"
        fid.write(f"{mtype:14s}{m:14d}{n:14d}{nnzeros:14d}{nrhs:14d}\n")
        # Line 4: formats (simplified)
        fid.write(f"({ptr_nperline}I{ptr_len})".ljust(16))
        fid.write(f"({ind_nperline}I{ind_len})".ljust(16))
        fid.write("(1E16.8)".ljust(20))
        fid.write("(1E16.8)\n")

        # Column pointers
        for i in range(0, n + 1, ptr_nperline):
            line = col_ptr[i:i + ptr_nperline]
            fid.write("".join(f"{v:{ptr_len}d}" for v in line) + "\n")

        # Row indices
        for i in range(0, nnzeros, ind_nperline):
            line = row_idx[i:i + ind_nperline]
            fid.write("".join(f"{v:{ind_len}d}" for v in line) + "\n")

        # Values
        if job > 1:
            for v in data:
                fid.write(f"{v:16.8e}\n")

        # RHS
        if nrhs > 0 and job >= 3:
            for r in range(nrhs):
                for v in rhs[r, :]:
                    fid.write(f"{v:16.8e}\n")


def estimate_condition_number(A):
    """
    Rough estimate of matrix condition number using power iteration.
    """
    n = A.shape[0]
    x = np.random.randn(n)
    x = x / np.linalg.norm(x)

    # Estimate largest eigenvalue
    for _ in range(10):
        x = A @ x
        x = x / (np.linalg.norm(x) + 1e-30)

    lambda_max = abs(x @ (A @ x)) / (x @ x + 1e-30)

    # Estimate smallest eigenvalue via inverse iteration
    try:
        Ainv = np.linalg.inv(A + 1e-12 * np.eye(n))
        y = np.random.randn(n)
        y = y / np.linalg.norm(y)
        for _ in range(10):
            y = Ainv @ y
            y = y / (np.linalg.norm(y) + 1e-30)
        lambda_min_inv = abs(y @ (Ainv @ y)) / (y @ y + 1e-30)
        lambda_min = 1.0 / (lambda_min_inv + 1e-30)
    except np.linalg.LinAlgError:
        lambda_min = 1e-14

    cond = lambda_max / (lambda_min + 1e-30)
    return cond
