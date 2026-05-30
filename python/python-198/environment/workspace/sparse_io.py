
import numpy as np


def dense_to_csr(A):
    A = np.asarray(A, dtype=float)
    n_rows, n_cols = A.shape
    data = []
    indices = []
    indptr = [0]
    
    for i in range(n_rows):
        row_nnz = 0
        for j in range(n_cols):
            if abs(A[i, j]) > 1e-15:
                data.append(A[i, j])
                indices.append(j)
                row_nnz += 1
        indptr.append(indptr[-1] + row_nnz)
    
    return {
        'data': np.array(data),
        'indices': np.array(indices, dtype=int),
        'indptr': np.array(indptr, dtype=int),
        'shape': (n_rows, n_cols)
    }


def csr_to_dense(csr):
    n_rows, n_cols = csr['shape']
    A = np.zeros((n_rows, n_cols))
    for i in range(n_rows):
        for idx in range(csr['indptr'][i], csr['indptr'][i + 1]):
            j = csr['indices'][idx]
            A[i, j] = csr['data'][idx]
    return A


def write_hb_simple(filename, A, title="SPARSE_MATRIX"):
    A = np.asarray(A, dtype=float)
    n_rows, n_cols = A.shape
    

    rows, cols = np.where(np.abs(A) > 1e-15)
    vals = A[rows, cols]
    nnz = len(vals)
    

    order = np.lexsort((rows, cols))
    rows = rows[order]
    cols = cols[order]
    vals = vals[order]
    

    colptr = np.zeros(n_cols + 1, dtype=int)
    colptr[n_cols] = nnz
    for c in range(n_cols):
        colptr[c] = np.searchsorted(cols, c, side='left')
    
    with open(filename, 'w') as f:

        f.write(f"{title:<72}{'EXPM1':>8}\n")

        f.write(f"{0:14d}{0:14d}{0:14d}{0:14d}{0:14d}\n")

        f.write(f"{'RUA':>3}{n_rows:14d}{n_cols:14d}{nnz:14d}{0:14d}\n")

        for i, cp in enumerate(colptr):
            f.write(f"{cp:8d}")
            if (i + 1) % 10 == 0:
                f.write("\n")
        if len(colptr) % 10 != 0:
            f.write("\n")

        for i, r in enumerate(rows):
            f.write(f"{r + 1:8d}")
            if (i + 1) % 10 == 0:
                f.write("\n")
        if len(rows) % 10 != 0:
            f.write("\n")

        for i, v in enumerate(vals):
            f.write(f"{v:16.8e}")
            if (i + 1) % 5 == 0:
                f.write("\n")
        if len(vals) % 5 != 0:
            f.write("\n")


def build_pce_block_sparse(spatial_A, pce_degree, alpha_mu, alpha_sigma):
    n_elem = spatial_A.shape[0]
    n_pce = pce_degree + 1
    N = n_elem * n_pce
    
    from pce_basis import build_pce_galerkin_matrix
    A_pce = build_pce_galerkin_matrix(pce_degree, alpha_mu, alpha_sigma)
    


    I_pce = np.eye(n_pce)
    I_spatial = np.eye(n_elem)
    
    A_total = np.kron(I_spatial, A_pce) + np.kron(spatial_A, I_pce)
    return A_total
