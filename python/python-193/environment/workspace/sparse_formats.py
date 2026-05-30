
import numpy as np
import os


def mm_read(filename):
    with open(filename, 'r') as f:

        line = f.readline()
        while line.strip() == '':
            line = f.readline()
        header = line.strip().lower().split()
        if len(header) < 5:
            raise ValueError("Invalid Matrix Market header")
        rep = header[2]
        field = header[3]
        symm = header[4]


        line = f.readline()
        while line.strip().startswith('%'):
            line = f.readline()
        dims = list(map(int, line.strip().split()))
        if rep == 'coordinate':
            rows, cols, entries = dims[0], dims[1], dims[2]
            data = []
            for _ in range(entries):
                parts = f.readline().strip().split()
                if field == 'pattern':
                    i, j = int(parts[0]), int(parts[1])
                    val = 1.0
                elif field in ('real', 'double', 'integer'):
                    i, j, val = int(parts[0]), int(parts[1]), float(parts[2])
                elif field == 'complex':
                    i, j = int(parts[0]), int(parts[1])
                    val = complex(float(parts[2]), float(parts[3]))
                else:
                    raise ValueError(f"Unsupported field: {field}")
                data.append((i - 1, j - 1, val))

            try:
                from scipy.sparse import coo_matrix
                rows_idx = [d[0] for d in data]
                cols_idx = [d[1] for d in data]
                vals = [d[2] for d in data]
                A = coo_matrix((vals, (rows_idx, cols_idx)), shape=(rows, cols))

                if symm == 'symmetric':
                    A = A + A.T - coo_matrix((A.diagonal(), (np.arange(min(rows, cols)), np.arange(min(rows, cols)))),
                                              shape=(rows, cols))
                elif symm == 'skew-symmetric':
                    A = A - A.T
                elif symm == 'hermitian':
                    A = A + A.conj().T - coo_matrix((A.diagonal().real, (np.arange(min(rows, cols)), np.arange(min(rows, cols)))),
                                                     shape=(rows, cols))
                return A.tocsr()
            except ImportError:

                A = np.zeros((rows, cols), dtype=complex if field == 'complex' else float)
                for i, j, val in data:
                    A[i, j] = val
                    if symm == 'symmetric' and i != j:
                        A[j, i] = val
                    elif symm == 'skew-symmetric' and i != j:
                        A[j, i] = -val
                    elif symm == 'hermitian' and i != j:
                        A[j, i] = np.conj(val)
                return A
        else:

            rows, cols = dims[0], dims[1]
            A = np.zeros((rows, cols), dtype=complex if field == 'complex' else float)
            idx = 0
            for j in range(cols):
                for i in range(rows):
                    parts = f.readline().strip().split()
                    if field == 'complex':
                        A[i, j] = complex(float(parts[0]), float(parts[1]))
                    else:
                        A[i, j] = float(parts[0])
                    idx += 1
            if symm == 'symmetric':
                for i in range(rows):
                    for j in range(i + 1, cols):
                        A[i, j] = A[j, i]
            elif symm == 'hermitian':
                for i in range(rows):
                    for j in range(i + 1, cols):
                        A[i, j] = np.conj(A[j, i])
            return A


def mm_write(filename, A, title="Generated Matrix", field="real", symm="general"):
    try:
        from scipy.sparse import coo_matrix, issparse
        if issparse(A):
            A_coo = coo_matrix(A)
            rows, cols = A_coo.shape
            entries = len(A_coo.data)
            with open(filename, 'w') as f:
                f.write(f"%%MatrixMarket matrix coordinate {field} {symm}\n")
                f.write(f"% {title}\n")
                f.write(f"{rows} {cols} {entries}\n")
                for i, j, v in zip(A_coo.row, A_coo.col, A_coo.data):
                    f.write(f"{i + 1} {j + 1} {v:.16e}\n")
        else:
            A = np.asarray(A)
            rows, cols = A.shape
            with open(filename, 'w') as f:
                f.write(f"%%MatrixMarket matrix array {field} {symm}\n")
                f.write(f"% {title}\n")
                f.write(f"{rows} {cols}\n")
                for j in range(cols):
                    for i in range(rows):
                        f.write(f"{A[i, j]:.16e}\n")
    except ImportError:
        A = np.asarray(A)
        rows, cols = A.shape
        with open(filename, 'w') as f:
            f.write(f"%%MatrixMarket matrix array {field} {symm}\n")
            f.write(f"% {title}\n")
            f.write(f"{rows} {cols}\n")
            for j in range(cols):
                for i in range(rows):
                    f.write(f"{A[i, j]:.16e}\n")


def _dense_to_csc(A):
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    data = []
    row_indices = []
    col_pointers = [0]
    for j in range(n):
        for i in range(m):
            if abs(A[i, j]) > 0.0:
                data.append(A[i, j])
                row_indices.append(i + 1)
        col_pointers.append(len(data))
    return np.array(data), np.array(row_indices, dtype=int), np.array(col_pointers, dtype=int)


def hb_write(filename, A, title="MATRIX", key="KEY", mtx_type="RUA", ifmt=8, job=3, rhs=None):
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    a, ia, ja = _dense_to_csc(A)
    nnzero = len(a)


    ptrfmt = f"({n + 1}I{ifmt})"
    indfmt = f"({nnzero}I{ifmt})"
    valfmt = "(1P,5E16.8)"
    rhsfmt = "(1P,5E16.8)"

    ptrcrd = (n + 1 + 4) // 5
    indcrd = (nnzero + 4) // 5
    valcrd = (nnzero + 4) // 5 if job >= 2 else 0
    rhscrd = 0
    has_rhs = (rhs is not None) and (job >= 3)
    if has_rhs:
        rhs = np.asarray(rhs, dtype=float).ravel()
        rhscrd = (len(rhs) + 4) // 5

    totcrd = ptrcrd + indcrd + valcrd + rhscrd

    with open(filename, 'w') as f:

        f.write(f"{title:72s}{key:8s}\n")

        f.write(f"{totcrd:14d}{ptrcrd:14d}{indcrd:14d}{valcrd:14d}{rhscrd:14d}\n")

        f.write(f"{mtx_type:3s}          {m:14d}{n:14d}{nnzero:14d}             0\n")

        f.write(f"{ptrfmt:16s}{indfmt:16s}{valfmt:20s}{rhsfmt:20s}\n")


        for i in range(0, len(ja), 5):
            vals = ja[i:i + 5]
            f.write("".join(f"{v:14d}" for v in vals) + "\n")


        for i in range(0, len(ia), 5):
            vals = ia[i:i + 5]
            f.write("".join(f"{v:14d}" for v in vals) + "\n")


        if job >= 2:
            for i in range(0, len(a), 5):
                vals = a[i:i + 5]
                f.write("".join(f"{v:16.8e}" for v in vals) + "\n")


        if has_rhs:
            for i in range(0, len(rhs), 5):
                vals = rhs[i:i + 5]
                f.write("".join(f"{v:16.8e}" for v in vals) + "\n")


def coo_to_csr(data, row, col, shape):
    try:
        from scipy.sparse import csr_matrix
        return csr_matrix((data, (row, col)), shape=shape)
    except ImportError:
        m, n = shape
        nnz = len(data)
        row_ptr = np.zeros(m + 1, dtype=int)
        for r in row:
            row_ptr[r + 1] += 1
        row_ptr = np.cumsum(row_ptr)
        col_idx = np.empty(nnz, dtype=int)
        vals = np.empty(nnz, dtype=float)
        next_pos = row_ptr[:-1].copy()
        for k in range(nnz):
            r = row[k]
            pos = next_pos[r]
            col_idx[pos] = col[k]
            vals[pos] = data[k]
            next_pos[r] += 1
        return row_ptr, col_idx, vals


def csr_matvec(row_ptr, col_idx, vals, x):
    m = len(row_ptr) - 1
    y = np.zeros(m)
    for i in range(m):
        s = 0.0
        for k in range(row_ptr[i], row_ptr[i + 1]):
            s += vals[k] * x[col_idx[k]]
        y[i] = s
    return y
