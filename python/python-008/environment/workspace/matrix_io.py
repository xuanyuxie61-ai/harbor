
import numpy as np


def dense_to_mm_array(A, symmetry='general'):
    A = np.asarray(A)
    m, n = A.shape
    lines = []
    lines.append('%%MatrixMarket matrix array real {}'.format(symmetry.lower()))
    lines.append('{}  {}'.format(m, n))

    for j in range(n):
        for i in range(m):
            if symmetry.lower() == 'symmetric' and i > j:
                continue
            if symmetry.lower() == 'skew-symmetric' and i >= j:
                continue
            lines.append('{:.16e}'.format(A[i, j]))

    return '\n'.join(lines)


def sparse_to_mm_coordinate(row, col, val, m, n, symmetry='general'):
    row = np.asarray(row, dtype=int)
    col = np.asarray(col, dtype=int)
    val = np.asarray(val, dtype=float)

    nz = row.size
    lines = []
    lines.append('%%MatrixMarket matrix coordinate real {}'.format(symmetry.lower()))
    lines.append('{}  {}  {}'.format(m, n, nz))

    for k in range(nz):
        lines.append('{}  {}  {:.16e}'.format(row[k] + 1, col[k] + 1, val[k]))

    return '\n'.join(lines)


def write_mm(filename, A, format_type='array', symmetry='general'):
    if format_type.lower() == 'array':
        text = dense_to_mm_array(A, symmetry)
    elif format_type.lower() == 'coordinate':
        row, col, val, m, n = A
        text = sparse_to_mm_coordinate(row, col, val, m, n, symmetry)
    else:
        raise ValueError("format_type must be 'array' or 'coordinate'")

    with open(filename, 'w') as f:
        f.write(text)


def read_mm_coordinate(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()


    idx = 0
    while lines[idx].strip().startswith('%'):
        idx += 1

    header = lines[idx].strip().split()
    m, n, nz = int(header[0]), int(header[1]), int(header[2])

    row = np.zeros(nz, dtype=int)
    col = np.zeros(nz, dtype=int)
    val = np.zeros(nz, dtype=float)

    for k in range(nz):
        parts = lines[idx + 1 + k].strip().split()
        row[k] = int(parts[0]) - 1
        col[k] = int(parts[1]) - 1
        val[k] = float(parts[2])

    return row, col, val, m, n


def export_grb_matrix(A_dense, filename_prefix='grb_matrix'):
    files = []


    fname_array = filename_prefix + '_array.mtx'
    write_mm(fname_array, A_dense, format_type='array', symmetry='general')
    files.append(fname_array)


    nz_mask = np.abs(A_dense) > 0.0
    row_idx, col_idx = np.where(nz_mask)
    vals = A_dense[row_idx, col_idx]
    fname_coord = filename_prefix + '_coordinate.mtx'
    write_mm(fname_coord, (row_idx, col_idx, vals, A_dense.shape[0], A_dense.shape[1]),
             format_type='coordinate', symmetry='general')
    files.append(fname_coord)

    return files
