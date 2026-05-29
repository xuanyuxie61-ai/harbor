"""
Matrix I/O Module
=================
Based on seed project 782_msm_to_mm:
- msm_to_mm.m  →  Matrix Market format conversion

Physics:
--------
Large sparse linear systems arising from radiative transfer
discretization in GRB afterglows are stored in standard formats
for interoperability.  The Matrix Market (MM) format represents
sparse matrices as:

    %%MatrixMarket matrix coordinate real general
    M  N  NZ
    i_1  j_1  a_{i1,j1}
    i_2  j_2  a_{i2,j2}
    ...

where M, N are dimensions and NZ is the number of nonzeros.

For dense arrays, the format is:

    %%MatrixMarket matrix array real general
    M  N
    a_{1,1}
    a_{2,1}
    ...

This module provides conversion between NumPy arrays/sparse
triplets and Matrix Market text representation, which is useful
for exporting GRB radiative-transfer matrices to external solvers
(e.g., PETSc, MUMPS).
"""

import numpy as np


def dense_to_mm_array(A, symmetry='general'):
    """
    Convert a dense NumPy array to Matrix Market array format string.

    Parameters
    ----------
    A : ndarray
        Dense matrix.
    symmetry : str
        'general', 'symmetric', 'skew-symmetric', or 'hermitian'.

    Returns
    -------
    text : str
        Matrix Market representation.
    """
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
    """
    Convert sparse triplet format to Matrix Market coordinate format.

    Parameters
    ----------
    row, col : ndarray
        0-based row and column indices.
    val : ndarray
        Nonzero values.
    m, n : int
        Matrix dimensions.
    symmetry : str
        'general', 'symmetric', 'skew-symmetric', or 'hermitian'.

    Returns
    -------
    text : str
        Matrix Market representation.
    """
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
    """
    Write a matrix to a file in Matrix Market format.

    Parameters
    ----------
    filename : str
        Output file path.
    A : ndarray or tuple
        Dense array, or (row, col, val, m, n) for sparse.
    format_type : str
        'array' or 'coordinate'.
    symmetry : str
        Symmetry type.
    """
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
    """
    Read a sparse matrix in Matrix Market coordinate format.

    Returns
    -------
    row, col, val, m, n : tuple
        0-based indices and dimensions.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Skip comments
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
    """
    Export a GRB radiative-transfer matrix in both dense and sparse
    Matrix Market formats.

    Returns
    -------
    files : list
        List of written file paths.
    """
    files = []

    # Dense array format
    fname_array = filename_prefix + '_array.mtx'
    write_mm(fname_array, A_dense, format_type='array', symmetry='general')
    files.append(fname_array)

    # Coordinate format (extract nonzeros)
    nz_mask = np.abs(A_dense) > 0.0
    row_idx, col_idx = np.where(nz_mask)
    vals = A_dense[row_idx, col_idx]
    fname_coord = filename_prefix + '_coordinate.mtx'
    write_mm(fname_coord, (row_idx, col_idx, vals, A_dense.shape[0], A_dense.shape[1]),
             format_type='coordinate', symmetry='general')
    files.append(fname_coord)

    return files
