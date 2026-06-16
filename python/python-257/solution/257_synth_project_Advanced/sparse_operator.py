"""
sparse_operator.py
==================
Conversion of dense finite-difference / FEM matrices into sparse
compressed-column-storage (CCS) format.

The CCS format is equivalent to SciPy's CSC and the Harwell-Boeing
"real unsymmetric assembled" (RUA) format.  Storage requirements:
  - colptr  : int array of length n+1
  - rowind  : int array of length nnz
  - values  : float array of length nnz

We implement three conversions:
  1. GE -> CCS        (from ge_to_ccs.m)
  2. COO -> CCS       (from element-wise assembly)
  3. Band storage CCS (optimised for banded operators like Laplacian)

We also provide sparse matrix-vector multiply and sparse Frobenius norm.
"""

from __future__ import annotations
import math
from typing import List, Tuple, Dict


# ---------------------------------------------------------------------------
# GE (dense) -> CCS
# ---------------------------------------------------------------------------
def ge_to_ccs(A: List[List[float]], tol: float = 1.0e-15) -> Tuple[int, List[int], List[int], List[float]]:
    """
    Convert a dense m x n matrix to CCS (compressed-column) form.
    Returns (nnz, colptr, rowind, values).
    """
    m = len(A)
    if m == 0:
        return 0, [0], [], []
    n = len(A[0])
    colptr = [0] * (n + 1)
    entries: List[Tuple[int, int, float]] = []
    for j in range(n):
        for i in range(m):
            if abs(A[i][j]) > tol:
                entries.append((i, j, A[i][j]))
    # Sort by (column, row)
    entries.sort(key=lambda x: (x[1], x[0]))
    nnz = len(entries)
    rowind = [0] * nnz
    values = [0.0] * nnz
    k = 0
    for j in range(n):
        colptr[j] = k + 1   # 1-based (Fortran-style)
        for (i, jj, v) in entries:
            if jj == j:
                rowind[k] = i + 1
                values[k] = v
                k += 1
    colptr[n] = nnz + 1
    return nnz, colptr, rowind, values


# ---------------------------------------------------------------------------
# COO (triplet) -> CCS
# ---------------------------------------------------------------------------
def coo_to_ccs(m: int, n: int, rows: List[int], cols: List[int], vals: List[float],
                tol: float = 1.0e-15) -> Tuple[int, List[int], List[int], List[float]]:
    """
    Convert COO (triplet) format to CCS.  Duplicate entries are summed.
    """
    # Aggregate duplicates
    entries: Dict[Tuple[int, int], float] = {}
    for (r, c, v) in zip(rows, cols, vals):
        if abs(v) <= tol:
            continue
        entries[(r, c)] = entries.get((r, c), 0.0) + v
    # Sort by (col, row)
    sorted_entries = sorted(entries.items(), key=lambda x: (x[0][1], x[0][0]))
    nnz = len(sorted_entries)
    colptr = [0] * (n + 1)
    rowind = [0] * nnz
    values = [0.0] * nnz
    k = 0
    col = 0
    colptr[0] = 1
    for ((r, c), v) in sorted_entries:
        while col < c:
            col += 1
            colptr[col] = k + 1
        rowind[k] = r + 1
        values[k] = v
        k += 1
    while col < n:
        col += 1
        colptr[col] = k + 1
    return nnz, colptr, rowind, values


# ---------------------------------------------------------------------------
# CCS -> GE (for verification)
# ---------------------------------------------------------------------------
def ccs_to_ge(m: int, n: int, colptr: List[int], rowind: List[int],
                values: List[float]) -> List[List[float]]:
    """Reconstruct a dense matrix from CCS format."""
    A = [[0.0] * n for _ in range(m)]
    for j in range(n):
        for k in range(colptr[j] - 1, colptr[j + 1] - 1):
            i = rowind[k] - 1
            A[i][j] = values[k]
    return A


# ---------------------------------------------------------------------------
# Sparse matrix-vector product  y = A x
# ---------------------------------------------------------------------------
def spmv(m: int, n: int, colptr: List[int], rowind: List[int],
          values: List[float], x: List[float]) -> List[float]:
    """Sparse matrix-vector product y = A x in CCS format."""
    y = [0.0] * m
    for j in range(n):
        xj = x[j]
        for k in range(colptr[j] - 1, colptr[j + 1] - 1):
            y[rowind[k] - 1] += values[k] * xj
    return y


# ---------------------------------------------------------------------------
# Frobenius norm
# ---------------------------------------------------------------------------
def ccs_frobenius_norm(colptr: List[int], values: List[float]) -> float:
    return math.sqrt(sum(v * v for v in values))


# ---------------------------------------------------------------------------
# CCS sparsity statistics
# ---------------------------------------------------------------------------
def ccs_stats(m: int, n: int, nnz: int) -> Dict[str, float]:
    total = m * n
    return {
        "nnz": nnz,
        "total": total,
        "sparsity": 1.0 - nnz / total if total > 0 else 0.0,
        "compression_ratio": total / nnz if nnz > 0 else float("inf"),
    }


# ---------------------------------------------------------------------------
# Band detection
# ---------------------------------------------------------------------------
def detect_bandwidth(colptr: List[int], rowind: List[int], n: int) -> Tuple[int, int]:
    """
    Compute the lower and upper bandwidths of a CCS matrix.
    lower_band = max(i - j) over nonzeros
    upper_band = max(j - i) over nonzeros
    """
    lb = 0
    ub = 0
    for j in range(n):
        for k in range(colptr[j] - 1, colptr[j + 1] - 1):
            i = rowind[k] - 1
            if i - j > lb:
                lb = i - j
            if j - i > ub:
                ub = j - i
    return lb, ub


# ---------------------------------------------------------------------------
# CCS symmetry check
# ---------------------------------------------------------------------------
def ccs_is_symmetric(m: int, n: int, colptr: List[int], rowind: List[int],
                       values: List[float], tol: float = 1.0e-12) -> bool:
    """Check if CCS matrix is symmetric."""
    if m != n:
        return False
    A = ccs_to_ge(m, n, colptr, rowind, values)
    for i in range(m):
        for j in range(i + 1, n):
            if abs(A[i][j] - A[j][i]) > tol:
                return False
    return True


# ---------------------------------------------------------------------------
# CCS diagonal extraction
# ---------------------------------------------------------------------------
def ccs_diagonal(colptr: List[int], rowind: List[int], values: List[float], n: int) -> List[float]:
    diag = [0.0] * n
    for j in range(n):
        for k in range(colptr[j] - 1, colptr[j + 1] - 1):
            i = rowind[k] - 1
            if i == j:
                diag[j] = values[k]
    return diag


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 5x5 tridiagonal Laplacian
    n = 5
    A = [[0.0] * n for _ in range(n)]
    for i in range(n):
        A[i][i] = -2.0
        if i > 0: A[i][i - 1] = 1.0
        if i < n - 1: A[i][i + 1] = 1.0
    nnz, colptr, rowind, values = ge_to_ccs(A)
    print(f"nnz = {nnz}, colptr = {colptr}")
    print(f"Sparsity = {ccs_stats(n, n, nnz)['sparsity']:.4f}")
    print(f"Bandwidth = {detect_bandwidth(colptr, rowind, n)}")
    print(f"Symmetric? {ccs_is_symmetric(n, n, colptr, rowind, values)}")
