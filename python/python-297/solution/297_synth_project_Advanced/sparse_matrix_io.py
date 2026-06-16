"""
sparse_matrix_io.py
====================
Sparse matrix I/O for the dusty plasma crystal interaction matrix.

Physical motivation:
    The Coulomb/dynamical matrix for a dusty plasma crystal is sparse
    because each dust grain only interacts significantly with its nearest
    neighbors (Yukawa potential decays exponentially). For a crystal of
    N_d grains in 2D, the dynamical matrix is 2N_d x 2N_d but has at
    most ~6 nonzero blocks per row (hexagonal coordination).

    We implement:
        1. Compressed Sparse Column (CSC) format I/O (Harwell-Boeing style, from 507_hb_io)
        2. Matrix Market format I/O (from 782_msm_to_mm)

    The interaction matrix between dust grains i and j is:
        K_ij = (Q_d^2 / (4*pi*eps0)) * d^2/dr_i dr_j [exp(-|r_i-r_j|/lambda_D) / |r_i-r_j|]

References:
    - Duff, Grimes, Reid, "User's Guide for the Harwell-Boeing Sparse Matrix Collection"
    - Boisvert et al., "Matrix Market Exchange Formats", NIST Tech Note (1996)
"""

import numpy as np
from typing import Tuple, Dict, Optional, Any
import os


# ============================================================
# Harwell-Boeing Format I/O (from 507_hb_io)
# ============================================================

class HarwellBoeingMatrix:
    """
    Sparse matrix in Harwell-Boeing format.

    The HB format stores a sparse matrix in CSC (Compressed Sparse Column):
        Header:
            Title (72 chars) + Key (8 chars)
            Totcrd, Ptrcrd, Indcrd, Valcrd, Rhscrd
            Mxtype (3 chars): R=real, C=complex, P=pattern
                              A=assembled, E=elemental
                              S=symmetric, U=unsymmetric,
                              Z=skew, H=Hermitian
            Nrow, Ncol, Nnzero, Neltvl (for elemental)
            Ptrfmt, Indfmt, Valfmt, Rhsfmt

        Data:
            Column pointers (colptr): length Ncol+1
            Row indices (rowind): length Nnzero
            Values (values): length Nnzero
            Optional RHS vectors
    """

    def __init__(self):
        self.title: str = ""
        self.key: str = ""
        self.mxtype: str = "RAU"  # Real, Assembled, Unsymmetric
        self.nrow: int = 0
        self.ncol: int = 0
        self.nnzero: int = 0
        self.colptr: Optional[np.ndarray] = None
        self.rowind: Optional[np.ndarray] = None
        self.values: Optional[np.ndarray] = None
        self.rhs: Optional[np.ndarray] = None
        self.rhstype: str = ""

    @classmethod
    def from_scipy_sparse(cls, matrix, title: str = "DustyPlasmaMatrix") -> "HarwellBoeingMatrix":
        """
        Create HB matrix from a scipy sparse matrix (or dense array).

        Parameters
        ----------
        matrix : scipy.sparse matrix or np.ndarray
            Input matrix.
        title : str
            Title for the HB file.

        Returns
        -------
        hb : HarwellBoeingMatrix
        """
        hb = cls()
        hb.title = title[:72].ljust(72)
        hb.key = "DUST".ljust(8)

        if hasattr(matrix, 'tocsc'):
            csc = matrix.tocsc()
            hb.nrow = csc.shape[0]
            hb.ncol = csc.shape[1]
            hb.colptr = csc.indptr.astype(np.int32)
            hb.rowind = csc.indices.astype(np.int32)
            hb.values = csc.data.astype(np.float64)
            hb.nnzero = csc.nnz
        else:
            arr = np.asarray(matrix, dtype=np.float64)
            hb.nrow, hb.ncol = arr.shape
            colptr_list = [0]
            rowind_list = []
            values_list = []
            for j in range(hb.ncol):
                for i in range(hb.nrow):
                    if abs(arr[i, j]) > 1e-30:
                        rowind_list.append(i)
                        values_list.append(arr[i, j])
                colptr_list.append(len(rowind_list))
            hb.colptr = np.array(colptr_list, dtype=np.int32)
            hb.rowind = np.array(rowind_list, dtype=np.int32)
            hb.values = np.array(values_list, dtype=np.float64)
            hb.nnzero = len(values_list)

        hb.mxtype = "RAU"
        return hb

    def write_hb(self, filename: str) -> None:
        """
        Write matrix in Harwell-Boeing format.

        Parameters
        ----------
        filename : str
            Output file path.
        """
        # Determine format widths
        nnz = self.nnzero
        ptr_width = max(len(str(self.ncol + 1)), 5)
        ind_width = max(len(str(self.nrow - 1)), 5)

        ptrfmt = f"({max(1, 80 // (ptr_width + 2))}I{ptr_width})"
        indfmt = f"({max(1, 80 // (ind_width + 2))}I{ind_width})"
        valfmt = "(5E16.8)"
        rhsfmt = "(5E16.8)"

        # Count lines
        ptr_per_line = max(1, 80 // (ptr_width + 2))
        ind_per_line = max(1, 80 // (ind_width + 2))
        ptrcrd = (self.ncol + 1 + ptr_per_line - 1) // ptr_per_line
        indcrd = (nnz + ind_per_line - 1) // ind_per_line if nnz > 0 else 0
        valcrd = (nnz + 4) // 5 if nnz > 0 else 0
        rhscrd = 0
        totcrd = ptrcrd + indcrd + valcrd + rhscrd

        with open(filename, 'w') as f:
            # Line 1: Title + Key
            f.write(f"{self.title}{self.key}\n")
            # Line 2: Line counts
            f.write(f"{totcrd:14d}{ptrcrd:14d}{indcrd:14d}{valcrd:14d}{rhscrd:14d}\n")
            # Line 3: Matrix type + dimensions (HB format: 3 chars + 11 blanks + 4*14 chars)
            f.write(f"{self.mxtype:3s}{'':11s}{self.nrow:14d}{self.ncol:14d}{self.nnzero:14d}{0:14d}\n")
            # Line 4: Formats
            f.write(f"{ptrfmt:16s}{indfmt:16s}{valfmt:20s}{rhsfmt:20s}\n")

            # Column pointers (1-indexed for HB)
            for i in range(len(self.colptr)):
                f.write(f"{self.colptr[i] + 1:{ptr_width}d} ")
                if (i + 1) % ptr_per_line == 0:
                    f.write("\n")
            if len(self.colptr) % ptr_per_line != 0:
                f.write("\n")

            # Row indices (1-indexed)
            for i in range(nnz):
                f.write(f"{self.rowind[i] + 1:{ind_width}d} ")
                if (i + 1) % ind_per_line == 0:
                    f.write("\n")
            if nnz % ind_per_line != 0:
                f.write("\n")

            # Values
            for i in range(nnz):
                f.write(f"{self.values[i]:16.8E}")
                if (i + 1) % 5 == 0:
                    f.write("\n")
            if nnz % 5 != 0:
                f.write("\n")

    @classmethod
    def read_hb(cls, filename: str) -> "HarwellBoeingMatrix":
        """
        Read matrix from Harwell-Boeing format file.

        Parameters
        ----------
        filename : str
            Input file path.

        Returns
        -------
        hb : HarwellBoeingMatrix
        """
        hb = cls()
        with open(filename, 'r') as f:
            # Line 1
            line1 = f.readline().rstrip('\n')
            hb.title = line1[:72].strip()
            hb.key = line1[72:80].strip() if len(line1) > 72 else ""

            # Line 2
            line2 = f.readline()
            totcrd = int(line2[0:14])
            ptrcrd = int(line2[14:28])
            indcrd = int(line2[28:42])
            valcrd = int(line2[42:56])
            rhscrd_line = line2[56:70].strip()
            rhscrd = int(rhscrd_line) if rhscrd_line else 0

            # Line 3
            line3 = f.readline()
            hb.mxtype = line3[0:3].strip()
            hb.nrow = int(line3[14:28].strip()) if len(line3) > 28 else 0
            hb.ncol = int(line3[28:42].strip()) if len(line3) > 42 else hb.nrow
            hb.nnzero = int(line3[42:56].strip()) if len(line3) > 56 else 0

            # Line 4
            line4 = f.readline()
            ptrfmt = line4[0:16].strip()
            indfmt = line4[16:32].strip()
            valfmt = line4[32:52].strip() if len(line4) > 32 else ""

            # Read column pointers - use split() for robustness
            colptr = []
            for _ in range(ptrcrd):
                line = f.readline()
                vals = line.split()
                for v in vals:
                    try:
                        colptr.append(int(v) - 1)  # Convert to 0-indexed
                    except ValueError:
                        pass
            hb.colptr = np.array(colptr[:hb.ncol + 1], dtype=np.int32)

            # Read row indices
            rowind = []
            for _ in range(indcrd):
                line = f.readline()
                vals = line.split()
                for v in vals:
                    try:
                        rowind.append(int(v) - 1)  # Convert to 0-indexed
                    except ValueError:
                        pass
            hb.rowind = np.array(rowind[:hb.nnzero], dtype=np.int32)

            # Read values
            if "R" in hb.mxtype or "C" in hb.mxtype:
                values = []
                for _ in range(valcrd):
                    line = f.readline()
                    vals = line.split()
                    for v in vals:
                        try:
                            values.append(float(v))
                        except ValueError:
                            pass
                hb.values = np.array(values[:hb.nnzero], dtype=np.float64)

        return hb

    def to_dense(self) -> np.ndarray:
        """Convert to dense numpy array."""
        mat = np.zeros((self.nrow, self.ncol), dtype=np.float64)
        for j in range(self.ncol):
            for idx in range(self.colptr[j], self.colptr[j + 1]):
                i = self.rowind[idx]
                mat[i, j] = self.values[idx]
        return mat


def _parse_format_width(fmt: str) -> int:
    """Parse the width from a Fortran format string like '(10I7)'."""
    import re
    m = re.search(r'[IE](\d+)', fmt)
    return int(m.group(1)) if m else 7


def _parse_format_count(fmt: str) -> int:
    """Parse the count from a Fortran format string like '(10I7)'."""
    import re
    m = re.match(r'\((\d+)', fmt)
    return int(m.group(1)) if m else 1


# ============================================================
# Matrix Market Format I/O (from 782_msm_to_mm)
# ============================================================

def write_matrix_market(
    filename: str,
    matrix: np.ndarray,
    comment: str = "Dusty plasma crystal dynamical matrix",
    symmetry: str = "general",
) -> None:
    """
    Write matrix in NIST Matrix Market format (from 782_msm_to_mm).

    Supports:
        - coordinate format (sparse): for large dynamical matrices
        - array format (dense): for small eigenvalue problems

    Parameters
    ----------
    filename : str
        Output file path.
    matrix : np.ndarray
        Matrix to write.
    comment : str
        Comment line.
    symmetry : str
        'general', 'symmetric', 'skew-symmetric', or 'hermitian'.
    """
    nrow, ncol = matrix.shape
    is_sparse = np.count_nonzero(matrix) < 0.5 * nrow * ncol

    with open(filename, 'w') as f:
        # Banner
        obj = "coordinate" if is_sparse else "array"
        f.write(f"%%MatrixMarket matrix {obj} real {symmetry}\n")
        f.write(f"% {comment}\n")
        f.write(f"% Generated for dusty plasma crystal simulation\n")

        if is_sparse:
            # Coordinate format
            entries = []
            for j in range(ncol):
                for i in range(nrow):
                    if abs(matrix[i, j]) > 1e-30:
                        if symmetry == "symmetric" and i < j:
                            continue
                        entries.append((i + 1, j + 1, matrix[i, j]))
            f.write(f"{nrow} {ncol} {len(entries)}\n")
            for row, col, val in entries:
                f.write(f"{row:8d} {col:8d} {val:20.12E}\n")
        else:
            # Array format
            f.write(f"{nrow} {ncol}\n")
            for j in range(ncol):
                for i in range(nrow):
                    f.write(f"{matrix[i, j]:20.12E}\n")


def read_matrix_market(filename: str) -> Tuple[np.ndarray, dict]:
    """
    Read matrix from Matrix Market format.

    Returns
    -------
    matrix : np.ndarray
        Dense matrix.
    info : dict
        Metadata from the banner line.
    """
    info = {"obj": "", "field": "", "symmetry": ""}
    with open(filename, 'r') as f:
        # Read banner
        banner = f.readline().strip()
        parts = banner.lower().split()
        if "%%matrixmarket" in parts[0]:
            info["obj"] = parts[2] if len(parts) > 2 else "array"
            info["field"] = parts[3] if len(parts) > 3 else "real"
            info["symmetry"] = parts[4] if len(parts) > 4 else "general"

        # Skip comments
        line = f.readline()
        while line.startswith('%'):
            line = f.readline()

        # Parse dimensions
        dims = line.strip().split()
        if info["obj"] == "coordinate":
            nrow, ncol, nnz = int(dims[0]), int(dims[1]), int(dims[2])
            matrix = np.zeros((nrow, ncol), dtype=np.float64)
            for _ in range(nnz):
                vals = f.readline().strip().split()
                i, j = int(vals[0]) - 1, int(vals[1]) - 1
                v = float(vals[2])
                matrix[i, j] = v
                if info["symmetry"] == "symmetric" and i != j:
                    matrix[j, i] = v
        else:
            nrow, ncol = int(dims[0]), int(dims[1])
            matrix = np.zeros((nrow, ncol), dtype=np.float64)
            for j in range(ncol):
                for i in range(nrow):
                    matrix[i, j] = float(f.readline().strip())

    return matrix, info
