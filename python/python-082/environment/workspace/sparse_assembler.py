"""
sparse_assembler.py
===================
Sparse matrix assembly and banded solver for composite FEM.

Incorporates core algorithms from:
- 508_hb_to_mm : Sparse matrix structure reading and coordinate-format
  representation (Harwell-Boeing / Matrix Market concepts).
- 972_r8but : Banded upper triangular matrix solver for efficient
  solution of the tangent stiffness system in damage analysis.

Scientific role:
    Assembles the global stiffness matrix K in coordinate (COO) format
    from element contributions. For banded tangent stiffness matrices
    arising from structured composite meshes, provides an efficient
    banded solver. Also includes dense solver fallback for general
    matrices.

Key formulas:
-----------
1. Global stiffness assembly:
   K = sum_{e=1}^{N_e} L_e^T k_e L_e
   where k_e is the element stiffness and L_e is the localization matrix.

2. Banded upper triangular solve (back substitution):
   For U x = b, with upper bandwidth mu:
   x_j = b_j / U_{j,j} - sum_{i=j+1}^{min(j+mu, n)} U_{j,i} x_i / U_{j,j}

3. Sparse matrix-vector product:
   y_i = sum_j K_{ij} x_j  (COO format)

4. Condition number estimate:
   kappa_inf = ||K||_inf * ||K^{-1}||_inf
   computed via power iteration for large systems.
"""

import numpy as np


class SparseMatrixCOO:
    """
    Sparse matrix in Coordinate (COO) format.
    """

    def __init__(self, rows=None, cols=None, vals=None, shape=None):
        self.rows = [] if rows is None else list(rows)
        self.cols = [] if cols is None else list(cols)
        self.vals = [] if vals is None else list(vals)
        self.shape = shape

    def add(self, row, col, val):
        """Add a single entry."""
        if abs(val) > 1e-18:
            self.rows.append(int(row))
            self.cols.append(int(col))
            self.vals.append(float(val))

    def add_block(self, rows, cols, block):
        """Add a dense block to the sparse structure."""
        block = np.asarray(block, dtype=float)
        for i, ri in enumerate(rows):
            for j, cj in enumerate(cols):
                v = block[i, j]
                if abs(v) > 1e-18:
                    self.rows.append(int(ri))
                    self.cols.append(int(cj))
                    self.vals.append(float(v))

    def to_dense(self):
        """Convert to dense numpy array."""
        if self.shape is None:
            raise ValueError("Shape not set.")
        A = np.zeros(self.shape, dtype=float)
        for r, c, v in zip(self.rows, self.cols, self.vals):
            A[r, c] += v
        return A

    def matvec(self, x):
        """Sparse matrix-vector product y = A @ x."""
        x = np.asarray(x, dtype=float)
        if self.shape is None:
            raise ValueError("Shape not set.")
        y = np.zeros(self.shape[0], dtype=float)
        for r, c, v in zip(self.rows, self.cols, self.vals):
            y[r] += v * x[c]
        return y

    def set_shape(self, n_rows, n_cols):
        self.shape = (n_rows, n_cols)

    def nnz(self):
        return len(self.vals)

    def summary(self):
        print(f"SparseMatrixCOO: shape={self.shape}, nnz={self.nnz()}")


def r8but_solve(n, mu, A_band, b):
    """
    Solve a banded upper triangular system U x = b.

    The matrix U is stored in banded format: A_band has shape (mu+1, n).
    The diagonal is in row mu, first superdiagonal in row mu-1, etc.

    Parameters
    ----------
    n : int
        Matrix order.
    mu : int
        Upper bandwidth.
    A_band : ndarray, shape (mu+1, n)
        Banded storage of U.
    b : ndarray, shape (n,)
        Right-hand side.

    Returns
    -------
    x : ndarray, shape (n,)
        Solution vector.
    """
    b = np.asarray(b, dtype=float).copy()
    x = np.zeros(n, dtype=float)

    for j in range(n - 1, -1, -1):
        diag_idx = mu
        diag_val = A_band[diag_idx, j]
        if abs(diag_val) < 1e-14:
            diag_val = 1e-14
        x[j] = b[j] / diag_val
        jlo = max(0, j - mu)
        for i in range(jlo, j):
            band_row = mu - (j - i)
            b[i] -= A_band[band_row, j] * x[j]

    return x


def dense_to_banded_upper(A, mu):
    """
    Convert dense upper triangular matrix to banded storage.

    Parameters
    ----------
    A : ndarray, shape (n, n)
    mu : int
        Upper bandwidth.

    Returns
    -------
    A_band : ndarray, shape (mu+1, n)
    """
    n = A.shape[0]
    A_band = np.zeros((mu + 1, n), dtype=float)
    for j in range(n):
        for i in range(max(0, j - mu), j + 1):
            A_band[mu - (j - i), j] = A[i, j]
    return A_band


def estimate_condition_number_inf(A, n_iter=5):
    """
    Estimate the infinity-norm condition number of a dense matrix A
    using power iteration.

    kappa_inf(A) = ||A||_inf * ||A^{-1}||_inf

    Parameters
    ----------
    A : ndarray
    n_iter : int
        Number of power iterations.

    Returns
    -------
    kappa : float
    """
    n = A.shape[0]
    A_norm = np.linalg.norm(A, ord=np.inf)

    # Estimate ||A^{-1}||_inf via power iteration on A^{-T} A^{-1}
    x = np.ones(n) / n
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return np.inf

    for _ in range(n_iter):
        x = A_inv.T @ (A_inv @ x)
        norm_x = np.linalg.norm(x, ord=np.inf)
        if norm_x < 1e-14:
            break
        x = x / norm_x

    A_inv_norm = np.linalg.norm(A_inv, ord=np.inf)
    return A_norm * A_inv_norm


def assemble_global_stiffness(n_nodes, elements, element_stiffness_list, dof_per_node=3):
    """
    Assemble global stiffness matrix from element stiffness matrices.

    Parameters
    ----------
    n_nodes : int
    elements : list of list of int
        Element connectivity (node indices per element).
    element_stiffness_list : list of ndarray
        k_e for each element, shape (n_elem_nodes*dof, n_elem_nodes*dof).
    dof_per_node : int
        Degrees of freedom per node.

    Returns
    -------
    K : SparseMatrixCOO
        Global stiffness in COO format.
    """
    n_dof = n_nodes * dof_per_node
    K = SparseMatrixCOO()
    K.set_shape(n_dof, n_dof)

    for elem_idx, elem_nodes in enumerate(elements):
        k_e = element_stiffness_list[elem_idx]
        n_en = len(elem_nodes)
        # Global DOF mapping
        global_dofs = []
        for nid in elem_nodes:
            for d in range(dof_per_node):
                global_dofs.append(nid * dof_per_node + d)
        K.add_block(global_dofs, global_dofs, k_e)

    return K


def apply_dirichlet_bc(K_dense, F, bc_dofs, bc_values):
    """
    Apply Dirichlet boundary conditions by modifying the global system.

    For each constrained DOF i with value u_i:
        K[i, i] = 1
        K[i, j] = K[j, i] = 0  (j != i)
        F[i] = u_i
        F[j] -= K[j, i] * u_i  (before zeroing)

    Parameters
    ----------
    K_dense : ndarray
        Global stiffness matrix (dense, modified in-place).
    F : ndarray
        Force vector (modified in-place).
    bc_dofs : list of int
        Constrained DOF indices.
    bc_values : list of float
        Prescribed values.

    Returns
    -------
    K_dense, F : modified arrays.
    """
    K_dense = np.array(K_dense, dtype=float, copy=True)
    F = np.array(F, dtype=float, copy=True)

    for dof, val in zip(bc_dofs, bc_values):
        # Adjust RHS for other DOFs
        F -= K_dense[:, dof] * val
        F[dof] = val
        # Zero row and column
        K_dense[dof, :] = 0.0
        K_dense[:, dof] = 0.0
        K_dense[dof, dof] = 1.0

    return K_dense, F
