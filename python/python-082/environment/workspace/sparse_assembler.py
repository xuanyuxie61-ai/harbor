
import numpy as np


class SparseMatrixCOO:

    def __init__(self, rows=None, cols=None, vals=None, shape=None):
        self.rows = [] if rows is None else list(rows)
        self.cols = [] if cols is None else list(cols)
        self.vals = [] if vals is None else list(vals)
        self.shape = shape

    def add(self, row, col, val):
        if abs(val) > 1e-18:
            self.rows.append(int(row))
            self.cols.append(int(col))
            self.vals.append(float(val))

    def add_block(self, rows, cols, block):
        block = np.asarray(block, dtype=float)
        for i, ri in enumerate(rows):
            for j, cj in enumerate(cols):
                v = block[i, j]
                if abs(v) > 1e-18:
                    self.rows.append(int(ri))
                    self.cols.append(int(cj))
                    self.vals.append(float(v))

    def to_dense(self):
        if self.shape is None:
            raise ValueError("Shape not set.")
        A = np.zeros(self.shape, dtype=float)
        for r, c, v in zip(self.rows, self.cols, self.vals):
            A[r, c] += v
        return A

    def matvec(self, x):
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
    n = A.shape[0]
    A_band = np.zeros((mu + 1, n), dtype=float)
    for j in range(n):
        for i in range(max(0, j - mu), j + 1):
            A_band[mu - (j - i), j] = A[i, j]
    return A_band


def estimate_condition_number_inf(A, n_iter=5):
    n = A.shape[0]
    A_norm = np.linalg.norm(A, ord=np.inf)


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
    n_dof = n_nodes * dof_per_node
    K = SparseMatrixCOO()
    K.set_shape(n_dof, n_dof)

    for elem_idx, elem_nodes in enumerate(elements):
        k_e = element_stiffness_list[elem_idx]
        n_en = len(elem_nodes)

        global_dofs = []
        for nid in elem_nodes:
            for d in range(dof_per_node):
                global_dofs.append(nid * dof_per_node + d)
        K.add_block(global_dofs, global_dofs, k_e)

    return K


def apply_dirichlet_bc(K_dense, F, bc_dofs, bc_values):
    K_dense = np.array(K_dense, dtype=float, copy=True)
    F = np.array(F, dtype=float, copy=True)

    for dof, val in zip(bc_dofs, bc_values):

        F -= K_dense[:, dof] * val
        F[dof] = val

        K_dense[dof, :] = 0.0
        K_dense[:, dof] = 0.0
        K_dense[dof, dof] = 1.0

    return K_dense, F
