
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from utils import validate_positive, compute_condition_number, compute_normalized_residual


class LaminateStiffness:

    def __init__(self, plies, thicknesses, material):
        if len(plies) != len(thicknesses):
            raise ValueError("plies and thicknesses must have same length.")
        self.n_plies = len(plies)
        self.plies = [float(p) for p in plies]
        self.thicknesses = [float(t) for t in thicknesses]
        self.material = material
        self._compute_abd()

    def _compute_abd(self):
        self.A = np.zeros((3, 3))
        self.B = np.zeros((3, 3))
        self.D = np.zeros((3, 3))


        z = [-sum(self.thicknesses) / 2.0]
        for t in self.thicknesses:
            z.append(z[-1] + t)
        self.z_coords = np.array(z)

        for k in range(self.n_plies):
            theta = self.plies[k]
            Q_bar = self.material.compute_transformed_stiffness(theta)
            h_k = self.thicknesses[k]
            z_k = z[k + 1]
            z_k1 = z[k]

            self.A += Q_bar * (z_k - z_k1)
            self.B += 0.5 * Q_bar * (z_k ** 2 - z_k1 ** 2)
            self.D += (1.0 / 3.0) * Q_bar * (z_k ** 3 - z_k1 ** 3)


        self.ABD = np.block([
            [self.A, self.B],
            [self.B, self.D]
        ])

    def compute_degraded_abd(self, damage_by_ply):
        A_deg = np.zeros((3, 3))
        B_deg = np.zeros((3, 3))
        D_deg = np.zeros((3, 3))
        z = self.z_coords














        raise NotImplementedError("Hole 2: compute_degraded_abd core loop needs implementation.")

        return A_deg, B_deg, D_deg

    def get_total_thickness(self):
        return sum(self.thicknesses)


class SparseStiffnessAssembler:

    def __init__(self, n_nodes, ndof_per_node=2):
        self.n_nodes = n_nodes
        self.ndof = ndof_per_node
        self.n_dof_total = n_nodes * ndof_per_node
        self.K_data = []
        self.K_row = []
        self.K_col = []

    def add_element_stiffness(self, element_nodes, k_e):
        nen = len(element_nodes)
        dof_map = []
        for node in element_nodes:
            for d in range(self.ndof):
                dof_map.append(node * self.ndof + d)

        for i_local, i_global in enumerate(dof_map):
            for j_local, j_global in enumerate(dof_map):
                val = k_e[i_local, j_local]
                if abs(val) > 1e-16:
                    self.K_data.append(val)
                    self.K_row.append(i_global)
                    self.K_col.append(j_global)

    def get_csr_matrix(self):
        K = csr_matrix((self.K_data, (self.K_row, self.K_col)),
                       shape=(self.n_dof_total, self.n_dof_total))
        return K

    def get_csc_matrix(self):
        K = csc_matrix((self.K_data, (self.K_row, self.K_col)),
                       shape=(self.n_dof_total, self.n_dof_total))
        return K


class BandedUpperTriangularSolver:

    def __init__(self, n, mu):
        validate_positive(n, "n")
        validate_positive(mu, "mu")
        self.n = int(n)
        self.mu = int(mu)

    def solve(self, A_band, b):
        if A_band.shape != (self.mu + 1, self.n):
            raise ValueError(f"A_band shape must be ({self.mu+1}, {self.n}), got {A_band.shape}")
        b = np.asarray(b, dtype=float)
        if len(b) != self.n:
            raise ValueError("b length mismatch.")

        x = b.copy()
        for j in range(self.n - 1, -1, -1):
            diag_row = j - j + self.mu
            diag_val = A_band[diag_row, j]
            if abs(diag_val) < 1e-15:
                diag_val = 1e-12
            x[j] = x[j] / diag_val
            jlo = max(0, j - self.mu)
            for i in range(jlo, j):
                row_idx = i - j + self.mu
                x[i] -= A_band[row_idx, j] * x[j]
        return x

    def multiply(self, A_band, x_vec):
        x_vec = np.asarray(x_vec, dtype=float)
        b = np.zeros(self.n)
        for i in range(self.n):
            for j in range(i, min(self.n, i + self.mu + 1)):
                row_idx = i - j + self.mu
                b[i] += A_band[row_idx, j] * x_vec[j]
        return b


def solve_equilibrium_dense(K, F):
    K = np.asarray(K, dtype=float)
    F = np.asarray(F, dtype=float)

    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("K must be a square matrix.")
    if len(F) != K.shape[0]:
        raise ValueError("F length must match K dimension.")

    cond_K = compute_condition_number(K)

    try:
        U = np.linalg.solve(K, F)
    except np.linalg.LinAlgError:

        U, _, _, _ = np.linalg.lstsq(K, F, rcond=None)

    r_norm = np.linalg.norm(F - K @ U, ord=np.inf)
    norm_res = compute_normalized_residual(K, U, F)

    return U, r_norm, norm_res, cond_K


def solve_equilibrium_sparse(K_csr, F):
    from scipy.sparse.linalg import spsolve
    F = np.asarray(F, dtype=float)
    U = spsolve(K_csr, F)
    if U is None:
        raise ValueError("Sparse solver failed.")
    U = np.asarray(U)
    r_norm = np.linalg.norm(F - K_csr @ U, ord=np.inf)
    return U, r_norm
