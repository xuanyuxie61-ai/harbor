# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix
from scipy.sparse.linalg import spsolve, norm as spnorm
from typing import Optional, Tuple


class StiffnessMatrixAssembler1D:

    def __init__(self, nodes: np.ndarray, A: float = 1.0, E0: float = 100e9):
        self.nodes = np.asarray(nodes)
        self.n_nodes = len(nodes)
        self.A = A
        self.E0 = E0
        self.n_elements = self.n_nodes - 1
        self.elem_sizes = np.diff(nodes)
        if np.any(self.elem_sizes <= 0):
            raise ValueError("Node coordinates must be strictly increasing.")

    def assemble_global_stiffness(self, damage: Optional[np.ndarray] = None,
                                   E_override: Optional[np.ndarray] = None) -> csr_matrix:
        ne = self.n_elements
        nn = self.n_nodes

        if E_override is not None:
            E_elem = np.asarray(E_override)
        else:
            E_elem = np.full(ne, self.E0)
            if damage is not None:
                damage = np.clip(np.asarray(damage), 0.0, 1.0)
                g_d = (1.0 - damage) ** 2 + 1e-6
                E_elem *= g_d

        rows = []
        cols = []
        data = []

        for e in range(ne):
            h_e = self.elem_sizes[e]
            k = E_elem[e] * self.A / h_e

            local_dofs = [e, e + 1]
            local_k = k * np.array([[1.0, -1.0], [-1.0, 1.0]])
            for i in range(2):
                for j in range(2):
                    rows.append(local_dofs[i])
                    cols.append(local_dofs[j])
                    data.append(local_k[i, j])

        K_coo = coo_matrix((data, (rows, cols)), shape=(nn, nn))
        return K_coo.tocsr()

    def apply_boundary_conditions(self, K: csr_matrix, F: np.ndarray,
                                   fixed_dofs: np.ndarray,
                                   penalty: float = 1e12) -> Tuple[csr_matrix, np.ndarray]:
        K = K.copy()
        F = F.copy()
        for dof in fixed_dofs:
            K[dof, dof] += penalty
            F[dof] += penalty * 0.0
        return K, F


class SparseMatrixConverter:

    @staticmethod
    def to_coo(K: csr_matrix) -> coo_matrix:
        return K.tocoo()

    @staticmethod
    def to_csc(K: csr_matrix) -> csc_matrix:
        return K.tocsc()

    @staticmethod
    def to_matrix_market_string(K: coo_matrix) -> str:
        nnz = K.nnz
        lines = ["%%MatrixMarket matrix coordinate real general",
                 f"{K.shape[0]} {K.shape[1]} {nnz}"]
        for i, j, v in zip(K.row, K.col, K.data):
            lines.append(f"{i+1} {j+1} {v:.16e}")
        return "\n".join(lines)

    @staticmethod
    def from_matrix_market_string(mm_string: str) -> coo_matrix:
        lines = mm_string.strip().splitlines()
        data_lines = [l for l in lines if not l.startswith("%")]
        header = data_lines[0].split()
        n_rows, n_cols, nnz = int(header[0]), int(header[1]), int(header[2])
        rows = []
        cols = []
        data = []
        for line in data_lines[1:]:
            parts = line.split()
            rows.append(int(parts[0]) - 1)
            cols.append(int(parts[1]) - 1)
            data.append(float(parts[2]))
        return coo_matrix((data, (rows, cols)), shape=(n_rows, n_cols))

    @staticmethod
    def bandwidth(K: csr_matrix) -> Tuple[int, int]:
        K_coo = K.tocoo()
        lower = 0
        upper = 0
        for i, j in zip(K_coo.row, K_coo.col):
            if i > j:
                lower = max(lower, i - j)
            elif j > i:
                upper = max(upper, j - i)
        return lower, upper


class BandedUpperTriangularSolver:

    def __init__(self, n: int, mu: int):
        self.n = n
        self.mu = mu

        self.data = np.zeros((mu + 1, n))

    def from_dense(self, U: np.ndarray):
        U = np.asarray(U)
        if U.shape != (self.n, self.n):
            raise ValueError("Dense matrix shape mismatch.")
        for j in range(self.n):
            for i in range(max(0, j - self.mu), j + 1):
                self.data[self.mu + i - j, j] = U[i, j]

    def solve(self, b: np.ndarray) -> np.ndarray:
        b = np.asarray(b, dtype=float).copy()
        x = np.zeros(self.n)
        for j in range(self.n - 1, -1, -1):
            diag = self.data[self.mu, j]
            if abs(diag) < 1e-30:
                diag = 1e-30 * np.sign(diag) if diag != 0 else 1e-30

            end_idx = min(j + 1 + self.mu, self.n)
            band_len = end_idx - (j + 1)
            if band_len > 0:
                x[j] = (b[j] - np.dot(self.data[self.mu - band_len:self.mu, j], x[j + 1:end_idx])) / diag
            else:
                x[j] = b[j] / diag
        return x

    def determinant(self) -> float:
        return np.prod(self.data[self.mu, :])

    def mv(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        y = np.zeros(self.n)
        for j in range(self.n):
            start = max(0, j - self.mu)
            for i in range(start, j + 1):
                y[i] += self.data[self.mu + i - j, j] * x[j]
        return y


class LinearSolverAnalysis:

    @staticmethod
    def solve_and_analyze(K: csr_matrix, F: np.ndarray) -> dict:
        u = spsolve(K, F)


        residual = F - K @ u
        residual_norm = np.linalg.norm(residual)
        rhs_norm = np.linalg.norm(F)
        sol_norm = np.linalg.norm(u)
        matrix_norm = spnorm(K, ord=2)


        eps_machine = np.finfo(float).eps
        relative_residual = residual_norm / (matrix_norm * sol_norm + 1e-30)
        normalized_residual = relative_residual / eps_machine



        cond_est = LinearSolverAnalysis._estimate_condition_number(K)

        return {
            "solution": u,
            "residual_norm": residual_norm,
            "relative_residual": relative_residual,
            "normalized_residual": normalized_residual,
            "condition_number_estimate": cond_est,
            "rhs_norm": rhs_norm,
            "solution_norm": sol_norm,
        }

    @staticmethod
    def _estimate_condition_number(K: csr_matrix, num_iter: int = 5) -> float:
        n = K.shape[0]

        x = np.random.randn(n)
        x = x / np.linalg.norm(x)
        for _ in range(num_iter):
            x = K @ x
            x = x / np.linalg.norm(x)
        sigma_max = np.linalg.norm(K @ x)


        try:
            y = np.random.randn(n)
            y = y / np.linalg.norm(y)
            for _ in range(num_iter):
                y = spsolve(K, y)
                y = y / np.linalg.norm(y)
            sigma_min = 1.0 / np.linalg.norm(spsolve(K, y))
        except Exception:
            sigma_min = 1e-16

        return sigma_max / (sigma_min + 1e-30)


if __name__ == "__main__":

    nodes = np.linspace(0.0, 1.0, 21)
    assembler = StiffnessMatrixAssembler1D(nodes, A=1e-4, E0=100e9)


    K0 = assembler.assemble_global_stiffness()
    F = np.zeros(assembler.n_nodes)
    F[-1] = 1000.0

    K_bc, F_bc = assembler.apply_boundary_conditions(K0, F, fixed_dofs=np.array([0]))
    result = LinearSolverAnalysis.solve_and_analyze(K_bc, F_bc)
    print("Displacement at free end:", result["solution"][-1])
    print("Normalized residual:", result["normalized_residual"])
    print("Condition number estimate:", result["condition_number_estimate"])


    damage = np.linspace(0.0, 0.5, assembler.n_elements)
    K_damaged = assembler.assemble_global_stiffness(damage=damage)
    Kd_bc, Fd_bc = assembler.apply_boundary_conditions(K_damaged, F, fixed_dofs=np.array([0]))
    result_d = LinearSolverAnalysis.solve_and_analyze(Kd_bc, Fd_bc)
    print("Damaged displacement at free end:", result_d["solution"][-1])


    U = np.triu(np.random.rand(10, 10)) + np.eye(10)
    band_solver = BandedUpperTriangularSolver(n=10, mu=9)
    band_solver.from_dense(U)
    b = np.random.rand(10)
    x_band = band_solver.solve(b)
    x_dense = np.linalg.solve(U, b)
    print("Banded solver error:", np.max(np.abs(x_band - x_dense)))
