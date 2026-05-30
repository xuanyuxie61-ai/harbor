
import numpy as np
from typing import Tuple, Optional, List


class RREFSolver:

    @staticmethod
    def rref_compute(A: np.ndarray,
                     tol: float = 1e-12) -> Tuple[np.ndarray, List[int]]:
        A = np.array(A, dtype=float, copy=True)
        m, n = A.shape
        pivot_cols = []
        r = 0

        for c in range(n):
            if r >= m:
                break

            pivot_val = abs(A[r, c])
            pivot_row = r
            for i in range(r + 1, m):
                if abs(A[i, c]) > pivot_val:
                    pivot_val = abs(A[i, c])
                    pivot_row = i

            if pivot_val < tol:
                continue


            if pivot_row != r:
                A[[r, pivot_row]] = A[[pivot_row, r]]


            A[r] = A[r] / A[r, c]


            for i in range(m):
                if i != r and abs(A[i, c]) > tol:
                    A[i] = A[i] - A[i, c] * A[r]

            pivot_cols.append(c)
            r += 1


        A[np.abs(A) < tol] = 0.0
        return A, pivot_cols

    @staticmethod
    def solve(A: np.ndarray, b: np.ndarray,
              tol: float = 1e-12) -> np.ndarray:
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float)
        m, n = A.shape

        if b.ndim == 1:
            b = b.reshape(-1, 1)

        if b.shape[0] != m:
            raise ValueError("b的行数必须与A的行数相同")


        Ab = np.hstack([A, b])
        R, pivot_cols = RREFSolver.rref_compute(Ab, tol)

        n_rhs = b.shape[1]
        x = np.zeros((n, n_rhs))

        for i, col in enumerate(pivot_cols):
            if col < n:
                x[col] = R[i, n:]

        return x.squeeze() if n_rhs == 1 else x

    @staticmethod
    def rank(A: np.ndarray, tol: float = 1e-12) -> int:
        R, pivot_cols = RREFSolver.rref_compute(A, tol)
        return len(pivot_cols)

    @staticmethod
    def determinant(A: np.ndarray, tol: float = 1e-12) -> float:
        A = np.array(A, dtype=float, copy=True)
        n = A.shape[0]
        if A.shape[0] != A.shape[1]:
            raise ValueError("必须是方阵")

        det_sign = 1
        for c in range(n):

            pivot_val = abs(A[c, c])
            pivot_row = c
            for i in range(c + 1, n):
                if abs(A[i, c]) > pivot_val:
                    pivot_val = abs(A[i, c])
                    pivot_row = i

            if pivot_val < tol:
                return 0.0

            if pivot_row != c:
                A[[c, pivot_row]] = A[[pivot_row, c]]
                det_sign *= -1

            pivot = A[c, c]

            for i in range(c + 1, n):
                factor = A[i, c] / pivot
                A[i, c:] -= factor * A[c, c:]

        det = det_sign
        for i in range(n):
            det *= A[i, i]
        return det

    @staticmethod
    def inverse(A: np.ndarray, tol: float = 1e-12) -> np.ndarray:
        n = A.shape[0]
        if A.shape[0] != A.shape[1]:
            raise ValueError("必须是方阵")
        I = np.eye(n)
        return RREFSolver.solve(A, I, tol)


class HammingErrorDetection:


    G = np.array([
        [1, 1, 0, 1],
        [1, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 1, 1, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=int)


    H = np.array([
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1]
    ], dtype=int)

    @staticmethod
    def encode(data: np.ndarray) -> np.ndarray:
        data = np.asarray(data).astype(int) % 2
        if len(data) != 4:
            raise ValueError("数据长度必须是4")
        codeword = (HammingErrorDetection.G @ data) % 2
        return codeword

    @staticmethod
    def decode(codeword: np.ndarray) -> Tuple[np.ndarray, bool]:
        codeword = np.asarray(codeword).astype(int) % 2
        if len(codeword) != 7:
            raise ValueError("码字长度必须是7")

        syndrome = (HammingErrorDetection.H @ codeword) % 2
        syndrome_int = syndrome[0] + 2 * syndrome[1] + 4 * syndrome[2]

        corrected = False
        if syndrome_int != 0:

            error_pos = syndrome_int - 1
            if 0 <= error_pos < 7:
                codeword[error_pos] = 1 - codeword[error_pos]
                corrected = True


        data = codeword[[2, 4, 5, 6]]
        return data, corrected

    @staticmethod
    def check_linear_system(A: np.ndarray, x: np.ndarray,
                            b: np.ndarray, tol: float = 1e-10) -> bool:
        residual = A @ x - b
        norm = np.linalg.norm(residual)
        return norm < tol

    @staticmethod
    def redundant_solve(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, float]:

        A_ext = np.vstack([A, np.sum(A, axis=0)])
        b_ext = np.append(b, np.sum(b))


        x, residuals, rank, s = np.linalg.lstsq(A_ext, b_ext, rcond=None)
        residual_norm = float(np.linalg.norm(A @ x - b))
        return x, residual_norm


class LatticeDiracSolver:

    def __init__(self, mass: float = 0.1, lattice_size: int = 8):
        self.mass = mass
        self.N = lattice_size

    def wilson_dirac_matrix(self, gauge_field: Optional[np.ndarray] = None) -> np.ndarray:
        N = self.N
        vol = N ** 4

        vol2 = N ** 2
        D = np.zeros((vol2, vol2))
        for i in range(N):
            for j in range(N):
                idx = i * N + j
                D[idx, idx] = 4.0 + self.mass

                neighbors = [
                    ((i + 1) % N, j),
                    ((i - 1 + N) % N, j),
                    (i, (j + 1) % N),
                    (i, (j - 1 + N) % N)
                ]
                for ni, nj in neighbors:
                    nidx = ni * N + nj
                    D[idx, nidx] -= 0.5
        return D

    def solve(self, eta: np.ndarray,
              gauge_field: Optional[np.ndarray] = None) -> np.ndarray:
        D = self.wilson_dirac_matrix(gauge_field)

        solver = RREFSolver()
        psi = solver.solve(D, eta)
        return psi
