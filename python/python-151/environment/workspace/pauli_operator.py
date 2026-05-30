
import numpy as np
from typing import List, Tuple, Optional, Dict
from functools import reduce



PAULI_MATRICES = {
    'I': np.array([[1, 0], [0, 1]], dtype=complex),
    'X': np.array([[0, 1], [1, 0]], dtype=complex),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
    'Z': np.array([[1, 0], [0, -1]], dtype=complex),
}


def pauli_commutator(p1: str, p2: str) -> Tuple[complex, str]:
    if p1 == 'I' or p2 == 'I':
        return 0.0, 'I'
    if p1 == p2:
        return 0.0, 'I'
    table = {
        ('X', 'Y'): (2j, 'Z'),
        ('Y', 'X'): (-2j, 'Z'),
        ('Y', 'Z'): (2j, 'X'),
        ('Z', 'Y'): (-2j, 'X'),
        ('Z', 'X'): (2j, 'Y'),
        ('X', 'Z'): (-2j, 'Y'),
    }
    return table.get((p1, p2), (0.0, 'I'))


class PauliString:
    def __init__(self, string: str, coefficient: complex = 1.0):
        self.string = string.upper()
        self.n_qubits = len(string)
        self.coefficient = complex(coefficient)
        for c in self.string:
            if c not in 'IXYZ':
                raise ValueError(f"非法Pauli字符: {c}")

    def __repr__(self):
        return f"{self.coefficient:.4g} * {self.string}"

    def __eq__(self, other):
        if not isinstance(other, PauliString):
            return False
        return self.string == other.string and np.isclose(self.coefficient, other.coefficient)

    def __hash__(self):
        return hash((self.string, round(self.coefficient.real, 12), round(self.coefficient.imag, 12)))

    def to_matrix(self) -> np.ndarray:
        mats = [PAULI_MATRICES[c] for c in self.string]
        return self.coefficient * reduce(np.kron, mats)

    def multiply(self, other: 'PauliString') -> 'PauliString':
        if self.n_qubits != other.n_qubits:
            raise ValueError("Pauli字符串量子比特数不匹配")
        new_string = []
        phase = 1.0 + 0.0j
        for a, b in zip(self.string, other.string):
            if a == 'I':
                new_string.append(b)
            elif b == 'I':
                new_string.append(a)
            elif a == b:
                new_string.append('I')
            else:

                comm, c = pauli_commutator(a, b)
                if c != 'I':
                    phase *= comm / (2j)
                    new_string.append(c)
        return PauliString(''.join(new_string), self.coefficient * other.coefficient * phase)

    def commutator(self, other: 'PauliString') -> 'PauliString':
        p1 = self.multiply(other)
        p2 = other.multiply(self)
        return PauliString(p1.string, p1.coefficient - (p2.coefficient if p1.string == p2.string else 0.0))

    def weight(self) -> int:
        return sum(1 for c in self.string if c != 'I')

    def support(self) -> List[int]:
        return [i for i, c in enumerate(self.string) if c != 'I']


def rref_compute(A: np.ndarray, tol: Optional[float] = None) -> Tuple[np.ndarray, List[int]]:
    A = A.astype(float).copy()
    rows, cols = A.shape
    if tol is None:
        tol = np.sqrt(np.finfo(float).eps)
    pivot_cols = []
    row = 0
    for col in range(cols):
        if row >= rows:
            break
        pivot_value = np.max(np.abs(A[row:rows, col]))
        if pivot_value <= tol:
            continue
        pivot_row = row + np.argmax(np.abs(A[row:rows, col]))
        pivot_cols.append(col)

        A[[row, pivot_row], col:cols] = A[[pivot_row, row], col:cols]

        A[row, col:cols] /= A[row, col]

        for i in range(rows):
            if i != row and abs(A[i, col]) > tol:
                A[i, col:cols] -= A[i, col] * A[row, col:cols]
        row += 1
    return A, pivot_cols


def rref_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    m1, n1 = A.shape
    if b.ndim == 1:
        b = b.reshape(-1, 1)
    m2, n2 = b.shape
    if m1 != m2:
        raise ValueError("A和b的行数不匹配")
    AI = np.hstack([A, b])
    AIRREF, pivot_cols = rref_compute(AI)
    x = AIRREF[:n1, n1:n1 + n2]
    return x.reshape(-1) if n2 == 1 else x


def extract_independent_paulis(paulis: List[PauliString]) -> List[PauliString]:
    if not paulis:
        return []
    mats = [p.to_matrix().reshape(-1).real for p in paulis]
    A = np.vstack(mats)

    _, pivot_rows = rref_compute(A.T)
    rank = min(len(pivot_rows), len(paulis))
    return [paulis[i] for i in pivot_rows[:rank]]


class ShermanMorrisonSolver:
    def __init__(self, A: np.ndarray):
        self.A = np.array(A, dtype=float)
        self.n = A.shape[0]
        if A.shape[0] != A.shape[1]:
            raise ValueError("矩阵必须是方阵")
        try:
            self.A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:

            self.A_inv = np.linalg.pinv(A)

    def update_inverse(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float).reshape(-1)
        v = np.asarray(v, dtype=float).reshape(-1)
        if u.shape[0] != self.n or v.shape[0] != self.n:
            raise ValueError("向量维度不匹配")


        vT_Ainv_u = float(v @ self.A_inv @ u)
        alpha_den = 1.0 - vT_Ainv_u
        if abs(alpha_den) < 1e-14:
            raise ValueError("Sherman-Morrison除数为零，更新不可逆")
        alpha = 1.0 / alpha_den


        Ainv_u = self.A_inv @ u
        vT_Ainv = v @ self.A_inv
        B_inv = self.A_inv + alpha * np.outer(Ainv_u, vT_Ainv)
        return B_inv

    def solve(self, u: np.ndarray, v: np.ndarray, b: np.ndarray) -> np.ndarray:
        b = np.asarray(b, dtype=float).reshape(-1)
        u = np.asarray(u, dtype=float).reshape(-1)
        v = np.asarray(v, dtype=float).reshape(-1)
        z = self.A_inv @ b
        w = self.A_inv @ u
        vT_w = float(v @ w)
        alpha_den = 1.0 - vT_w
        if abs(alpha_den) < 1e-14:
            raise ValueError("Sherman-Morrison除数为零")
        alpha = 1.0 / alpha_den
        beta = float(v @ z)
        x = z + alpha * beta * w
        return x

    def mv(self, u: np.ndarray, v: np.ndarray, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        u = np.asarray(u, dtype=float).reshape(-1)
        v = np.asarray(v, dtype=float).reshape(-1)
        return self.A @ x - u * (v @ x)


def build_pauli_hamiltonian(n_qubits: int, coefficients: Dict[str, complex]) -> np.ndarray:

    raise NotImplementedError("Hole 2: 请实现Pauli字符串到哈密顿量矩阵的构建")

