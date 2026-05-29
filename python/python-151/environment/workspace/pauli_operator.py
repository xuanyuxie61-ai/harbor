"""
pauli_operator.py
=================
Pauli算符代数与稀疏矩阵线性系统求解工具

原项目映射:
- 1048_rref2: 行最简形 (RREF) 计算，用于判断Pauli字符串集合的线性无关性
- 995_r8sm: Sherman-Morrison公式，用于低秩更新逆矩阵（在量子态层析中应用）

科学功能:
本模块实现了VQE核心的Pauli算符字符串表示、对易关系计算、
以及用于量子测量后处理的线性代数工具。RREF用于提取Pauli群的
生成元基，Sherman-Morrison公式用于在迭代优化中快速更新估计的
费米子-玻色子耦合矩阵的逆。
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from functools import reduce


# Pauli矩阵
PAULI_MATRICES = {
    'I': np.array([[1, 0], [0, 1]], dtype=complex),
    'X': np.array([[0, 1], [1, 0]], dtype=complex),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
    'Z': np.array([[1, 0], [0, -1]], dtype=complex),
}


def pauli_commutator(p1: str, p2: str) -> Tuple[complex, str]:
    """
    计算两个单量子比特Pauli算符的对易子 [P1, P2] = c * P3。
    满足关系: [sigma_a, sigma_b] = 2i * epsilon_{abc} * sigma_c
    """
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
    """
    多量子比特Pauli字符串，例如 'XIZY' 表示 X_0 \otimes I_1 \otimes Z_2 \otimes Y_3。
    """
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
        """将Pauli字符串转换为 2^n x 2^n 的复矩阵。"""
        mats = [PAULI_MATRICES[c] for c in self.string]
        return self.coefficient * reduce(np.kron, mats)

    def multiply(self, other: 'PauliString') -> 'PauliString':
        """
        计算两个Pauli字符串的乘积: (c1 P1) * (c2 P2) = c1*c2 * phase * P3。
        使用关系: sigma_a * sigma_b = delta_{ab} I + i * epsilon_{abc} sigma_c
        """
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
                # sigma_a * sigma_b = i * epsilon_{abc} * sigma_c
                comm, c = pauli_commutator(a, b)
                if c != 'I':
                    phase *= comm / (2j)
                    new_string.append(c)
        return PauliString(''.join(new_string), self.coefficient * other.coefficient * phase)

    def commutator(self, other: 'PauliString') -> 'PauliString':
        """对易子 [self, other] = self*other - other*self。"""
        p1 = self.multiply(other)
        p2 = other.multiply(self)
        return PauliString(p1.string, p1.coefficient - (p2.coefficient if p1.string == p2.string else 0.0))

    def weight(self) -> int:
        """非恒等Pauli算符的个数（Hamming权重）。"""
        return sum(1 for c in self.string if c != 'I')

    def support(self) -> List[int]:
        """返回非恒等算符作用的量子比特索引列表。"""
        return [i for i, c in enumerate(self.string) if c != 'I']


def rref_compute(A: np.ndarray, tol: Optional[float] = None) -> Tuple[np.ndarray, List[int]]:
    """
    计算矩阵的行最简形 (Reduced Row Echelon Form)，对应 1048_rref2/rref_compute。

    算法流程:
    1. 对每一列，在当前行以下的子向量中寻找主元（绝对值最大）。
    2. 若主元大于容差，则交换行、归一化、消去该列其他元素。
    3. 记录主元列索引。

    RREF在VQE中的应用: 从过完备的Pauli测量集合中提取线性无关的
    生成元，减少量子电路执行次数。

    参数:
        A: MxN 实矩阵
        tol: 主元容差，默认 sqrt(eps)
    返回:
        A_rref: 行最简形矩阵
        pivot_cols: 主元列索引列表
    """
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
        # 交换行
        A[[row, pivot_row], col:cols] = A[[pivot_row, row], col:cols]
        # 归一化
        A[row, col:cols] /= A[row, col]
        # 消去
        for i in range(rows):
            if i != row and abs(A[i, col]) > tol:
                A[i, col:cols] -= A[i, col] * A[row, col:cols]
        row += 1
    return A, pivot_cols


def rref_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    使用RREF求解线性系统 A x = b，对应 1048_rref2/rref_solve。
    假设方程组是一致的（相容的）。
    """
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
    """
    从Pauli字符串列表中提取线性无关的子集（基于矩阵表示的RREF）。
    这是减少VQE测量次数的关键步骤。
    """
    if not paulis:
        return []
    mats = [p.to_matrix().reshape(-1).real for p in paulis]
    A = np.vstack(mats)
    # 对 A.T 做RREF，pivot_cols 对应 A 中线性无关行的索引
    _, pivot_rows = rref_compute(A.T)
    rank = min(len(pivot_rows), len(paulis))
    return [paulis[i] for i in pivot_rows[:rank]]


class ShermanMorrisonSolver:
    """
    Sherman-Morrison公式实现，用于低秩更新逆矩阵，基于 995_r8sm。

    若 B = A - u v^T，且已知 A^{-1}，则:
        B^{-1} = A^{-1} + (A^{-1} u v^T A^{-1}) / (1 - v^T A^{-1} u)

    在VQE中应用于: 迭代更新有效单粒子哈密顿量的逆，或
    在量子态层析中更新密度矩阵估计。
    """
    def __init__(self, A: np.ndarray):
        self.A = np.array(A, dtype=float)
        self.n = A.shape[0]
        if A.shape[0] != A.shape[1]:
            raise ValueError("矩阵必须是方阵")
        try:
            self.A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            # 对接近奇异的矩阵使用伪逆
            self.A_inv = np.linalg.pinv(A)

    def update_inverse(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        计算 (A - u v^T)^{-1}，利用Sherman-Morrison公式。
        对应 r8sm_sl 的思想。
        """
        u = np.asarray(u, dtype=float).reshape(-1)
        v = np.asarray(v, dtype=float).reshape(-1)
        if u.shape[0] != self.n or v.shape[0] != self.n:
            raise ValueError("向量维度不匹配")

        # 计算 alpha = 1 / (1 - v^T A^{-1} u)
        vT_Ainv_u = float(v @ self.A_inv @ u)
        alpha_den = 1.0 - vT_Ainv_u
        if abs(alpha_den) < 1e-14:
            raise ValueError("Sherman-Morrison除数为零，更新不可逆")
        alpha = 1.0 / alpha_den

        # B^{-1} = A^{-1} + alpha * A^{-1} u v^T A^{-1}
        Ainv_u = self.A_inv @ u
        vT_Ainv = v @ self.A_inv
        B_inv = self.A_inv + alpha * np.outer(Ainv_u, vT_Ainv)
        return B_inv

    def solve(self, u: np.ndarray, v: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        直接求解 (A - u v^T) x = b，不显式构造逆矩阵。
        步骤:
            1. 解 A z = b  得 z
            2. 解 A w = u  得 w
            3. alpha = 1 / (1 - v^T w)
            4. beta = v^T z
            5. x = z + alpha * beta * w
        """
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
        """
        矩阵-向量乘法 y = (A - u v^T) x，对应 r8sm_mv。
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        u = np.asarray(u, dtype=float).reshape(-1)
        v = np.asarray(v, dtype=float).reshape(-1)
        return self.A @ x - u * (v @ x)


def build_pauli_hamiltonian(n_qubits: int, coefficients: Dict[str, complex]) -> np.ndarray:
    """
    从Pauli字符串字典构建哈密顿量矩阵:
        H = sum_{P} c_P * P
    """
    # TODO: 从Pauli字符串字典构建哈密顿量稠密矩阵
    raise NotImplementedError("Hole 2: 请实现Pauli字符串到哈密顿量矩阵的构建")

