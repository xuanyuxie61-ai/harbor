"""
linear_solver.py
格点QCD线性系统求解与数值误差检测

基于种子项目:
- 1048_rref2: 行简化阶梯形(RREF)、线性方程组求解
- 499_hamming: Hamming纠错码思想 → 数值计算中的冗余校验

物理应用:
1. 格点QCD中Dirac矩阵方程 D ψ = η 的求解
2. 有限元法中的大规模线性系统
3. 流体力学稳定性分析的矩阵特征值问题
4. 数值误差检测与校正

数学模型:
- 线性系统 Ax = b
- RREF: 将A化为 [I | x] 形式
- Hamming校验: 在计算中引入冗余方程检测数值误差
"""

import numpy as np
from typing import Tuple, Optional, List


class RREFSolver:
    """
    基于行简化阶梯形(RREF)的线性系统求解器。
    """

    @staticmethod
    def rref_compute(A: np.ndarray,
                     tol: float = 1e-12) -> Tuple[np.ndarray, List[int]]:
        """
        计算矩阵A的行简化阶梯形(RREF)。

        算法步骤:
        1. 找到主元列
        2. 将主元行归一化
        3. 消去主元列的其他元素

        Parameters
        ----------
        A : np.ndarray
            输入矩阵 (m, n)
        tol : float
            零元素容差

        Returns
        -------
        R : np.ndarray
            RREF矩阵
        pivot_cols : List[int]
            主元列索引
        """
        A = np.array(A, dtype=float, copy=True)
        m, n = A.shape
        pivot_cols = []
        r = 0

        for c in range(n):
            if r >= m:
                break
            # 找主元
            pivot_val = abs(A[r, c])
            pivot_row = r
            for i in range(r + 1, m):
                if abs(A[i, c]) > pivot_val:
                    pivot_val = abs(A[i, c])
                    pivot_row = i

            if pivot_val < tol:
                continue

            # 交换行
            if pivot_row != r:
                A[[r, pivot_row]] = A[[pivot_row, r]]

            # 归一化
            A[r] = A[r] / A[r, c]

            # 消去其他行
            for i in range(m):
                if i != r and abs(A[i, c]) > tol:
                    A[i] = A[i] - A[i, c] * A[r]

            pivot_cols.append(c)
            r += 1

        # 清理小数值
        A[np.abs(A) < tol] = 0.0
        return A, pivot_cols

    @staticmethod
    def solve(A: np.ndarray, b: np.ndarray,
              tol: float = 1e-12) -> np.ndarray:
        """
        使用RREF求解线性系统 Ax = b。

        Parameters
        ----------
        A : np.ndarray
            系数矩阵 (m, n)
        b : np.ndarray
            右端项 (m,) 或 (m, k)
        tol : float
            容差

        Returns
        -------
        x : np.ndarray
            解 (n,) 或 (n, k)
        """
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float)
        m, n = A.shape

        if b.ndim == 1:
            b = b.reshape(-1, 1)

        if b.shape[0] != m:
            raise ValueError("b的行数必须与A的行数相同")

        # 构造增广矩阵
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
        """
        计算矩阵的秩。

        Parameters
        ----------
        A : np.ndarray
            输入矩阵
        tol : float
            容差

        Returns
        -------
        int
            秩
        """
        R, pivot_cols = RREFSolver.rref_compute(A, tol)
        return len(pivot_cols)

    @staticmethod
    def determinant(A: np.ndarray, tol: float = 1e-12) -> float:
        """
        计算方阵的行列式 (通过RREF)。

        det(A) = (-1)^{交换次数} · Π 主元

        Parameters
        ----------
        A : np.ndarray
            方阵
        tol : float
            容差

        Returns
        -------
        float
            行列式
        """
        A = np.array(A, dtype=float, copy=True)
        n = A.shape[0]
        if A.shape[0] != A.shape[1]:
            raise ValueError("必须是方阵")

        det_sign = 1
        for c in range(n):
            # 找主元
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
            # 消去下方元素
            for i in range(c + 1, n):
                factor = A[i, c] / pivot
                A[i, c:] -= factor * A[c, c:]

        det = det_sign
        for i in range(n):
            det *= A[i, i]
        return det

    @staticmethod
    def inverse(A: np.ndarray, tol: float = 1e-12) -> np.ndarray:
        """
        计算矩阵的逆。

        Parameters
        ----------
        A : np.ndarray
            方阵
        tol : float
            容差

        Returns
        -------
        np.ndarray
            逆矩阵
        """
        n = A.shape[0]
        if A.shape[0] != A.shape[1]:
            raise ValueError("必须是方阵")
        I = np.eye(n)
        return RREFSolver.solve(A, I, tol)


class HammingErrorDetection:
    """
    基于Hamming(7,4)码思想的数值误差检测。

    在物理计算中，将4个数值编码为7位码字，
    可以检测并纠正1位错误。
    """

    # Hamming(7,4) 生成矩阵 G (7x4)
    G = np.array([
        [1, 1, 0, 1],
        [1, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 1, 1, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=int)

    # 校验矩阵 H (3x7)
    H = np.array([
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1]
    ], dtype=int)

    @staticmethod
    def encode(data: np.ndarray) -> np.ndarray:
        """
        将4位数据编码为7位Hamming码。

        Parameters
        ----------
        data : np.ndarray
            4位二进制数据

        Returns
        -------
        np.ndarray
            7位码字
        """
        data = np.asarray(data).astype(int) % 2
        if len(data) != 4:
            raise ValueError("数据长度必须是4")
        codeword = (HammingErrorDetection.G @ data) % 2
        return codeword

    @staticmethod
    def decode(codeword: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        解码Hamming码并纠正单比特错误。

        Parameters
        ----------
        codeword : np.ndarray
            7位码字

        Returns
        -------
        data : np.ndarray
            4位数据
        corrected : bool
            是否纠正了错误
        """
        codeword = np.asarray(codeword).astype(int) % 2
        if len(codeword) != 7:
            raise ValueError("码字长度必须是7")

        syndrome = (HammingErrorDetection.H @ codeword) % 2
        syndrome_int = syndrome[0] + 2 * syndrome[1] + 4 * syndrome[2]

        corrected = False
        if syndrome_int != 0:
            # 纠正错误
            error_pos = syndrome_int - 1
            if 0 <= error_pos < 7:
                codeword[error_pos] = 1 - codeword[error_pos]
                corrected = True

        # 提取数据位
        data = codeword[[2, 4, 5, 6]]
        return data, corrected

    @staticmethod
    def check_linear_system(A: np.ndarray, x: np.ndarray,
                            b: np.ndarray, tol: float = 1e-10) -> bool:
        """
        校验线性方程组的解是否正确。

        通过计算残差 ||Ax - b|| 并与容差比较。

        Parameters
        ----------
        A : np.ndarray
            系数矩阵
        x : np.ndarray
            解向量
        b : np.ndarray
            右端项
        tol : float
            容差

        Returns
        -------
        bool
            解是否通过校验
        """
        residual = A @ x - b
        norm = np.linalg.norm(residual)
        return norm < tol

    @staticmethod
    def redundant_solve(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        使用冗余方程求解并检测数值误差。

        将原方程 Ax = b 扩展为包含校验方程的系统。

        Parameters
        ----------
        A : np.ndarray
            系数矩阵
        b : np.ndarray
            右端项

        Returns
        -------
        x : np.ndarray
            解
        residual_norm : float
            残差范数
        """
        # 添加校验行（行的和）
        A_ext = np.vstack([A, np.sum(A, axis=0)])
        b_ext = np.append(b, np.sum(b))

        # 使用最小二乘求解超定系统
        x, residuals, rank, s = np.linalg.lstsq(A_ext, b_ext, rcond=None)
        residual_norm = float(np.linalg.norm(A @ x - b))
        return x, residual_norm


class LatticeDiracSolver:
    """
    格点QCD Dirac方程求解器 (简化模型)。

    方程: (D + m) ψ = η
    其中D为Wilson-Dirac算子。
    """

    def __init__(self, mass: float = 0.1, lattice_size: int = 8):
        """
        初始化格点参数。

        Parameters
        ----------
        mass : float
            夸克质量 [lattice units]
        lattice_size : int
            格点大小
        """
        self.mass = mass
        self.N = lattice_size

    def wilson_dirac_matrix(self, gauge_field: Optional[np.ndarray] = None) -> np.ndarray:
        """
        构造简化Wilson-Dirac矩阵。

        D_w(x,y) = (4 + m) δ_{x,y}
                    - 0.5 Σ_μ [ (1 - γ_μ) U_μ(x) δ_{x+μ,y}
                               + (1 + γ_μ) U_μ^†(x-μ) δ_{x-μ,y} ]

        简化: 使用U_μ = I (自由场)

        Returns
        -------
        np.ndarray
            Dirac矩阵 (N⁴ × N⁴)
        """
        N = self.N
        vol = N ** 4
        # 仅构造一个小规模示例 (使用2D简化)
        vol2 = N ** 2
        D = np.zeros((vol2, vol2))
        for i in range(N):
            for j in range(N):
                idx = i * N + j
                D[idx, idx] = 4.0 + self.mass
                # 最近邻耦合
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
        """
        求解Dirac方程。

        Parameters
        ----------
        eta : np.ndarray
            源项
        gauge_field : np.ndarray, optional
            规范场

        Returns
        -------
        np.ndarray
            解 ψ
        """
        D = self.wilson_dirac_matrix(gauge_field)
        # 使用RREF求解
        solver = RREFSolver()
        psi = solver.solve(D, eta)
        return psi
