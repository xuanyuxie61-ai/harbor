"""
s_matrix.py — S 矩阵计算与幺正性约束 (PROJECT_232)

融合种子项目:
  - 969_r8bb: 带状矩阵运算 (banded matrix factorization & solve)
  - 569_i4mat_rref2: 整数矩阵行简化阶梯形 (IRREF)

核心物理:
  S 矩阵 (散射矩阵) 是量子力学散射理论的核心对象:
    S = 1 + iT
  其中 T 为转移矩阵。

  幺正性: S†S = 1  ⟹  T - T† = iT†T

  分波 S 矩阵 (对角化后):
    S_l = η_l e^{2iδ_l}
  其中 η_l ∈ [0,1] 为弹性参数 (η_l=1 纯弹性)。

  多通道 S 矩阵:
    S_{ij} = δ_{ij} + 2i √(ρ_i) T_{ij} √(ρ_j)
  幺正性: Σ_k S_{ki}* S_{kj} = δ_{ij}

  本模块使用带状矩阵技术存储和求解多通道 S 矩阵，
  并使用整数行简化方法分析幺正性约束的秩。
"""
import numpy as np
from constants import EPS_MACH, TOL_CONVERGE, PI


# ===================================================================
# 带状矩阵 (融合 969_r8bb)
# ===================================================================

class BandedMatrix:
    """
    带状矩阵存储与运算

    对于 n×n 矩阵 A，若 A_{ij} = 0 当 |i-j| > k，
    则 A 为带宽 k 的带状矩阵。

    存储: 使用 (2k+1) × n 的数组，仅存储非零带。
    A[i, j] → storage[k + i - j, j]

    物理背景: 多通道散射的 coupled-channel 方程
    在分波基下产生的 S 矩阵通常是带状的，
    因为通道间耦合通常是短程的。

    Attributes
    ----------
    n : int
        矩阵维度
    kl, ku : int
        下带宽和上带宽
    data : ndarray, shape (kl+ku+1, n)
        带状存储
    """

    def __init__(self, n, kl, ku):
        """
        Parameters
        ----------
        n : int
            矩阵维度
        kl : int
            下带宽
        ku : int
            上带宽
        """
        self.n = n
        self.kl = kl
        self.ku = ku
        self.data = np.zeros((kl + ku + 1, n), dtype=np.complex128)

    def __setitem__(self, key, value):
        i, j = key
        if abs(i - j) > max(self.kl, self.ku):
            if abs(value) > EPS_MACH:
                raise ValueError(
                    f"BandedMatrix: ({i},{j}) outside bandwidth")
            return
        self.data[self.ku + i - j, j] = value

    def __getitem__(self, key):
        i, j = key
        if i < 0 or i >= self.n or j < 0 or j >= self.n:
            return 0.0
        if abs(i - j) > max(self.kl, self.ku):
            return 0.0
        return self.data[self.ku + i - j, j]

    def to_dense(self):
        """转换为密集矩阵"""
        A = np.zeros((self.n, self.n), dtype=np.complex128)
        for j in range(self.n):
            for i in range(max(0, j - self.ku), min(self.n, j + self.kl + 1)):
                A[i, j] = self.data[self.ku + i - j, j]
        return A

    @classmethod
    def from_dense(cls, A, kl=None, ku=None):
        """从密集矩阵创建带状矩阵"""
        n = A.shape[0]
        if kl is None:
            kl = n - 1
        if ku is None:
            ku = n - 1
        bm = cls(n, kl, ku)
        for j in range(n):
            for i in range(max(0, j - ku), min(n, j + kl + 1)):
                bm.data[bm.ku + i - j, j] = A[i, j]
        return bm

    def banded_solve(self, b_vec):
        """
        带状矩阵线性方程组求解 (LU 分解)

        使用带状 LU 分解:
          A = L·U
        其中 L 为单位下三角带状矩阵，U 为上三角带状矩阵。

        算法:
          1. 前向消去 (band-preserving Gaussian elimination)
          2. 回代求解

        复杂度: O(n · kl · ku) 而非 O(n³)

        Parameters
        ----------
        b_vec : ndarray, shape (n,)
            右端向量

        Returns
        -------
        x : ndarray, shape (n,)
            解向量
        """
        n = self.n
        kl, ku = self.kl, self.ku
        # 复制数据
        LU = self.data.copy()
        x = b_vec.copy().astype(np.complex128)

        # LU 分解 (带状)
        for j in range(n - 1):
            pivot = LU[ku, j]
            if abs(pivot) < EPS_MACH:
                # 需要选主元
                max_val = 0.0
                max_row = j
                for i in range(j + 1, min(j + kl + 1, n)):
                    val = abs(LU[ku + i - j, j])
                    if val > max_val:
                        max_val = val
                        max_row = i
                if max_val < EPS_MACH:
                    raise ValueError("BandedMatrix: singular matrix in banded_solve")
                # 行交换
                if max_row != j:
                    for col in range(j, min(j + ku + 1, n)):
                        LU[ku + j - col, col], LU[ku + max_row - col, col] = \
                            LU[ku + max_row - col, col], LU[ku + j - col, col]
                    x[j], x[max_row] = x[max_row], x[j]
                pivot = LU[ku, j]

            for i in range(j + 1, min(j + kl + 1, n)):
                factor = LU[ku + i - j, j] / pivot
                LU[ku + i - j, j] = factor
                for col in range(j + 1, min(j + ku + 1, n)):
                    LU[ku + i - col, col] -= factor * LU[ku + j - col, col]
                x[i] -= factor * x[j]

        # 回代
        for j in range(n - 1, -1, -1):
            if abs(LU[ku, j]) < EPS_MACH:
                x[j] = 0.0
                continue
            for col in range(j + 1, min(j + ku + 1, n)):
                x[j] -= LU[ku + j - col, col] * x[col]
            x[j] /= LU[ku, j]

        return x

    def condition_number_estimate(self):
        """
        估计带状矩阵的条件数

        使用 Gershgorin 圆盘定理:
          λ_i ∈ ∪_i {z : |z - A_{ii}| ≤ Σ_{j≠i} |A_{ij}|}

        条件数估计: κ ≈ max|λ| / min|λ|

        Returns
        -------
        float
            条件数估计
        """
        diag = np.array([self.data[self.ku, j] for j in range(self.n)])
        radii = np.zeros(self.n)
        for j in range(self.n):
            for i in range(self.n):
                if i != j:
                    radii[j] += abs(self[i, j])

        lambda_max = np.max(np.abs(diag) + radii)
        lambda_min = np.max([np.min(np.abs(diag) - radii), EPS_MACH])
        return lambda_max / lambda_min


# ===================================================================
# S 矩阵 (散射矩阵)
# ===================================================================

class ScatteringMatrix:
    """
    多通道 S 矩阵

    对于 N 通道的散射问题:
      S = I + 2i √ρ T √ρ

    其中 ρ_i = 2k_i/√s 为通道 i 的相空间因子。

    幺正性约束:
      S†S = I
      T - T† = 2i T† ρ T

    参数化 (K 矩阵):
      T = K(I - iρK)⁻¹
      S = (I + iK√ρ)(I - iK√ρ)⁻¹

    这自动保证幺正性 (当 K 为实对称时)。

    Attributes
    ----------
    n_channels : int
        通道数
    S : ndarray, shape (n_channels, n_channels)
        S 矩阵
    rho : ndarray, shape (n_channels,)
        相空间因子
    """

    def __init__(self, n_channels):
        self.n_channels = n_channels
        self.S = np.eye(n_channels, dtype=np.complex128)
        self.rho = np.zeros(n_channels, dtype=np.float64)
        self.K = np.zeros((n_channels, n_channels), dtype=np.float64)

    def set_k_matrix(self, K_matrix, rho_vec):
        """
        从 K 矩阵和相空间构造 S 矩阵

        S = (I + i√ρ K √ρ)(I - i√ρ K √ρ)⁻¹

        这等价于:
          T = K(I - iρK)⁻¹
          S = I + 2i√ρ T √ρ

        Parameters
        ----------
        K_matrix : ndarray, shape (N, N)
            实对称 K 矩阵
        rho_vec : ndarray, shape (N,)
            相空间因子
        """
        self.K = np.asarray(K_matrix, dtype=np.float64)
        self.rho = np.asarray(rho_vec, dtype=np.float64)

        sqrt_rho = np.sqrt(np.maximum(self.rho, 0.0))
        sqrt_rho_mat = np.diag(sqrt_rho)

        # K̃ = √ρ K √ρ
        K_tilde = sqrt_rho_mat @ self.K @ sqrt_rho_mat

        # S = (I + iK̃)(I - iK̃)⁻¹
        I = np.eye(self.n_channels)
        num = I + 1j * K_tilde
        den = I - 1j * K_tilde

        try:
            self.S = num @ np.linalg.inv(den)
        except np.linalg.LinAlgError:
            # 使用带状求解器回退
            den_banded = BandedMatrix.from_dense(den)
            self.S = np.zeros_like(num)
            for j in range(self.n_channels):
                self.S[:, j] = den_banded.banded_solve(num[:, j])

    def check_unitarity(self, tol=1e-10):
        """
        检查 S 矩阵幺正性 S†S = I

        Parameters
        ----------
        tol : float
            容差

        Returns
        -------
        is_unitary : bool
            是否满足幺正性
        max_deviation : float
            最大偏差 ||S†S - I||
        """
        SdS = self.S.conj().T @ self.S
        I = np.eye(self.n_channels)
        deviation = np.max(np.abs(SdS - I))
        return deviation < tol, deviation

    def inelasticity(self):
        """
        计算非弹性参数 η_l

        对于单通道: η = |S| = 1 (纯弹性)
        对于多通道对角元: η_i = |S_{ii}|

        η_i < 1 表示通道 i 存在非弹性散射。

        Returns
        -------
        eta : ndarray, shape (N,)
            各通道非弹性参数
        """
        return np.abs(np.diag(self.S))

    def eigenphases(self):
        """
        计算 S 矩阵的本征相移

        S 为酉矩阵，本征值为 e^{2iδ_α}:
          S = U diag(e^{2iδ_α}) U†

        本征相移 δ_α 通过共振时快速增加 π 来标识共振态。

        Returns
        -------
        eigenphases : ndarray, shape (N,)
            本征相移 [弧度]
        """
        eigenvalues = np.linalg.eigvals(self.S)
        phases = np.angle(eigenvalues) / 2.0
        return np.sort(phases)

    def to_banded(self, bandwidth=None):
        """
        将 S 矩阵转换为带状存储

        Parameters
        ----------
        bandwidth : int, optional
            带宽 (默认为 n_channels - 1)

        Returns
        -------
        BandedMatrix
        """
        if bandwidth is None:
            bandwidth = self.n_channels - 1
        return BandedMatrix.from_dense(self.S, bandwidth, bandwidth)


# ===================================================================
# 整数行简化 (融合 569_i4mat_rref2)
# ===================================================================

def integer_rref(matrix, n_pivot_cols=None):
    """
    整数矩阵的精确行简化阶梯形 (IRREF)

    使用纯整数运算进行高斯消去，避免浮点舍入误差。
    这对分析散射约束的精确秩至关重要。

    算法 (来自 569_i4mat_rref2/i4mat_rref2.m):
      1. 搜索非零主元 (整数)
      2. 若主元为负，整行取反
      3. 消除行公因子 (除以 GCD)
      4. 用主元行消去其他行

    物理应用:
      幺正性约束 Σ_k S_{ki}*S_{kj} = δ_{ij} 可表示为
      关于 S 矩阵实部和虚部的线性约束。
      整数 RREF 用于精确确定独立约束的数量。

    Parameters
    ----------
    matrix : ndarray of int, shape (m, n)
        输入整数矩阵
    n_pivot_cols : int, optional
        主元列数 (默认为 n)

    Returns
    -------
    irref_matrix : ndarray of int, shape (m, n)
        整数行简化阶梯形
    rank : int
        矩阵秩
    """
    A = np.array(matrix, dtype=np.int64)
    m, n = A.shape
    if n_pivot_cols is None:
        n_pivot_cols = n

    rank = 0
    lead = 0

    for r in range(m):
        if lead >= n_pivot_cols:
            break

        i = r
        while A[i, lead] == 0:
            i += 1
            if i >= m:
                i = r
                lead += 1
                if lead >= n_pivot_cols:
                    return A, rank

        # 行交换
        if i != r:
            A[[i, r]] = A[[r, i]]

        # 确保主元为正
        if A[r, lead] < 0:
            A[r] = -A[r]

        # 消除行公因子
        from math import gcd
        from functools import reduce
        row_gcd = reduce(gcd, A[r].tolist())
        if row_gcd > 1:
            A[r] //= row_gcd

        # 消去其他行
        for i in range(m):
            if i != r and A[i, lead] != 0:
                factor_r = A[r, lead]
                factor_i = A[i, lead]
                A[i] = factor_r * A[i] - factor_i * A[r]
                row_gcd = reduce(gcd, A[i].tolist())
                if row_gcd > 1:
                    A[i] //= row_gcd

        rank += 1
        lead += 1

    return A, rank


def unitarity_constraint_rank(n_channels):
    """
    计算 N 通道 S 矩阵幺正性约束的独立约束数

    幺正性 S†S = I 给出 N² 个实约束:
      - N 个对角约束: Σ_k |S_{ki}|² = 1
      - N(N-1)/2 个非对角实部约束: Re(Σ_k S_{ki}*S_{kj}) = 0
      - N(N-1)/2 个非对角虚部约束: Im(Σ_k S_{ki}*S_{kj}) = 0

    但 S 为 N×N 复矩阵有 2N² 个实参数，
    所以独立参数数为 2N² - N² = N² (U(N) 群维度)。

    Parameters
    ----------
    n_channels : int
        通道数

    Returns
    -------
    n_constraints : int
        独立约束数
    n_free_params : int
        自由参数数
    """
    n_constraints = n_channels * n_channels
    n_real_params = 2 * n_channels * n_channels
    n_free = n_real_params - n_constraints
    return n_constraints, n_free


def build_unitarity_system(n_channels):
    """
    构建幺正性约束的线性化系统

    在 S = I 附近线性化: S = I + iT
    幺正性: (I-iT†)(I+iT) = I → T - T† = 0 → T 为 Hermitian

    对于 T = A + iB:
      T = T† → B = 0, A = A^T

    独立约束数 = N² (A 为 N×N 实对称矩阵的自由度)

    Parameters
    ----------
    n_channels : int
        通道数

    Returns
    -------
    constraint_matrix : ndarray of int
        约束矩阵 (整数形式)
    rank : int
        独立约束的秩
    """
    # 构建 T 矩阵的 Hermiticity 约束
    # T_{ij} = T_{ji}* → Re(T_{ij}) = Re(T_{ji}), Im(T_{ij}) = -Im(T_{ji})
    n = n_channels
    n_vars = 2 * n * n  # N² 实部 + N² 虚部

    constraints = []
    # Re(T_{ij}) - Re(T_{ji}) = 0 for i < j
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n_vars, dtype=np.int64)
            row[i * n + j] = 1      # Re T_{ij}
            row[j * n + i] = -1     # Re T_{ji}
            constraints.append(row)

    # Im(T_{ij}) + Im(T_{ji}) = 0 for i < j
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n_vars, dtype=np.int64)
            row[n * n + i * n + j] = 1    # Im T_{ij}
            row[n * n + j * n + i] = 1    # Im T_{ji}
            constraints.append(row)

    # Im(T_{ii}) = 0 (对角元为实数)
    for i in range(n):
        row = np.zeros(n_vars, dtype=np.int64)
        row[n * n + i * n + i] = 1
        constraints.append(row)

    if len(constraints) == 0:
        return np.zeros((0, n_vars), dtype=np.int64), 0

    constraint_matrix = np.array(constraints, dtype=np.int64)
    _, rank = integer_rref(constraint_matrix)
    return constraint_matrix, rank
