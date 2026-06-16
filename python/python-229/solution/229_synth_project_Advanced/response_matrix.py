"""
response_matrix.py
==================

来源:
  - 1401_wathen_matrix :  Wathen 有限元质量矩阵组装, 稀疏正定系统, CG 求解
  - 996_r8sr           :  CSR 稀疏矩阵存储与操作
  - 285_digraph_adj    :  有向图邻接矩阵 → 转移矩阵 (Markov 链) → 传递闭包

物理重构: 探测器响应矩阵 R
--------------------------------
在高能物理 unfold 中, 探测器响应矩阵 R 满足:

    O_i = Σ_j R_{ij} T_j,      i = 1..N_rec,   j = 1..N_true

其中:
    O_i    : 重建能量落在第 i 个 bin 的计数
    T_j    : 真实能量落在第 j 个 bin 的粒子数
    R_{ij} : 真实能量 E ∈ bin j 的粒子被重建到 E_rec ∈ bin i 的概率

R 的构造:
    R_{ij} = (1/N_j^MC) ∫_{bin i}^{rec} dE_rec ∫_{bin j}^{true} dE_true
              ε(E_true) A(E_true) G(E_rec; E_true, σ(E_true))

其中 G 为高斯弥散核, σ(E)/E = a/√E ⊕ b (量能器分辨率).

组装类比 (Wathen):
    Wathen 的 FE 质量矩阵 M = Σ_e ρ_e m_e 通过对单元刚度求和得到稀疏矩阵.
    类似地, R = Σ_{migration path} w_p · r_p,
    每条迁移路径贡献一个局部秩-1 矩阵到全局 R.

迁移图 (Digraph):
    将 bin 间迁移建模为有向图: 节点 = bin, 边 = 迁移概率.
    邻接矩阵 A_{ij} = R_{ij} (i ≠ j), 行归一化后得到转移矩阵 P.

CSR 存储:
    当 N_true, N_rec 较大时 (数百 ~ 数千), R 是带状稀疏矩阵.
    使用 CSR 格式: values[], col_idx[], row_ptr[].
"""

from __future__ import annotations
from typing import List, Tuple, Dict
import math
import special_functions as sf


# ===========================================================================
#            CSR 稀疏矩阵 (移植自 996_r8sr)
# ===========================================================================

class CSRMatrix:
    """
    压缩稀疏行 (CSR) 矩阵.

    存储结构:
        values   : 非零元素值 (按行主序)
        col_idx  : 对应列索引
        row_ptr  : 每行起始在 values 中的位置, 长度 = nrows + 1

    矩阵向量乘法 y = A x:
        for i in range(nrows):
            s = 0
            for k in range(row_ptr[i], row_ptr[i+1]):
                s += values[k] * x[col_idx[k]]
            y[i] = s

    对角元素单独存储以加速预处理.
    """

    def __init__(self, nrows: int, ncols: int):
        if nrows <= 0 or ncols <= 0:
            raise ValueError("CSRMatrix: 维度必须为正")
        self.nrows = nrows
        self.ncols = ncols
        self.values: List[float] = []
        self.col_idx: List[int] = []
        self.row_ptr: List[int] = [0] * (nrows + 1)
        self.diag: List[float] = [0.0] * nrows

    @classmethod
    def from_dense(cls, A: List[List[float]]) -> "CSRMatrix":
        """从稠密二维列表构造 CSR."""
        nrows = len(A)
        ncols = len(A[0]) if nrows > 0 else 0
        m = cls(nrows, ncols)
        m.values = []
        m.col_idx = []
        m.row_ptr = [0]
        for i in range(nrows):
            for j in range(ncols):
                if abs(A[i][j]) > 1e-300:
                    m.values.append(A[i][j])
                    m.col_idx.append(j)
                    if i == j:
                        m.diag[i] = A[i][j]
            m.row_ptr.append(len(m.values))
        return m

    def to_dense(self) -> List[List[float]]:
        """转为稠密二维列表."""
        A = [[0.0] * self.ncols for _ in range(self.nrows)]
        for i in range(self.nrows):
            for k in range(self.row_ptr[i], self.row_ptr[i + 1]):
                A[i][self.col_idx[k]] = self.values[k]
        return A

    def mv(self, x: List[float]) -> List[float]:
        """稀疏矩阵向量乘 y = A x."""
        if len(x) != self.ncols:
            raise ValueError(f"CSRMatrix.mv: x 长度 {len(x)} ≠ ncols {self.ncols}")
        y = [0.0] * self.nrows
        for i in range(self.nrows):
            s = 0.0
            for k in range(self.row_ptr[i], self.row_ptr[i + 1]):
                s += self.values[k] * x[self.col_idx[k]]
            y[i] = s
        return y

    def nnz(self) -> int:
        return len(self.values)

    def bandwidth(self) -> int:
        """矩阵带宽 = max |i - j| for A_{ij} ≠ 0."""
        bw = 0
        for i in range(self.nrows):
            for k in range(self.row_ptr[i], self.row_ptr[i + 1]):
                j = self.col_idx[k]
                bw = max(bw, abs(i - j))
        return bw


# ===========================================================================
#        Wathen 风格响应矩阵组装 (移植自 1401_wathen_matrix)
# ===========================================================================

def gaussian_smearing_kernel(
    E_rec: float, E_true: float, sigma: float,
) -> float:
    """
    高斯弥散核 (截断到 ±5σ):
        G(E_rec; E_true, σ) = (1 / (√(2π) σ)) exp( - (E_rec - E_true)^2 / (2σ^2) )

    物理意义: 量能器能量分辨率导致的重建能量分布.
    """
    if sigma <= 0.0:
        return 1.0 if abs(E_rec - E_true) < 1e-10 else 0.0
    z = (E_rec - E_true) / sigma
    if abs(z) > 5.0:
        return 0.0
    return math.exp(-0.5 * z * z) / (math.sqrt(2.0 * math.pi) * sigma)


def detector_resolution(E: float, a_stoch: float = 0.10, b_const: float = 0.01) -> float:
    """
    量能器能量分辨率参数化:
        σ(E) / E = a / √E  ⊕  b
        σ(E) = E * sqrt( (a/√E)^2 + b^2 ) = sqrt( a^2 E + b^2 E^2 )

    典型电磁量能器: a ≈ 10%/√GeV, b ≈ 1%.
    边界处理: E → 0 时 σ → 0 (无弥散).
    """
    if E <= 0.0:
        return 1e-10
    return math.sqrt(a_stoch * a_stoch * E + b_const * b_const * E * E)


def assemble_response_matrix(
    E_true_edges: List[float],
    E_rec_edges: List[float],
    resolution_a: float = 0.10,
    resolution_b: float = 0.01,
    efficiency: float = 0.95,
    quadrature_order: int = 5,
) -> CSRMatrix:
    """
    组装探测器响应矩阵 R (N_rec × N_true).

    算法 (Wathen 组装类比):
        对每个 (i, j) ∈ [0, N_rec) × [0, N_true):
            R_{ij} = ε · ∫_{bin i}^{rec} dE_rec ∫_{bin j}^{true} dE_true
                        G(E_rec; E_true, σ(E_true)) / (E_true_max - E_true_min)

        使用 Gauss-Legendre 求积 (quadrature_order 点).

    类比 Wathen:
        每个 (i, j) 对相当于一个 "单元", 其贡献为局部积分值.
        全局 R 为所有单元贡献的叠加.

    返回 CSRMatrix.
    """
    N_rec = len(E_rec_edges) - 1
    N_true = len(E_true_edges) - 1
    if N_rec <= 0 or N_true <= 0:
        raise ValueError("响应矩阵维度必须 > 0")

    # Gauss-Legendre 节点 (在 [-1, 1])
    import phase_space_grid as psg
    nodes, weights = psg.gauss_legendre_cos_theta(quadrature_order)
    # 映射到 [0, 1]
    nodes_half = [0.5 * (u + 1.0) for u in nodes]
    weights_half = [0.5 * w for w in weights]

    # 稠密组装 (后续转 CSR)
    R = [[0.0] * N_true for _ in range(N_rec)]

    for j in range(N_true):
        Ej_lo, Ej_hi = E_true_edges[j], E_true_edges[j + 1]
        Dj = Ej_hi - Ej_lo
        if Dj <= 0:
            continue
        for k_q in range(quadrature_order):
            E_true = Ej_lo + Dj * nodes_half[k_q]
            w_true = weights_half[k_q] * Dj
            sigma = detector_resolution(E_true, resolution_a, resolution_b)
            for i in range(N_rec):
                Ei_lo, Ei_hi = E_rec_edges[i], E_rec_edges[i + 1]
                Di = Ei_hi - Ei_lo
                if Di <= 0:
                    continue
                s = 0.0
                for l_q in range(quadrature_order):
                    E_rec = Ei_lo + Di * nodes_half[l_q]
                    w_rec = weights_half[l_q] * Di
                    s += w_rec * gaussian_smearing_kernel(E_rec, E_true, sigma)
                R[i][j] += efficiency * w_true * s / Dj

    # 行归一化检查: 每列之和应 ≈ ε (效率)
    col_sums = [sum(R[i][j] for i in range(N_rec)) for j in range(N_true)]
    for j in range(N_true):
        if col_sums[j] > 0:
            norm = efficiency / col_sums[j] if col_sums[j] > 1e-300 else 1.0
            for i in range(N_rec):
                R[i][j] *= norm

    return CSRMatrix.from_dense(R)


# ===========================================================================
#          迁移图 (Digraph) 与 Markov 转移矩阵 (285_digraph_adj)
# ===========================================================================

def migration_adjacency(R: CSRMatrix, threshold: float = 1e-6) -> CSRMatrix:
    """
    从响应矩阵 R 提取迁移邻接矩阵 A.

    A_{ij} = R_{ij}  if R_{ij} > threshold
           = 0       otherwise

    这是有向图: 边 j → i 表示 "从 true bin j 迁移到 rec bin i".
    """
    A = CSRMatrix(R.nrows, R.ncols)
    for i in range(R.nrows):
        for k in range(R.row_ptr[i], R.row_ptr[i + 1]):
            j = R.col_idx[k]
            v = R.values[k]
            if v > threshold:
                A.values.append(v)
                A.col_idx.append(j)
        A.row_ptr[i + 1] = len(A.values)
    return A


def adjacency_to_transition(A: CSRMatrix) -> CSRMatrix:
    """
    行归一化: P_{ij} = A_{ij} / Σ_k A_{ik}.
    将邻接矩阵转为 Markov 转移矩阵.

    物理含义: P_{ij} = P(从 rec bin i 回溯到 true bin j | 观测到 i).
    """
    P = CSRMatrix(A.nrows, A.ncols)
    for i in range(A.nrows):
        row_sum = sum(A.values[k] for k in range(A.row_ptr[i], A.row_ptr[i + 1]))
        if row_sum < 1e-300:
            # 无出边 → 均匀分布
            n_out = A.row_ptr[i + 1] - A.row_ptr[i]
            if n_out > 0:
                p = 1.0 / n_out
                for k in range(A.row_ptr[i], A.row_ptr[i + 1]):
                    P.values.append(p)
                    P.col_idx.append(A.col_idx[k])
        else:
            for k in range(A.row_ptr[i], A.row_ptr[i + 1]):
                P.values.append(A.values[k] / row_sum)
                P.col_idx.append(A.col_idx[k])
        P.row_ptr[i + 1] = len(P.values)
    return P


def transitive_closure(P: CSRMatrix, n_iter: int = 10) -> List[List[float]]:
    """
    传递闭包 (Warshall 风格):
        T^{(k+1)} = T^{(k)} ⊗ P,  其中 ⊗ 为矩阵乘法 (概率路径累加).

    经过 n_iter 步, T_{ij} 表示从 i 到 j 在 ≤ n_iter 步内的总概率.
    用于分析 bin 间多步迁移的连通性.
    """
    # 稠密化做乘法 (小规模)
    T = P.to_dense()
    for _ in range(n_iter):
        T_new = [[0.0] * P.ncols for _ in range(P.nrows)]
        for i in range(P.nrows):
            for k in range(P.ncols):
                if T[i][k] > 1e-300:
                    for j in range(P.ncols):
                        # P 可能不是方阵; 假设 nrows = ncols for simplicity
                        if k < len(P.row_ptr) - 1:
                            # 查 P[k][j]
                            pkj = 0.0
                            for kk in range(P.row_ptr[k], P.row_ptr[k + 1] if k < P.nrows else len(P.values)):
                                if P.col_idx[kk] == j:
                                    pkj = P.values[kk]
                                    break
                            T_new[i][j] += T[i][k] * pkj
        T = T_new
    return T


# ===========================================================================
#            共轭梯度法求解 R^T R x = R^T O (Wathen CG)
# ===========================================================================

def cg_solve(
    A_op,
    b: List[float],
    x0: List[float] = None,
    tol: float = 1e-8,
    max_iter: int = 1000,
) -> Tuple[List[float], int, List[float]]:
    """
    共轭梯度法求解 A x = b, A 对称正定.

    返回 (x, iterations, residuals).

    用于求解正规方程 R^T R T = R^T O (最小二乘 unfold).
    """
    n = len(b)
    if x0 is None:
        x = [0.0] * n
    else:
        x = list(x0)

    def mv(v):
        return A_op(v)

    r = [b[i] - mv(x)[i] for i in range(n)]
    p = list(r)
    rs_old = sum(ri * ri for ri in r)
    residuals = [math.sqrt(rs_old)]

    for it in range(max_iter):
        Ap = mv(p)
        pAp = sum(pi * Api for pi, Api in zip(p, Ap))
        if abs(pAp) < 1e-300:
            break
        alpha = rs_old / pAp
        x = [x[i] + alpha * p[i] for i in range(n)]
        r = [r[i] - alpha * Ap[i] for i in range(n)]
        rs_new = sum(ri * ri for ri in r)
        residuals.append(math.sqrt(rs_new))
        if math.sqrt(rs_new) < tol:
            return x, it + 1, residuals
        beta = rs_new / rs_old
        p = [r[i] + beta * p[i] for i in range(n)]
        rs_old = rs_new

    return x, max_iter, residuals


__all__ = [
    "CSRMatrix",
    "gaussian_smearing_kernel", "detector_resolution",
    "assemble_response_matrix",
    "migration_adjacency", "adjacency_to_transition", "transitive_closure",
    "cg_solve",
]
