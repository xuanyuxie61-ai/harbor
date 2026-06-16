"""
linalg_solver.py
================
线性代数核心 —— 高斯消元、主元选取、行列式、逆矩阵

融合种子项目:
  - 337_eros: Gauss 消元, 部分主元, 行列式, 矩阵逆, PLU 分解

核心公式:
  1. Gauss 消元: a_{ij}^{(k+1)} = a_{ij}^{(k)} - (a_{ik}^{(k)}/a_{kk}^{(k)}) · a_{kj}^{(k)}
  2. 部分主元: 选 |a_{pk}| = max_{i≥k} |a_{ik}|, 交换行 p ↔ k
  3. PLU 分解: PA = LU, L 下三角单位对角, U 上三角
  4. 行列式: det(A) = (-1)^s · Π U_{ii}, s 为交换次数
  5. 条件数: κ(A) = ||A|| · ||A^{-1}|| (衡量线性系统病态程度)
"""

import numpy as np
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# 1. 全局增广矩阵 (模拟 337_eros 的 global 结构, 但用类封装)
# ---------------------------------------------------------------------------

class GaussianElimination:
    """带部分主元的高斯消元法 (源自 337_eros).

    维护增广矩阵 Ab = [A | B], 其中 A 是 n×n 系统矩阵,
    B 是 n×k 右端项 (可同时求解 k 个系统).

    核心操作:
      - pivot(j): 对第 j 列选主元
      - elim(i, j): 用第 j 行消去第 i 行
      - swap(i, j): 行交换
      - scale(i, s): 第 i 行缩放
    """

    def __init__(self, A: np.ndarray, B: Optional[np.ndarray] = None):
        """初始化增广矩阵.
        A: (n, n) 系统矩阵
        B: (n, k) 右端项, 若 None 则计算逆矩阵
        """
        m, n = A.shape
        if m != n:
            raise ValueError(f"矩阵必须是方阵, 得到 {m}×{n}")
        self.m = m
        self.n = n
        if B is None:
            # 构造 [A | I] 用于求逆
            self.Ab = np.hstack([A.astype(float).copy(), np.eye(n)])
            self.k = n
        else:
            if B.ndim == 1:
                B = B.reshape(-1, 1)
            self.Ab = np.hstack([A.astype(float).copy(), B.astype(float).copy()])
            self.k = B.shape[1]

    def scale_row(self, i: int, s: float):
        """行缩放: row_i ← s · row_i (源自 337_eros/scale)."""
        self.Ab[i, :] *= s

    def swap_rows(self, i: int, j: int):
        """行交换: row_i ↔ row_j (源自 337_eros/swap)."""
        temp = self.Ab[i, :].copy()
        self.Ab[i, :] = self.Ab[j, :]
        self.Ab[j, :] = temp

    def eliminate(self, i: int, j: int):
        """消元: row_i ← row_i - (Ab[i,j]/Ab[j,j]) · row_j (源自 337_eros/elim)."""
        if abs(self.Ab[j, j]) < 1e-300:
            raise ValueError(f"主元 Ab[{j},{j}] 为零, 无法消元")
        factor = self.Ab[i, j] / self.Ab[j, j]
        self.Ab[i, :] -= factor * self.Ab[j, :]

    def pivot(self, j: int) -> int:
        """部分主元选取 (源自 337_eros/pivot).
        在 j:n 行中选取 |Ab[i,j]| 最大的行, 交换到第 j 行.
        返回主元行索引.
        """
        col_vals = np.abs(self.Ab[j:self.n, j])
        max_idx = np.argmax(col_vals) + j
        max_val = col_vals[max_idx - j]

        if max_val < 1e-300:
            raise ValueError(f"第 {j} 列找不到非零主元 (矩阵奇异)")

        if max_idx != j:
            self.swap_rows(j, max_idx)
        return max_idx

    def forward_elimination(self) -> int:
        """前向消元 (源自 337_eros/gauss).
        将 Ab 化为行阶梯形 [U | B'].
        返回交换次数的符号 (-1)^swaps.
        """
        sign = 1
        for j in range(self.n):
            p = self.pivot(j)
            if p != j:
                sign = -sign
            for i in range(j + 1, self.n):
                self.eliminate(i, j)
        return sign

    def back_substitution(self) -> np.ndarray:
        """回代求解 (源自 337_eros/solution).
        假设 Ab 已被化为 [I | X] 或 [U | B'].
        返回解 X (n, k).
        """
        # 先化为简化行阶梯形
        for j in range(self.n - 1, -1, -1):
            # 缩放主元为 1
            if abs(self.Ab[j, j]) < 1e-300:
                raise ValueError(f"第 {j} 个主元为零")
            self.scale_row(j, 1.0 / self.Ab[j, j])
            # 向上消元
            for i in range(j):
                self.eliminate(i, j)

        return self.Ab[:, self.n:self.n + self.k]

    def solve(self) -> np.ndarray:
        """完整求解流程: 前向消元 + 回代.
        返回解 X (n, k).
        """
        self.forward_elimination()
        return self.back_substitution()

    def determinant(self) -> float:
        """计算行列式 (源自 337_eros/gauss_det).
        det(A) = (-1)^swaps · Π U_{ii}
        """
        Ab_save = self.Ab.copy()
        sign = self.forward_elimination()
        det_val = float(sign)
        for j in range(self.n):
            det_val *= self.Ab[j, j]
        self.Ab = Ab_save
        return det_val

    def inverse(self) -> np.ndarray:
        """计算逆矩阵 (源自 337_eros/gauss_inverse).
        通过 [A | I] → [I | A^{-1}].
        """
        self.forward_elimination()
        return self.back_substitution()


# ---------------------------------------------------------------------------
# 2. PLU 分解 (源自 337_eros/gauss_plu)
# ---------------------------------------------------------------------------

def plu_decomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """PLU 分解: PA = LU.
    源自 337_eros/gauss_plu.

    返回 (P, L, U, swaps).
    P: 置换矩阵
    L: 下三角 (单位对角)
    U: 上三角
    swaps: 行交换次数
    """
    n = A.shape[0]
    U = A.astype(float).copy()
    L = np.eye(n)
    P = np.eye(n)
    swaps = 0

    for j in range(n):
        # 部分主元
        max_idx = j + np.argmax(np.abs(U[j:, j]))
        if abs(U[max_idx, j]) < 1e-300:
            continue  # 奇异列

        if max_idx != j:
            U[[j, max_idx]] = U[[max_idx, j]]
            P[[j, max_idx]] = P[[max_idx, j]]
            swaps += 1
            # 交换 L 的前 j 列
            if j > 0:
                L[j, :j], L[max_idx, :j] = L[max_idx, :j].copy(), L[j, :j].copy()

        for i in range(j + 1, n):
            if abs(U[j, j]) < 1e-300:
                continue
            L[i, j] = U[i, j] / U[j, j]
            U[i, j:] -= L[i, j] * U[j, j:]

    return P, L, U, swaps


# ---------------------------------------------------------------------------
# 3. 矩阵条件数估计
# ---------------------------------------------------------------------------

def condition_number(A: np.ndarray) -> float:
    """估计矩阵条件数 κ(A) = ||A||_∞ · ||A^{-1}||_∞.
    条件数衡量线性系统 Ax=b 的病态程度:
      - κ ≈ 1: 良态
      - κ >> 1: 病态 (解对扰动敏感)
      - κ = ∞: 奇异

    在 Newton 法中, Hessian 条件数直接影响收敛速度.
    """
    try:
        A_inv = np.linalg.inv(A)
        return float(np.linalg.norm(A, np.inf) * np.linalg.norm(A_inv, np.inf))
    except np.linalg.LinAlgError:
        return float('inf')


def condition_number_2(A: np.ndarray) -> float:
    """2-范数条件数: κ₂(A) = σ_max/σ_min (奇异值之比)."""
    try:
        s = np.linalg.svd(A, compute_uv=False)
        if s[-1] < 1e-300:
            return float('inf')
        return float(s[0] / s[-1])
    except np.linalg.LinAlgError:
        return float('inf')


# ---------------------------------------------------------------------------
# 4. 对称正定矩阵的 Cholesky 分解
# ---------------------------------------------------------------------------

def cholesky_factorization(A: np.ndarray) -> Optional[np.ndarray]:
    """Cholesky 分解: A = LL^T, A 对称正定.
    L[i,i] = √(A[i,i] - Σ_{k<i} L[i,k]²)
    L[i,j] = (A[i,j] - Σ_{k<j} L[i,k]L[j,k]) / L[j,j],  i > j

    若 A 不正定, 返回 None.
    """
    n = A.shape[0]
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i, k] * L[j, k] for k in range(j))
            if i == j:
                val = A[i, i] - s
                if val <= 0:
                    return None  # 非正定
                L[i, j] = math_sqrt(val)
            else:
                if abs(L[j, j]) < 1e-300:
                    return None
                L[i, j] = (A[i, j] - s) / L[j, j]
    return L


def math_sqrt(x: float) -> float:
    """安全平方根, 处理负数边界."""
    if x < 0:
        return 0.0
    return x ** 0.5


# ---------------------------------------------------------------------------
# 5. 共轭梯度法 (大型稀疏对称正定系统)
# ---------------------------------------------------------------------------

def conjugate_gradient(A: np.ndarray, b: np.ndarray,
                       x0: Optional[np.ndarray] = None,
                       tol: float = 1e-10,
                       max_iter: int = 1000) -> Tuple[np.ndarray, int, float]:
    """共轭梯度法求解 Ax = b (A 对称正定).

    算法 (Hestenes & Stiefel, 1952):
      r₀ = b - Ax₀
      p₀ = r₀
      for k = 0, 1, ...
        α_k = r_k^T r_k / (p_k^T A p_k)
        x_{k+1} = x_k + α_k p_k
        r_{k+1} = r_k - α_k A p_k
        β_k = r_{k+1}^T r_{k+1} / (r_k^T r_k)
        p_{k+1} = r_{k+1} + β_k p_k

    收敛性: 至多 n 步到达精确解 (算术精确), 实际 O(√κ · log(1/ε)) 步.

    返回 (x, iterations, residual_norm).
    """
    n = len(b)
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    r = b - A @ x
    p = r.copy()
    rs_old = np.dot(r, r)

    for k in range(max_iter):
        Ap = A @ p
        pAp = np.dot(p, Ap)
        if abs(pAp) < 1e-300:
            break
        alpha = rs_old / pAp
        x += alpha * p
        r -= alpha * Ap
        rs_new = np.dot(r, r)

        if np.sqrt(rs_new) < tol:
            return x, k + 1, np.sqrt(rs_new)

        beta = rs_new / max(rs_old, 1e-300)
        p = r + beta * p
        rs_old = rs_new

    return x, max_iter, np.sqrt(rs_new)


# ---------------------------------------------------------------------------
# 6. 对称特征值问题 (用于 Newton 修正)
# ---------------------------------------------------------------------------

def symmetric_eigenvalues(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """对称矩阵特征值分解 (用于修正非正定 Hessian).
    A = Q Λ Q^T, Λ = diag(λ₁,...,λₙ).

    在优化中, 若 Hessian 有负特征值, 则当前点非极小点.
    修正策略: 将负特征值替换为小正数 (Levenberg-Marquardt 修正).
    """
    # 对称化
    A_sym = 0.5 * (A + A.T)
    eigvals, eigvecs = np.linalg.eigh(A_sym)
    return eigvals, eigvecs


def hessian_modification(H: np.ndarray, delta: float = 1e-6) -> np.ndarray:
    """修正 Hessian 使其正定 (用于 Newton 法).
    策略 1: H_mod = H + δI (Tikhonov 正则化)
    策略 2: 特征值修正, 将 λ_min 提升到 δ

    返回修正后的正定 Hessian.
    """
    eigvals, eigvecs = symmetric_eigenvalues(H)
    min_eig = eigvals[0]

    if min_eig > delta:
        return H.copy()  # 已正定

    # 特征值修正
    tau = delta - min_eig + 1e-8
    H_mod = H + tau * np.eye(H.shape[0])
    return H_mod
