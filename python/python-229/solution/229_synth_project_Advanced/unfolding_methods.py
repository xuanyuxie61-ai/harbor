"""
unfolding_methods.py
====================

来源:
  - 1074_Akiraichi_Explicit-quantum-surrogates :  SVM 核矩阵特征值分解 → SVD unfold
  - 1130_RaphaelPellegrin_PINNs                :  Physics-Informed Neural Net 正则化

物理重构: Unfolding 方法
--------------------------------
三种 unfolding 方法:

1. 矩阵求逆 (naive):
       T = R^{-1} O
   当 R 病态时 (κ(R) >> 1), 噪声被剧烈放大.

2. SVD 截断 unfold (来自 1074 的特征值分解思想):
       R = U Σ V^T
       T_k = Σ_{i=1}^k (u_i^T O / σ_i) v_i
   截断参数 k 控制正则化强度.

3. 物理约束 Neural Network unfold (来自 1130 PINNs):
       T(E) = NN(E; θ),   θ 通过最小化:
         L(θ) = ‖R · T(θ) - O‖^2 + λ_smooth · ∫ (T'')^2 dE + λ_nn · ∫ max(-T, 0)^2 dE
       约束: T(E) ≥ 0 (非负性),  T'' 小 (光滑性)

本模块实现上述三种方法.
"""

from __future__ import annotations
from typing import List, Tuple, Callable
import math
import random
import response_matrix as rm


# ===========================================================================
#             SVD 分解 (Jacobi 旋转, 小型矩阵)
# ===========================================================================

def svd_decompose(
    A: List[List[float]],
    max_iter: int = 200,
) -> Tuple[List[List[float]], List[float], List[List[float]]]:
    """
    小型矩阵 SVD: A = U Σ V^T.

    算法: 先计算 A^T A 的特征分解 → V, Σ^2;
         U = A V Σ^{-1}.

    特征分解使用 Jacobi 旋转 (对称矩阵).

    返回 (U, sigma, V).
    """
    m = len(A)
    n = len(A[0]) if m > 0 else 0

    # ATA = A^T A (n × n)
    ATA = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(m):
                s += A[k][i] * A[k][j]
            ATA[i][j] = s

    # Jacobi 特征分解
    evals, evecs = jacobi_eigen(ATA, max_iter)

    # 排序 (降序)
    order = sorted(range(n), key=lambda i: -evals[i])
    sigma = [math.sqrt(max(evals[i], 0.0)) for i in order]

    V = [[evecs[i][order[j]] for j in range(n)] for i in range(n)]
    # V 按列: V[:, j] = j-th right singular vector
    # 存储为 V_T (行 = 右奇异向量)
    V_T = [[V[i][j] for i in range(n)] for j in range(n)]

    # U = A V Σ^{-1}
    U = [[0.0] * n for _ in range(m)]
    for j in range(n):
        if sigma[j] > 1e-14:
            for i in range(m):
                s = 0.0
                for k in range(n):
                    s += A[i][k] * V[k][j]
                U[i][j] = s / sigma[j]
        else:
            # 零奇异值 → U 列正交化
            for i in range(m):
                U[i][j] = 1.0 / math.sqrt(m) if i == j % m else 0.0

    return U, sigma, V_T


def jacobi_eigen(
    A: List[List[float]],
    max_iter: int = 200,
) -> Tuple[List[float], List[List[float]]]:
    """
    对称矩阵 Jacobi 特征分解.

    返回 (特征值, 特征向量矩阵 [按列]).
    """
    n = len(A)
    S = [list(row) for row in A]
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    for _ in range(max_iter):
        # 找最大非对角元
        off = sum(S[i][j] ** 2 for i in range(n) for j in range(n) if i != j)
        if off < 1e-28:
            break
        p, q = 0, 1
        max_val = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(S[i][j]) > max_val:
                    max_val = abs(S[i][j])
                    p, q = i, j

        # 旋转角
        if abs(S[p][p] - S[q][q]) < 1e-300:
            theta = math.pi / 4.0
        else:
            theta = 0.5 * math.atan2(2.0 * S[p][q], S[p][p] - S[q][q])
        c = math.cos(theta)
        s = math.sin(theta)

        # 更新 S
        S_new = [list(row) for row in S]
        for i in range(n):
            if i != p and i != q:
                S_new[i][p] = c * S[i][p] + s * S[i][q]
                S_new[p][i] = S_new[i][p]
                S_new[i][q] = -s * S[i][p] + c * S[i][q]
                S_new[q][i] = S_new[i][q]
        S_new[p][p] = c * c * S[p][p] + 2 * s * c * S[p][q] + s * s * S[q][q]
        S_new[q][q] = s * s * S[p][p] - 2 * s * c * S[p][q] + c * c * S[q][q]
        S_new[p][q] = 0.0
        S_new[q][p] = 0.0
        S = S_new

        # 更新 V
        for i in range(n):
            vip = V[i][p]
            viq = V[i][q]
            V[i][p] = c * vip + s * viq
            V[i][q] = -s * vip + c * viq

    evals = [S[i][i] for i in range(n)]
    return evals, V


# ===========================================================================
#         SVD 截断 unfolding
# ===========================================================================

def svd_unfold(
    R_dense: List[List[float]],
    O: List[float],
    k_cutoff: int = None,
    tau: float = 1e-3,
) -> Tuple[List[float], List[float], List[List[float]], List[List[float]]]:
    """
    SVD 截断 unfolding.

    R = U Σ V^T
    T_k = Σ_{i=1}^k (u_i^T O / σ_i) v_i

    参数:
        k_cutoff : 截断阶数 (若 None, 自动选择 σ_i > τ σ_max)
        tau      : 截断阈值 (相对)

    返回 (T_unfolded, sigma, U, V_T).
    """
    U, sigma, V_T = svd_decompose(R_dense)
    m = len(R_dense)
    n = len(R_dense[0])

    if k_cutoff is None:
        sigma_max = max(sigma) if sigma else 1.0
        k_cutoff = sum(1 for s in sigma if s > tau * sigma_max)
        k_cutoff = max(1, k_cutoff)

    k_cutoff = min(k_cutoff, len(sigma))

    # T = Σ_{i=0}^{k-1} (U[:,i]^T O / σ_i) V_T[i,:]
    T = [0.0] * n
    for i in range(k_cutoff):
        if sigma[i] < 1e-300:
            continue
        # u_i^T O
        u_dot_O = sum(U[r][i] * O[r] for r in range(m))
        coeff = u_dot_O / sigma[i]
        for j in range(n):
            T[j] += coeff * V_T[i][j]

    return T, sigma, U, V_T


# ===========================================================================
#         物理约束 Neural Network unfolding (PINN 风格)
# ===========================================================================

class PhysicsInformedUnfolder:
    """
    简单 2 层前馈神经网络, 用于 unfolding.

    结构:
        input: E (1D)
        hidden: N_h 个神经元, tanh 激活
        output: T(E) ≥ 0 (softplus 激活)

    损失函数 (PINN 风格):
        L = ‖R · T - O‖^2 / N_rec
          + λ_s ∫ (T''(E))^2 dE             (光滑性)
          + λ_n · mean(max(-T, 0)^2)       (非负性, softplus 自动满足)

    训练: 简单 SGD (无自动微分, 数值梯度).
    """

    def __init__(
        self,
        E_nodes: List[float],
        N_hidden: int = 16,
        lr: float = 0.01,
        lambda_smooth: float = 0.01,
        seed: int = 42,
    ):
        self.E_nodes = list(E_nodes)
        self.N = len(E_nodes)
        self.N_h = N_hidden
        self.lr = lr
        self.lambda_s = lambda_smooth
        random.seed(seed)

        # 权重初始化 (Xavier)
        scale1 = math.sqrt(2.0 / (1 + N_hidden))
        scale2 = math.sqrt(2.0 / N_hidden)
        self.W1 = [[random.gauss(0, scale1) for _ in range(1)] for _ in range(N_hidden)]
        self.b1 = [0.0] * N_hidden
        self.W2 = [[random.gauss(0, scale2)] for _ in range(N_hidden)]
        self.b2 = 0.0

    def _tanh(self, x):
        if x > 20:
            return 1.0
        if x < -20:
            return -1.0
        return math.tanh(x)

    def _softplus(self, x):
        if x > 20:
            return x
        if x < -20:
            return 0.0
        return math.log(1.0 + math.exp(x))

    def forward(self, E: float) -> float:
        """前向: E → T(E)."""
        # 归一化 E 到 [-1, 1]
        E_min, E_max = min(self.E_nodes), max(self.E_nodes)
        E_norm = 2.0 * (E - E_min) / (E_max - E_min + 1e-300) - 1.0

        # hidden layer
        h = [self._tanh(self.W1[i][0] * E_norm + self.b1[i]) for i in range(self.N_h)]
        # output
        out = sum(self.W2[i][0] * h[i] for i in range(self.N_h)) + self.b2
        # softplus 保证非负
        return self._softplus(out)

    def predict_spectrum(self) -> List[float]:
        """在所有节点上预测 T(E)."""
        return [self.forward(E) for E in self.E_nodes]

    def compute_loss(
        self,
        R_dense: List[List[float]],
        O: List[float],
    ) -> float:
        """
        损失函数:
            L = ‖R T - O‖^2 / N_rec + λ_s Σ_i (T_{i+1} - 2 T_i + T_{i-1})^2 / h^2
        """
        T = self.predict_spectrum()
        N_rec = len(R_dense)
        N_true = len(T)

        # 数据项
        RT = [sum(R_dense[i][j] * T[j] for j in range(N_true)) for i in range(N_rec)]
        data_loss = sum((RT[i] - O[i]) ** 2 for i in range(N_rec)) / max(N_rec, 1)

        # 光滑性项 (二阶 FD)
        if self.N > 2:
            h = (self.E_nodes[-1] - self.E_nodes[0]) / max(self.N - 1, 1)
            smooth = 0.0
            for i in range(1, self.N - 1):
                d2 = (T[i + 1] - 2 * T[i] + T[i - 1]) / (h * h)
                smooth += d2 * d2
            smooth /= max(self.N - 2, 1)
        else:
            smooth = 0.0

        return data_loss + self.lambda_s * smooth

    def train(
        self,
        R_dense: List[List[float]],
        O: List[float],
        n_epochs: int = 200,
    ) -> List[float]:
        """
        训练 (数值梯度 SGD).

        返回训练后的 T(E) 在各节点的值.
        """
        eps = 1e-5
        losses = []
        for epoch in range(n_epochs):
            L0 = self.compute_loss(R_dense, O)
            losses.append(L0)

            # 对 W1, b1, W2, b2 计算数值梯度 (抽样部分参数)
            # 为效率, 每 epoch 更新一部分参数
            params = []
            for i in range(self.N_h):
                params.append(("W1", i, 0))
                params.append(("b1", i, 0))
            for i in range(self.N_h):
                params.append(("W2", i, 0))
            params.append(("b2", 0, 0))

            # 随机抽样 n_update 个参数
            n_update = min(16, len(params))
            sampled = random.sample(params, n_update)

            for (name, i, j) in sampled:
                # +eps
                self._perturb(name, i, j, eps)
                Lp = self.compute_loss(R_dense, O)
                # -eps
                self._perturb(name, i, j, -2 * eps)
                Lm = self.compute_loss(R_dense, O)
                # 恢复
                self._perturb(name, i, j, eps)
                grad = (Lp - Lm) / (2 * eps)
                # 更新
                self._perturb(name, i, j, -self.lr * grad)

        return self.predict_spectrum()

    def _perturb(self, name: str, i: int, j: int, delta: float):
        if name == "W1":
            self.W1[i][j] += delta
        elif name == "b1":
            self.b1[i] += delta
        elif name == "W2":
            self.W2[i][j] += delta
        elif name == "b2":
            self.b2 += delta


# ===========================================================================
#         BINNY 风格迭代 unfolding (D'Agostini)
# ===========================================================================

def dAgostini_unfold(
    R_dense: List[List[float]],
    O: List[float],
    T_prior: List[float] = None,
    n_iter: int = 10,
) -> List[float]:
    """
    D'Agostini 迭代贝叶斯 unfold.

    初始化: T^{(0)} = prior (默认均匀)
    迭代:
        O_exp_i = Σ_j R_{ij} T^{(n)}_j
        T^{(n+1)}_j = T^{(n)}_j · Σ_i R_{ij} O_i / (O_exp_i · Σ_k R_{ik})

    收敛后 T 趋于最大似然解.

    返回 T_unfolded.
    """
    N_rec = len(R_dense)
    N_true = len(R_dense[0])
    if T_prior is None:
        T = [1.0] * N_true
    else:
        T = list(T_prior)

    for it in range(n_iter):
        # O_exp
        O_exp = [sum(R_dense[i][j] * T[j] for j in range(N_true)) for i in range(N_rec)]
        # 列归一化: Σ_i R_{ij} / O_exp_i
        T_new = [0.0] * N_true
        for j in range(N_true):
            s = 0.0
            for i in range(N_rec):
                if O_exp[i] > 1e-300:
                    s += R_dense[i][j] * O[i] / O_exp[i]
            # 归一化: Σ_k R_{ik}
            T_new[j] = T[j] * s
        T = T_new
        # 归一化 (保持总计数)
        total = sum(T)
        if total > 0:
            O_total = sum(O)
            T = [t * O_total / total for t in T]

    return T


__all__ = [
    "svd_decompose", "jacobi_eigen",
    "svd_unfold",
    "PhysicsInformedUnfolder",
    "dAgostini_unfold",
]
