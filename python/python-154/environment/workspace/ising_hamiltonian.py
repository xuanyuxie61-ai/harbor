"""
ising_hamiltonian.py
================================================================================
Constructs the Ising Hamiltonian and QUBO formulation for quantum annealing
combinatorial optimization.

融合来源：622_knapsack_01_brute（0/1 背包 Gray 码枚举）

核心物理模型：
  对于一个 N  spin 的伊辛系统，哈密顿量为

      H_Ising = Σ_{i<j} J_{ij} σ_i^z σ_j^z + Σ_i h_i σ_i^z

  其中 σ_i^z ∈ {+1, -1}。通过映射 s_i = (1 - x_i)/2 可将二进制变量
  x_i ∈ {0,1} 的 QUBO 问题转化为 Ising 模型：

      H_QUBO = Σ_{i≤j} Q_{ij} x_i x_j  →  H_Ising(s)

  转换关系：
      J_{ij} = Q_{ij}/4   (i≠j)
      h_i    = Q_{ii}/2 + Σ_{j≠i} Q_{ij}/4
      C      = Σ_{i≤j} Q_{ij}/4
"""

import numpy as np
from typing import Tuple, Optional


class IsingHamiltonian:
    """
    伊辛哈密顿量构造器，支持从通用 QUBO 矩阵或背包约束生成。
    """

    def __init__(self, n_spins: int, seed: int = 154):
        if n_spins <= 0:
            raise ValueError("n_spins must be positive")
        self.n_spins = n_spins
        self.rng = np.random.default_rng(seed)
        self.J: Optional[np.ndarray] = None
        self.h: Optional[np.ndarray] = None
        self.offset: float = 0.0
        self.qubo_matrix: Optional[np.ndarray] = None

    def build_from_qubo(self, Q: np.ndarray) -> None:
        """
        从 QUBO 矩阵 Q 构造 Ising 参数 (J, h)。

        公式推导：
            x_i = (1 - s_i)/2
            H_QUBO = x^T Q x = Σ_{i,j} Q_{ij} x_i x_j
                   = Σ_{i,j} Q_{ij} (1-s_i)(1-s_j)/4
                   = Σ_{i<j} (Q_{ij}/4) s_i s_j
                     + Σ_i [ -Q_{ii}/2 - Σ_{j≠i} Q_{ij}/4 ] s_i
                     + const
        """
        if Q.shape != (self.n_spins, self.n_spins):
            raise ValueError("QUBO matrix shape mismatch")
        self.qubo_matrix = Q.astype(float)

        # 对称化 Q
        Qs = (Q + Q.T) / 2.0
        np.fill_diagonal(Qs, np.diag(Q))

        n = self.n_spins
        self.J = np.zeros((n, n))
        self.h = np.zeros(n)

        for i in range(n):
            for j in range(i + 1, n):
                self.J[i, j] = Qs[i, j] / 4.0
                self.J[j, i] = self.J[i, j]

        for i in range(n):
            self.h[i] = Qs[i, i] / 2.0
            for j in range(n):
                if j != i:
                    self.h[i] += Qs[i, j] / 4.0

        self.offset = 0.0
        for i in range(n):
            self.offset += Qs[i, i] / 4.0
            for j in range(i + 1, n):
                self.offset += Qs[i, j] / 4.0

    def build_knapsack_qubo(self, weights: np.ndarray, values: np.ndarray,
                            capacity: float, penalty: float = 5.0) -> None:
        """
        将 0/1 背包问题编码为 QUBO：

            max   Σ v_i x_i
            s.t.  Σ w_i x_i ≤ C

        引入松弛变量与惩罚项：
            H = -Σ v_i x_i + penalty * ( Σ w_i x_i - C )^2

        注意：为控制规模，当 n_spins 较大时自动截断为稀疏高斯随机场。
        """
        n = self.n_spins
        if len(weights) != n or len(values) != n:
            raise ValueError("weights/values length must equal n_spins")
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if any(w < 0 for w in weights):
            raise ValueError("weights must be non-negative")

        Q = np.zeros((n, n))
        for i in range(n):
            Q[i, i] = -values[i] + penalty * (weights[i] ** 2 - 2 * capacity * weights[i])
            for j in range(i + 1, n):
                Q[i, j] = penalty * weights[i] * weights[j]
                Q[j, i] = Q[i, j]
        self.build_from_qubo(Q)

    def build_random_ensemble(self, connectivity: float = 0.3,
                              j_mean: float = 0.0, j_std: float = 1.0,
                              h_mean: float = 0.0, h_std: float = 0.5) -> None:
        """
        生成随机自旋玻璃实例（Sherrington-Kirkpatrick 型）：

            J_{ij} ~ N(j_mean, j_std^2)   (以概率 connectivity)
            h_i    ~ N(h_mean, h_std^2)

        满足高斯随机场伊辛模型（RFIM）的统计特性。
        """
        n = self.n_spins
        if not (0.0 <= connectivity <= 1.0):
            raise ValueError("connectivity must be in [0,1]")

        self.J = np.zeros((n, n))
        mask = self.rng.random((n, n)) < connectivity
        # 上三角掩码
        triu_mask = np.triu(mask, k=1)
        self.J[triu_mask] = self.rng.normal(j_mean, j_std, size=np.count_nonzero(triu_mask))
        self.J = self.J + self.J.T
        self.h = self.rng.normal(h_mean, h_std, size=n)
        self.offset = 0.0

    def energy(self, spin_config: np.ndarray) -> float:
        """
        计算给定自旋构型的伊辛能量：

            E(s) = Σ_{i<j} J_{ij} s_i s_j + Σ_i h_i s_i + offset
        """
        # TODO: 实现伊辛能量计算公式
        # 提示：需要考虑 J 矩阵的对称性、自旋取值为 ±1、以及 offset 常数项
        raise NotImplementedError("Hole 3: 请补全伊辛哈密顿量能量计算公式")

    def exact_ground_state_brute_force(self) -> Tuple[np.ndarray, float]:
        """
        对小规模系统 (n_spins ≤ 20) 使用 Gray 码枚举精确求解基态。

        基于种子项目 622_knapsack_01_brute 的 Gray 码子集遍历思想，
        将二进制子集映射为自旋构型 s_i = 2*b_i - 1，逐位翻转更新能量差。

        能量递推更新公式：
            当第 k 位由 b_k → 1-b_k（即 s_k → -s_k）时，
            ΔE = -2 * s_k * ( Σ_j J_{kj} s_j + h_k )
        """
        n = self.n_spins
        if n > 20:
            raise ValueError("Brute force only feasible for n_spins <= 20")

        # Gray 码初始状态
        bits = np.zeros(n, dtype=int)
        spins = 2 * bits - 1
        e_min = self.energy(spins)
        s_opt = spins.copy()

        e_current = e_min
        # Gray 码生成：共 2^n 个状态，第 g 个状态与第 g-1 个状态仅差 1 位
        for g in range(1, 2 ** n):
            # 找到 Gray 码翻转位：gray = g ^ (g>>1)，与 g-1 的 gray 比较
            gray_g = g ^ (g >> 1)
            gray_prev = (g - 1) ^ ((g - 1) >> 1)
            diff = gray_g ^ gray_prev
            k = int(np.log2(diff))  # 翻转位的索引

            # 更新能量差
            delta_e = -2.0 * spins[k] * (np.dot(self.J[k, :], spins) + self.h[k])
            # 由于对称矩阵且含对角，更严谨地直接重算或修正
            # 为数值鲁棒，使用增量公式但跳过对角自作用
            delta_e = -2.0 * spins[k] * (
                np.dot(self.J[k, :], spins) - self.J[k, k] * spins[k] + self.h[k]
            )
            e_current += delta_e
            spins[k] *= -1

            if e_current < e_min:
                e_min = e_current
                s_opt = spins.copy()

        return s_opt, e_min

    def qubo_energy(self, binary_config: np.ndarray) -> float:
        """计算 QUBO 能量 x^T Q x"""
        if self.qubo_matrix is None:
            raise RuntimeError("QUBO matrix not initialized")
        x = binary_config.astype(float)
        return float(x @ self.qubo_matrix @ x)
