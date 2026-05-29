"""
cell_cycle.py
=============
细胞周期相位动力学模型

融合原始项目：
  - 132_caesar：Caesar 移位密码 → 重新诠释为细胞周期相位的循环置换算子

数学物理模型：
  细胞周期包含四个主要相位：G1 → S → G2 → M → G1，形成一个循环群 C₄。
  设相位状态向量 p = [p_G1, p_S, p_G2, p_M]^T，则周期推进算子为：
      P · p(t) = p(t+Δt)
  其中 P 是 4×4 循环置换矩阵：
      P = [ 0 0 0 1 ;
            1 0 0 0 ;
            0 1 0 0 ;
            0 0 1 0 ]
  此即 Caesar 移位的矩阵形式（模 4 加法）。

  同时引入 chemotaxis 敏感性调制：细胞在 G1 期对趋化因子最敏感，
  M 期最不敏感。敏感性权重：
      w(φ) = w_max · (1 + cos(2π φ / 4 - π)) / 2
  其中 φ ∈ {0,1,2,3} 分别对应 G1, S, G2, M。
"""

import numpy as np


class CellCyclePhase:
    """细胞周期相位枚举"""
    G1 = 0
    S = 1
    G2 = 2
    M = 3


def caesar_cycle_shift(phase_state: np.ndarray, k: int = 1):
    """
    细胞周期相位循环置换（源自 Caesar 密码的模运算思想）。

    将相位状态向量向右循环移动 k 位：
        p'_i = p_{(i - k) mod 4}
    对应细胞从当前相位进入下一个相位。

    参数
    ----
    phase_state : np.ndarray, shape (4,)
        当前各相位细胞比例 [G1, S, G2, M]
    k : int
        移位量（k=1 表示正常推进一个相位）

    返回
    ----
    shifted : np.ndarray, shape (4,)
        移位后的相位分布
    """
    phase_state = np.asarray(phase_state, dtype=float)
    if phase_state.size != 4:
        raise ValueError("caesar_cycle_shift: phase_state 长度必须为 4")
    k = int(k) % 4
    if k == 0:
        return phase_state.copy()
    return np.roll(phase_state, k)


def cycle_transition_matrix(k: int = 1):
    """
    构造细胞周期推进的置换矩阵 P_k，满足 p(t+k) = P_k p(t)。

    矩阵形式：
        (P_k)_{i,j} = δ_{i, (j+k) mod 4}
    其中 δ 为 Kronecker delta。
    """
    k = int(k) % 4
    P = np.zeros((4, 4), dtype=float)
    for j in range(4):
        i = (j + k) % 4
        P[i, j] = 1.0
    return P


def chemotaxis_sensitivity_by_phase(phase_index: int,
                                    w_max: float = 1.0,
                                    w_min: float = 0.1):
    """
    根据细胞周期相位返回 chemotaxis 敏感性权重。

    公式：
        w(φ) = w_min + (w_max - w_min) · (1 + cos(π φ / 2 - π)) / 2

    参数
    ----
    phase_index : int
        0=G1, 1=S, 2=G2, 3=M
    w_max, w_min : float
        最大/最小敏感性

    返回
    ----
    weight : float
    """
    phase_index = int(phase_index) % 4
    theta = np.pi * phase_index / 2.0 - np.pi
    val = (1.0 + np.cos(theta)) / 2.0
    return float(w_min + (w_max - w_min) * val)


def advance_cell_cycle(phase_dist: np.ndarray,
                       dt: float,
                       transition_rates: np.ndarray = None):
    """
    以连续时间 Markov 链模型推进细胞周期。

    速率方程：
        d p_{G1} / dt = -k_{G1→S} p_{G1} + k_{M→G1} p_M
        d p_S   / dt =  k_{G1→S} p_{G1} - k_{S→G2} p_S
        d p_{G2}/ dt =  k_{S→G2} p_S    - k_{G2→M} p_{G2}
        d p_M   / dt =  k_{G2→M} p_{G2} - k_{M→G1} p_M

    参数
    ----
    phase_dist : np.ndarray, shape (4,)
        当前相位分布（归一化和为 1）
    dt : float
        时间步长
    transition_rates : np.ndarray, shape (4,), optional
        [k_G1S, k_SG2, k_G2M, k_MG1]，默认 [0.3, 0.5, 0.4, 0.6]

    返回
    ----
    new_dist : np.ndarray, shape (4,)
        推进后的相位分布
    """
    p = np.asarray(phase_dist, dtype=float)
    if abs(p.sum()) < 1e-15:
        raise ValueError("advance_cell_cycle: phase_dist 全为零")
    p = p / p.sum()

    if transition_rates is None:
        transition_rates = np.array([0.3, 0.5, 0.4, 0.6], dtype=float)
    else:
        transition_rates = np.asarray(transition_rates, dtype=float)

    if transition_rates.size != 4:
        raise ValueError("advance_cell_cycle: transition_rates 长度必须为 4")

    k = transition_rates
    # 构造生成元矩阵 Q
    Q = np.array([
        [-k[0], 0.0,   0.0,   k[3]],
        [ k[0], -k[1], 0.0,   0.0 ],
        [ 0.0,  k[1], -k[2], 0.0 ],
        [ 0.0,  0.0,   k[2], -k[3]]
    ], dtype=float)

    # 使用矩阵指数的一阶近似（向后 Euler 保证稳定性）
    # (I - dt Q) p_new = p
    A = np.eye(4) - dt * Q
    p_new = np.linalg.solve(A, p)
    p_new = np.maximum(p_new, 0.0)
    s = p_new.sum()
    if s > 1e-15:
        p_new /= s
    return p_new


def population_weighted_chemotaxis_sensitivity(phase_dist: np.ndarray,
                                                w_max: float = 1.0,
                                                w_min: float = 0.1):
    """
    计算群体加权平均 chemotaxis 敏感性。

    公式：
        \bar{w} = Σ_{φ=0}^{3} p_φ · w(φ)
    """
    p = np.asarray(phase_dist, dtype=float)
    if p.sum() < 1e-15:
        return 0.0
    p = p / p.sum()
    w = 0.0
    for phi in range(4):
        w += p[phi] * chemotaxis_sensitivity_by_phase(phi, w_max, w_min)
    return float(w)
