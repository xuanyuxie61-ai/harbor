"""
mhd_stability.py
磁流体动力学稳定性分析与状态转移模型。

核心物理模型：
  托卡马克中的 MHD 稳定性分析涉及大量本征值问题。
  本模块创新性地将原 tennis_matrix 的 Markov 转移矩阵思想
  应用于 MHD 模的稳定性分类：

  将等离子体状态空间离散为若干宏观态：
      1.  完全约束 (Confined)
      2.  边缘局域模 (ELM) 爆发
      3.  撕裂模 (Tearing Mode) 增长
      4.  电阻壁模 (RWM) 增长
      5.  大破裂 (Disruption) 前兆
      6.  大破裂 (Disruption)
      7.  垂直位移事件 (VDE)
      8.  恢复 (Recovery)

  转移矩阵 P 描述各状态之间的转移概率，
  类比网球计分系统的 Markov 链：

      P_{ij} = Prob(状态 j → 状态 i 在 Δt 内)

  稳态分布 π 满足 π = P^T π，对应 MHD 平衡态。
  若存在 P_{ii} → 1 的吸收态（如 Disruption），
  则系统在该态吸收。

  理想 MHD 能量原理：
      δW = 0.5 ∫ [ |Q_⊥|² / μ₀ + γ p |∇·ξ_⊥|²
                   + (ξ_⊥ · ∇p)(ξ_⊥* · κ)
                   - J_∥ (ξ_⊥* × b) · Q_⊥ ] dV

  其中 Q = ∇ × (ξ × B) 为磁场扰动，κ = b · ∇b 为曲率。
  若 δW < 0，则系统不稳定。

  本模块同时实现简化 δW 计算与基于 Markov 链的
  稳定性概率演化模型。
"""

import numpy as np
from parameters import R0, a_minor, B0, q0, q_edge, MU0


# ============================================================
# 1. MHD 状态转移矩阵 (基于原 tennis_matrix)
# ============================================================

def build_mhd_transition_matrix(p_stable=0.85, p_elm=0.08,
                                 p_tearing=0.04, p_rwm=0.02,
                                 p_disruption=0.01):
    """
    构建 MHD 稳定性状态转移矩阵。

    状态索引：
        0: Confined
        1: ELM
        2: Tearing
        3: RWM
        4: Pre-Disruption
        5: Disruption
        6: VDE
        7: Recovery

    参数
    ------
    p_stable : float
        保持在约束态的概率。
    p_elm, p_tearing, p_rwm, p_disruption : float
        各类不稳定性触发概率。

    返回
    ------
    P : ndarray, shape (8, 8)
        转移概率矩阵（列随机）。
    labels : list
        状态标签。
    """
    labels = [
        "Confined", "ELM", "Tearing", "RWM",
        "Pre-Disruption", "Disruption", "VDE", "Recovery"
    ]

    P = np.zeros((8, 8))

    # 从 Confined (0)
    P[0, 0] = p_stable
    P[1, 0] = p_elm
    P[2, 0] = p_tearing
    P[3, 0] = p_rwm
    P[4, 0] = p_disruption

    # 从 ELM (1) -> 通常回到 Confined
    P[0, 1] = 0.9
    P[1, 1] = 0.05
    P[4, 1] = 0.05

    # 从 Tearing (2)
    P[0, 2] = 0.6
    P[2, 2] = 0.2
    P[4, 2] = 0.15
    P[5, 2] = 0.05

    # 从 RWM (3)
    P[0, 3] = 0.5
    P[3, 3] = 0.2
    P[4, 3] = 0.2
    P[5, 3] = 0.1

    # 从 Pre-Disruption (4)
    P[4, 4] = 0.3
    P[5, 4] = 0.4
    P[6, 4] = 0.2
    P[7, 4] = 0.1

    # Disruption (5) 为吸收态
    P[5, 5] = 1.0

    # 从 VDE (6)
    P[5, 6] = 0.7
    P[6, 6] = 0.2
    P[7, 6] = 0.1

    # 从 Recovery (7)
    P[0, 7] = 0.8
    P[7, 7] = 0.2

    # 列归一化
    col_sums = P.sum(axis=0)
    col_sums = np.where(col_sums < 1e-15, 1.0, col_sums)
    P = P / col_sums

    return P, labels


def mhd_markov_evolution(P, initial_state, n_steps=100):
    """
    模拟 MHD 状态的 Markov 链演化。

    参数
    ------
    P : ndarray
        转移矩阵。
    initial_state : ndarray
        初始概率分布。
    n_steps : int
        演化步数。

    返回
    ------
    history : ndarray, shape (n_steps+1, n_states)
        状态概率历史。
    absorption_time : float
        平均吸收时间（到达 Disruption 的期望步数）。
    """
    n_states = P.shape[0]
    state = np.asarray(initial_state, dtype=float)
    state /= (state.sum() + 1e-30)

    history = np.zeros((n_steps + 1, n_states))
    history[0, :] = state

    absorption_step = None
    for step in range(n_steps):
        state = P @ state
        history[step + 1, :] = state
        if absorption_step is None and state[5] > 0.5:
            absorption_step = step + 1

    if absorption_step is not None:
        absorption_time = float(absorption_step)
    else:
        # 计算期望吸收时间（基本矩阵法）
        Q = np.delete(np.delete(P, 5, axis=0), 5, axis=1)
        I = np.eye(n_states - 1)
        try:
            N_fund = np.linalg.inv(I - Q)
            t_expect = N_fund.sum(axis=1)
            absorption_time = float(t_expect[0])
        except np.linalg.LinAlgError:
            absorption_time = float(n_steps)

    return history, absorption_time


def compute_ideal_mhd_delta_w(m_mode, n_mode, q_profile, r_grid, p_profile,
                               B_theta, B_phi, R=R0, gamma=5.0 / 3.0):
    """
    简化理想 MHD 能量变分 δW 计算。

    公式（简化 cylindrical 近似）
    ----------------------------
        δW = π ∫_0^a [ f(r) (dξ/dr)² + g(r) ξ² ] dr

        f(r) = (r³ / R²) B_θ² (m - n q)²
        g(r) = (2 r / R²) B_θ² (m - n q) [m - n q - (r / q) dq/dr]
                + (2 μ₀ r dp/dr) (m² - 1) / m²

    参数
    ------
    m_mode, n_mode : int
        极向与环向模数。
    q_profile : ndarray
        安全因子剖面。
    r_grid : ndarray
        小半径网格 [m]。
    p_profile : ndarray
        压强剖面 [Pa]。
    B_theta, B_phi : ndarray
        极向与环向磁场。
    R : float
        大半径 [m]。
    gamma : float
        绝热指数。

    返回
    ------
    delta_w : float
        能量变分值 [J]。
    stability : str
        "stable" 或 "unstable"。
    """
    r = np.asarray(r_grid)
    q = np.asarray(q_profile)
    p = np.asarray(p_profile)
    Bt = np.asarray(B_theta)

    if len(r) < 3:
        return 0.0, "stable"

    dr = r[1] - r[0] if len(r) > 1 else 1.0
    dq_dr = np.gradient(q, dr)
    dp_dr = np.gradient(p, dr)

    # 共振面位置
    resonant_idx = np.argmin(np.abs(m_mode - n_mode * q))

    # f(r) 与 g(r)
    m_minus_nq = m_mode - n_mode * q
    f = (r ** 3 / (R ** 2)) * (Bt ** 2) * (m_minus_nq ** 2)
    g = ((2.0 * r / (R ** 2)) * (Bt ** 2) * m_minus_nq *
         (m_minus_nq - (r / (q + 1e-20)) * dq_dr))
    g += (2.0 * MU0 * r * dp_dr * (m_mode ** 2 - 1.0) / (m_mode ** 2 + 1e-20))

    # 简化的试探函数 ξ(r) = 1 - (r/a)²
    a = r[-1]
    xi = 1.0 - (r / a) ** 2
    dxi_dr = -2.0 * r / (a ** 2)

    integrand = f * (dxi_dr ** 2) + g * (xi ** 2)
    delta_w = np.pi * np.trapezoid(integrand, r)

    stability = "unstable" if delta_w < 0 else "stable"
    return float(delta_w), stability


def compute_mercier_criterion(q_profile, r_grid, p_profile, B_phi, B_theta, R=R0):
    """
    Mercier 稳定性判据（局部判据）。

    公式
    ----
        D_M = (q / r)² [ (r⁴ / (4 R² q⁴)) (1 - q²)²
               - (2 μ₀ R² q² / B_φ²) (dp/dr) (1 - 1/q²) ]

    若 D_M < 0 在共振面处，则存在局域互换不稳定性。

    参数
    ------
    q_profile, r_grid, p_profile, B_phi, B_theta : ndarray
    R : float

    返回
    ------
    D_M : ndarray
        Mercier 函数值。
    unstable_regions : list of tuple
        (r_start, r_end) 不稳定区间。
    """
    from parameters import MU0
    r = np.asarray(r_grid)
    q = np.asarray(q_profile)
    p = np.asarray(p_profile)
    Bp = np.asarray(B_phi)
    Bt = np.asarray(B_theta)

    dr = r[1] - r[0] if len(r) > 1 else 1.0
    dq_dr = np.gradient(q, dr)
    dp_dr = np.gradient(p, dr)

    # 简化 Mercier 判据
    term1 = (r ** 4 / (4.0 * R ** 2 * (q ** 4) + 1e-30)) * (1.0 - q ** 2) ** 2
    term2 = ((2.0 * MU0 * R ** 2 * (q ** 2)) / (Bp ** 2 + 1e-30)) * dp_dr * (1.0 - 1.0 / (q ** 2 + 1e-30))
    D_M = ((q / (r + 1e-20)) ** 2) * (term1 - term2)

    # 检测不稳定区域 D_M < 0
    unstable = []
    in_unstable = False
    r_start = None
    for i in range(len(r)):
        if D_M[i] < 0:
            if not in_unstable:
                r_start = r[i]
                in_unstable = True
        else:
            if in_unstable:
                unstable.append((float(r_start), float(r[i])))
                in_unstable = False
    if in_unstable:
        unstable.append((float(r_start), float(r[-1])))

    return D_M, unstable


def compute_critical_beta(q_profile, r_grid, B_phi, R=R0, a=a_minor):
    """
    估算临界比压 β_c（Troyon 极限简化）。

    公式
    ----
        β_c [%] ≈ 3.5 · (I_p [MA] / (a [m] B_φ [T])) · (%)
        简化：β_c ≈ 3.5 · ε / q_edge

    参数
    ------
    q_profile, r_grid, B_phi : ndarray
    R, a : float

    返回
    ------
    beta_c : float
        临界比压 [%]。
    """
    epsilon = a / R
    q_edge_val = q_profile[-1] if len(q_profile) > 0 else 3.0
    beta_c = 3.5 * epsilon / (q_edge_val + 1e-10)
    return float(beta_c * 100.0)  # 转为百分比
