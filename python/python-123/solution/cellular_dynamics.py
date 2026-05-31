"""
cellular_dynamics.py

肿瘤细胞群体状态演化与细胞自动机模块

本模块融合以下种子项目的核心算法：
  - 673_lights_out_game: 网格邻居交互状态更新
  - 1200_tennis_matrix: 马尔可夫链转移矩阵

科学背景：
  肿瘤微环境中的细胞存在多种状态：增殖态(P)、静息态(Q)、凋亡态(A)、
  坏死态(N)。细胞状态转移受局部微环境（氧气、营养、细胞密度）调控。

  我们建立连续时间马尔可夫链（CTMC）模型：
    dP/dt = lambda_PP * P + lambda_QP * Q - (lambda_PQ + lambda_PA) * P
    dQ/dt = lambda_PQ * P + lambda_AQ * A - (lambda_QP + lambda_QA) * Q
    dA/dt = lambda_PA * P + lambda_QA * Q - lambda_AQ * A

  离散化后得到转移概率矩阵 M，满足 sum_j M_{ij} = 1。

  细胞自动机（CA）规则基于 "Lights Out" 的邻域交互思想：
    每个格点的细胞状态受其 4/8 邻域内细胞密度的影响，
    高密度邻域触发接触抑制（contact inhibition），
    低氧邻域触发凋亡或坏死。
"""

import numpy as np
from typing import Tuple


# 细胞状态枚举
STATE_PROLIFERATION = 0
STATE_QUIESCENCE = 1
STATE_APOPTOSIS = 2
STATE_NECROSIS = 3


def cell_transition_matrix(
    p_prolif_to_quies: float = 0.15,
    p_prolif_to_apop: float = 0.05,
    p_quies_to_prolif: float = 0.20,
    p_quies_to_apop: float = 0.10,
    p_apop_to_quies: float = 0.02,
    p_necrosis_recovery: float = 0.0
) -> np.ndarray:
    """
    构建肿瘤细胞状态转移矩阵（基于网球矩阵的 Markov 链思想）。

    状态空间: [P, Q, A, N] 对应 0,1,2,3

    转移矩阵 M 满足 M[i,j] = P(下一状态=j | 当前状态=i)

    参数约束：每行和为 1，所有概率在 [0,1] 内。
    """
    for p in [p_prolif_to_quies, p_prolif_to_apop, p_quies_to_prolif,
              p_quies_to_apop, p_apop_to_quies, p_necrosis_recovery]:
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"cell_transition_matrix: 概率 {p} 超出 [0,1]")

    M = np.zeros((4, 4))

    # 从 Proliferation (P)
    M[0, 0] = 1.0 - p_prolif_to_quies - p_prolif_to_apop
    M[0, 1] = p_prolif_to_quies
    M[0, 2] = p_prolif_to_apop
    M[0, 3] = 0.0

    # 从 Quiescence (Q)
    M[1, 0] = p_quies_to_prolif
    M[1, 1] = 1.0 - p_quies_to_prolif - p_quies_to_apop
    M[1, 2] = p_quies_to_apop
    M[1, 3] = 0.0

    # 从 Apoptosis (A) -> 不可逆，但允许极缓慢回到 Q（免疫清除后再定植）
    M[2, 0] = 0.0
    M[2, 1] = p_apop_to_quies
    M[2, 2] = 1.0 - p_apop_to_quies
    M[2, 3] = 0.0

    # 从 Necrosis (N) -> 不可逆
    M[3, 0] = 0.0
    M[3, 1] = 0.0
    M[3, 2] = 0.0
    M[3, 3] = 1.0

    # 边界处理：确保每行和为 1
    row_sums = M.sum(axis=1)
    for i in range(4):
        if abs(row_sums[i] - 1.0) > 1e-12:
            if row_sums[i] > 1e-15:
                M[i, :] /= row_sums[i]
            else:
                M[i, i] = 1.0

    return M


def evolve_cell_population_markov(
    initial_counts: np.ndarray, trans_matrix: np.ndarray, steps: int
) -> np.ndarray:
    """
    使用马尔可夫链演化细胞群体分布。

    递推公式:
        N_{t+1}^T = N_t^T * M
      或等价地 N_{t+1} = M^T * N_t

    参数:
        initial_counts: 长度为 4 的数组 [P, Q, A, N]
        trans_matrix: 4x4 转移矩阵
        steps: 演化步数

    返回:
        history: (steps+1, 4) 的演化历史
    """
    initial_counts = np.asarray(initial_counts, dtype=float)
    if initial_counts.shape[0] != 4:
        raise ValueError("evolve_cell_population_markov: initial_counts 长度必须为 4")
    if trans_matrix.shape != (4, 4):
        raise ValueError("evolve_cell_population_markov: 转移矩阵必须是 4x4")
    if steps < 0:
        raise ValueError("evolve_cell_population_markov: steps >= 0")

    history = np.zeros((steps + 1, 4))
    history[0, :] = initial_counts
    state = initial_counts.copy()

    for t in range(1, steps + 1):
        state = trans_matrix.T @ state
        # 数值鲁棒性：防止负值
        state = np.where(state < 0, 0.0, state)
        history[t, :] = state

    return history


def ca_contact_inhibition_update(
    cell_grid: np.ndarray, nutrient_grid: np.ndarray,
    threshold_nutrient: float = 0.1,
    inhibition_threshold: int = 5
) -> np.ndarray:
    """
    基于 Lights Out 邻域交互思想的细胞自动机更新。

    规则：
      1. 对每个格点 (i,j)，统计其 8-邻域内活细胞数 neighbor_count
      2. 若 neighbor_count >= inhibition_threshold：
            触发接触抑制 -> 细胞进入静息态 (Q)
      3. 若 nutrient_grid[i,j] < threshold_nutrient：
            低氧 -> 细胞凋亡 (A)
      4. 否则保持或恢复增殖 (P)

    细胞状态编码：
        0 = P (Proliferation), 1 = Q (Quiescence),
        2 = A (Apoptosis),     3 = N (Necrosis)

    参数:
        cell_grid: (H, W) 整数数组，当前细胞状态
        nutrient_grid: (H, W) 浮点数组，局部营养浓度
        threshold_nutrient: 营养阈值
        inhibition_threshold: 接触抑制阈值（邻居数）

    返回:
        new_grid: 更新后的细胞状态网格
    """
    H, W = cell_grid.shape
    if nutrient_grid.shape != (H, W):
        raise ValueError("ca_contact_inhibition_update: 网格尺寸不匹配")

    new_grid = cell_grid.copy()

    for i in range(H):
        for j in range(W):
            # 统计 8-邻域内活细胞（P 或 Q 视为活细胞）
            neighbor_count = 0
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < H and 0 <= nj < W:
                        if cell_grid[ni, nj] in (STATE_PROLIFERATION, STATE_QUIESCENCE):
                            neighbor_count += 1

            current = cell_grid[i, j]

            # Necrosis 不可逆
            if current == STATE_NECROSIS:
                new_grid[i, j] = STATE_NECROSIS
                continue

            # 极低氧 -> 坏死（如果已经凋亡或长期缺氧）
            if nutrient_grid[i, j] < threshold_nutrient * 0.3:
                if current == STATE_APOPTOSIS:
                    new_grid[i, j] = STATE_NECROSIS
                else:
                    new_grid[i, j] = STATE_APOPTOSIS
                continue

            # 低氧 -> 凋亡
            if nutrient_grid[i, j] < threshold_nutrient:
                if current == STATE_PROLIFERATION:
                    new_grid[i, j] = STATE_APOPTOSIS
                else:
                    new_grid[i, j] = current
                continue

            # 接触抑制
            if neighbor_count >= inhibition_threshold:
                if current == STATE_PROLIFERATION:
                    new_grid[i, j] = STATE_QUIESCENCE
                else:
                    new_grid[i, j] = current
                continue

            # 正常环境 -> 趋向增殖
            if current == STATE_QUIESCENCE and nutrient_grid[i, j] > threshold_nutrient * 2.0:
                new_grid[i, j] = STATE_PROLIFERATION
            elif current == STATE_APOPTOSIS and nutrient_grid[i, j] > threshold_nutrient * 1.5:
                # 低概率恢复（模拟免疫编辑后残余细胞）
                new_grid[i, j] = STATE_QUIESCENCE
            else:
                new_grid[i, j] = current

    return new_grid


def ca_proliferation_step(
    cell_grid: np.ndarray, empty_probability: float = 0.1
) -> np.ndarray:
    """
    细胞增殖步：增殖态细胞以一定概率向空位（或较弱竞争位）扩展。

    实现方式：对 grid 中的每个 P 细胞，随机选择一个空邻域填充新的 P。
    """
    H, W = cell_grid.shape
    new_grid = cell_grid.copy()
    rng = np.random.default_rng(seed=42)

    for i in range(H):
        for j in range(W):
            if cell_grid[i, j] == STATE_PROLIFERATION:
                if rng.random() < empty_probability:
                    # 寻找空位或凋亡/坏死位
                    candidates = []
                    for di in (-1, 0, 1):
                        for dj in (-1, 0, 1):
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < H and 0 <= nj < W:
                                if cell_grid[ni, nj] in (STATE_APOPTOSIS, STATE_NECROSIS):
                                    candidates.append((ni, nj))
                    if candidates:
                        ni, nj = candidates[rng.integers(len(candidates))]
                        new_grid[ni, nj] = STATE_PROLIFERATION
    return new_grid


def compute_tumor_cellularity(cell_grid: np.ndarray) -> Tuple[float, float, float, float]:
    """
    计算肿瘤网格中各状态细胞的比例。

    返回:
        frac_P, frac_Q, frac_A, frac_N
    """
    total = cell_grid.size
    if total == 0:
        return 0.0, 0.0, 0.0, 0.0

    frac_P = np.sum(cell_grid == STATE_PROLIFERATION) / total
    frac_Q = np.sum(cell_grid == STATE_QUIESCENCE) / total
    frac_A = np.sum(cell_grid == STATE_APOPTOSIS) / total
    frac_N = np.sum(cell_grid == STATE_NECROSIS) / total
    return float(frac_P), float(frac_Q), float(frac_A), float(frac_N)


def compute_doubling_time(population_history: np.ndarray) -> float:
    """
    基于马尔可夫链历史估算有效群体倍增时间。

    假设增殖态细胞指数增长: N(t) = N0 * exp(lambda * t)
    倍增时间 T_d = ln(2) / lambda
    """
    if population_history.shape[0] < 2:
        return np.inf

    total = population_history.sum(axis=1)
    proliferative = population_history[:, STATE_PROLIFERATION]

    # 只对 total > 0 的时段拟合
    valid = total > 1e-12
    if np.sum(valid) < 2:
        return np.inf

    t_vals = np.arange(population_history.shape[0])[valid]
    y_vals = np.log(total[valid] + 1e-12)

    # 线性拟合 ln(N) ~ a + lambda * t
    n = t_vals.shape[0]
    sum_t = np.sum(t_vals)
    sum_y = np.sum(y_vals)
    sum_tt = np.sum(t_vals ** 2)
    sum_ty = np.sum(t_vals * y_vals)
    denom = n * sum_tt - sum_t ** 2
    if abs(denom) < 1e-15:
        return np.inf

    lam = (n * sum_ty - sum_t * sum_y) / denom
    if lam <= 1e-15:
        return np.inf

    T_d = np.log(2.0) / lam
    return float(T_d)
