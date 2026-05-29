"""
optimal_source_selection.py
主动噪声控制中最优次级声源子集选择

融合原始项目:
  - 1181_subset_sum_swap (子集和交换启发式)

科学背景:
  在工程实践中,可用的次级声源数量和功率预算有限.
  需要从 N 个候选位置中选择 K 个,使得在给定功率预算 P_max 下,
  目标区域的总声压平方和最小.

  目标函数:
      min_{S \subseteq {1..N}, |S|<=K}  J(S) = \sum_{m=1}^{M} |p_m|^2
      s.t.  \sum_{n \in S} P_n <= P_max

  其中 p_m = d_m + \sum_{n \in S} H_{mn} s_n,
  d_m 为初级噪声, H_{mn} 为传递函数, s_n 为次级源信号.

  这是一个NP难的组合优化问题.本模块采用基于subset_sum_swap思想的
  贪心-交换启发式算法求解.
"""

import numpy as np


def subset_sum_swap_anc(candidate_powers, desired_power_budget):
    """
    基于subset_sum_swap的功率预算子集选择.

    参数:
        candidate_powers: 候选声源的功率消耗数组
        desired_power_budget: 总功率预算

    返回:
        selected: bool数组,表示每个候选是否被选中
        achieved_power: 实际使用的功率
    """
    n = len(candidate_powers)
    a = np.array(candidate_powers, dtype=float)
    sum_desired = float(desired_power_budget)

    # 降序排序
    order = np.argsort(-a)
    a_sorted = a[order]

    selected = np.zeros(n, dtype=bool)
    sum_achieved = 0.0

    while True:
        nmove = 0
        for idx in range(n):
            i = order[idx]
            if not selected[i]:
                if sum_achieved + a_sorted[idx] <= sum_desired + 1e-9:
                    selected[i] = True
                    sum_achieved += a_sorted[idx]
                    nmove += 1
                    continue

            if not selected[i]:
                # 尝试交换: 用一个已选的较小功率替换未选的较大功率
                for jdx in range(n):
                    j = order[jdx]
                    if selected[j]:
                        new_sum = sum_achieved + a_sorted[idx] - a_sorted[jdx]
                        if sum_achieved < new_sum <= sum_desired + 1e-9:
                            selected[j] = False
                            selected[i] = True
                            sum_achieved = new_sum
                            nmove += 2
                            break

        if nmove == 0:
            break

    return selected, sum_achieved


def greedy_source_selection(H, d, max_sources, power_budget, source_powers):
    """
    贪心算法选择使残余声压最小的次级声源.

    参数:
        H: (M, N) 传递函数矩阵 (复数)
        d: (M,) 初级噪声向量 (复数)
        max_sources: 最大可选声源数
        power_budget: 功率预算
        source_powers: (N,) 每个声源的功率消耗

    返回:
        selected: bool数组
        filters: 选中声源的最优滤波器系数
    """
    M, N = H.shape
    H = np.asarray(H, dtype=complex)
    d = np.asarray(d, dtype=complex)

    selected = np.zeros(N, dtype=bool)
    total_power = 0.0

    # 迭代贪心选择
    for _ in range(min(max_sources, N)):
        best_j = -1
        best_cost = np.inf
        best_s = None

        for j in range(N):
            if selected[j]:
                continue
            if total_power + source_powers[j] > power_budget + 1e-9:
                continue

            # 临时选中j,求解最小二乘最优系数
            temp_sel = selected.copy()
            temp_sel[j] = True
            H_sub = H[:, temp_sel]
            try:
                s = np.linalg.lstsq(H_sub, -d, rcond=None)[0]
                residual = d + H_sub @ s
                cost = np.vdot(residual, residual).real
            except np.linalg.LinAlgError:
                cost = np.inf

            if cost < best_cost:
                best_cost = cost
                best_j = j
                best_s = s

        if best_j < 0:
            break

        selected[best_j] = True
        total_power += source_powers[best_j]

    # 最终求解所有选中源的最优系数
    if np.any(selected):
        H_sel = H[:, selected]
        try:
            filters = np.linalg.lstsq(H_sel, -d, rcond=None)[0]
        except np.linalg.LinAlgError:
            filters = np.zeros(np.sum(selected), dtype=complex)
    else:
        filters = np.array([], dtype=complex)

    return selected, filters
