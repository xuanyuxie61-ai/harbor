"""
knapsack_combinatoric.py — 组合优化与喷注配对选择
=================================================
本模块实现组合优化算法用于 tt̄ 事件中
喷注-部分子配对的最佳选择问题。

物理问题:
  在 tt̄ → bjjb̄lν 事件中, 有 N_jets 个喷注,
  需要选择 2 个 b-tagged jets 和 2 个轻 jets 分别对应:
    (b, q, q̄') → 强子衰变 W → top_had
    (b̄, l, ν) → 轻子衰变 W → top_lep

  对于 N_jets 个喷注, 可能的配对数为:
    C(N,2) × C(N-2,2) × 2 = N(N-1)(N-2)(N-3)/2

  目标: 最小化 χ² = Σ (M_reco - M_expected)²/σ²

  这是一个 0/1 整数规划问题:
    min Σ_i Σ_j c_{ij} x_{ij}
    s.t. Σ_j x_{ij} ≤ 1 (每个 jet 最多分配一次)
         Σ_i x_{ij} = 1 (每个 parton 必须分配)
         x_{ij} ∈ {0, 1}

  算法:
    - 暴力枚举 (N ≤ 10 可行)
    - 分支定界 (N ≤ 20)
    - 匈牙利算法 (对松弛问题)

  映射种子项目:
    - 623_knapsack_brute: 0/1 背包暴力枚举 + 二进制计数器
"""

import numpy as np
from itertools import combinations


def binary_subset_next(s):
    """
    二进制子集计数器递增 (从 knapsack_brute)。

    将二进制向量 s 视为 n-bit 整数, 加 1。
    进位传播: 从最低位开始, 1→0 并进位, 0→1 则停止。
    全 1 后回到全 0 (溢出)。

    参数:
        s: 二进制向量 (就地修改)

    返回:
        修改后的 s
    """
    n = len(s)
    s = list(s)

    for i in range(n - 1, -1, -1):
        if s[i] == 0:
            s[i] = 1
            return s
        else:
            s[i] = 0

    return s  # 溢出: 全 0


def brute_force_jet_assignment(cost_matrix, n_jets, n_slots):
    """
    暴力枚举所有喷注分配方案, 选择最小代价。

    对于 n_slots 个部分子位和 n_jets 个喷注:
    枚举所有 C(n_jets, n_slots) × n_slots! 种分配。

    使用二进制计数器枚举所有 2^n_jets 个子集,
    筛选出恰好选择 n_slots 个喷注的子集。

    映射种子项目:
      - 623_knapsack_brute: 二进制枚举 + 约束筛选

    参数:
        cost_matrix: (n_slots, n_jets) 代价矩阵
        n_jets: 总喷注数
        n_slots: 需要选择的喷注数

    返回:
        best_assignment: 最佳分配索引列表
        best_cost: 最小代价
    """
    best_assignment = None
    best_cost = float('inf')

    # 二进制枚举
    s = [0] * n_jets
    total_checked = 0

    while True:
        s = binary_subset_next(s)
        total_checked += 1

        # 检查是否选择了正确数量的 jets
        n_selected = sum(s)
        if n_selected != n_slots:
            # 检查是否已枚举所有子集 (回到全 0)
            if n_selected == 0:
                break
            continue

        # 获取选中的 jet 索引
        selected = [i for i in range(n_jets) if s[i] == 1]

        # 对选中的 jets, 尝试所有排列 (分配到 slots)
        for perm in _permutations(selected):
            cost = 0.0
            for slot in range(n_slots):
                cost += cost_matrix[slot, perm[slot]]

            if cost < best_cost:
                best_cost = cost
                best_assignment = list(perm)

    return best_assignment, best_cost


def _permutations(items):
    """生成列表的所有排列。"""
    if len(items) <= 1:
        yield items
        return
    for i in range(len(items)):
        rest = items[:i] + items[i + 1:]
        for p in _permutations(rest):
            yield [items[i]] + p


def chi2_jet_assignment(jets_4vec, b_tagged_indices, top_mass=172.5,
                        W_mass=80.377):
    """
    计算喷注分配的 χ² 代价函数。

    对于强子衰变侧:
      χ²_had = ((M_bjj - m_top)/σ_top)² + ((M_jj - m_W)/σ_W)²

    对于轻子衰变侧 (需要中微子重建):
      χ²_lep = ((M_blν - m_top)/σ_top)² + ((M_lν - m_W)/σ_W)² +
               ((p_T_miss - p_T_ν)/σ_MET)²

    参数:
        jets_4vec: (N_jets, 4) 四动量 [E, px, py, pz]
        b_tagged_indices: b-tagged jets 的索引列表
        top_mass: 顶夸克质量假设
        W_mass: W 玻色子质量

    返回:
        cost_matrix: (n_slots, n_jets) 代价矩阵
    """
    n_jets = len(jets_4vec)
    n_b = len(b_tagged_indices)
    n_light = n_jets - n_b

    # 代价矩阵: 4 slots (b_had, j1, j2, b_lep) × n_jets
    n_slots = min(4, n_jets)
    cost = np.full((n_slots, n_jets), 1e10)

    sigma_top = 15.0  # GeV
    sigma_W = 10.0  # GeV

    # 填充代价矩阵
    for slot in range(n_slots):
        for j in range(n_jets):
            E_j = jets_4vec[j, 0]
            # 简化的代价: 基于喷注能量与期望值的偏差
            if slot < 2:  # b-jets
                if j in b_tagged_indices:
                    cost[slot, j] = (E_j - top_mass / 3.0)**2 / sigma_top**2
                else:
                    cost[slot, j] = 1e8  # 惩罚非 b-jet
            else:  # light jets
                if j not in b_tagged_indices:
                    cost[slot, j] = (E_j - W_mass / 2.0)**2 / sigma_W**2
                else:
                    cost[slot, j] = 1e8  # 惩罚 b-jet 作为 light

    return cost


def efficient_jet_matching(jets_4vec, b_indices, light_indices,
                           top_mass=172.5, W_mass=80.377):
    """
    高效喷注匹配算法 (贪心 + 局部优化)。

    对于大规模事件样本, 暴力法不可行。
    使用贪心初始分配 + 局部交换优化。

    参数:
        jets_4vec: 四动量数组
        b_indices: b-tagged jet 索引
        light_indices: light jet 索引
        top_mass: 顶夸克质量
        W_mass: W 质量

    返回:
        assignment: 分配方案
        chi2: 总 χ²
    """
    n_b = len(b_indices)
    n_light = len(light_indices)

    if n_b < 2 or n_light < 2:
        return None, float('inf')

    # 贪心初始化: 按能量排序
    b_sorted = sorted(b_indices, key=lambda i: jets_4vec[i, 0], reverse=True)
    light_sorted = sorted(light_indices, key=lambda i: jets_4vec[i, 0], reverse=True)

    best_assignment = {
        'b_had': b_sorted[0],
        'j1': light_sorted[0],
        'j2': light_sorted[1],
        'b_lep': b_sorted[1],
    }

    # 计算 χ²
    def compute_chi2(asgn):
        p_b = jets_4vec[asgn['b_had']]
        p_j1 = jets_4vec[asgn['j1']]
        p_j2 = jets_4vec[asgn['j2']]

        p_W = p_j1 + p_j2
        M_W = np.sqrt(max(p_W[0]**2 - p_W[1]**2 - p_W[2]**2 - p_W[3]**2, 0))
        p_top = p_b + p_W
        M_top = np.sqrt(max(p_top[0]**2 - p_top[1]**2 - p_top[2]**2 - p_top[3]**2, 0))

        chi2 = ((M_top - top_mass) / 15.0)**2 + ((M_W - W_mass) / 10.0)**2
        return chi2

    chi2 = compute_chi2(best_assignment)

    # 局部优化: 尝试交换
    improved = True
    while improved:
        improved = False
        for key1 in best_assignment:
            for key2 in best_assignment:
                if key1 >= key2:
                    continue
                # 尝试交换
                new_asgn = dict(best_assignment)
                new_asgn[key1], new_asgn[key2] = new_asgn[key2], new_asgn[key1]
                new_chi2 = compute_chi2(new_asgn)
                if new_chi2 < chi2:
                    best_assignment = new_asgn
                    chi2 = new_chi2
                    improved = True

    return best_assignment, chi2


def combinatorial_background_estimate(n_jets, n_btags, signal_fraction=0.3):
    """
    组合背景的估计。

    总配对数:
      N_total = C(n_btags, 2) × C(n_jets - n_btags, 2) × 2

    信号占比 ~ signal_fraction (基于 Monte Carlo)
    背景 = (1 - signal_fraction) × N_total

    参数:
        n_jets: 总喷注数
        n_btags: b-tagged 数
        signal_fraction: 信号占比

    返回:
        (n_total, n_signal, n_background)
    """
    from scipy.special import comb

    if n_btags < 2 or (n_jets - n_btags) < 2:
        return 0, 0, 0

    n_total = comb(n_btags, 2) * comb(n_jets - n_btags, 2) * 2
    n_signal = int(n_total * signal_fraction)
    n_background = int(n_total * (1.0 - signal_fraction))

    return int(n_total), n_signal, n_background
