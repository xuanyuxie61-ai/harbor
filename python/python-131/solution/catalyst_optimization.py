"""
catalyst_optimization.py
========================
基于 623_knapsack_brute 与 289_diophantine_nd 改造的催化剂分布优化模块。

在浆态床气泡柱反应器中，催化剂颗粒的装载量直接影响反应速率、
传热效率与压降。本模块解决两个优化问题：
1. 连续型背包问题：在给定总催化剂质量（重量上限）下，
   将不同活性等级的催化剂分配到反应器各轴向段，使总转化率达到最大。
2. 离散型 Diophantine 整数规划：各段催化剂颗粒数为整数，
   求满足总颗粒数约束且使目标函数最优的整数解。

核心公式
--------
1. 背包问题（连续）：
       max Σ v_i x_i
       s.t. Σ w_i x_i ≤ K,   0 ≤ x_i ≤ 1
   其中 v_i 为第 i 段催化剂的反应价值（基于转化率增量），
         w_i 为第 i 段所需催化剂质量，
         x_i ∈ {0,1} 为选择变量（0-1 背包）。

2. Fischer-Tropsch 转化率增量模型：
       ΔX_i = 1 - exp(-k_i W_i / Q)
       v_i = ΔX_i · Y_{C5+} · ρ_{wax}
       k_i = k_0 exp(-E_a / (R T_i))
   其中 W_i 为第 i 段催化剂质量，Q 为气体体积流率。

3. Diophantine 整数约束：
       a_1 x_1 + a_2 x_2 + ... + a_n x_n = b
       0 ≤ x_i ≤ m_i,   x_i ∈ ℤ
   其中 a_i 为单颗粒催化剂在第 i 段的活性当量，
         b 为总活性当量预算，
         m_i 为第 i 段最大可容纳颗粒数（受空间约束）。

4. 搜索算法：
   - 对低维问题（n≤20）采用 brute-force 枚举所有子集。
   - 对高维整数约束采用回溯法枚举 Diophantine 解。
"""

import numpy as np


# ---------------------------------------------------------------------------
# Brute force knapsack (from 623_knapsack_brute)
# ---------------------------------------------------------------------------

def knapsack_brute_force(values, weights, capacity):
    """
    0-1 背包问题的暴力枚举解法（适用于 n ≤ 20）。

    Parameters
    ----------
    values : ndarray, shape (n,)
        各物品价值。
    weights : ndarray, shape (n,)
        各物品重量。
    capacity : float
        背包容量。

    Returns
    -------
    vmax : float
        最大总价值。
    wmax : float
        对应总重量。
    smax : ndarray, shape (n,)
        最优选择向量（0/1）。
    """
    n = len(values)
    vmax = 0.0
    wmax = 0.0
    smax = np.zeros(n, dtype=int)

    # 枚举所有 2^n 个子集
    total_subsets = 1 << n
    for s in range(total_subsets):
        selection = np.array([(s >> i) & 1 for i in range(n)], dtype=int)
        w_test = np.dot(selection, weights)
        if w_test <= capacity:
            v_test = np.dot(selection, values)
            if v_test > vmax:
                vmax = v_test
                wmax = w_test
                smax = selection.copy()

    return vmax, wmax, smax


def subset_next(s):
    """
    生成下一个 0-1 子集（格雷码风格递增）。
    """
    s = np.asarray(s, dtype=int).copy()
    n = len(s)
    for i in range(n):
        if s[i] == 0:
            s[i] = 1
            return s
        else:
            s[i] = 0
    return s


# ---------------------------------------------------------------------------
# Diophantine integer optimization (from 289_diophantine_nd)
# ---------------------------------------------------------------------------

def diophantine_bounded_solutions(a, b, m):
    """
    求有界非负 Diophantine 方程 a·x = b 的所有整数解。

    Parameters
    ----------
    a : ndarray, shape (n,)
        正整数系数。
    b : int
        右端项（非负）。
    m : ndarray, shape (n,)
        各变量上界。

    Returns
    -------
    solutions : ndarray, shape (k, n)
        所有满足条件的整数解。
    """
    a = np.asarray(a, dtype=int)
    m = np.asarray(m, dtype=int)
    n = len(a)
    solutions = []
    y = np.zeros(n, dtype=int)
    j = 0

    while True:
        r = b - np.dot(a[:j], y[:j])
        if j < n:
            j += 1
            y[j - 1] = min(r // a[j - 1], m[j - 1])
        else:
            if r == 0:
                solutions.append(y.copy())
            # 回溯
            while j > 0:
                if y[j - 1] > 0:
                    y[j - 1] -= 1
                    break
                j -= 1
            if j == 0:
                break

    if not solutions:
        return np.empty((0, n), dtype=int)
    return np.array(solutions, dtype=int)


# ---------------------------------------------------------------------------
# Reactor catalyst optimization
# ---------------------------------------------------------------------------

def catalyst_value_per_segment(W_cat, T_segment, Q_gas,
                                k0=1.2e-3, Ea=85000.0, R=8.314,
                                yield_heavy=0.75, rho_wax=780.0):
    """
    计算各反应器段的催化剂价值（转化率增量 × 产物收益）。

    Parameters
    ----------
    W_cat : ndarray
        各段催化剂质量 [kg]。
    T_segment : ndarray
        各段温度 [K]。
    Q_gas : float
        气体体积流率 [m³/s]。
    k0, Ea, R : float
        阿伦尼乌斯参数。
    yield_heavy : float
        C5+ 产物选择性。
    rho_wax : float
        蜡密度 [kg/m³]。

    Returns
    -------
    values : ndarray
        各段价值 [$/段]（以产物质量计）。
    weights : ndarray
        各段重量 = W_cat。
    """
    W_cat = np.asarray(W_cat, dtype=float)
    T_segment = np.asarray(T_segment, dtype=float)

    # 阿伦尼乌斯反应速率常数
    k = k0 * np.exp(-Ea / (R * T_segment))

    # 转化率（简化一级反应模型）
    X = 1.0 - np.exp(-k * W_cat / max(Q_gas, 1e-12))
    X = np.clip(X, 0.0, 0.999)

    # 价值 = 转化的合成气质量 × 选择性 × 产物密度（简化经济价值）
    values = X * yield_heavy * rho_wax * W_cat
    weights = W_cat.copy()
    return values, weights


def optimize_catalyst_loading(W_total, n_segments, T_profile,
                              Q_gas=0.01, method='brute_force'):
    """
    优化催化剂在反应器各段的分布。

    Parameters
    ----------
    W_total : float
        总催化剂质量 [kg]。
    n_segments : int
        反应器段数。
    T_profile : ndarray
        各段温度 [K]。
    Q_gas : float
        气体流率。
    method : str
        'brute_force' 或 'diophantine'。

    Returns
    -------
    result : dict
    """
    if n_segments <= 0:
        raise ValueError("n_segments must be positive")

    # 离散化候选质量
    n_candidates = min(n_segments * 2, 20)
    W_candidates = np.linspace(0, W_total, n_candidates)

    if method == 'brute_force':
        values, weights = catalyst_value_per_segment(
            W_candidates, np.interp(np.linspace(0, 1, n_candidates),
                                    np.linspace(0, 1, len(T_profile)), T_profile),
            Q_gas)
        vmax, wmax, smax = knapsack_brute_force(values, weights, W_total)
        selected_W = W_candidates[smax == 1]
        return {
            'method': 'brute_force',
            'max_value': vmax,
            'total_weight': wmax,
            'selection': smax,
            'selected_weights': selected_W,
        }

    elif method == 'diophantine':
        # 将总质量离散为颗粒数
        m_particle = 0.1  # kg/颗粒
        N_total = int(W_total / m_particle)
        a = np.ones(n_segments, dtype=int)
        m = np.full(n_segments, N_total, dtype=int)
        solutions = diophantine_bounded_solutions(a, N_total, m)

        if solutions.shape[0] == 0:
            return {
                'method': 'diophantine',
                'max_value': 0.0,
                'best_solution': np.zeros(n_segments, dtype=int),
            }

        best_val = -1.0
        best_sol = solutions[0]
        for sol in solutions:
            W_seg = sol * m_particle
            vals, _ = catalyst_value_per_segment(W_seg, T_profile, Q_gas)
            total_val = np.sum(vals)
            if total_val > best_val:
                best_val = total_val
                best_sol = sol.copy()

        return {
            'method': 'diophantine',
            'max_value': best_val,
            'best_solution': best_sol,
            'particle_mass': m_particle,
        }
    else:
        raise ValueError(f"Unknown method: {method}")
