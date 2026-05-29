"""
stoichiometric_polynomial.py
============================
基于多项式乘法的化学计量组合计数模块。

核心算法源自 change_polynomial (Project 158)，并改造用于
计算化学反应网络中的可能化学计量组合数。

原始问题（找零钱问题）：
    给定硬币面值 value = [v₁, v₂, ..., v_m]，
    求恰好使用 coin_num 枚硬币凑成和为 sum 的方案数。

在燃烧化学中，这对应于：给定一组基元反应（化学计量系数），
计算不同反应路径的组合数。例如，对于甲烷燃烧机理：

    CH4 + O2 → CH3 + HO2          (反应1)
    CH4 + H → CH3 + H2            (反应2)
    CH3 + O → CH2O + H            (反应3)
    ...

从 CH4 到 CO2 的转化路径可视为一个组合计数问题。

生成函数方法：
    对于每个反应 i，其化学计量变化向量为 Δn_i，
    生成函数为 P_i(x) = x^{Δn_i}。

    总生成函数为各反应生成函数的乘积：
        P_total(x) = Π_i P_i(x)^{m_i}

    其中 m_i 为反应 i 发生的次数。

    x^S 的系数即为从初始组分转化为目标组分（净变化为 S）的方案数。

本模块实现：
1. 多项式乘法（离散卷积）；
2. 化学计量路径计数；
3. 反应机理的生成函数分析。
"""

import numpy as np


def polynomial_multiply(a, b):
    """
    计算两个多项式的乘积（离散卷积）。

    公式：
        c[k] = Σ_{i=0}^{k} a[i] * b[k-i]

    Parameters
    ----------
    a, b : ndarray
        多项式系数向量。

    Returns
    -------
    c : ndarray
        乘积多项式系数。
    """
    a = np.asarray(a)
    b = np.asarray(b)
    na, nb = len(a), len(b)
    c = np.zeros(na + nb - 1)

    for i in range(na):
        for j in range(nb):
            c[i + j] += a[i] * b[j]

    return c


def stoichiometric_paths(reaction_steps, target_change, max_steps):
    """
    计算达到目标化学计量变化的路径数。

    Parameters
    ----------
    reaction_steps : list of int
        各反应的化学计量变化量（正整数列表）。
    target_change : int
        目标总变化量。
    max_steps : int
        最大反应步数。

    Returns
    -------
    count : int
        路径数。
    polynomial : ndarray
        生成函数多项式系数。
    """
    if target_change < 0 or max_steps <= 0:
        return 0, np.array([1.0])

    # 初始化：0步时变化为0的方案数为1
    p = np.zeros(target_change + 1)
    p[0] = 1.0

    # 每步的生成函数
    step_poly = np.zeros(max(reaction_steps) + 1)
    for s in reaction_steps:
        if 0 <= s <= max(reaction_steps):
            step_poly[s] += 1.0

    # 迭代乘法
    current = p.copy()
    for _ in range(max_steps):
        current = polynomial_multiply(current, step_poly)
        if len(current) > target_change:
            current = current[:target_change + 1]
        else:
            # 补齐
            temp = np.zeros(target_change + 1)
            temp[:len(current)] = current
            current = temp

    count = int(round(current[target_change]))
    return count, current


def reaction_mechanism_complexity(reactions, max_depth=10):
    """
    评估反应机理的复杂度（路径分支数）。

    Parameters
    ----------
    reactions : list of dict
        每个反应为 {'name': str, 'stoich_change': int, 'rate': float}。
    max_depth : int
        最大搜索深度。

    Returns
    -------
    complexity : dict
        复杂度分析结果。
    """
    steps = [r['stoich_change'] for r in reactions]
    rates = [r['rate'] for r in reactions]

    # 计算不同目标变化下的路径数
    max_change = max(steps) * max_depth
    total_paths = 0
    path_distribution = []

    for target in range(1, max_change + 1):
        count, _ = stoichiometric_paths(steps, target, max_depth)
        total_paths += count
        path_distribution.append((target, count))

    # 有效反应速率加权的平均路径长度
    avg_rate = np.mean(rates) if rates else 1.0
    effective_complexity = total_paths * avg_rate

    complexity = {
        'num_reactions': len(reactions),
        'max_depth': max_depth,
        'total_paths': total_paths,
        'average_rate': avg_rate,
        'effective_complexity': effective_complexity,
        'path_distribution': path_distribution[:20],
    }

    return complexity
