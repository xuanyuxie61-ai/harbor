"""
combinatorial_stats.py
组合统计与构型枚举模块

基于种子项目 1273_toms515 的核心算法：
- comb: 按字典序索引生成组合
- binom: 二项式系数
- gamma_log_values: Gamma 函数对数值

在离子通道问题中的应用：
- 组合计数用于统计离子在多个结合位点上的占据构型数
- 二项式系数用于系综平均中的状态计数
- Gamma 函数用于统计力学配分函数的解析延拓
"""

import numpy as np


def binomial_coefficient(n, k):
    """
    计算二项式系数 C(n,k)（源自 binom.m）。

    公式：
        C(n,k) = n! / (k! (n-k)!)

    边界处理：若 k<0 或 k>n，返回 0。
    """
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    # 利用对称性减小计算量
    k = min(k, n - k)
    result = 1
    for i in range(1, k + 1):
        result = result * (n - k + i) // i
    return result


def combination_lex_index(n, p, l):
    """
    按字典序索引 l 生成组合 C(n,p) 中的第 l 个组合（源自 comb.m）。

    Parameters
    ----------
    n : int
        集合大小
    p : int
        子集大小
    l : int
        字典序索引（1-based）

    Returns
    -------
    c : ndarray
        组合元素（1-based 索引）
    """
    if p <= 0 or p > n:
        raise ValueError("p 必须在 (0, n] 范围内")
    if l < 1 or l > binomial_coefficient(n, p):
        raise ValueError("索引 l 超出范围")

    c = np.zeros(p, dtype=int)
    if p == 1:
        c[0] = l
        return c

    k = 0
    p1 = p - 1
    c[0] = 0

    for i in range(p1):
        if i > 0:
            c[i] = c[i - 1]
        while True:
            c[i] += 1
            r = binomial_coefficient(n - c[i], p - i - 1)
            k += r
            if l <= k:
                break
        k -= r

    c[p - 1] = c[p1 - 1] + l - k
    return c


def enumerate_occupations(n_sites, n_ions):
    """
    枚举 n_sites 个结合位点被 n_ions 个离子占据的所有构型。

    返回所有组合（0-based 索引列表）。
    """
    total = binomial_coefficient(n_sites, n_ions)
    configs = []
    for l in range(1, total + 1):
        c = combination_lex_index(n_sites, n_ions, l)
        configs.append(c - 1)  # 转为 0-based
    return configs


def multinomial_coefficient(n, groups):
    """
    多项式系数：
        (n; n1, n2, ..., nk) = n! / (n1! n2! ... nk!)

    用于多物种离子（K+, Na+）在结合位点上的分布计数。
    """
    if sum(groups) != n:
        return 0
    result = np.math.factorial(n)
    for g in groups:
        result //= np.math.factorial(g)
    return result


def gamma_log_table(n_values):
    """
    预计算 ln Γ(n) 的查找表（源自 gamma_log_values.m）。

    用于配分函数的数值稳定性计算：
        Z = Σ_i exp(-E_i/kT) 中的大数处理
    """
    from special_functions import log_gamma_lanczos
    return np.array([log_gamma_lanczos(v) for v in n_values])


def canonical_partition_function(energies, T=300.0):
    """
    正则系综配分函数：
        Z = Σ_i exp(-β E_i)

    采用 log-sum-exp 技巧保证数值稳定性：
        ln Z = E_min + ln Σ_i exp(-β(E_i - E_min))
    """
    kB = 1.380649e-23
    beta = 1.0 / (kB * T)
    e_min = np.min(energies)
    z = np.sum(np.exp(-beta * (energies - e_min)))
    return e_min + np.log(z)


def occupancy_probability(n_sites, n_ions, energy_func, T=300.0):
    """
    计算每个位点被占据的概率（基于正则系综平均）。

    P(位点 j 被占据) = Σ_{包含 j 的构型} exp(-β E_conf) / Z
    """
    configs = enumerate_occupations(n_sites, n_ions)
    kB = 1.380649e-23
    beta = 1.0 / (kB * T)

    weights = []
    for conf in configs:
        e = energy_func(conf)
        weights.append(np.exp(-beta * e))
    weights = np.array(weights)
    Z = np.sum(weights)

    probs = np.zeros(n_sites)
    for idx, conf in enumerate(configs):
        for site in conf:
            probs[site] += weights[idx]
    probs /= Z
    return probs
