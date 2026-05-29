"""
 bayesian_sampling.py
 
 融合种子项目:
   - 1376_urn_simulation: 超几何分布 urn 采样与概率质量函数
 
 科学应用:
   在全波形反演中，贝叶斯框架提供了反演结果的不确定性量化方法。
   本模块实现基于超几何分布的先验采样，用于表征地下介质参数空间中的
   离散不确定性（例如岩相分类的不确定性）。通过 urn 模型，我们可以从
   包含不同岩相类别的总体中进行无放回抽样，量化反演结果的统计置信度。
"""

import numpy as np


def ksub_random2(n, k, rng=None):
    """
    从 {1, ..., n} 中随机无放回抽取 k 个不同整数。
    
    这是超几何分布采样的核心步骤。算法采用 Fisher-Yates 洗牌思想，
    但只生成前 k 个元素，时间复杂度 O(k)。
    
    Parameters
    ----------
    n : int
        总体大小。
    k : int
        抽样数量，需满足 0 <= k <= n。
    rng : numpy.random.Generator, optional
        随机数生成器。
    
    Returns
    -------
    y : ndarray, shape (k,)
        抽取的 k 个不同整数（已排序）。
    """
    if rng is None:
        rng = np.random.default_rng()
    if k < 0 or k > n:
        raise ValueError("k must satisfy 0 <= k <= n")
    if k == 0:
        return np.array([], dtype=int)
    # 使用 numpy 的 choice 实现无放回抽样
    y = rng.choice(n, size=k, replace=False)
    return np.sort(y + 1)  # 转换为 1-based 索引


def urn_sample(marble_num, draw_num, color_num, color_count, rng=None):
    """
    模拟从 urn 中抽取彩色弹珠。
    
    物理模型:
      一个 urn 中包含 marble_num 个弹珠，分为 color_num 种颜色，
      第 i 种颜色有 color_count[i] 个弹珠。从中无放回抽取 draw_num 个弹珠，
      返回每种颜色被抽取的个数。
    
    在全波形反演中，此模型对应于：从包含多种岩相类型的总体中抽取观测样本，
    估计各岩相的分布比例。
    
    Parameters
    ----------
    marble_num : int
        弹珠总数。
    draw_num : int
        抽取数量。
    color_num : int
        颜色种类数。
    color_count : ndarray, shape (color_num,)
        每种颜色的弹珠数量。
    rng : numpy.random.Generator, optional
    
    Returns
    -------
    draw_color : ndarray, shape (color_num,)
        每种颜色被抽取的数量。
    """
    if rng is None:
        rng = np.random.default_rng()
    color_count = np.asarray(color_count, dtype=int)
    if np.sum(color_count) != marble_num:
        raise ValueError("sum(color_count) must equal marble_num")
    y = ksub_random2(marble_num, draw_num, rng=rng)
    draw_color = np.zeros(color_num, dtype=int)
    t = 0
    for i in range(color_num):
        b = t
        t = t + color_count[i]
        # 计算在区间 (b, t] 中的 y 的数量
        draw_color[i] = np.sum((b < y) & (y <= t))
    return draw_color


def urn_two_color_pdf(w, draw_num, color_count):
    """
    计算双颜色 urn 问题的概率质量函数（PMF）。
    
    对于超几何分布，抽取 w 个白色弹珠的概率为:
      P(W=w) = C(K, w) * C(N-K, draw_num-w) / C(N, draw_num)
    其中 N = sum(color_count) 为总数，K = color_count[0] 为白色弹珠数。
    
    Parameters
    ----------
    w : ndarray
        待计算概率的白色弹珠数量。
    draw_num : int
        抽取总数。
    color_count : ndarray, shape (2,)
        color_count[0] = 白色弹珠数, color_count[1] = 黑色弹珠数。
    
    Returns
    -------
    pw : ndarray
        对应每个 w 的概率值。
    """
    from scipy.special import comb
    marble_num = np.sum(color_count)
    w = np.asarray(w, dtype=int)
    pw = np.zeros_like(w, dtype=float)
    for i in range(len(w)):
        if w[i] < 0 or w[i] > color_count[0] or (draw_num - w[i]) > color_count[1]:
            pw[i] = 0.0
        else:
            pw[i] = (
                comb(color_count[0], w[i], exact=True) *
                comb(color_count[1], draw_num - w[i], exact=True) /
                comb(marble_num, draw_num, exact=True)
            )
    return pw


def bayesian_posterior_sample(likelihood_func, prior_sampler, n_samples=1000, rng=None):
    """
    使用拒绝采样法进行贝叶斯后验采样。
    
    贝叶斯定理:
      p(m | d) = p(d | m) * p(m) / p(d)
    其中 p(d | m) 为似然函数，p(m) 为先验分布。
    
    Parameters
    ----------
    likelihood_func : callable
        似然函数，输入参数样本，输出似然值。
    prior_sampler : callable
        先验采样函数，返回一个参数样本。
    n_samples : int
        目标后验样本数。
    rng : numpy.random.Generator, optional
    
    Returns
    -------
    samples : list
        后验样本列表。
    acceptance_rate : float
        接受率。
    """
    if rng is None:
        rng = np.random.default_rng()
    samples = []
    trials = 0
    max_trials = n_samples * 100
    # 先进行预采样估计最大似然值
    pre_samples = [prior_sampler() for _ in range(200)]
    pre_likes = [likelihood_func(s) for s in pre_samples]
    max_like = np.max(pre_likes)
    if max_like <= 0:
        max_like = 1.0
    while len(samples) < n_samples and trials < max_trials:
        m = prior_sampler()
        like = likelihood_func(m)
        trials += 1
        u = rng.random()
        if u < like / max_like:
            samples.append(m)
    acceptance_rate = len(samples) / max(trials, 1)
    return samples, acceptance_rate
