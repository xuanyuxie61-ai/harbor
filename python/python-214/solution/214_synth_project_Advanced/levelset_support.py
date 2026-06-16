"""
levelset_support.py — 水平集支撑集识别模块
===========================================
来源项目映射:
  - 667_levels → 水平集 (等值面) 计算
    (levels_xy, surface_xy)

科学背景:
  L1 正则化解的支撑集 supp(x) = {i : x_i ≠ 0} 的精确识别
  是稀疏恢复的关键. 当系数 |x_i| 量级差异大时, 硬阈值
  难以选取. 本模块采用水平集方法:
    (1) 将 |x| 视为 1D 离散标量场
    (2) 对水平值 τ 计算等值集 {i : |x_i| = τ}
    (3) 通过扫描 τ 观察支撑集拓扑变化
    (4) 用持续同调选取最优 τ

  这与连续水平集 (667_levels) 思想一致: 通过扫描水平值
  揭示标量场的拓扑结构.

核心公式:
  水平集:
    L_τ = {i ∈ {1,...,M} : |x_i| = τ}
  支撑集:
    S_τ = {i : |x_i| > τ}
  持续同调 (0-维):
     births at τ = |x_{(i)}|  (排序后的系数)
     deaths at τ = |x_{(i+1)}|
  持续性 = death - birth,  大持续性 → 重要非零系数.
"""
import numpy as np


# ----------------------------------------------------------------------
# 离散水平集 (源自 667_levels)
# ----------------------------------------------------------------------
def levels_1d(values, n_levels=20):
    """计算 1D 标量场的水平值序列.

    输入:
        values    : (M,) 标量场
        n_levels  : 水平级数
    返回:
        level_set : (n_levels,) 水平值
    """
    v_min, v_max = np.min(values), np.max(values)
    if v_max - v_min < 1e-14:
        return np.linspace(v_min, v_max, n_levels)
    return np.linspace(v_min, v_max, n_levels)


def level_crossing_indices(values, tau):
    """找出 |x_i| 穿过水平 τ 的指标集.

    返回:
        above : {i : |x_i| > τ}
        at    : {i : ||x_i| - τ| < ε}
    """
    abs_vals = np.abs(values)
    eps = 1e-10 * (np.max(abs_vals) + 1e-14)
    above = set(np.where(abs_vals > tau + eps)[0])
    at = set(np.where(np.abs(abs_vals - tau) <= eps)[0])
    return above, at


def contour_tree_1d(values):
    """构建 1D 标量场的等高线树 (简化版持续同调).

    返回:
        births     : 各连通分支诞生水平
        lifetimes  : 各分支的持续长度
        important  : 重要分支的指标 (按持续性排序)
    """
    abs_vals = np.abs(values)
    sorted_idx = np.argsort(-abs_vals)  # 降序
    sorted_vals = abs_vals[sorted_idx]
    births = sorted_vals[:-1]
    deaths = sorted_vals[1:]
    lifetimes = births - deaths
    # 持续长度排序
    order = np.argsort(-lifetimes)
    return births, lifetimes, sorted_idx[order]


# ----------------------------------------------------------------------
# 支撑集识别
# ----------------------------------------------------------------------
def support_by_threshold(values, tau):
    """硬阈值支撑集: {i : |x_i| > τ}."""
    return set(np.where(np.abs(values) > tau)[0])


def support_by_persistence(values, min_persistence_ratio=0.1):
    """基于持续同调的支撑集识别.

    保留持续性 > min_persistence_ratio · max(|x|) 的分量.
    """
    abs_vals = np.abs(values)
    max_val = np.max(abs_vals) if len(abs_vals) > 0 else 1.0
    threshold = min_persistence_ratio * max_val

    sorted_idx = np.argsort(-abs_vals)
    sorted_vals = abs_vals[sorted_idx]

    support = set()
    for k in range(len(sorted_vals) - 1):
        lifetime = sorted_vals[k] - sorted_vals[k + 1]
        if lifetime >= threshold:
            support.add(sorted_idx[k])
    # 最后一个分量单独处理
    if len(sorted_vals) > 0 and sorted_vals[-1] >= threshold:
        support.add(sorted_idx[-1])
    return support


def adaptive_threshold(values, method='otsu'):
    """自适应阈值选取.

    Otsu 法: 最大化类间方差
        τ* = argmax_τ σ_B²(τ)
    其中 σ_B² = n_1 n_2 (μ_1 - μ_2)² / (n_1 + n_2)²
    """
    abs_vals = np.abs(values)
    if method == 'otsu':
        sorted_v = np.sort(abs_vals)
        n = len(sorted_v)
        best_tau = 0.0
        best_var = -1.0
        for tau in sorted_v[::max(1, n // 100)]:
            below = abs_vals[abs_vals <= tau]
            above = abs_vals[abs_vals > tau]
            if len(below) == 0 or len(above) == 0:
                continue
            n1, n2 = len(below), len(above)
            mu1, mu2 = np.mean(below), np.mean(above)
            var_between = (n1 * n2 * (mu1 - mu2) ** 2) / ((n1 + n2) ** 2)
            if var_between > best_var:
                best_var = var_between
                best_tau = tau
        return float(best_tau)
    elif method == 'median_abs_deviation':
        # MAD: τ = median(|x|) + 1.4826 · MAD
        med = np.median(abs_vals)
        mad = np.median(np.abs(abs_vals - med))
        return float(med + 1.4826 * mad)
    else:
        raise ValueError(f"未知方法: {method}")


# ----------------------------------------------------------------------
# 水平集演化监控
# ----------------------------------------------------------------------
def track_support_evolution(values, n_levels=50):
    """扫描水平值 τ 观察支撑集变化.

    返回:
        taus     : 水平值序列
        sizes    : |S_τ| 随 τ 变化
        new_adds : 每个水平新增的元素数
    """
    abs_vals = np.abs(values)
    taus = np.linspace(0, np.max(abs_vals) * 1.01, n_levels)
    sizes = []
    new_adds = []
    prev_sup = set()
    for tau in taus:
        sup = support_by_threshold(values, tau)
        sizes.append(len(sup))
        new_adds.append(len(sup - prev_sup))
        prev_sup = sup
    return taus, sizes, new_adds


def stability_estimate(values, noise_level, n_trials=100, seed=0):
    """支撑集稳定性估计.

    扰动 x → x + ε,  ε ~ N(0, σ²), 观察支撑集变化频率.
    返回: 各指标的稳定概率 (越高越稳定).
    """
    rng = np.random.RandomState(seed)
    M = len(values)
    counts = np.zeros(M)
    tau = adaptive_threshold(values)
    for _ in range(n_trials):
        perturbed = values + noise_level * rng.randn(M)
        sup = support_by_threshold(perturbed, tau)
        for i in sup:
            counts[i] += 1
    return counts / n_trials
