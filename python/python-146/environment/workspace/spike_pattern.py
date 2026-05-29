"""
spike_pattern.py
脉冲模式组合分析模块

融合 polyominoes (多格拼板组合枚举)
与 r8col (数值列向量排序去重 / 聚类)。

核心科学模型：
  脉冲时间编码的组合分析：
    给定时间窗 [0, T] 划分为 N_bin 个等宽 bin，
    脉冲模式为二进制向量 p in {0,1}^{N_bin}。
    总模式数 = 2^{N_bin}。

  连通模式计数 (类比 polyominoes):
    两个脉冲若在相邻 bin 则视为"连通"。
    连通模式数 = 所有 1 的位置在 bin 索引上形成连通集的模式数。
    该问题等价于一维"polyomino"枚举：
      C(N) = sum_{k=1}^{N} C(N, k)
    其中 C(N,k) 为在 N 个 bin 中放置 k 个连通 1 的方式数。
    在一维情况下，k 个连通的 1 形成长度为 k 的块，
    可在 N-k+1 个位置放置，故:
      C(N) = sum_{k=1}^{N} (N - k + 1) = N(N+1)/2

  高维推广 (二维感受野脉冲模式):
    将 N_bin = n_x * n_y 排列为二维网格，
    连通模式类比二维 polyominoes，计数随阶数指数增长。
    固定阶数 m 的连通模式数 = polyominoes_fixed(m)。

  脉冲模式相似性与去重 (r8col 思想):
    将每个脉冲模式视为 R^N 中的列向量，
    使用排序与容差去重识别等价模式类。
    距离度量: d(p1, p2) = sum |p1_i - p2_i| (汉明距离)
    若 d <= tol，视为同一模式类。
"""

import numpy as np


# ------------------------------------------------------------------
# polyominoes 固定多格拼板枚举 (直接移植核心数据)
# ------------------------------------------------------------------
_POLYOMINO_FIXED_COUNTS = [
    1, 1, 2, 6, 19, 63,
    216, 760, 2725, 9910, 36446,
    135268, 505861, 1903890, 7204874, 27394666,
    104592937, 400795844, 1540820542, 5940738676, 22964779660,
    88983512783, 345532572678, 1344372335524, 5239988770268, 20457802016011,
    79992676367108, 313224032098244, 1228088671826973
]


def polyomino_enumerate_fixed(order):
    """
    返回固定多格拼板 (fixed polyomino) 的计数。
    order: 0 <= order <= 28
    融合 polyomino_enumerate_fixed 的核心数据。
    """
    if not (0 <= order <= 28):
        raise ValueError("order must be between 0 and 28.")
    return _POLYOMINO_FIXED_COUNTS[order]


def connected_spike_patterns_1d(n_bins):
    """
    一维连通脉冲模式数。
    在 n_bins 个时间 bin 中，所有 1 形成单个连通块的模式数。
    理论公式: n_bins * (n_bins + 1) / 2
    """
    if n_bins < 0:
        return 0
    return n_bins * (n_bins + 1) // 2


def connected_spike_patterns_2d(nx, ny, max_order=10):
    """
    二维感受野连通脉冲模式数 (类比 polyominoes)。
    返回阶数 1..max_order 的连通模式数列表。
    """
    results = []
    for m in range(1, min(max_order, 28) + 1):
        # 使用 polyomino 计数作为上界
        count = polyomino_enumerate_fixed(m)
        results.append((m, count))
    return results


# ------------------------------------------------------------------
# r8col 排序去重思想用于脉冲模式聚类
# ------------------------------------------------------------------
def r8col_sorted_tol_unique(patterns, tol=0.0):
    """
    对脉冲模式矩阵进行排序与容差去重。
    融合 r8col_sorted_tol_unique 的核心算法。

    参数:
      patterns: (M, N) 矩阵，每列是一个 M 维脉冲模式
      tol: 容差阈值，max(|a[:,i] - a[:,j]|) <= tol 视为相同

    返回:
      unique_patterns: 去重后的模式矩阵
      unique_num: 唯一模式数
    """
    patterns = np.asarray(patterns, dtype=float)
    if patterns.ndim == 1:
        patterns = patterns.reshape(-1, 1)
    m, n = patterns.shape
    if n <= 0:
        return np.zeros((m, 0)), 0

    # 按第一行排序 (简化排序策略)
    order = np.lexsort(patterns)
    a_sorted = patterns[:, order]

    unique_num = 1
    for i in range(1, n):
        is_unique = True
        for j in range(unique_num):
            if np.max(np.abs(a_sorted[:, j] - a_sorted[:, i])) <= tol:
                is_unique = False
                break
        if is_unique:
            unique_num += 1
            a_sorted[:, unique_num - 1] = a_sorted[:, i]

    return a_sorted[:, :unique_num], unique_num


def hamming_distance(p1, p2):
    """二进制脉冲模式间的汉明距离。"""
    return np.sum(np.abs(p1 - p2))


def pattern_clustering(patterns, max_distance=2):
    """
    基于汉明距离的脉冲模式聚类。
    返回每个模式的簇标签。
    """
    patterns = np.asarray(patterns, dtype=float)
    n = patterns.shape[1] if patterns.ndim > 1 else len(patterns)
    if patterns.ndim == 1:
        patterns = patterns.reshape(1, -1)

    labels = -np.ones(n, dtype=int)
    current_label = 0
    for i in range(n):
        if labels[i] >= 0:
            continue
        # 新建簇
        labels[i] = current_label
        # 贪婪扩展
        cluster_members = [i]
        changed = True
        while changed:
            changed = False
            for j in range(n):
                if labels[j] >= 0:
                    continue
                for member in cluster_members:
                    if hamming_distance(patterns[:, j], patterns[:, member]) <= max_distance:
                        labels[j] = current_label
                        cluster_members.append(j)
                        changed = True
                        break
        current_label += 1
    return labels


class SpikePatternAnalyzer:
    """
    脉冲模式分析器：组合计数、去重、聚类。
    """

    def __init__(self, n_bins, pattern_binwidth=1.0):
        self.n_bins = n_bins
        self.pattern_binwidth = pattern_binwidth

    def encode_spike_train(self, spike_times, T_window):
        """
        将脉冲序列编码为二进制模式向量。
        spike_times: 脉冲时刻列表
        T_window: 时间窗长度
        """
        pattern = np.zeros(self.n_bins, dtype=int)
        bin_edges = np.linspace(0, T_window, self.n_bins + 1)
        for t in spike_times:
            if 0 <= t < T_window:
                idx = int(np.floor(t / self.pattern_binwidth)) % self.n_bins
                pattern[idx] = 1
        return pattern

    def pattern_entropy(self, patterns):
        """
        计算脉冲模式的经验香农熵。
        H = - sum p_i log2(p_i)
        """
        patterns = np.asarray(patterns, dtype=float)
        if patterns.ndim == 1:
            patterns = patterns.reshape(-1, 1)
        n = patterns.shape[1]
        if n == 0:
            return 0.0

        # 去重计数
        unique_patterns, unique_num = r8col_sorted_tol_unique(patterns, tol=0.0)
        counts = np.zeros(unique_num)
        for i in range(unique_num):
            for j in range(n):
                if np.array_equal(unique_patterns[:, i], patterns[:, j]):
                    counts[i] += 1

        probs = counts / n
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs))
        return entropy

    def pattern_capacity(self):
        """
        计算一维时间编码的连通模式容量 (bits)。
        capacity = log2(connected_spike_patterns_1d(n_bins))
        """
        n = connected_spike_patterns_1d(self.n_bins)
        if n <= 0:
            return 0.0
        return np.log2(n)


def demo_pattern_analysis():
    """脉冲模式分析 demo。"""
    analyzer = SpikePatternAnalyzer(n_bins=8)
    # 生成随机脉冲模式
    np.random.seed(3)
    patterns = []
    for _ in range(50):
        p = np.random.randint(0, 2, size=8)
        patterns.append(p)
    patterns = np.column_stack(patterns)
    entropy = analyzer.pattern_entropy(patterns)
    labels = pattern_clustering(patterns, max_distance=1)
    capacity = analyzer.pattern_capacity()
    return entropy, labels, capacity


def demo_polyomino_mapping():
    """多格拼板映射 demo：将二维 3x3 感受野脉冲模式映射为多格拼板计数。"""
    # 3x3 网格，最大阶数 9
    max_order = 9
    counts = connected_spike_patterns_2d(3, 3, max_order=max_order)
    return counts
