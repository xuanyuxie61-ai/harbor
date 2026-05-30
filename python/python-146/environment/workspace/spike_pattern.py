
import numpy as np





_POLYOMINO_FIXED_COUNTS = [
    1, 1, 2, 6, 19, 63,
    216, 760, 2725, 9910, 36446,
    135268, 505861, 1903890, 7204874, 27394666,
    104592937, 400795844, 1540820542, 5940738676, 22964779660,
    88983512783, 345532572678, 1344372335524, 5239988770268, 20457802016011,
    79992676367108, 313224032098244, 1228088671826973
]


def polyomino_enumerate_fixed(order):
    if not (0 <= order <= 28):
        raise ValueError("order must be between 0 and 28.")
    return _POLYOMINO_FIXED_COUNTS[order]


def connected_spike_patterns_1d(n_bins):
    if n_bins < 0:
        return 0
    return n_bins * (n_bins + 1) // 2


def connected_spike_patterns_2d(nx, ny, max_order=10):
    results = []
    for m in range(1, min(max_order, 28) + 1):

        count = polyomino_enumerate_fixed(m)
        results.append((m, count))
    return results





def r8col_sorted_tol_unique(patterns, tol=0.0):
    patterns = np.asarray(patterns, dtype=float)
    if patterns.ndim == 1:
        patterns = patterns.reshape(-1, 1)
    m, n = patterns.shape
    if n <= 0:
        return np.zeros((m, 0)), 0


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
    return np.sum(np.abs(p1 - p2))


def pattern_clustering(patterns, max_distance=2):
    patterns = np.asarray(patterns, dtype=float)
    n = patterns.shape[1] if patterns.ndim > 1 else len(patterns)
    if patterns.ndim == 1:
        patterns = patterns.reshape(1, -1)

    labels = -np.ones(n, dtype=int)
    current_label = 0
    for i in range(n):
        if labels[i] >= 0:
            continue

        labels[i] = current_label

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

    def __init__(self, n_bins, pattern_binwidth=1.0):
        self.n_bins = n_bins
        self.pattern_binwidth = pattern_binwidth

    def encode_spike_train(self, spike_times, T_window):
        pattern = np.zeros(self.n_bins, dtype=int)
        bin_edges = np.linspace(0, T_window, self.n_bins + 1)
        for t in spike_times:
            if 0 <= t < T_window:
                idx = int(np.floor(t / self.pattern_binwidth)) % self.n_bins
                pattern[idx] = 1
        return pattern

    def pattern_entropy(self, patterns):
        patterns = np.asarray(patterns, dtype=float)
        if patterns.ndim == 1:
            patterns = patterns.reshape(-1, 1)
        n = patterns.shape[1]
        if n == 0:
            return 0.0


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
        n = connected_spike_patterns_1d(self.n_bins)
        if n <= 0:
            return 0.0
        return np.log2(n)


def demo_pattern_analysis():
    analyzer = SpikePatternAnalyzer(n_bins=8)

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

    max_order = 9
    counts = connected_spike_patterns_2d(3, 3, max_order=max_order)
    return counts
