
import numpy as np


class BacktrackSampler:

    def __init__(self, n_vars, n_states, constraint_func=None):
        if n_vars <= 0 or n_states <= 0:
            raise ValueError("n_vars 和 n_states 必须为正。")
        self.n_vars = n_vars
        self.n_states = n_states
        self.constraint = constraint_func

    def search(self, max_solutions=100):
        solutions = []
        iarray = np.zeros(self.n_vars, dtype=int)
        k = 0

        def backtrack_dfs(pos):
            if len(solutions) >= max_solutions:
                return
            if pos == self.n_vars:
                solutions.append(iarray.copy())
                return
            for val in range(self.n_states):
                iarray[pos] = val
                if self.constraint is not None:
                    if not self.constraint(iarray[:pos + 1]):
                        continue
                backtrack_dfs(pos + 1)

        backtrack_dfs(0)
        return solutions


class CombinatorialEnumerators:

    @staticmethod
    def gray_code(n_bits):
        if n_bits < 0:
            raise ValueError("n_bits 必须非负。")
        codes = []
        for k in range(1 << n_bits):
            g = k ^ (k >> 1)
            codes.append(g)
        return codes

    @staticmethod
    def k_subset_lex(n, k):
        if k < 0 or k > n:
            return []
        if k == 0:
            return [[]]
        result = []
        subset = list(range(k))
        while True:
            result.append(subset.copy())

            i = k - 1
            while i >= 0 and subset[i] == i + n - k:
                i -= 1
            if i < 0:
                break
            subset[i] += 1
            for j in range(i + 1, k):
                subset[j] = subset[j - 1] + 1
        return result

    @staticmethod
    def integer_partitions(n, max_part=None):
        if max_part is None:
            max_part = n
        if n == 0:
            return [[]]
        result = []

        def helper(remaining, max_p, current):
            if remaining == 0:
                result.append(current.copy())
                return
            for p in range(min(remaining, max_p), 0, -1):
                current.append(p)
                helper(remaining - p, p, current)
                current.pop()

        helper(n, max_part, [])
        return result

    @staticmethod
    def stirling_second(n, k):
        if n < 0 or k < 0:
            return 0
        if k == 0:
            return 1 if n == 0 else 0
        if k > n:
            return 0
        S = np.zeros((n + 1, k + 1), dtype=object)
        S[0, 0] = 1
        for i in range(1, n + 1):
            for j in range(1, min(i, k) + 1):
                S[i, j] = j * S[i - 1, j] + S[i - 1, j - 1]
        return int(S[n, k])


class ConfigurationSampler:

    def __init__(self, nx, ny, n_orient_states=6):
        self.nx = nx
        self.ny = ny
        self.n = nx * ny
        self.m = n_orient_states

    def random_configuration(self, seed=None):
        rng = np.random.default_rng(seed)
        return rng.integers(0, self.m, size=self.n)

    def gray_code_walk(self, n_steps=100, seed=None):
        rng = np.random.default_rng(seed)
        n_bits = self.n * int(np.ceil(np.log2(self.m)))
        codes = CombinatorialEnumerators.gray_code(min(n_bits, 12))
        configs = []
        base = self.random_configuration(seed)
        for step in range(n_steps):
            g = codes[step % len(codes)]
            config = base.copy()

            bit = step % n_bits
            lipid_idx = bit % self.n
            config[lipid_idx] = (config[lipid_idx] + 1) % self.m
            configs.append(config.copy())
        return configs

    def domain_partitions(self, max_domains=5):
        n = self.nx * self.ny
        partitions = []
        for k in range(1, max_domains + 1):
            partitions.extend(CombinatorialEnumerators.integer_partitions(n, k))
        return partitions

    def knapsack_lipid_selection(self, capacities, values, max_weight):
        capacities = np.asarray(capacities, dtype=int)
        values = np.asarray(values, dtype=float)
        n = len(capacities)
        dp = np.zeros(max_weight + 1)
        choice = np.full((max_weight + 1, n), False)

        for i in range(n):
            for w in range(max_weight, capacities[i] - 1, -1):
                if dp[w - capacities[i]] + values[i] > dp[w]:
                    dp[w] = dp[w - capacities[i]] + values[i]
                    if w - capacities[i] >= 0:
                        choice[w, :] = choice[w - capacities[i], :]
                    choice[w, i] = True

        best_w = int(np.argmax(dp))
        return best_w, dp[best_w], choice[best_w]
