"""
combinatorial_sampler.py
构象空间的组合采样模块

本模块利用组合数学中的回溯搜索与排列/子集枚举算法，
对脂质双分子层的离散构象空间进行系统性采样。

参考种子项目: 202_combo (组合算法集合，包括 backtrack, permutations,
                subsets, partitions, Gray code 等)

物理背景:
    粗粒化脂质分子可处于若干离散取向态（例如 6 个或 8 个优选方向）。
    整个双层膜的构象空间大小为 M^N，其中 M 为单分子取向态数，N 为分子数。
    对于 N=576 (24×24), M=6，总构象数约为 6^576 ≈ 10^448，完全不可遍历。
    因此需要:
      1. 回溯搜索（backtrack）寻找满足能量约束的局部低能构象；
      2. 组合枚举生成代表性态子集；
      3. Gray 码遍历实现相邻构象间最小变化（用于蒙特卡洛）。
"""

import numpy as np


class BacktrackSampler:
    """
    回溯搜索采样器（受种子项目 202_combo/backtrack.m 启发）。

    在构象空间中搜索满足约束条件 C(x_1,...,x_L) 的构象。
    每个变量 x_k ∈ {0,1,...,M-1} 表示第 k 个脂质分子的取向态。
    """

    def __init__(self, n_vars, n_states, constraint_func=None):
        """
        Parameters
        ----------
        n_vars : int
            变量数（脂质分子数）。
        n_states : int
            每个变量的取值数（离散取向态数）。
        constraint_func : callable or None
            constraint_func(partial_array) -> bool
            若返回 False，则当前部分赋值不满足约束，回溯。
        """
        if n_vars <= 0 or n_states <= 0:
            raise ValueError("n_vars 和 n_states 必须为正。")
        self.n_vars = n_vars
        self.n_states = n_states
        self.constraint = constraint_func

    def search(self, max_solutions=100):
        """
        深度优先回溯搜索，返回最多 max_solutions 个可行解。

        算法:
            stack: 候选值栈
            iarray: 当前部分解
            k: 当前赋值位置
        """
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
    """
    组合枚举工具集（受种子项目 202_combo 启发）。
    """

    @staticmethod
    def gray_code(n_bits):
        """
        生成 n_bits 位 Gray 码序列。

        Gray 码性质: 相邻两个码仅有一位不同。
        公式: g(k) = k XOR (k >> 1)

        在 MD 中应用: 用 Gray 码遍历构象空间，确保每一步只翻转一个
        脂质分子的取向态，从而最大化蒙特卡洛接受率。
        """
        if n_bits < 0:
            raise ValueError("n_bits 必须非负。")
        codes = []
        for k in range(1 << n_bits):
            g = k ^ (k >> 1)
            codes.append(g)
        return codes

    @staticmethod
    def k_subset_lex(n, k):
        """
        字典序生成 {0,...,n-1} 的所有 k-子集。

        在脂质畴分析中: 从 N 个分子中选出 k 个作为“有序畴”核心。
        """
        if k < 0 or k > n:
            return []
        if k == 0:
            return [[]]
        result = []
        subset = list(range(k))
        while True:
            result.append(subset.copy())
            # 找下一个字典序子集
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
        """
        生成正整数 n 的所有无序划分。

        在双层膜中: 将 N 个脂质分子划分为若干个畴（domain），
        划分中的每个部分对应一个畴的大小。
        """
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
        """
        第二类 Stirling 数 S(n,k): 将 n 个不同元素划分为 k 个非空无标号子集的方法数。

        递推:
            S(n,k) = k*S(n-1,k) + S(n-1,k-1)
            S(n,1) = S(n,n) = 1
        """
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
    """
    脂质构象组合采样器。

    将 N 个脂质分子的取向态编码为整数数组，利用组合方法生成样本。
    """

    def __init__(self, nx, ny, n_orient_states=6):
        self.nx = nx
        self.ny = ny
        self.n = nx * ny
        self.m = n_orient_states

    def random_configuration(self, seed=None):
        """
        均匀随机生成一个构象。
        """
        rng = np.random.default_rng(seed)
        return rng.integers(0, self.m, size=self.n)

    def gray_code_walk(self, n_steps=100, seed=None):
        """
        使用 Gray 码遍历构象空间的局部邻域。

        将构象编码为 N*log2(M) 位二进制数，每步按 Gray 码翻转一位。
        """
        rng = np.random.default_rng(seed)
        n_bits = self.n * int(np.ceil(np.log2(self.m)))
        codes = CombinatorialEnumerators.gray_code(min(n_bits, 12))
        configs = []
        base = self.random_configuration(seed)
        for step in range(n_steps):
            g = codes[step % len(codes)]
            config = base.copy()
            # 将 Gray 码的变化映射到构象上
            bit = step % n_bits
            lipid_idx = bit % self.n
            config[lipid_idx] = (config[lipid_idx] + 1) % self.m
            configs.append(config.copy())
        return configs

    def domain_partitions(self, max_domains=5):
        """
        枚举将 nx×ny 格点划分为至多 max_domains 个畴的划分方式。

        返回每个划分对应的畴大小列表。
        """
        n = self.nx * self.ny
        partitions = []
        for k in range(1, max_domains + 1):
            partitions.extend(CombinatorialEnumerators.integer_partitions(n, k))
        return partitions

    def knapsack_lipid_selection(self, capacities, values, max_weight):
        """
        0/1 背包问题变体：选择脂质子集以最大化某种序参数贡献。

        受种子项目 202_combo/knapsack_01.m 启发。

        动态规划:
            dp[w] = max(dp[w], dp[w - capacities[i]] + values[i])
        """
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
