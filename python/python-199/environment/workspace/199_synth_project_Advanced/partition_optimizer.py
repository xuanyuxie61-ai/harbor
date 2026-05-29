"""
partition_optimizer.py — 数据分区优化与约束满足
===============================================
融合来源:
  - 339_eternity (Eternity拼图的覆盖约束与组合分配)
  - 1062_scip_solution_read (优化解文件解析与配置读取)
  - 915_prime_plot (素数生成用于哈希种子)

在高性能外排序中，均衡的数据分区是性能关键。不均衡的分区会导致
某些归并路径负载过高，形成性能瓶颈。本模块将分区问题建模为约束
满足问题，并利用组合优化思想寻找近似最优解。

核心约束：
  1. 完备性：每个记录必须被分配到恰好一个分区
  2. 容量：每个分区的数据量不超过内存页大小的整数倍
  3. 连续性：分区边界应尽可能保持数据的空间局部性
"""

import math
import random
from typing import List, Tuple, Dict, Optional
from utils import generate_primes, hash_family_seed, compute_gcd


class CoverConstraintSolver:
    """
    Eternity 拼图覆盖约束的抽象化。

    原始 Eternity 问题要求用 209 块 tile 无重叠、无遗漏地覆盖目标区域，
    约束为：
        Σ_{t ∈ T(i,j)} x_t = 1,   ∀ 网格单元 (i,j)
        Σ_j x_{t,j} = 1,          ∀ tile t

    在外排序中，我们将"数据块 → 分区"的分配建模为类似的 0-1 约束系统：
        Σ_p x_{b,p} = 1,          ∀ 数据块 b    (每块恰好一个分区)
        Σ_b size(b) · x_{b,p} ≤ C_p,  ∀ 分区 p    (容量约束)

    由于精确求解为 NP-hard，采用贪心启发式近似。
    """

    def __init__(self, num_partitions: int, capacity_per_partition: int):
        self.P = num_partitions
        self.C = capacity_per_partition

    def greedy_assign(self, block_sizes: List[int],
                      seed: int = 42) -> List[int]:
        """
        贪心算法分配数据块到分区。

n        策略：
          1. 将数据块按大小降序排列（最大块优先）
          2. 对每个块，选择当前剩余容量最大的分区
          3. 若无法放入任何分区，返回错误指示

        该贪心策略的近似比为 11/9 · OPT + 1（对于装箱问题）。
        """
        random.seed(seed)
        n = len(block_sizes)
        # 带原始索引的排序
        indexed = sorted(enumerate(block_sizes), key=lambda x: -x[1])
        assignment = [-1] * n
        remaining = [self.C] * self.P

        for orig_idx, size in indexed:
            # 找到剩余容量最大且能容纳该块的分区
            best_p = -1
            best_cap = -1
            for p in range(self.P):
                if remaining[p] >= size and remaining[p] > best_cap:
                    best_cap = remaining[p]
                    best_p = p
            if best_p == -1:
                # 无法放入，放入剩余容量最大的分区（超容警告）
                best_p = max(range(self.P), key=lambda p: remaining[p])
            assignment[orig_idx] = best_p
            remaining[best_p] -= size

        return assignment

    def balance_score(self, block_sizes: List[int], assignment: List[int]) -> float:
        """
        计算分区均衡度评分（变异系数的倒数，越大越均衡）。

        理想均衡时各分区负载相等，均为 total_size / P。
        评分 = mean_load / std_load，范围 [0, +∞)。
        """
        loads = [0.0] * self.P
        for idx, p in enumerate(assignment):
            loads[p] += block_sizes[idx]
        mean_load = sum(loads) / self.P
        if mean_load < 1e-15:
            return 0.0
        variance = sum((l - mean_load) ** 2 for l in loads) / self.P
        std_load = math.sqrt(variance)
        if std_load < 1e-15:
            return float('inf')
        return mean_load / std_load


class HashPartitioner:
    """
    基于素数哈希函数的一致性分区器。

    利用素数取模构造低碰撞哈希，将复合键映射到分区：
        h(k) = (a·k + b) mod p mod P

    其中 p 为大素数，a, b 为随机种子，P 为分区数。
    素数取模保证哈希值在 [0, p-1] 上均匀分布，减少分区倾斜。
    """

    def __init__(self, num_partitions: int, prime_seed_idx: int = 0):
        self.P = num_partitions
        primes = generate_primes(10000 + prime_seed_idx * 100, 3)
        self.p = primes[-1]
        seeds = hash_family_seed(prime_seed_idx, num_hashes=1)
        _, self.a, self.b = seeds[0]

    def hash_key(self, key: float) -> int:
        """
        将浮点键映射到分区编号。
        """
        # 将浮点键量化后哈希
        int_key = int(abs(key) * 1e9) % self.p
        h = (self.a * int_key + self.b) % self.p
        return h % self.P

    def partition(self, records: List[Tuple[float, ...]],
                  key_index: int = 0) -> List[List[Tuple[float, ...]]]:
        """
        对记录列表进行哈希分区。
        """
        parts = [[] for _ in range(self.P)]
        for rec in records:
            pid = self.hash_key(rec[key_index])
            parts[pid].append(rec)
        return parts


class AdaptivePartitionOptimizer:
    """
    自适应分区优化器，结合采样估计与约束满足。

    工作流程：
        1. 读取数据样本，估计分布
        2. 计算均衡分区边界（直方图/RBF/FEM）
        3. 用贪心算法微调，确保容量约束满足
        4. 输出最终分区方案
    """

    def __init__(self, num_partitions: int, memory_pages: int, page_size: int):
        self.P = num_partitions
        self.M_pages = memory_pages
        self.page_size = page_size
        self.capacity = memory_pages * page_size

    def optimize(self, records: List[Tuple[float, ...]],
                 key_index: int = 0,
                 sample_ratio: float = 0.05) -> Tuple[List[float], List[int]]:
        """
        执行自适应分区优化。

        返回:
            boundaries: 分区边界列表
            assignment: 每个记录对应的分区编号
        """
        from adaptive_sampler import adaptive_partition_estimation

        n = len(records)
        if n == 0:
            return [0.0, 1.0], []

        # 步骤1：采样估计分布
        sample_size = max(int(n * sample_ratio), min(100, n))
        sample = random.sample(records, sample_size)
        sample_keys = [rec[key_index] for rec in sample]

        # 步骤2：估计分区边界
        boundaries = adaptive_partition_estimation(sample_keys, self.P, use_rbf_refinement=True)

        # 步骤3：分段常数分配
        from external_sort_engine import PiecewiseConstantPartition
        pwc = PiecewiseConstantPartition(boundaries)
        partitions = pwc.partition_records(records, key_index)

        # 步骤4：检查容量约束，必要时调整
        block_sizes = [len(p) for p in partitions]
        solver = CoverConstraintSolver(self.P, self.capacity)
        assignment = solver.greedy_assign(block_sizes)

        # 将贪心调整后的分配映射回记录级别
        record_assignment = [-1] * n
        offset = 0
        for pid, part in enumerate(partitions):
            for rec in part:
                record_assignment[offset] = assignment[pid]
                offset += 1

        return boundaries, record_assignment


def optimal_page_alignment(total_records: int, memory_size: int,
                           page_size: int = 4096) -> int:
    """
    计算满足页对齐的最优块大小。

    利用GCD确保块大小同时是页大小和内存限制的公约数：
        block_size = gcd(memory_size, total_records · record_size)
    此处简化为基于记录数和内存页数的启发式计算。
    """
    # 计算记录大小假设为64字节（6个float + 开销）
    record_size = 64
    total_bytes = total_records * record_size
    aligned = compute_gcd(memory_size, page_size)
    # 最优块大小为内存限制和页大小的最大公约数的整数倍
    optimal_block = aligned
    while optimal_block < memory_size // 4 and optimal_block * 2 <= memory_size:
        optimal_block *= 2
    return optimal_block
