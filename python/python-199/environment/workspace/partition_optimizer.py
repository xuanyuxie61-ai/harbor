
import math
import random
from typing import List, Tuple, Dict, Optional
from utils import generate_primes, hash_family_seed, compute_gcd


class CoverConstraintSolver:

    def __init__(self, num_partitions: int, capacity_per_partition: int):
        self.P = num_partitions
        self.C = capacity_per_partition

    def greedy_assign(self, block_sizes: List[int],
                      seed: int = 42) -> List[int]:
        random.seed(seed)
        n = len(block_sizes)

        indexed = sorted(enumerate(block_sizes), key=lambda x: -x[1])
        assignment = [-1] * n
        remaining = [self.C] * self.P

        for orig_idx, size in indexed:

            best_p = -1
            best_cap = -1
            for p in range(self.P):
                if remaining[p] >= size and remaining[p] > best_cap:
                    best_cap = remaining[p]
                    best_p = p
            if best_p == -1:

                best_p = max(range(self.P), key=lambda p: remaining[p])
            assignment[orig_idx] = best_p
            remaining[best_p] -= size

        return assignment

    def balance_score(self, block_sizes: List[int], assignment: List[int]) -> float:
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

    def __init__(self, num_partitions: int, prime_seed_idx: int = 0):
        self.P = num_partitions
        primes = generate_primes(10000 + prime_seed_idx * 100, 3)
        self.p = primes[-1]
        seeds = hash_family_seed(prime_seed_idx, num_hashes=1)
        _, self.a, self.b = seeds[0]

    def hash_key(self, key: float) -> int:

        int_key = int(abs(key) * 1e9) % self.p
        h = (self.a * int_key + self.b) % self.p
        return h % self.P

    def partition(self, records: List[Tuple[float, ...]],
                  key_index: int = 0) -> List[List[Tuple[float, ...]]]:
        parts = [[] for _ in range(self.P)]
        for rec in records:
            pid = self.hash_key(rec[key_index])
            parts[pid].append(rec)
        return parts


class AdaptivePartitionOptimizer:

    def __init__(self, num_partitions: int, memory_pages: int, page_size: int):
        self.P = num_partitions
        self.M_pages = memory_pages
        self.page_size = page_size
        self.capacity = memory_pages * page_size

    def optimize(self, records: List[Tuple[float, ...]],
                 key_index: int = 0,
                 sample_ratio: float = 0.05) -> Tuple[List[float], List[int]]:
        from adaptive_sampler import adaptive_partition_estimation

        n = len(records)
        if n == 0:
            return [0.0, 1.0], []


        sample_size = max(int(n * sample_ratio), min(100, n))
        sample = random.sample(records, sample_size)
        sample_keys = [rec[key_index] for rec in sample]


        boundaries = adaptive_partition_estimation(sample_keys, self.P, use_rbf_refinement=True)


        from external_sort_engine import PiecewiseConstantPartition
        pwc = PiecewiseConstantPartition(boundaries)
        partitions = pwc.partition_records(records, key_index)


        block_sizes = [len(p) for p in partitions]
        solver = CoverConstraintSolver(self.P, self.capacity)
        assignment = solver.greedy_assign(block_sizes)


        record_assignment = [-1] * n
        offset = 0
        for pid, part in enumerate(partitions):
            for rec in part:
                record_assignment[offset] = assignment[pid]
                offset += 1

        return boundaries, record_assignment


def optimal_page_alignment(total_records: int, memory_size: int,
                           page_size: int = 4096) -> int:

    record_size = 64
    total_bytes = total_records * record_size
    aligned = compute_gcd(memory_size, page_size)

    optimal_block = aligned
    while optimal_block < memory_size // 4 and optimal_block * 2 <= memory_size:
        optimal_block *= 2
    return optimal_block
