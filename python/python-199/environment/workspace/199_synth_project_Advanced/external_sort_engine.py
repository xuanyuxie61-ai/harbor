
import math
import heapq
import os
import tempfile
from typing import List, Tuple, Optional, Callable


class ReplacementSelection:

    def __init__(self, memory_capacity: int):
        self.M = max(memory_capacity, 2)
        self.runs = []

    def generate_runs(self, records: List[Tuple[float, ...]],
                      key_index: int = 0) -> List[List[Tuple[float, ...]]]:
        if not records:
            return []

        idx = 0
        n = len(records)
        runs = []

        while idx < n:

            batch = records[idx:min(idx + self.M, n)]
            idx += len(batch)


            heap = []
            dead = []
            for rec in batch:
                key = rec[key_index]
                heapq.heappush(heap, (key, 0, rec))

            current_run = []
            last_output_key = -float('inf')

            while heap:
                key, flag, rec = heapq.heappop(heap)
                if flag == 1:

                    dead.append((key, rec))
                    continue

                if key < last_output_key:

                    dead.append((key, rec))
                else:
                    current_run.append(rec)
                    last_output_key = key


                if idx < n:
                    next_rec = records[idx]
                    idx += 1
                    next_key = next_rec[key_index]
                    if next_key >= last_output_key and current_run:
                        heapq.heappush(heap, (next_key, 0, next_rec))
                    else:
                        heapq.heappush(heap, (next_key, 1, next_rec))

            if current_run:
                runs.append(current_run)


            if dead:
                dead_records = [rec for _, rec in sorted(dead, key=lambda x: x[0])]


                if dead_records:
                    extra_runs = self._process_dead(dead_records, key_index)
                    runs.extend(extra_runs)

        return runs

    def _process_dead(self, records: List[Tuple[float, ...]],
                      key_index: int) -> List[List[Tuple[float, ...]]]:
        if not records:
            return []
        if len(records) <= self.M:
            return [sorted(records, key=lambda r: r[key_index])]

        result = []
        for i in range(0, len(records), self.M):
            batch = records[i:i + self.M]
            result.append(sorted(batch, key=lambda r: r[key_index]))
        return result


class KWayMerge:

    def __init__(self, runs: List[List[Tuple[float, ...]]], k: int,
                 key_index: int = 0):
        self.runs = [list(r) for r in runs]
        self.k = max(k, 2)
        self.key_index = key_index
        self.num_runs = len(runs)

    def merge(self) -> List[Tuple[float, ...]]:
        if not self.runs:
            return []
        if len(self.runs) == 1:
            return self.runs[0]



        heap = []
        for ri, run in enumerate(self.runs):
            if run:
                rec = run[0]
                heapq.heappush(heap, (rec[self.key_index], ri, 0, rec))

        merged = []
        while heap:
            key, ri, ei, rec = heapq.heappop(heap)
            merged.append(rec)
            next_ei = ei + 1
            if next_ei < len(self.runs[ri]):
                next_rec = self.runs[ri][next_ei]
                heapq.heappush(heap, (next_rec[self.key_index], ri, next_ei, next_rec))
        return merged


class PiecewiseConstantPartition:

    def __init__(self, boundaries: List[float]):
        if len(boundaries) < 2:
            raise ValueError("Need at least 2 boundaries.")
        self.boundaries = sorted(boundaries)
        self.n_intervals = len(boundaries) - 1

    def assign_partition(self, key: float) -> int:
        if key <= self.boundaries[0]:
            return 0
        if key >= self.boundaries[-1]:
            return self.n_intervals - 1

        lo, hi = 0, self.n_intervals
        while lo < hi:
            mid = (lo + hi) // 2
            if key < self.boundaries[mid]:
                hi = mid
            else:
                lo = mid + 1
        return min(lo - 1, self.n_intervals - 1)

    def partition_records(self, records: List[Tuple[float, ...]],
                          key_index: int = 0) -> List[List[Tuple[float, ...]]]:
        partitions = [[] for _ in range(self.n_intervals)]
        for rec in records:
            pid = self.assign_partition(rec[key_index])
            partitions[pid].append(rec)
        return partitions


class ExternalSortPipeline:

    def __init__(self, memory_capacity: int, k_way: int = 4, key_index: int = 0):
        self.M = memory_capacity
        self.k = k_way
        self.key_index = key_index
        self.io_count = 0

    def sort(self, records: List[Tuple[float, ...]]) -> List[Tuple[float, ...]]:
        if not records:
            return []
        if len(records) <= self.M:

            return sorted(records, key=lambda r: r[self.key_index])


        rs = ReplacementSelection(self.M)
        runs = rs.generate_runs(records, self.key_index)
        self.io_count += len(runs)


        current_runs = runs
        merge_pass = 0
        while len(current_runs) > 1:
            merge_pass += 1
            next_runs = []
            for i in range(0, len(current_runs), self.k):
                batch = current_runs[i:i + self.k]
                merger = KWayMerge(batch, len(batch), self.key_index)
                merged = merger.merge()
                next_runs.append(merged)
                self.io_count += len(batch) + 1
            current_runs = next_runs

        result = current_runs[0] if current_runs else []


        if len(result) != len(records):
            raise RuntimeError(
                f"Coverage constraint violated: input {len(records)} != output {len(result)}"
            )
        return result

    def theoretical_io_cost(self, N: int, B: int) -> float:
        if self.M <= B:
            return float('inf')
        ratio = N / self.M
        log_base = self.M / B
        passes = math.ceil(math.log(ratio) / math.log(log_base)) if ratio > 1 else 1
        return 2.0 * (N / B) * max(passes, 1)


def verify_sorted(records: List[Tuple[float, ...]], key_index: int = 0) -> bool:
    for i in range(1, len(records)):
        if records[i][key_index] < records[i - 1][key_index]:
            return False
    return True
