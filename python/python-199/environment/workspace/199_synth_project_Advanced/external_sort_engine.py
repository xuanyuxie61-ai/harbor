"""
external_sort_engine.py — 内存受限外排序核心引擎
=================================================
融合来源:
  - 923_pwc_plot_1d (分段常数函数结构，用于快速分区)
  - 339_eternity (组合覆盖约束与数据块分配思想)

实现基于替换选择（Replacement Selection）的多路外排序算法。
核心目标：在内存容量 M 远小于数据总量 N 的条件下，以最小 I/O 代价
完成全序排列。

I/O 复杂度理论下界（Aggarwal-Vitter 外排序模型）：
    最优 I/O 次数 = Θ( (N/B) · log_{M/B}(N/M) )
其中 B 为磁盘块大小，M 为内存容量。

替换选择产生的初始归并段平均长度为 2M（随机数据假设下），
显著优于简单分块排序的 M。
"""

import math
import heapq
import os
import tempfile
from typing import List, Tuple, Optional, Callable


class ReplacementSelection:
    """
    替换选择算法生成初始归并段（runs）。

    算法描述：
        1. 从输入流读入 M 个记录到内存数组
        2. 建立最小堆（键值为排序键）
        3. 反复提取堆顶最小元素输出到当前 run
        4. 读入下一个输入记录：
           - 若其键 ≥ 最后输出键，放入堆中（延长当前 run）
           - 否则放入“死区”（dead space），待下一 run 处理
        5. 堆空时当前 run 结束，将死区记录移回堆开始下一 run

    对随机输入，平均 run 长度 ≈ 2M（Knuth 证明）。
    """

    def __init__(self, memory_capacity: int):
        self.M = max(memory_capacity, 2)
        self.runs = []

    def generate_runs(self, records: List[Tuple[float, ...]],
                      key_index: int = 0) -> List[List[Tuple[float, ...]]]:
        """
        从记录列表生成有序归并段。

        参数:
            records: 输入记录列表，每条记录为元组，key_index 指定排序键位置
            key_index: 排序键在元组中的索引
        """
        if not records:
            return []

        idx = 0
        n = len(records)
        runs = []

        while idx < n:
            # 读入最多 M 个记录
            batch = records[idx:min(idx + self.M, n)]
            idx += len(batch)

            # 初始化堆：元素格式 (key, is_dead, record)
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
                    # 死区元素直接暂存
                    dead.append((key, rec))
                    continue

                if key < last_output_key:
                    # 当前元素无法延续当前 run，标记为死区
                    dead.append((key, rec))
                else:
                    current_run.append(rec)
                    last_output_key = key

                # 读入下一个输入记录
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

            # 死区记录作为下一批输入
            if dead:
                dead_records = [rec for _, rec in sorted(dead, key=lambda x: x[0])]
                # 将未处理的死区插回到records末尾（逻辑上）
                # 为简化，直接递归处理
                if dead_records:
                    extra_runs = self._process_dead(dead_records, key_index)
                    runs.extend(extra_runs)

        return runs

    def _process_dead(self, records: List[Tuple[float, ...]],
                      key_index: int) -> List[List[Tuple[float, ...]]]:
        """处理死区记录，产生额外的runs。"""
        if not records:
            return []
        if len(records) <= self.M:
            return [sorted(records, key=lambda r: r[key_index])]
        # 分批处理
        result = []
        for i in range(0, len(records), self.M):
            batch = records[i:i + self.M]
            result.append(sorted(batch, key=lambda r: r[key_index]))
        return result


class KWayMerge:
    """
    k路归并，使用败者树（Loser Tree）优化。

    败者树将每次选取最小元素的时间复杂度从 O(k) 降低到 O(log k)，
    是外排序归并阶段的标准数据结构。

    对于 R 个初始归并段，k 路归并的趟数为 ⌈log_k R⌉。
    总 I/O 复杂度为 O( (N/B) · log_k R )。
    """

    def __init__(self, runs: List[List[Tuple[float, ...]]], k: int,
                 key_index: int = 0):
        self.runs = [list(r) for r in runs]
        self.k = max(k, 2)
        self.key_index = key_index
        self.num_runs = len(runs)

    def merge(self) -> List[Tuple[float, ...]]:
        """
        执行 k 路归并，返回全局有序记录列表。
        """
        if not self.runs:
            return []
        if len(self.runs) == 1:
            return self.runs[0]

        # 使用堆模拟败者树（Python heapq 为最小堆）
        # 每个元素：(key, run_index, element_index, record)
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
    """
    分段常数近似快速分区。

    将数据范围划分为若干区间，每个区间赋予一个常数值（此处为分区编号）。
    该结构用于快速确定记录所属的目标分区，无需全局比较。

    源于 pwc_plot_1d 的分段常数思想，去除可视化部分，保留数值结构。
    """

    def __init__(self, boundaries: List[float]):
        if len(boundaries) < 2:
            raise ValueError("Need at least 2 boundaries.")
        self.boundaries = sorted(boundaries)
        self.n_intervals = len(boundaries) - 1

    def assign_partition(self, key: float) -> int:
        """
        将键值分配到对应分区编号 [0, n_intervals-1]。
        边界外键值进行饱和处理。
        """
        if key <= self.boundaries[0]:
            return 0
        if key >= self.boundaries[-1]:
            return self.n_intervals - 1
        # 二分查找
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
        """
        将记录列表按分段常数边界分配到各分区。
        """
        partitions = [[] for _ in range(self.n_intervals)]
        for rec in records:
            pid = self.assign_partition(rec[key_index])
            partitions[pid].append(rec)
        return partitions


class ExternalSortPipeline:
    """
    外排序完整流水线：替换选择 → k路归并 → 验证。

    Eternity 拼图的覆盖约束思想体现在：每个数据记录必须被恰好一个
    归并段覆盖，且所有归并段的并集必须等于全集（完整性约束）。
    """

    def __init__(self, memory_capacity: int, k_way: int = 4, key_index: int = 0):
        self.M = memory_capacity
        self.k = k_way
        self.key_index = key_index
        self.io_count = 0  # 模拟I/O计数

    def sort(self, records: List[Tuple[float, ...]]) -> List[Tuple[float, ...]]:
        """
        执行完整外排序流程。
        """
        if not records:
            return []
        if len(records) <= self.M:
            # 内存可容纳，直接排序
            return sorted(records, key=lambda r: r[self.key_index])

        # Phase 1: 替换选择生成初始归并段
        rs = ReplacementSelection(self.M)
        runs = rs.generate_runs(records, self.key_index)
        self.io_count += len(runs)  # 每个run一次写操作

        # Phase 2: k路归并（多趟直至一个run）
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
                self.io_count += len(batch) + 1  # 读 + 写
            current_runs = next_runs

        result = current_runs[0] if current_runs else []

        # 覆盖约束验证：输入输出记录数一致
        if len(result) != len(records):
            raise RuntimeError(
                f"Coverage constraint violated: input {len(records)} != output {len(result)}"
            )
        return result

    def theoretical_io_cost(self, N: int, B: int) -> float:
        """
        理论I/O代价估计（Aggarwal-Vitter模型）：
            IO(N, M, B) ≈ 2 · (N/B) · ⌈log_{M/B}(N/M)⌉
        """
        if self.M <= B:
            return float('inf')
        ratio = N / self.M
        log_base = self.M / B
        passes = math.ceil(math.log(ratio) / math.log(log_base)) if ratio > 1 else 1
        return 2.0 * (N / B) * max(passes, 1)


def verify_sorted(records: List[Tuple[float, ...]], key_index: int = 0) -> bool:
    """
    验证记录列表是否按指定键严格非递减排序。
    """
    for i in range(1, len(records)):
        if records[i][key_index] < records[i - 1][key_index]:
            return False
    return True
