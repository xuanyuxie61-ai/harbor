"""
任务调度器 (task_scheduler.py)
==============================
基于种子项目 1196_task_division 的负载均衡思想，为地核发电机多尺度计算
提供任务划分与调度功能。

地核发电机模拟涉及不同球谐阶数 l 的独立演化方程，可将 (l, m) 模式空间
划分为若干子集，分配到不同计算批次中。本模块提供：
  - 模式空间任务划分 (mode_space_partition)
  - 多尺度时间步调度 (multiscale_dt_scheduler)
  - 径向网格负载均衡 (radial_load_balance)
"""

import math
from typing import List, Tuple


# ---------------------------------------------------------------------------
# 1. 模式空间任务划分
#    将 (l, m) 对（其中 l=1..Lmax, m=-l..l）均匀分配到 n_proc 个处理器
# ---------------------------------------------------------------------------
def mode_space_partition(l_max: int, n_proc: int) -> List[List[Tuple[int, int]]]:
    """
    将球谐模式空间划分为 n_proc 个任务块。

    总模式数 N = sum_{l=1}^{Lmax} (2l+1) = Lmax*(Lmax+2)。
    每个处理器获得约 N/n_proc 个模式。
    """
    if l_max < 1:
        return [[]]
    if n_proc < 1:
        n_proc = 1

    modes: List[Tuple[int, int]] = []
    for l in range(1, l_max + 1):
        for m in range(-l, l + 1):
            modes.append((l, m))

    total = len(modes)
    base = total // n_proc
    remainder = total % n_proc

    partitions = []
    idx = 0
    for p in range(n_proc):
        size = base + (1 if p < remainder else 0)
        partitions.append(modes[idx: idx + size])
        idx += size

    return partitions


# ---------------------------------------------------------------------------
# 2. 多尺度时间步调度
#    根据地磁场各阶模式的扩散时间尺度 tau_l = r^2 / (eta * l*(l+1))
#    为不同 l 分配不同的有效时间步上限
# ---------------------------------------------------------------------------
def multiscale_dt_scheduler(l_max: int, radius: float, eta: float,
                            base_dt: float, safety_factor: float = 0.1) -> List[float]:
    """
    计算每个球谐阶数 l 允许的最大显式时间步。

    磁扩散稳定性条件（显式欧拉）：
        dt_l <= safety_factor * r^2 / (eta * l * (l+1))

    返回 dt_max[l]，索引从 1 到 l_max。
    """
    if radius <= 0.0 or eta <= 0.0 or base_dt <= 0.0:
        raise ValueError("物理参数必须为正")

    dt_max = [0.0]  # 占位，索引 0 不用
    for l in range(1, l_max + 1):
        tau_diff = radius * radius / (eta * l * (l + 1.0))
        dt_limit = safety_factor * tau_diff
        dt_max.append(min(base_dt, dt_limit))
    return dt_max


# ---------------------------------------------------------------------------
# 3. 径向网格负载均衡
#    基于种子项目 task_division 的贪心分配策略
# ---------------------------------------------------------------------------
def radial_load_balance(n_radial: int, n_workers: int) -> List[Tuple[int, int]]:
    """
    将 n_radial 个径向网格层分配到 n_workers 个工作者。
    返回每个工作者的 (start, end) 索引区间（含端点）。
    """
    if n_radial < 1:
        return []
    if n_workers < 1:
        n_workers = 1

    tasks_remain = n_radial
    workers_remain = n_workers
    intervals = []
    start = 0

    while workers_remain > 0 and tasks_remain > 0:
        # 贪心：round(tasks_remain / workers_remain)
        share = tasks_remain // workers_remain
        if (2 * share + 1) * workers_remain < 2 * tasks_remain:
            share += 1
        share = max(1, share)
        share = min(share, tasks_remain)
        end = start + share - 1
        intervals.append((start, end))
        start = end + 1
        tasks_remain -= share
        workers_remain -= 1

    return intervals


# ---------------------------------------------------------------------------
# 4. 计算批次管理器（地核发电机专用）
# ---------------------------------------------------------------------------
class DynamoBatchScheduler:
    """
    管理地核发电机模拟的计算批次：
      - 将 (l,m) 模式分块
      - 为每批分配时间步上限
      - 跟踪各批完成状态
    """

    def __init__(self, l_max: int, n_radial: int, n_batches: int,
                 radius: float, eta: float, base_dt: float):
        self.l_max = l_max
        self.n_radial = n_radial
        self.n_batches = max(1, n_batches)
        self.mode_partitions = mode_space_partition(l_max, self.n_batches)
        self.radial_intervals = radial_load_balance(n_radial, self.n_batches)
        self.dt_limits = multiscale_dt_scheduler(l_max, radius, eta, base_dt)
        self.completed = [False] * self.n_batches

    def get_batch_modes(self, batch_id: int) -> List[Tuple[int, int]]:
        if 0 <= batch_id < self.n_batches:
            return self.mode_partitions[batch_id]
        return []

    def get_batch_radial_range(self, batch_id: int) -> Tuple[int, int]:
        if 0 <= batch_id < len(self.radial_intervals):
            return self.radial_intervals[batch_id]
        return (0, 0)

    def get_effective_dt(self, batch_id: int) -> float:
        """取该批次中所有模式对应 dt 上限的最小值。"""
        modes = self.get_batch_modes(batch_id)
        if not modes:
            return float('inf')
        dt_min = float('inf')
        for l, _ in modes:
            if l < len(self.dt_limits):
                dt_min = min(dt_min, self.dt_limits[l])
        return dt_min if dt_min < float('inf') else 1.0

    def mark_completed(self, batch_id: int):
        if 0 <= batch_id < self.n_batches:
            self.completed[batch_id] = True

    def all_completed(self) -> bool:
        return all(self.completed)


# ---------------------------------------------------------------------------
# 自测试
# ---------------------------------------------------------------------------
def _self_test():
    parts = mode_space_partition(3, 2)
    total_modes = sum(len(p) for p in parts)
    assert total_modes == 3 * (3 + 2)  # l=1..3, N = l*(l+2) = 15

    dt_limits = multiscale_dt_scheduler(4, 3480e3, 2.0, 1e10)
    assert len(dt_limits) == 5
    assert dt_limits[1] >= dt_limits[4]  # 高 l 扩散更快，时间步更小

    intervals = radial_load_balance(32, 5)
    total = sum(e - s + 1 for s, e in intervals)
    assert total == 32

    sched = DynamoBatchScheduler(l_max=4, n_radial=32, n_batches=3,
                                  radius=3480e3, eta=2.0, base_dt=1e10)
    assert sched.get_effective_dt(0) > 0.0
    print("task_scheduler: self-test passed.")


if __name__ == "__main__":
    _self_test()
