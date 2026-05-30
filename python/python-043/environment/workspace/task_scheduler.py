
import math
from typing import List, Tuple






def mode_space_partition(l_max: int, n_proc: int) -> List[List[Tuple[int, int]]]:
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







def multiscale_dt_scheduler(l_max: int, radius: float, eta: float,
                            base_dt: float, safety_factor: float = 0.1) -> List[float]:
    if radius <= 0.0 or eta <= 0.0 or base_dt <= 0.0:
        raise ValueError("物理参数必须为正")

    dt_max = [0.0]
    for l in range(1, l_max + 1):
        tau_diff = radius * radius / (eta * l * (l + 1.0))
        dt_limit = safety_factor * tau_diff
        dt_max.append(min(base_dt, dt_limit))
    return dt_max






def radial_load_balance(n_radial: int, n_workers: int) -> List[Tuple[int, int]]:
    if n_radial < 1:
        return []
    if n_workers < 1:
        n_workers = 1

    tasks_remain = n_radial
    workers_remain = n_workers
    intervals = []
    start = 0

    while workers_remain > 0 and tasks_remain > 0:

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





class DynamoBatchScheduler:

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





def _self_test():
    parts = mode_space_partition(3, 2)
    total_modes = sum(len(p) for p in parts)
    assert total_modes == 3 * (3 + 2)

    dt_limits = multiscale_dt_scheduler(4, 3480e3, 2.0, 1e10)
    assert len(dt_limits) == 5
    assert dt_limits[1] >= dt_limits[4]

    intervals = radial_load_balance(32, 5)
    total = sum(e - s + 1 for s, e in intervals)
    assert total == 32

    sched = DynamoBatchScheduler(l_max=4, n_radial=32, n_batches=3,
                                  radius=3480e3, eta=2.0, base_dt=1e10)
    assert sched.get_effective_dt(0) > 0.0
    print("task_scheduler: self-test passed.")


if __name__ == "__main__":
    _self_test()
