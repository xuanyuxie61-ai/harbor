"""
parallel_utils.py
=================
并行计算工具模块（源自 seed 514_hello_parfor 的并行 for 思想）

在粗粒化分子动力学中，非键相互作用（范德华、静电）的计算复杂度为 O(N^2)，
是模拟的绝对瓶颈。本模块利用 Python multiprocessing 实现**空间分解并行力计算**，
将模拟盒子沿某一维度划分为多个域，每个域独立计算局部粒子受力后汇总。

关键约束：
    - 为保证数值可复现，随机力必须在主进程统一生成后分发给子进程；
    - 为避免竞态条件，力向量采用进程独立累加后归约（reduce）。
"""

import multiprocessing as mp
from typing import Callable, List, Any, Sequence
import numpy as np


def parallel_map(func: Callable[[Any], Any],
                 data: Sequence[Any],
                 n_workers: int = None) -> List[Any]:
    """
    对 data 中的每个元素并行调用 func，返回结果列表。

    在 MD 中，典型用法是将粒子索引列表分片后并行计算每片的受力贡献：

        forces = parallel_map(_compute_slice_force, slices, n_workers=4)
        total_force = np.sum(forces, axis=0)

    Parameters
    ----------
    func : callable
        单参数函数，接收 data 中的一个元素。
    data : sequence
        待处理的数据序列。
    n_workers : int, optional
        并行进程数；默认使用 cpu_count() // 2 以避免超线程带来的缓存抖动。

    Returns
    -------
    results : list
        与 data 等长的结果列表，顺序保持。
    """
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() // 2)
    if n_workers <= 1 or len(data) <= 1:
        return [func(d) for d in data]
    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(func, data)
    return results


def parallel_force_reduction(compute_local: Callable[[int, int], np.ndarray],
                             n_particles: int,
                             n_workers: int = None) -> np.ndarray:
    """
    并行计算并归约总受力向量。

    将 N 个粒子的受力计算划分为 n_workers 个区间：
        slice_i = [start_i, end_i),  start_i = i * N // n_workers
    每个进程计算其负责区间内的局部力矩阵 F_local[start:end, :]，
    最后将各局部矩阵相加得到总力。

    Parameters
    ----------
    compute_local : callable(start, end) -> ndarray
        计算粒子索引范围 [start, end) 内的局部力矩阵 (end-start, 3)。
    n_particles : int
        总粒子数。
    n_workers : int, optional
        进程数。

    Returns
    -------
    total_force : ndarray, shape (n_particles, 3)
        每个粒子在三维空间中的总受力。
    """
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() // 2)
    n_workers = min(n_workers, n_particles)

    def worker(args):
        start, end = args
        return compute_local(start, end)

    slices = []
    for i in range(n_workers):
        start = i * n_particles // n_workers
        end = (i + 1) * n_particles // n_workers
        slices.append((start, end))

    local_forces = parallel_map(worker, slices, n_workers=n_workers)
    total_force = np.vstack(local_forces)
    return total_force
