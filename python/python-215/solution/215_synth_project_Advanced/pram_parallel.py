#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pram_parallel.py — PRAM 并行拓扑评估协调器

对应种子项目:
  - 906_pram_view: PRAM (Parallel Random Access Machine) 视图与分块配置

核心思想:
  PRAM 模型的三种并发读写策略:
    EREW (Exclusive Read Exclusive Write)
    CREW (Concurrent Read Exclusive Write)
    CRCW (Concurrent Read Concurrent Write)

  在优化中映射为:
    - 种群分块: 将种群分成多个子块, 每块独立评估
    - 并行评估: 使用线程池并发计算目标函数
    - 结果合并: CREW 方式收集结果 (共享读取, 独占写入)

  分块策略:
      n_blocks = ceil(N / block_size)
      每个 block 包含 block_size 个个体
      各 block 可独立并行评估
"""

import numpy as np
from typing import Callable, List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed


# ---------------------------------------------------------------------------
# PRAM 分块配置
# ---------------------------------------------------------------------------
class PRAMConfig:
    """
    PRAM 并行配置.

    参数:
        n_workers   : 并行工作线程数
        block_size  : 每块大小
        strategy    : 'EREW' | 'CREW' | 'CRCW'
    """

    def __init__(self, n_workers: int = 4,
                 block_size: int = 10,
                 strategy: str = 'CREW'):
        self.n_workers = max(1, n_workers)
        self.block_size = max(1, block_size)
        self.strategy = strategy
        self._validate()

    def _validate(self):
        if self.strategy not in ('EREW', 'CREW', 'CRCW'):
            raise ValueError(f"未知 PRAM 策略: {self.strategy}")

    def compute_blocks(self, n_items: int) -> List[Tuple[int, int]]:
        """
        计算分块 — 将 n_items 分成多个块.

        Returns: List of (start, end) 索引对
        """
        blocks = []
        for i in range(0, n_items, self.block_size):
            end = min(i + self.block_size, n_items)
            blocks.append((i, end))
        return blocks


def tile_configurations(n_total: int,
                        max_tile_size: int = 10,
                        n_tiles_range: Tuple[int, int] = (2, 8)
                        ) -> List[Dict[str, int]]:
    """
    生成不同的分块配置 (来自 906 pram_view 的思想).

    对给定的总数量, 列举不同的分块方式.

    Parameters
    ----------
    n_total       : 总个体数
    max_tile_size : 最大块大小
    n_tiles_range : 块数范围

    Returns
    -------
    List of dicts with 'n_tiles', 'tile_size', 'remainder'
    """
    configs = []
    for n_tiles in range(n_tiles_range[0], n_tiles_range[1] + 1):
        tile_size = n_total // n_tiles
        remainder = n_total % n_tiles
        tile_size = min(tile_size, max_tile_size)
        configs.append({
            'n_tiles': n_tiles,
            'tile_size': tile_size,
            'remainder': remainder,
            'total_covered': tile_size * n_tiles + remainder
        })
    return configs


# ---------------------------------------------------------------------------
# 并行评估器
# ---------------------------------------------------------------------------
class ParallelEvaluator:
    """
    基于 PRAM 模型的并行目标函数评估.

    使用线程池并发评估种群中的个体.
    支持 EREW (无共享), CREW (共享读取), CRCW (共享读写).

    对于纯函数评估 (无状态), EREW 和 CREW 等价.
    """

    def __init__(self, evaluator: Callable,
                 config: PRAMConfig = None):
        self.evaluator = evaluator
        self.config = config or PRAMConfig()
        self._eval_count = 0

    def evaluate_batch(self, population: np.ndarray
                       ) -> np.ndarray:
        """
        并行评估种群.

        Parameters
        ----------
        population : shape (N, D)

        Returns
        -------
        fitness : shape (N, M)
        """
        N = population.shape[0]
        blocks = self.config.compute_blocks(N)

        results = [None] * N

        if self.config.n_workers == 1 or N <= self.config.block_size:
            # 串行
            for i in range(N):
                results[i] = self.evaluator(population[i])
        else:
            # 并行
            with ThreadPoolExecutor(max_workers=self.config.n_workers) as pool:
                futures = {}
                for start, end in blocks:
                    for i in range(start, end):
                        f = pool.submit(self.evaluator, population[i])
                        futures[f] = i

                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        # 鲁棒性: 评估失败时返回大值
                        print(f"  [PRAM] 个体 {idx} 评估失败: {e}")
                        results[idx] = np.full(3, 1e6)

        self._eval_count += N
        return np.array(results)

    @property
    def total_evals(self) -> int:
        return self._eval_count


# ---------------------------------------------------------------------------
# 并行诊断
# ---------------------------------------------------------------------------
def benchmark_parallel_speedup(evaluator: Callable,
                               population: np.ndarray,
                               worker_range: List[int] = None
                               ) -> Dict[int, float]:
    """
    测试不同并行度下的加速比.

    Returns: dict {n_workers: time_seconds}
    """
    import time

    if worker_range is None:
        worker_range = [1, 2, 4]

    results = {}
    for nw in worker_range:
        config = PRAMConfig(n_workers=nw, block_size=max(1, len(population) // nw))
        pe = ParallelEvaluator(evaluator, config)
        t0 = time.time()
        pe.evaluate_batch(population)
        t1 = time.time()
        results[nw] = t1 - t0

    return results
