#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
timing_utils.py  ——  计时工具 & 性能基准

融合种子项目:
  - 1417_wtime : 挂钟时间 → 模拟性能计时

核心: 记录各阶段耗时, 计算 MFlops, 输出性能报告.
"""

from __future__ import annotations
import time
from typing import Dict, List
from dataclasses import dataclass, field


@dataclass
class TimerSection:
    name: str
    elapsed: float = 0.0
    count: int = 0

    @property
    def avg(self) -> float:
        return self.elapsed / max(self.count, 1)


class PerformanceTimer:
    """
    多区段计时器 (复刻 1417_wtime 的 wallclock 思想).
    """

    def __init__(self) -> None:
        self.sections: Dict[str, TimerSection] = {}
        self._start_times: Dict[str, float] = {}
        self._origin = time.time()

    def start(self, name: str) -> None:
        self._start_times[name] = time.time()

    def stop(self, name: str) -> float:
        if name not in self._start_times:
            return 0.0
        elapsed = time.time() - self._start_times[name]
        if name not in self.sections:
            self.sections[name] = TimerSection(name=name)
        self.sections[name].elapsed += elapsed
        self.sections[name].count += 1
        del self._start_times[name]
        return elapsed

    def report(self) -> Dict[str, Dict[str, float]]:
        """返回各区段的耗时统计."""
        result = {}
        for name, sec in self.sections.items():
            result[name] = {
                "total": sec.elapsed,
                "count": sec.count,
                "avg": sec.avg,
            }
        return result

    def walltime_since_origin(self) -> float:
        return time.time() - self._origin

    def print_report(self) -> None:
        """打印性能报告."""
        print("\n" + "=" * 60)
        print("性能计时报告")
        print("=" * 60)
        total = 0.0
        for name, stats in sorted(self.report().items()):
            print(f"  {name:<30s}  "
                  f"总耗时: {stats['total']:8.4f}s  "
                  f"调用: {int(stats['count']):4d}  "
                  f"平均: {stats['avg']:8.4f}s")
            total += stats['total']
        print("-" * 60)
        print(f"  {'合计':<30s}  总耗时: {total:8.4f}s")
        print(f"  {'挂钟时间':<30s}  {self.walltime_since_origin():8.4f}s")
        print("=" * 60)


def compute_mflops(n_ops: int, elapsed: float) -> float:
    """返回 MFlops (百万次浮点运算/秒)."""
    return n_ops / max(elapsed, 1e-30) / 1e6
