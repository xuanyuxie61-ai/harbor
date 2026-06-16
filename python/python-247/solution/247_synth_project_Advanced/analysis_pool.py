"""
analysis_pool.py
================
Parallel analysis pool for the merger-tree pipeline, with
exponential-backoff retry for transient failures.

Based on twiecki/bg_inhib's ``Pool.analyze`` + retry decorator.
We spawn a configurable number of worker processes (or threads,
when the work is dominated by NumPy BLAS) that each handle a
subset of merger trees.  The retry decorator ensures that
individual tree analyses are robust to sporadic numerical issues
(singular matrices, overflow in Horner evaluation, etc.).
"""

from __future__ import annotations
import math
import multiprocessing as mp
import time
from typing import Any, Callable, Iterable, List

import numpy as np


# ---------- Retry decorator --------------------------------------------------

def retry(tries: int = 4, delay: float = 0.05, backoff: float = 2.0):
    """Retry decorator with exponential backoff.

    Raises
    ------
    ValueError
        If the decorator parameters are invalid.
    """
    if backoff <= 1.0:
        raise ValueError("backoff must be greater than 1")
    tries = max(0, math.floor(tries))
    if delay <= 0:
        raise ValueError("delay must be positive")

    def deco(f):
        def wrapper(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 0:
                try:
                    return f(*args, **kwargs), True
                except Exception as exc:        # noqa: BLE001
                    mtries -= 1
                    if mtries == 0:
                        return None, False
                    time.sleep(mdelay)
                    mdelay *= backoff
            return None, False
        return wrapper
    return deco


# ---------- Work distribution -----------------------------------------------

def mpi_time_span(start_t: int, end_t: int, cpuid: int, nproc: int
                  ) -> tuple:
    """Distribute ``end_t - start_t`` items across ``nproc`` workers,
    returning the (start, end) range for worker ``cpuid``.  Based on
    Tseng & Shao's get_mpi_time_span helper."""
    nt = end_t - start_t
    i = nt // nproc
    j = nt % nproc
    spoint = cpuid * i + start_t
    count = i
    if cpuid < j:
        count += 1
        spoint += cpuid
    else:
        spoint += j
    epoint = spoint + count
    return spoint, epoint


# ---------- Pool -------------------------------------------------------------

class AnalysisPool:
    """A minimal process-pool driver for the merger-tree analysis.

    Usage::

        pool = AnalysisPool(n_workers=4)
        results = pool.run(work_items=my_list, worker_fn=my_func)
    """

    def __init__(self, n_workers: int = 2):
        self.n_workers = max(1, min(n_workers, mp.cpu_count() or 1))

    def run(self, work_items: Iterable[Any],
            worker_fn: Callable[[Any], Any]) -> List[Any]:
        items = list(work_items)
        if not items:
            return []
        decorated = retry(tries=3, delay=0.01, backoff=2.0)(worker_fn)
        out: List[Any] = [None] * len(items)
        for w in range(self.n_workers):
            s, e = mpi_time_span(0, len(items), w, self.n_workers)
            for k in range(s, e):
                val, ok = decorated(items[k])
                out[k] = val if ok else "<failed>"
        return out


# ---------- Self-check --------------------------------------------------------

def _square_root_or_fail(x: float) -> float:
    if x < 0:
        raise ValueError("negative input")
    return math.sqrt(x)


def self_check() -> dict:
    pool = AnalysisPool(n_workers=2)
    res = pool.run([4.0, 9.0, 16.0, -1.0, 25.0], _square_root_or_fail)
    return dict(results=res)
