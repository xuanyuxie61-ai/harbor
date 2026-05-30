
import multiprocessing as mp
from typing import Callable, List, Any, Sequence
import numpy as np


def parallel_map(func: Callable[[Any], Any],
                 data: Sequence[Any],
                 n_workers: int = None) -> List[Any]:
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
