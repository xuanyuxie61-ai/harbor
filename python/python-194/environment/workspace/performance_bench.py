"""
performance_bench.py
====================
Performance benchmarking utilities for subdomain direct solvers
and parallel finite element assembly, based on the LINPACK benchmark.

Integrates concepts from:
  * linpack_bench_backslash (dense linear solve performance metrics)

Mathematical background
-----------------------
For a dense n x n linear system solved by LU factorization
(2/3 n^3 floating-point operations) plus forward/backward
substitution (2 n^2 operations), the total FLOP count is:
    FLOPs = (2/3) n^3 + 2 n^2

Performance in MFLOPS:
    MFLOPS = FLOPs / (10^6 * t_seconds)

For banded SPD matrices with bandwidth m, the Cholesky factorization
cost is approximately:
    FLOPs_band = n m^2 - (1/3) m^3

Error metrics:
    Residual infinity norm:    ||r||_inf = ||b - A x||_inf
    Normalized residual:       ||r||_inf / (||A||_inf * ||x||_inf * epsilon)
    where epsilon is machine epsilon (~2.2e-16 for double precision).
"""

import numpy as np
import time
from typing import Tuple, Optional
from sparse_matrix import BandedSPDMatrix, residual_norm, banded_cholesky_solve


def linpack_benchmark_dense(n: int) -> Tuple[float, float, float]:
    """
    Dense LINPACK-style benchmark using numpy's LU solver.

    Returns
    -------
    time_sec   : Wall-clock time for solve.
    mflops     : Mega-FLOPS achieved.
    norm_res   : Normalized residual.
    """
    # Generate well-conditioned random matrix
    A = np.random.randn(n, n)
    A = A @ A.T + n * np.eye(n)  # Make SPD
    x_exact = np.ones(n, dtype=float)
    b = A @ x_exact

    t0 = time.perf_counter()
    x = np.linalg.solve(A, b)
    t1 = time.perf_counter()

    time_sec = t1 - t0
    flops = (2.0 / 3.0) * n ** 3 + 2.0 * n ** 2
    mflops = flops / (1e6 * max(time_sec, 1e-9))

    r = b - A @ x
    norm_r = float(np.linalg.norm(r, ord=np.inf))
    norm_A = float(np.linalg.norm(A, ord=np.inf))
    norm_x = float(np.linalg.norm(x, ord=np.inf))
    eps = np.finfo(float).eps
    norm_res = norm_r / (norm_A * norm_x * eps + 1e-15)

    return time_sec, mflops, norm_res


def banded_benchmark(
    n: int,
    bandwidth: int,
    n_runs: int = 5
) -> Tuple[float, float, float]:
    """
    Benchmark banded Cholesky solve for subdomain-local systems.

    Returns
    -------
    avg_time   : Average solve time in seconds.
    mflops     : Estimated MFLOPS.
    avg_res    : Average normalized residual.
    """
    times = []
    ress = []
    for _ in range(n_runs):
        # Random SPD banded matrix
        A = BandedSPDMatrix(n, bandwidth)
        for i in range(n):
            diag_val = 2.0 * bandwidth + 1.0 + np.random.rand()
            A.set(i, i, diag_val)
            for off in range(1, bandwidth + 1):
                if i + off < n:
                    off_val = np.random.rand() * 0.1
                    A.set(i + off, i, off_val)

        x_exact = np.ones(n, dtype=float)
        # Band matvec
        b = np.zeros(n, dtype=float)
        for j in range(n):
            for i in range(j, min(n, j + bandwidth + 1)):
                v = A.get(i, j)
                b[i] += v * x_exact[j]
                if i != j:
                    b[j] += v * x_exact[i]

        t0 = time.perf_counter()
        x = banded_cholesky_solve(A, b)
        t1 = time.perf_counter()

        times.append(t1 - t0)
        res = residual_norm(A, x, b)
        ress.append(res)

    avg_time = float(np.mean(times))
    # FLOPs for banded Cholesky: ~ n * bandwidth^2
    flops = n * bandwidth ** 2
    mflops = flops / (1e6 * max(avg_time, 1e-9))
    avg_res = float(np.mean(ress))
    return avg_time, mflops, avg_res


def subdomain_solver_benchmark(
    subdomain_sizes: list,
    bandwidth: int = 3,
    n_runs: int = 3
) -> dict:
    """
    Benchmark suite across multiple subdomain sizes.
    Returns a dictionary of results.
    """
    results = {}
    for n in subdomain_sizes:
        t, m, r = banded_benchmark(n, bandwidth, n_runs)
        results[n] = {"time": t, "mflops": m, "residual": r}
    return results


def parallel_efficiency_estimate(
    serial_time: float,
    n_subdomains: int,
    comm_fraction: float = 0.1
) -> float:
    """
    Amdahl's law estimate of parallel speedup:
        S(p) = 1 / (s + (1-s)/p)
    where s is the serial fraction.

    Efficiency = S(p) / p.
    """
    if n_subdomains <= 0:
        return 0.0
    if serial_time <= 0:
        return 1.0
    s = max(0.0, min(1.0, comm_fraction))
    speedup = 1.0 / (s + (1.0 - s) / n_subdomains)
    efficiency = speedup / n_subdomains
    return float(efficiency)


def report_benchmark(results: dict):
    """Print formatted benchmark report."""
    print("=" * 60)
    print("Subdomain Solver Benchmark Report")
    print("=" * 60)
    print(f"{'Size':>10} {'Time(s)':>12} {'MFLOPS':>12} {'Residual':>14}")
    print("-" * 60)
    for n, res in sorted(results.items()):
        print(f"{n:>10} {res['time']:>12.4e} {res['mflops']:>12.2f} {res['residual']:>14.4e}")
    print("=" * 60)
