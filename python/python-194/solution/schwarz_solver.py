"""
schwarz_solver.py
=================
Overlapping Schwarz domain decomposition solver for the discrete
Stokes system arising from spectral element discretization.

Integrates concepts from:
  * r8pbl / linpack_bench_backslash (banded SPD direct solves for subdomain problems)
  * diophantine / polyomino_parity (integer constraints for load balancing)
  * hb_to_st (sparse matrix assembly)

Mathematical background
-----------------------
Given the global discrete Stokes system:
    [ A   B^T ] [ u ]   [ f ]
    [ B   0   ] [ p ] = [ g ]

where A is the viscous block (vector Laplacian), B is the divergence
operator, we apply an additive overlapping Schwarz preconditioner.

For each subdomain Omega_i with overlap:
    Solve local saddle-point system restricted to Omega_i:
        [ A_i   B_i^T ] [ u_i ]   [ f_i ]
        [ B_i   0     ] [ p_i ] = [ g_i ]

The additive Schwarz iteration updates:
    u^{k+1} = u^k + sum_i R_i^T A_i^{-1} R_i (f - A u^k - B^T p^k)
    p^{k+1} = p^k + sum_i R_i^T S_i^{-1} (g - B u^k)
where R_i is the restriction operator to subdomain i, and S_i is the
Schur complement of the pressure on subdomain i.

For numerical stability, we solve the local systems via a penalized
Uzawa approach or directly via the banded Cholesky for the velocity
block combined with a pressure correction.

Load balancing via Diophantine constraints:
    Given total elements N_e and n subdomains, find integer vector
    {m_i} such that sum_i m_i = N_e and each m_i is proportional to
    the estimated workload. This is a bounded linear Diophantine problem.
"""

import numpy as np
from typing import List, Tuple, Callable, Optional
from sparse_matrix import BandedSPDMatrix, banded_cholesky_solve


def diophantine_partition(N: int, n_parts: int) -> List[int]:
    """
    Solve the Diophantine partition problem: find integers m_i >= 1
    such that sum_i m_i = N.  This is a simple constructive solution
    using the division algorithm.

    For load balancing, we additionally try to make the m_i as equal
    as possible, satisfying:
        m_i = floor(N / n_parts)  or  ceil(N / n_parts)
        sum_i m_i = N
        m_i >= 1
    """
    if n_parts <= 0 or N < n_parts:
        raise ValueError("Invalid partition parameters.")
    base = N // n_parts
    rem = N % n_parts
    parts = [base] * n_parts
    for i in range(rem):
        parts[i] += 1
    # Ensure all >= 1
    for i in range(n_parts):
        if parts[i] < 1:
            parts[i] = 1
    # Fix sum if needed
    diff = N - sum(parts)
    idx = 0
    while diff != 0:
        if diff > 0:
            parts[idx % n_parts] += 1
            diff -= 1
        else:
            if parts[idx % n_parts] > 1:
                parts[idx % n_parts] -= 1
                diff += 1
        idx += 1
    return parts


def build_local_stokes_matrix(
    n_local: int,
    dx: float,
    nu: float = 1.0,
    penalty: float = 1e-8
) -> Tuple[BandedSPDMatrix, np.ndarray, np.ndarray]:
    """
    Build a simplified 1-D local viscous matrix A_i (banded SPD)
    and discrete divergence operator B_i for a local subdomain.

    For demonstration, we construct a 1-D second-difference operator
    (analogous to the viscous Laplacian in each velocity component)
    with n_local interior nodes.

    Returns
    -------
    A_loc : BandedSPDMatrix representing the local viscous block.
    B_loc : (n_local,) discrete divergence (du/dx approximated by central diff).
    M_loc : (n_local,) mass lumping diagonal.
    """
    if n_local <= 0:
        raise ValueError("n_local must be positive.")
    A_loc = BandedSPDMatrix(n_local, 1)
    coeff = nu / (dx ** 2)
    for i in range(n_local):
        A_loc.set(i, i, 2.0 * coeff + penalty)
        if i > 0:
            A_loc.set(i, i - 1, -coeff)
    B_loc = np.zeros(n_local, dtype=float)
    # Central difference divergence (1D)
    if n_local >= 3:
        B_loc[0] = -1.0 / dx
        B_loc[1] = 1.0 / dx
        for i in range(1, n_local - 1):
            B_loc[i] = -1.0 / dx
            B_loc[i + 1] = 1.0 / dx
    M_loc = np.ones(n_local, dtype=float) * dx
    return A_loc, B_loc, M_loc


def solve_local_schur_complement(
    A_loc: BandedSPDMatrix,
    B_loc: np.ndarray,
    f_loc: np.ndarray,
    g_loc: float,
    max_iter: int = 50,
    tol: float = 1e-10
) -> Tuple[np.ndarray, float]:
    """
    Solve local saddle-point system using Uzawa iteration with
    exact solve for velocity (via banded Cholesky) and Richardson
    update for pressure.

    Algorithm:
        u^{k+1} = A^{-1} (f - B^T p^k)
        p^{k+1} = p^k + alpha * (B u^{k+1} - g)

    where alpha is chosen via a line search or fixed as
    alpha = 1 / (B A^{-1} B^T) approximated by power iteration.
    """
    n = A_loc.n
    u = np.zeros(n, dtype=float)
    p = 0.0

    # Precompute Cholesky factor
    L = A_loc.cholesky_band()

    # Estimate alpha via power iteration on S = B A^{-1} B^T (scalar in 1D)
    # S approx = B^T (A^{-1} B)
    AinvB = A_loc.solve_cholesky(L, B_loc)
    S_est = float(np.dot(B_loc, AinvB))
    if abs(S_est) < 1e-15:
        S_est = 1e-15
    alpha = 1.0 / S_est

    for it in range(max_iter):
        rhs = f_loc - B_loc * p
        u_new = A_loc.solve_cholesky(L, rhs)
        res = float(np.dot(B_loc, u_new)) - g_loc
        p_new = p + alpha * res
        du = np.linalg.norm(u_new - u)
        dp = abs(p_new - p)
        u = u_new
        p = p_new
        if du < tol and dp < tol:
            break
    return u, p


def additive_schwarz_iteration(
    global_n: int,
    subdomains: List[Tuple[int, int]],  # (start, end) inclusive-exclusive
    overlap: int,
    f_global: np.ndarray,
    g_global: np.ndarray,
    dx: float,
    nu: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Additive overlapping Schwarz iteration for a 1-D-like global system.

    Parameters
    ----------
    global_n   : Total number of global degrees of freedom.
    subdomains : List of (start, end) index ranges for each subdomain.
    overlap    : Number of overlapping grid points between subdomains.
    f_global   : Global forcing for velocity.
    g_global   : Global divergence constraint (one per element).
    dx         : Grid spacing.
    nu         : Viscosity.
    max_iter   : Maximum Schwarz iterations.
    tol        : Convergence tolerance.

    Returns
    -------
    u_global   : Global velocity solution.
    p_global   : Global pressure solution (element-centered).
    res_norm   : Final residual norm.
    """
    u_global = np.zeros(global_n, dtype=float)
    p_global = np.zeros(global_n, dtype=float)

    n_sub = len(subdomains)

    # Pre-build local matrices
    local_systems = []
    for start, end in subdomains:
        n_loc = end - start
        if n_loc <= 0:
            n_loc = 1
        A_loc, B_loc, M_loc = build_local_stokes_matrix(n_loc, dx, nu)
        local_systems.append((A_loc, B_loc, M_loc, start, end))

    for it in range(max_iter):
        du_total = np.zeros(global_n, dtype=float)
        dp_total = np.zeros(global_n, dtype=float)
        counts = np.zeros(global_n, dtype=float)

        for idx, (A_loc, B_loc, M_loc, start, end) in enumerate(local_systems):
            n_loc = end - start
            f_loc = f_global[start:end].copy()
            g_loc = float(np.mean(g_global[start:end]))

            # Restrict current global solution for overlap correction
            u_loc = u_global[start:end].copy()
            p_loc = float(np.mean(p_global[start:end]))

            # Local residual
            # r_f = f - A u - B^T p
            # For simplicity in this 1D model, we solve the correction equation
            # A delta_u + B^T delta_p = r_f
            # B delta_u = r_g
            # Here we just solve the full local system with residual RHS
            r_f = f_loc - A_loc.to_dense() @ u_loc - B_loc * p_loc
            r_g = g_loc - float(np.dot(B_loc, u_loc))

            delta_u, delta_p = solve_local_schur_complement(
                A_loc, B_loc, r_f, r_g, max_iter=30, tol=1e-10
            )

            # Additive update with restriction
            du_total[start:end] += delta_u
            dp_total[start:end] += delta_p
            counts[start:end] += 1.0

        # Average overlapping updates
        for i in range(global_n):
            if counts[i] > 0:
                du_total[i] /= counts[i]
                dp_total[i] /= counts[i]

        u_global += du_total
        p_global += dp_total

        # Global residual
        # Approximate global A as DIF2 scaled
        A_global = BandedSPDMatrix.dif2_band(global_n)
        A_global_scaled = BandedSPDMatrix(global_n, 1)
        coeff = nu / (dx ** 2)
        for i in range(global_n):
            A_global_scaled.set(i, i, 2.0 * coeff)
            if i > 0:
                A_global_scaled.set(i, i - 1, -coeff)

        # Simple divergence operator
        Bu = np.zeros(global_n, dtype=float)
        for i in range(global_n - 1):
            Bu[i] = (u_global[i + 1] - u_global[i]) / dx

        res_u = f_global - A_global_scaled.to_dense() @ u_global - Bu * 0.0  # simplified
        res_g = g_global - Bu
        res_norm = float(np.linalg.norm(res_u) + np.linalg.norm(res_g))

        if res_norm < tol:
            break

    return u_global, p_global, res_norm


def partition_domain_1d(
    N: int,
    n_subdomains: int,
    overlap: int
) -> List[Tuple[int, int]]:
    """
    Partition N global points into n_subdomains with specified overlap.
    Uses Diophantine partitioning for balanced load.
    """
    # Each subdomain must have at least overlap+1 interior points
    min_per_sub = overlap + 2
    if N < n_subdomains * min_per_sub:
        # Reduce subdomains or overlap
        n_subdomains = max(1, N // min_per_sub)
        if n_subdomains < 1:
            n_subdomains = 1

    # Diophantine partition of interior points
    interior = N - (n_subdomains - 1) * overlap
    if interior < n_subdomains:
        interior = n_subdomains
        overlap = (N - interior) // max(1, n_subdomains - 1)

    parts = diophantine_partition(interior, n_subdomains)
    ranges = []
    start = 0
    for i, sz in enumerate(parts):
        end = start + sz
        if end > N:
            end = N
        ranges.append((start, end))
        start = end - overlap
        if start < 0:
            start = 0
    return ranges
