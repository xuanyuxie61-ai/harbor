# -*- coding: utf-8 -*-
"""
diophantine_phonon.py
---------------------
Enumeration of non-negative integer solutions of linear Diophantine equations
that arise when decomposing multi-phonon scattering channels in the
isotropic Eliashberg formalism.

Scientific origin of the fused algorithms
-----------------------------------------
* Mcnuggets / diophantine_nd_nonnegative   (seed project 742_mcnuggets)
    -> recursive backtracking search over non-negative integer tuples
       (n_1, ..., n_p) such that  a_1 n_1 + ... + a_p n_p = N.

Core physics / mathematics
--------------------------
* In a truncated phonon spectrum with p Einstein branches of frequencies
      Omega_j = a_j * Delta_omega    (j = 1..p),
  the total energy of a p-phonon excitation is  N * Delta_omega when
      a_1 n_1 + ... + a_p n_p = N.
* The number of solutions  W(N) is the degeneracy of the N-th phonon shell
  and enters the density of states as
      g_{ph}(N Delta_omega) = W(N) * g_0
* We implement:
      - diophantine_nd_nonnegative(a, b)   : enumerate all solutions
      - ways_count(a, b)                    : count solutions without storing
      - solvable(a, b)                      : boolean reachability
      - ways_list(a, b)                     : list of solution vectors
      - mcnugget_number_values(a, b_max)    : reachability table up to b_max

Stability / boundary notes
--------------------------
* Empty coefficient vector => no solutions (returns []).
* Right-hand side b < 0     => no solutions.
* Coefficients a_j must be positive integers (checked).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 1. Enumeration of all non-negative integer solutions
# ---------------------------------------------------------------------------
def diophantine_nd_nonnegative(
    a: Sequence[int], b: int
) -> np.ndarray:
    """Return an (K, p) array of all non-negative integer solutions x of
        a_1 x_1 + ... + a_p x_p = b.
    """
    a = np.asarray(a, dtype=int).reshape(-1)
    if a.size == 0 or b < 0 or np.any(a <= 0):
        return np.zeros((0, a.size), dtype=int)

    n = a.size
    solutions: List[List[int]] = []
    y = np.zeros(n, dtype=int)

    def _recurse(j: int, remaining: int) -> None:
        if j < n - 1:
            # Choose y[j] in {0, 1, ..., floor(remaining / a[j])}
            max_val = remaining // a[j]
            for v in range(max_val + 1):
                y[j] = v
                _recurse(j + 1, remaining - v * a[j])
            y[j] = 0
        else:
            # Last variable: must exactly match the remainder
            if remaining % a[j] == 0:
                y[j] = remaining // a[j]
                solutions.append(y.tolist())
            y[j] = 0

    _recurse(0, b)
    if not solutions:
        return np.zeros((0, n), dtype=int)
    return np.array(solutions, dtype=int)


# ---------------------------------------------------------------------------
# 2. Counting solutions without enumerating  (DP)
# ---------------------------------------------------------------------------
def ways_count(a: Sequence[int], b: int) -> int:
    """Number of non-negative integer solutions of  a.x = b."""
    a = np.asarray(a, dtype=int).reshape(-1)
    if a.size == 0 or b < 0 or np.any(a <= 0):
        return 0
    dp = np.zeros(b + 1, dtype=int)
    dp[0] = 1
    for aj in a:
        for r in range(aj, b + 1):
            dp[r] += dp[r - aj]
    return int(dp[b])


# ---------------------------------------------------------------------------
# 3. Reachability  (Frobenius / Mcnugget)
# ---------------------------------------------------------------------------
def solvable(a: Sequence[int], b: int) -> bool:
    """Can b be expressed as a non-negative integer combination of a?"""
    return ways_count(a, b) > 0


def mcnugget_number_values(a: Sequence[int], b_max: int) -> np.ndarray:
    """Return an array R[0..b_max] with R[k] = ways_count(a, k)."""
    a = np.asarray(a, dtype=int).reshape(-1)
    if b_max < 0 or np.any(a <= 0):
        return np.zeros(max(0, b_max + 1), dtype=int)
    dp = np.zeros(b_max + 1, dtype=int)
    dp[0] = 1
    for aj in a:
        for r in range(aj, b_max + 1):
            dp[r] += dp[r - aj]
    return dp


# ---------------------------------------------------------------------------
# 4. Physics driver: phonon-shell degeneracies
# ---------------------------------------------------------------------------
@dataclass
class PhononShellSpectrum:
    """Result of the phonon-shell enumeration.

    Attributes
    ----------
    branch_multiplicities : (p,) int array
        Integer branch labels  a_j  (proportional to Einstein frequencies).
    max_shell : int
        Maximum phonon shell number N considered.
    degeneracy : (max_shell+1,) int array
        W(N) = number of p-tuples (n_1,...,n_p) such that sum a_j n_j = N.
    reachability : (max_shell+1,) bool array
        True if shell N is reachable from the vacuum.
    frobenius_number : int or None
        Largest unreachable N (None if all are reachable above some cutoff).
    """
    branch_multiplicities: np.ndarray
    max_shell: int
    degeneracy: np.ndarray
    reachability: np.ndarray
    frobenius_number: Optional[int]


def phonon_shell_spectrum(
    branch_multiplicities: Sequence[int],
    max_shell: int = 60,
) -> PhononShellSpectrum:
    """Compute the degeneracy spectrum of a truncated multi-Einstein model.

    Parameters
    ----------
    branch_multiplicities : sequence of p positive ints
        Integer branch labels  a_1, ..., a_p  (proportional to the Einstein
        frequencies Omega_j / Delta_omega).
    max_shell : int
        Largest phonon shell number to consider.
    """
    a = np.asarray(branch_multiplicities, dtype=int).reshape(-1)
    if a.size == 0 or np.any(a <= 0):
        raise ValueError("diophantine_phonon: branch_multiplicities must be positive ints")

    deg = mcnugget_number_values(a, max_shell)
    reach = deg > 0
    frob: Optional[int] = None
    # Frobenius number exists only if gcd(a) = 1
    from math import gcd
    from functools import reduce
    g = reduce(gcd, a.tolist())
    if g == 1:
        unreachable = np.where(~reach)[0]
        if unreachable.size > 0:
            frob = int(unreachable[-1])
    return PhononShellSpectrum(
        branch_multiplicities=a,
        max_shell=max_shell,
        degeneracy=deg,
        reachability=reach,
        frobenius_number=frob,
    )


# ---------------------------------------------------------------------------
# 5. Migdal-Eliashberg coupling constants from shell degeneracies
# ---------------------------------------------------------------------------
def migdal_lambda_from_shells(
    shell_deg: np.ndarray,
    V_ph: float,
    mu_star: float,
) -> Tuple[float, np.ndarray]:
    """Return (lambda_eff, lambda_per_shell) the effective EPC coupling.

    The Migdal coupling constant on the N-th phonon shell is
        lambda_N = V_ph * W(N) / (N + 1)
    where the 1/(N+1) factor mimics the Matsubara denominator.
    The *effective* coupling entering the McMillan formula is
        lambda_eff = sum_N lambda_N  -  mu_star .
    """
    shell_deg = np.asarray(shell_deg, dtype=float)
    N = np.arange(shell_deg.size, dtype=float)
    lam_per_shell = V_ph * shell_deg / (N + 1.0)
    lam_eff = float(lam_per_shell.sum() - mu_star)
    return lam_eff, lam_per_shell
