"""
Discrete Cascade Module
=======================
Based on seed project 1178_subset_sum:
- subset_sum_table.m  →  subset-sum dynamic programming

Physics:
--------
In inverse-Compton (IC) cascades within GRB afterglows, a seed
photon of energy ε_0 upscatters off relativistic electrons,
gaining energy by a factor ~ γ².  After n scatterings, the
photon energy is approximately:

    ε_n ≈ ε_0 · γ_1² · γ_2² · ... · γ_n²

If electrons have discrete Lorentz factors {γ_1, γ_2, ..., γ_m},
the set of achievable photon energies after one scattering is:

    { ε_0 · γ_i² : i = 1..m }

After multiple scatterings, the energy spectrum forms a discrete
subset-sum structure: each additional scattering adds a term
proportional to log(γ_i²) to the total log-energy.

The subset-sum table tracks which total energy shifts ΔE are
achievable as sums of individual electron contributions:

    ΔE = Σ_i w_i   with   w_i ∝ log(γ_i²)

This discrete representation is essential for computing the
line-like features (spectral edges) in the IC cascade spectrum
that would be smoothed out by a purely continuous treatment.

The Compton parameter for each scattering is:

    y_i = (4/3) τ_T γ_i²

and the total cascade compactness is:

    ℓ = (L σ_T) / (R m_e c³)
"""

import numpy as np


def subset_sum_table(target, weights):
    """
    Dynamic-programming subset-sum table.

    table[j] = w means that sum j is achievable with last weight w.

    Parameters
    ----------
    target : int
        Target sum.
    weights : ndarray
        Positive integer weights.

    Returns
    -------
    table : ndarray, shape (target,)
        Subset-sum table (0 = unattainable).
    """
    n = weights.size
    table = np.zeros(target, dtype=int)

    for i in range(n):
        w = int(weights[i])
        if w <= 0 or w > target:
            continue
        for j in range(target - w, -1, -1):
            if j == 0:
                if table[w - 1] == 0:
                    table[w - 1] = w
            elif table[j - 1] != 0 and table[j + w - 1] == 0:
                table[j + w - 1] = w

    return table


def subset_sum_find(target, weights):
    """
    Recover a subset that sums to target using the DP table.

    Returns
    -------
    subset : list or None
        List of weights if found, else None.
    """
    table = subset_sum_table(target, weights)
    if target <= 0 or target > table.size or table[target - 1] == 0:
        return None

    subset = []
    remaining = target
    while remaining > 0:
        w = table[remaining - 1]
        if w == 0:
            return None
        subset.append(w)
        remaining -= w
        if remaining < 0:
            return None

    return subset


def discrete_ic_cascade(epsilon_0, gamma_discrete, n_scatter_max=5):
    """
    Compute the discrete energy spectrum of an inverse-Compton cascade.

    Parameters
    ----------
    epsilon_0 : float
        Seed photon energy (eV).
    gamma_discrete : ndarray
        Discrete electron Lorentz factors.
    n_scatter_max : int
        Maximum number of scatterings.

    Returns
    -------
    energies : list of ndarray
        Achievable photon energies after 1, 2, ..., n_scatter_max scatterings.
    """
    # Convert to integer log-energy weights
    log_factors = np.round(100.0 * np.log(gamma_discrete ** 2)).astype(int)
    log_factors = np.clip(log_factors, 1, None)

    energies_list = []
    current_log = np.round(100.0 * np.log(epsilon_0)).astype(int)

    for _ in range(n_scatter_max):
        # All possible sums of current_log + any combination of log_factors
        target_max = current_log + int(np.sum(log_factors)) + 1
        target_max = min(target_max, 50000)
        table = subset_sum_table(target_max, log_factors)

        achievable = []
        for s in range(target_max):
            if table[s] != 0:
                achievable.append(current_log + s + 1)

        if len(achievable) == 0:
            break

        energies = np.exp(np.array(achievable, dtype=float) / 100.0)
        energies_list.append(energies)
        current_log = int(np.median(achievable))

    return energies_list


def cascade_compactness(L, R):
    """
    Compute the compactness parameter:

        ℓ = (L σ_T) / (R m_e c³)

    Parameters
    ----------
    L : float
        Luminosity (erg/s).
    R : float
        Emission radius (cm).

    Returns
    -------
    ell : float
        Compactness parameter.
    """
    sigma_T = 6.6524587158e-25
    m_e = 9.10938356e-28
    c = 2.99792458e10
    ell = (L * sigma_T) / (R * m_e * c ** 3)
    return ell
