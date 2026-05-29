"""
phase_transition.py
===================
Phase transition detection and order parameter analysis for dusty plasma crystals.

Core algorithms:
  - Lindemann melting criterion
  - Radial distribution function g(r)
  - Static structure factor S(q)
  - Phase state classification based on coupling parameter and Lindemann parameter
  - Pair correlation analysis for crystalline ordering
"""

import numpy as np


def lindemann_parameter(positions, equilibrium_positions, a_ws):
    """
    Compute the Lindemann parameter, a key indicator of the melting transition.
    
    L = sqrt( <u^2> ) / a_WS
    
    where <u^2> is the mean-square displacement from equilibrium positions
    and a_WS is the Wigner-Seitz radius.
    
    Melting criterion (Ikezi, 1986; Kalman et al., 2000):
      - L < 0.1   : crystalline phase (plasma crystal)
      - L ~ 0.1   : melting point
      - L > 0.15  : liquid/gaseous phase
    
    For Yukawa systems, the critical Lindemann parameter depends on kappa:
      L_c ~ 0.07 - 0.12 for kappa in [1, 5].
    """
    displacements = positions - equilibrium_positions
    u2 = np.mean(np.sum(displacements**2, axis=1))
    return np.sqrt(max(u2, 0.0)) / a_ws


def radial_distribution_function(positions, dr, r_max):
    """
    Compute the radial distribution function g(r).
    
    g(r) measures the probability of finding a particle at distance r
    relative to a uniform distribution. Peaks indicate shell structure.
    
    In the crystalline phase, g(r) shows sharp peaks at lattice neighbor distances.
    In the liquid phase, peaks are broadened.
    In the gaseous phase, g(r) -> 1 for all r.
    
    Algorithm:
      1. Bin all pairwise distances into histogram bins of width dr
      2. Normalize by ideal gas density in each spherical shell
    """
    N = positions.shape[0]
    dim = positions.shape[1]
    n_bins = max(1, int(r_max / dr))
    g = np.zeros(n_bins, dtype=float)
    counts = np.zeros(n_bins, dtype=float)
    
    for i in range(N):
        for j in range(i + 1, N):
            r = np.linalg.norm(positions[i] - positions[j])
            if r < r_max:
                idx = int(r / dr)
                if idx < n_bins:
                    g[idx] += 2.0
    
    # Normalize by ideal gas shell density
    r_bins = np.arange(n_bins) * dr + dr / 2.0
    for i in range(n_bins):
        r_inner = i * dr
        r_outer = (i + 1) * dr
        if dim == 2:
            shell_vol = np.pi * (r_outer**2 - r_inner**2)
        else:
            shell_vol = (4.0 / 3.0) * np.pi * (r_outer**3 - r_inner**3)
        if shell_vol > 0:
            g[i] /= (N * shell_vol)
    
    return r_bins, g


def structure_factor(q, positions):
    """
    Compute the static structure factor S(q).
    
    S(q) = (1/N) * | sum_j exp(i * q . r_j) |^2
    
    For a crystal, S(q) shows sharp Bragg peaks at reciprocal lattice vectors.
    For a liquid, S(q) has a broad first peak and damped oscillations.
    For a gas, S(q) ~ 1.
    
    The Debye-Waller factor relates peak height to thermal vibrations:
      S(G) ~ N * exp(-G^2 * <u^2> / 2)
    """
    N = positions.shape[0]
    total = 0.0 + 0.0j
    for j in range(N):
        qr = np.dot(q, positions[j])
        total += np.cos(qr) + 1j * np.sin(qr)
    return float(np.abs(total)**2 / N)


def detect_phase(gamma, lindemann, gamma_c=170.0, lindemann_c=0.10):
    """
    Detect the phase of the dusty plasma based on multiple order parameters.
    
    Criteria:
      CRYSTALLINE: Gamma > gamma_c AND L < lindemann_c
      GASEOUS    : Gamma < 0.5 * gamma_c OR L > 2.0 * lindemann_c
      LIQUID     : intermediate regime
    
    The phase diagram of Yukawa systems (Hamaguchi et al., 1997) shows:
      - For kappa = 0 (Coulomb): Gamma_c ~ 170
      - For kappa > 0 (Yukawa): Gamma_c increases with kappa
    """
    if gamma > gamma_c and lindemann < lindemann_c:
        return "CRYSTALLINE"
    elif gamma < 0.5 * gamma_c or lindemann > 2.0 * lindemann_c:
        return "GASEOUS"
    else:
        return "LIQUID"


def compute_bond_orientational_order(positions, n_neighbors=6):
    """
    Compute 2D bond orientational order parameter psi_6 for hexagonal crystals.
    
    psi_6 = (1/N) * sum_j | (1/n_j) * sum_{k in neighbors(j)} exp(i * 6 * theta_{jk}) |
    
    where theta_{jk} is the angle of the bond from particle j to neighbor k.
    
    For perfect hexagonal lattice: psi_6 = 1.0
    For disordered system: psi_6 ~ 0.0
    """
    N = positions.shape[0]
    if N == 0:
        return 0.0
    
    psi_sum = 0.0
    for j in range(N):
        # Find nearest neighbors by distance
        dists = np.array([np.linalg.norm(positions[j] - positions[k]) 
                          for k in range(N) if k != j])
        if len(dists) == 0:
            continue
        cutoff = np.sort(dists)[min(n_neighbors, len(dists)) - 1] * 1.1
        
        local_psi = 0.0 + 0.0j
        n_count = 0
        for k in range(N):
            if j == k:
                continue
            r_jk = positions[k, :2] - positions[j, :2]
            r = np.linalg.norm(r_jk)
            if r < 1e-10:
                continue
            if r <= cutoff:
                theta = np.arctan2(r_jk[1], r_jk[0])
                local_psi += np.cos(6.0 * theta) + 1j * np.sin(6.0 * theta)
                n_count += 1
        
        if n_count > 0:
            psi_sum += np.abs(local_psi / n_count)
    
    return psi_sum / N


def debye_waller_factor(q_vec, u_rms):
    """
    Compute Debye-Waller factor for Bragg peak attenuation.
    
    DWF = exp( -q^2 * <u^2> / 2 )
    
    where <u^2> is the mean-square displacement and q is the reciprocal
    lattice vector magnitude.
    """
    q2 = np.dot(q_vec, q_vec)
    return np.exp(-0.5 * q2 * u_rms**2)
