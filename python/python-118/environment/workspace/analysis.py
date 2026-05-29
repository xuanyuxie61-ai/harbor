"""
analysis.py

Statistical analysis tools for solid-liquid interface dynamics.

Synthesizes concepts from:
    - 081_besselzero: Bessel functions for radial distribution function
    - 641_laguerre_polynomial: Radial basis expansion
    - 638_lagrange_nd: N-dimensional interpolation for smooth field reconstruction
    
Physical Model:
    Radial Distribution Function (RDF):
        g(r) = (V / N^2) * sum_{i != j} delta(r - r_{ij}) / (4*pi*r^2*dr)
    
    The RDF measures the probability of finding an atom at distance r
    from a reference atom, normalized by the ideal gas probability.
    
    For a perfect FCC crystal, g(r) shows sharp peaks at:
        r/a = sqrt(2)/2, 1, sqrt(3/2), sqrt(2), sqrt(5/2), ...
    
    In the liquid, these peaks broaden and diminish with distance.
    
    Mean Square Displacement (MSD):
        MSD(t) = < |r(t) - r(0)|^2 >
    
    In the diffusive regime:
        MSD(t) = 6 * D * t
    
    where D is the diffusion coefficient.
    
    Velocity Autocorrelation Function (VACF):
        C_vv(t) = < v(t) . v(0) > / < v(0)^2 >
    
    The diffusion coefficient can also be computed from the VACF:
        D = (1/3) * integral_0^inf C_vv(t) dt
    
    Structure Factor:
        S(q) = (1/N) * sum_{i,j} exp(i * q . (r_i - r_j))
    
    For a crystal, S(q) has sharp Bragg peaks; for a liquid, it is smooth.
"""

import numpy as np
from scipy.special import jv
from config import R_CUTOFF, BOX_X, BOX_Y, BOX_Z


# =============================================================================
# Radial Distribution Function
# =============================================================================

def compute_rdf(positions, box, species=None, dr=0.05, r_max=None,
                target_species=None, neighbor_species=None):
    """
    Compute radial distribution function g(r).
    
    g(r) = (V / N^2) * (1 / (4*pi*r^2*dr)) * sum_{i != j} H(r - r_{ij}) * H(r_{ij} - (r-dr))
    
    where H is the Heaviside step function.
    
    Args:
        positions: (N, 3) array
        box: (3,) array
        species: (N,) array, if provided compute partial RDF
        dr: bin width
        r_max: maximum distance
        target_species: species index for central atoms
        neighbor_species: species index for neighbor atoms
        
    Returns:
        r_bins: bin centers
        g_r: RDF values
    """
    n_atoms = positions.shape[0]
    
    if r_max is None:
        r_max = min(box) / 2.0
    
    n_bins = int(r_max / dr)
    r_bins = np.linspace(0, r_max, n_bins + 1)
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    
    counts = np.zeros(n_bins)
    
    volume = np.prod(box)
    
    # Select atoms
    if target_species is not None and species is not None:
        i_list = np.where(species == target_species)[0]
    else:
        i_list = range(n_atoms)
    
    n_targets = len(i_list)
    
    for i in i_list:
        for j in range(n_atoms):
            if i == j:
                continue
            if neighbor_species is not None and species is not None:
                if species[j] != neighbor_species:
                    continue
            
            dr_vec = positions[j] - positions[i]
            dr_vec -= box * np.round(dr_vec / box)
            r = np.sqrt(np.sum(dr_vec ** 2))
            
            if r < r_max:
                bin_idx = int(r / dr)
                if bin_idx < n_bins:
                    counts[bin_idx] += 1
    
    # Normalize
    # Number of pairs
    if neighbor_species is not None and species is not None:
        n_neighbors = np.sum(species == neighbor_species)
    else:
        n_neighbors = n_atoms
    
    # Ideal gas normalization: 4*pi*r^2*dr * (N/V)
    rho = n_neighbors / volume
    
    for b in range(n_bins):
        shell_volume = 4.0 * np.pi * r_centers[b] ** 2 * dr
        ideal_count = n_targets * rho * shell_volume
        if ideal_count > 0:
            counts[b] /= ideal_count
    
    return r_centers, counts


# =============================================================================
# Mean Square Displacement
# =============================================================================

def compute_msd(trajectory, box, time_step=1.0):
    """
    Compute mean square displacement from trajectory.
    
    MSD(t) = (1/N) * sum_i |r_i(t) - r_i(0)|^2
    
    For diffusive motion: MSD(t) = 6 * D * t
    
    Args:
        trajectory: list or array of positions over time
                   shape (n_frames, N, 3)
        box: (3,) array
        time_step: time between frames
        
    Returns:
        times: time array
        msd: MSD values
        diffusion_coeff: estimated diffusion coefficient
    """
    trajectory = np.asarray(trajectory)
    n_frames, n_atoms, _ = trajectory.shape
    
    # Unwrap trajectories (remove periodic boundary jumps)
    trajectory_unwrapped = unwrap_trajectory(trajectory, box)
    
    ref_positions = trajectory_unwrapped[0]
    
    times = np.arange(n_frames) * time_step
    msd = np.zeros(n_frames)
    
    for t in range(n_frames):
        displacements = trajectory_unwrapped[t] - ref_positions
        msd[t] = np.mean(np.sum(displacements ** 2, axis=1))
    
    # Estimate diffusion coefficient from linear fit
    # MSD = 6*D*t for 3D
    fit_start = n_frames // 4
    if fit_start < n_frames - 1:
        slope, intercept = np.polyfit(times[fit_start:], msd[fit_start:], 1)
        diffusion_coeff = slope / 6.0
    else:
        diffusion_coeff = 0.0
    
    return times, msd, diffusion_coeff


def unwrap_trajectory(trajectory, box):
    """
    Remove periodic boundary condition jumps from trajectory.
    
    Args:
        trajectory: (n_frames, N, 3) array
        box: (3,) array
        
    Returns:
        unwrapped: (n_frames, N, 3) array
    """
    n_frames = trajectory.shape[0]
    unwrapped = trajectory.copy()
    
    for t in range(1, n_frames):
        delta = trajectory[t] - trajectory[t - 1]
        delta -= box * np.round(delta / box)
        unwrapped[t] = unwrapped[t - 1] + delta
    
    return unwrapped


# =============================================================================
# Velocity Autocorrelation Function
# =============================================================================

def compute_vacf(velocities_history, dt=1.0):
    """
    Compute velocity autocorrelation function.
    
    C_vv(t) = < v(t0 + t) . v(t0) > / < v(t0)^2 >
    
    Averaged over all time origins t0.
    
    Args:
        velocities_history: (n_frames, N, 3) array
        dt: time step
        
    Returns:
        times: time array
        vacf: normalized VACF
    """
    n_frames, n_atoms, _ = velocities_history.shape
    
    max_lag = n_frames // 2
    vacf = np.zeros(max_lag)
    counts = np.zeros(max_lag)
    
    # Normalize by initial velocity squared
    v0_sq = np.mean(np.sum(velocities_history[0] ** 2, axis=1))
    
    for lag in range(max_lag):
        for t0 in range(n_frames - lag):
            corr = np.mean(np.sum(velocities_history[t0] * velocities_history[t0 + lag], axis=1))
            vacf[lag] += corr
            counts[lag] += 1
    
    vacf /= counts
    vacf /= v0_sq
    
    times = np.arange(max_lag) * dt
    
    return times, vacf


# =============================================================================
# Bessel Function Expansion of RDF (from 081_besselzero)
# =============================================================================

def bessel_expansion_rdf(r, g_r, n_terms=5):
    """
    Expand RDF using Bessel function basis.
    
    g(r) = sum_{k=1}^K c_k * J_0(alpha_k * r / R_max)
    
    where alpha_k are zeros of J_0.
    
    This provides a smooth representation of the RDF.
    
    Args:
        r: radial distances
        g_r: RDF values
        n_terms: number of Bessel terms
        
    Returns:
        coefficients: Bessel coefficients
        g_smooth: smoothed RDF
    """
    from utils import bessel_zero_j
    
    r_max = np.max(r)
    
    # Get Bessel zeros
    zeros = np.array([bessel_zero_j(0, k + 1) for k in range(n_terms)])
    
    # Construct basis matrix
    n_points = len(r)
    basis = np.zeros((n_points, n_terms))
    
    for k in range(n_terms):
        alpha = zeros[k] / r_max
        basis[:, k] = jv(0, alpha * r)
    
    # Least squares fit
    coefficients, residuals, rank, s = np.linalg.lstsq(basis, g_r, rcond=None)
    
    g_smooth = basis @ coefficients
    
    return coefficients, g_smooth


# =============================================================================
# Laguerre Expansion of Density Profile
# =============================================================================

def laguerre_expansion_density(z, rho_z, n_terms=8, alpha=0.0):
    """
    Expand density profile using generalized Laguerre polynomials.
    
    rho(z) = sum_{n=0}^{N-1} c_n * L_n^{(alpha)}(z) * z^alpha * exp(-z)
    
    for z >= 0 (map to positive domain first).
    
    Args:
        z: spatial coordinates
        rho_z: density values
        n_terms: number of expansion terms
        alpha: generalized Laguerre parameter
        
    Returns:
        coefficients: expansion coefficients
        rho_smooth: smoothed density
    """
    from utils import laguerre_polynomial
    
    # Map to positive domain
    z_shifted = z - np.min(z)
    if np.max(z_shifted) > 0:
        z_scaled = z_shifted / np.max(z_shifted) * 10.0  # scale to ~10
    else:
        z_scaled = z_shifted
    
    # Compute Laguerre polynomials
    L = laguerre_polynomial(len(z_scaled), n_terms - 1, z_scaled)
    
    # Weight function
    weight = z_scaled ** alpha * np.exp(-z_scaled)
    
    # Basis: L_n(z) * weight(z)
    basis = L * weight[:, np.newaxis]
    
    # Least squares fit
    coefficients, residuals, rank, s = np.linalg.lstsq(basis, rho_z, rcond=None)
    
    rho_smooth = basis @ coefficients
    
    return coefficients, rho_smooth


# =============================================================================
# Structure Factor
# =============================================================================

def compute_structure_factor(positions, box, n_q=50):
    """
    Compute structure factor S(q).
    
    S(q) = (1/N) * | sum_j exp(i * q . r_j) |^2
    
    Args:
        positions: (N, 3) array
        box: (3,) array
        n_q: number of q-points along each direction
        
    Returns:
        q_magnitudes: q magnitudes
        S_q: structure factor values
    """
    n_atoms = positions.shape[0]
    
    # q-points along z-direction (interface normal)
    qz_values = 2.0 * np.pi / box[2] * np.arange(1, n_q + 1)
    
    S_q = np.zeros(n_q)
    
    for iq, qz in enumerate(qz_values):
        q_vec = np.array([0.0, 0.0, qz])
        
        phase_sum = np.sum(np.exp(1j * np.dot(positions, q_vec)))
        S_q[iq] = np.abs(phase_sum) ** 2 / n_atoms
    
    return qz_values, S_q


# =============================================================================
# Composition Profile and Warren-Cowley Parameters
# =============================================================================

def compute_composition_profile(positions, species, box, n_bins=50):
    """
    Compute composition profile c(z) along interface normal.
    
    c(z) = N_B(z) / (N_A(z) + N_B(z))
    
    Args:
        positions: (N, 3) array
        species: (N,) array
        box: (3,) array
        n_bins: number of bins
        
    Returns:
        z_centers: bin centers
        composition: composition profile
    """
    z = positions[:, 2]
    z_min, z_max = -box[2] / 2, box[2] / 2
    
    bin_edges = np.linspace(z_min, z_max, n_bins + 1)
    z_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    composition = np.zeros(n_bins)
    
    for b in range(n_bins):
        mask = (z >= bin_edges[b]) & (z < bin_edges[b + 1])
        if np.any(mask):
            n_B = np.sum(species[mask] == 1)
            n_total = np.sum(mask)
            composition[b] = n_B / n_total
    
    return z_centers, composition


def warren_cowley_parameter(positions, species, neighbors, dists_sq, rcut):
    """
    Compute Warren-Cowley short-range order parameter.
    
    alpha_{AB} = 1 - P_{AB} / (x_B)
    
    where P_{AB} is the probability of finding a B neighbor around an A atom,
    and x_B is the overall composition of B.
    
    alpha = 0: random alloy
    alpha < 0: ordering (unlike neighbors preferred)
    alpha > 0: clustering (like neighbors preferred)
    
    Args:
        positions: (N, 3) array
        species: (N,) array
        neighbors: list of neighbor lists
        dists_sq: list of squared distances
        rcut: neighbor cutoff
        
    Returns:
        alpha_AB: Warren-Cowley parameter for A-B pairs
    """
    n_atoms = len(species)
    x_B = np.mean(species == 1)
    x_A = 1.0 - x_B
    
    if x_B < 1e-10 or x_A < 1e-10:
        return 0.0
    
    # Count A-B bonds
    n_AB = 0
    n_A_total = 0
    
    for i in range(n_atoms):
        if species[i] != 0:
            continue
        
        n_neighbors = 0
        n_B_neighbors = 0
        
        for idx_j, r_sq in enumerate(dists_sq[i]):
            if np.sqrt(r_sq) < rcut:
                j = neighbors[i][idx_j]
                n_neighbors += 1
                if species[j] == 1:
                    n_B_neighbors += 1
        
        if n_neighbors > 0:
            n_AB += n_B_neighbors
            n_A_total += n_neighbors
    
    if n_A_total == 0:
        return 0.0
    
    P_AB = n_AB / n_A_total
    alpha_AB = 1.0 - P_AB / x_B
    
    return alpha_AB
