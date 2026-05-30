
import numpy as np
from scipy.special import jv
from config import R_CUTOFF, BOX_X, BOX_Y, BOX_Z






def compute_rdf(positions, box, species=None, dr=0.05, r_max=None,
                target_species=None, neighbor_species=None):
    n_atoms = positions.shape[0]
    
    if r_max is None:
        r_max = min(box) / 2.0
    
    n_bins = int(r_max / dr)
    r_bins = np.linspace(0, r_max, n_bins + 1)
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    
    counts = np.zeros(n_bins)
    
    volume = np.prod(box)
    

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
    


    if neighbor_species is not None and species is not None:
        n_neighbors = np.sum(species == neighbor_species)
    else:
        n_neighbors = n_atoms
    

    rho = n_neighbors / volume
    
    for b in range(n_bins):
        shell_volume = 4.0 * np.pi * r_centers[b] ** 2 * dr
        ideal_count = n_targets * rho * shell_volume
        if ideal_count > 0:
            counts[b] /= ideal_count
    
    return r_centers, counts






def compute_msd(trajectory, box, time_step=1.0):
    trajectory = np.asarray(trajectory)
    n_frames, n_atoms, _ = trajectory.shape
    

    trajectory_unwrapped = unwrap_trajectory(trajectory, box)
    
    ref_positions = trajectory_unwrapped[0]
    
    times = np.arange(n_frames) * time_step
    msd = np.zeros(n_frames)
    
    for t in range(n_frames):
        displacements = trajectory_unwrapped[t] - ref_positions
        msd[t] = np.mean(np.sum(displacements ** 2, axis=1))
    


    fit_start = n_frames // 4
    if fit_start < n_frames - 1:
        slope, intercept = np.polyfit(times[fit_start:], msd[fit_start:], 1)
        diffusion_coeff = slope / 6.0
    else:
        diffusion_coeff = 0.0
    
    return times, msd, diffusion_coeff


def unwrap_trajectory(trajectory, box):
    n_frames = trajectory.shape[0]
    unwrapped = trajectory.copy()
    
    for t in range(1, n_frames):
        delta = trajectory[t] - trajectory[t - 1]
        delta -= box * np.round(delta / box)
        unwrapped[t] = unwrapped[t - 1] + delta
    
    return unwrapped






def compute_vacf(velocities_history, dt=1.0):
    n_frames, n_atoms, _ = velocities_history.shape
    
    max_lag = n_frames // 2
    vacf = np.zeros(max_lag)
    counts = np.zeros(max_lag)
    

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






def bessel_expansion_rdf(r, g_r, n_terms=5):
    from utils import bessel_zero_j
    
    r_max = np.max(r)
    

    zeros = np.array([bessel_zero_j(0, k + 1) for k in range(n_terms)])
    

    n_points = len(r)
    basis = np.zeros((n_points, n_terms))
    
    for k in range(n_terms):
        alpha = zeros[k] / r_max
        basis[:, k] = jv(0, alpha * r)
    

    coefficients, residuals, rank, s = np.linalg.lstsq(basis, g_r, rcond=None)
    
    g_smooth = basis @ coefficients
    
    return coefficients, g_smooth






def laguerre_expansion_density(z, rho_z, n_terms=8, alpha=0.0):
    from utils import laguerre_polynomial
    

    z_shifted = z - np.min(z)
    if np.max(z_shifted) > 0:
        z_scaled = z_shifted / np.max(z_shifted) * 10.0
    else:
        z_scaled = z_shifted
    

    L = laguerre_polynomial(len(z_scaled), n_terms - 1, z_scaled)
    

    weight = z_scaled ** alpha * np.exp(-z_scaled)
    

    basis = L * weight[:, np.newaxis]
    

    coefficients, residuals, rank, s = np.linalg.lstsq(basis, rho_z, rcond=None)
    
    rho_smooth = basis @ coefficients
    
    return coefficients, rho_smooth






def compute_structure_factor(positions, box, n_q=50):
    n_atoms = positions.shape[0]
    

    qz_values = 2.0 * np.pi / box[2] * np.arange(1, n_q + 1)
    
    S_q = np.zeros(n_q)
    
    for iq, qz in enumerate(qz_values):
        q_vec = np.array([0.0, 0.0, qz])
        
        phase_sum = np.sum(np.exp(1j * np.dot(positions, q_vec)))
        S_q[iq] = np.abs(phase_sum) ** 2 / n_atoms
    
    return qz_values, S_q






def compute_composition_profile(positions, species, box, n_bins=50):
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
    n_atoms = len(species)
    x_B = np.mean(species == 1)
    x_A = 1.0 - x_B
    
    if x_B < 1e-10 or x_A < 1e-10:
        return 0.0
    

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
