
import numpy as np
from config import (
    EPSILON_AA, EPSILON_BB, EPSILON_AB,
    SIGMA_AA, SIGMA_BB, SIGMA_AB,
    EAM_A, EAM_B, EAM_RHO0,
    R_CUTOFF, R_CUTOFF_SQ, MAX_FORCE
)


def lj_pair_potential(r, epsilon, sigma):
    sr = sigma / r
    sr6 = sr ** 6
    sr12 = sr6 ** 2
    return 4.0 * epsilon * (sr12 - sr6)


def lj_pair_force(r, epsilon, sigma):
    sr = sigma / r
    sr6 = sr ** 6
    sr12 = sr6 ** 2
    return 24.0 * epsilon / r * (2.0 * sr12 - sr6)


def electron_density(r, f0=1.0, eta=2.0, r0=2.5):
    return f0 * np.exp(-eta * (r - r0) ** 2)


def electron_density_derivative(r, f0=1.0, eta=2.0, r0=2.5):
    return -2.0 * eta * (r - r0) * f0 * np.exp(-eta * (r - r0) ** 2)


def embedding_energy(rho, A=EAM_A, B=EAM_B):
    rho_safe = np.maximum(rho, 1e-10)
    return -A * np.sqrt(rho_safe) + B * rho_safe ** 2


def embedding_energy_derivative(rho, A=EAM_A, B=EAM_B):
    rho_safe = np.maximum(rho, 1e-10)
    return -A / (2.0 * np.sqrt(rho_safe)) + 2.0 * B * rho_safe


def compute_energy_and_forces(positions, species, box, neighbors, dists_sq):
    n_atoms = positions.shape[0]
    
    eps_matrix = np.array([
        [EPSILON_AA, EPSILON_AB],
        [EPSILON_AB, EPSILON_BB]
    ])
    sig_matrix = np.array([
        [SIGMA_AA, SIGMA_AB],
        [SIGMA_AB, SIGMA_BB]
    ])
    

    rho = np.zeros(n_atoms)
    
    for i in range(n_atoms):
        if len(dists_sq[i]) == 0:
            continue
        r_arr = np.sqrt(np.array(dists_sq[i]))
        mask = r_arr < R_CUTOFF
        rho[i] += np.sum(electron_density(r_arr[mask]))
        

        for idx_j, r in zip(np.array(neighbors[i])[mask], r_arr[mask]):
            rho[idx_j] += electron_density(r)
    

    embed_energy = embedding_energy(rho)
    dF_drho = embedding_energy_derivative(rho)
    

    pair_energy = 0.0
    forces = np.zeros((n_atoms, 3))
    virial = 0.0
    
    for i in range(n_atoms):
        if len(neighbors[i]) == 0:
            continue
        
        j_arr = np.array(neighbors[i])
        r_arr = np.sqrt(np.array(dists_sq[i]))
        mask = (r_arr < R_CUTOFF) & (r_arr > 1e-3)
        
        if not np.any(mask):
            continue
        
        j_valid = j_arr[mask]
        r_valid = r_arr[mask]
        
        si = species[i]
        sj = species[j_valid]
        
        eps = eps_matrix[si, sj]
        sig = sig_matrix[si, sj]
        

        sr = sig / r_valid
        sr6 = sr ** 6
        sr12 = sr6 ** 2
        pot = 4.0 * eps * (sr12 - sr6)
        fmag = 24.0 * eps / r_valid * (2.0 * sr12 - sr6)
        

        df_dr = -2.0 * 2.0 * (r_valid - 2.5) * np.exp(-2.0 * (r_valid - 2.5) ** 2)
        f_embed = dF_drho[i] * df_dr + dF_drho[j_valid] * df_dr
        
        f_total = fmag + f_embed
        f_total = np.clip(f_total, -MAX_FORCE, MAX_FORCE)
        

        dr = positions[j_valid] - positions[i]
        dr -= box * np.round(dr / box)
        
        f_vec = (f_total[:, np.newaxis] * dr) / r_valid[:, np.newaxis]
        
        forces[i] -= np.sum(f_vec, axis=0)
        forces[j_valid] += f_vec
        
        pair_energy += np.sum(pot)
        virial += np.sum(f_total * r_valid)
    
    pair_energy *= 0.5
    total_energy = np.sum(embed_energy) + pair_energy
    
    return total_energy, forces, virial


def compute_pressure(positions, forces, box, temperature, n_atoms):
    from config import BOLTZMANN_KB, EV_TO_J, ANGSTROM
    
    kbt = BOLTZMANN_KB * temperature / EV_TO_J
    virial = np.sum(positions * forces)
    pressure = (n_atoms * kbt + virial / 3.0) / (np.prod(box))
    
    return pressure
