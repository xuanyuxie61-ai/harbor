
import numpy as np
from config import (
    TARGET_TEMPERATURE, BOLTZMANN_KB, EV_TO_J,
    MC_SWAP_PROBABILITY, MC_N_SWAPS_PER_CYCLE, RANDOM_SEED
)


def metropolis_acceptance(delta_e, temperature, rng=None):
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)
    
    if delta_e <= 0:
        return True
    
    kB_eV = BOLTZMANN_KB / EV_TO_J
    beta = 1.0 / (kB_eV * temperature)
    
    prob = np.exp(-beta * delta_e)
    return rng.random() < prob


def attempt_species_swap(positions, species, masses, energy_func, temperature,
                         box, rng=None, max_trials=100):
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)
    
    n_atoms = len(species)
    new_species = species.copy()
    new_masses = masses.copy()
    

    for _ in range(max_trials):
        i = rng.integers(0, n_atoms)
        


        j = rng.integers(0, n_atoms)
        if i == j:
            continue
        
        if species[i] != species[j]:

            new_species[i] = species[j]
            new_species[j] = species[i]
            new_masses[i] = masses[j]
            new_masses[j] = masses[i]
            

            e_old = energy_func(positions, species)
            e_new = energy_func(positions, new_species)
            delta_e = e_new - e_old
            
            if metropolis_acceptance(delta_e, temperature, rng):
                return True, new_species, new_masses, delta_e
            else:

                return False, species.copy(), masses.copy(), delta_e
    
    return False, species.copy(), masses.copy(), 0.0


def monte_carlo_cycle(positions, species, masses, energy_func, temperature,
                      box, n_swaps=MC_N_SWAPS_PER_CYCLE, rng=None):
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)
    
    n_accepted = 0
    total_delta_e = 0.0
    
    for _ in range(n_swaps):
        accepted, species, masses, delta_e = attempt_species_swap(
            positions, species, masses, energy_func, temperature, box, rng
        )
        
        if accepted:
            n_accepted += 1
            total_delta_e += delta_e
    
    return species, masses, n_accepted, total_delta_e


def ising_model_energy(spins, coupling=1.0, field=0.0, neighbor_pairs=None):
    n_atoms = len(spins)
    energy = 0.0
    
    if neighbor_pairs is None:

        for i in range(n_atoms - 1):
            energy -= coupling * spins[i] * spins[i + 1]
    else:
        for i, j in neighbor_pairs:
            energy -= coupling * spins[i] * spins[j]
    

    energy -= field * np.sum(spins)
    
    return energy


def ising_monte_carlo_step(spins, temperature, coupling=1.0, field=0.0,
                           neighbor_pairs=None, rng=None):
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)
    
    n_atoms = len(spins)
    new_spins = spins.copy()
    
    i = rng.integers(0, n_atoms)
    

    local_field = 0.0
    if neighbor_pairs is not None:
        for ni, nj in neighbor_pairs:
            if ni == i:
                local_field += coupling * spins[nj]
            elif nj == i:
                local_field += coupling * spins[ni]
    
    local_field += field
    

    delta_e = 2.0 * spins[i] * local_field
    
    if metropolis_acceptance(delta_e, temperature, rng):
        new_spins[i] *= -1
        return new_spins, True
    
    return new_spins, False


def compute_mixing_enthalpy(positions, species, neighbors, dists_sq,
                            eps_matrix, sig_matrix, rcut):
    from potential import lj_pair_potential
    
    n_atoms = len(species)
    

    e_AA = 0.0
    e_BB = 0.0
    e_AB = 0.0
    n_AA = 0
    n_BB = 0
    n_AB = 0
    
    for i in range(n_atoms):
        for idx_j, r_sq in enumerate(dists_sq[i]):
            r = np.sqrt(r_sq)
            if r >= rcut:
                continue
            
            j = neighbors[i][idx_j]
            si = species[i]
            sj = species[j]
            eps = eps_matrix[si, sj]
            sig = sig_matrix[si, sj]
            pot = lj_pair_potential(r, eps, sig)
            
            if si == 0 and sj == 0:
                e_AA += pot
                n_AA += 1
            elif si == 1 and sj == 1:
                e_BB += pot
                n_BB += 1
            else:
                e_AB += pot
                n_AB += 1
    

    x_A = np.mean(species == 0)
    x_B = 1.0 - x_A
    
    e_AA_avg = e_AA / max(n_AA, 1)
    e_BB_avg = e_BB / max(n_BB, 1)
    e_AB_avg = e_AB / max(n_AB, 1)
    

    omega = e_AB_avg - 0.5 * (e_AA_avg + e_BB_avg)
    

    delta_h = omega * x_A * x_B
    
    return delta_h
