"""
monte_carlo.py

Monte Carlo moves for alloy equilibration.

Synthesizes concepts from:
    - 601_ising_2d_simulation: Spin-flip Monte Carlo with Metropolis acceptance
    - 1373_uniform: Random number generators for trial moves
    
Physical Model:
    In a binary alloy, Monte Carlo (MC) moves attempt to swap the species
    of two atoms (A <-> B) while maintaining the total number of each species.
    
    This is a semi-grand canonical Monte Carlo move:
        1. Select two atoms i and j with different species
        2. Propose a swap: species_i <-> species_j
        3. Compute energy change dE = E_new - E_old
        4. Accept with probability:
            P_acc = min(1, exp(-beta * dE))
    
    The acceptance probability satisfies detailed balance:
        P(old -> new) / P(new -> old) = exp(-beta * (E_new - E_old))
    
    For the Ising model analogy:
        - Each lattice site has a "spin" s_i = +1 (A) or -1 (B)
        - The Hamiltonian is:
            H = -J * sum_{<i,j>} s_i * s_j - h * sum_i s_i
        where J is the interaction strength and h is the chemical potential difference.
    
    The alloy mixing energy is:
        E_mix = E_AB - (E_AA + E_BB) / 2
    
    Positive E_mix favors phase separation; negative E_mix favors mixing.
"""

import numpy as np
from config import (
    TARGET_TEMPERATURE, BOLTZMANN_KB, EV_TO_J,
    MC_SWAP_PROBABILITY, MC_N_SWAPS_PER_CYCLE, RANDOM_SEED
)


def metropolis_acceptance(delta_e, temperature, rng=None):
    """
    Metropolis acceptance criterion.
    
    P_acc = min(1, exp(-beta * delta_e))
    
    where beta = 1 / (k_B * T).
    
    Args:
        delta_e: energy change (new - old) in eV
        temperature: temperature in K
        rng: random generator
        
    Returns:
        bool: whether to accept the move
    """
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
    """
    Attempt a semi-grand canonical Monte Carlo swap move.
    
    Algorithm:
        1. Randomly select atom i
        2. Find a neighbor j with different species
        3. Compute energy before and after swap
        4. Apply Metropolis criterion
    
    Args:
        positions: (N, 3) array
        species: (N,) array
        masses: (N,) array
        energy_func: callable returning energy given positions, species
        temperature: temperature in K
        box: (3,) array
        rng: random generator
        max_trials: maximum attempts to find different-species pair
        
    Returns:
        accepted: whether swap was accepted
        new_species: updated species array
        new_masses: updated masses array
        delta_e: energy change
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)
    
    n_atoms = len(species)
    new_species = species.copy()
    new_masses = masses.copy()
    
    # Try to find a pair with different species
    for _ in range(max_trials):
        i = rng.integers(0, n_atoms)
        
        # Find a random neighbor within cutoff
        # For simplicity, search all atoms
        j = rng.integers(0, n_atoms)
        if i == j:
            continue
        
        if species[i] != species[j]:
            # Propose swap
            new_species[i] = species[j]
            new_species[j] = species[i]
            new_masses[i] = masses[j]
            new_masses[j] = masses[i]
            
            # Compute energy change
            e_old = energy_func(positions, species)
            e_new = energy_func(positions, new_species)
            delta_e = e_new - e_old
            
            if metropolis_acceptance(delta_e, temperature, rng):
                return True, new_species, new_masses, delta_e
            else:
                # Reject: restore original
                return False, species.copy(), masses.copy(), delta_e
    
    return False, species.copy(), masses.copy(), 0.0


def monte_carlo_cycle(positions, species, masses, energy_func, temperature,
                      box, n_swaps=MC_N_SWAPS_PER_CYCLE, rng=None):
    """
    Perform a Monte Carlo cycle with multiple swap attempts.
    
    Args:
        positions: (N, 3) array
        species: (N,) array
        masses: (N,) array
        energy_func: callable returning energy
        temperature: temperature in K
        box: (3,) array
        n_swaps: number of swap attempts
        rng: random generator
        
    Returns:
        species: updated species array
        masses: updated masses array
        n_accepted: number of accepted moves
        total_delta_e: total energy change
    """
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
    """
    Compute Ising model energy for species configuration.
    
    H = -J * sum_{<i,j>} s_i * s_j - h * sum_i s_i
    
    where s_i = +1 for species A, -1 for species B.
    
    Args:
        spins: (N,) array with +1 or -1
        coupling: J (positive = ferromagnetic, favors same neighbors)
        field: external field h
        neighbor_pairs: list of (i, j) neighbor pairs
        
    Returns:
        energy
    """
    n_atoms = len(spins)
    energy = 0.0
    
    if neighbor_pairs is None:
        # Assume 1D chain for simplicity
        for i in range(n_atoms - 1):
            energy -= coupling * spins[i] * spins[i + 1]
    else:
        for i, j in neighbor_pairs:
            energy -= coupling * spins[i] * spins[j]
    
    # Field term
    energy -= field * np.sum(spins)
    
    return energy


def ising_monte_carlo_step(spins, temperature, coupling=1.0, field=0.0,
                           neighbor_pairs=None, rng=None):
    """
    Single-spin flip Monte Carlo step (Glauber/Metropolis dynamics).
    
    Algorithm:
        1. Select random spin i
        2. Compute energy change for flip: dE = 2 * s_i * (J * sum_j s_j + h)
        3. Accept with Metropolis probability
    
    Args:
        spins: (N,) array
        temperature: temperature
        coupling: J
        field: h
        neighbor_pairs: list of neighbor pairs
        rng: random generator
        
    Returns:
        new_spins, accepted
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)
    
    n_atoms = len(spins)
    new_spins = spins.copy()
    
    i = rng.integers(0, n_atoms)
    
    # Compute local field
    local_field = 0.0
    if neighbor_pairs is not None:
        for ni, nj in neighbor_pairs:
            if ni == i:
                local_field += coupling * spins[nj]
            elif nj == i:
                local_field += coupling * spins[ni]
    
    local_field += field
    
    # Energy change for flip
    delta_e = 2.0 * spins[i] * local_field
    
    if metropolis_acceptance(delta_e, temperature, rng):
        new_spins[i] *= -1
        return new_spins, True
    
    return new_spins, False


def compute_mixing_enthalpy(positions, species, neighbors, dists_sq,
                            eps_matrix, sig_matrix, rcut):
    """
    Compute the mixing enthalpy of the alloy.
    
    Delta_H_mix = E_AB - (x_A * E_AA + x_B * E_BB)
    
    where E_AB is the energy of the alloy and E_AA, E_BB are the energies
    of the pure components.
    
    Args:
        positions: (N, 3) array
        species: (N,) array
        neighbors: list of neighbor lists
        dists_sq: list of squared distances
        eps_matrix: (2, 2) LJ epsilon matrix
        sig_matrix: (2, 2) LJ sigma matrix
        rcut: cutoff distance
        
    Returns:
        mixing_enthalpy per atom in eV
    """
    from potential import lj_pair_potential
    
    n_atoms = len(species)
    
    # Count pair energies by species pair
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
    
    # Average per pair
    x_A = np.mean(species == 0)
    x_B = 1.0 - x_A
    
    e_AA_avg = e_AA / max(n_AA, 1)
    e_BB_avg = e_BB / max(n_BB, 1)
    e_AB_avg = e_AB / max(n_AB, 1)
    
    # Regular solution model parameter
    omega = e_AB_avg - 0.5 * (e_AA_avg + e_BB_avg)
    
    # Mixing enthalpy
    delta_h = omega * x_A * x_B
    
    return delta_h
