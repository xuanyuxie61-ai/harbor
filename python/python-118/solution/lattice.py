"""
lattice.py

Crystal lattice initialization and mesh generation for MD simulation.

Synthesizes concepts from:
    - 1233_tet_mesh_l2q: Tetrahedral mesh topology, linear-to-quadratic nodes
    - 489_grf_display: Graph structure for node connectivity
    
Physical Model:
    FCC (face-centered cubic) lattice for binary alloy.
    
    The conventional cubic cell contains 4 atoms at positions:
        (0, 0, 0)
        (0, 1/2, 1/2)
        (1/2, 0, 1/2)
        (1/2, 1/2, 0)
    
    scaled by lattice constant a0.
    
    For a binary alloy, the A and B atoms occupy lattice sites according to
    a specified composition profile. At the solid-liquid interface, the
    composition varies smoothly from the solid composition to the liquid
    composition.
    
    The interface position is defined by an order parameter field:
        phi(z) = 0.5 * [ 1 + tanh( (z - z_interface) / w ) ]
    where w is the interface width and z_interface is the nominal interface
    position.
"""

import numpy as np
from config import (
    LATTICE_CONSTANT, BOX_X, BOX_Y, BOX_Z,
    MASS_A, MASS_B, R_CUTOFF, RANDOM_SEED
)


# =============================================================================
# FCC Lattice Basis (Conventional Cell)
# =============================================================================
FCC_BASIS = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 0.5, 0.5],
    [0.5, 0.0, 0.5],
    [0.5, 0.5, 0.0]
])


def create_fcc_lattice(a0, nx, ny, nz, composition=0.5, interface_z=None, interface_width=3.0, rng=None):
    """
    Create an FCC supercell with specified dimensions and binary composition.
    
    The supercell is constructed by replicating the conventional cubic cell
    nx x ny x nz times along the x, y, z directions.
    
    Total number of atoms = 4 * nx * ny * nz.
    
    The composition profile along z is:
        c(z) = c_solid + (c_liquid - c_solid) * phi(z)
    where phi(z) is an error-function-like profile:
        phi(z) = 0.5 * [ 1 + erf( (z - z_interface) / (sqrt(2) * w) ) ]
    
    Args:
        a0: lattice constant (Angstrom)
        nx, ny, nz: number of unit cells in each direction
        composition: nominal composition of species B (0 to 1)
        interface_z: z-coordinate of interface center (default: middle of box)
        interface_width: interface width parameter (Angstrom)
        rng: random number generator
        
    Returns:
        positions: (N, 3) array of atomic positions
        species: (N,) array of species indices (0 = A, 1 = B)
        masses: (N,) array of atomic masses
        is_solid: (N,) boolean array indicating solid-phase assignment
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)
    
    natoms = 4 * nx * ny * nz
    positions = np.zeros((natoms, 3))
    species = np.zeros(natoms, dtype=int)
    
    idx = 0
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                for basis in FCC_BASIS:
                    pos = a0 * (np.array([ix, iy, iz]) + basis)
                    positions[idx] = pos
                    idx += 1
    
    # Apply periodic boundary conditions
    box = np.array([nx * a0, ny * a0, nz * a0])
    
    # Determine interface position
    if interface_z is None:
        interface_z = box[2] * 0.5
    
    # Assign species based on composition profile
    # Solid phase (low z): composition c_solid
    # Liquid phase (high z): composition c_liquid
    c_solid = composition * 0.8
    c_liquid = composition * 1.2
    c_liquid = min(c_liquid, 0.95)
    c_solid = max(c_solid, 0.05)
    
    is_solid = np.zeros(natoms, dtype=bool)
    
    for i in range(natoms):
        z = positions[i, 2]
        # Smooth interface profile
        phi = 0.5 * (1.0 + np.tanh((z - interface_z) / (np.sqrt(2.0) * interface_width)))
        c_local = c_solid + (c_liquid - c_solid) * phi
        
        # Determine phase assignment
        is_solid[i] = (z < interface_z)
        
        # Random assignment based on local composition
        if rng.random() < c_local:
            species[i] = 1  # Species B
        else:
            species[i] = 0  # Species A
    
    masses = np.where(species == 0, MASS_A, MASS_B)
    
    # Center the system
    positions -= box * 0.5
    
    return positions, species, masses, is_solid, box


def add_thermal_displacement(positions, temperature, masses, rng=None):
    """
    Add random thermal displacements to atomic positions according to
    the equipartition theorem.
    
    The mean square displacement in each direction is:
        <u^2> = k_B * T / (m * omega^2)
    
    For a rough estimate using the Einstein model:
        <u^2> ~ 3 * k_B * T / (m * omega_E^2)
    
    where omega_E is the Einstein frequency, approximately:
        omega_E ~ sqrt( k_spring / m )
    
    with k_spring ~ epsilon / sigma^2 in LJ units.
    
    Args:
        positions: (N, 3) array
        temperature: temperature in K
        masses: (N,) array in AMU
        rng: random generator
        
    Returns:
        displaced positions
    """
    from config import BOLTZMANN_KB, EV_TO_J, ANGSTROM, AMU
    
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)
    
    n_atoms = positions.shape[0]
    
    # Estimate Einstein frequency
    # k_B * T in eV
    kbt_ev = BOLTZMANN_KB * temperature / EV_TO_J
    
    # Rough spring constant in eV/Angstrom^2
    k_spring = 10.0  # eV/Angstrom^2
    
    # RMS displacement
    # u_rms = sqrt(k_B * T / k_spring)
    # But this is in reduced units; convert properly
    # k_B * T [J] / (k_spring [J/m^2]) -> m^2 -> Angstrom^2
    k_spring_si = k_spring * EV_TO_J / (ANGSTROM ** 2)  # J/m^2
    u_rms = np.sqrt(BOLTZMANN_KB * temperature / k_spring_si) / ANGSTROM
    
    displacements = rng.normal(0.0, u_rms * 0.3, size=(n_atoms, 3))
    
    return positions + displacements


def compute_lattice_energy_per_atom(a0, epsilon, sigma):
    """
    Compute the cohesive energy of a perfect FCC lattice using the
    Lennard-Jones potential.
    
    The total lattice energy per atom is:
        E_coh = (1/2) * sum_{j != i} 4*epsilon * [ (sigma/r_ij)^12 - (sigma/r_ij)^6 ]
    
    For an FCC lattice, this sum can be evaluated analytically:
        E_coh = 2 * epsilon * [ A_12 * (sigma/a0)^12 - A_6 * (sigma/a0)^6 ]
    
    where A_12 and A_6 are lattice sums:
        A_12 = sum_{j != 0} (a0/r_0j)^12 = 12.13188
        A_6  = sum_{j != 0} (a0/r_0j)^6  = 14.45392
    
    Args:
        a0: lattice constant
        epsilon: LJ well depth
        sigma: LJ diameter
        
    Returns:
        cohesive energy per atom
    """
    A_12 = 12.13188
    A_6 = 14.45392
    
    x = sigma / a0
    e_coh = 2.0 * epsilon * (A_12 * x ** 12 - A_6 * x ** 6)
    return e_coh
