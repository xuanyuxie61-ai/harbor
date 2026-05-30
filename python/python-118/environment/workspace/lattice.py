
import numpy as np
from config import (
    LATTICE_CONSTANT, BOX_X, BOX_Y, BOX_Z,
    MASS_A, MASS_B, R_CUTOFF, RANDOM_SEED
)





FCC_BASIS = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 0.5, 0.5],
    [0.5, 0.0, 0.5],
    [0.5, 0.5, 0.0]
])


def create_fcc_lattice(a0, nx, ny, nz, composition=0.5, interface_z=None, interface_width=3.0, rng=None):
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
    

    box = np.array([nx * a0, ny * a0, nz * a0])
    

    if interface_z is None:
        interface_z = box[2] * 0.5
    



    c_solid = composition * 0.8
    c_liquid = composition * 1.2
    c_liquid = min(c_liquid, 0.95)
    c_solid = max(c_solid, 0.05)
    
    is_solid = np.zeros(natoms, dtype=bool)
    
    for i in range(natoms):
        z = positions[i, 2]

        phi = 0.5 * (1.0 + np.tanh((z - interface_z) / (np.sqrt(2.0) * interface_width)))
        c_local = c_solid + (c_liquid - c_solid) * phi
        

        is_solid[i] = (z < interface_z)
        

        if rng.random() < c_local:
            species[i] = 1
        else:
            species[i] = 0
    
    masses = np.where(species == 0, MASS_A, MASS_B)
    

    positions -= box * 0.5
    
    return positions, species, masses, is_solid, box


def add_thermal_displacement(positions, temperature, masses, rng=None):
    from config import BOLTZMANN_KB, EV_TO_J, ANGSTROM, AMU
    
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)
    
    n_atoms = positions.shape[0]
    


    kbt_ev = BOLTZMANN_KB * temperature / EV_TO_J
    

    k_spring = 10.0
    




    k_spring_si = k_spring * EV_TO_J / (ANGSTROM ** 2)
    u_rms = np.sqrt(BOLTZMANN_KB * temperature / k_spring_si) / ANGSTROM
    
    displacements = rng.normal(0.0, u_rms * 0.3, size=(n_atoms, 3))
    
    return positions + displacements


def compute_lattice_energy_per_atom(a0, epsilon, sigma):
    A_12 = 12.13188
    A_6 = 14.45392
    
    x = sigma / a0
    e_coh = 2.0 * epsilon * (A_12 * x ** 12 - A_6 * x ** 6)
    return e_coh
