
import numpy as np
from config import (
    TARGET_TEMPERATURE, BOLTZMANN_KB,
    NOSE_HOOVER_Q, NOSE_HOOVER_XI, TIME_STEP,
    EV_TO_J, AMU, RANDOM_SEED
)
from utils import r8vec_normal_01


def compute_temperature(velocities, masses):









    
    conversion = 103.642
    
    ke_per_atom = 0.5 * masses * np.sum(velocities ** 2, axis=1)
    total_ke = np.sum(ke_per_atom) * conversion
    
    n_atoms = len(masses)
    n_dof = 3 * n_atoms - 3
    


    kB_eV = BOLTZMANN_KB / EV_TO_J
    temperature = 2.0 * total_ke / (n_dof * kB_eV)
    
    return temperature, total_ke


def langevin_thermostat(velocities, masses, target_temperature, dt, gamma=0.1, rng=None):
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)
    
    n_atoms = len(masses)
    
    c1 = np.exp(-gamma * dt)
    c2 = np.sqrt(1.0 - c1 ** 2)
    

    kB_eV = BOLTZMANN_KB / EV_TO_J
    v_thermal = np.sqrt(kB_eV * target_temperature / masses)
    

    noise = r8vec_normal_01(n_atoms * 3, rng).reshape(n_atoms, 3)
    
    new_velocities = c1 * velocities + c2 * v_thermal[:, np.newaxis] * noise
    
    return new_velocities


class NoseHooverThermostat:
    
    def __init__(self, target_temperature, n_atoms, q_mass=NOSE_HOOVER_Q, dt=TIME_STEP):
        self.target_temperature = target_temperature
        self.n_dof = 3 * n_atoms - 3
        self.q_mass = q_mass
        self.dt = dt
        self.xi = 0.0
        self.eta = 0.0
        
    def apply(self, velocities, masses):

        temperature, ke = compute_temperature(velocities, masses)
        
        kB_eV = BOLTZMANN_KB / EV_TO_J
        ke_target = 0.5 * self.n_dof * kB_eV * self.target_temperature
        

        dxi_dt = 2.0 * (ke - ke_target) / self.q_mass
        self.xi += dxi_dt * self.dt
        

        scale = np.exp(-self.xi * self.dt * 0.5)
        

        scale = np.clip(scale, 0.5, 2.0)
        
        return velocities * scale
    
    def get_conserved_quantity(self, ke, pe):
        kB_eV = BOLTZMANN_KB / EV_TO_J
        thermostat_energy = 0.5 * self.q_mass * self.xi ** 2
        bath_energy = self.n_dof * kB_eV * self.target_temperature * self.eta
        return ke + pe + thermostat_energy + bath_energy


def andersen_thermostat(velocities, masses, target_temperature, nu, dt, rng=None):
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)
    
    n_atoms = len(masses)
    new_velocities = velocities.copy()
    
    kB_eV = BOLTZMANN_KB / EV_TO_J
    
    for i in range(n_atoms):
        if rng.random() < nu * dt:

            sigma_v = np.sqrt(kB_eV * target_temperature / masses[i])
            new_velocities[i] = r8vec_normal_01(3, rng) * sigma_v
    
    return new_velocities
