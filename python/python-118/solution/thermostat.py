"""
thermostat.py

Temperature control algorithms for molecular dynamics.

Synthesizes concepts from:
    - 1373_uniform: Uniform and normal random number generators
    - 601_ising_2d_simulation: Monte Carlo acceptance criterion for energy fluctuations
    
Physical Model:
    To maintain constant temperature (NVT ensemble), we apply a thermostat
    that modifies the velocities while preserving the canonical distribution.
    
    Langevin Thermostat:
        m_i * dv_i/dt = F_i - gamma * m_i * v_i + sqrt(2 * gamma * m_i * k_B * T) * xi_i(t)
    
    where:
        - gamma is the friction coefficient
        - xi_i(t) is delta-correlated white noise: <xi_i(t) * xi_j(t')> = delta_{ij} * delta(t-t')
    
    The discretized velocity update (with time step dt):
        v_i' = c_1 * v_i + c_2 * sqrt(k_B * T / m_i) * R_i
    
    where:
        c_1 = exp(-gamma * dt)
        c_2 = sqrt(1 - c_1^2)
        R_i ~ N(0, 1) for each component
    
    Nose-Hoover Thermostat:
        Introduces a fictitious variable s with Hamiltonian:
        H_NH = sum_i p_i^2 / (2 * m_i * s^2) + U(r) + p_s^2 / (2 * Q) + g * k_B * T * ln(s)
    
    Equations of motion:
        dr_i/dt = p_i / (m_i * s^2)
        dp_i/dt = F_i - (p_s / Q) * p_i / s^2
        ds/dt = p_s / Q
        dp_s/dt = sum_i p_i^2 / (m_i * s^2) - g * k_B * T
    
    where g = 3N is the number of degrees of freedom and Q is the fictitious mass.
"""

import numpy as np
from config import (
    TARGET_TEMPERATURE, BOLTZMANN_KB,
    NOSE_HOOVER_Q, NOSE_HOOVER_XI, TIME_STEP,
    EV_TO_J, AMU, RANDOM_SEED
)
from utils import r8vec_normal_01


def compute_temperature(velocities, masses):
    """
    Compute instantaneous kinetic temperature.
    
    T = (2 / (3 * N * k_B)) * sum_i (1/2) * m_i * |v_i|^2
    
    In reduced units where k_B = 1:
        T = (2 / (3 * N)) * K
    where K is the total kinetic energy.
    
    Args:
        velocities: (N, 3) array in Angstrom/fs
        masses: (N,) array in AMU
        
    Returns:
        temperature in Kelvin
    """
    # Kinetic energy in eV
    # KE = 0.5 * m * v^2
    # m in kg, v in m/s -> KE in J
    # Convert: m[amu] * 1.6605e-27 kg/amu
    #          v[Angstrom/fs] * 1e5 m/s per Angstrom/fs
    # KE[J] = 0.5 * m[amu] * 1.6605e-27 * v^2 * 1e10
    #       = 0.5 * m * v^2 * 1.6605e-17 J
    #       = 0.5 * m * v^2 * 1.6605e-17 / 1.602e-19 eV
    #       = 0.5 * m * v^2 * 103.642 eV
    
    conversion = 103.642  # eV / (AMU * (Angstrom/fs)^2)
    
    ke_per_atom = 0.5 * masses * np.sum(velocities ** 2, axis=1)
    total_ke = np.sum(ke_per_atom) * conversion  # in eV
    
    n_atoms = len(masses)
    n_dof = 3 * n_atoms - 3  # subtract 3 for center of mass motion
    
    # T = 2 * KE / (n_dof * k_B)
    # k_B in eV/K = 8.617e-5
    kB_eV = BOLTZMANN_KB / EV_TO_J
    temperature = 2.0 * total_ke / (n_dof * kB_eV)
    
    return temperature, total_ke


def langevin_thermostat(velocities, masses, target_temperature, dt, gamma=0.1, rng=None):
    """
    Apply Langevin thermostat velocity update.
    
    The Ornstein-Uhlenbeck process for velocities:
        v(t+dt) = c_1 * v(t) + c_2 * sqrt(k_B * T / m) * R
    
    where:
        c_1 = exp(-gamma * dt)
        c_2 = sqrt(1 - c_1^2)
        R ~ N(0, 1)
    
    This exactly samples the Maxwell-Boltzmann distribution in the limit
    of large gamma * dt.
    
    The Maxwell-Boltzmann velocity distribution:
        P(v) = (m / (2*pi*k_B*T))^{3/2} * exp( -m*v^2 / (2*k_B*T) )
    
    Args:
        velocities: (N, 3) array
        masses: (N,) array
        target_temperature: target temperature in K
        dt: time step in fs
        gamma: friction coefficient in 1/fs
        rng: random generator
        
    Returns:
        new_velocities
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)
    
    n_atoms = len(masses)
    
    c1 = np.exp(-gamma * dt)
    c2 = np.sqrt(1.0 - c1 ** 2)
    
    # Thermal velocity scale
    kB_eV = BOLTZMANN_KB / EV_TO_J
    v_thermal = np.sqrt(kB_eV * target_temperature / masses)  # in reduced units
    
    # Random kicks
    noise = r8vec_normal_01(n_atoms * 3, rng).reshape(n_atoms, 3)
    
    new_velocities = c1 * velocities + c2 * v_thermal[:, np.newaxis] * noise
    
    return new_velocities


class NoseHooverThermostat:
    """
    Nose-Hoover chain thermostat for canonical ensemble sampling.
    
    The extended Hamiltonian is:
        H = sum_i p_i^2/(2*m_i*s^2) + U(r) + p_s^2/(2*Q) + g*k_B*T*ln(s)
    
    Chain formulation (for better ergodicity):
        dxi_1/dt = (2*K - g*k_B*T) / Q_1
        dxi_2/dt = (Q_1*xi_1^2 - k_B*T) / Q_2
        ...
    
    Velocity scaling:
        v_i -> v_i * exp(-xi_1 * dt/2)
    
    Args:
        target_temperature: target temperature
        n_atoms: number of atoms
        q_mass: fictitious mass parameter Q
        dt: time step
    """
    
    def __init__(self, target_temperature, n_atoms, q_mass=NOSE_HOOVER_Q, dt=TIME_STEP):
        self.target_temperature = target_temperature
        self.n_dof = 3 * n_atoms - 3
        self.q_mass = q_mass
        self.dt = dt
        self.xi = 0.0  # thermostat variable
        self.eta = 0.0  # accumulated log scale factor
        
    def apply(self, velocities, masses):
        """
        Apply Nose-Hoover thermostat to velocities.
        
        Uses a simple velocity scaling with the thermostat variable:
            v_new = v * exp(-xi * dt/2)
        
        and updates xi using the energy difference.
        
        Args:
            velocities: (N, 3) array
            masses: (N,) array
            
        Returns:
            scaled_velocities
        """
        # Compute current kinetic energy and temperature
        temperature, ke = compute_temperature(velocities, masses)
        
        kB_eV = BOLTZMANN_KB / EV_TO_J
        ke_target = 0.5 * self.n_dof * kB_eV * self.target_temperature
        
        # Update thermostat variable
        dxi_dt = 2.0 * (ke - ke_target) / self.q_mass
        self.xi += dxi_dt * self.dt
        
        # Scale velocities
        scale = np.exp(-self.xi * self.dt * 0.5)
        
        # Boundary handling: prevent runaway scaling
        scale = np.clip(scale, 0.5, 2.0)
        
        return velocities * scale
    
    def get_conserved_quantity(self, ke, pe):
        """
        Compute the conserved quantity (Hamiltonian + thermostat energy).
        
        H_conserved = K + U + (Q/2) * xi^2 + g * k_B * T * eta
        
        Args:
            ke: kinetic energy
            pe: potential energy
            
        Returns:
            conserved energy
        """
        kB_eV = BOLTZMANN_KB / EV_TO_J
        thermostat_energy = 0.5 * self.q_mass * self.xi ** 2
        bath_energy = self.n_dof * kB_eV * self.target_temperature * self.eta
        return ke + pe + thermostat_energy + bath_energy


def andersen_thermostat(velocities, masses, target_temperature, nu, dt, rng=None):
    """
    Andersen thermostat: stochastic collisions with heat bath.
    
    With probability nu * dt, each atom's velocity is redrawn from the
    Maxwell-Boltzmann distribution:
        P(v) ~ exp( -m*v^2 / (2*k_B*T) )
    
    This creates a Markov chain in phase space that samples the canonical
    distribution.
    
    Args:
        velocities: (N, 3) array
        masses: (N,) array
        target_temperature: target temperature in K
        nu: collision frequency in 1/fs
        dt: time step in fs
        rng: random generator
        
    Returns:
        new_velocities
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)
    
    n_atoms = len(masses)
    new_velocities = velocities.copy()
    
    kB_eV = BOLTZMANN_KB / EV_TO_J
    
    for i in range(n_atoms):
        if rng.random() < nu * dt:
            # Redraw velocity from Maxwell-Boltzmann
            sigma_v = np.sqrt(kB_eV * target_temperature / masses[i])
            new_velocities[i] = r8vec_normal_01(3, rng) * sigma_v
    
    return new_velocities
