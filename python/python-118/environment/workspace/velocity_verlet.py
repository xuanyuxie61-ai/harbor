
import numpy as np
from utils_numeric import check_bounds, relative_convergence_check


class VelocityVerletNVT:

    def __init__(self, dt=1.0, T_target=1200.0, nhc_chain_length=3, Q_factor=1.0):
        self.dt = float(dt)
        self.T_target = float(T_target)
        self.kb = 8.617333e-5
        self.nhc_length = max(1, int(nhc_chain_length))
        self.Q_factor = float(Q_factor)


        self.xi = None
        self.v_xi = None
        self.Q = None

    def _initialize_nhc(self, n_dof):

        omega = 1.0 / self.dt
        self.Q = np.zeros(self.nhc_length, dtype=np.float64)
        self.Q[0] = self.Q_factor * n_dof * self.kb * self.T_target / (omega ** 2)
        for i in range(1, self.nhc_length):
            self.Q[i] = self.kb * self.T_target / (omega ** 2)
        self.v_xi = np.zeros(self.nhc_length, dtype=np.float64)

    def _nhc_scale_factor(self, v, masses, n_steps=3):




        return 1.0

    def step(self, positions, velocities, masses, species_idx, potential, box):
        n_atoms = positions.shape[0]
        n_dof = 3 * n_atoms
        if self.Q is None:
            self._initialize_nhc(n_dof)


        _, forces, virial = potential.compute_forces_and_energies(positions, species_idx, box)


        scale = self._nhc_scale_factor(velocities, masses, n_steps=3)
        velocities *= scale


        velocities_half = velocities + 0.5 * self.dt * forces / masses[:, None]


        new_positions = positions + self.dt * velocities_half

        new_positions -= box * np.floor(new_positions / box)


        pot_energy, new_forces, new_virial = potential.compute_forces_and_energies(
            new_positions, species_idx, box)


        new_velocities = velocities_half + 0.5 * self.dt * new_forces / masses[:, None]


        scale2 = self._nhc_scale_factor(new_velocities, masses, n_steps=3)
        new_velocities *= scale2


        kinetic = 0.5 * np.sum(masses[:, None] * new_velocities ** 2)
        temperature = 2.0 * kinetic / (n_dof * self.kb)
        total_energy = pot_energy + kinetic


        nhc_energy = 0.0
        for i in range(self.nhc_length):
            nhc_energy += 0.5 * self.Q[i] * self.v_xi[i] ** 2
            if i == 0:
                nhc_energy += n_dof * self.kb * self.T_target * 0.0
        total_energy += nhc_energy

        return new_positions, new_velocities, total_energy, temperature, pot_energy, kinetic, new_virial


class TrapezoidalThermostatIntegrator:

    def __init__(self, dt=1.0, T_target=1200.0, max_iter=10, tol=1e-8):
        self.dt = float(dt)
        self.T_target = float(T_target)
        self.kb = 8.617333e-5
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.xi = 0.0

    def step(self, positions, velocities, masses, species_idx, potential, box):
        n_atoms = positions.shape[0]
        n_dof = 3 * n_atoms
        Q = n_dof * self.kb * self.T_target * (self.dt ** 2) * 10.0


        _, forces, virial = potential.compute_forces_and_energies(positions, species_idx, box)


        v_pred = velocities + self.dt * (forces / masses[:, None] - self.xi * velocities)
        pos_pred = positions + self.dt * v_pred
        pos_pred -= box * np.floor(pos_pred / box)


        xi_old = self.xi
        for iteration in range(self.max_iter):

            v_half = velocities + 0.5 * self.dt * (forces / masses[:, None] - xi_old * velocities)
            pos_new = positions + self.dt * v_half
            pos_new -= box * np.floor(pos_new / box)

            _, f_new, _ = potential.compute_forces_and_energies(pos_new, species_idx, box)
            v_new = v_half + 0.5 * self.dt * (f_new / masses[:, None] - xi_old * v_half)

            K_new = np.sum(masses[:, None] * v_new ** 2)
            G_new = K_new - n_dof * self.kb * self.T_target
            K_old = np.sum(masses[:, None] * velocities ** 2)
            G_old = K_old - n_dof * self.kb * self.T_target

            xi_new = xi_old + 0.5 * self.dt * (G_old + G_new) / Q
            if abs(xi_new - xi_old) < self.tol:
                self.xi = xi_new
                break
            xi_old = xi_new
        else:
            self.xi = xi_old

        kinetic = 0.5 * np.sum(masses[:, None] * v_new ** 2)
        _, f_final, virial_final = potential.compute_forces_and_energies(pos_new, species_idx, box)
        pot_energy = _
        temperature = 2.0 * kinetic / (n_dof * self.kb)
        total_energy = pot_energy + kinetic
        return pos_new, v_new, total_energy, temperature, pot_energy, kinetic, virial_final
