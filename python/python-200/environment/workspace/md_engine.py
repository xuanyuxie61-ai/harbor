
import numpy as np
from typing import Callable, Tuple, Optional
from potential_models import total_forces_lj, total_potential_lj


class MDEngine:

    def __init__(self, n_particles: int, dim: int = 2,
                 mass: float = 1.0, dt: float = 0.001,
                 box_size: float = 10.0,
                 epsilon: float = 1.0, sigma: float = 1.0,
                 rcut: float = 2.5,
                 temperature: float = 1.0,
                 tau_thermostat: float = 0.1):
        self.n = n_particles
        self.dim = dim
        self.mass = mass
        self.dt = dt
        self.box = box_size
        self.epsilon = epsilon
        self.sigma = sigma
        self.rcut = rcut
        self.target_temp = temperature
        self.tau = tau_thermostat


        self.pos = np.zeros((n_particles, dim))
        self.vel = np.zeros((n_particles, dim))
        self.acc = np.zeros((n_particles, dim))
        self.force = np.zeros((n_particles, dim))


        self.time_history = []
        self.potential_history = []
        self.kinetic_history = []
        self.total_energy_history = []
        self.temperature_history = []
        self.pressure_history = []

    def initialize_positions_lattice(self, lattice_type: str = "square"):
        if lattice_type == "square":
            n_side = int(np.ceil(np.sqrt(self.n)))
            spacing = self.box / n_side
            idx = 0
            for i in range(n_side):
                for j in range(n_side):
                    if idx >= self.n:
                        break
                    self.pos[idx, 0] = (i + 0.5) * spacing
                    self.pos[idx, 1] = (j + 0.5) * spacing
                    if self.dim == 3:
                        self.pos[idx, 2] = self.box / 2.0
                    idx += 1
                if idx >= self.n:
                    break
        elif lattice_type == "hexagonal":

            spacing = self.sigma * 1.12
            n_x = int(np.ceil(self.box / spacing))
            n_y = int(np.ceil(self.box / (spacing * np.sqrt(3.0) / 2.0)))
            idx = 0
            for j in range(n_y):
                offset = 0.0 if j % 2 == 0 else spacing * 0.5
                for i in range(n_x):
                    if idx >= self.n:
                        break
                    x = offset + i * spacing
                    y = j * spacing * np.sqrt(3.0) / 2.0
                    if x < self.box and y < self.box:
                        self.pos[idx, 0] = x
                        self.pos[idx, 1] = y
                        idx += 1

            while idx < self.n:
                self.pos[idx] = np.random.rand(self.dim) * self.box
                idx += 1
        else:

            self.pos = np.random.rand(self.n, self.dim) * self.box

    def initialize_velocities_maxwell_boltzmann(self):
        std = np.sqrt(self.target_temp / self.mass)
        self.vel = np.random.normal(0.0, std, (self.n, self.dim))

        v_cm = np.mean(self.vel, axis=0)
        self.vel -= v_cm

    def apply_periodic_boundary(self):
        self.pos -= self.box * np.floor(self.pos / self.box)

    def compute_forces_and_energies(self) -> Tuple[float, float]:
        self.force = total_forces_lj(self.pos, self.epsilon,
                                      self.sigma, self.rcut, self.box)
        potential = total_potential_lj(self.pos, self.epsilon,
                                       self.sigma, self.rcut, self.box)
        kinetic = 0.5 * self.mass * np.sum(self.vel ** 2)
        return potential, kinetic

    def compute_temperature(self, kinetic: float) -> float:
        dof = self.n * self.dim
        if dof < 1:
            return 0.0
        return 2.0 * kinetic / dof

    def compute_pressure(self, potential: float, kinetic: float) -> float:
        volume = self.box ** self.dim
        n_kb_t = 2.0 * kinetic / self.dim

        from potential_models import virial_stress_lj
        stress = virial_stress_lj(self.pos, self.epsilon, self.sigma,
                                   self.rcut, volume, self.box)
        virial = np.trace(stress) * volume
        pressure = (2.0 * kinetic + virial) / (self.dim * volume)
        return pressure

    def berendsen_thermostat(self, current_temp: float):
        if current_temp < 1e-12:
            return
        lam = np.sqrt(1.0 + self.dt / self.tau * (self.target_temp / current_temp - 1.0))
        lam = np.clip(lam, 0.8, 1.2)
        self.vel *= lam

    def velocity_verlet_step(self, apply_thermostat: bool = True):


        raise NotImplementedError("Hole_2: 请补全 velocity_verlet_step 积分方案")

    def run(self, n_steps: int, equilibration_steps: int = 100,
            apply_thermostat: bool = True) -> dict:

        _, _ = self.compute_forces_and_energies()
        self.acc = self.force / self.mass

        self.time_history.clear()
        self.potential_history.clear()
        self.kinetic_history.clear()
        self.total_energy_history.clear()
        self.temperature_history.clear()
        self.pressure_history.clear()

        for step in range(n_steps):
            pot, kin, temp, press, etot = self.velocity_verlet_step(
                apply_thermostat=(apply_thermostat and step < equilibration_steps)
            )
            t = step * self.dt
            self.time_history.append(t)
            self.potential_history.append(pot)
            self.kinetic_history.append(kin)
            self.total_energy_history.append(etot)
            self.temperature_history.append(temp)
            self.pressure_history.append(press)

        return {
            'time': np.array(self.time_history),
            'potential': np.array(self.potential_history),
            'kinetic': np.array(self.kinetic_history),
            'total_energy': np.array(self.total_energy_history),
            'temperature': np.array(self.temperature_history),
            'pressure': np.array(self.pressure_history),
        }

    def get_final_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.pos.copy(), self.vel.copy(), self.acc.copy()
