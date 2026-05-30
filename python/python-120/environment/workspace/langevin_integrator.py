
import numpy as np
from typing import Callable, Tuple, Optional, List


class LangevinIntegrator:

    def __init__(self, mass_amu: np.ndarray, gamma_ps: float,
                 temperature_k: float, dt_fs: float = 1.0,
                 n_dims: int = 3):
        from utils import AMU_TO_KG, FS_TO_S, BOLTZMANN_KB
        self.mass = np.asarray(mass_amu, dtype=float) * AMU_TO_KG
        self.gamma = gamma_ps / (1.0e12)
        self.temperature = temperature_k
        self.dt = dt_fs * FS_TO_S
        self.n_dims = n_dims
        self.kb = BOLTZMANN_KB
        self.amu_to_kg = AMU_TO_KG


        self.c1 = np.exp(-self.gamma * self.dt)
        self.c2 = np.sqrt(self.kb * self.temperature * (1.0 - self.c1 ** 2))

        self.positions = None
        self.velocities = None
        self.n_particles = None

    def initialize(self, positions: np.ndarray, velocities: Optional[np.ndarray] = None):
        positions = np.asarray(positions, dtype=float)
        if positions.ndim == 1:
            positions = positions.reshape(-1, self.n_dims)
        self.n_particles = positions.shape[0]
        self.positions = positions.copy()

        if velocities is None:

            sigma_v = np.sqrt(self.kb * self.temperature / self.mass)
            self.velocities = np.random.normal(0.0, sigma_v[:, None],
                                               size=(self.n_particles, self.n_dims))
        else:
            self.velocities = np.asarray(velocities, dtype=float).copy()

    def step(self, force_func: Callable[[np.ndarray], np.ndarray]):
        if self.positions is None:
            raise RuntimeError("必须先调用 initialize()")


        forces = force_func(self.positions)
        acc = forces / self.mass[:, None]
        self.velocities += acc * (self.dt * 0.5)


        self.positions += self.velocities * (self.dt * 0.5)



        noise = np.random.normal(0.0, 1.0, size=(self.n_particles, self.n_dims))
        mass_factor = 1.0 / np.sqrt(self.mass[:, None])
        self.velocities = (self.c1 * self.velocities +
                           self.c2 * mass_factor * noise)


        self.positions += self.velocities * (self.dt * 0.5)


        forces = force_func(self.positions)
        acc = forces / self.mass[:, None]
        self.velocities += acc * (self.dt * 0.5)

    def run(self, n_steps: int, force_func: Callable[[np.ndarray], np.ndarray],
            callback: Optional[Callable[[int, np.ndarray, np.ndarray], None]] = None):
        if n_steps < 0:
            raise ValueError("n_steps >= 0")
        for step in range(n_steps):
            self.step(force_func)
            if callback is not None:
                callback(step, self.positions, self.velocities)

    def kinetic_energy(self) -> float:
        if self.velocities is None:
            return 0.0
        return float(np.sum(0.5 * self.mass[:, None] * self.velocities ** 2))

    def temperature_instantaneous(self) -> float:
        e_kin = self.kinetic_energy()
        n_dof = self.n_particles * self.n_dims
        return (2.0 * e_kin) / (n_dof * self.kb)

    def compute_mean_square_displacement(self, traj: List[np.ndarray]) -> np.ndarray:
        if len(traj) == 0:
            return np.array([])
        r0 = traj[0]
        msd = np.zeros(len(traj))
        for i, ri in enumerate(traj):
            dr = ri - r0
            msd[i] = np.mean(np.sum(dr ** 2, axis=1))
        return msd

    def diffusion_coefficient_from_msd(self, traj: List[np.ndarray],
                                       time_interval_fs: float) -> float:
        msd = self.compute_mean_square_displacement(traj)
        n_frames = len(msd)
        if n_frames < 10:
            return 0.0

        start_idx = n_frames // 2
        t_vals = np.arange(start_idx, n_frames) * time_interval_fs * 1e-15
        msd_vals = msd[start_idx:]

        A = np.vstack([t_vals, np.ones_like(t_vals)]).T
        slope, _ = np.linalg.lstsq(A, msd_vals, rcond=None)[0]
        D = slope / (2.0 * self.n_dims)
        return float(D)


class StochasticReactionDynamics:

    def __init__(self, surface, pes, temperature_k: float = 500.0):
        self.surface = surface
        self.pes = pes
        self.temperature = temperature_k
        self.time = 0.0
        self.event_log = []

    def compute_event_rates(self, species_map: np.ndarray) -> np.ndarray:









        raise NotImplementedError("Hole_2: 请实现 compute_event_rates 方法")

    def gillespie_step(self, species_map: np.ndarray) -> Tuple[float, int, int]:
        rates = self.compute_event_rates(species_map)
        R_total = np.sum(rates)
        if R_total < 1e-300:
            return 1.0e10, -1, -1

        u1 = np.random.random()
        tau = -np.log(u1) / R_total


        cumsum = np.cumsum(rates) / R_total
        u2 = np.random.random()
        event_idx = int(np.searchsorted(cumsum, u2))
        site_idx = event_idx // 3
        event_type = event_idx % 3

        return tau, event_type, site_idx
