
import numpy as np
from typing import Tuple, Callable


class NanoparticleLangevinDynamics:

    def __init__(self,
                 R_np: float = 2.5,
                 eta: float = 0.89e-9,
                 T: float = 300.0,
                 k_B: float = 8.314e-3,
                 z0: float = 8.0,
                 z_cutoff: float = 1.0,
                 F_max_bind: float = 50.0,
                 kappa_bind: float = 2.0,
                 epsilon_LJ: float = 4.0,
                 sigma_LJ: float = 3.0,
                 k_spring_bend: float = 10.0):
        self.R_np = float(R_np)
        self.eta = float(eta)
        self.T = float(T)
        self.k_B = float(k_B)
        self.z = float(z0)
        self.z_cutoff = float(z_cutoff)
        self.F_max_bind = float(F_max_bind)
        self.kappa_bind = float(kappa_bind)
        self.epsilon_LJ = float(epsilon_LJ)
        self.sigma_LJ = float(sigma_LJ)
        self.k_spring_bend = float(k_spring_bend)


        self.gamma = max(6.0 * np.pi * self.eta * self.R_np, 10.0)

        self.dt = 1e-5

    def force_electrostatic(self, z: float, debye_length: float = 1.0,
                            zeta_np: float = -0.05,
                            zeta_mem: float = -0.03) -> float:
        eps_rel = 80.0
        eps0 = 8.854e-12


        prefactor = 10.0
        F = prefactor * (zeta_np * zeta_mem / debye_length) * np.exp(-z / debye_length)
        return float(F)

    def force_vdw(self, z: float) -> float:





        raise NotImplementedError("HOLE 2: 请补全 force_vdw 的 LJ 力计算")

    def force_bending(self, z: float) -> float:
        z_eq = 0.5
        return -self.k_spring_bend * (z - z_eq)

    def force_binding(self, z: float) -> float:
        dz = max(self.z_cutoff - z, 0.0)

        F = -self.F_max_bind * (1.0 - np.exp(-self.kappa_bind * dz))
        return float(F)

    def total_force(self, z: float, debye_length: float = 1.0) -> float:
        return (self.force_electrostatic(z, debye_length)
                + self.force_vdw(z)
                + self.force_bending(z)
                + self.force_binding(z))

    def step_euler_maruyama(self, debye_length: float = 1.0) -> float:
        F = self.total_force(self.z, debye_length)
        F_capped = np.clip(F, -1000.0, 1000.0)
        drift = self.dt / self.gamma * F_capped
        diffusion = np.sqrt(2.0 * self.k_B * self.T * self.dt / self.gamma)
        noise = diffusion * np.random.randn()
        self.z = self.z + drift + noise

        if self.z < 0.1:
            self.z = 0.1 + (0.1 - self.z)
        return float(self.z)

    def simulate(self, n_steps: int = 50000,
                 debye_length: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        t = np.arange(n_steps) * self.dt
        z_traj = np.zeros(n_steps, dtype=np.float64)
        F_traj = np.zeros(n_steps, dtype=np.float64)
        for i in range(n_steps):
            z_traj[i] = self.z
            F_traj[i] = self.total_force(self.z, debye_length)
            self.step_euler_maruyama(debye_length)
        return t, z_traj, F_traj
