
import numpy as np
from physics_constants import (
    curl_electric_to_magnetic,
    curl_magnetic_to_electric,
    electromagnetic_energy_density,
    cfl_condition_3d,
)


class FDTD3DEngine:

    def __init__(self, grid, epsilon, mu, sigma, source=None, dt=None, cfl_factor=0.95):
        self.grid = grid
        self.epsilon = epsilon
        self.mu = mu
        self.sigma = sigma
        self.source = source


        c_max = np.max(1.0 / np.sqrt(epsilon * mu))
        self.c_max = c_max


        if dt is None:
            dt_cfl = cfl_condition_3d(grid.dx, grid.dy, grid.dz, c_max)
            self.dt = cfl_factor * dt_cfl
        else:
            dt_cfl = cfl_condition_3d(grid.dx, grid.dy, grid.dz, c_max)
            if dt > dt_cfl:
                raise ValueError(f"时间步长{dt}超过CFL极限{dt_cfl}")
            self.dt = dt

        self.cfl_factor = cfl_factor
        self.time = 0.0
        self.step_count = 0


        nx, ny, nz = grid.nx, grid.ny, grid.nz
        shape = (nx, ny, nz)
        self.Ex = np.zeros(shape)
        self.Ey = np.zeros(shape)
        self.Ez = np.zeros(shape)
        self.Hx = np.zeros(shape)
        self.Hy = np.zeros(shape)
        self.Hz = np.zeros(shape)


        self._compute_update_coefficients()


        self.energy_history = []
        self.time_history = []

    def _compute_update_coefficients(self):
        dt = self.dt
        eps = self.epsilon
        mu = self.mu
        sig = self.sigma


        self.ch = dt / mu




        denom = eps + 0.5 * sig * dt
        denom = np.where(np.abs(denom) < 1e-30, 1e-30, denom)
        self.ce1 = (eps - 0.5 * sig * dt) / denom
        self.ce2 = dt / denom

    def update_magnetic(self):



        raise NotImplementedError("Hole 2a: 请实现update_magnetic的磁场更新逻辑")

    def update_electric(self):



        raise NotImplementedError("Hole 2b: 请实现update_electric的电场更新逻辑")

    def apply_pec_boundary(self):
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz


        self.Ey[0, :, :] = 0.0
        self.Ez[0, :, :] = 0.0
        self.Ey[-1, :, :] = 0.0
        self.Ez[-1, :, :] = 0.0


        self.Ex[:, 0, :] = 0.0
        self.Ez[:, 0, :] = 0.0
        self.Ex[:, -1, :] = 0.0
        self.Ez[:, -1, :] = 0.0


        self.Ex[:, :, 0] = 0.0
        self.Ey[:, :, 0] = 0.0
        self.Ex[:, :, -1] = 0.0
        self.Ey[:, :, -1] = 0.0

    def apply_source(self):
        if self.source is not None:
            self.source(self.time, self)

    def compute_energy(self):
        E = (self.Ex, self.Ey, self.Ez)
        H = (self.Hx, self.Hy, self.Hz)
        w = electromagnetic_energy_density(E, H, self.epsilon, self.mu)
        return np.sum(w) * self.grid.cell_volume()

    def compute_power_loss(self):
        E_mag_sq = self.Ex**2 + self.Ey**2 + self.Ez**2
        p_loss = self.sigma * E_mag_sq
        return np.sum(p_loss) * self.grid.cell_volume()

    def step(self):
        self.update_magnetic()
        self.update_electric()
        self.apply_pec_boundary()
        self.time += self.dt
        self.step_count += 1
        self.apply_source()

    def run(self, n_steps, energy_sample_interval=10):
        for i in range(n_steps):
            self.step()
            if i % energy_sample_interval == 0:
                W = self.compute_energy()
                P = self.compute_power_loss()
                self.energy_history.append(W)
                self.time_history.append(self.time)

        return {
            'time_history': np.array(self.time_history),
            'energy_history': np.array(self.energy_history),
            'final_E': (self.Ex.copy(), self.Ey.copy(), self.Ez.copy()),
            'final_H': (self.Hx.copy(), self.Hy.copy(), self.Hz.copy()),
            'dt': self.dt,
            'n_steps': n_steps,
        }


class HarmonicSource:

    def __init__(self, amplitude, frequency, t0, tau, position, component='Ez'):
        self.amplitude = amplitude
        self.frequency = frequency
        self.t0 = t0
        self.tau = tau
        self.position = position
        self.component = component

    def __call__(self, t, engine):
        envelope = np.exp(-((t - self.t0) ** 2) / (2.0 * self.tau ** 2))
        value = self.amplitude * envelope * np.sin(2.0 * np.pi * self.frequency * t)

        ix, iy, iz = self.position
        if self.component == 'Ex':
            engine.Ex[ix, iy, iz] += value
        elif self.component == 'Ey':
            engine.Ey[ix, iy, iz] += value
        elif self.component == 'Ez':
            engine.Ez[ix, iy, iz] += value
        elif self.component == 'Hx':
            engine.Hx[ix, iy, iz] += value
        elif self.component == 'Hy':
            engine.Hy[ix, iy, iz] += value
        elif self.component == 'Hz':
            engine.Hz[ix, iy, iz] += value


def stability_analysis_2d_scalar(kx, ky, dx, dy, dt, c):
    rhs = (np.sin(kx * dx / 2.0) / dx) ** 2 + (np.sin(ky * dy / 2.0) / dy) ** 2
    arg = (c * dt) ** 2 * rhs
    arg = min(arg, 1.0)
    omega_numerical = 2.0 / dt * np.arcsin(np.sqrt(arg))
    omega_exact = c * np.sqrt(kx ** 2 + ky ** 2)
    return omega_numerical, omega_exact
