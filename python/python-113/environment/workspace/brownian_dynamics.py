
import numpy as np


class IonParticle:
    def __init__(self, pos, charge, radius, mass_amu=39.0983):
        self.pos = np.array(pos, dtype=float)
        self.charge = charge
        self.radius = radius
        self.mass = mass_amu * 1.66053906660e-27
        self.trajectory = [self.pos.copy()]


class BrownianDynamicsEngine:
    def __init__(self, temperature=300.0, dt=1e-15, friction=1e-11):
        self.T = temperature
        self.dt = dt
        self.gamma = friction
        self.kB = 1.380649e-23
        self.kT = self.kB * temperature

    def _random_displacement(self, D):
        sigma = np.sqrt(2.0 * D * self.dt)
        return sigma * np.random.randn(3)

    def _force_field(self, particle, phi_field, grid_origin, grid_spacing):

        pos = particle.pos
        ix = int((pos[0] - grid_origin[0]) / grid_spacing[0])
        iy = int((pos[1] - grid_origin[1]) / grid_spacing[1])
        iz = int((pos[2] - grid_origin[2]) / grid_spacing[2])

        Nx, Ny, Nz = phi_field.shape
        ix = np.clip(ix, 1, Nx - 2)
        iy = np.clip(iy, 1, Ny - 2)
        iz = np.clip(iz, 1, Nz - 2)

        dx, dy, dz = grid_spacing

        Ex = -(phi_field[ix + 1, iy, iz] - phi_field[ix - 1, iy, iz]) / (2.0 * dx)
        Ey = -(phi_field[ix, iy + 1, iz] - phi_field[ix, iy - 1, iz]) / (2.0 * dy)
        Ez = -(phi_field[ix, iy, iz + 1] - phi_field[ix, iy, iz - 1]) / (2.0 * dz)
        E_field = np.array([Ex, Ey, Ez])


        e_charge = 1.602176634e-19
        F_electric = -particle.charge * e_charge * E_field * 1e9


        r_xy = np.sqrt(pos[0] ** 2 + pos[1] ** 2)
        z = pos[2]

        if 1.5 <= z <= 2.7:
            r_channel = 0.15
        elif 0.5 <= z < 1.5:
            r_channel = 0.5
        elif z < 0.5:
            r_channel = 0.2 + 0.4 * z
        else:
            r_channel = 0.6

        k_wall = 1e-8
        if r_xy > r_channel:
            dr = r_xy - r_channel
            n_r = np.array([pos[0], pos[1], 0.0]) / (r_xy + 1e-12)
            F_wall = -k_wall * dr * n_r
        else:
            F_wall = np.zeros(3)

        return F_electric + F_wall

    def step(self, particle, phi_field, grid_origin, grid_spacing, D_coeff):
        F = self._force_field(particle, phi_field, grid_origin, grid_spacing)

        drift = (D_coeff / self.kT) * F * self.dt
        diffusion = self._random_displacement(D_coeff)
        particle.pos += drift + diffusion
        particle.trajectory.append(particle.pos.copy())
        return particle

    def run(self, particles, phi_field, grid_origin, grid_spacing,
            D_k=1.96e-9, D_na=1.33e-9, n_steps=1000):
        for step in range(n_steps):
            for p in particles:
                D = D_k if p.mass > 3e-26 else D_na
                self.step(p, phi_field, grid_origin, grid_spacing, D)
        return particles


def compute_mean_square_displacement(trajectories, dt, max_lag=None):
    if max_lag is None:
        max_lag = len(trajectories[0]) // 4

    msd = np.zeros(max_lag)
    counts = np.zeros(max_lag)

    for traj in trajectories:
        traj = np.array(traj)
        n = len(traj)
        for lag in range(1, max_lag):
            disp = traj[lag:] - traj[:-lag]
            sq = np.sum(disp ** 2, axis=1)
            msd[lag] += np.sum(sq)
            counts[lag] += len(sq)

    msd = msd / (counts + 1e-30)
    tau = np.arange(max_lag) * dt

    valid = (tau > 0) & (msd > 0)
    if np.sum(valid) > 5:
        slope = np.polyfit(tau[valid], msd[valid], 1)[0]
        D_estimated = slope / 6.0
    else:
        D_estimated = 0.0

    return tau, msd, D_estimated
