
import numpy as np


class LagrangianParticleTransport:

    def __init__(self, nx, nz, Lx, Lz, nparticles=5000, dt=3600.0):
        if nx < 4 or nz < 4:
            raise ValueError("nx, nz >= 4")
        if nparticles < 100:
            raise ValueError("nparticles >= 100")
        if dt <= 0:
            raise ValueError("dt > 0")

        self.nx = nx
        self.nz = nz
        self.Lx = Lx
        self.Lz = Lz
        self.dx = Lx / (nx - 1)
        self.dz = Lz / (nz - 1)
        self.nparticles = nparticles
        self.dt = dt


        self.pos = np.zeros((nparticles, 2))

        self.pos[:, 0] = np.random.rand(nparticles) * Lx
        self.pos[:, 1] = np.random.rand(nparticles) * Lz


        self.nutrient_conc = np.random.uniform(0.5, 5.0, nparticles)
        self.chlorophyll = np.random.uniform(0.01, 0.5, nparticles)
        self.temperature = np.zeros(nparticles)
        self.salinity = np.zeros(nparticles)


        self.spwt = (Lx * Lz) / nparticles


        self.active = np.ones(nparticles, dtype=bool)

    def bilinear_weights(self, pos_x, pos_z):
        dx = self.dx
        dz = self.dz
        nx = self.nx
        nz = self.nz


        if not (np.isfinite(pos_x) and np.isfinite(pos_z)):
            return 0, 0, 0.0, 0.0

        pos_x = max(0.0, min(self.Lx - 1e-12, pos_x))
        pos_z = max(0.0, min(self.Lz - 1e-12, pos_z))

        fi = 1.0 + pos_x / dx
        i = int(np.floor(fi))
        i = max(0, min(i, nx - 2))
        hx = fi - i
        hx = max(0.0, min(1.0, hx))

        fj = 1.0 + pos_z / dz
        j = int(np.floor(fj))
        j = max(0, min(j, nz - 2))
        hy = fj - j
        hy = max(0.0, min(1.0, hy))

        return i, j, hx, hy

    def interpolate_field_to_particle(self, field):
        vals = np.zeros(self.nparticles)
        for p in range(self.nparticles):
            if not self.active[p]:
                continue
            i, j, hx, hy = self.bilinear_weights(self.pos[p, 0], self.pos[p, 1])
            vals[p] = (
                (1.0 - hx) * (1.0 - hy) * field[i, j] +
                hx * (1.0 - hy) * field[i + 1, j] +
                (1.0 - hx) * hy * field[i, j + 1] +
                hx * hy * field[i + 1, j + 1]
            )
        return vals

    def deposit_particles_to_grid(self, particle_scalar):
        grid = np.zeros((self.nx, self.nz))
        count = np.zeros((self.nx, self.nz))

        for p in range(self.nparticles):
            if not self.active[p]:
                continue
            i, j, hx, hy = self.bilinear_weights(self.pos[p, 0], self.pos[p, 1])
            w00 = (1.0 - hx) * (1.0 - hy)
            w10 = hx * (1.0 - hy)
            w01 = (1.0 - hx) * hy
            w11 = hx * hy

            grid[i, j] += w00 * particle_scalar[p]
            grid[i + 1, j] += w10 * particle_scalar[p]
            grid[i, j + 1] += w01 * particle_scalar[p]
            grid[i + 1, j + 1] += w11 * particle_scalar[p]
            count[i, j] += w00
            count[i + 1, j] += w10
            count[i, j + 1] += w01
            count[i + 1, j + 1] += w11


        with np.errstate(divide='ignore', invalid='ignore'):
            grid = np.where(count > 0, grid / count, 0.0)
        return grid

    def step(self, u_field, w_field, omega_bio=0.0):

        u_p = self.interpolate_field_to_particle(u_field)
        w_p = self.interpolate_field_to_particle(w_field)


        self.pos[:, 0] += self.dt * u_p
        self.pos[:, 1] += self.dt * w_p


        for p in range(self.nparticles):
            if not self.active[p]:
                continue


            if self.pos[p, 1] < 0.0:
                self.pos[p, 1] = -self.pos[p, 1]


            if self.pos[p, 1] > self.Lz:
                self.pos[p, 1] = 2.0 * self.Lz - self.pos[p, 1]


            if self.pos[p, 0] < 0.0:
                self.pos[p, 0] = self.Lx - 1e-6
                self.nutrient_conc[p] = np.random.uniform(3.0, 6.0)
            if self.pos[p, 0] > self.Lx:
                self.pos[p, 0] = 1e-6
                self.nutrient_conc[p] = np.random.uniform(1.0, 3.0)


        if omega_bio > 0:
            self.nutrient_conc *= np.exp(-omega_bio * self.dt)
            self.chlorophyll *= np.exp(-omega_bio * self.dt)

    def get_particle_density_field(self):
        ones = np.ones(self.nparticles)
        return self.deposit_particles_to_grid(ones)

    def get_mean_nutrient_field(self):
        return self.deposit_particles_to_grid(self.nutrient_conc)

    def resample_particles(self, N_grid, P_grid, T_grid, S_grid):

        N_env = self.interpolate_field_to_particle(N_grid)
        P_env = self.interpolate_field_to_particle(P_grid)
        self.temperature = self.interpolate_field_to_particle(T_grid)
        self.salinity = self.interpolate_field_to_particle(S_grid)


        relax = 0.01
        self.nutrient_conc += relax * (N_env - self.nutrient_conc)
        self.chlorophyll += relax * (P_env - self.chlorophyll)


        self.nutrient_conc = np.maximum(self.nutrient_conc, 0.0)
        self.chlorophyll = np.maximum(self.chlorophyll, 0.0)
