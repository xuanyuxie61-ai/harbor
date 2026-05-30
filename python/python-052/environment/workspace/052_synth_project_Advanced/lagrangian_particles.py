
import numpy as np
from typing import Tuple, Optional, Callable






class CMRG:

    M1 = 2147483563
    M2 = 2147483399
    A1 = 40014
    A2 = 40692

    def __init__(self, seed1: int = 12345, seed2: int = 67890):
        self.s1 = int(seed1) % self.M1
        self.s2 = int(seed2) % self.M2
        if self.s1 == 0:
            self.s1 = 1
        if self.s2 == 0:
            self.s2 = 1

    def _advance(self):
        self.s1 = (self.A1 * self.s1) % self.M1
        self.s2 = (self.A2 * self.s2) % self.M2

    def rand(self) -> float:
        self._advance()
        z = self.s1 - self.s2
        if z < 0:
            z += self.M1
        return z / self.M1

    def randn(self) -> float:
        u1 = self.rand()
        u2 = self.rand()
        if u1 < 1e-15:
            u1 = 1e-15
        return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)

    def rand_array(self, shape: Tuple[int, ...]) -> np.ndarray:
        return np.array([self.rand() for _ in range(int(np.prod(shape)))]).reshape(shape)

    def randn_array(self, shape: Tuple[int, ...]) -> np.ndarray:
        return np.array([self.randn() for _ in range(int(np.prod(shape)))]).reshape(shape)






def fibonacci_spiral_2d(n: int, Lx: float = 1.0, Ly: float = 1.0,
                        center: Tuple[float, float] = None) -> Tuple[np.ndarray, np.ndarray]:
    if n < 1:
        raise ValueError("n must be positive")
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    golden_angle = 2.0 * np.pi / (phi ** 2)

    i = np.arange(n, dtype=float)

    r = np.sqrt(i / n)
    theta = i * golden_angle


    x_circ = r * np.cos(theta)
    y_circ = r * np.sin(theta)


    if center is None:
        cx, cy = Lx / 2.0, Ly / 2.0
    else:
        cx, cy = center


    x = cx + 0.5 * Lx * x_circ
    y = cy + 0.5 * Ly * y_circ


    x = x % Lx
    y = y % Ly

    return x, y






def periodic_distance(a: np.ndarray, b: np.ndarray, L: float) -> np.ndarray:
    d = np.abs(a - b)
    return np.minimum(d, L - d)


def cvtp_optimize_2d(x: np.ndarray, y: np.ndarray, Lx: float, Ly: float,
                     n_samples: int = 5000, n_iter: int = 10,
                     rng: Optional[CMRG] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_particles = len(x)
    if rng is None:
        rng = CMRG(seed1=12345, seed2=67890)

    gx = x.copy()
    gy = y.copy()
    energy_history = []

    for _ in range(n_iter):

        sx = rng.rand_array((n_samples,)) * Lx
        sy = rng.rand_array((n_samples,)) * Ly


        belongs_to = np.zeros(n_samples, dtype=int)
        for s in range(n_samples):
            dx = periodic_distance(sx[s], gx, Lx)
            dy = periodic_distance(sy[s], gy, Ly)
            dist2 = dx ** 2 + dy ** 2
            belongs_to[s] = int(np.argmin(dist2))


        new_gx = np.zeros(n_particles)
        new_gy = np.zeros(n_particles)
        counts = np.zeros(n_particles)

        for i in range(n_particles):
            mask = belongs_to == i
            if np.sum(mask) > 0:

                sx_m = sx[mask]
                sy_m = sy[mask]

                angles_x = 2.0 * np.pi * sx_m / Lx
                cx = np.mean(np.cos(angles_x))
                sx_mean = (np.arctan2(np.mean(np.sin(angles_x)), cx) / (2.0 * np.pi)) * Lx
                if sx_mean < 0:
                    sx_mean += Lx

                angles_y = 2.0 * np.pi * sy_m / Ly
                cy = np.mean(np.cos(angles_y))
                sy_mean = (np.arctan2(np.mean(np.sin(angles_y)), cy) / (2.0 * np.pi)) * Ly
                if sy_mean < 0:
                    sy_mean += Ly
                new_gx[i] = sx_mean
                new_gy[i] = sy_mean
                counts[i] = np.sum(mask)
            else:
                new_gx[i] = gx[i]
                new_gy[i] = gy[i]
                counts[i] = 1

        gx = new_gx
        gy = new_gy


        energy = 0.0
        for s in range(n_samples):
            i = belongs_to[s]
            dx = periodic_distance(sx[s], gx[i], Lx)
            dy = periodic_distance(sy[s], gy[i], Ly)
            energy += dx ** 2 + dy ** 2
        energy_history.append(energy / n_samples)

    return gx, gy, np.array(energy_history)






class LagrangianParticleTracker:

    def __init__(self, x0: np.ndarray, y0: np.ndarray,
                 Lx: float, Ly: float, dt: float = 0.01,
                 diffusivity: float = 0.0, rng: Optional[CMRG] = None):
        self.x = np.asarray(x0, dtype=float).copy()
        self.y = np.asarray(y0, dtype=float).copy()
        self.n_particles = len(self.x)
        self.Lx = float(Lx)
        self.Ly = float(Ly)
        self.dt = float(dt)
        self.diffusivity = float(diffusivity)
        self.rng = rng if rng is not None else CMRG()


        self.trajectory_x = [self.x.copy()]
        self.trajectory_y = [self.y.copy()]
        self.t_history = [0.0]

    def _periodic_wrap(self):
        self.x = self.x % self.Lx
        self.y = self.y % self.Ly

    def _interpolate_velocity(self, x_query: np.ndarray, y_query: np.ndarray,
                              u_field: np.ndarray, v_field: np.ndarray,
                              x_grid: np.ndarray, y_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Ny, Nx = u_field.shape
        dx = x_grid[1] - x_grid[0]
        dy = y_grid[1] - y_grid[0]

        ix = np.floor((x_query - x_grid[0]) / dx).astype(int)
        iy = np.floor((y_query - y_grid[0]) / dy).astype(int)
        ix = np.clip(ix, 0, Nx - 2)
        iy = np.clip(iy, 0, Ny - 2)

        fx = (x_query - x_grid[ix]) / dx
        fy = (y_query - y_grid[iy]) / dy
        fx = np.clip(fx, 0.0, 1.0)
        fy = np.clip(fy, 0.0, 1.0)

        u_q = (1 - fx) * (1 - fy) * u_field[iy, ix] + \
              fx * (1 - fy) * u_field[iy, ix + 1] + \
              (1 - fx) * fy * u_field[iy + 1, ix] + \
              fx * fy * u_field[iy + 1, ix + 1]

        v_q = (1 - fx) * (1 - fy) * v_field[iy, ix] + \
              fx * (1 - fy) * v_field[iy, ix + 1] + \
              (1 - fx) * fy * v_field[iy + 1, ix] + \
              fx * fy * v_field[iy + 1, ix + 1]

        return u_q, v_q

    def step_rk4(self, u_field: np.ndarray, v_field: np.ndarray,
                 x_grid: np.ndarray, y_grid: np.ndarray):
        def vel(xp, yp):
            return self._interpolate_velocity(xp, yp, u_field, v_field, x_grid, y_grid)


        k1x, k1y = vel(self.x, self.y)
        if self.diffusivity > 0:
            k1x += np.sqrt(2.0 * self.diffusivity / self.dt) * self.rng.randn_array((self.n_particles,))
            k1y += np.sqrt(2.0 * self.diffusivity / self.dt) * self.rng.randn_array((self.n_particles,))


        x2 = self.x + 0.5 * self.dt * k1x
        y2 = self.y + 0.5 * self.dt * k1y
        k2x, k2y = vel(x2, y2)


        x3 = self.x + 0.5 * self.dt * k2x
        y3 = self.y + 0.5 * self.dt * k2y
        k3x, k3y = vel(x3, y3)


        x4 = self.x + self.dt * k3x
        y4 = self.y + self.dt * k3y
        k4x, k4y = vel(x4, y4)

        self.x = self.x + self.dt * (k1x + 2.0 * k2x + 2.0 * k3x + k4x) / 6.0
        self.y = self.y + self.dt * (k1y + 2.0 * k2y + 2.0 * k3y + k4y) / 6.0
        self._periodic_wrap()

        self.trajectory_x.append(self.x.copy())
        self.trajectory_y.append(self.y.copy())
        self.t_history.append(self.t_history[-1] + self.dt)

    def compute_mean_square_displacement(self) -> Tuple[np.ndarray, np.ndarray]:
        tx = np.array(self.trajectory_x)
        ty = np.array(self.trajectory_y)
        dx = periodic_distance(tx, tx[0, :], self.Lx)
        dy = periodic_distance(ty, ty[0, :], self.Ly)
        msd = np.mean(dx ** 2 + dy ** 2, axis=1)
        t = np.array(self.t_history)
        return t, msd

    def compute_pair_separation(self, n_pairs: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        tx = np.array(self.trajectory_x)
        ty = np.array(self.trajectory_y)
        nt = tx.shape[0]

        rng_local = np.random.default_rng(42)
        pairs = rng_local.integers(0, self.n_particles, size=(n_pairs, 2))

        r2_mean = np.zeros(nt)
        for t_idx in range(nt):
            dx = periodic_distance(tx[t_idx, pairs[:, 0]], tx[t_idx, pairs[:, 1]], self.Lx)
            dy = periodic_distance(ty[t_idx, pairs[:, 0]], ty[t_idx, pairs[:, 1]], self.Ly)
            r2_mean[t_idx] = np.mean(dx ** 2 + dy ** 2)

        return np.array(self.t_history), r2_mean

    def compute_diffusivity(self) -> float:
        t, msd = self.compute_mean_square_displacement()
        if len(t) < 3:
            return 0.0

        half = len(t) // 2
        if half < 2:
            half = 1
        D_est = np.mean(msd[half:] / (4.0 * t[half:]))
        return float(D_est)






def compute_ftle_grid(u_field: np.ndarray, v_field: np.ndarray,
                      x_grid: np.ndarray, y_grid: np.ndarray,
                      dt: float, n_steps: int,
                      dx_grid: float, dy_grid: float) -> np.ndarray:
    Ny, Nx = u_field.shape
    ftle = np.zeros((Ny, Nx))


    eps = min(dx_grid, dy_grid) * 0.1

    for j in range(Ny):
        for i in range(Nx):
            x0 = x_grid[i]
            y0 = y_grid[j]



            offsets = np.array([[eps, 0], [0, eps], [-eps, 0], [0, -eps]])
            x_traj = np.zeros((n_steps + 1, 4))
            y_traj = np.zeros((n_steps + 1, 4))
            x_traj[0, :] = x0 + offsets[:, 0]
            y_traj[0, :] = y0 + offsets[:, 1]


            for step in range(n_steps):
                idx = step % u_field.shape[0]

                ux = u_field[j, i]
                vy = v_field[j, i]
                x_traj[step + 1, :] = x_traj[step, :] + ux * dt
                y_traj[step + 1, :] = y_traj[step, :] + vy * dt


            dx_final = x_traj[-1, :] - x_traj[-1, :].mean()
            dy_final = y_traj[-1, :] - y_traj[-1, :].mean()


            J11 = (x_traj[-1, 0] - x_traj[-1, 2]) / (2 * eps)
            J12 = (x_traj[-1, 1] - x_traj[-1, 3]) / (2 * eps)
            J21 = (y_traj[-1, 0] - y_traj[-1, 2]) / (2 * eps)
            J22 = (y_traj[-1, 1] - y_traj[-1, 3]) / (2 * eps)

            C = np.array([[J11 ** 2 + J21 ** 2, J11 * J12 + J21 * J22],
                          [J11 * J12 + J21 * J22, J12 ** 2 + J22 ** 2]])
            eigvals = np.linalg.eigvalsh(C)
            lambda_max = np.max(eigvals)
            T = n_steps * dt
            if lambda_max > 1e-15 and abs(T) > 1e-15:
                ftle[j, i] = 0.5 * np.log(lambda_max) / abs(T)
            else:
                ftle[j, i] = 0.0

    return ftle


if __name__ == "__main__":

    x, y = fibonacci_spiral_2d(500, Lx=2*np.pi, Ly=2*np.pi)
    print("Fibonacci spiral mean x:", np.mean(x), "std:", np.std(x))


    rng = CMRG()
    vals = [rng.rand() for _ in range(5)]
    print("CMRG samples:", vals)


    x_opt, y_opt, E = cvtp_optimize_2d(x, y, 2*np.pi, 2*np.pi, n_samples=2000, n_iter=3)
    print("CVT energy history:", E)
