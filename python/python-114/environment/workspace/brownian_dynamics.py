
import numpy as np
from typing import Callable, Tuple, Optional


class OverdampedLangevinIntegrator:

    def __init__(
        self,
        diffusion_coeff: float = 5.0,
        friction_coeff: float = 1.2e-3,
        temperature: float = 310.0,
        dt: float = 1e-3,
    ):
        self.D = float(diffusion_coeff)
        self.gamma = float(friction_coeff)
        self.T = float(temperature)
        self.dt = float(dt)
        self.k_B = 1.380649e-5


        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.D <= 0.0:
            raise ValueError("diffusion coefficient must be positive")


        D_fdt = self.k_B * self.T / self.gamma
        if not np.isclose(self.D, D_fdt, rtol=0.5):

            pass

    def step(
        self,
        x: np.ndarray,
        force: np.ndarray,
    ) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        force = np.asarray(force, dtype=np.float64)

        if x.shape != force.shape:
            raise ValueError("x and force must have the same shape")

        n_particles = x.shape[0]


        drift = (self.dt / self.gamma) * force


        noise_amp = np.sqrt(2.0 * self.D * self.dt)
        noise = noise_amp * np.random.randn(*x.shape)

        x_new = x + drift + noise


        nuclear_radius = 5000.0
        for i in range(n_particles):
            r = np.linalg.norm(x_new[i, :])
            if r > nuclear_radius:
                x_new[i, :] *= nuclear_radius / (r + 1e-12)

        return x_new

    def integrate(
        self,
        x0: np.ndarray,
        force_func: Callable[[np.ndarray], np.ndarray],
        n_steps: int,
    ) -> np.ndarray:
        x0 = np.asarray(x0, dtype=np.float64)
        traj = np.zeros((n_steps + 1,) + x0.shape, dtype=np.float64)
        traj[0, :, :] = x0

        x = x0.copy()
        for step in range(n_steps):
            f = force_func(x)
            x = self.step(x, f)
            traj[step + 1, :, :] = x

        return traj


def dna_repair_protein_force(
    x: np.ndarray,
    dsb_site: np.ndarray,
    binding_energy: float = 15.0,
    binding_range: float = 5.0,
    repulsion_strength: float = 2.0,
    protein_radius: float = 3.0,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    dsb_site = np.asarray(dsb_site, dtype=np.float64)
    n = x.shape[0]

    if n == 0:
        return np.zeros_like(x)


    dx = x - dsb_site[np.newaxis, :]
    r = np.linalg.norm(dx, axis=1) + 1e-12



    f_bind_mag = binding_energy * (r / (binding_range ** 2)) * np.exp(-r ** 2 / (2.0 * binding_range ** 2))
    f_bind = -f_bind_mag[:, np.newaxis] * (dx / r[:, np.newaxis])


    f_rep = np.zeros_like(x)
    if n > 1:
        for i in range(n):
            dx_ij = x[i, :] - x[i + 1 :, :]
            r_ij = np.linalg.norm(dx_ij, axis=1) + 1e-12
            mask = r_ij < 2.5 * protein_radius
            if np.any(mask):
                dr = r_ij[mask] - 2.0 * protein_radius
                f_mag = repulsion_strength * np.exp(-dr ** 2 / (2.0 * (protein_radius ** 2)))
                f_vec = f_mag[:, np.newaxis] * (dx_ij[mask, :] / r_ij[mask, np.newaxis])
                f_rep[i, :] += np.sum(f_vec, axis=0)

                idx = np.where(mask)[0] + i + 1
                for j_local, j_global in enumerate(idx):
                    f_rep[j_global, :] -= f_vec[j_local, :]


    k_B = 1.380649e-5
    T = 310.0
    total_force = (f_bind + f_rep) * k_B * T

    return total_force


def henon_crowding_map(
    x: np.ndarray,
    y: np.ndarray,
    c: float = 0.98,
    n_iter: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    if not (-1.0 <= c <= 1.0):
        raise ValueError("c must be in [-1, 1]")
    s = np.sqrt(max(0.0, 1.0 - c ** 2))

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)


    r = np.sqrt(x ** 2 + y ** 2)
    valid = r < 1.0

    xv = x[valid].copy()
    yv = y[valid].copy()

    for _ in range(n_iter):
        xnew = xv * c - (yv - xv ** 2) * s
        ynew = xv * s + (yv - xv ** 2) * c
        xv, yv = xnew, ynew

    x_out = x.copy()
    y_out = y.copy()
    x_out[valid] = xv
    y_out[valid] = yv

    return x_out, y_out


def normal_distribution_ode_solution(t: np.ndarray, sigma0: float = 1.0) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    y = np.exp(-(t / sigma0) ** 2 / 2.0) / np.sqrt(2.0 * np.pi) / sigma0


    y = np.where(np.isfinite(y), y, 0.0)
    y = np.where(y >= 0, y, 0.0)
    return y


def simulate_ku80_search_time(
    n_proteins: int = 50,
    n_steps: int = 5000,
    dsb_position: Optional[np.ndarray] = None,
) -> dict:
    if dsb_position is None:
        dsb_position = np.array([0.0, 0.0, 0.0])


    np.random.seed(42)
    x0 = np.random.randn(n_proteins, 3) * 1000.0
    nuclear_r = 5000.0
    for i in range(n_proteins):
        r = np.linalg.norm(x0[i, :])
        if r > nuclear_r:
            x0[i, :] *= nuclear_r / r

    integrator = OverdampedLangevinIntegrator(dt=0.01)

    force_func = lambda x: dna_repair_protein_force(x, dsb_position)

    traj = integrator.integrate(x0, force_func, n_steps)


    binding_radius = 5.0
    distances = np.linalg.norm(traj - dsb_position[np.newaxis, np.newaxis, :], axis=2)
    bound = distances < binding_radius


    first_pass = np.full(n_proteins, n_steps + 1, dtype=np.int64)
    for p in range(n_proteins):
        hit = np.where(bound[:, p])[0]
        if len(hit) > 0:
            first_pass[p] = hit[0]

    bound_proteins = first_pass <= n_steps
    if np.any(bound_proteins):
        mftp = float(np.mean(first_pass[bound_proteins])) * integrator.dt
    else:
        mftp = float(n_steps) * integrator.dt


    msd = np.mean(np.sum((traj - traj[0, :, :][np.newaxis, :, :]) ** 2, axis=2), axis=1)

    return {
        "mfpt_us": mftp,
        "binding_fraction": float(np.mean(first_pass <= n_steps)),
        "msd_final_nm2": float(msd[-1]),
        "trajectory_shape": traj.shape,
    }
