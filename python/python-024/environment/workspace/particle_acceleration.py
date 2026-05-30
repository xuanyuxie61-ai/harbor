
import numpy as np
from typing import Tuple, List, Optional


Q_E = 1.602176634e-19
M_E = 9.10938356e-31
M_P = 1.6726219e-27
C_LIGHT = 2.99792458e8


class MonteCarloParticleAccelerator:

    def __init__(self,
                 n_particles: int = 10000,
                 particle_charge: float = Q_E,
                 particle_mass: float = M_P,
                 B_strength: float = 1.0e-2,
                 E_reconnection: float = 1.0e3,
                 lambda_acc: float = 1.0e4):
        if n_particles <= 0:
            raise ValueError("n_particles 必须为正")
        if particle_mass <= 0:
            raise ValueError("particle_mass 必须为正")
        self.n = n_particles
        self.q = particle_charge
        self.m = particle_mass
        self.B = B_strength
        self.E = E_reconnection
        self.L = lambda_acc

    def sample_initial_velocities(self,
                                   T_thermal: float = 1.0e6,
                                   seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)


        k_B = 1.380649e-23
        v_th = np.sqrt(k_B * T_thermal / self.m)


        v_parallel = np.random.normal(0.0, v_th, size=self.n)

        v_perp_mag = np.random.rayleigh(v_th / np.sqrt(2.0), size=self.n)

        theta = 2.0 * np.pi * np.random.rand(self.n)
        v_perp_x = v_perp_mag * np.cos(theta)
        v_perp_y = v_perp_mag * np.sin(theta)


        v = np.column_stack([v_parallel, v_perp_x, v_perp_y])
        return v

    def accelerate_stochastic(self,
                               v0: np.ndarray,
                               dt: float = 1.0e-6,
                               n_steps: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        if dt <= 0:
            raise ValueError("dt 必须为正")
        v = np.copy(v0)
        n = len(v)
        energies = np.zeros((n_steps + 1, n))


        v_sq = np.sum(v ** 2, axis=1)
        gamma = 1.0 / np.sqrt(1.0 - np.minimum(v_sq / C_LIGHT ** 2, 0.9999))
        energies[0] = (gamma - 1.0) * self.m * C_LIGHT ** 2 / Q_E

        for step in range(n_steps):

            v[:, 0] += dt * self.q * self.E / self.m


            scatter_angle = np.random.normal(0.0, 0.05, size=n)
            v_perp_mag = np.sqrt(v[:, 1] ** 2 + v[:, 2] ** 2)
            theta_old = np.arctan2(v[:, 2], v[:, 1])
            theta_new = theta_old + scatter_angle
            v[:, 1] = v_perp_mag * np.cos(theta_new)
            v[:, 2] = v_perp_mag * np.sin(theta_new)


            v_mag = np.sqrt(np.sum(v ** 2, axis=1))
            mask = v_mag > 0.99 * C_LIGHT
            if np.any(mask):
                v[mask] = 0.99 * C_LIGHT * v[mask] / v_mag[mask, None]


            v_sq = np.sum(v ** 2, axis=1)
            gamma = 1.0 / np.sqrt(1.0 - np.minimum(v_sq / C_LIGHT ** 2, 0.9999))
            energies[step + 1] = (gamma - 1.0) * self.m * C_LIGHT ** 2 / Q_E

        return v, energies

    def energy_spectrum(self, energies_ev: np.ndarray, n_bins: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        e_flat = energies_ev.ravel()
        e_min = max(np.min(e_flat), 1.0)
        e_max = np.max(e_flat)
        if e_max <= e_min:
            e_max = e_min * 10.0
        bins = np.logspace(np.log10(e_min), np.log10(e_max), n_bins)
        hist, edges = np.histogram(e_flat, bins=bins)
        bin_centers = np.sqrt(edges[:-1] * edges[1:])
        dE = edges[1:] - edges[:-1]
        spectrum = hist / (dE * self.n)
        return bin_centers, spectrum


class NonlinearOrbitTracker:

    def __init__(self,
                 q: float = Q_E,
                 m: float = M_P,
                 B_prime: float = 1.0e-7,
                 E_field: float = 1.0e3,
                 length: float = 1.0):
        self.q = q
        self.m = m
        self.Bp = B_prime
        self.E = E_field
        self.L = length

        self.omega0_sq = q * B_prime / m
        if self.omega0_sq < 0:
            raise ValueError("omega0_sq 为负，检查参数符号")
        self.omega0 = np.sqrt(abs(self.omega0_sq))

    def deriv(self, t: float, y: np.ndarray) -> np.ndarray:
        x, yp, vx, vy = y

        x_clip = np.clip(x, -self.L, self.L)
        ax = self.q / self.m * self.E + self.q / self.m * vy * self.Bp * x_clip
        ay = -self.q / self.m * vx * self.Bp * x_clip
        return np.array([vx, vy, ax, ay])

    def integrate_rk4(self,
                      y0: np.ndarray,
                      t_span: Tuple[float, float],
                      n_steps: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
        t0, tf = t_span
        dt = (tf - t0) / n_steps
        t = np.linspace(t0, tf, n_steps + 1)
        states = np.zeros((n_steps + 1, 4))
        states[0] = y0
        y = np.copy(y0)

        for i in range(n_steps):
            k1 = self.deriv(t[i], y)
            k2 = self.deriv(t[i] + 0.5 * dt, y + 0.5 * dt * k1)
            k3 = self.deriv(t[i] + 0.5 * dt, y + 0.5 * dt * k2)
            k4 = self.deriv(t[i] + dt, y + dt * k3)
            y = y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            states[i + 1] = y
        return t, states

    def compute_lyapunov_exponent(self,
                                   y0: np.ndarray,
                                   t_span: Tuple[float, float],
                                   n_steps: int = 5000,
                                   delta0: float = 1e-8) -> float:
        t, states = self.integrate_rk4(y0, t_span, n_steps)

        y0_pert = y0.copy()
        y0_pert[0] += delta0
        t2, states_pert = self.integrate_rk4(y0_pert, t_span, n_steps)

        delta = states_pert - states
        norms = np.sqrt(np.sum(delta ** 2, axis=1))

        mid = n_steps // 2
        if norms[mid] > 0 and norms[-1] > 0:
            lambda_max = np.log(norms[-1] / norms[mid]) / (t[-1] - t[mid])
        else:
            lambda_max = -np.inf
        return lambda_max


def demo_particles():
    print("\n[Particles] 演示: 蒙特卡洛粒子加速")
    acc = MonteCarloParticleAccelerator(n_particles=5000)
    v0 = acc.sample_initial_velocities(T_thermal=1.0e6, seed=42)
    v_final, energies = acc.accelerate_stochastic(v0, dt=1.0e-7, n_steps=500)
    E_centers, spectrum = acc.energy_spectrum(energies[-100:])
    print(f"  初始平均能量: {np.mean(energies[0]):.3e} eV")
    print(f"  最终平均能量: {np.mean(energies[-1]):.3e} eV")
    print(f"  最大能量: {np.max(energies[-1]):.3e} eV")

    print("\n[Particles] 演示: 非线性轨道追踪")
    tracker = NonlinearOrbitTracker()
    y0 = np.array([0.1, 0.0, 1.0e5, 0.0])
    t, states = tracker.integrate_rk4(y0, (0.0, 1.0e-3), n_steps=2000)
    print(f"  初始位置: ({y0[0]:.3e}, {y0[1]:.3e})")
    print(f"  最终位置: ({states[-1, 0]:.3e}, {states[-1, 1]:.3e})")
    lyap = tracker.compute_lyapunov_exponent(y0, (0.0, 5.0e-3), n_steps=5000)
    print(f"  最大 Lyapunov 指数: {lyap:.3e} (正值表示混沌)")


if __name__ == "__main__":
    demo_particles()
