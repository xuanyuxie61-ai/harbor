
import numpy as np
from scipy.special import ellipj, ellipk
from typing import Tuple






class VesselElasticPendulum:
    def __init__(self, equilibrium_radius: float = 0.005,
                 elastic_modulus_pa: float = 1.0e6,
                 wall_thickness_m: float = 1.0e-3,
                 wall_density_kg_m3: float = 1050.0):
        self.R0 = equilibrium_radius
        self.E = elastic_modulus_pa
        self.h = wall_thickness_m
        self.rho_w = wall_density_kg_m3
        self.g_eff = self.E * self.h / (self.rho_w * self.R0 ** 2)
        self.mass = 2.0 * np.pi * self.R0 * self.h * self.rho_w

    def deriv(self, t: float, y: np.ndarray, external_force: float = 0.0) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if len(y) != 2:
            raise ValueError("State vector must have length 2")
        xi = y[0]
        xi_dot = y[1]
        dydt = np.zeros(2)
        dydt[0] = xi_dot
        dydt[1] = -self.g_eff * np.sin(xi) + external_force / (self.mass + 1e-15)
        return dydt

    def exact_solution(self, t: np.ndarray, xi0: float) -> np.ndarray:
        t = np.atleast_1d(t)
        if abs(xi0) < 1e-12:
            return np.zeros_like(t)

        m = np.sin(xi0 / 2.0) ** 2
        m = np.clip(m, 0.0, 1.0 - 1e-15)
        omega0 = np.sqrt(self.g_eff)
        K = ellipk(m)

        u = K - omega0 * t
        sn, cn, dn, ph = ellipj(u, m)
        xi = 2.0 * np.arcsin(np.clip(np.sin(xi0 / 2.0) * sn, -1.0, 1.0))
        return xi

    def period(self, xi0: float) -> float:
        if abs(xi0) < 1e-12:
            return 2.0 * np.pi / np.sqrt(self.g_eff)
        m = np.sin(xi0 / 2.0) ** 2
        m = np.clip(m, 0.0, 1.0 - 1e-15)
        K = ellipk(m)
        return 4.0 * K / np.sqrt(self.g_eff)

    def energy(self, xi: float, xi_dot: float) -> float:
        kinetic = 0.5 * self.mass * xi_dot ** 2
        potential = self.mass * self.g_eff * (1.0 - np.cos(xi))
        return kinetic + potential


def simulate_vessel_oscillation(pendulum: VesselElasticPendulum,
                                t_span: np.ndarray,
                                xi0: float,
                                external_pressure_pa: np.ndarray = None) -> dict:
    n = len(t_span)
    xi = np.zeros(n)
    xi_dot = np.zeros(n)
    energy = np.zeros(n)

    xi[0] = xi0
    xi_dot[0] = 0.0
    energy[0] = pendulum.energy(xi0, 0.0)

    dt = t_span[1] - t_span[0] if n > 1 else 1e-3

    for i in range(1, n):
        f_ext = 0.0
        if external_pressure_pa is not None:

            f_ext = external_pressure_pa[i] * 2.0 * np.pi * pendulum.R0


        y = np.array([xi[i - 1], xi_dot[i - 1]])
        dydt = pendulum.deriv(t_span[i - 1], y, f_ext)

        xi_dot[i] = xi_dot[i - 1] + dt * dydt[1]
        xi[i] = xi[i - 1] + dt * xi_dot[i]


        xi[i] = np.clip(xi[i], -0.5, 0.5)

        energy[i] = pendulum.energy(xi[i], xi_dot[i])

    return {
        "displacement": xi,
        "velocity": xi_dot,
        "energy": energy,
        "radius": pendulum.R0 * (1.0 + xi)
    }






def rbc_interaction_force(positions: np.ndarray,
                          radii: float = 2.5e-6,
                          repulsion_strength: float = 1e-12,
                          attraction_strength: float = 1e-14,
                          attraction_range: float = 1e-5) -> np.ndarray:
    positions = np.asarray(positions, dtype=float)
    n_particles, dim = positions.shape
    forces = np.zeros_like(positions)

    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            r_vec = positions[j] - positions[i]
            r = np.linalg.norm(r_vec)
            if r < 1e-15 or r > 5.0 * attraction_range:
                continue


            if r < 2.0 * radii:
                f_mag = repulsion_strength * ((2.0 * radii / r) ** 6 - 1.0)
            else:
                f_mag = 0.0


            if r < attraction_range:
                f_mag -= attraction_strength / (r ** 3)

            f_vec = f_mag * r_vec / (r + 1e-15)
            forces[i] -= f_vec
            forces[j] += f_vec

    return forces


def update_rbc_positions_euler(positions: np.ndarray, velocities: np.ndarray,
                               forces: np.ndarray, dt: float,
                               mass_kg: float = 1e-13) -> Tuple[np.ndarray, np.ndarray]:
    accel = forces / (mass_kg + 1e-20)
    new_vel = velocities + dt * accel
    new_pos = positions + dt * new_vel
    return new_pos, new_vel


def apparent_viscosity_from_rbc(n_rbc: int, domain_volume: float,
                                base_viscosity: float = 0.0012) -> float:
    rbc_volume = 4.0 / 3.0 * np.pi * (2.5e-6) ** 3
    phi = n_rbc * rbc_volume / (domain_volume + 1e-20)
    phi = np.clip(phi, 0.0, 0.6)
    return base_viscosity * (1.0 + 2.5 * phi + 6.2 * phi * phi)
