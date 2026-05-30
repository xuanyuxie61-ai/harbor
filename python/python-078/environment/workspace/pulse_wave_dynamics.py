
import numpy as np
from typing import Tuple






def shallow_water_lax_wendroff(A: np.ndarray, Q: np.ndarray,
                               dx: float, dt: float,
                               g_eff: float = 1.0,
                               n_steps: int = 1,
                               boundary_type: str = "reflecting") -> Tuple[np.ndarray, np.ndarray]:
    A = np.asarray(A, dtype=float).copy()
    Q = np.asarray(Q, dtype=float).copy()
    n_grid = len(A)


    cfl = dt / dx
    max_speed = np.max(np.sqrt(g_eff * A + 1e-15) + np.abs(Q / (A + 1e-15)))
    if max_speed * cfl > 1.0:

        dt = 0.9 * dx / (max_speed + 1e-15)
        cfl = dt / dx

    def flux(AA, QQ):
        F1 = QQ
        F2 = QQ ** 2 / (AA + 1e-15) + 0.5 * g_eff * AA ** 2
        return np.array([F1, F2])

    def apply_boundary(AA, QQ):
        if boundary_type == "reflecting":
            AA[0] = AA[1]
            AA[-1] = AA[-2]
            QQ[0] = 0.0
            QQ[-1] = QQ[-2]
        elif boundary_type == "periodic":
            AA[0] = AA[-2]
            AA[-1] = AA[1]
            QQ[0] = QQ[-2]
            QQ[-1] = QQ[1]
        return AA, QQ

    for _ in range(n_steps):
        A, Q = apply_boundary(A, Q)


        U = np.vstack([A, Q])
        F = np.zeros((2, n_grid))
        for j in range(n_grid):
            F[:, j] = flux(A[j], Q[j])

        U_half = np.zeros((2, n_grid - 1))
        for j in range(n_grid - 1):
            U_half[:, j] = 0.5 * (U[:, j] + U[:, j + 1]) - 0.5 * cfl * (F[:, j + 1] - F[:, j])

        A_half = U_half[0, :]
        Q_half = U_half[1, :]


        F_half = np.zeros((2, n_grid - 1))
        for j in range(n_grid - 1):
            F_half[:, j] = flux(A_half[j], Q_half[j])

        U_new = U.copy()
        for j in range(1, n_grid - 1):
            U_new[:, j] = U[:, j] - cfl * (F_half[:, j] - F_half[:, j - 1])

        A = U_new[0, :]
        Q = U_new[1, :]

    return A, Q


def pressure_wave_speed(elastic_modulus_pa: float,
                        wall_thickness_m: float,
                        vessel_radius_m: float,
                        blood_density_kg_m3: float = 1060.0) -> float:
    if vessel_radius_m <= 0 or wall_thickness_m <= 0:
        raise ValueError("Radius and thickness must be positive")
    return np.sqrt(elastic_modulus_pa * wall_thickness_m /
                   (2.0 * blood_density_kg_m3 * vessel_radius_m))






class NLSEPressurePulse:
    def __init__(self, nx: int, z_min: float, z_max: float,
                 gamma: float = 0.5, dx: float = None):
        self.nx = nx
        self.z = np.linspace(z_min, z_max, nx)
        self.dz = self.z[1] - self.z[0] if dx is None else dx
        self.gamma = gamma


        self.im1 = np.array([1] + list(range(nx - 2)) + [nx - 2])
        self.i = np.arange(nx)
        self.ip1 = np.array([1] + list(range(2, nx)) + [nx - 2])

    def initial_double_soliton(self, z: np.ndarray,
                                amplitude: float = 1.0,
                                c1: float = 2.0, c2: float = 0.5,
                                delta: float = 5.0) -> np.ndarray:
        alpha = amplitude
        prefactor = np.sqrt(alpha) * np.sqrt(2.0 / (self.gamma + 1e-15))

        xi1 = np.sqrt(alpha) * (z - c1 * 0.0)
        xi2 = np.sqrt(alpha) * (z - delta - c2 * 0.0)

        phi1 = 1j * ((c1 / 2.0) * z)
        phi2 = 1j * ((c2 / 2.0) * (z - delta))

        psi = prefactor * (np.exp(phi1) / np.cosh(xi1 + 1e-15) +
                           np.exp(phi2) / np.cosh(xi2 + 1e-15))
        return psi

    def deriv(self, psi: np.ndarray, t: float = 0.0) -> np.ndarray:
        psi_im1 = psi[self.im1]
        psi_i = psi[self.i]
        psi_ip1 = psi[self.ip1]

        psi_zz = (psi_ip1 - 2.0 * psi_i + psi_im1) / (self.dz ** 2)
        nonlinear = self.gamma * np.abs(psi_i) ** 2 * psi_i
        dpsi_dt = 1j * (psi_zz + nonlinear)
        return dpsi_dt

    def mass_conservation(self, psi: np.ndarray) -> float:
        n = len(psi)
        m = self.dz * (-0.5 * np.abs(psi[0]) ** 2 +
                       np.sum(np.abs(psi[1:n - 1]) ** 2) -
                       0.5 * np.abs(psi[-1]) ** 2)
        return float(np.abs(m))

    def step_rk4(self, psi: np.ndarray, dt: float) -> np.ndarray:
        k1 = self.deriv(psi)
        k2 = self.deriv(psi + 0.5 * dt * k1)
        k3 = self.deriv(psi + 0.5 * dt * k2)
        k4 = self.deriv(psi + dt * k3)
        return psi + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def evolve(self, psi0: np.ndarray, dt: float, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        psi = psi0.copy()
        mass_hist = np.zeros(n_steps)

        for n in range(n_steps):
            psi = self.step_rk4(psi, dt)
            mass_hist[n] = self.mass_conservation(psi)

        return psi, mass_hist


def nlse_to_pressure_amplitude(psi: np.ndarray,
                               base_pressure_pa: float = 10000.0,
                               conversion_factor: float = 5000.0) -> np.ndarray:
    return base_pressure_pa + conversion_factor * np.abs(psi)
