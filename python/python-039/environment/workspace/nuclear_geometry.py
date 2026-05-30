
import numpy as np
from typing import Tuple, List


class NuclearGeometry:

    def __init__(self, mass_number_a: int = 197, mass_number_b: int = 197,
                 radius_param: float = 1.12, diffuseness: float = 0.54,
                 nucleon_cross_section: float = 4.2):
        self.A = mass_number_a
        self.B = mass_number_b
        self.r0 = radius_param
        self.a = diffuseness
        self.sigma_nn = nucleon_cross_section
        self.R_A = self.r0 * (self.A ** (1.0 / 3.0))
        self.R_B = self.r0 * (self.B ** (1.0 / 3.0))

        self.rho0 = self._compute_rho0(self.A, self.R_A, self.a)

    def _compute_rho0(self, A: int, R: float, a: float) -> float:
        correction = 1.0 + (np.pi ** 2) * (a ** 2) / (R ** 2)
        rho0 = 3.0 * A / (4.0 * np.pi * (R ** 3) * correction)
        return rho0

    def woods_saxon_density(self, r: np.ndarray) -> np.ndarray:
        r = np.asarray(r)

        exponent = (r - self.R_A) / self.a

        exponent = np.clip(exponent, -700, 700)
        density = self.rho0 / (1.0 + np.exp(exponent))
        return density

    def thickness_function(self, x: np.ndarray, y: np.ndarray,
                           nucleus: str = 'A') -> np.ndarray:
        if nucleus == 'A':
            R = self.R_A
        else:
            R = self.R_B

        x = np.asarray(x)
        y = np.asarray(y)
        s2 = x ** 2 + y ** 2

        z_max = R + 10.0 * self.a
        n_z = 200
        z_grid = np.linspace(-z_max, z_max, n_z)
        dz = z_grid[1] - z_grid[0]


        r = np.sqrt(s2[..., np.newaxis] + z_grid ** 2)
        if nucleus == 'A':
            rho_vals = self.woods_saxon_density(r.flatten()).reshape(r.shape)
        else:

            rho_vals = self.woods_saxon_density(r.flatten()).reshape(r.shape)

        thickness = np.trapezoid(rho_vals, z_grid, axis=-1)
        return thickness

    def overlap_function(self, b: float, x_grid: np.ndarray,
                         y_grid: np.ndarray) -> np.ndarray:
        X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
        T_A = self.thickness_function(X, Y, 'A')
        T_B = self.thickness_function(X - b, Y, 'B')
        T_AB = T_A * T_B
        return T_AB

    def compute_npart_ncoll(self, b: float, x_grid: np.ndarray,
                            y_grid: np.ndarray) -> Tuple[float, float]:
        dx = x_grid[1] - x_grid[0]
        dy = y_grid[1] - y_grid[0]
        X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')

        T_A = self.thickness_function(X, Y, 'A')
        T_B = self.thickness_function(X - b, Y, 'B')


        sigma_fm2 = self.sigma_nn * 0.1


        term_A = 1.0 - (1.0 - sigma_fm2 * T_B / self.B) ** self.B
        term_B = 1.0 - (1.0 - sigma_fm2 * T_A / self.A) ** self.A

        N_part = (np.trapezoid(np.trapezoid(T_A * term_A, y_grid, axis=1),
                           x_grid, axis=0) +
                  np.trapezoid(np.trapezoid(T_B * term_B, y_grid, axis=1),
                           x_grid, axis=0))


        N_coll = sigma_fm2 * np.trapezoid(np.trapezoid(T_A * T_B, y_grid, axis=1),
                                      x_grid, axis=0)

        return float(N_part), float(N_coll)

    def eccentricity(self, b: float, x_grid: np.ndarray,
                     y_grid: np.ndarray) -> Tuple[float, float]:
        X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
        T_AB = self.overlap_function(b, x_grid, y_grid)

        dx = x_grid[1] - x_grid[0]
        dy = y_grid[1] - y_grid[0]
        dA = dx * dy

        total = np.sum(T_AB) * dA
        if total < 1e-15:
            return 0.0, 0.0


        x2 = np.sum(X ** 2 * T_AB) * dA / total
        y2 = np.sum(Y ** 2 * T_AB) * dA / total
        xy = np.sum(X * Y * T_AB) * dA / total
        r2 = x2 + y2


        r4 = np.sum((X ** 2 + Y ** 2) ** 2 * T_AB) * dA / total
        cos4phi = np.sum(((X ** 2 + Y ** 2) ** 2) * np.cos(4.0 * np.arctan2(Y, X)) * T_AB) * dA

        eps2 = np.sqrt((x2 - y2) ** 2 + 4.0 * xy ** 2) / r2 if r2 > 1e-15 else 0.0
        eps4 = cos4phi / (r4 * total) if r4 > 1e-15 else 0.0

        return float(eps2), float(eps4)

    def tortoise_boundary_word(self, n_segments: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        phi = np.linspace(0, 2 * np.pi, n_segments, endpoint=False)

        r_surface = self.R_A + self.a * np.log(1.0 / 0.05 - 1.0)
        x = r_surface * np.cos(phi)
        y = r_surface * np.sin(phi)
        return x, y

    def participant_density_profile(self, b: float, x_grid: np.ndarray,
                                    y_grid: np.ndarray) -> np.ndarray:
        X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
        T_A = self.thickness_function(X, Y, 'A')
        T_B = self.thickness_function(X - b, Y, 'B')
        sigma_fm2 = self.sigma_nn * 0.1

        rho_part = (T_A * (1.0 - (1.0 - sigma_fm2 * T_B / self.B) ** self.B) +
                    T_B * (1.0 - (1.0 - sigma_fm2 * T_A / self.A) ** self.A))
        return rho_part
