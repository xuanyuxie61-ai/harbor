# -*- coding: utf-8 -*-

import numpy as np
from typing import Tuple, Optional, Callable


class TransportSolver:

    def __init__(self, nx: int, ny: int, dx: float, dy: float,
                 dt: float, R: float = 1.0, lambda_decay: float = 0.0):
        if nx <= 0 or ny <= 0:
            raise ValueError("nx 和 ny 必须为正")
        if dx <= 0 or dy <= 0 or dt <= 0:
            raise ValueError("dx, dy, dt 必须为正")
        if R <= 0:
            raise ValueError("R 必须为正")
        if lambda_decay < 0:
            raise ValueError("lambda_decay 必须为非负")

        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.R = R
        self.lambda_decay = lambda_decay

        self.concentration = np.zeros((ny, nx))
        self.source = np.zeros((ny, nx))


        self.mass_history = []
        self.time_history = []

    def set_velocity_field(self, vx: np.ndarray, vy: np.ndarray):
        if vx.shape != (self.ny, self.nx) or vy.shape != (self.ny, self.nx):
            raise ValueError("流速场形状必须与网格匹配")
        self.vx = vx
        self.vy = vy

    def set_dispersivity(self, alpha_L: float, alpha_T: float,
                         D_m: float = 1.0e-9):
        if alpha_L < 0 or alpha_T < 0 or D_m < 0:
            raise ValueError("弥散参数必须为非负")

        self.alpha_L = alpha_L
        self.alpha_T = alpha_T
        self.D_m = D_m


        v_mag = np.sqrt(self.vx ** 2 + self.vy ** 2)
        self.Dxx = D_m + alpha_L * v_mag
        self.Dyy = D_m + alpha_T * v_mag
        self.Dxy = np.zeros_like(v_mag)

    def _compute_flux_x(self, C: np.ndarray, i: int, j: int) -> float:
        if j <= 0 or j >= self.nx - 1:
            return 0.0


        vx_ij = self.vx[i, j]
        if vx_ij > 0:
            adv = vx_ij * (C[i, j] - C[i, j-1]) / self.dx
        else:
            adv = vx_ij * (C[i, j+1] - C[i, j]) / self.dx


        D_face = 0.5 * (self.Dxx[i, j] + self.Dxx[i, j+1])
        diff_p = D_face * (C[i, j+1] - C[i, j]) / self.dx**2

        D_face_m = 0.5 * (self.Dxx[i, j] + self.Dxx[i, j-1])
        diff_m = D_face_m * (C[i, j] - C[i, j-1]) / self.dx**2

        return -adv + diff_p - diff_m

    def _compute_flux_y(self, C: np.ndarray, i: int, j: int) -> float:
        if i <= 0 or i >= self.ny - 1:
            return 0.0

        vy_ij = self.vy[i, j]
        if vy_ij > 0:
            adv = vy_ij * (C[i, j] - C[i-1, j]) / self.dy
        else:
            adv = vy_ij * (C[i+1, j] - C[i, j]) / self.dy

        D_face = 0.5 * (self.Dyy[i, j] + self.Dyy[i+1, j])
        diff_p = D_face * (C[i+1, j] - C[i, j]) / self.dy**2

        D_face_m = 0.5 * (self.Dyy[i, j] + self.Dyy[i-1, j])
        diff_m = D_face_m * (C[i, j] - C[i-1, j]) / self.dy**2

        return -adv + diff_p - diff_m

    def explicit_step(self) -> np.ndarray:

        pass

    def solve(self, n_steps: int, injection_zone: Optional[Tuple] = None,
              C_inject: float = 1.0, check_mass: bool = True) -> dict:
        if n_steps <= 0:
            raise ValueError("n_steps 必须为正")

        if injection_zone is not None:
            i_min, i_max, j_min, j_max = injection_zone
            self.concentration[i_min:i_max, j_min:j_max] = C_inject

        self.mass_history = []
        self.time_history = []

        for step in range(n_steps):
            self.explicit_step()

            if check_mass:
                mass = self.compute_total_mass()
                self.mass_history.append(mass)
                self.time_history.append((step + 1) * self.dt)

        result = {
            'concentration': self.concentration.copy(),
            'final_mass': self.compute_total_mass(),
            'mass_history': np.array(self.mass_history),
            'time_history': np.array(self.time_history)
        }

        if check_mass and len(self.mass_history) > 0:
            result['mass_conservation_error'] = self._mass_conservation_error()

        return result

    def compute_total_mass(self) -> float:
        return float(np.sum(self.concentration) * self.dx * self.dy)

    def _mass_conservation_error(self) -> float:
        if len(self.mass_history) == 0:
            return 0.0

        M0 = self.mass_history[0] if len(self.mass_history) > 0 else 1.0
        if abs(M0) < 1e-20:
            return 0.0

        t_final = self.time_history[-1]
        if self.lambda_decay > 0:
            M_expected = M0 * np.exp(-self.lambda_decay * t_final)
        else:
            M_expected = M0

        M_actual = self.mass_history[-1]
        rel_error = abs(M_actual - M_expected) / abs(M_expected) if abs(M_expected) > 1e-20 else 0.0
        return float(rel_error)

    def breakthrough_curve(self, outlet_zone: Tuple,
                            n_steps: int, injection_zone: Tuple,
                            C_inject: float = 1.0) -> dict:
        times = []
        concentrations = []
        masses = []


        self.concentration = np.zeros((self.ny, self.nx))
        i_min, i_max, j_min, j_max = injection_zone
        self.concentration[i_min:i_max, j_min:j_max] = C_inject

        oi_min, oi_max, oj_min, oj_max = outlet_zone

        for step in range(n_steps):
            self.explicit_step()


            C_out = np.mean(self.concentration[oi_min:oi_max, oj_min:oj_max])
            times.append((step + 1) * self.dt)
            concentrations.append(C_out)
            masses.append(self.compute_total_mass())

        return {
            'times': np.array(times),
            'concentrations': np.array(concentrations),
            'masses': np.array(masses)
        }

    def stability_check(self) -> dict:
        vx_max = np.max(np.abs(self.vx))
        vy_max = np.max(np.abs(self.vy))

        cfl_x = vx_max * self.dt / self.dx
        cfl_y = vy_max * self.dt / self.dy
        cfl = cfl_x + cfl_y

        D_max = np.max(self.Dxx)
        diff_num = D_max * self.dt / (self.dx ** 2)

        return {
            'CFL': float(cfl),
            'CFL_x': float(cfl_x),
            'CFL_y': float(cfl_y),
            'diffusion_number': float(diff_num),
            'stable': cfl < 1.0 and diff_num < 0.5
        }
