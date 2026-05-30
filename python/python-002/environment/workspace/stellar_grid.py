# -*- coding: utf-8 -*-

import numpy as np
from typing import List, Tuple, Optional


class StellarGrid:

    def __init__(self, M_total: float, N_shells: int = 200,
                 core_fraction: float = 0.05, envelope_fraction: float = 0.95):
        if M_total <= 0:
            raise ValueError("恒星总质量必须为正")
        if N_shells < 10:
            raise ValueError("壳层数至少为10")
        self.M_total = M_total
        self.N_shells = N_shells
        self.core_fraction = max(0.0, min(core_fraction, 1.0))
        self.envelope_fraction = max(0.0, min(envelope_fraction, 1.0))




        self.mass = self._generate_mass_grid()
        self.dm = np.diff(self.mass)
        self.dm = np.append(self.dm, self.dm[-1])

        self.mass = np.clip(self.mass, 0.0, M_total)


        self.radius = np.zeros(N_shells, dtype=np.float64)
        self.density = np.zeros(N_shells, dtype=np.float64)
        self.temperature = np.zeros(N_shells, dtype=np.float64)
        self.pressure = np.zeros(N_shells, dtype=np.float64)
        self.luminosity = np.zeros(N_shells, dtype=np.float64)


        self.boundary_words: List[str] = []
        self._generate_boundary_words()

    def _generate_mass_grid(self) -> np.ndarray:
        u = np.linspace(0.0, 1.0, self.N_shells)

        alpha = 3.0
        uc = 0.5

        tanh_alpha = np.tanh(alpha * uc)
        if tanh_alpha > 1e-10:
            s = 0.5 * (np.tanh(alpha * (u - uc)) / tanh_alpha + 1.0)
        else:
            s = u
        s = np.clip(s, 0.0, 1.0)
        return s * self.M_total

    def _generate_boundary_words(self):
        self.boundary_words = []
        for i in range(self.N_shells):
            word = []
            if i == 0:
                word.append('A')
            else:
                dm = self.mass[i] - self.mass[i - 1]
                if dm > 0:
                    word.append('A')
                else:
                    word.append('a')
            if i == self.N_shells - 1:
                word.append('B')
            else:
                word.append('b')
            self.boundary_words.append(''.join(word))

    def get_shell_index(self, mass_coord: float) -> int:
        if mass_coord <= 0:
            return 0
        if mass_coord >= self.M_total:
            return self.N_shells - 1
        idx = np.searchsorted(self.mass, mass_coord)
        return min(idx, self.N_shells - 1)

    def get_core_shells(self) -> slice:
        m_core = self.core_fraction * self.M_total
        end = np.searchsorted(self.mass, m_core)
        return slice(0, min(end + 1, self.N_shells))

    def get_envelope_shells(self) -> slice:
        m_env = (1.0 - self.envelope_fraction) * self.M_total
        start = np.searchsorted(self.mass, m_env)
        return slice(max(start, 0), self.N_shells)

    def get_radiative_zone(self, convection_mask: np.ndarray) -> np.ndarray:
        conv = np.asarray(convection_mask, dtype=bool)
        return np.where(~conv)[0]

    def shell_mass(self, i: int) -> float:
        if i < 0 or i >= self.N_shells:
            raise IndexError("壳层索引越界")
        if i == 0:
            return self.mass[1] - self.mass[0] if self.N_shells > 1 else self.mass[0]
        elif i == self.N_shells - 1:
            return self.mass[-1] - self.mass[-2] if self.N_shells > 1 else self.mass[-1]
        else:
            return 0.5 * (self.mass[i + 1] - self.mass[i - 1])

    def remap_grid(self, new_mass: np.ndarray):
        new_mass = np.asarray(new_mass, dtype=np.float64)
        new_mass = np.clip(new_mass, 0.0, self.M_total)
        new_N = len(new_mass)

        old_mass = self.mass.copy()
        old_radius = self.radius.copy()
        old_density = self.density.copy()
        old_temperature = self.temperature.copy()
        old_pressure = self.pressure.copy()
        old_luminosity = self.luminosity.copy()

        self.N_shells = new_N
        self.mass = new_mass
        self.dm = np.diff(new_mass)
        self.dm = np.append(self.dm, self.dm[-1])


        def safe_interp(x_old, y_old, x_new):
            y_new = np.interp(x_new, x_old, y_old)
            return y_new

        self.radius = safe_interp(old_mass, old_radius, new_mass)
        self.density = safe_interp(old_mass, old_density, new_mass)
        self.temperature = safe_interp(old_mass, old_temperature, new_mass)
        self.pressure = safe_interp(old_mass, old_pressure, new_mass)
        self.luminosity = safe_interp(old_mass, old_luminosity, new_mass)


        self.radius = np.maximum(self.radius, 0.0)
        self.density = np.maximum(self.density, 1e-10)
        self.temperature = np.maximum(self.temperature, 1e3)
        self.pressure = np.maximum(self.pressure, 1e-5)
        self.luminosity = np.maximum(self.luminosity, 0.0)

        self._generate_boundary_words()

    def compute_volumes(self) -> np.ndarray:
        r = self.radius

        r_interface = np.zeros(self.N_shells + 1, dtype=np.float64)
        r_interface[0] = 0.0
        for i in range(1, self.N_shells):
            r_interface[i] = 0.5 * (r[i - 1] + r[i])
        r_interface[-1] = r[-1] if self.N_shells > 1 else r[0]
        volumes = (4.0 / 3.0) * np.pi * (r_interface[1:] ** 3 - r_interface[:-1] ** 3)
        return np.maximum(volumes, 1e-30)

    def to_mass_coordinates(self, f_r: np.ndarray) -> np.ndarray:
        return np.asarray(f_r, dtype=np.float64)
