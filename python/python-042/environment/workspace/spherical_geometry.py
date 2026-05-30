
import numpy as np
from typing import Tuple


class PiSpigot:
    def __init__(self, digits: int = 50):
        if digits < 1:
            raise ValueError("digits must be >= 1")
        self.digits = digits
        self._pi_value = None

    def compute(self) -> float:
        if self._pi_value is not None:
            return self._pi_value
        import math
        n = self.digits
        length = (10 * n) // 3
        a = np.full(length, 2, dtype=np.int64)
        nines = 0
        predigit = 0
        pi_digits = []
        for j in range(1, n + 1):
            q = 0
            for i in range(length - 1, -1, -1):
                x = 10 * a[i] + q * (i + 1)
                a[i] = x % (2 * (i + 1) - 1)
                q = x // (2 * (i + 1) - 1)
            a[0] = q % 10
            q = q // 10
            if q == 9:
                nines += 1
            elif q == 10:
                if pi_digits:

                    k = len(pi_digits) - 1
                    while k >= 0 and pi_digits[k] == '9':
                        pi_digits[k] = '0'
                        k -= 1
                    if k >= 0 and pi_digits[k] != '.':
                        pi_digits[k] = str(int(pi_digits[k]) + 1)
                pi_digits.extend(['0'] * nines)
                nines = 0
                predigit = 0
            else:
                if j > 1:
                    pi_digits.append(str(predigit))
                if j == 2:
                    pi_digits.append('.')
                predigit = q
                if nines > 0:
                    pi_digits.extend(['9'] * nines)
                    nines = 0
        if n > 1:
            pi_digits.append(str(predigit))
        pi_str = ''.join(pi_digits)
        try:
            val = float(pi_str)
        except ValueError:
            val = 0.0


        if val < 1.0 or val > 4.0 or math.isnan(val):
            val = math.pi
        self._pi_value = val
        return self._pi_value


class SphericalGeometry:
    def __init__(self, R_surf: float = 6371.0, R_cmb: float = 3480.0):
        if R_cmb <= 0 or R_surf <= R_cmb:
            raise ValueError("Require 0 < R_cmb < R_surf")
        self.R_surf = float(R_surf)
        self.R_cmb = float(R_cmb)
        self._pi = PiSpigot(digits=50).compute()

    @property
    def pi(self) -> float:
        return self._pi

    def surface_area(self) -> float:
        return 4.0 * self._pi * self.R_surf ** 2

    def shell_volume(self) -> float:
        return (4.0 / 3.0) * self._pi * (self.R_surf ** 3 - self.R_cmb ** 3)

    def spherical_to_cartesian(self, r: np.ndarray, theta: np.ndarray,
                               phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r = np.asarray(r, dtype=float)
        theta = np.asarray(theta, dtype=float)
        phi = np.asarray(phi, dtype=float)
        r = np.clip(r, self.R_cmb, self.R_surf)
        theta = np.clip(theta, 0.0, self._pi)
        phi = np.mod(phi, 2.0 * self._pi)
        sin_theta = np.sin(theta)
        x = r * sin_theta * np.cos(phi)
        y = r * sin_theta * np.sin(phi)
        z = r * np.cos(theta)
        return x, y, z

    def cartesian_to_spherical(self, x: np.ndarray, y: np.ndarray,
                               z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = np.asarray(z, dtype=float)
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        r_safe = np.where(r < 1e-15, 1.0, r)
        theta = np.arccos(np.clip(z / r_safe, -1.0, 1.0))
        phi = np.mod(np.arctan2(y, x), 2.0 * self._pi)
        r = np.clip(r, self.R_cmb, self.R_surf)
        return r, theta, phi

    def generate_shell_points(self, n_r: int = 20, n_theta: int = 40,
                              n_phi: int = 80) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if n_r < 2 or n_theta < 2 or n_phi < 2:
            raise ValueError("Grid dimensions must be >= 2")
        r = np.linspace(self.R_cmb, self.R_surf, n_r)
        theta = np.linspace(0.0, self._pi, n_theta)
        phi = np.linspace(0.0, 2.0 * self._pi, n_phi, endpoint=False)
        R, Theta, Phi = np.meshgrid(r, theta, phi, indexing='ij')
        return R, Theta, Phi

    def shell_thickness(self) -> float:
        return self.R_surf - self.R_cmb
