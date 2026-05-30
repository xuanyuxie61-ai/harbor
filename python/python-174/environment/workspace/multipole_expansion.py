
import numpy as np
from spherical_geometry import legendre_associated_normalized


class MultipoleExpansion:

    def __init__(self, center, order):
        self.center = np.asarray(center, dtype=float)
        self.order = int(order)
        if self.order < 0:
            raise ValueError("order必须非负")


        self.moments_real = []
        self.moments_imag = []
        for l in range(self.order + 1):
            self.moments_real.append(np.zeros(l + 1))
            self.moments_imag.append(np.zeros(l + 1))
        self.total_charge = 0.0

    def _shift_to_local(self, points):
        return points - self.center

    def add_particles(self, points, charges):
        points = np.atleast_2d(points)
        charges = np.asarray(charges, dtype=float)
        if points.shape[0] != charges.shape[0]:
            raise ValueError("points和charges长度不匹配")

        local = self._shift_to_local(points)
        r = np.linalg.norm(local, axis=1)

        theta = np.arccos(np.clip(local[:, 2] / (r + 1e-15), -1.0, 1.0))
        phi = np.arctan2(local[:, 1], local[:, 0])
        phi = np.where(phi < 0, phi + 2.0 * np.pi, phi)






        raise NotImplementedError("Hole_1: 请实现add_particles中的多极矩计算")

    def evaluate_potential(self, target):
        target = np.atleast_2d(target)
        local = target - self.center
        R = np.linalg.norm(local, axis=1)

        R = np.where(R < 1e-15, 1e-15, R)
        theta = np.arccos(np.clip(local[:, 2] / R, -1.0, 1.0))
        phi = np.arctan2(local[:, 1], local[:, 0])
        phi = np.where(phi < 0, phi + 2.0 * np.pi, phi)

        N = target.shape[0]
        potential = np.zeros(N)

        for l in range(self.order + 1):
            inv_R_lp1 = 1.0 / (R ** (l + 1))
            plm_0 = np.array([legendre_associated_normalized(l, 0, np.cos(t)) for t in theta])
            y_0 = plm_0[:, l]
            potential += self.moments_real[l][0] * inv_R_lp1 * y_0
            for m in range(1, l + 1):
                plm_m = np.array([legendre_associated_normalized(l, m, np.cos(t)) for t in theta])
                y_real = plm_m[:, l] * np.cos(m * phi)
                y_imag = plm_m[:, l] * np.sin(m * phi)
                potential += inv_R_lp1 * (
                    self.moments_real[l][m] * y_real +
                    self.moments_imag[l][m] * y_imag
                )
        return potential

    def evaluate_field(self, target):
        target = np.atleast_2d(target)
        N = target.shape[0]
        h = 1e-6
        field = np.zeros((N, 3))
        for d in range(3):
            offset = np.zeros(3)
            offset[d] = h
            phi_plus = self.evaluate_potential(target + offset)
            phi_minus = self.evaluate_potential(target - offset)
            field[:, d] = -(phi_plus - phi_minus) / (2.0 * h)
        return field

    def get_moments_l2_norm(self):
        norm_sq = 0.0
        for l in range(self.order + 1):
            for m in range(l + 1):
                norm_sq += self.moments_real[l][m] ** 2 + self.moments_imag[l][m] ** 2
        return np.sqrt(norm_sq)

    def truncation_error_bound(self, target, total_charge_magnitude, source_radius):
        target = np.atleast_2d(target)
        R = np.linalg.norm(target - self.center, axis=1)
        R = np.where(R < 1e-15, 1e-15, R)
        L = self.order
        a = source_radius
        Q = total_charge_magnitude
        denom = R ** (L + 2) - a * (R ** (L + 1))
        denom = np.where(denom < 1e-15, 1e-15, denom)
        error = Q * (a ** (L + 1)) / denom
        return error
