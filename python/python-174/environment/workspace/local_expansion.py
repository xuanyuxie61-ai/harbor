
import numpy as np
from spherical_geometry import legendre_associated_normalized


def hermite_gauss_nodes_weights(n):
    if n <= 0:
        raise ValueError("n必须为正整数")





    from numpy.polynomial.hermite import hermgauss
    x, w = hermgauss(n)
    return x, w


def integrate_kernel_hermite(f, n=16):
    x, w = hermite_gauss_nodes_weights(n)
    vals = np.array([f(xi) for xi in x])



    return float(np.sum(w * vals))


class LocalExpansion:

    def __init__(self, center, order):
        self.center = np.asarray(center, dtype=float)
        self.order = int(order)
        if self.order < 0:
            raise ValueError("order必须非负")
        self.coeffs_real = []
        self.coeffs_imag = []
        for l in range(self.order + 1):
            self.coeffs_real.append(np.zeros(l + 1))
            self.coeffs_imag.append(np.zeros(l + 1))

    def add_source_contribution(self, source_points, source_charges):
        source_points = np.atleast_2d(source_points)
        source_charges = np.asarray(source_charges, dtype=float)
        if source_points.shape[0] != source_charges.shape[0]:
            raise ValueError("长度不匹配")

        local = source_points - self.center
        r = np.linalg.norm(local, axis=1)
        r = np.where(r < 1e-15, 1e-15, r)
        theta = np.arccos(np.clip(local[:, 2] / r, -1.0, 1.0))
        phi = np.arctan2(local[:, 1], local[:, 0])
        phi = np.where(phi < 0, phi + 2.0 * np.pi, phi)

        for i in range(source_points.shape[0]):
            for l in range(self.order + 1):
                inv_r_lp1 = 1.0 / (r[i] ** (l + 1))
                plm_0 = legendre_associated_normalized(l, 0, np.cos(theta[i]))
                y_0 = plm_0[l]
                self.coeffs_real[l][0] += source_charges[i] * inv_r_lp1 * y_0
                for m in range(1, l + 1):
                    plm_m = legendre_associated_normalized(l, m, np.cos(theta[i]))
                    y_real = plm_m[l] * np.cos(m * phi[i])
                    y_imag = plm_m[l] * np.sin(m * phi[i])
                    self.coeffs_real[l][m] += source_charges[i] * inv_r_lp1 * y_real
                    self.coeffs_imag[l][m] += source_charges[i] * inv_r_lp1 * y_imag

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
            R_l = R ** l
            plm_0 = np.array([legendre_associated_normalized(l, 0, np.cos(t)) for t in theta])
            y_0 = plm_0[:, l]
            potential += self.coeffs_real[l][0] * R_l * y_0
            for m in range(1, l + 1):
                plm_m = np.array([legendre_associated_normalized(l, m, np.cos(t)) for t in theta])
                y_real = plm_m[:, l] * np.cos(m * phi)
                y_imag = plm_m[:, l] * np.sin(m * phi)
                potential += R_l * (
                    self.coeffs_real[l][m] * y_real +
                    self.coeffs_imag[l][m] * y_imag
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

    def wedge_moment_integral(self, exponents):
        e1, e2, e3 = int(exponents[0]), int(exponents[1]), int(exponents[2])
        if e1 < 0 or e2 < 0:
            raise ValueError("e1,e2必须非负")
        if e3 == -1:
            raise ValueError("e3不能为-1")


        value = 1.0
        k = e1
        for i in range(1, e2 + 1):
            k = k + 1
            value = value * i / k
        k = k + 1
        value = value / k
        k = k + 1
        value = value / k


        if e3 % 2 == 1:
            value = 0.0
        else:
            value = value * 2.0 / (e3 + 1)

        return float(value)

    def get_coefficients_norm(self):
        norm_sq = 0.0
        for l in range(self.order + 1):
            for m in range(l + 1):
                norm_sq += self.coeffs_real[l][m] ** 2 + self.coeffs_imag[l][m] ** 2
        return np.sqrt(norm_sq)
