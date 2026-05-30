# -*- coding: utf-8 -*-

import numpy as np
from parameters import get_parameters


class ErosionQuadrature:

    def __init__(self, params=None):
        if params is None:
            params = get_parameters()
        self.params = params
        self.U_0 = params.get('E_bind')
        self.E_th = params.get('E_threshold')
        self.Z_wall = params.get('wall_Z')
        self.M_wall = params.get('wall_M')
        self.Z_i = params.get('Z_i')
        self.M_i = params.get('m_i')

    def nuclear_stopping_krc(self, epsilon):
        epsilon = np.asarray(epsilon, dtype=float)

        eps_safe = np.where(epsilon < 1.0e-10, 1.0e-10, epsilon)

        numerator = 0.5 * np.log(1.0 + 1.2288 * eps_safe)
        denominator = eps_safe + 0.1728 * np.sqrt(eps_safe) + 0.008 * eps_safe**0.1504

        s_n = numerator / denominator
        return s_n

    def reduced_energy(self, E_ev):
        a0 = 0.5291772108e-10
        e_charge = 1.602176634e-19
        epsilon_0 = 8.854187817e-12


        z_sum = self.Z_i**(2.0/3.0) + self.Z_wall**(2.0/3.0)
        a_screen = 0.8854 * a0 / np.sqrt(z_sum)


        coulomb_factor = self.Z_i * self.Z_wall * e_charge**2 / (4.0 * np.pi * epsilon_0)


        mass_ratio = self.M_wall / (self.M_i + self.M_wall)
        E_joule = E_ev * e_charge
        epsilon = mass_ratio * E_joule * a_screen / coulomb_factor


        if epsilon < 1.0e-10:
            epsilon = 1.0e-10
        return epsilon

    def sputtering_yield_bohdansky(self, E_ev, theta=0.0):
        E_ev = float(E_ev)
        if E_ev <= self.E_th:
            return 0.0

        epsilon = self.reduced_energy(E_ev)
        s_n = self.nuclear_stopping_krc(epsilon)

        Q = (self.Z_wall / max(self.Z_i, 1))**0.2 * (self.M_i / max(self.M_wall, 1))**0.15

        ratio = self.E_th / E_ev
        if ratio >= 1.0:
            return 0.0

        Y = 0.042 * Q * s_n * (ratio**0.25) * ((1.0 - ratio)**2.5)


        if theta != 0.0:
            theta_r = np.radians(78.0)
            cos_theta = np.cos(theta)
            f_theta = np.exp(-((theta - theta_r) / 0.8)**2) + 0.5 * cos_theta**(-1.5)
            f_theta = min(f_theta, 5.0)
            Y *= f_theta


        if Y < 0:
            Y = 0.0
        if Y > 100.0:
            Y = 100.0

        return Y

    def energy_deposition_profile(self, E_ev, x_depth):


        R_p = 1.0e-9 * 10.0 * (E_ev**0.6) / (self.Z_i * self.Z_wall * max(self.M_i, 1))
        sigma = R_p * 0.3

        if sigma <= 0:
            return np.zeros_like(x_depth)

        dep = E_ev * np.exp(-0.5 * ((x_depth - R_p) / sigma)**2) / (sigma * np.sqrt(2.0 * np.pi))
        return dep



    @staticmethod
    def clenshaw_curtis_rule(n):
        if n < 1:
            raise ValueError("n 必须 >= 1")

        x = np.zeros(n)
        w = np.zeros(n)

        if n == 1:
            x[0] = 0.0
            w[0] = 2.0
            return x, w

        for i in range(n):
            x[i] = np.cos(np.pi * (n - 1 - i) / (n - 1))

        w[:] = 1.0
        for i in range(n):
            theta = np.pi * i / (n - 1)
            jhi = (n - 1) // 2
            for j in range(1, jhi + 1):
                if 2 * j == n - 1:
                    b = 1.0
                else:
                    b = 2.0
                w[i] -= b * np.cos(2.0 * j * theta) / (4.0 * j * j - 1.0)

        w[0] /= (n - 1)
        w[1:-1] = 2.0 * w[1:-1] / (n - 1)
        w[-1] /= (n - 1)

        return x, w

    @staticmethod
    def chebyshev1_rule(n):
        if n < 1:
            raise ValueError("n 必须 >= 1")

        x = np.zeros(n)
        w = np.full(n, np.pi / n)

        for i in range(n):
            x[i] = np.cos(np.pi * (2.0 * n - 1.0 - 2.0 * i) / (2.0 * n))

        return x, w

    @staticmethod
    def triangle_unit_o03():
        w = np.ones(3) / 3.0
        xy = np.array([
            [2.0/3.0, 1.0/6.0],
            [1.0/6.0, 2.0/3.0],
            [1.0/6.0, 1.0/6.0]
        ]).T
        return w, xy

    @staticmethod
    def triangle_unit_o12():
        w = np.array([
            0.050844906370206816921,
            0.050844906370206816921,
            0.050844906370206816921,
            0.11678627572637936603,
            0.11678627572637936603,
            0.11678627572637936603,
            0.082851075618373575194,
            0.082851075618373575194,
            0.082851075618373575194,
            0.082851075618373575194,
            0.082851075618373575194,
            0.082851075618373575194,
        ])
        xy = np.array([
            [0.87382197101699554332, 0.063089014491502228340],
            [0.063089014491502228340, 0.87382197101699554332],
            [0.063089014491502228340, 0.063089014491502228340],
            [0.50142650965817915742, 0.24928674517091042129],
            [0.24928674517091042129, 0.50142650965817915742],
            [0.24928674517091042129, 0.24928674517091042129],
            [0.053145049844816947353, 0.31035245103378440542],
            [0.31035245103378440542, 0.053145049844816947353],
            [0.053145049844816947353, 0.63650249912139864723],
            [0.31035245103378440542, 0.63650249912139864723],
            [0.63650249912139864723, 0.053145049844816947353],
            [0.63650249912139864723, 0.31035245103378440542],
        ]).T
        return w, xy

    @staticmethod
    def triangle_unit_monomial_integral(expon):
        m, n = int(expon[0]), int(expon[1])
        if m < 0 or n < 0:
            return 0.0

        value = 1.0
        k = m
        for i in range(1, n + 1):
            k += 1
            value *= i / k
        k += 1
        value /= k
        k += 1
        value /= k
        return value

    def integrate_sputtering_yield_1d(self, E_min, E_max, n_points=64):
        x_cc, w_cc = self.clenshaw_curtis_rule(n_points)


        E_nodes = 0.5 * (E_max - E_min) * x_cc + 0.5 * (E_max + E_min)
        jacobian = 0.5 * (E_max - E_min)


        T_eff = self.params.get('T_e')
        if T_eff <= 0:
            T_eff = 50.0

        integrand = np.zeros(n_points)
        for i in range(n_points):
            E = E_nodes[i]
            if E <= self.E_th:
                integrand[i] = 0.0
            else:
                Y = self.sputtering_yield_bohdansky(E)
                f_E = (E / T_eff) * np.exp(-E / T_eff)
                integrand[i] = Y * f_E

        result = jacobian * np.sum(w_cc * integrand)
        return result, E_nodes, integrand

    def integrate_erosion_over_triangle(self, triangle_vertices, gamma_func, E_func):
        w, xy_ref = self.triangle_unit_o12()


        v0 = triangle_vertices[0]
        v1 = triangle_vertices[1]
        v2 = triangle_vertices[2]


        J = np.array([[v1[0]-v0[0], v2[0]-v0[0]],
                      [v1[1]-v0[1], v2[1]-v0[1]]])
        det_J = abs(np.linalg.det(J))
        if det_J < 1.0e-20:
            return 0.0

        area = 0.5 * det_J

        total = 0.0
        for i in range(len(w)):

            xi, eta = xy_ref[0, i], xy_ref[1, i]

            x = v0[0] + xi * (v1[0] - v0[0]) + eta * (v2[0] - v0[0])
            y = v0[1] + xi * (v1[1] - v0[1]) + eta * (v2[1] - v0[1])

            gamma = gamma_func(x, y)
            E = E_func(x, y)

            if E > self.E_th:
                Y = self.sputtering_yield_bohdansky(E)
                total += w[i] * Y * gamma


        total *= area
        return total


def demo_erosion():
    eq = ErosionQuadrature()


    energies = [50, 100, 200, 500, 1000, 2000]
    print("物理溅射产额 (D -> W):")
    for E in energies:
        Y = eq.sputtering_yield_bohdansky(E)
        print(f"  E = {E:5d} eV, Y = {Y:.4f}")


    result, _, _ = eq.integrate_sputtering_yield_1d(10.0, 5000.0, n_points=64)
    print(f"\n能量加权平均溅射产额 = {result:.4f}")


    tri = np.array([[0.0, 0.0], [1.0e-3, 0.0], [0.5e-3, 1.0e-3]])
    def gamma_f(x, y):
        return 1.0e22
    def E_f(x, y):
        return 500.0
    erosion = eq.integrate_erosion_over_triangle(tri, gamma_f, E_f)
    print(f"三角形单元侵蚀率 = {erosion:.3e} 原子/s")

    return eq


if __name__ == "__main__":
    demo_erosion()
