# -*- coding: utf-8 -*-

import numpy as np


E_CHARGE = 1.43996448



def triangle01_area():
    return 0.5


def triangle01_monomial_integral(e):
    e = np.asarray(e)
    if np.any(e < 0):
        raise ValueError("指数必须非负")
    k = 0
    integral = 1.0
    for i in range(2):
        for j in range(1, e[i] + 1):
            k += 1
            integral = integral * j / k
    for _ in range(2):
        k += 1
        integral = integral / k
    return integral


def triangle01_sample(n):
    u = np.random.rand(n)
    v = np.random.rand(n)
    mask = u + v > 1.0
    u[mask] = 1.0 - u[mask]
    v[mask] = 1.0 - v[mask]
    return np.column_stack((u, v))



def wedge01_volume():
    return 1.0


def wedge01_monomial_integral(e):
    e = np.asarray(e)
    if np.any(e[:2] < 0):
        raise ValueError("x,y指数必须非负")
    if e[2] == -1:
        raise ValueError("e[3] = -1非法")

    value = 1.0
    k = e[0]
    for i in range(1, e[1] + 1):
        k += 1
        value = value * i / k
    k += 1
    value = value / k
    k += 1
    value = value / k

    if e[2] % 2 == 1:
        value = 0.0
    else:
        value = value * 2.0 / (e[2] + 1)
    return value



def tetrahedron_unit_volume():
    return 1.0 / 6.0


def tetrahedron_unit_monomial(expon):
    expon = np.asarray(expon)
    if np.any(expon < 0):
        raise ValueError("指数必须非负")

    value = 1.0
    k = expon[0]
    for i in range(1, expon[1] + 1):
        k += 1
        value = value * i / k
    for i in range(1, expon[2] + 1):
        k += 1
        value = value * i / k
    for _ in range(3):
        k += 1
        value = value / k
    return value



def hexagon01_area():
    return 2.0 * np.sqrt(3.0)


def hexagon_stroud_rule1():
    n = 1
    p = 1
    x = np.array([0.0])
    y = np.array([0.0])
    w = np.array([1.0])
    return n, p, x, y, w


def hexagon_stroud_rule2():
    n = 6
    p = 3
    r = np.sqrt(2.0 / 3.0)
    theta = np.linspace(0, 2 * np.pi, 7)[:-1]
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    w = np.ones(n) / n
    return n, p, x, y, w


def hexagon_stroud_rule3():
    n = 7
    p = 5
    x = np.zeros(n)
    y = np.zeros(n)
    w = np.zeros(n)

    x[0] = 0.0
    y[0] = 0.0
    w[0] = 1.0 / 4.0

    r = np.sqrt(6.0 / 7.0)
    theta = np.linspace(0, 2 * np.pi, 7)[:-1]
    x[1:] = r * np.cos(theta)
    y[1:] = r * np.sin(theta)
    w[1:] = 1.0 / 8.0
    return n, p, x, y, w


def hexagon_stroud_rule4():
    n = 12
    p = 7
    x = np.zeros(n)
    y = np.zeros(n)
    w = np.zeros(n)

    r1 = np.sqrt((6.0 - np.sqrt(6.0)) / 10.0)
    r2 = np.sqrt((6.0 + np.sqrt(6.0)) / 10.0)
    theta = np.linspace(0, 2 * np.pi, 7)[:-1]
    x[:6] = r1 * np.cos(theta)
    y[:6] = r1 * np.sin(theta)
    w[:6] = (3.0 + 2.0 * np.sqrt(6.0)) / 72.0
    x[6:] = r2 * np.cos(theta)
    y[6:] = r2 * np.sin(theta)
    w[6:] = (3.0 - 2.0 * np.sqrt(6.0)) / 72.0
    return n, p, x, y, w


def hexagon_integral(func, rule=3):
    rules = {
        1: hexagon_stroud_rule1,
        2: hexagon_stroud_rule2,
        3: hexagon_stroud_rule3,
        4: hexagon_stroud_rule4,
    }
    if rule not in rules:
        raise ValueError("rule必须在1-4之间")
    n, p, x, y, w = rules[rule]()
    area = hexagon01_area()
    result = 0.0
    for i in range(n):
        result += w[i] * func(x[i], y[i])
    return result * area



def quadrilateral_bilinear_interpolate(x, y, z_values, xi, yi):


    x = np.asarray(x)
    y = np.asarray(y)
    z_values = np.asarray(z_values)

    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)

    if xmax - xmin < 1e-15 or ymax - ymin < 1e-15:
        return np.mean(z_values)

    s = 2.0 * (xi - xmin) / (xmax - xmin) - 1.0
    t = 2.0 * (yi - ymin) / (ymax - ymin) - 1.0


    N1 = 0.25 * (1.0 - s) * (1.0 - t)
    N2 = 0.25 * (1.0 + s) * (1.0 - t)
    N3 = 0.25 * (1.0 + s) * (1.0 + t)
    N4 = 0.25 * (1.0 - s) * (1.0 + t)

    zi = N1 * z_values[0] + N2 * z_values[1] + N3 * z_values[2] + N4 * z_values[3]
    return zi



class PastaPhase:

    PHASE_NAMES = {
        1: 'gnocchi',
        2: 'spaghetti',
        3: 'lasagna',
        4: 'anti-spaghetti',
        5: 'anti-gnocchi'
    }

    def __init__(self, phase_id, density, proton_fraction, u=None):
        if phase_id not in self.PHASE_NAMES:
            raise ValueError(f"phase_id必须在1-5之间, 得到{phase_id}")
        if density <= 0.0:
            raise ValueError("密度必须大于0")
        if proton_fraction < 0.0 or proton_fraction > 1.0:
            raise ValueError("质子分数必须在[0,1]之间")

        self.phase_id = phase_id
        self.density = density
        self.proton_fraction = proton_fraction
        self.rho_n = density * (1.0 - proton_fraction)
        self.rho_p = density * proton_fraction


        self.V_WS = 1.0 / density
        self.a_WS = self.V_WS ** (1.0 / 3.0)


        if u is None:
            self.u = self._optimal_filling()
        else:
            if u <= 0.0 or u >= 1.0:
                raise ValueError("填充率必须在(0,1)之间")
            self.u = u

        self._compute_geometry()

    def _optimal_filling(self):

        return 0.3 + 0.1 * self.proton_fraction

    def _compute_geometry(self):
        raise NotImplementedError

    def surface_area(self):
        raise NotImplementedError

    def volume(self):
        raise NotImplementedError

    def coulomb_factor(self):
        raise NotImplementedError

    def surface_to_volume(self):
        return self.surface_area() / self.volume()


class GnocchiPhase(PastaPhase):

    def __init__(self, density, proton_fraction, u=None):
        super().__init__(1, density, proton_fraction, u)

    def _compute_geometry(self):
        self.R = (3.0 * self.u * self.V_WS / (4.0 * np.pi)) ** (1.0 / 3.0)

    def surface_area(self):
        return 4.0 * np.pi * self.R**2

    def volume(self):
        return (4.0 / 3.0) * np.pi * self.R**3

    def coulomb_factor(self):
        return (3.0 / 5.0) * self.u ** (2.0 / 3.0)


class SpaghettiPhase(PastaPhase):

    def __init__(self, density, proton_fraction, u=None):
        super().__init__(2, density, proton_fraction, u)

    def _compute_geometry(self):

        self.L = self.a_WS
        self.R = np.sqrt(self.u * self.V_WS / (np.pi * self.L))

    def surface_area(self):
        return 2.0 * np.pi * self.R * self.L

    def volume(self):
        return np.pi * self.R**2 * self.L

    def coulomb_factor(self):
        return 0.5 * self.u


class LasagnaPhase(PastaPhase):

    def __init__(self, density, proton_fraction, u=None):
        super().__init__(3, density, proton_fraction, u)

    def _compute_geometry(self):
        self.t = self.u * self.a_WS
        self.A_slice = self.a_WS**2

    def surface_area(self):
        return 2.0 * self.A_slice

    def volume(self):
        return self.t * self.A_slice

    def coulomb_factor(self):
        return self.u ** (2.0 / 3.0)


class AntiSpaghettiPhase(PastaPhase):

    def __init__(self, density, proton_fraction, u=None):
        super().__init__(4, density, proton_fraction, u)

    def _compute_geometry(self):
        self.L = self.a_WS

        self.R = np.sqrt((1.0 - self.u) * self.V_WS / (np.pi * self.L))

    def surface_area(self):
        return 2.0 * np.pi * self.R * self.L

    def volume(self):

        return np.pi * self.R**2 * self.L

    def coulomb_factor(self):

        return 0.5 * (1.0 - self.u)


class AntiGnocchiPhase(PastaPhase):

    def __init__(self, density, proton_fraction, u=None):
        super().__init__(5, density, proton_fraction, u)

    def _compute_geometry(self):
        self.R = (3.0 * (1.0 - self.u) * self.V_WS / (4.0 * np.pi)) ** (1.0 / 3.0)

    def surface_area(self):
        return 4.0 * np.pi * self.R**2

    def volume(self):
        return (4.0 / 3.0) * np.pi * self.R**3

    def coulomb_factor(self):
        return (3.0 / 5.0) * (1.0 - self.u) ** (2.0 / 3.0)


def create_pasta_phase(phase_id, density, proton_fraction, u=None):
    constructors = {
        1: GnocchiPhase,
        2: SpaghettiPhase,
        3: LasagnaPhase,
        4: AntiSpaghettiPhase,
        5: AntiGnocchiPhase,
    }
    return constructors[phase_id](density, proton_fraction, u)


def pasta_energy_landscape(density_range, proton_fraction, n_points=20):
    results = {}
    u_grid = np.linspace(0.05, 0.95, n_points)

    for pid in range(1, 6):
        name = PastaPhase.PHASE_NAMES[pid]
        energies = []
        for rho in density_range:
            e_vals = []
            for u in u_grid:
                try:
                    phase = create_pasta_phase(pid, rho, proton_fraction, u)

                    sigma = 1.0
                    e_surf = sigma * phase.surface_to_volume()
                    e_coul = 0.5 * E_CHARGE * phase.rho_p**2 * phase.coulomb_factor() * phase.volume()
                    e_total = e_surf + e_coul
                    e_vals.append(e_total)
                except ValueError:
                    e_vals.append(np.inf)
            if len(e_vals) > 0:
                energies.append(np.min(e_vals))
            else:
                energies.append(np.inf)
        results[name] = np.array(energies)

    return results


if __name__ == '__main__':

    rho = 0.08
    x_p = 0.3
    for pid in range(1, 6):
        p = create_pasta_phase(pid, rho, x_p)
        print(f"{p.PHASE_NAMES[pid]}: R/t={getattr(p, 'R', getattr(p, 't', 'N/A')):.3f} fm, "
              f"S/V={p.surface_to_volume():.3f} fm^-1")
