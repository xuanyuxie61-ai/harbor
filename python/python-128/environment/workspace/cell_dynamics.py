
import numpy as np
from special_math import jacobi_elliptic





def ellipsoid_surface_area_rudolf(a: float, b: float, c: float, p: float = 1.6075):
    a, b, c = abs(float(a)), abs(float(b)), abs(float(c))

    arr = sorted([a, b, c], reverse=True)
    a, b, c = arr[0], arr[1], arr[2]
    if a < 1e-15 or b < 1e-15 or c < 1e-15:
        raise ValueError("ellipsoid_surface_area_rudolf: 半轴必须为正")
    term = ((a * b) ** p + (a * c) ** p + (b * c) ** p) / 3.0
    return 4.0 * np.pi * (term ** (1.0 / p))


def ellipsoid_surface_area_elliptic(a: float, b: float, c: float):
    a, b, c = abs(float(a)), abs(float(b)), abs(float(c))
    arr = sorted([a, b, c], reverse=True)
    a, b, c = arr[0], arr[1], arr[2]
    if a < 1e-15 or c < 1e-15:
        raise ValueError("ellipsoid_surface_area_elliptic: 半轴必须为正")

    phi = np.arccos(np.clip(c / a, -1.0, 1.0))
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    denom = b * b * (a * a - c * c)
    if abs(denom) < 1e-15:
        m = 1.0
    else:
        m = (a * a * (b * b - c * c)) / denom
        m = np.clip(m, 0.0, 1.0)




    def integrand_E(theta):
        return np.sqrt(1.0 - m * np.sin(theta) ** 2)

    def integrand_F(theta):
        return 1.0 / np.sqrt(1.0 - m * np.sin(theta) ** 2)

    n_quad = 200
    theta = np.linspace(0.0, phi, n_quad)
    dth = phi / (n_quad - 1)
    E_val = np.trapz(integrand_E(theta), theta)
    F_val = np.trapz(integrand_F(theta), theta)

    temp = E_val * sin_phi ** 2 + F_val * cos_phi ** 2
    if abs(sin_phi) < 1e-15:
        temp2 = 1.0
    else:
        temp2 = temp / sin_phi

    return 2.0 * np.pi * (c ** 2 + a * b * temp2)


def ellipsoid_volume(a: float, b: float, c: float):
    a, b, c = abs(float(a)), abs(float(b)), abs(float(c))
    return (4.0 / 3.0) * np.pi * a * b * c





class CellAgent:

    def __init__(self, position, shape=(5.0, 3.0, 2.0), phase=0):
        self.position = np.asarray(position, dtype=float).reshape(3)
        self.shape = tuple(float(s) for s in shape)
        self.phase = int(phase) % 4
        self.velocity = np.zeros(3, dtype=float)
        self.sensitivity = 1.0

    def chemotaxis_velocity(self, grad_c, mu=0.5, gamma=0.3):

        raise NotImplementedError("Hole 3: chemotaxis_velocity 尚未实现")


    def ecm_drag(self, ecm_density_func, beta=0.2):
        rho = ecm_density_func(self.position)
        return -beta * rho

    def stochastic_component(self, sigma=0.05):
        return np.random.normal(0.0, sigma, size=3)

    def step(self, grad_c, dt, ecm_density_func=None, mu=0.5, gamma=0.3,
             beta=0.2, sigma=0.05):
        v = self.chemotaxis_velocity(grad_c, mu, gamma)
        v += self.stochastic_component(sigma)
        if ecm_density_func is not None:
            drag = self.ecm_drag(ecm_density_func, beta)

            v *= np.exp(drag * dt)
        self.velocity = v
        self.position += dt * v
        return self.position


class CellPopulation:

    def __init__(self, n_cells: int = 50, domain=((-1, 1), (-1, 1), (-0.5, 0.5))):
        self.n_cells = int(n_cells)
        self.domain = domain
        self.cells = []
        for _ in range(self.n_cells):
            pos = np.array([
                np.random.uniform(domain[0][0], domain[0][1]),
                np.random.uniform(domain[1][0], domain[1][1]),
                np.random.uniform(domain[2][0], domain[2][1]),
            ])

            shape = tuple(sorted([
                np.random.uniform(3.0, 7.0),
                np.random.uniform(2.0, 5.0),
                np.random.uniform(1.5, 4.0),
            ], reverse=True))
            self.cells.append(CellAgent(pos, shape))

    def compute_mean_position(self):
        if not self.cells:
            return np.zeros(3)
        return np.mean([c.position for c in self.cells], axis=0)

    def compute_spread(self):
        if len(self.cells) < 2:
            return 0.0
        pos = np.array([c.position for c in self.cells])
        return np.mean(np.std(pos, axis=0))

    def sensitivity_analysis(self, grad_c_func, dt, n_steps=10, eps=1e-4):
        if not self.cells:
            return np.array([])
        base_cell = self.cells[0]
        traj_base = []
        traj_pert = []
        pos0 = base_cell.position.copy()
        pos_pert = pos0 + eps * np.ones(3)

        cell_base = CellAgent(pos0, base_cell.shape)
        cell_pert = CellAgent(pos_pert, base_cell.shape)

        for _ in range(n_steps):
            g = grad_c_func(cell_base.position)
            cell_base.step(g, dt)
            traj_base.append(cell_base.position.copy())

            g = grad_c_func(cell_pert.position)
            cell_pert.step(g, dt)
            traj_pert.append(cell_pert.position.copy())

        diff = np.array([
            np.linalg.norm(traj_base[i] - traj_pert[i])
            for i in range(n_steps)
        ])
        return diff

    def step_all(self, grad_c_func, dt, **kwargs):
        for cell in self.cells:
            g = grad_c_func(cell.position)
            cell.step(g, dt, **kwargs)

    def total_surface_area(self):
        total = 0.0
        for cell in self.cells:
            total += ellipsoid_surface_area_rudolf(*cell.shape)
        return total

    def total_volume(self):
        total = 0.0
        for cell in self.cells:
            total += ellipsoid_volume(*cell.shape)
        return total
