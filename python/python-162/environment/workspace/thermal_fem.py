
import numpy as np
from typing import Tuple, Callable
from fem_assembler import assemble_thermal_matrices, apply_dirichlet_bc
from mesh_generator import build_boundary_mask


class ThermalFEMSolver:

    def __init__(self, nodes: np.ndarray, elements: np.ndarray,
                 region_tags: np.ndarray, thermal_conductivity: dict,
                 rho_cp: float = 2.5e6, dt: float = 1.0,
                 T_ambient: float = 298.15):
        self.nodes = nodes
        self.elements = elements
        self.region_tags = region_tags
        self.rho_cp = rho_cp
        self.dt = dt
        self.T_ambient = T_ambient
        self.n_nodes = len(nodes)


        self.K, self.M = assemble_thermal_matrices(
            nodes, elements, thermal_conductivity, region_tags, rho_cp
        )
        self.boundary_mask = build_boundary_mask(nodes, elements)
        self.bc_nodes = np.where(self.boundary_mask)[0]


        self.A_cn = self.M + 0.5 * dt * self.K
        self.B_cn = self.M - 0.5 * dt * self.K

    def step(self, T_n: np.ndarray, Q_gen: np.ndarray,
             bc_values: np.ndarray = None) -> np.ndarray:
        rhs = self.B_cn @ T_n + self.dt * Q_gen
        if bc_values is None:
            bc_values = np.full(len(self.bc_nodes), self.T_ambient)
        A, rhs_bc = apply_dirichlet_bc(self.A_cn, rhs, self.bc_nodes, bc_values)

        T_next = np.linalg.solve(A + 1e-10 * np.eye(self.n_nodes), rhs_bc)

        T_next = np.clip(T_next, self.T_ambient, 400.0)
        return T_next

    def solve_transient(self, T0: np.ndarray, n_steps: int,
                        Q_gen_func: Callable[[int, np.ndarray], np.ndarray],
                        bc_func: Callable[[int], np.ndarray] = None) -> np.ndarray:
        T_hist = np.zeros((n_steps + 1, self.n_nodes), dtype=float)
        T_hist[0] = T0
        for step in range(n_steps):
            Q = Q_gen_func(step, T_hist[step])
            bc_vals = bc_func(step) if bc_func is not None else None
            T_hist[step + 1] = self.step(T_hist[step], Q, bc_vals)
        return T_hist

    def compute_max_temperature(self, T: np.ndarray) -> float:
        return float(np.max(T))

    def compute_temperature_gradient(self, T: np.ndarray) -> np.ndarray:
        grad = np.zeros(self.n_nodes, dtype=float)
        counts = np.zeros(self.n_nodes, dtype=float)
        for tri in self.elements:
            p = self.nodes[tri]

            J = np.array([
                [p[1, 0] - p[0, 0], p[2, 0] - p[0, 0]],
                [p[1, 1] - p[0, 1], p[2, 1] - p[0, 1]]
            ])
            detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
            if abs(detJ) < 1e-14:
                continue
            invJ = np.linalg.inv(J)
            dT = np.array([T[tri[1]] - T[tri[0]], T[tri[2]] - T[tri[0]]])
            dTdx = invJ[0, 0] * dT[0] + invJ[1, 0] * dT[1]
            dTdy = invJ[0, 1] * dT[0] + invJ[1, 1] * dT[1]
            gmag = np.sqrt(dTdx ** 2 + dTdy ** 2)
            for node in tri:
                grad[node] += gmag
                counts[node] += 1.0
        mask = counts > 0
        grad[mask] /= counts[mask]
        return grad


def compute_heat_generation(
    region_tags: np.ndarray,
    current: float,
    overpotential: np.ndarray,
    reaction_flux: np.ndarray,
    specific_area: float = 885000.0,
    faraday: float = 96485.33212,
    T_nodes: np.ndarray = None,
    dUdT: float = -0.0002
) -> np.ndarray:




    n_elem = len(region_tags)
    Q_elem = np.zeros(n_elem, dtype=float)

    return Q_elem
