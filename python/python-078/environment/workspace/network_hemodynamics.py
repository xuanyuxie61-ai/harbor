
import numpy as np
from typing import Tuple, List, Dict






def incidence_to_transition(adjacency: np.ndarray) -> np.ndarray:
    A = np.asarray(adjacency, dtype=float)
    n = A.shape[0]
    row_sums = A.sum(axis=1)

    T_row = np.zeros_like(A)
    for i in range(n):
        if row_sums[i] > 0:
            T_row[i, :] = A[i, :] / row_sums[i]
        else:

            T_row[i, i] = 1.0

    T = T_row.T
    return T


def power_rank(T: np.ndarray, max_iter: int = 200,
               tol: float = 1e-10) -> np.ndarray:
    n = T.shape[0]
    x = np.ones(n) / n

    for it in range(max_iter):
        x_new = T @ x
        diff = np.linalg.norm(x_new - x, ord=np.inf)
        x = x_new
        if diff < tol:
            break


    s = x.sum()
    if s > 1e-15:
        x = x / s
    return x


def page_rank_with_damping(adjacency: np.ndarray,
                           damping: float = 0.85,
                           max_iter: int = 200,
                           tol: float = 1e-10) -> np.ndarray:
    T = incidence_to_transition(adjacency)
    n = T.shape[0]
    x = np.ones(n) / n

    for it in range(max_iter):
        x_new = damping * (T @ x) + (1.0 - damping) / n
        diff = np.linalg.norm(x_new - x, ord=np.inf)
        x = x_new
        if diff < tol:
            break

    s = x.sum()
    if s > 1e-15:
        x = x / s
    return x






class ArterialNetwork:
    def __init__(self):
        self.node_names = [
            "Ascending_Aorta", "Aortic_Arch", "Descending_Aorta",
            "Brachiocephalic", "R_Common_Carotid", "R_Subclavian",
            "L_Common_Carotid", "L_Subclavian"
        ]
        self.n_nodes = len(self.node_names)


        self.adjacency = np.zeros((self.n_nodes, self.n_nodes), dtype=int)
        edges = [
            (0, 1),
            (1, 2),
            (1, 3),
            (1, 6),
            (1, 7),
            (3, 4),
            (3, 5),
        ]
        for i, j in edges:
            self.adjacency[i, j] = 1


        self.radii = np.array([
            0.014, 0.012, 0.010,
            0.006, 0.004, 0.004,
            0.004, 0.004
        ])


        self.lengths = np.array([
            0.05, 0.08, 0.30,
            0.04, 0.12, 0.20,
            0.12, 0.20
        ])

    def compute_flow_distribution(self, total_flow: float = 5.0e-5) -> Dict[str, float]:
        pi = page_rank_with_damping(self.adjacency, damping=0.85)


        root_idx = 0
        scale = total_flow / (pi[root_idx] + 1e-15)

        flows = {}
        for i, name in enumerate(self.node_names):
            flows[name] = float(pi[i] * scale)

        return flows

    def compute_wss_from_flow(self, flow_dict: Dict[str, float],
                              blood_viscosity_pa_s: float = 0.0035) -> Dict[str, float]:
        wss = {}
        for i, name in enumerate(self.node_names):
            Q = flow_dict.get(name, 0.0)
            R = self.radii[i]
            if R > 1e-6:
                tau_w = 4.0 * blood_viscosity_pa_s * Q / (np.pi * R ** 3)
            else:
                tau_w = 0.0
            wss[name] = float(tau_w)
        return wss

    def network_resistance(self, blood_viscosity_pa_s: float = 0.0035) -> Dict[str, float]:
        resistances = {}
        for i, name in enumerate(self.node_names):
            mu = blood_viscosity_pa_s
            L = self.lengths[i]
            R = self.radii[i]
            if R > 1e-6:
                resistances[name] = float(8.0 * mu * L / (np.pi * R ** 4))
            else:
                resistances[name] = float('inf')
        return resistances

    def womersley_numbers(self, heart_rate_bpm: float = 72.0,
                          kinematic_viscosity: float = 3.3e-6) -> Dict[str, float]:
        f = heart_rate_bpm / 60.0
        omega = 2.0 * np.pi * f
        alpha_dict = {}
        for i, name in enumerate(self.node_names):
            R = self.radii[i]
            alpha = R * np.sqrt(omega / kinematic_viscosity)
            alpha_dict[name] = float(alpha)
        return alpha_dict


def bifurcation_flow_split(r_parent: float, r_child1: float,
                           r_child2: float) -> Tuple[float, float]:
    r1_cubed = r_child1 ** 3
    r2_cubed = r_child2 ** 3
    total = r1_cubed + r2_cubed + 1e-15
    q1_ratio = r1_cubed / total
    q2_ratio = r2_cubed / total
    return q1_ratio, q2_ratio
