
import numpy as np
from typing import Tuple, List
from utils import spherical_volume, safe_divide
from icf_parameters import TP, NP


class RadialMesh:

    def __init__(self, n_cells: int = NP.N_RADIAL):
        self.n_cells = n_cells
        self.n_nodes = n_cells + 1
        self.r = np.zeros(self.n_nodes)
        self._generate_initial_mesh()

    def _generate_initial_mesh(self):
        n = self.n_cells


        s = np.linspace(0.0, 1.0, self.n_nodes)



        alpha = 8.0
        beta = 0.85
        s_mapped = s + alpha * s * (1.0 - s) * (s - beta) / (1.0 + alpha * 0.25)
        s_mapped = np.clip(s_mapped, 0.0, 1.0)
        s_mapped = (s_mapped - s_mapped[0]) / (s_mapped[-1] - s_mapped[0])

        self.r = s_mapped * TP.R_ABLATION


        self._enforce_interface_nodes()

    def _enforce_interface_nodes(self):
        for target in [TP.R_DT_ICE, TP.R_GAS]:
            idx = np.searchsorted(self.r, target)
            if idx > 0 and idx < self.n_nodes:
                if not np.isclose(self.r[idx], target):

                    self.r = np.insert(self.r, idx, target)
                    self.n_nodes += 1
                    self.n_cells += 1

    def cell_volumes(self) -> np.ndarray:
        vol = np.zeros(self.n_cells)
        for i in range(self.n_cells):
            vol[i] = spherical_volume(self.r[i], self.r[i + 1])
        return vol

    def cell_centers(self) -> np.ndarray:
        centers = np.zeros(self.n_cells)
        for i in range(self.n_cells):
            r1, r2 = self.r[i], self.r[i + 1]

            num = r2**4 - r1**4
            den = r2**3 - r1**3
            centers[i] = 0.75 * safe_divide(np.array([num]), np.array([den]))[0]
            if den < 1.0e-30:
                centers[i] = 0.5 * (r1 + r2)
        return centers

    def cell_widths(self) -> np.ndarray:
        return np.diff(self.r)

    def get_material_zone(self, cell_idx: int) -> str:
        rc = self.cell_centers()[cell_idx]
        if rc >= TP.R_DT_ICE:
            return "ablator"
        elif rc >= TP.R_GAS:
            return "dt_ice"
        else:
            return "gas"

    def get_density_by_zone(self, cell_idx: int) -> float:
        zone = self.get_material_zone(cell_idx)
        return {"ablator": TP.RHO_CH, "dt_ice": TP.RHO_DT, "gas": TP.RHO_GAS}[zone]

    def remap_lagrangian(self, mass: np.ndarray):
        new_r = np.zeros_like(self.r)
        new_r[0] = self.r[0]
        for i in range(self.n_cells):
            rho = self.get_density_by_zone(i)
            vol = mass[i] / rho
            new_r[i + 1] = (new_r[i]**3 + 3.0 * vol / (4.0 * np.pi))**(1.0 / 3.0)
        self.r = new_r






def rcm_ordering(adjacency: List[List[int]], n: int) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)


    mask = np.ones(n, dtype=int)
    perm = np.zeros(n, dtype=int)
    num = 0

    for start in range(n):
        if mask[start] == 0:
            continue


        root = _pseudo_peripheral_node(adjacency, n, mask, start)


        level_order = _bfs_level_order(adjacency, n, mask, root)


        level_order.reverse()

        for node in level_order:
            perm[num] = node
            mask[node] = 0
            num += 1

        if num >= n:
            break

    return perm


def _pseudo_peripheral_node(adjacency, n, mask, start):
    root = start
    while True:
        level_nodes, level_ptr = _build_level_structure(adjacency, n, mask, root)

        last_level = level_nodes[level_ptr[-2]:level_ptr[-1]] if len(level_ptr) > 1 else [root]
        if not last_level:
            break
        min_deg_node = min(last_level, key=lambda x: len(adjacency[x]))
        if min_deg_node == root:
            break
        root = min_deg_node
    return root


def _build_level_structure(adjacency, n, mask, root):
    visited = np.zeros(n, dtype=int)
    level_nodes = []
    level_ptr = [0]
    queue = [root]
    visited[root] = 1

    while queue:
        level_ptr.append(len(level_nodes) + len(queue))
        next_queue = []
        for node in queue:
            level_nodes.append(node)
            for nb in adjacency[node]:
                if visited[nb] == 0 and mask[nb] == 1:
                    visited[nb] = 1
                    next_queue.append(nb)
        queue = next_queue

    return np.array(level_nodes, dtype=int), level_ptr


def _bfs_level_order(adjacency, n, mask, root):
    visited = np.zeros(n, dtype=int)
    order = []
    queue = [root]
    visited[root] = 1
    mask[root] = 0

    while queue:

        queue.sort(key=lambda x: len(adjacency[x]))
        next_queue = []
        for node in queue:
            order.append(node)
            for nb in adjacency[node]:
                if visited[nb] == 0 and mask[nb] == 1:
                    visited[nb] = 1
                    mask[nb] = 0
                    next_queue.append(nb)
        queue = next_queue

    return order


def build_1d_fem_adjacency(n_nodes: int) -> List[List[int]]:
    adj = [[] for _ in range(n_nodes)]
    for i in range(n_nodes):
        if i > 0:
            adj[i].append(i - 1)
        if i < n_nodes - 1:
            adj[i].append(i + 1)
    return adj






def fem_stiffness_mass_1d_spherical(r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = len(r) - 1
    n_nodes = len(r)


    K_main = np.zeros(n_nodes)
    K_upper = np.zeros(n_nodes - 1)
    K_lower = np.zeros(n_nodes - 1)

    M_main = np.zeros(n_nodes)
    M_upper = np.zeros(n_nodes - 1)
    M_lower = np.zeros(n_nodes - 1)

    for e in range(n):
        r1, r2 = r[e], r[e + 1]
        h = r2 - r1
        if h <= 1.0e-30:
            continue




        vol_factor = (4.0 * np.pi * h / 3.0) * (r1**2 + r1 * r2 + r2**2)



        r_c = 0.5 * (r1 + r2)
        k_local = (4.0 * np.pi * r_c**2) / h * np.array([[1.0, -1.0], [-1.0, 1.0]])


        m_local = vol_factor / h * np.array([[1.0/3.0, 1.0/6.0], [1.0/6.0, 1.0/3.0]])


        idx = [e, e + 1]
        for i_local in range(2):
            gi = idx[i_local]
            K_main[gi] += k_local[i_local, i_local]
            M_main[gi] += m_local[i_local, i_local]
            if i_local == 0:
                K_upper[e] += k_local[0, 1]
                K_lower[e] += k_local[1, 0]
                M_upper[e] += m_local[0, 1]
                M_lower[e] += m_local[1, 0]


    class Tridiag:
        def __init__(self, lower, main, upper):
            self.lower = lower
            self.main = main
            self.upper = upper
            self.n = len(main)

    return Tridiag(K_lower, K_main, K_upper), Tridiag(M_lower, M_main, M_upper)


def apply_rcm_to_tridiag(K, M, perm: np.ndarray):
    n = K.n
    K_new_main = np.zeros(n)
    K_new_upper = np.zeros(n - 1)
    K_new_lower = np.zeros(n - 1)
    M_new_main = np.zeros(n)
    M_new_upper = np.zeros(n - 1)
    M_new_lower = np.zeros(n - 1)

    inv_perm = np.zeros(n, dtype=int)
    inv_perm[perm] = np.arange(n)

    for i in range(n):
        ii = perm[i]
        K_new_main[i] = K.main[ii]
        M_new_main[i] = M.main[ii]
        if i < n - 1:

            j = i + 1
            jj = perm[j]
            if abs(ii - jj) == 1:
                idx = min(ii, jj)
                K_new_upper[i] = K.upper[idx] if ii < jj else K.lower[idx]
                K_new_lower[i] = K.lower[idx] if ii < jj else K.upper[idx]
                M_new_upper[i] = M.upper[idx] if ii < jj else M.lower[idx]
                M_new_lower[i] = M.lower[idx] if ii < jj else M.upper[idx]

    class Tridiag:
        def __init__(self, lower, main, upper):
            self.lower = lower
            self.main = main
            self.upper = upper
            self.n = len(main)

    return Tridiag(K_new_lower, K_new_main, K_new_upper), Tridiag(M_new_lower, M_new_main, M_new_upper)
