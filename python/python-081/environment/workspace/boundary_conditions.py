
import numpy as np
from typing import Tuple, List, Optional


def find_nodes_on_plane(nodes: np.ndarray,
                        coord_idx: int,
                        coord_value: float,
                        tol: float = 1e-8) -> np.ndarray:
    return np.where(np.abs(nodes[:, coord_idx] - coord_value) < tol)[0]


def apply_dirichlet_bc(nodes: np.ndarray,
                        fixed_planes: List[Tuple[int, float]],
                        fixed_dofs_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    if fixed_dofs_mask is None:
        fixed_dofs_mask = np.array([True, True, True])

    constrained_nodes = set()
    for coord_idx, coord_value in fixed_planes:
        ids = find_nodes_on_plane(nodes, coord_idx, coord_value)
        constrained_nodes.update(ids.tolist())

    bc_dofs = []
    bc_values = []
    for n in sorted(constrained_nodes):
        for d in range(3):
            if fixed_dofs_mask[d]:
                bc_dofs.append(3 * n + d)
                bc_values.append(0.0)

    return np.array(bc_dofs, dtype=np.int32), np.array(bc_values, dtype=np.float64)


def apply_pressure_load(nodes: np.ndarray,
                        surface_tris: np.ndarray,
                        pressure: float,
                        direction: Optional[np.ndarray] = None) -> np.ndarray:
    n_nodes = nodes.shape[0]
    F_ext = np.zeros(3 * n_nodes, dtype=np.float64)

    for tri in surface_tris:
        p0, p1, p2 = nodes[tri[0]], nodes[tri[1]], nodes[tri[2]]
        v1 = p1 - p0
        v2 = p2 - p0
        cross = np.cross(v1, v2)
        area = 0.5 * np.linalg.norm(cross)
        if area < 1e-14:
            continue
        if direction is not None:
            n_vec = direction / (np.linalg.norm(direction) + 1e-14)
        else:
            n_vec = cross / (np.linalg.norm(cross) + 1e-14)
        force_per_node = -pressure * area / 3.0 * n_vec
        for n in tri:
            F_ext[3 * n:3 * n + 3] += force_per_node

    return F_ext


def apply_body_force(nodes: np.ndarray, elements: np.ndarray,
                     body_force: np.ndarray) -> np.ndarray:
    n_nodes = nodes.shape[0]
    F_ext = np.zeros(3 * n_nodes, dtype=np.float64)

    for e in elements:
        x0, x1, x2, x3 = nodes[e[0]], nodes[e[1]], nodes[e[2]], nodes[e[3]]
        mat = np.vstack([x1 - x0, x2 - x0, x3 - x0])
        vol = abs(np.linalg.det(mat)) / 6.0
        force_total = body_force * vol
        for n in e:
            F_ext[3 * n:3 * n + 3] += force_total / 4.0

    return F_ext


def penalty_contact_force(nodes: np.ndarray,
                          slave_nodes: np.ndarray,
                          master_plane: Tuple[int, float],
                          penalty: float = 1e12,
                          contact_normal: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    n_nodes = nodes.shape[0]
    F_contact = np.zeros(3 * n_nodes, dtype=np.float64)
    active = np.zeros(n_nodes, dtype=np.bool_)

    coord_idx, coord_value = master_plane
    if contact_normal is None:
        n_vec = np.zeros(3, dtype=np.float64)
        n_vec[coord_idx] = 1.0
    else:
        n_vec = contact_normal / (np.linalg.norm(contact_normal) + 1e-14)

    for n in slave_nodes:
        plane_point = np.zeros(3)
        plane_point[coord_idx] = coord_value
        signed_dist = np.dot(nodes[n] - plane_point, n_vec)
        penetration = -signed_dist
        if penetration > 0:
            active[n] = True
            force_mag = penalty * penetration
            F_contact[3 * n:3 * n + 3] += force_mag * n_vec

    return F_contact, active
