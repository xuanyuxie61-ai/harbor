
import numpy as np
from typing import Tuple, List, Optional



A_CC_NM = 0.142
A_LATTICE_NM = 0.246
T0_INTRALAYER = -2.7
DELTA_DECAY = 0.184 * A_CC_NM
ALPHA_PI = 0.48
W0_INTERLAYER = 0.11
XI_INTERLAYER = A_LATTICE_NM / np.sqrt(3.0)


def graphene_lattice_vectors(a: float = A_LATTICE_NM) -> np.ndarray:
    a1 = np.array([a, 0.0])
    a2 = np.array([a * 0.5, a * np.sqrt(3.0) * 0.5])
    return np.vstack([a1, a2])


def moire_reciprocal_vectors(theta_deg: float) -> np.ndarray:
    theta = float(theta_deg)
    if theta <= 0.0 or theta >= 30.0:
        raise ValueError("Twist angle must be in (0°, 30°).")
    theta_rad = np.deg2rad(theta)
    k_theta = (8.0 * np.pi / (3.0 * A_LATTICE_NM)) * np.sin(theta_rad * 0.5)
    q_vecs = np.zeros((3, 2))
    for j in range(3):
        angle = 2.0 * np.pi * j / 3.0
        q_vecs[j] = k_theta * np.array([np.cos(angle), np.sin(angle)])
    return q_vecs


def moire_lattice_constant(theta_deg: float) -> float:
    theta = float(theta_deg)
    if theta <= 0.0 or theta >= 30.0:
        raise ValueError("Twist angle must be in (0°, 30°).")
    theta_rad = np.deg2rad(theta)
    return A_LATTICE_NM / (2.0 * np.sin(theta_rad * 0.5))


def generate_monolayer_sites(
    n_cells: int = 4, a: float = A_LATTICE_NM
) -> Tuple[np.ndarray, np.ndarray]:
    if n_cells < 1:
        raise ValueError("n_cells must be positive.")
    a1, a2 = graphene_lattice_vectors(a)
    delta = np.array([0.0, a / np.sqrt(3.0)])

    n_atoms = 2 * n_cells * n_cells
    positions = np.zeros((n_atoms, 2))
    sublattice = np.zeros(n_atoms, dtype=int)

    idx = 0
    for n1 in range(n_cells):
        for n2 in range(n_cells):
            base = n1 * a1 + n2 * a2
            positions[idx] = base
            sublattice[idx] = 0
            idx += 1
            positions[idx] = base + delta
            sublattice[idx] = 1
            idx += 1


    cell_x = n_cells * a1[0] + n_cells * a2[0]
    cell_y = n_cells * a1[1] + n_cells * a2[1]


    cell_matrix = n_cells * a1 + n_cells * a2

    sc_a1 = n_cells * a1
    sc_a2 = n_cells * a2

    for i in range(idx):
        r = positions[i]

        det = sc_a1[0] * sc_a2[1] - sc_a1[1] * sc_a2[0]
        if abs(det) < 1e-14:
            raise RuntimeError("Degenerate supercell vectors.")
        c1 = (r[0] * sc_a2[1] - r[1] * sc_a2[0]) / det
        c2 = (sc_a1[0] * r[1] - sc_a1[1] * r[0]) / det
        c1 -= np.floor(c1)
        c2 -= np.floor(c2)
        positions[i] = c1 * sc_a1 + c2 * sc_a2

    return positions, sublattice


def intralayer_hopping(r: float, theta: float = 0.0) -> float:




    raise NotImplementedError("Hole 1: implement intralayer hopping Slater-Koster formula")


def interlayer_hopping(
    r_inplane: float, dz: float = 0.335
) -> float:
    if r_inplane < 0.0:
        raise ValueError("In-plane distance cannot be negative.")

    dz_factor = np.exp(-0.1 * (dz - 0.335))
    val = W0_INTERLAYER * np.exp(-(r_inplane ** 2) / (2.0 * XI_INTERLAYER ** 2))
    return float(val * dz_factor)


def build_tight_binding_hamiltonian(
    theta_deg: float,
    n_super: int = 4,
    onsite_energy: Optional[np.ndarray] = None,
    intralayer_cutoff: float = 3.0 * A_CC_NM,
    interlayer_cutoff: float = 2.0 * A_LATTICE_NM,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = float(theta_deg)
    if theta <= 0.0 or theta >= 30.0:
        raise ValueError("Twist angle must be in (0°, 30°).")

    half_theta = np.deg2rad(theta * 0.5)


    pos_layer0, sub_layer0 = generate_monolayer_sites(n_super)
    pos_layer1, sub_layer1 = generate_monolayer_sites(n_super)

    n_atoms_per_layer = pos_layer0.shape[0]
    n_total = 2 * n_atoms_per_layer


    R0 = np.array([
        [np.cos(-half_theta), -np.sin(-half_theta)],
        [np.sin(-half_theta),  np.cos(-half_theta)],
    ])
    R1 = np.array([
        [np.cos(half_theta), -np.sin(half_theta)],
        [np.sin(half_theta),  np.cos(half_theta)],
    ])

    positions = np.zeros((n_total, 3))
    layer_index = np.zeros(n_total, dtype=int)
    sublattice = np.zeros(n_total, dtype=int)

    for i in range(n_atoms_per_layer):
        positions[i, :2] = R0 @ pos_layer0[i]
        positions[i, 2] = -0.1675
        layer_index[i] = 0
        sublattice[i] = sub_layer0[i]

        j = i + n_atoms_per_layer
        positions[j, :2] = R1 @ pos_layer1[i]
        positions[j, 2] = 0.1675
        layer_index[j] = 1
        sublattice[j] = sub_layer1[i]


    H = np.zeros((n_total, n_total))

    if onsite_energy is None:
        onsite_energy = np.zeros(n_total)
    else:
        onsite_energy = np.asarray(onsite_energy, dtype=float)
        if onsite_energy.size != n_total:
            raise ValueError("Onsite energy array size mismatch.")

    for i in range(n_total):
        H[i, i] = onsite_energy[i]


    for layer in range(2):
        offset = layer * n_atoms_per_layer
        for i in range(n_atoms_per_layer):
            ii = offset + i
            ri = positions[ii, :2]
            for j in range(i + 1, n_atoms_per_layer):
                jj = offset + j
                rj = positions[jj, :2]
                dr = ri - rj


                sc_a1, sc_a2 = graphene_lattice_vectors()
                sc_a1 *= n_super
                sc_a2 *= n_super

                dr = minimum_image_2d(dr, sc_a1, sc_a2)
                dist = np.linalg.norm(dr)
                if dist < intralayer_cutoff and dist > 1e-6:
                    t = intralayer_hopping(dist)
                    H[ii, jj] = t
                    H[jj, ii] = t


    for i in range(n_atoms_per_layer):
        ii = i
        ri = positions[ii, :2]
        for j in range(n_atoms_per_layer):
            jj = n_atoms_per_layer + j
            rj = positions[jj, :2]
            dr = ri - rj

            L_m = moire_lattice_constant(theta)


            for nx in [-1, 0, 1]:
                for ny in [-1, 0, 1]:
                    shift = np.array([nx * L_m, ny * L_m])
                    dr_shifted = dr + shift
                    dist = np.linalg.norm(dr_shifted)
                    if dist < interlayer_cutoff:
                        w = interlayer_hopping(dist)
                        H[ii, jj] += w
                        H[jj, ii] += w


    H = 0.5 * (H + H.T)

    return H, positions, layer_index


def minimum_image_2d(
    dr: np.ndarray, a1: np.ndarray, a2: np.ndarray
) -> np.ndarray:
    dr = np.asarray(dr, dtype=float)
    a1 = np.asarray(a1, dtype=float)
    a2 = np.asarray(a2, dtype=float)

    det = a1[0] * a2[1] - a1[1] * a2[0]
    if abs(det) < 1e-14:
        raise ValueError("Degenerate lattice vectors.")


    b1 = (2.0 * np.pi / det) * np.array([a2[1], -a2[0]])
    b2 = (2.0 * np.pi / det) * np.array([-a1[1], a1[0]])


    c1 = (dr[0] * b1[0] + dr[1] * b1[1]) / (2.0 * np.pi)
    c2 = (dr[0] * b2[0] + dr[1] * b2[1]) / (2.0 * np.pi)

    n1 = int(np.round(c1))
    n2 = int(np.round(c2))

    return dr - n1 * a1 - n2 * a2


def stacking_registry(
    positions: np.ndarray,
    layer_index: np.ndarray,
    theta_deg: float,
) -> np.ndarray:
    q_vecs = moire_reciprocal_vectors(theta_deg)
    N = positions.shape[0]
    registry = np.zeros(N, dtype=int)


    for i in range(N):
        r = positions[i, :2]
        psi = 0.0 + 0.0j
        for q in q_vecs:
            psi += np.exp(1j * np.dot(q, r))
        amplitude = np.abs(psi)
        phase = np.angle(psi)


        if amplitude > 2.5:
            registry[i] = 0
        elif np.cos(phase) > 0:
            registry[i] = 1
        else:
            registry[i] = 2

    return registry


def apply_electric_field(
    H: np.ndarray,
    positions: np.ndarray,
    layer_index: np.ndarray,
    field_strength: float,
) -> np.ndarray:
    H = np.array(H, copy=True)
    N = H.shape[0]
    if positions.shape[0] != N or layer_index.shape[0] != N:
        raise ValueError("Dimension mismatch.")
    e_charge = 1.0
    for i in range(N):
        H[i, i] += e_charge * field_strength * positions[i, 2]
    return H
