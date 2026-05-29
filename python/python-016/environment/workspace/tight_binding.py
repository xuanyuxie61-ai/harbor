"""
Tight-Binding Hamiltonian for Twisted Bilayer Graphene (TBG)
=================================================================
Constructs the real-space tight-binding Hamiltonian for twisted bilayer
graphene heterostructures, incorporating intralayer nearest-neighbor (NN)
and next-nearest-neighbor (NNN) hopping, as well as interlayer coupling
that depends on the local stacking registry (Bistritzer-MacDonald model).

Scientific Background
---------------------
The low-energy electronic structure of TBG near the magic angle
θ_m ≈ 1.05° is described by a continuum model or a real-space
atomistic tight-binding model.  The moiré superlattice period is

    L_M = a / (2 sin(θ/2))

where a = 0.246 nm is the graphene lattice constant.

The Hamiltonian in second-quantized form reads

    Ĥ = Σ_n ε_n c_n^† c_n
        + Σ_{⟨n,m⟩} t_{nm} c_n^† c_m
        + Σ_{n∈A, m∈B} w(r_n - r_m) c_n^† c_m  +  h.c.

Intralayer hopping (Slater-Koster-like form):

    t(r) = t₀ exp[−(r − a_CC)/δ₀] · [cos²(φ_{r,π}) + α sin²(φ_{r,π})]

with t₀ ≈ −2.7 eV, a_CC = a/√3 ≈ 0.142 nm, δ₀ ≈ 0.184 a_CC,
and α = 0.48 parametrizing the angular dependence of the p_z−p_z overlap.

Interlayer coupling (Bistritzer-MacDonald):

    w(r) = w₀ exp(−|r|² / (2ξ²))

with w₀ ≈ 0.11 eV and ξ ≈ a/√3 the decay length.

For the moiré pattern we distinguish three high-symmetry stacking
regions: AA (perfectly aligned), AB (one sublattice on top, the other
farside), and BA (the opposite of AB).  The interlayer hopping amplitude
varies periodically across the moiré unit cell as

    w(r) = w₀ Σ_{j=1}^{3} exp(−i q_j·r)

where q_j are the three moiré reciprocal lattice vectors.
"""

import numpy as np
from typing import Tuple, List, Optional


# Physical constants (in eV, nm units)
A_CC_NM = 0.142  # carbon-carbon distance in nm
A_LATTICE_NM = 0.246  # graphene lattice constant in nm
T0_INTRALAYER = -2.7  # eV
DELTA_DECAY = 0.184 * A_CC_NM  # decay length for intralayer hopping
ALPHA_PI = 0.48  # angular anisotropy parameter
W0_INTERLAYER = 0.11  # eV, interlayer coupling strength
XI_INTERLAYER = A_LATTICE_NM / np.sqrt(3.0)  # decay length for interlayer


def graphene_lattice_vectors(a: float = A_LATTICE_NM) -> np.ndarray:
    """
    Return the real-space lattice vectors of monolayer graphene.

    a1 = a * [1, 0]
    a2 = a * [1/2, sqrt(3)/2]

    Parameters
    ----------
    a : float
        Lattice constant in nm.

    Returns
    -------
    np.ndarray of shape (2, 2)
    """
    a1 = np.array([a, 0.0])
    a2 = np.array([a * 0.5, a * np.sqrt(3.0) * 0.5])
    return np.vstack([a1, a2])


def moire_reciprocal_vectors(theta_deg: float) -> np.ndarray:
    """
    Compute the three moiré reciprocal lattice vectors for twisted bilayer
    graphene with twist angle θ (degrees).

    The moiré wave-vectors are

        q_j = k_θ * [cos(2π j/3), sin(2π j/3)]

    with magnitude

        |q| = (8π / (3a)) · sin(θ/2)

    Parameters
    ----------
    theta_deg : float
        Twist angle in degrees.  Must be positive and not too large
        (typically < 10° for the moiré expansion to be valid).

    Returns
    -------
    np.ndarray of shape (3, 2)
        The three moiré reciprocal vectors.

    Raises
    ------
    ValueError
        If θ ≤ 0 or θ ≥ 30°.
    """
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
    """
    Moiré superlattice constant L_M.

        L_M = a / (2 sin(θ/2))

    Parameters
    ----------
    theta_deg : float
        Twist angle in degrees.

    Returns
    -------
    float
        Moiré period in nm.
    """
    theta = float(theta_deg)
    if theta <= 0.0 or theta >= 30.0:
        raise ValueError("Twist angle must be in (0°, 30°).")
    theta_rad = np.deg2rad(theta)
    return A_LATTICE_NM / (2.0 * np.sin(theta_rad * 0.5))


def generate_monolayer_sites(
    n_cells: int = 4, a: float = A_LATTICE_NM
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate the atomic positions of a single graphene layer with periodic
    boundary conditions over an n_cells × n_cells supercell.

    Each unit cell contains two atoms (A and B sublattices):

        r_A = n1 a1 + n2 a2
        r_B = n1 a1 + n2 a2 + δ

    with δ = [0, a/√3]  (the intracell basis vector).

    Parameters
    ----------
    n_cells : int
        Number of unit cells along each lattice direction.
    a : float
        Lattice constant.

    Returns
    -------
    positions : np.ndarray of shape (2 * n_cells^2, 2)
        (x, y) coordinates of all atoms.
    sublattice : np.ndarray of shape (2 * n_cells^2,)
        0 for A sublattice, 1 for B sublattice.
    """
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

    # Enforce periodic boundaries: wrap coordinates into the supercell
    cell_x = n_cells * a1[0] + n_cells * a2[0]
    cell_y = n_cells * a1[1] + n_cells * a2[1]
    # For the specific graphene lattice vectors, cell_x = n_cells * a * 1.5,
    # cell_y = n_cells * a * sqrt(3)/2
    cell_matrix = n_cells * a1 + n_cells * a2
    # More robust: use the actual supercell vectors
    sc_a1 = n_cells * a1
    sc_a2 = n_cells * a2
    # Project onto supercell basis and wrap
    for i in range(idx):
        r = positions[i]
        # Solve r = c1 * sc_a1 + c2 * sc_a2
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
    """
    Intralayer hopping amplitude as a function of distance r (nm) and
    relative angle θ between the p_z orbitals.

    The model form is

        t(r,θ) = t₀ exp[−(r − a_CC)/δ₀] · [cos²θ + α sin²θ]

    with a smooth cutoff at large r.

    Parameters
    ----------
    r : float
        Interatomic distance in nm.
    theta : float
        Angle between p_z orbitals in radians (default 0 for in-plane
        nearest neighbors).

    Returns
    -------
    float
        Hopping amplitude in eV.
    """
    # TODO: Hole 1 - implement intralayer hopping amplitude
    # Scientific formula: Slater-Koster form
    #   t(r,θ) = t₀ exp[−(r − a_CC)/δ₀] · [cos²θ + α sin²θ]
    # with smooth cutoff at large r.
    raise NotImplementedError("Hole 1: implement intralayer hopping Slater-Koster formula")


def interlayer_hopping(
    r_inplane: float, dz: float = 0.335
) -> float:
    """
    Interlayer hopping amplitude for two atoms separated by in-plane
    distance r_inplane and out-of-plane distance dz (nm).

    The model is a 2D Gaussian attenuation:

        w(r_inplane) = w₀ exp(−r_inplane² / (2 ξ²))

    Parameters
    ----------
    r_inplane : float
        In-plane separation in nm.
    dz : float
        Interlayer distance in nm (default 0.335 nm for graphite).

    Returns
    -------
    float
        Interlayer hopping in eV.
    """
    if r_inplane < 0.0:
        raise ValueError("In-plane distance cannot be negative.")
    # Add a small attenuation with dz (not used in simplest model)
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
    """
    Build the full tight-binding Hamiltonian matrix for twisted bilayer
    graphene.

    The system consists of two layers rotated by +θ/2 and −θ/2.
    The moiré unit cell is approximated by an n_super × n_super
    periodic supercell.  The total Hamiltonian is a real symmetric
    (or complex Hermitian if magnetic fields are included) matrix of
    size 4 × n_super².

    Parameters
    ----------
    theta_deg : float
        Twist angle in degrees.
    n_super : int
        Supercell size (number of unit cells along each direction).
    onsite_energy : np.ndarray, optional
        Onsite energies for each atom.  If None, all set to 0.
    intralayer_cutoff : float
        Cutoff distance for intralayer hopping in nm.
    interlayer_cutoff : float
        Cutoff distance for interlayer hopping in nm.

    Returns
    -------
    H : np.ndarray of shape (N, N)
        Hamiltonian matrix (N = 4 * n_super²).
    positions : np.ndarray of shape (N, 3)
        Atomic positions (x, y, z).
    layer_index : np.ndarray of shape (N,)
        0 for bottom layer, 1 for top layer.
    """
    theta = float(theta_deg)
    if theta <= 0.0 or theta >= 30.0:
        raise ValueError("Twist angle must be in (0°, 30°).")

    half_theta = np.deg2rad(theta * 0.5)

    # Generate monolayer sites
    pos_layer0, sub_layer0 = generate_monolayer_sites(n_super)
    pos_layer1, sub_layer1 = generate_monolayer_sites(n_super)

    n_atoms_per_layer = pos_layer0.shape[0]
    n_total = 2 * n_atoms_per_layer

    # Rotate layer 1 by +θ/2, layer 0 by −θ/2 around origin
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
        positions[i, 2] = -0.1675  # half interlayer spacing
        layer_index[i] = 0
        sublattice[i] = sub_layer0[i]

        j = i + n_atoms_per_layer
        positions[j, :2] = R1 @ pos_layer1[i]
        positions[j, 2] = 0.1675
        layer_index[j] = 1
        sublattice[j] = sub_layer1[i]

    # Build Hamiltonian
    H = np.zeros((n_total, n_total))

    if onsite_energy is None:
        onsite_energy = np.zeros(n_total)
    else:
        onsite_energy = np.asarray(onsite_energy, dtype=float)
        if onsite_energy.size != n_total:
            raise ValueError("Onsite energy array size mismatch.")

    for i in range(n_total):
        H[i, i] = onsite_energy[i]

    # Intralayer hopping
    for layer in range(2):
        offset = layer * n_atoms_per_layer
        for i in range(n_atoms_per_layer):
            ii = offset + i
            ri = positions[ii, :2]
            for j in range(i + 1, n_atoms_per_layer):
                jj = offset + j
                rj = positions[jj, :2]
                dr = ri - rj
                # Apply minimum image convention for periodic supercell
                # Supercell size in real space
                sc_a1, sc_a2 = graphene_lattice_vectors()
                sc_a1 *= n_super
                sc_a2 *= n_super
                # Minimum image in 2D
                dr = minimum_image_2d(dr, sc_a1, sc_a2)
                dist = np.linalg.norm(dr)
                if dist < intralayer_cutoff and dist > 1e-6:
                    t = intralayer_hopping(dist)
                    H[ii, jj] = t
                    H[jj, ii] = t

    # Interlayer hopping
    for i in range(n_atoms_per_layer):
        ii = i  # layer 0
        ri = positions[ii, :2]
        for j in range(n_atoms_per_layer):
            jj = n_atoms_per_layer + j  # layer 1
            rj = positions[jj, :2]
            dr = ri - rj
            # Moiré cell minimum image: use the moiré lattice constant
            L_m = moire_lattice_constant(theta)
            # For minimum image in moiré cell, we use a square approximation
            # with side L_m (sufficient for nearest-moiré-cell images)
            for nx in [-1, 0, 1]:
                for ny in [-1, 0, 1]:
                    shift = np.array([nx * L_m, ny * L_m])
                    dr_shifted = dr + shift
                    dist = np.linalg.norm(dr_shifted)
                    if dist < interlayer_cutoff:
                        w = interlayer_hopping(dist)
                        H[ii, jj] += w
                        H[jj, ii] += w

    # Symmetrize to avoid any tiny numerical asymmetry
    H = 0.5 * (H + H.T)

    return H, positions, layer_index


def minimum_image_2d(
    dr: np.ndarray, a1: np.ndarray, a2: np.ndarray
) -> np.ndarray:
    """
    Apply the minimum image convention for a 2D periodic supercell
    defined by lattice vectors a1 and a2.

    The algorithm projects dr onto the reciprocal lattice, rounds to
    the nearest integer multiple of the direct lattice, and subtracts:

        dr_min = dr − n1 a1 − n2 a2

    with n1, n2 chosen to minimize |dr_min|.

    Parameters
    ----------
    dr : np.ndarray of shape (2,)
        Displacement vector.
    a1, a2 : np.ndarray of shape (2,)
        Direct lattice vectors.

    Returns
    -------
    np.ndarray of shape (2,)
        Minimum-image displacement.
    """
    dr = np.asarray(dr, dtype=float)
    a1 = np.asarray(a1, dtype=float)
    a2 = np.asarray(a2, dtype=float)

    det = a1[0] * a2[1] - a1[1] * a2[0]
    if abs(det) < 1e-14:
        raise ValueError("Degenerate lattice vectors.")

    # Reciprocal lattice vectors (not normalized)
    b1 = (2.0 * np.pi / det) * np.array([a2[1], -a2[0]])
    b2 = (2.0 * np.pi / det) * np.array([-a1[1], a1[0]])

    # Coefficients in direct lattice
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
    """
    Classify each atom into one of three stacking regions (AA, AB, BA)
    based on its in-plane position within the moiré unit cell.

    The moiré pattern produces a spatially varying stacking parameter

        ψ(r) = Σ_{j=1}^{3} exp(i q_j · r)

    with |ψ| maximal at AA, intermediate at AB/BA, and minimal at the
    saddle points.

    Parameters
    ----------
    positions : np.ndarray of shape (N, 3)
    layer_index : np.ndarray of shape (N,)
    theta_deg : float

    Returns
    -------
    registry : np.ndarray of shape (N,)
        0 = AA-like, 1 = AB-like, 2 = BA-like.
    """
    q_vecs = moire_reciprocal_vectors(theta_deg)
    N = positions.shape[0]
    registry = np.zeros(N, dtype=int)

    # Use only layer-0 atoms for registry (they sit near the bottom)
    for i in range(N):
        r = positions[i, :2]
        psi = 0.0 + 0.0j
        for q in q_vecs:
            psi += np.exp(1j * np.dot(q, r))
        amplitude = np.abs(psi)
        phase = np.angle(psi)

        # Heuristic thresholds based on |ψ|
        if amplitude > 2.5:
            registry[i] = 0  # AA
        elif np.cos(phase) > 0:
            registry[i] = 1  # AB
        else:
            registry[i] = 2  # BA

    return registry


def apply_electric_field(
    H: np.ndarray,
    positions: np.ndarray,
    layer_index: np.ndarray,
    field_strength: float,
) -> np.ndarray:
    """
    Apply a perpendicular electric field E_z (in V/nm) by adding a
    Stark shift to the onsite energies:

        Δε_i = e · E_z · z_i

    where z_i is the out-of-plane coordinate of atom i.

    Parameters
    ----------
    H : np.ndarray
        Hamiltonian matrix (modified in place).
    positions : np.ndarray
        Atomic positions.
    layer_index : np.ndarray
        Layer indices.
    field_strength : float
        Electric field in V/nm.

    Returns
    -------
    np.ndarray
        Modified Hamiltonian.
    """
    H = np.array(H, copy=True)
    N = H.shape[0]
    if positions.shape[0] != N or layer_index.shape[0] != N:
        raise ValueError("Dimension mismatch.")
    e_charge = 1.0  # in eV·nm/V units for convenience (e=1 in natural units)
    for i in range(N):
        H[i, i] += e_charge * field_strength * positions[i, 2]
    return H
