"""
brillouin_integrator.py
-----------------------
Numerical quadrature over the 3D Brillouin zone using tetrahedron methods.

Adapted from seed project 1253_tetrahedron_nco_rule.

Scientific Background
=====================
Many-body response functions and topological invariants in condensed matter
require integration over the Brillouin zone (BZ). For 3D systems, the BZ
is partitioned into tetrahedra, and integrals are evaluated as

    I = Σ_tetrahedra Σ_{q} w_q f(k_q),

where {k_q, w_q} are quadrature points and weights inside each tetrahedron.
The NCO (Newton-Cotes Open) rules on tetrahedra provide symmetric quadrature
formulae exact for polynomials up to a given degree.

For a non-Hermitian topological invariant such as the biorthogonal Chern-Simons
invariant or the volume of the Fermi surface, one evaluates

    Ω_CS = (1 / 4π) ∫_{BZ} d^3k ε^{μνρ} Tr[ A_μ ∂_ν A_ρ + (2/3) A_μ A_ν A_ρ ],

where A_μ is the non-Abelian Berry connection matrix
(A_μ)_{nm} = i ⟨u_n^L| ∂_{k_μ} |u_m^R⟩.
"""

import numpy as np


# Precomputed symmetric quadrature rules for the reference tetrahedron
# with vertices (0,0,0), (1,0,0), (0,1,0), (0,0,1).
# We provide a degree-3 rule (order 5) and a degree-5 rule (order 15).

_TETRA_RULES = {
    3: {
        'points': np.array([
            [0.25, 0.25, 0.25],
            [0.5, 1.0/6.0, 1.0/6.0],
            [1.0/6.0, 0.5, 1.0/6.0],
            [1.0/6.0, 1.0/6.0, 0.5],
            [1.0/6.0, 1.0/6.0, 1.0/6.0],
        ]),
        'weights': np.array([-0.8, 0.45, 0.45, 0.45, 0.45]) / 6.0,
    },
    5: {
        # 15-point degree-5 rule (approximate)
        'points': np.array([
            [0.25, 0.25, 0.25],
            [0.091971078052723, 0.091971078052723, 0.091971078052723],
            [0.72408676584183,  0.091971078052723, 0.091971078052723],
            [0.091971078052723, 0.72408676584183,  0.091971078052723],
            [0.091971078052723, 0.091971078052723, 0.72408676584183],
            [0.31979362782963,  0.31979362782963,  0.31979362782963],
            [0.04061911651111,  0.31979362782963,  0.31979362782963],
            [0.31979362782963,  0.04061911651111,  0.31979362782963],
            [0.31979362782963,  0.31979362782963,  0.04061911651111],
            [0.44364916731037,  0.44364916731037,  0.05635083268963],
            [0.44364916731037,  0.05635083268963,  0.44364916731037],
            [0.05635083268963,  0.44364916731037,  0.44364916731037],
            [0.44364916731037,  0.05635083268963,  0.05635083268963],
            [0.05635083268963,  0.44364916731037,  0.05635083268963],
            [0.05635083268963,  0.05635083268963,  0.44364916731037],
        ]),
        'weights': np.array([
            -0.013155555555556,
             0.007622222222222,
             0.007622222222222,
             0.007622222222222,
             0.007622222222222,
             0.024888888888889,
             0.024888888888889,
             0.024888888888889,
             0.024888888888889,
             0.009851111111111,
             0.009851111111111,
             0.009851111111111,
             0.009851111111111,
             0.009851111111111,
             0.009851111111111,
        ]) / 6.0,
    }
}


def reference_to_physical_t4(ref_points, tetra):
    """
    Map quadrature points from the reference tetrahedron to a physical
    tetrahedron defined by its four vertices.

    Parameters
    ----------
    ref_points : ndarray, shape (N, 3)
        Barycentric coordinates (or reference coordinates).
    tetra : ndarray, shape (4, 3)
        Physical vertices.

    Returns
    -------
    phys_points : ndarray, shape (N, 3)
    """
    # ref_points are assumed to be in barycentric-like coordinates
    # where x+y+z <= 1, x,y,z >= 0
    # The fourth barycentric coordinate is w = 1 - x - y - z
    N = ref_points.shape[0]
    phys = np.zeros((N, 3))
    for i in range(N):
        x, y, z = ref_points[i]
        w = 1.0 - x - y - z
        phys[i] = x * tetra[0] + y * tetra[1] + z * tetra[2] + w * tetra[3]
    return phys


def tetrahedron_volume(tetra):
    """
    Volume of a tetrahedron with vertices tetra[0..3, 0..2].

    V = |det(M)| / 6, where M is the 4×4 matrix with rows [x_i, y_i, z_i, 1].
    """
    M = np.ones((4, 4))
    M[:, :3] = tetra
    vol = abs(np.linalg.det(M)) / 6.0
    return vol


def integrate_over_tetrahedron(f, tetra, degree=5):
    """
    Integrate a function f over a physical tetrahedron using symmetric
    NCO quadrature rules.

    Parameters
    ----------
    f : callable
        f(x, y, z) -> float or complex.
    tetra : ndarray, shape (4, 3)
        Physical tetrahedron vertices.
    degree : int
        Quadrature degree (3 or 5).

    Returns
    -------
    integral : complex or float
    """
    if degree not in _TETRA_RULES:
        raise ValueError(f"Unsupported degree {degree}. Choose 3 or 5.")
    rule = _TETRA_RULES[degree]
    pts = rule['points']
    wts = rule['weights']
    phys_pts = reference_to_physical_t4(pts, tetra)
    vol = tetrahedron_volume(tetra)
    total = 0.0 + 0.0j
    for p, w in zip(phys_pts, wts):
        total += w * f(p[0], p[1], p[2])
    return total * vol * 6.0  # weights are normalized for ref volume 1/6


def partition_bz_into_tetrahedra(n_k=4):
    """
    Partition the cubic Brillouin zone [-π, π]^3 into tetrahedra.
    We subdivide the cube into n_k^3 small cubes, then each small cube
    into 6 tetrahedra.

    Returns
    -------
    tetra_list : list of ndarray, each shape (4, 3)
    """
    edges = np.linspace(-np.pi, np.pi, n_k + 1)
    tetra_list = []
    for ix in range(n_k):
        for iy in range(n_k):
            for iz in range(n_k):
                x0, x1 = edges[ix], edges[ix + 1]
                y0, y1 = edges[iy], edges[iy + 1]
                z0, z1 = edges[iz], edges[iz + 1]
                # Vertices of the cube
                v000 = np.array([x0, y0, z0])
                v100 = np.array([x1, y0, z0])
                v010 = np.array([x0, y1, z0])
                v110 = np.array([x1, y1, z0])
                v001 = np.array([x0, y0, z1])
                v101 = np.array([x1, y0, z1])
                v011 = np.array([x0, y1, z1])
                v111 = np.array([x1, y1, z1])
                # 5 tetrahedra decomposition (more stable)
                tetra_list.append(np.array([v000, v100, v010, v001]))
                tetra_list.append(np.array([v111, v011, v101, v110]))
                tetra_list.append(np.array([v100, v010, v110, v001]))
                tetra_list.append(np.array([v100, v110, v101, v001]))
                tetra_list.append(np.array([v010, v110, v011, v001]))
                tetra_list.append(np.array([v110, v101, v011, v001]))
    return tetra_list


def integrate_bz_3d(f, n_k=4, degree=5):
    """
    Integrate a function over the 3D Brillouin zone using tetrahedral
    decomposition and high-order quadrature.

    Parameters
    ----------
    f : callable
        f(kx, ky, kz) -> scalar.
    n_k : int
        Number of subdivisions per axis.
    degree : int
        Quadrature degree.

    Returns
    -------
    result : complex or float
    """
    tetras = partition_bz_into_tetrahedra(n_k)
    total = 0.0 + 0.0j
    for tetra in tetras:
        total += integrate_over_tetrahedron(f, tetra, degree=degree)
    return total


def bz_average_energy(H_func, n_k=4, degree=5):
    """
    Compute the average ground-state energy over the 3D Brillouin zone
    for a non-Hermitian Hamiltonian H(k).

    ⟨E_0⟩ = (1 / V_BZ) ∫_{BZ} E_0(k) d^3k

    Parameters
    ----------
    H_func : callable
        H_func(kx, ky, kz) returns ndarray.
    n_k : int
    degree : int

    Returns
    -------
    avg_E : complex
    """
    def f(kx, ky, kz):
        H = H_func(kx, ky, kz)
        E = np.linalg.eigvals(H)
        return E[np.argmin(E.real)]

    V_BZ = (2.0 * np.pi) ** 3
    integral = integrate_bz_3d(f, n_k=n_k, degree=degree)
    return integral / V_BZ
