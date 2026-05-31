"""
mesh_discretization.py
----------------------
Mesh handling and finite-element discretization for non-Hermitian
spatial operators, adapted from GMSH I/O utilities.

Adapted from seed project 474_gmsh_io.

Scientific Background
=====================
For continuous non-Hermitian Schrödinger operators in 2D or 3D,

    H = - (ℏ^2 / 2m) ∇^2 + V(r) + i W(r),

where V(r) is the real potential and W(r) is the imaginary (gain/loss)
potential, we discretize space using a finite-element mesh. The weak
form leads to a generalized eigenvalue problem

    (K + M_V + i M_W) ψ = E M_mass ψ,

where K is the stiffness matrix (Laplacian), M_V and M_W are potential
mass matrices, and M_mass is the standard mass matrix.

The mesh reader supports the GMSH ASCII format 2.2, parsing $Nodes and
$Elements sections. Elements are triangles (2D) or tetrahedra (3D).
"""

import numpy as np


class SimpleMesh:
    """
    Minimal finite-element mesh container.
    """
    def __init__(self, nodes, elements, element_type='triangle'):
        """
        Parameters
        ----------
        nodes : ndarray, shape (N_nodes, dim)
        elements : ndarray, shape (N_elements, n_vertices)
            Zero-based node indices.
        element_type : str
            'triangle' or 'tetrahedron'.
        """
        self.nodes = np.asarray(nodes, dtype=float)
        self.elements = np.asarray(elements, dtype=int)
        self.element_type = element_type
        self.dim = self.nodes.shape[1]
        if self.dim not in (2, 3):
            raise ValueError("Only 2D and 3D meshes supported.")

    def bounding_box(self):
        """
        Return the axis-aligned bounding box as (min_coords, max_coords).
        """
        return self.nodes.min(axis=0), self.nodes.max(axis=0)

    def element_centroid(self, elem_idx):
        """
        Compute the centroid of a single element.
        """
        verts = self.nodes[self.elements[elem_idx]]
        return verts.mean(axis=0)

    def element_area_or_volume(self, elem_idx):
        """
        Compute area (2D triangle) or volume (3D tetrahedron).
        """
        verts = self.nodes[self.elements[elem_idx]]
        if self.element_type == 'triangle' and self.dim == 2:
            # Shoelace formula
            x, y = verts[:, 0], verts[:, 1]
            return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        elif self.element_type == 'tetrahedron' and self.dim == 3:
            M = np.ones((4, 4))
            M[:, :3] = verts
            return abs(np.linalg.det(M)) / 6.0
        else:
            raise NotImplementedError("Only 2D triangles and 3D tetrahedra.")


def read_gmsh_ascii(filepath):
    """
    Read a minimal GMSH ASCII mesh file (format 2.2).
    Parses $Nodes and $Elements, returns a SimpleMesh.

    Parameters
    ----------
    filepath : str

    Returns
    -------
    mesh : SimpleMesh
    """
    nodes = []
    elements = []
    reading_nodes = False
    reading_elements = False
    node_count = 0
    elem_count = 0
    node_idx = 0
    elem_idx = 0

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('$Nodes'):
                reading_nodes = True
                continue
            if line.startswith('$EndNodes'):
                reading_nodes = False
                continue
            if line.startswith('$Elements'):
                reading_elements = True
                continue
            if line.startswith('$EndElements'):
                reading_elements = False
                continue

            if reading_nodes and node_count == 0:
                node_count = int(line)
                continue
            if reading_elements and elem_count == 0:
                elem_count = int(line)
                continue

            if reading_nodes and node_idx < node_count:
                parts = line.split()
                # index x y z
                if len(parts) >= 4:
                    nodes.append([float(parts[1]), float(parts[2]), float(parts[3])])
                node_idx += 1
                continue

            if reading_elements and elem_idx < elem_count:
                parts = line.split()
                # elm-number elm-type number-of-tags tag ... node-number-list
                if len(parts) >= 3:
                    elem_type = int(parts[1])
                    num_tags = int(parts[2])
                    if elem_type == 2:
                        # 3-node triangle
                        n_vert = 3
                        elem_type_str = 'triangle'
                    elif elem_type == 4:
                        # 4-node tetrahedron
                        n_vert = 4
                        elem_type_str = 'tetrahedron'
                    else:
                        elem_idx += 1
                        continue
                    verts = [int(p) - 1 for p in parts[3 + num_tags:3 + num_tags + n_vert]]
                    elements.append(verts)
                elem_idx += 1
                continue

    nodes = np.array(nodes)
    if nodes.size == 0:
        raise ValueError("No nodes found in mesh file.")
    elements = np.array(elements)

    # Infer dimension from node coordinates variance
    dim = 2 if np.allclose(nodes[:, 2], nodes[0, 2]) else 3
    nodes = nodes[:, :dim]

    etype = 'triangle' if (elements.shape[1] == 3) else 'tetrahedron'
    return SimpleMesh(nodes, elements, element_type=etype)


def build_mass_matrix(mesh):
    """
    Build the lumped mass matrix for a mesh using the lumped-mass
    approximation:

        M_ii = Σ_{elements containing node i} (area_e / n_vertices).

    Parameters
    ----------
    mesh : SimpleMesh

    Returns
    -------
    M : ndarray, shape (N_nodes, N_nodes)
        Diagonal mass matrix.
    """
    N = mesh.nodes.shape[0]
    M_diag = np.zeros(N)
    n_vert = mesh.elements.shape[1]
    for e_idx in range(mesh.elements.shape[0]):
        vol = mesh.element_area_or_volume(e_idx)
        contrib = vol / n_vert
        for node in mesh.elements[e_idx]:
            M_diag[node] += contrib
    return np.diag(M_diag)


def build_stiffness_matrix_2d(mesh):
    """
    Build the stiffness matrix (discrete Laplacian) for a 2D triangle mesh
    using the cotangent formula:

        K_{ij} = - (1/2) (cot θ_{ij}^{(k)} + cot θ_{ij}^{(l)})
        K_{ii} = - Σ_{j≠i} K_{ij}

    where θ_{ij}^{(k)} is the angle opposite edge (i,j) in triangle k.
    """
    if mesh.dim != 2 or mesh.element_type != 'triangle':
        raise ValueError("Cotangent stiffness only for 2D triangle meshes.")

    N = mesh.nodes.shape[0]
    K = np.zeros((N, N))

    for tri in mesh.elements:
        i, j, k = tri
        pi, pj, pk = mesh.nodes[i], mesh.nodes[j], mesh.nodes[k]
        # Edge vectors
        eij = pj - pi
        eik = pk - pi
        eji = pi - pj
        ejk = pk - pj
        eki = pi - pk
        ekj = pj - pk

        # Cotangents of angles opposite each edge
        def cot(u, v):
            cos_angle = np.dot(u, v)
            sin_angle = abs(np.cross(u, v))
            if sin_angle < 1e-15:
                return 0.0
            return cos_angle / sin_angle

        cot_i = cot(eij, eik)
        cot_j = cot(eji, ejk)
        cot_k = cot(eki, ekj)

        K[i, j] -= 0.5 * cot_k
        K[j, i] -= 0.5 * cot_k
        K[i, k] -= 0.5 * cot_j
        K[k, i] -= 0.5 * cot_j
        K[j, k] -= 0.5 * cot_i
        K[k, j] -= 0.5 * cot_i

    # Diagonal entries
    for i in range(N):
        K[i, i] = -np.sum(K[i, :])

    return K


def assemble_nonhermitian_hamiltonian_fe(mesh, V_func, W_func, hbar=1.0, m_eff=1.0):
    """
    Assemble the finite-element non-Hermitian Hamiltonian matrix:

        H = -(ℏ^2 / 2m) K + diag(V(r_i) + i W(r_i))

    using the lumped-mass approximation.

    Parameters
    ----------
    mesh : SimpleMesh
    V_func : callable
        Real potential V(x, y) or V(x, y, z).
    W_func : callable
        Imaginary potential W(x, y) or W(x, y, z).
    hbar, m_eff : float

    Returns
    -------
    H : ndarray, dtype=complex
    M : ndarray
        Mass matrix.
    """
    if mesh.dim == 2:
        K = build_stiffness_matrix_2d(mesh)
    else:
        raise NotImplementedError("Only 2D implemented.")

    M = build_mass_matrix(mesh)
    N = mesh.nodes.shape[0]
    H = -(hbar ** 2 / (2.0 * m_eff)) * K.astype(complex)
    for i in range(N):
        x, y = mesh.nodes[i]
        H[i, i] += V_func(x, y) + 1j * W_func(x, y)
    return H, M
