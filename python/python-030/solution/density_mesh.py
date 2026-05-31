# -*- coding: utf-8 -*-
r"""
density_mesh.py
===============
Three-dimensional nuclear density discretisation on tetrahedral meshes,
merging concepts from **tet_mesh_to_xml** (tetrahedral mesh topology)
and **triangulation_svg** / **mario** / **peaks_movie** / **usa_box_plot**
(structured grid generation and pixel/triangle mapping).

Physical model
--------------
The nuclear density distribution :math:`\rho(\mathbf{r})` for a deformed
nucleus is parametrised by a Fermi-type profile:

.. math::
    \rho(r,\theta) = \frac{\rho_0}{1 + \exp\!\left(
    \dfrac{r - R(\theta)}{a}\right)} \;,

where :math:`R(\theta)` is the angle-dependent nuclear radius and
:math:`a\approx 0.5` fm is the surface diffuseness.

The central density :math:`\rho_0` is fixed by normalisation:

.. math::
    \int \rho(\mathbf{r})\,d^3r = A \;.

Mesh generation
---------------
The spherical domain :math:`r\in[0, r_{\max}]` is discretised into
concentric spherical shells.  Each shell is subdivided into tetrahedra
by projecting a triangulated icosahedron onto the sphere and connecting
adjacent shells.  The resulting mesh is stored as

* *nodes*: :math:`(x,y,z)` coordinates,
* *elements*: tetrahedron node-index lists.

The isoparametric mapping for a tetrahedron with vertices
:math:`\mathbf{r}_1,\dots,\mathbf{r}_4` is

.. math::
    \mathbf{r}(\boldsymbol{\xi}) = \sum_{i=1}^{4} N_i(\boldsymbol{\xi})\,
    \mathbf{r}_i \;,

with linear shape functions
:math:`N_1=\xi_1, N_2=\xi_2, N_3=\xi_3, N_4=1-\xi_1-\xi_2-\xi_3`.
"""

import numpy as np


def icosahedron_vertices():
    r"""
    Vertices of a regular icosahedron inscribed in the unit sphere.

    Returns
    -------
    verts : ndarray, shape (12, 3)
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0  # golden ratio
    verts = np.array([
        [-1,  phi,  0], [ 1,  phi,  0], [-1, -phi,  0], [ 1, -phi,  0],
        [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
        [ phi,  0, -1], [ phi,  0,  1], [-phi,  0, -1], [-phi,  0,  1]
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1)[:, np.newaxis]
    return verts


def icosahedron_faces():
    r"""
    Face indices of a regular icosahedron.

    Returns
    -------
    faces : ndarray, shape (20, 3)
    """
    return np.array([
        [0,11,5], [0,5,1], [0,1,7], [0,7,10], [0,10,11],
        [1,5,9], [5,11,4], [11,10,2], [10,7,6], [7,1,8],
        [3,9,4], [3,4,2], [3,2,6], [3,6,8], [3,8,9],
        [4,9,5], [2,4,11], [6,2,10], [8,6,7], [9,8,1]
    ], dtype=int)


def subdivide_triangle(v1, v2, v3):
    r"""
    Subdivide a spherical triangle into 4 smaller spherical triangles.

    Parameters
    ----------
    v1, v2, v3 : ndarray, shape (3,)
        Unit vectors of vertices.

    Returns
    -------
    tri_list : list of ndarray
        List of 4 triangles.
    """
    a = (v1 + v2)
    a /= np.linalg.norm(a)
    b = (v2 + v3)
    b /= np.linalg.norm(b)
    c = (v3 + v1)
    c /= np.linalg.norm(c)
    return [
        np.vstack([v1, a, c]),
        np.vstack([v2, b, a]),
        np.vstack([v3, c, b]),
        np.vstack([a, b, c])
    ]


def spherical_triangulation(n_subdiv=2):
    r"""
    Generate a triangulated sphere by recursive icosahedron subdivision.

    Parameters
    ----------
    n_subdiv : int
        Number of subdivision levels.

    Returns
    -------
    vertices : ndarray, shape (N_v, 3)
        Unique vertices on unit sphere.
    faces : ndarray, shape (N_f, 3)
        Triangle face indices.
    """
    verts = icosahedron_vertices()
    faces = icosahedron_faces()

    # Subdivide
    for _ in range(n_subdiv):
        new_faces = []
        for f in faces:
            tri_list = subdivide_triangle(verts[f[0]], verts[f[1]], verts[f[2]])
            for tri in tri_list:
                # Find or add vertices
                idx = []
                for vv in tri:
                    dists = np.linalg.norm(verts - vv, axis=1)
                    match = np.where(dists < 1e-8)[0]
                    if match.size > 0:
                        idx.append(match[0])
                    else:
                        verts = np.vstack([verts, vv])
                        idx.append(verts.shape[0] - 1)
                new_faces.append(idx)
        faces = np.array(new_faces, dtype=int)

    return verts, faces


def build_tetrahedral_sphere_mesh(rmin, rmax, n_shells, n_subdiv=2):
    r"""
    Build a tetrahedral mesh of spherical shells.

    Parameters
    ----------
    rmin, rmax : float
        Inner and outer radii in fm.
    n_shells : int
        Number of radial shells.
    n_subdiv : int
        Icosahedron subdivision level.

    Returns
    -------
    nodes : ndarray, shape (n_nodes, 3)
        Node coordinates.
    elements : ndarray, shape (n_elements, 4)
        Tetrahedron node indices (0-based).
    """
    sphere_verts, sphere_faces = spherical_triangulation(n_subdiv)
    n_surf = sphere_verts.shape[0]
    n_faces = sphere_faces.shape[0]

    radii = np.linspace(rmin, rmax, n_shells)
    nodes = []
    for r in radii:
        nodes.append(r * sphere_verts)
    nodes = np.vstack(nodes)

    elements = []
    for shell in range(n_shells - 1):
        base_inner = shell * n_surf
        base_outer = (shell + 1) * n_surf
        for f in sphere_faces:
            vi = [base_inner + f[0], base_inner + f[1], base_inner + f[2]]
            vo = [base_outer + f[0], base_outer + f[1], base_outer + f[2]]
            # Split triangular prism into 3 tetrahedra
            elements.append([vi[0], vi[1], vi[2], vo[0]])
            elements.append([vo[0], vo[1], vo[2], vi[2]])
            elements.append([vi[1], vi[2], vo[0], vo[2]])

    elements = np.array(elements, dtype=int)
    return nodes, elements


def tetrahedron_volume(nodes, element):
    r"""
    Volume of a tetrahedron given 4 node indices.

    .. math::
        V = \frac{1}{6}\bigl|(\mathbf{r}_2-\mathbf{r}_1)
        \cdot\bigl[(\mathbf{r}_3-\mathbf{r}_1)
        \times(\mathbf{r}_4-\mathbf{r}_1)\bigr]\bigr|

    Parameters
    ----------
    nodes : ndarray, shape (n_nodes, 3)
    element : iterable of 4 ints
        Node indices.

    Returns
    -------
    vol : float
        Volume in fm³.
    """
    r1, r2, r3, r4 = nodes[element[0]], nodes[element[1]], nodes[element[2]], nodes[element[3]]
    return abs(np.dot(r2 - r1, np.cross(r3 - r1, r4 - r1))) / 6.0


def integrate_density_on_mesh(rho_func, nodes, elements):
    r"""
    Integrate a density function over a tetrahedral mesh using one-point
    quadrature (centroid).

    .. math::
        \int \rho\,dV \approx \sum_{e} \rho(\mathbf{r}_c^{(e)})\,V_e

    Parameters
    ----------
    rho_func : callable
        Density function :math:`\rho(x,y,z)`.
    nodes : ndarray
    elements : ndarray

    Returns
    -------
    total : float
        Integrated density.
    element_data : list of dict
        Per-element volume and centroid density.
    """
    total = 0.0
    element_data = []
    for el in elements:
        coords = nodes[el, :]
        centroid = np.mean(coords, axis=0)
        vol = tetrahedron_volume(nodes, el)
        rho_c = rho_func(centroid[0], centroid[1], centroid[2])
        total += rho_c * vol
        element_data.append({
            'volume': vol,
            'centroid': centroid,
            'rho': rho_c
        })
    return total, element_data


def deformed_fermi_density(x, y, z, A, beta2=0.0, beta3=0.0, beta4=0.0,
                           rho0=None, a_diff=0.5):
    r"""
    Three-parameter deformed Fermi density.

    .. math::
        \rho(r,\theta) = \frac{\rho_0}
        {1 + \exp\!\left(\dfrac{r - R(\theta)}{a}\right)}

    Parameters
    ----------
    x, y, z : float
        Cartesian coordinates in fm.
    A : int
        Mass number.
    beta2, beta3, beta4 : float
        Deformation parameters.
    rho0 : float, optional
        Central density in fm⁻³.  Computed from normalisation if None.
    a_diff : float
        Surface diffuseness in fm.

    Returns
    -------
    rho : float
        Density in fm⁻³.
    """
    r = np.sqrt(x * x + y * y + z * z)
    if r < 1e-6:
        theta = 0.0
    else:
        theta = np.arccos(np.clip(z / r, -1.0, 1.0))

    from nuclear_potential import deformed_radius
    R_th = deformed_radius(theta, A, beta2, beta3, beta4)

    if rho0 is None:
        # Approximate normalisation: rho0 ≈ 3A / (4π R³)
        R0 = 1.2 * (A ** (1.0 / 3.0))
        rho0 = 3.0 * A / (4.0 * np.pi * R0 ** 3)

    arg = (r - R_th) / a_diff
    arg = np.clip(arg, -500.0, 500.0)
    return rho0 / (1.0 + np.exp(arg))


def rms_radius_from_mesh(nodes, elements, rho_func):
    r"""
    Root-mean-square charge radius from mesh density.

    .. math::
        \langle r^2 \rangle^{1/2}
        = \left[\frac{\int r^2 \rho(\mathbf{r})\,dV}
                      {\int \rho(\mathbf{r})\,dV}\right]^{1/2}

    Parameters
    ----------
    nodes, elements : ndarray
    rho_func : callable

    Returns
    -------
    rms : float
        RMS radius in fm.
    """
    num = 0.0
    den = 0.0
    for el in elements:
        coords = nodes[el, :]
        centroid = np.mean(coords, axis=0)
        vol = tetrahedron_volume(nodes, el)
        rho_c = rho_func(centroid[0], centroid[1], centroid[2])
        r2 = np.sum(centroid ** 2)
        num += r2 * rho_c * vol
        den += rho_c * vol
    if den <= 0:
        return 0.0
    return np.sqrt(num / den)


def write_mesh_to_xml(nodes, elements, filename):
    r"""
    Write a tetrahedral mesh to a simplified XML format
    (inspired by *tet_mesh_to_xml* / DOLFIN XML).

    Parameters
    ----------
    nodes : ndarray, shape (n_nodes, 3)
    elements : ndarray, shape (n_elements, 4)
    filename : str
        Output file path.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<dolfin xmlns:dolfin="http://www.fenicsproject.org">\n')
        f.write(f'  <mesh celltype="tetrahedron" dim="3">\n')
        f.write(f'    <vertices size="{nodes.shape[0]}">\n')
        for i, (x, y, z) in enumerate(nodes):
            f.write(f'      <vertex index="{i}" x="{x:.8e}" y="{y:.8e}" z="{z:.8e}"/>\n')
        f.write('    </vertices>\n')
        f.write(f'    <cells size="{elements.shape[0]}">\n')
        for i, el in enumerate(elements):
            f.write(f'      <tetrahedron index="{i}" '
                    f'v0="{el[0]}" v1="{el[1]}" v2="{el[2]}" v3="{el[3]}"/>\n')
        f.write('    </cells>\n')
        f.write('  </mesh>\n')
        f.write('</dolfin>\n')
