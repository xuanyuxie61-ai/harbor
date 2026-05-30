# -*- coding: utf-8 -*-

import numpy as np


def icosahedron_vertices():
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1,  phi,  0], [ 1,  phi,  0], [-1, -phi,  0], [ 1, -phi,  0],
        [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
        [ phi,  0, -1], [ phi,  0,  1], [-phi,  0, -1], [-phi,  0,  1]
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1)[:, np.newaxis]
    return verts


def icosahedron_faces():
    return np.array([
        [0,11,5], [0,5,1], [0,1,7], [0,7,10], [0,10,11],
        [1,5,9], [5,11,4], [11,10,2], [10,7,6], [7,1,8],
        [3,9,4], [3,4,2], [3,2,6], [3,6,8], [3,8,9],
        [4,9,5], [2,4,11], [6,2,10], [8,6,7], [9,8,1]
    ], dtype=int)


def subdivide_triangle(v1, v2, v3):
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
    verts = icosahedron_vertices()
    faces = icosahedron_faces()


    for _ in range(n_subdiv):
        new_faces = []
        for f in faces:
            tri_list = subdivide_triangle(verts[f[0]], verts[f[1]], verts[f[2]])
            for tri in tri_list:

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

            elements.append([vi[0], vi[1], vi[2], vo[0]])
            elements.append([vo[0], vo[1], vo[2], vi[2]])
            elements.append([vi[1], vi[2], vo[0], vo[2]])

    elements = np.array(elements, dtype=int)
    return nodes, elements


def tetrahedron_volume(nodes, element):
    r1, r2, r3, r4 = nodes[element[0]], nodes[element[1]], nodes[element[2]], nodes[element[3]]
    return abs(np.dot(r2 - r1, np.cross(r3 - r1, r4 - r1))) / 6.0


def integrate_density_on_mesh(rho_func, nodes, elements):
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
    r = np.sqrt(x * x + y * y + z * z)
    if r < 1e-6:
        theta = 0.0
    else:
        theta = np.arccos(np.clip(z / r, -1.0, 1.0))

    from nuclear_potential import deformed_radius
    R_th = deformed_radius(theta, A, beta2, beta3, beta4)

    if rho0 is None:

        R0 = 1.2 * (A ** (1.0 / 3.0))
        rho0 = 3.0 * A / (4.0 * np.pi * R0 ** 3)

    arg = (r - R_th) / a_diff
    arg = np.clip(arg, -500.0, 500.0)
    return rho0 / (1.0 + np.exp(arg))


def rms_radius_from_mesh(nodes, elements, rho_func):
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
