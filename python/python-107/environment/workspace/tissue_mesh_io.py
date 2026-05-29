"""
tissue_mesh_io.py

Mesh I/O utilities for multi-layered tissue geometries.
Combines functionality from:
- gmsh_io: Gmsh mesh file reading and writing
- freefem_msh_io: FreeFem++ mesh format handling
- xy_io: Simple 2D coordinate point I/O for scan patterns

Supports 1D layered boundaries, 2D triangular meshes, and coordinate
serialization for OCT scan patterns and tissue interfaces.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Gmsh mesh I/O (from gmsh_io)
# ---------------------------------------------------------------------------

def gmsh_mesh2d_write(filename, nodes, elements):
    """
    Write a 2D triangular mesh in Gmsh format (version 2.2).

    Parameters
    ----------
    filename : str
        Output file path.
    nodes : ndarray, shape (n_nodes, 2)
        Node coordinates.
    elements : ndarray, shape (n_elements, 3)
        Triangle connectivity (0-based, converted to 1-based in file).
    """
    nodes = np.asarray(nodes, dtype=float)
    elements = np.asarray(elements, dtype=int)
    n_nodes = nodes.shape[0]
    n_elements = elements.shape[0]

    with open(filename, 'w') as f:
        f.write("$MeshFormat\n")
        f.write("2.2 0 8\n")
        f.write("$EndMeshFormat\n")
        f.write("$Nodes\n")
        f.write(f"{n_nodes}\n")
        for i in range(n_nodes):
            f.write(f"{i+1} {nodes[i,0]:.16g} {nodes[i,1]:.16g} 0.0\n")
        f.write("$EndNodes\n")
        f.write("$Elements\n")
        f.write(f"{n_elements}\n")
        for e in range(n_elements):
            f.write(f"{e+1} 2 2 0 {e+1} "
                    f"{elements[e,0]+1} {elements[e,1]+1} {elements[e,2]+1}\n")
        f.write("$EndElements\n")


def gmsh_mesh2d_read(filename):
    """
    Read a 2D triangular mesh from Gmsh format.

    Parameters
    ----------
    filename : str

    Returns
    -------
    nodes : ndarray, shape (n_nodes, 2)
    elements : ndarray, shape (n_elements, 3)
        0-based connectivity.
    """
    nodes = []
    elements = []
    in_nodes = False
    in_elements = False
    node_count = 0
    elem_count = 0
    nodes_read = 0
    elems_read = 0

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('$Nodes'):
                in_nodes = True
                continue
            if line.startswith('$EndNodes'):
                in_nodes = False
                continue
            if line.startswith('$Elements'):
                in_elements = True
                continue
            if line.startswith('$EndElements'):
                in_elements = False
                continue

            if in_nodes and nodes_read < node_count:
                parts = line.split()
                if len(parts) >= 3:
                    nodes.append([float(parts[1]), float(parts[2])])
                    nodes_read += 1
            elif in_nodes and nodes_read == 0:
                node_count = int(line)

            if in_elements and elems_read < elem_count:
                parts = line.split()
                if len(parts) >= 6:
                    # parts: elem_num type tag_num tag1 tag2 n1 n2 n3
                    elements.append([int(parts[5]) - 1, int(parts[6]) - 1, int(parts[7]) - 1])
                    elems_read += 1
            elif in_elements and elems_read == 0:
                elem_count = int(line)

    return np.array(nodes, dtype=float), np.array(elements, dtype=int)


# ---------------------------------------------------------------------------
# FreeFem++ mesh I/O (from freefem_msh_io)
# ---------------------------------------------------------------------------

def freefem_msh_read(filename):
    """
    Read a 2D FreeFem++ mesh file.

    Format:
      n_vertices n_triangles n_boundary_edges
      vertices: x y label
      triangles: v1 v2 v3 label
      edges: e1 e2 label

    Parameters
    ----------
    filename : str

    Returns
    -------
    nodes : ndarray, shape (n_vertices, 2)
    elements : ndarray, shape (n_triangles, 3)
        0-based.
    node_labels : ndarray, shape (n_vertices,)
    edge_data : list of tuple
        (v1, v2, label)
    """
    with open(filename, 'r') as f:
        parts = f.readline().strip().split()
        n_v = int(parts[0])
        n_t = int(parts[1])
        n_e = int(parts[2])

        nodes = []
        node_labels = []
        for _ in range(n_v):
            parts = f.readline().strip().split()
            nodes.append([float(parts[0]), float(parts[1])])
            node_labels.append(int(parts[2]))

        elements = []
        tri_labels = []
        for _ in range(n_t):
            parts = f.readline().strip().split()
            elements.append([int(parts[0]) - 1, int(parts[1]) - 1, int(parts[2]) - 1])
            tri_labels.append(int(parts[3]))

        edge_data = []
        for _ in range(n_e):
            parts = f.readline().strip().split()
            edge_data.append((int(parts[0]) - 1, int(parts[1]) - 1, int(parts[2])))

    return (np.array(nodes, dtype=float),
            np.array(elements, dtype=int),
            np.array(node_labels, dtype=int),
            edge_data)


def freefem_msh_write(filename, nodes, elements, node_labels=None, edge_data=None):
    """
    Write a 2D FreeFem++ mesh file.

    Parameters
    ----------
    filename : str
    nodes : ndarray, shape (n_v, 2)
    elements : ndarray, shape (n_t, 3)
        0-based, written as 1-based.
    node_labels : ndarray, optional
    edge_data : list of tuple, optional
    """
    nodes = np.asarray(nodes, dtype=float)
    elements = np.asarray(elements, dtype=int)
    n_v = nodes.shape[0]
    n_t = elements.shape[0]

    if node_labels is None:
        node_labels = np.zeros(n_v, dtype=int)
    if edge_data is None:
        edge_data = []
    n_e = len(edge_data)

    with open(filename, 'w') as f:
        f.write(f"{n_v} {n_t} {n_e}\n")
        for i in range(n_v):
            f.write(f"{nodes[i,0]:.16g} {nodes[i,1]:.16g} {node_labels[i]}\n")
        for i in range(n_t):
            f.write(f"{elements[i,0]+1} {elements[i,1]+1} {elements[i,2]+1} 0\n")
        for e in edge_data:
            f.write(f"{e[0]+1} {e[1]+1} {e[2]}\n")


# ---------------------------------------------------------------------------
# XY coordinate I/O (from xy_io)
# ---------------------------------------------------------------------------

def xy_header_write(filename, point_num):
    """
    Write XY file header.

    Format: first line is the number of points.

    Parameters
    ----------
    filename : str
    point_num : int
    """
    with open(filename, 'w') as f:
        f.write(f"{point_num}\n")


def xy_data_write(filename, coordinates, append=False):
    """
    Write 2D coordinates to XY file.

    Parameters
    ----------
    filename : str
    coordinates : ndarray, shape (n, 2)
    append : bool
        If True, append to existing file.
    """
    coordinates = np.asarray(coordinates, dtype=float)
    mode = 'a' if append else 'w'
    with open(filename, mode) as f:
        for i in range(coordinates.shape[0]):
            f.write(f"{coordinates[i,0]:.16g}  {coordinates[i,1]:.16g}\n")


def xy_data_read(filename):
    """
    Read 2D coordinates from XY file.

    Skips comment lines starting with '#' and blank lines.

    Parameters
    ----------
    filename : str

    Returns
    -------
    coordinates : ndarray, shape (n, 2)
    """
    coords = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                coords.append([float(parts[0]), float(parts[1])])
    return np.array(coords, dtype=float)


# ---------------------------------------------------------------------------
# Layered tissue geometry generation
# ---------------------------------------------------------------------------

def generate_layered_tissue_mesh(z_boundaries, radial_extent, n_r, n_z_per_layer):
    """
    Generate a simple 2D triangular mesh for layered tissue (cylindrical geometry).

    Creates a quadrilateral grid in (r, z) and splits each quad into 2 triangles.

    Parameters
    ----------
    z_boundaries : array_like
        z-coordinates of layer interfaces.
    radial_extent : float
        Maximum radial coordinate.
    n_r : int
        Number of radial divisions.
    n_z_per_layer : int
        Number of z divisions per layer.

    Returns
    -------
    nodes : ndarray, shape (n_nodes, 2)
        Coordinates (r, z).
    elements : ndarray, shape (n_elements, 3)
    """
    z_boundaries = np.asarray(z_boundaries, dtype=float)
    n_layers = len(z_boundaries) - 1
    r_vals = np.linspace(0.0, radial_extent, n_r + 1)

    nodes = []
    node_map = {}
    idx = 0
    for li in range(n_layers):
        z_vals = np.linspace(z_boundaries[li], z_boundaries[li + 1], n_z_per_layer + 1)
        for zi, z in enumerate(z_vals):
            for ri, r in enumerate(r_vals):
                # Only include boundary z nodes once between layers
                if li > 0 and zi == 0:
                    # Reuse node from previous layer top boundary
                    global_z_idx = sum(n_z_per_layer for _ in range(li)) + zi
                    key = (ri, global_z_idx)
                else:
                    global_z_idx = sum(n_z_per_layer for _ in range(li)) + zi
                    key = (ri, global_z_idx)
                if key not in node_map:
                    node_map[key] = idx
                    nodes.append([r, z])
                    idx += 1

    elements = []
    for li in range(n_layers):
        for zi in range(n_z_per_layer):
            for ri in range(n_r):
                global_z_base = sum(n_z_per_layer for _ in range(li)) + zi
                n00 = node_map[(ri, global_z_base)]
                n10 = node_map[(ri + 1, global_z_base)]
                n01 = node_map[(ri, global_z_base + 1)]
                n11 = node_map[(ri + 1, global_z_base + 1)]
                elements.append([n00, n10, n11])
                elements.append([n00, n11, n01])

    return np.array(nodes, dtype=float), np.array(elements, dtype=int)
