
import numpy as np
import os


def write_node_file(filename, node_coord):
    node_coord = np.asarray(node_coord, dtype=np.float64)
    if node_coord.ndim == 1:
        node_coord = node_coord.reshape(-1, 1)
    with open(filename, 'w') as f:
        for i in range(node_coord.shape[0]):
            line = "  ".join(f"{node_coord[i, j]:18.10e}" for j in range(node_coord.shape[1]))
            f.write(line + "\n")


def write_element_file(filename, element_node):
    element_node = np.asarray(element_node, dtype=np.int64)
    if element_node.ndim == 1:
        element_node = element_node.reshape(1, -1)
    with open(filename, 'w') as f:
        for i in range(element_node.shape[0]):
            line = "  ".join(f"{element_node[i, j] + 1:12d}" for j in range(element_node.shape[1]))
            f.write(line + "\n")


def write_value_file(filename, values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    with open(filename, 'w') as f:
        for i in range(values.shape[0]):
            line = "  ".join(f"{values[i, j]:18.10e}" for j in range(values.shape[1]))
            f.write(line + "\n")


def read_node_file(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            data.append([float(p) for p in parts])
    return np.array(data, dtype=np.float64)


def read_element_file(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            data.append([int(p) - 1 for p in parts])
    return np.array(data, dtype=np.int64)


def read_value_file(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            data.append([float(p) for p in parts])
    return np.array(data, dtype=np.float64)


def assemble_fem_data(prefix, node_coord, element_node, node_values):
    node_file = prefix + "_nodes.txt"
    element_file = prefix + "_elements.txt"
    value_file = prefix + "_values.txt"

    write_node_file(node_file, node_coord)
    write_element_file(element_file, element_node)
    write_value_file(value_file, node_values)

    return node_file, element_file, value_file


def write_tecplot_ascii(filename, node_coord, element_node, node_values, var_names=None):
    node_coord = np.asarray(node_coord, dtype=np.float64)
    element_node = np.asarray(element_node, dtype=np.int64)
    node_values = np.asarray(node_values, dtype=np.float64)

    if node_coord.ndim == 1:
        node_coord = node_coord.reshape(-1, 1)
    if node_values.ndim == 1:
        node_values = node_values.reshape(-1, 1)

    dim = node_coord.shape[1]
    n_nodes = node_coord.shape[0]
    n_elements = element_node.shape[0]
    n_vars = dim + node_values.shape[1]

    if var_names is None:
        var_names = [f"Var{i + 1}" for i in range(n_vars)]

    with open(filename, 'w') as f:
        f.write(f'TITLE = "{filename}"\n')
        f.write('VARIABLES = ' + ", ".join(f'"{v}"' for v in var_names) + '\n')
        f.write(f'ZONE N={n_nodes} E={n_elements} F=FEPOINT ET=TRIANGLE\n')


        for i in range(n_nodes):
            coords = "  ".join(f"{node_coord[i, j]:18.10e}" for j in range(dim))
            vals = "  ".join(f"{node_values[i, j]:18.10e}" for j in range(node_values.shape[1]))
            f.write(coords + "  " + vals + "\n")


        for i in range(n_elements):
            conn = "  ".join(f"{element_node[i, j] + 1:12d}" for j in range(element_node.shape[1]))
            f.write(conn + "\n")


def compute_fem_mass_matrix(node_coord, element_node):
    node_coord = np.asarray(node_coord, dtype=np.float64)
    element_node = np.asarray(element_node, dtype=np.int64)
    n_nodes = node_coord.shape[0]
    M = np.zeros(n_nodes, dtype=np.float64)

    for e in element_node:
        p1, p2, p3 = node_coord[e[0]], node_coord[e[1]], node_coord[e[2]]
        area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
        for idx in e:
            M[idx] += area / 3.0

    return M


def compute_fem_stiffness_matrix(node_coord, element_node, kappa_element):
    node_coord = np.asarray(node_coord, dtype=np.float64)
    element_node = np.asarray(element_node, dtype=np.int64)
    n_nodes = node_coord.shape[0]
    K = np.zeros((n_nodes, n_nodes), dtype=np.float64)

    for idx_e, e in enumerate(element_node):
        p1, p2, p3 = node_coord[e[0]], node_coord[e[1]], node_coord[e[2]]
        area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
        if area < 1e-14:
            continue

        b = np.array([p2[1] - p3[1], p3[1] - p1[1], p1[1] - p2[1]], dtype=np.float64)
        c = np.array([p3[0] - p2[0], p1[0] - p3[0], p2[0] - p1[0]], dtype=np.float64)
        kappa = kappa_element[idx_e] if hasattr(kappa_element, '__len__') else kappa_element
        for i in range(3):
            for j in range(3):
                K[e[i], e[j]] += kappa * (b[i] * b[j] + c[i] * c[j]) / (4.0 * area)

    return K
