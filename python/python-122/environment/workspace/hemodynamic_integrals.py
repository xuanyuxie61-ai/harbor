
import numpy as np



def tetrahedron01_monomial_integral(e):
    e = np.asarray(e, dtype=int)
    if np.any(e < 0):
        return 0.0
    m = len(e)
    k = 0
    integral = 1.0
    for i in range(m):
        for j in range(1, e[i] + 1):
            k += 1
            integral *= j / k
    for i in range(m):
        k += 1
        integral /= k
    return integral


def tetrahedron01_volume():
    return 1.0 / 6.0


def tetrahedron01_sample(n):
    u = np.random.rand(n, 3)

    x = 1.0 - u[:, 0] ** (1.0 / 3.0)
    y = (1.0 - u[:, 1] ** (1.0 / 2.0)) * (1.0 - x)
    z = u[:, 2] * (1.0 - x - y)
    return np.column_stack((x, y, z))


def integrate_blood_volume_tetrahedral(p, t):
    total_vol = 0.0
    for i in range(t.shape[0]):
        idx = t[i]
        v0, v1, v2, v3 = p[idx[0]], p[idx[1]], p[idx[2]], p[idx[3]]
        mat = np.column_stack((v1 - v0, v2 - v0, v3 - v0))
        vol = abs(np.linalg.det(mat)) / 6.0
        total_vol += vol
    return total_vol



def hypersphere_01_surface_uniform(m, n):
    x = np.zeros((m, n))
    for j in range(n):
        v = np.random.randn(m)
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-14:
            v_norm = 1.0
        x[:, j] = v / v_norm
    return x


def hypersphere_01_interior_uniform(m, n):
    exponent = 1.0 / m
    surface = hypersphere_01_surface_uniform(m, n)
    r = np.random.rand(n) ** exponent
    return surface * r[None, :]


def sample_vascular_cross_section(n_points, radius, center, normal):
    normal = np.asarray(normal, dtype=float)
    normal = normal / (np.linalg.norm(normal) + 1e-14)


    if abs(normal[2]) < 0.9:
        tangent1 = np.cross(normal, np.array([0.0, 0.0, 1.0]))
    else:
        tangent1 = np.cross(normal, np.array([0.0, 1.0, 0.0]))
    tangent1 = tangent1 / (np.linalg.norm(tangent1) + 1e-14)
    tangent2 = np.cross(normal, tangent1)
    tangent2 = tangent2 / (np.linalg.norm(tangent2) + 1e-14)


    disk = hypersphere_01_interior_uniform(2, n_points)
    points = center[None, :] + radius * (disk[0, :][:, None] * tangent1[None, :] +
                                          disk[1, :][:, None] * tangent2[None, :])
    return points


def monte_carlo_flow_rate_integral(n_samples, radius, velocity_profile_func):
    area = np.pi * radius ** 2

    disk = hypersphere_01_interior_uniform(2, n_samples)
    r_local = radius * np.sqrt(disk[0, :] ** 2 + disk[1, :] ** 2)
    r_local = np.clip(r_local, 0.0, radius)
    v_samples = velocity_profile_func(r_local)
    Q_est = area * np.mean(v_samples)
    return Q_est



def triangle_node_write(filename, node_coord, node_att=None):
    n_nodes = node_coord.shape[0]
    dim = node_coord.shape[1]
    with open(filename, 'w') as f:
        f.write(f"{n_nodes} {dim} 0 0\n")
        for i in range(n_nodes):
            line = f"{i} " + " ".join(f"{c:.6e}" for c in node_coord[i])
            if node_att is not None:
                line += " " + " ".join(f"{a:.6e}" for a in np.atleast_1d(node_att[i]))
            f.write(line + "\n")


def triangle_element_write(filename, element_node, element_att=None):
    n_elem = element_node.shape[0]
    order = element_node.shape[1]
    with open(filename, 'w') as f:
        f.write(f"{n_elem} {order} 0\n")
        for i in range(n_elem):
            line = f"{i} " + " ".join(str(idx) for idx in element_node[i])
            if element_att is not None:
                line += " " + " ".join(f"{a:.6e}" for a in np.atleast_1d(element_att[i]))
            f.write(line + "\n")


def triangle_node_read(filename):
    coords = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                x = float(parts[1])
                y = float(parts[2])
                coords.append([x, y])
            except ValueError:
                continue
    return np.array(coords)


def triangle_element_read(filename):
    elems = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                n1 = int(parts[1])
                n2 = int(parts[2])
                n3 = int(parts[3])
                elems.append([n1, n2, n3])
            except ValueError:
                continue
    return np.array(elems)
