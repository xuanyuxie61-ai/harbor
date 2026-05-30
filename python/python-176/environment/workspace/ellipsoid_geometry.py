
import numpy as np
from scipy.spatial import Delaunay


def carlson_rf(x, y, z, tol=1.0e-12, max_iter=100):
    x = float(x)
    y = float(y)
    z = float(z)

    if x < 0.0 or y < 0.0 or z < 0.0:
        raise ValueError("RF: 输入参数必须非负。")

    lolim = 5.0e-26
    if x + y < lolim or x + z < lolim or y + z < lolim:
        raise ValueError("RF: 参数两两之和过小，接近奇点。")


    for _ in range(max_iter):
        rx = np.sqrt(x)
        ry = np.sqrt(y)
        rz = np.sqrt(z)
        lam = rx * ry + rx * rz + ry * rz
        x = 0.25 * (x + lam)
        y = 0.25 * (y + lam)
        z = 0.25 * (z + lam)
        if abs(x - y) < tol and abs(x - z) < tol and abs(y - z) < tol:
            break

    e2 = (x - y) * (x - z)
    e3 = (y - z) * e2

    c1 = 1.0 / 24.0
    c2 = 3.0 / 44.0
    c3 = 1.0 / 14.0
    u = (x + y + z) / 3.0
    val = (1.0 + c1 * e2 / u**2 - c2 * e3 / u**3 + c3 * e2**2 / u**4) / np.sqrt(u)
    return val


def carlson_rd(x, y, z, tol=1.0e-12, max_iter=100):
    x = float(x)
    y = float(y)
    z = float(z)

    if x < 0.0 or y < 0.0 or z < 0.0:
        raise ValueError("RD: 输入参数必须非负。")

    lolim = 5.0e-26
    if x + y < lolim or z < lolim:
        raise ValueError("RD: 参数接近奇点。")

    sigma = 0.0
    fac = 1.0
    for _ in range(max_iter):
        rx = np.sqrt(x)
        ry = np.sqrt(y)
        rz = np.sqrt(z)
        lam = rx * ry + rx * rz + ry * rz
        sigma += fac / (rz * (z + lam))
        fac *= 0.25
        x = 0.25 * (x + lam)
        y = 0.25 * (y + lam)
        z = 0.25 * (z + lam)
        if abs(x - y) < tol and abs(x - z) < tol and abs(y - z) < tol:
            break

    e2 = (x - y) * (x - z)
    e3 = (y - z) * e2
    c1 = 1.0 / 24.0
    c2 = 3.0 / 44.0
    c3 = 1.0 / 14.0
    u = (x + y + 3.0 * z) / 5.0
    val = (1.0 + c1 * e2 / u**2 - c2 * e3 / u**3 + c3 * e2**2 / u**4) / u**1.5
    val *= 3.0
    val += 6.0 * sigma
    return val


def elliptic_inc_fm(phi, m):
    s = np.sin(phi)
    c = np.cos(phi)
    if abs(s) < 1.0e-15:
        return phi
    val = s * carlson_rf(c * c, 1.0 - m * s * s, 1.0)
    return val


def elliptic_inc_em(phi, m):
    s = np.sin(phi)
    c = np.cos(phi)
    if abs(s) < 1.0e-15:
        return phi
    ss = s * s
    rf_val = carlson_rf(c * c, 1.0 - m * ss, 1.0)
    rd_val = carlson_rd(c * c, 1.0 - m * ss, 1.0)
    val = s * rf_val - (1.0 / 3.0) * m * ss * s * rd_val
    return val


def ellipsoid_surface_area(a, b, c):
    abc = np.array([a, b, c], dtype=float)
    abc = np.sort(abc)[::-1]
    a, b, c = abc


    if abs(a - b) < 1.0e-12 and abs(b - c) < 1.0e-12:

        return 4.0 * np.pi * a * a
    if abs(a - b) < 1.0e-12 and b > c:

        e2 = 1.0 - (c * c) / (a * a)
        e = np.sqrt(e2)
        return 2.0 * np.pi * a * a * (1.0 + (1.0 - e2) / e * np.arctanh(e))
    if a > b and abs(b - c) < 1.0e-12:

        e2 = 1.0 - (b * b) / (a * a)
        e = np.sqrt(e2)
        return 2.0 * np.pi * b * b * (1.0 + a / (b * e) * np.arcsin(e))


    phi = np.arccos(c / a)
    m = (a * a * (b * b - c * c)) / (b * b * (a * a - c * c))

    if m < 0.0:
        m = 0.0
    if m > 1.0:
        m = 1.0
    e_val = elliptic_inc_em(phi, m)
    f_val = elliptic_inc_fm(phi, m)
    s = 2.0 * np.pi * c * c + (2.0 * np.pi * a * b / np.sin(phi)) * (
        e_val * np.sin(phi)**2 + f_val * np.cos(phi)**2
    )
    return s


def ellipsoid_volume(a, b, c):
    return (4.0 / 3.0) * np.pi * a * b * c


def ellipse_area_2d(a, b):
    return np.pi * a * b


def generate_ellipse_mesh_2d(a, b, n_boundary=32, n_inner=80, seed=42):
    rng = np.random.default_rng(seed)


    theta = np.linspace(0.0, 2.0 * np.pi, n_boundary, endpoint=False)
    x_bnd = a * np.cos(theta)
    y_bnd = b * np.sin(theta)
    boundary_nodes = list(range(n_boundary))



    x_in = []
    y_in = []
    batch = 0
    while len(x_in) < n_inner and batch < 100:
        u = rng.random(n_inner * 2)
        r = np.sqrt(u[:n_inner * 2 // 2])
        t = u[n_inner * 2 // 2:] * 2.0 * np.pi

        xi = a * r * np.cos(t)
        yi = b * r * np.sin(t)
        for xx, yy in zip(xi, yi):
            if len(x_in) >= n_inner:
                break

            x_in.append(xx)
            y_in.append(yy)
        batch += 1


    if len(x_in) < n_inner:
        nx = int(np.sqrt(n_inner)) + 2
        xs = np.linspace(-a, a, nx)
        ys = np.linspace(-b, b, nx)
        for xx in xs:
            for yy in ys:
                if len(x_in) >= n_inner:
                    break
                if (xx / a) ** 2 + (yy / b) ** 2 < 0.99:
                    x_in.append(xx)
                    y_in.append(yy)

    x_in = np.array(x_in[:n_inner])
    y_in = np.array(y_in[:n_inner])


    nodes_bnd = np.column_stack((x_bnd, y_bnd))
    nodes_in = np.column_stack((x_in, y_in))
    nodes = np.vstack((nodes_bnd, nodes_in))


    tri = Delaunay(nodes)
    elements = tri.simplices.copy()


    centroid = np.mean(nodes[elements], axis=1)
    inside = ((centroid[:, 0] / a) ** 2 + (centroid[:, 1] / b) ** 2) <= 1.05
    elements = elements[inside]


    for e in range(elements.shape[0]):
        i, j, k = elements[e]
        xi, yi = nodes[i]
        xj, yj = nodes[j]
        xk, yk = nodes[k]
        area2 = (xj - xi) * (yk - yi) - (xk - xi) * (yj - yi)
        if area2 < 0:
            elements[e, 1], elements[e, 2] = elements[e, 2], elements[e, 1]


    edge_count = {}
    for e in elements:
        edges = [(e[0], e[1]), (e[1], e[2]), (e[2], e[0])]
        for v1, v2 in edges:
            key = tuple(sorted((int(v1), int(v2))))
            edge_count[key] = edge_count.get(key, 0) + 1

    boundary_edges = [e for e, c in edge_count.items() if c == 1]

    boundary_nodes = sorted(set([n for edge in boundary_edges for n in edge]))

    return nodes, elements, boundary_nodes


def compute_element_areas(nodes, elements):
    p1 = nodes[elements[:, 0]]
    p2 = nodes[elements[:, 1]]
    p3 = nodes[elements[:, 2]]
    areas = 0.5 * np.abs((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1])
                         - (p3[:, 0] - p1[:, 0]) * (p2[:, 1] - p1[:, 1]))
    return areas


def identify_boundary_edges(elements):
    edge_count = {}
    for e in elements:
        edges = [(e[0], e[1]), (e[1], e[2]), (e[2], e[0])]
        for v1, v2 in edges:
            key = tuple(sorted((int(v1), int(v2))))
            edge_count[key] = edge_count.get(key, 0) + 1
    boundary_edges = [e for e, c in edge_count.items() if c == 1]
    return boundary_edges


def write_tecplot_mesh(filename, nodes, elements, node_data=None, var_names=None):
    n_nodes = nodes.shape[0]
    n_elements = elements.shape[0]

    with open(filename, 'w') as f:
        f.write('TITLE = "Ellipse FEM Mesh"\n')
        if node_data is not None and var_names is not None:
            vars_str = ', '.join(['X', 'Y'] + list(var_names))
        else:
            vars_str = 'X, Y'
        f.write(f'VARIABLES = "{vars_str}"\n')
        f.write(f'ZONE N={n_nodes}, E={n_elements}, DATAPACKING=POINT, ZONETYPE=FETRIANGLE\n')
        for i in range(n_nodes):
            line = f"{nodes[i, 0]:.12e} {nodes[i, 1]:.12e}"
            if node_data is not None:
                for j in range(node_data.shape[1]):
                    line += f" {node_data[i, j]:.12e}"
            f.write(line + '\n')
        for e in elements:
            f.write(f"{e[0]+1} {e[1]+1} {e[2]+1}\n")


def read_tecplot_mesh(filename):
    nodes = []
    elements = []
    data_started = False
    elem_started = False
    n_nodes_expected = 0
    n_elements_expected = 0
    node_dim = 2

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.upper().startswith('VARIABLES='):
                parts = line.split('=')[1]

                var_count = parts.count(',') + 1
                node_dim = var_count
                continue
            if line.upper().startswith('ZONE'):

                parts = line.upper().split(',')
                for p in parts:
                    if 'N=' in p:
                        n_nodes_expected = int(p.split('=')[1].strip())
                    if 'E=' in p:
                        n_elements_expected = int(p.split('=')[1].strip())
                continue
            if n_nodes_expected > 0 and len(nodes) < n_nodes_expected:
                vals = [float(v) for v in line.split()]
                nodes.append(vals)
                if len(nodes) == n_nodes_expected:
                    elem_started = True
                continue
            if elem_started and len(elements) < n_elements_expected:
                vals = [int(v) - 1 for v in line.split()]
                elements.append(vals)
                continue

    nodes = np.array(nodes, dtype=float)
    if nodes.shape[1] > 2:
        node_data = nodes[:, 2:]
        nodes = nodes[:, :2]
    else:
        node_data = None
    elements = np.array(elements, dtype=int)
    return nodes, elements, node_data
