
import numpy as np






def parse_stl_ascii(data_lines):
    vertices = []
    normals = []
    current_normal = np.array([0.0, 0.0, 1.0])
    current_verts = []

    for line in data_lines:
        line = line.strip().lower()
        if line.startswith('facet normal'):
            parts = line.split()
            if len(parts) >= 6:
                current_normal = np.array([
                    float(parts[2]), float(parts[3]), float(parts[4])
                ])
        elif line.startswith('vertex'):
            parts = line.split()
            if len(parts) >= 4:
                current_verts.append([
                    float(parts[1]), float(parts[2]), float(parts[3])
                ])
        elif line.startswith('endfacet'):
            if len(current_verts) == 3:
                vertices.append(current_verts)
                normals.append(current_normal)
            current_verts = []

    if len(vertices) == 0:
        return np.array([]), np.array([])
    return np.array(vertices, dtype=float), np.array(normals, dtype=float)


def compute_face_normals(vertices):
    v0 = vertices[:, 0, :]
    v1 = vertices[:, 1, :]
    v2 = vertices[:, 2, :]
    cross = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1e-15, norms)
    return cross / norms


def stla_check(vertices, normals):
    if vertices.size == 0:
        return 2


    v0 = vertices[:, 0, :]
    v1 = vertices[:, 1, :]
    v2 = vertices[:, 2, :]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = np.linalg.norm(cross, axis=1)
    if np.any(areas < 1e-12):
        return 1


    n_norms = np.linalg.norm(normals, axis=1)
    if np.any(np.abs(n_norms - 1.0) > 0.01):
        return 2

    return 0






def subset_sum_swap(weights, budget):
    weights = np.asarray(weights, dtype=float)
    if np.any(weights <= 0):
        raise ValueError("权重必须为正。")
    if budget <= 0:
        raise ValueError("预算必须为正。")

    n = len(weights)

    sorted_idx = np.argsort(weights)[::-1]
    sorted_w = weights[sorted_idx]

    selected = np.zeros(n, dtype=bool)
    achieved = 0.0

    while True:
        nmove = 0
        for i in range(n):
            if not selected[i]:
                if achieved + sorted_w[i] <= budget:
                    selected[i] = True
                    achieved += sorted_w[i]
                    nmove += 1
                    continue

            if not selected[i]:
                for j in range(n):
                    if selected[j]:
                        delta = sorted_w[i] - sorted_w[j]
                        if delta > 0 and achieved + delta <= budget:
                            selected[j] = False
                            selected[i] = True
                            achieved += delta
                            nmove += 2
                            break

        if nmove == 0:
            break


    result = np.zeros(n, dtype=bool)
    result[sorted_idx[selected]] = True
    return result, achieved


def sensor_placement_optimization(candidate_positions, field_values,
                                  budget_num, influence_radius):
    ny, nx = field_values.shape
    weights = np.zeros(len(candidate_positions))

    for idx, (j, i) in enumerate(candidate_positions):

        if 0 < i < nx - 1 and 0 < j < ny - 1:
            grad_x = abs(field_values[j, i + 1] - field_values[j, i - 1]) / 2.0
            grad_y = abs(field_values[j + 1, i] - field_values[j - 1, i]) / 2.0
            grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        else:
            grad_mag = 0.0
        weights[idx] = grad_mag


    max_w = np.max(weights)
    if max_w < 1e-15:
        weights[:] = 1.0


    total = np.sum(weights)
    budget = total * budget_num / len(candidate_positions) if len(candidate_positions) > 0 else 0

    selected, achieved = subset_sum_swap(weights, budget)
    return selected, achieved






def vortex_caustic_map(n_points, m_ratio, cylinder_center, radius_scale):
    theta = 2.0 * np.pi * np.arange(n_points + 1) / n_points
    z = np.exp(1j * theta)

    cx, cy = cylinder_center
    connections = []
    for j in range(n_points):
        idx1 = j
        idx2 = (j * m_ratio) % n_points
        x1 = cx + radius_scale * np.real(z[idx1])
        y1 = cy + radius_scale * np.imag(z[idx1])
        x2 = cx + radius_scale * np.real(z[idx2])
        y2 = cy + radius_scale * np.imag(z[idx2])
        connections.append(((x1, y1), (x2, y2)))


    from math import gcd
    w = gcd(n_points, m_ratio)
    winding_number = n_points // w if w > 0 else n_points

    return connections, winding_number






def extract_iso_line_segments(field, threshold, x_coords, y_coords):
    ny, nx = field.shape
    segments = []
    adjacency = {}
    node_id = 0
    node_map = {}

    def get_or_create_node(key, px, py):
        nonlocal node_id
        if key not in node_map:
            node_map[key] = node_id
            adjacency[node_id] = []
            node_id += 1
        return node_map[key]


    for j in range(ny):
        for i in range(nx - 1):
            f1 = field[j, i]
            f2 = field[j, i + 1]
            if (f1 - threshold) * (f2 - threshold) < 0:
                t = (threshold - f1) / (f2 - f1)
                px = x_coords[i] + t * (x_coords[i + 1] - x_coords[i])
                py = y_coords[j]
                key = (j, i, 'h')
                nid = get_or_create_node(key, px, py)


                segments.append((px, py))


    for j in range(ny - 1):
        for i in range(nx):
            f1 = field[j, i]
            f2 = field[j + 1, i]
            if (f1 - threshold) * (f2 - threshold) < 0:
                t = (threshold - f1) / (f2 - f1)
                px = x_coords[i]
                py = y_coords[j] + t * (y_coords[j + 1] - y_coords[j])
                key = (j, i, 'v')
                nid = get_or_create_node(key, px, py)
                segments.append((px, py))

    return segments, adjacency


def generate_simple_cylinder_stl_lines(diameter=1.0, num_facets=36):
    lines = ["solid cylinder"]
    r = diameter / 2.0
    for i in range(num_facets):
        theta1 = 2.0 * np.pi * i / num_facets
        theta2 = 2.0 * np.pi * (i + 1) / num_facets
        x1, y1 = r * np.cos(theta1), r * np.sin(theta1)
        x2, y2 = r * np.cos(theta2), r * np.sin(theta2)

        nx_n = (y2 - y1)
        ny_n = -(x2 - x1)
        nz_n = 0.0
        nn = np.sqrt(nx_n**2 + ny_n**2 + nz_n**2)
        if nn < 1e-15:
            nn = 1.0
        lines.append(f"  facet normal {nx_n/nn:.6f} {ny_n/nn:.6f} {nz_n/nn:.6f}")
        lines.append("    outer loop")
        lines.append(f"      vertex {x1:.6f} {y1:.6f} 0.0")
        lines.append(f"      vertex {x2:.6f} {y2:.6f} 0.0")
        lines.append(f"      vertex 0.0 0.0 0.0")
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append("endsolid cylinder")
    return lines
