
import numpy as np



HEX_DIRECTIONS = np.array([
    [1.0, 0.0],
    [0.5, np.sqrt(3.0) / 2.0],
    [-0.5, np.sqrt(3.0) / 2.0],
    [-1.0, 0.0],
    [-0.5, -np.sqrt(3.0) / 2.0],
    [0.5, -np.sqrt(3.0) / 2.0]
])


def axial_to_cartesian(q, r, size=1.0):
    x = size * (3.0 / 2.0 * q)
    y = size * (np.sqrt(3.0) / 2.0 * q + np.sqrt(3.0) * r)
    return x, y


def cartesian_to_axial(x, y, size=1.0):
    q = (2.0 / 3.0 * x) / size
    r = (-1.0 / 3.0 * x + np.sqrt(3.0) / 3.0 * y) / size
    return q, r


def hex_round(q, r):
    if isinstance(q, np.ndarray):
        s = -q - r
        x = np.round(q)
        y = np.round(r)
        z = np.round(s)

        x_diff = np.abs(x - q)
        y_diff = np.abs(y - r)
        z_diff = np.abs(z - s)

        mask_x = (x_diff > y_diff) & (x_diff > z_diff)
        x[mask_x] = -y[mask_x] - z[mask_x]

        mask_y = (~mask_x) & (y_diff > z_diff)
        y[mask_y] = -x[mask_y] - z[mask_y]

        mask_z = (~mask_x) & (~mask_y)
        z[mask_z] = -x[mask_z] - y[mask_z]

        return x.astype(int), y.astype(int)
    else:
        s = -q - r
        x = round(q)
        y = round(r)
        z = round(s)

        x_diff = abs(x - q)
        y_diff = abs(y - r)
        z_diff = abs(z - s)

        if x_diff > y_diff and x_diff > z_diff:
            x = -y - z
        elif y_diff > z_diff:
            y = -x - z
        else:
            z = -x - y

        return int(x), int(y)


def generate_hexagonal_lattice(radius, size=1.0):
    points = []
    axial = []

    for q in range(-radius, radius + 1):
        r1 = max(-radius, -q - radius)
        r2 = min(radius, -q + radius)
        for r in range(r1, r2 + 1):
            x, y = axial_to_cartesian(q, r, size)
            points.append([x, y])
            axial.append([q, r])

    return np.array(points), np.array(axial, dtype=int)


def boundary_word_to_polygon(boundary_word, start_q=0, start_r=0, size=1.0):
    vertices = []
    q, r = start_q, start_r

    x, y = axial_to_cartesian(q, r, size)
    vertices.append([x, y])

    for direction in boundary_word:
        dq, dr = HEX_DIRECTIONS[direction][:2]

        if direction == 0:
            q += 1
        elif direction == 1:
            r += 1
        elif direction == 2:
            q -= 1
            r += 1
        elif direction == 3:
            q -= 1
        elif direction == 4:
            r -= 1
        elif direction == 5:
            q += 1
            r -= 1

        x, y = axial_to_cartesian(q, r, size)
        vertices.append([x, y])

    return np.array(vertices)


def approximate_boundary_with_hexagons(
    boundary_func,
    domain_bounds,
    hex_size,
    n_samples=1000
):
    t_vals = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    boundary_points = np.array([boundary_func(t) for t in t_vals])


    q_vals, r_vals = cartesian_to_axial(
        boundary_points[:, 0], boundary_points[:, 1], hex_size
    )
    q_int, r_int = hex_round(q_vals, r_vals)


    unique_coords = set()
    hex_points = []
    for q, r in zip(q_int, r_int):
        key = (q, r)
        if key not in unique_coords:
            unique_coords.add(key)
            x, y = axial_to_cartesian(q, r, hex_size)
            hex_points.append([x, y])

    return np.array(hex_points), boundary_points


def hex_boundary_refinement_indicator(hex_points, solution_values, hex_size):
    n = len(hex_points)
    indicators = np.zeros(n)

    for i in range(n):

        dists = np.linalg.norm(hex_points - hex_points[i], axis=1)
        neighbor_mask = (dists > 1e-10) & (dists < 2.5 * hex_size)

        if np.sum(neighbor_mask) > 0:
            neighbor_vals = solution_values[neighbor_mask]

            local_grad = np.max(np.abs(neighbor_vals - solution_values[i])) / hex_size
            indicators[i] = local_grad * hex_size
        else:
            indicators[i] = 0.0

    return indicators
