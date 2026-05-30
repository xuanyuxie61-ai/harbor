
import numpy as np
from utils_numerical import safe_divide


def generate_voronoi_mesh(nc: int, bounds: tuple = ((0.0, 1.0), (0.0, 1.0)),
                          m: int = 100, n: int = 100, p_norm: float = 2.0) -> dict:
    xmin, xmax = bounds[0]
    ymin, ymax = bounds[1]


    np.random.seed(42)
    generators = np.random.rand(2, nc)
    generators[0, :] = xmin + generators[0, :] * (xmax - xmin)
    generators[1, :] = ymin + generators[1, :] * (ymax - ymin)


    boundary_pts = np.array([
        [xmin, xmin, xmax, xmax],
        [ymin, ymax, ymin, ymax]
    ])
    generators = np.hstack([generators, boundary_pts])
    nc_total = generators.shape[1]


    x_grid = np.linspace(xmin, xmax, n)
    y_grid = np.linspace(ymin, ymax, m)
    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]


    voronoi_map = np.zeros((m, n), dtype=int)
    for i in range(m):
        y = y_grid[i]
        for j in range(n):
            x = x_grid[j]

            min_dist = np.inf
            nearest = 0
            for k in range(nc_total):
                gx, gy = generators[:, k]
                if p_norm == np.inf:
                    dist = max(abs(x - gx), abs(y - gy))
                elif p_norm == 1.0:
                    dist = abs(x - gx) + abs(y - gy)
                elif p_norm == 2.0:
                    dist = (x - gx) ** 2 + (y - gy) ** 2
                else:
                    dx_ = abs(x - gx)
                    dy_ = abs(y - gy)
                    dist = (dx_ ** p_norm + dy_ ** p_norm) ** (1.0 / p_norm)

                if dist < min_dist:
                    min_dist = dist
                    nearest = k

            voronoi_map[i, j] = nearest


    cell_areas = np.zeros(nc_total)
    for k in range(nc_total):
        cell_areas[k] = np.sum(voronoi_map == k) * dx * dy

    return {
        'generators': generators[:, :nc],
        'voronoi_map': voronoi_map,
        'x_grid': x_grid,
        'y_grid': y_grid,
        'cell_areas': cell_areas[:nc],
        'dx': dx,
        'dy': dy
    }


def ifs_adaptive_refinement(n_iterations: int = 5000, bounds: tuple = ((0.0, 1.0), (0.0, 1.0)),
                            refinement_regions: list = None) -> np.ndarray:

    transforms = [
        {'A': np.array([[0.0, 0.0], [0.0, 0.5]]), 'b': np.array([0.5, 0.0]), 'prob': 0.25},
        {'A': np.array([[0.1, 0.0], [0.0, 0.1]]), 'b': np.array([0.45, 0.15]), 'prob': 0.25},
        {'A': np.array([[0.42, -0.42], [0.42, 0.42]]), 'b': np.array([0.29, -0.01]), 'prob': 0.25},
        {'A': np.array([[0.42, 0.42], [-0.42, 0.42]]), 'b': np.array([0.29, 0.41]), 'prob': 0.25}
    ]


    cum_prob = np.cumsum([t['prob'] for t in transforms])

    points = np.zeros((2, n_iterations))
    x = np.random.rand(2)


    for _ in range(100):
        r = np.random.rand()
        idx = np.searchsorted(cum_prob, r)
        x = transforms[idx]['A'] @ x + transforms[idx]['b']


    accepted = 0
    i = 0
    while accepted < n_iterations and i < n_iterations * 10:
        i += 1
        r = np.random.rand()
        idx = np.searchsorted(cum_prob, r)
        x = transforms[idx]['A'] @ x + transforms[idx]['b']


        px = bounds[0][0] + x[0] * (bounds[0][1] - bounds[0][0])
        py = bounds[1][0] + x[1] * (bounds[1][1] - bounds[1][0])


        if refinement_regions:
            in_refined = any(
                rx[0] <= px <= rx[1] and ry[0] <= py <= ry[1]
                for rx, ry in refinement_regions
            )
            if in_refined:

                if np.random.rand() < 0.8:
                    points[:, accepted] = [px, py]
                    accepted += 1
            else:
                if np.random.rand() < 0.3:
                    points[:, accepted] = [px, py]
                    accepted += 1
        else:
            points[:, accepted] = [px, py]
            accepted += 1

    return points[:, :accepted]


def sample_boundary_points(n_points: int = 50, boundary_type: str = 'plate',
                           Re: float = 1e5, x_range: tuple = (0.0, 1.0)) -> tuple:
    if boundary_type == 'plate':


        x_sample = np.linspace(x_range[0] + 0.01, x_range[1], n_points)
        delta = 5.0 * x_sample / np.sqrt(np.maximum(Re * x_sample, 1.0))


        beta = 2.5
        eta = np.linspace(0.0, 1.0, n_points)
        y_norm = np.tanh(beta * eta) / np.tanh(beta)


        x_grid = np.tile(x_sample, n_points)
        y_grid = np.outer(delta, y_norm).flatten()


        dy_wall = delta[0] * (np.tanh(beta / n_points) / np.tanh(beta))

        return x_grid, y_grid, float(dy_wall)

    else:

        theta = np.linspace(0.0, 2.0 * np.pi, n_points)
        x_airfoil = 0.5 * (1.0 + np.cos(theta))
        y_airfoil = 0.06 * (0.2969 * np.sqrt(x_airfoil) - 0.1260 * x_airfoil
                          - 0.3516 * x_airfoil ** 2 + 0.2843 * x_airfoil ** 3
                          - 0.1015 * x_airfoil ** 4)


        x_points = np.concatenate([x_airfoil, x_airfoil[::-1]])
        y_points = np.concatenate([y_airfoil, -y_airfoil[::-1]])

        return x_points, y_points, 0.01


def generate_spectral_element_mesh(nx: int = 16, ny: int = 16,
                                   x_bounds: tuple = (0.0, 1.0),
                                   y_bounds: tuple = (0.0, 1.0),
                                   stretch_y: bool = True) -> dict:

    def chebyshev_nodes(n):
        return np.cos(np.pi * np.arange(n + 1) / n)


    n_gll = 8
    xi = chebyshev_nodes(n_gll)

    x_nodes = np.linspace(x_bounds[0], x_bounds[1], nx + 1)
    y_nodes = np.linspace(y_bounds[0], y_bounds[1], ny + 1)


    npx = nx * n_gll + 1
    npy = ny * n_gll + 1
    x = np.zeros(npx)
    y = np.zeros(npy)


    for i in range(nx):
        for j in range(n_gll + 1):
            idx = i * n_gll + j
            if idx < npx:
                x_local = 0.5 * ((1 - xi[j]) * x_nodes[i] + (1 + xi[j]) * x_nodes[i + 1])
                x[idx] = x_local


    for i in range(ny):
        for j in range(n_gll + 1):
            idx = i * n_gll + j
            if idx < npy:
                y_local = 0.5 * ((1 - xi[j]) * y_nodes[i] + (1 + xi[j]) * y_nodes[i + 1])
                if stretch_y:

                    y_local = y_bounds[1] * (np.exp(2.0 * y_local / y_bounds[1]) - 1.0) / (np.exp(2.0) - 1.0)
                y[idx] = y_local


    x = np.unique(np.round(x, 12))
    y = np.unique(np.round(y, 12))


    X, Y = np.meshgrid(x, y)


    dx = np.diff(x)
    dy = np.diff(y)

    return {
        'x': x,
        'y': y,
        'X': X,
        'Y': Y,
        'dx_min': float(np.min(dx)) if len(dx) > 0 else 1e-3,
        'dy_min': float(np.min(dy)) if len(dy) > 0 else 1e-3,
        'nx': len(x),
        'ny': len(y)
    }
