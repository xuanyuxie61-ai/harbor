
import numpy as np


class GridGenerator:

    @staticmethod
    def rectangular_grid(nx, ny, xlim=(-5.0, 5.0), ylim=(-5.0, 5.0)):
        if nx < 2 or ny < 2:
            raise ValueError("nx, ny 必须至少为 2。")
        x = np.linspace(xlim[0], xlim[1], nx)
        y = np.linspace(ylim[0], ylim[1], ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        dx = (xlim[1] - xlim[0]) / (nx - 1) if nx > 1 else 1.0
        dy = (ylim[1] - ylim[0]) / (ny - 1) if ny > 1 else 1.0
        return X, Y, dx, dy

    @staticmethod
    def polar_grid(r_min, r_max, nr, n_ang, center=(0.0, 0.0)):
        if r_min < 0 or r_max <= r_min:
            raise ValueError("径向范围无效。")
        if nr < 1 or n_ang < 3:
            raise ValueError("nr ≥ 1, n_ang ≥ 3。")

        r = np.linspace(r_min, r_max, nr)
        theta = np.linspace(0.0, 2.0 * np.pi, n_ang, endpoint=False)
        R, Theta = np.meshgrid(r, theta, indexing='ij')
        X = center[0] + R * np.cos(Theta)
        Y = center[1] + R * np.sin(Theta)
        return R, Theta, X, Y

    @staticmethod
    def triangular_grid(nx, ny, xlim=(-5.0, 5.0), ylim=(-5.0, 5.0)):
        if nx < 2 or ny < 2:
            raise ValueError("nx, ny 必须至少为 2。")
        x = np.linspace(xlim[0], xlim[1], nx)
        y = np.linspace(ylim[0], ylim[1], ny)
        X, Y = np.meshgrid(x, y, indexing='ij')

        nodes = []
        node_id = {}
        idx = 0
        for i in range(nx):
            for j in range(ny):
                nodes.append([X[i, j], Y[i, j]])
                node_id[(i, j)] = idx
                idx += 1
        nodes = np.array(nodes)

        triangles = []
        for i in range(nx - 1):
            for j in range(ny - 1):
                n00 = node_id[(i, j)]
                n10 = node_id[(i + 1, j)]
                n01 = node_id[(i, j + 1)]
                n11 = node_id[(i + 1, j + 1)]

                triangles.append([n00, n10, n11])
                triangles.append([n00, n11, n01])

        return nodes, np.array(triangles, dtype=int)


class BoundaryTracer:

    HEX_STEPS = np.array([
        [0, 1], [1, 0], [1, -1], [0, -1], [-1, 0], [-1, 1]
    ], dtype=int)

    SQR_STEPS = np.array([
        [0, 1], [1, 0], [0, -1], [-1, 0]
    ], dtype=int)

    def __init__(self, grid_type='hex'):
        if grid_type not in ('hex', 'square'):
            raise ValueError("grid_type 必须是 'hex' 或 'square'。")
        self.grid_type = grid_type
        self.steps = self.HEX_STEPS if grid_type == 'hex' else self.SQR_STEPS

    def trace_boundary(self, mask, start_ij):
        mask = np.asarray(mask, dtype=bool)
        nx, ny = mask.shape
        i, j = start_ij
        if not (0 <= i < nx and 0 <= j < ny):
            raise ValueError("起点超出网格范围。")
        if not mask[i, j]:
            raise ValueError("起点必须位于膜区域内。")

        boundary_word = []
        path = [(i, j)]
        visited = set()
        visited.add((i, j))


        current_dir = 0
        max_steps = 4 * nx * ny
        for _ in range(max_steps):
            found = False
            for d in range(len(self.steps)):
                nd = (current_dir + d) % len(self.steps)
                di, dj = self.steps[nd]
                ni, nj = i + di, j + dj
                if 0 <= ni < nx and 0 <= nj < ny and mask[ni, nj]:
                    if (ni, nj) not in visited or len(path) > 2:
                        boundary_word.append(int(nd))
                        i, j = ni, nj
                        path.append((i, j))
                        visited.add((i, j))
                        current_dir = (nd - 1) % len(self.steps)
                        found = True
                        break
            if not found or (len(path) > 2 and path[-1] == path[0]):
                break

        return boundary_word, np.array(path, dtype=int)

    def word_to_polygon(self, word, start=(0, 0)):
        pts = [np.array(start, dtype=float)]
        p = np.array(start, dtype=float)
        for d in word:
            if d < 0 or d >= len(self.steps):
                continue
            p = p + self.steps[d].astype(float)
            pts.append(p.copy())
        return np.array(pts)

    def pram_like_boundary(self, scale=1.0):


        word = (
            [1] * (8 * scale) +
            [0] * (8 * scale) +
            [4] * (14 * scale) +
            [3] * (8 * scale) +
            [2] * (4 * scale) +
            [5] * (4 * scale)
        )
        return [int(w) for w in word]

    def equilateral_triangle_boundary(self, side_length=5):
        if side_length <= 0:
            raise ValueError("边长必须为正。")

        word = []
        for _ in range(side_length):
            word.extend([1, 0])
        for _ in range(side_length):
            word.extend([4, 3])
        for _ in range(side_length):
            word.extend([2, 2])
        return word

    def compute_perimeter_and_area(self, word):
        poly = self.word_to_polygon(word)
        if len(poly) < 3:
            return 0.0, 0.0

        diffs = np.diff(poly, axis=0, append=poly[:1])
        perimeter = np.sum(np.sqrt(np.sum(diffs ** 2, axis=1)))

        x = poly[:, 0]
        y = poly[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        return float(perimeter), float(area)


def membrane_surface_metric(nodes, triangles):
    if len(triangles) == 0:
        return 1.0, 0.0, 1.0

    E_total = F_total = G_total = 0.0
    for tri in triangles:
        p0, p1, p2 = nodes[tri[0]], nodes[tri[1]], nodes[tri[2]]

        ru = p1 - p0
        rv = p2 - p0
        E_total += np.dot(ru, ru)
        F_total += np.dot(ru, rv)
        G_total += np.dot(rv, rv)

    n_tri = len(triangles)
    return E_total / n_tri, F_total / n_tri, G_total / n_tri
