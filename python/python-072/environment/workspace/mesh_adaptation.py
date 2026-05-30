
import numpy as np


class MeshAdaptation:

    def __init__(self, nx, ny, x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0):
        self.nx = nx
        self.ny = ny
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

        self.dx_base = (x_max - x_min) / (nx - 1)
        self.dy_base = (y_max - y_min) / (ny - 1)

    def compute_error_indicator(self, phi):
        nx, ny = phi.shape
        dx = (self.x_max - self.x_min) / (nx - 1)
        dy = (self.y_max - self.y_min) / (ny - 1)

        grad_x = np.zeros_like(phi)
        grad_y = np.zeros_like(phi)

        grad_x[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2.0 * dx)
        grad_y[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2.0 * dy)

        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        h_local = np.sqrt(dx ** 2 + dy ** 2)

        return grad_mag * h_local

    def mark_refinement(self, phi, threshold_ratio=0.5):
        eta = self.compute_error_indicator(phi)
        max_eta = np.max(eta)
        if max_eta < 1e-14:
            return np.zeros_like(phi, dtype=bool)

        threshold = threshold_ratio * max_eta
        return eta > threshold

    def dynamic_programming_mesh_distribution(self, n_total, error_funcs, regions):
        n_min = 2


        dp = np.full((regions + 1, n_total + 1), np.inf)
        dp[0, 0] = 0.0


        decision = np.zeros((regions + 1, n_total + 1), dtype=int)

        for i in range(1, regions + 1):
            for j in range(n_total + 1):
                for k in range(n_min, j + 1):
                    err = error_funcs[i - 1](k)
                    if dp[i - 1, j - k] + err < dp[i, j]:
                        dp[i, j] = dp[i - 1, j - k] + err
                        decision[i, j] = k


        distribution = []
        remaining = n_total
        for i in range(regions, 0, -1):
            k = decision[i, remaining]
            distribution.append(k)
            remaining -= k

        distribution.reverse()
        return distribution

    def interface_focused_mesh(self, phi, refinement_level=2):
        nx, ny = phi.shape


        interface_mask = np.abs(phi) < 0.8


        x_uniform = np.linspace(self.x_min, self.x_max, nx)
        y_uniform = np.linspace(self.y_min, self.y_max, ny)


        x_interface_indices = np.where(np.any(interface_mask, axis=1))[0]
        y_interface_indices = np.where(np.any(interface_mask, axis=0))[0]

        if len(x_interface_indices) == 0:
            return x_uniform, y_uniform

        x_min_int = x_uniform[x_interface_indices[0]]
        x_max_int = x_uniform[x_interface_indices[-1]]
        y_min_int = y_uniform[y_interface_indices[0]]
        y_max_int = y_uniform[y_interface_indices[-1]]


        x_coords = []
        for x in x_uniform:
            if x_min_int <= x <= x_max_int:

                x_coords.append(x)
                for r in range(1, refinement_level):
                    dx_fine = self.dx_base / (2 ** r)
                    x_coords.append(x + dx_fine)
            else:
                x_coords.append(x)

        y_coords = []
        for y in y_uniform:
            if y_min_int <= y <= y_max_int:
                y_coords.append(y)
                for r in range(1, refinement_level):
                    dy_fine = self.dy_base / (2 ** r)
                    y_coords.append(y + dy_fine)
            else:
                y_coords.append(y)

        return np.array(sorted(set(np.round(x_coords, 10)))), \
               np.array(sorted(set(np.round(y_coords, 10))))

    def compute_mesh_quality(self, phi):
        nx, ny = phi.shape
        dx = (self.x_max - self.x_min) / (nx - 1)
        dy = (self.y_max - self.y_min) / (ny - 1)


        interface_mask = np.abs(phi) < 0.8


        h_interface = np.sqrt(dx ** 2 + dy ** 2)
        if np.any(interface_mask):

            q_res = h_interface / 0.01
        else:
            q_res = np.inf


        eta = self.compute_error_indicator(phi)
        max_eta = np.max(eta)
        if max_eta > 1e-14:
            q_grad = np.mean(eta) / max_eta
        else:
            q_grad = 1.0

        return {
            'interface_resolution': q_res,
            'gradient_adaptivity': q_grad,
            'max_error_indicator': max_eta
        }


class TriangleGridTopology:

    @staticmethod
    def quadrilateral_to_triangles(nx, ny):

        vertices = []
        for i in range(nx):
            for j in range(ny):
                vertices.append((i, j))
        vertices = np.array(vertices)


        triangles = []
        for i in range(nx - 1):
            for j in range(ny - 1):
                v00 = i * ny + j
                v10 = (i + 1) * ny + j
                v01 = i * ny + (j + 1)
                v11 = (i + 1) * ny + (j + 1)

                triangles.append([v00, v10, v01])
                triangles.append([v10, v11, v01])

        return vertices, np.array(triangles)

    @staticmethod
    def triangle_quality(p1, p2, p3):
        a = np.linalg.norm(p2 - p3)
        b = np.linalg.norm(p1 - p3)
        c = np.linalg.norm(p1 - p2)

        if a * b * c < 1e-14:
            return 0.0

        quality = (b + c - a) * (c + a - b) * (a + b - c) / (a * b * c)
        return quality
