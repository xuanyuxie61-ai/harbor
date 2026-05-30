
import numpy as np
import math
from numerics_core import safe_divide
from typing import Tuple, List, Dict, Optional, Callable






def compute_okubo_weiss(u: np.ndarray, v: np.ndarray,
                        dx: float, dy: float) -> np.ndarray:
    Ny, Nx = u.shape
    if u.shape != v.shape:
        raise ValueError("u and v must have same shape")


    ux = np.zeros_like(u)
    uy = np.zeros_like(u)
    vx = np.zeros_like(v)
    vy = np.zeros_like(v)

    ux[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2.0 * dx)
    uy[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2.0 * dy)
    vx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (2.0 * dx)
    vy[1:-1, :] = (v[2:, :] - v[:-2, :]) / (2.0 * dy)


    ux[:, 0] = (u[:, 1] - u[:, -1]) / (2.0 * dx)
    ux[:, -1] = (u[:, 0] - u[:, -2]) / (2.0 * dx)
    uy[0, :] = (u[1, :] - u[-1, :]) / (2.0 * dy)
    uy[-1, :] = (u[0, :] - u[-2, :]) / (2.0 * dy)
    vx[:, 0] = (v[:, 1] - v[:, -1]) / (2.0 * dx)
    vx[:, -1] = (v[:, 0] - v[:, -2]) / (2.0 * dx)
    vy[0, :] = (v[1, :] - v[-1, :]) / (2.0 * dy)
    vy[-1, :] = (v[0, :] - v[-2, :]) / (2.0 * dy)

    Sn = ux - vy
    Ss = vx + uy
    omega = vx - uy

    W = Sn ** 2 + Ss ** 2 - omega ** 2
    return W


def detect_vortex_cores(W: np.ndarray, threshold_factor: float = 0.2) -> Tuple[np.ndarray, float]:
    W_mean = np.mean(W)
    W_std = np.std(W)
    threshold = W_mean - threshold_factor * W_std
    cores = W < threshold
    return cores, threshold






def polygon_moments(vertices: np.ndarray, max_order: int = 2) -> Dict[Tuple[int, int], float]:
    n = vertices.shape[0]
    if n < 3:
        return {(0, 0): 0.0}

    moments = {}
    for p in range(max_order + 1):
        for q in range(max_order + 1 - p):
            moments[(p, q)] = 0.0

    for i in range(n):
        x0, y0 = vertices[i]
        x1, y1 = vertices[(i + 1) % n]
        dx = x1 - x0
        dy = y1 - y0
        cross = x0 * dy - y0 * dx



        for p in range(max_order + 1):
            for q in range(max_order + 1 - p):
                val = 0.0
                for a in range(p + 1):
                    for b in range(q + 1):
                        coeff = (math.comb(p, a) * math.comb(q, b) *
                                 (x0 ** (p - a)) * (dx ** a) *
                                 (y0 ** (q - b)) * (dy ** b))
                        power = a + b
                        if power >= 0:
                            coeff /= (power + 1)
                        val += coeff
                moments[(p, q)] += val * cross


    if moments[(0, 0)] < 0:
        for key in moments:
            moments[key] *= -1

    return moments


def polygon_central_moments(vertices: np.ndarray, max_order: int = 2) -> Dict[Tuple[int, int], float]:
    M = polygon_moments(vertices, max_order)
    area = M[(0, 0)]
    if area < 1e-15:
        return M

    xc = M[(1, 0)] / area
    yc = M[(0, 1)] / area

    mu = {}
    for p in range(max_order + 1):
        for q in range(max_order + 1 - p):
            val = 0.0
            for i in range(p + 1):
                for j in range(q + 1):
                    val += (math.comb(p, i) * math.comb(q, j) *
                            ((-xc) ** (p - i)) * ((-yc) ** (q - j)) *
                            M.get((i, j), 0.0))
            mu[(p, q)] = val

    return mu


def polygon_inertia_tensor(vertices: np.ndarray) -> np.ndarray:
    mu = polygon_central_moments(vertices, max_order=2)
    return np.array([[mu[(0, 2)], -mu[(1, 1)]],
                     [-mu[(1, 1)], mu[(2, 0)]]])


def polygon_eccentricity(vertices: np.ndarray) -> float:
    I = polygon_inertia_tensor(vertices)
    eigvals = np.linalg.eigvalsh(I)
    if eigvals[1] < 1e-15:
        return 0.0
    return np.sqrt(1.0 - eigvals[0] / eigvals[1])






def signed_distance_circle(p: np.ndarray, xc: float, yc: float, r: float) -> np.ndarray:
    return np.sqrt((p[:, 0] - xc) ** 2 + (p[:, 1] - yc) ** 2) - r


def signed_distance_rectangle(p: np.ndarray, x1: float, y1: float,
                              x2: float, y2: float) -> np.ndarray:
    dx = np.maximum(np.maximum(x1 - p[:, 0], p[:, 0] - x2), 0.0)
    dy = np.maximum(np.maximum(y1 - p[:, 1], p[:, 1] - y2), 0.0)
    return np.sqrt(dx ** 2 + dy ** 2)


def signed_distance_union(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    return np.minimum(d1, d2)


def signed_distance_difference(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    return np.maximum(d1, -d2)


def simple_mesh_2d(fd: Callable, fh: Callable, bbox: Tuple[float, float, float, float],
                   h0: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import Delaunay

    xmin, ymin, xmax, ymax = bbox

    nx = int(np.ceil((xmax - xmin) / h0))
    ny = int(np.ceil((ymax - ymin) / h0))
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)
    p_init = np.column_stack([X.flatten(), Y.flatten()])


    d = fd(p_init)
    p = p_init[d < 0]

    if len(p) < 3:
        return p, np.zeros((0, 3), dtype=int)


    tri = Delaunay(p)
    t = tri.simplices


    pc = (p[t[:, 0]] + p[t[:, 1]] + p[t[:, 2]]) / 3.0
    t = t[fd(pc) < 0]

    return p, t






class Eddy:

    def __init__(self, label: int, mask: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray,
                 q: np.ndarray, u: np.ndarray, v: np.ndarray):
        self.label = label
        self.mask = mask
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.dx = x_grid[1] - x_grid[0]
        self.dy = y_grid[1] - y_grid[0]


        self._compute_geometry()

        self._compute_physics(q, u, v)

    def _compute_geometry(self):

        try:
            from skimage.measure import find_contours
            contours = find_contours(self.mask, 0.5)
        except Exception:
            contours = self._simple_contour()

        if contours:

            longest = max(contours, key=len)

            vertices = np.zeros_like(longest)
            vertices[:, 0] = self.x_grid[np.clip(longest[:, 1].astype(int), 0, len(self.x_grid) - 1)]
            vertices[:, 1] = self.y_grid[np.clip(longest[:, 0].astype(int), 0, len(self.y_grid) - 1)]
            self.vertices = vertices

            M = polygon_moments(vertices, max_order=2)
            self.area = M[(0, 0)]
            if self.area > 1e-15:
                self.centroid_x = M[(1, 0)] / self.area
                self.centroid_y = M[(0, 1)] / self.area
            else:
                self.centroid_x = np.mean(self.x_grid[self.mask.any(axis=0)])
                self.centroid_y = np.mean(self.y_grid[self.mask.any(axis=1)])
            self.eccentricity = polygon_eccentricity(vertices)
        else:
            self.vertices = np.zeros((0, 2))
            self.area = 0.0
            self.centroid_x = 0.0
            self.centroid_y = 0.0
            self.eccentricity = 0.0

    def _simple_contour(self) -> List[np.ndarray]:
        Ny, Nx = self.mask.shape
        boundary = np.zeros_like(self.mask, dtype=bool)
        for j in range(1, Ny - 1):
            for i in range(1, Nx - 1):
                if self.mask[j, i]:
                    if not (self.mask[j - 1, i] and self.mask[j + 1, i] and
                            self.mask[j, i - 1] and self.mask[j, i + 1]):
                        boundary[j, i] = True

        points = np.column_stack(np.where(boundary))
        if len(points) < 3:
            return []

        return [points[:, [1, 0]].astype(float)]

    def _compute_physics(self, q: np.ndarray, u: np.ndarray, v: np.ndarray):
        if np.any(self.mask):
            self.mean_vorticity = float(np.mean(q[self.mask]))
            self.max_vorticity = float(np.max(np.abs(q[self.mask])))
            self.kinetic_energy = float(0.5 * np.sum((u[self.mask] ** 2 + v[self.mask] ** 2)) *
                                        self.dx * self.dy)
            self.circulation = float(np.sum(q[self.mask]) * self.dx * self.dy)
        else:
            self.mean_vorticity = 0.0
            self.max_vorticity = 0.0
            self.kinetic_energy = 0.0
            self.circulation = 0.0

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "centroid": (float(self.centroid_x), float(self.centroid_y)),
            "area": float(self.area),
            "eccentricity": float(self.eccentricity),
            "mean_vorticity": float(self.mean_vorticity),
            "max_vorticity": float(self.max_vorticity),
            "kinetic_energy": float(self.kinetic_energy),
            "circulation": float(self.circulation)
        }


def extract_eddies(q: np.ndarray, u: np.ndarray, v: np.ndarray,
                   x_grid: np.ndarray, y_grid: np.ndarray,
                   threshold_factor: float = 0.2) -> List[Eddy]:
    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]
    W = compute_okubo_weiss(u, v, dx, dy)
    cores, _ = detect_vortex_cores(W, threshold_factor)


    Ny, Nx = cores.shape
    labels = np.zeros_like(cores, dtype=int)
    current_label = 0

    for j in range(Ny):
        for i in range(Nx):
            if cores[j, i] and labels[j, i] == 0:
                current_label += 1

                stack = [(j, i)]
                while stack:
                    cj, ci = stack.pop()
                    if 0 <= cj < Ny and 0 <= ci < Nx and cores[cj, ci] and labels[cj, ci] == 0:
                        labels[cj, ci] = current_label
                        stack.extend([(cj - 1, ci), (cj + 1, ci), (cj, ci - 1), (cj, ci + 1)])

    eddies = []
    for lbl in range(1, current_label + 1):
        mask = labels == lbl

        if np.sum(mask) < 4:
            continue
        eddy = Eddy(lbl, mask, x_grid, y_grid, q, u, v)
        if eddy.area > 1e-10:
            eddies.append(eddy)

    return eddies


if __name__ == "__main__":

    verts = np.array([[0, 0], [2, 0], [2, 1], [0, 1]])
    M = polygon_moments(verts, max_order=2)
    print("Rectangle moments:", M)
    print("Area:", M[(0, 0)])


    x = np.linspace(0, 2*np.pi, 64)
    y = np.linspace(0, 2*np.pi, 64)
    X, Y = np.meshgrid(x, y)
    u = -np.sin(X) * np.cos(Y)
    v = np.cos(X) * np.sin(Y)
    W = compute_okubo_weiss(u, v, x[1]-x[0], y[1]-y[0])
    print("OW range:", np.min(W), np.max(W))
