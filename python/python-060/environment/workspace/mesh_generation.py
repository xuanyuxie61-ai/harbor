
import numpy as np
from typing import Tuple, List, Optional


class HexagonalGridGenerator:

    def __init__(self, radius: float = 6371.0e3):
        self.radius = radius

    def hexagon_area(self, side_length: float) -> float:
        if side_length <= 0:
            raise ValueError("边长必须为正")
        return 3.0 * np.sqrt(3.0) / 2.0 * side_length ** 2

    def generate_planar_hex_grid(self, center: Tuple[float, float] = (0.0, 0.0),
                                  side_length: float = 1.0e5,
                                  n_rings: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        if side_length <= 0 or n_rings < 0:
            raise ValueError("参数无效")

        points = []
        cx, cy = center


        points.append((cx, cy))


        dx = side_length * np.sqrt(3.0)
        dy = side_length * 1.5

        for ring in range(1, n_rings + 1):
            for i in range(6 * ring):
                angle = np.pi / 3.0 * (i / ring)
                r = side_length * ring * np.sqrt(3.0)
                px = cx + r * np.cos(angle)
                py = cy + r * np.sin(angle)
                points.append((px, py))

        x = np.array([p[0] for p in points])
        y = np.array([p[1] for p in points])
        return x, y

    def hex_lyness_rule(self, rule_id: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if rule_id == 1:

            x = np.array([0.5])
            y = np.array([0.5])
            w = np.array([1.0])
        elif rule_id == 2:

            p = (3.0 - np.sqrt(3.0)) / 6.0
            q = (3.0 + np.sqrt(3.0)) / 6.0
            x = np.array([p, p, q, q])
            y = np.array([p, q, p, q])
            w = np.array([0.25, 0.25, 0.25, 0.25])
        elif rule_id == 3:

            p = np.sqrt(3.0 / 5.0)
            x = np.array([0.5, 0.5 - p/2, 0.5 + p/2, 0.5 - p/2, 0.5 + p/2,
                          0.5 - p/2, 0.5 + p/2, 0.5, 0.5])
            y = np.array([0.5, 0.5 - p/2, 0.5 - p/2, 0.5 + p/2, 0.5 + p/2,
                          0.5, 0.5, 0.5 - p/2, 0.5 + p/2])
            w0 = 64.0 / 81.0
            w1 = 40.0 / 81.0
            w2 = 25.0 / 81.0
            w = np.array([w0, w2, w2, w2, w2, w1, w1, w1, w1]) / 4.0
        else:

            x = np.array([0.5])
            y = np.array([0.5])
            w = np.array([1.0])

        return x, y, w

    def integrate_over_hex(self, f: callable, center: Tuple[float, float],
                           side_length: float, rule_id: int = 3) -> float:
        xi, eta, w = self.hex_lyness_rule(rule_id)
        area = self.hexagon_area(side_length)



        cx, cy = center
        dx = side_length * np.sqrt(3.0)
        x = cx + (xi - 0.5) * dx
        y = cy + (eta - 0.5) * dx

        integral = 0.0
        for i in range(len(w)):
            integral += w[i] * f(x[i], y[i])

        return integral * area


class CVTMeshOptimizer:

    def __init__(self, dim: int = 2, n_generators: int = 100):
        self.dim = dim
        self.n_generators = n_generators

    def find_closest_generator(self, x: np.ndarray,
                               generators: np.ndarray) -> int:
        dists = np.sum((generators - x) ** 2, axis=1)
        return int(np.argmin(dists))

    def cvt_iteration(self, generators: np.ndarray,
                      a: np.ndarray, b: np.ndarray,
                      sample_num: int = 5000,
                      modular: bool = False) -> Tuple[np.ndarray, float]:
        n_gen = generators.shape[0]
        dim = generators.shape[1]

        generator_new = np.zeros_like(generators)
        counts = np.zeros(n_gen)


        np.random.seed(42)
        for _ in range(sample_num):
            x = a + np.random.rand(dim) * (b - a)

            if modular:

                nearest = self._find_closest_modular(x, generators, a, b)
            else:
                nearest = self.find_closest_generator(x, generators)

            generator_new[nearest] += x
            counts[nearest] += 1.0


        for j in range(n_gen):
            if counts[j] > 0:
                generator_new[j] /= counts[j]
            else:
                generator_new[j] = generators[j].copy()


        change = np.max(np.sqrt(np.sum((generator_new - generators) ** 2, axis=1)))


        if modular:
            for j in range(n_gen):
                for i in range(dim):
                    if generator_new[j, i] < a[i]:
                        generator_new[j, i] += b[i] - a[i]
                    elif generator_new[j, i] > b[i]:
                        generator_new[j, i] -= b[i] - a[i]

        return generator_new, change

    def _find_closest_modular(self, x: np.ndarray, generators: np.ndarray,
                              a: np.ndarray, b: np.ndarray) -> int:
        dim = len(x)

        min_dist = float('inf')
        nearest = 0

        for offset in np.ndindex(*([3] * dim)):
            shift = np.array(offset) - 1.0
            x_shifted = x.copy()
            for i in range(dim):
                period = b[i] - a[i]
                x_shifted[i] += shift[i] * period

            for j, gen in enumerate(generators):
                dist = np.sum((x_shifted - gen) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    nearest = j

        return nearest

    def optimize(self, a: np.ndarray, b: np.ndarray,
                 n_iter: int = 50, tol: float = 1e-4,
                 sample_num: int = 5000) -> np.ndarray:
        dim = len(a)
        generators = np.random.rand(self.n_generators, dim)
        for i in range(dim):
            generators[:, i] = a[i] + generators[:, i] * (b[i] - a[i])

        for it in range(n_iter):
            generators_new, change = self.cvt_iteration(
                generators, a, b, sample_num, modular=False)

            if change < tol:
                break
            generators = generators_new

        return generators


class MeshQualityEvaluator:

    def __init__(self):
        pass

    def triangle_area_2d(self, a: np.ndarray, b: np.ndarray,
                         c: np.ndarray) -> float:
        area = 0.5 * abs(
            a[0] * (b[1] - c[1]) +
            b[0] * (c[1] - a[1]) +
            c[0] * (a[1] - b[1])
        )
        return area

    def triangle_angles(self, a: np.ndarray, b: np.ndarray,
                        c: np.ndarray) -> Tuple[float, float, float]:
        ab = np.linalg.norm(b - a)
        bc = np.linalg.norm(c - b)
        ca = np.linalg.norm(a - c)


        if ab < 1e-14 and bc < 1e-14 and ca < 1e-14:
            return 2.0 * np.pi / 3.0, 2.0 * np.pi / 3.0, 2.0 * np.pi / 3.0

        def safe_acos(val):
            return np.arccos(np.clip(val, -1.0, 1.0))

        if ca < 1e-14 or ab < 1e-14:
            a_angle = np.pi
        else:
            a_angle = safe_acos((ca ** 2 + ab ** 2 - bc ** 2) / (2.0 * ca * ab))

        if ab < 1e-14 or bc < 1e-14:
            b_angle = np.pi
        else:
            b_angle = safe_acos((ab ** 2 + bc ** 2 - ca ** 2) / (2.0 * ab * bc))

        if bc < 1e-14 or ca < 1e-14:
            c_angle = np.pi
        else:
            c_angle = safe_acos((bc ** 2 + ca ** 2 - ab ** 2) / (2.0 * bc * ca))

        return a_angle, b_angle, c_angle

    def alpha_measure(self, points: np.ndarray,
                      triangles: List[Tuple[int, int, int]]) -> dict:
        if len(triangles) == 0:
            return {'alpha_min': 0.0, 'alpha_ave': 0.0, 'alpha_area': 0.0}

        alpha_min = float('inf')
        alpha_ave = 0.0
        alpha_area = 0.0
        total_area = 0.0

        for tri in triangles:
            a = points[tri[0]]
            b = points[tri[1]]
            c = points[tri[2]]

            area = self.triangle_area_2d(a, b, c)
            angles = self.triangle_angles(a, b, c)
            min_angle = min(angles)

            alpha_min = min(alpha_min, min_angle)
            alpha_ave += min_angle
            alpha_area += area * min_angle
            total_area += area

        n_tri = len(triangles)
        alpha_ave /= n_tri
        if total_area > 0:
            alpha_area /= total_area


        norm = 3.0 / np.pi
        return {
            'alpha_min': alpha_min * norm,
            'alpha_ave': alpha_ave * norm,
            'alpha_area': alpha_area * norm,
            'n_triangles': n_tri,
            'total_area': total_area
        }

    def evaluate_grid_quality(self, x: np.ndarray, y: np.ndarray) -> dict:
        n = len(x)
        if n < 3:
            return {'uniformity': 0.0, 'coverage': 0.0}


        min_dists = []
        for i in range(n):
            dists = np.sqrt((x - x[i]) ** 2 + (y - y[i]) ** 2)
            dists[i] = float('inf')
            min_dists.append(np.min(dists))

        min_dists = np.array(min_dists)
        mean_dist = np.mean(min_dists)
        std_dist = np.std(min_dists)

        uniformity = 1.0 / (1.0 + std_dist / (mean_dist + 1e-30))


        x_range = np.max(x) - np.min(x)
        y_range = np.max(y) - np.min(y)
        coverage = (n * mean_dist ** 2) / (x_range * y_range + 1e-30)

        return {
            'uniformity': np.clip(uniformity, 0.0, 1.0),
            'coverage': np.clip(coverage, 0.0, 1.0),
            'mean_neighbor_dist': mean_dist,
            'std_neighbor_dist': std_dist
        }


def generate_atmospheric_mesh(n_horizontal: int = 50,
                               n_vertical: int = 40,
                               z_min: float = 10000.0,
                               z_max: float = 50000.0) -> dict:
    hex_gen = HexagonalGridGenerator()
    cvt_opt = CVTMeshOptimizer(dim=2, n_generators=n_horizontal)
    quality = MeshQualityEvaluator()


    n_rings = int(np.sqrt(n_horizontal / 3.0))
    x_hex, y_hex = hex_gen.generate_planar_hex_grid(
        center=(0.0, 0.0), side_length=2.0e5, n_rings=n_rings)


    a = np.array([np.min(x_hex), np.min(y_hex)])
    b = np.array([np.max(x_hex), np.max(y_hex)])

    generators = np.column_stack([x_hex, y_hex])
    for _ in range(10):
        generators_new, change = cvt_opt.cvt_iteration(
            generators, a, b, sample_num=2000)
        generators = generators_new
        if change < 1e-3:
            break


    z_km = np.linspace(z_min / 1000.0, z_max / 1000.0, n_vertical)

    z_refined = z_km + 2.0 * np.exp(-((z_km - 25.0) / 3.0) ** 2)
    z = np.clip(z_refined * 1000.0, z_min, z_max)


    grid_quality = quality.evaluate_grid_quality(generators[:, 0], generators[:, 1])

    return {
        'xy_horizontal': generators,
        'z_vertical': z,
        'n_horizontal': len(generators),
        'n_vertical': n_vertical,
        'horizontal_quality': grid_quality,
        'area_per_cell': hex_gen.hexagon_area(2.0e5) if len(generators) > 0 else 0.0
    }
