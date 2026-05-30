
import numpy as np


class CylindricalCavity:

    def __init__(self, radius, height, epsilon_r=1.0, mu_r=1.0, sigma=0.0):
        if radius <= 0 or height <= 0:
            raise ValueError("半径和高度必须为正")
        self.radius = radius
        self.height = height
        self.epsilon_r = epsilon_r
        self.mu_r = mu_r
        self.sigma = sigma

    def volume(self):
        return np.pi * self.radius ** 2 * self.height

    def surface_area(self):
        return 2.0 * np.pi * self.radius ** 2 + 2.0 * np.pi * self.radius * self.height

    def te_cutoff_frequency(self, m, n, p, c=3e8):

        j_prime_zeros = {
            (0, 1): 3.8317, (0, 2): 7.0156,
            (1, 1): 1.8412, (1, 2): 5.3314,
            (2, 1): 3.0542, (2, 2): 6.7061,
        }
        key = (m, n)
        if key not in j_prime_zeros:

            x_mn = (m + 2 * n - 0.5) * np.pi / 2.0
        else:
            x_mn = j_prime_zeros[key]

        k_r = x_mn / self.radius
        k_z = p * np.pi / self.height
        k = np.sqrt(k_r ** 2 + k_z ** 2)
        return c * k / (2.0 * np.pi)

    def tm_cutoff_frequency(self, m, n, p, c=3e8):
        j_zeros = {
            (0, 1): 2.4048, (0, 2): 5.5201,
            (1, 1): 3.8317, (1, 2): 7.0156,
            (2, 1): 5.1356, (2, 2): 8.4172,
        }
        key = (m, n)
        if key not in j_zeros:
            x_mn = (m + 2 * n - 0.25) * np.pi
        else:
            x_mn = j_zeros[key]

        k_r = x_mn / self.radius
        k_z = p * np.pi / self.height
        k = np.sqrt(k_r ** 2 + k_z ** 2)
        return c * k / (2.0 * np.pi)

    def is_inside(self, x, y, z, center=(0, 0, 0)):
        cx, cy, cz = center
        r_sq = (x - cx) ** 2 + (y - cy) ** 2
        r = np.sqrt(r_sq)
        in_radius = r <= self.radius
        in_height = (z >= cz) & (z <= cz + self.height)
        return in_radius & in_height


class CoaxialCavity:

    def __init__(self, a, b, length, epsilon_r=1.0):
        if a <= 0 or b <= a or length <= 0:
            raise ValueError("几何参数必须满足: 0 < a < b, length > 0")
        self.a = a
        self.b = b
        self.length = length
        self.epsilon_r = epsilon_r

    def characteristic_impedance(self, mu_r=1.0):
        from physics_constants import ETA_0
        return ETA_0 / (2.0 * np.pi) * np.sqrt(mu_r / self.epsilon_r) * np.log(self.b / self.a)

    def capacitance_per_unit_length(self, epsilon_0=8.854e-12):
        return 2.0 * np.pi * epsilon_0 * self.epsilon_r / np.log(self.b / self.a)

    def is_inside(self, x, y, z, center=(0, 0, 0)):
        cx, cy, cz = center
        r_sq = (x - cx) ** 2 + (y - cy) ** 2
        r = np.sqrt(r_sq)
        in_radial = (r >= self.a) & (r <= self.b)
        in_axial = (z >= cz) & (z <= cz + self.length)
        return in_radial & in_axial


class CircleSegmentDielectric:

    def __init__(self, r, theta, height, epsilon_r=10.0, center=(0, 0)):
        self.r = r
        self.theta = theta
        self.height = height
        self.epsilon_r = epsilon_r
        self.center = center

    def area(self):
        return self.r ** 2 * (self.theta - np.sin(self.theta)) / 2.0

    def centroid(self):
        if self.theta < 1e-10:
            return (0.0, 0.0)
        d = 4.0 * self.r * (np.sin(self.theta / 2.0)) ** 3 / (3.0 * (self.theta - np.sin(self.theta)))

        return (d, 0.0)

    def is_inside(self, x, y, z, z_bottom=0.0):
        cx, cy = self.center
        dx = x - cx
        dy = y - cy
        r_p = np.sqrt(dx ** 2 + dy ** 2)
        in_radius = r_p <= self.r

        angle = np.arctan2(dy, dx)
        in_angle = np.abs(angle) <= self.theta / 2.0
        in_height = (z >= z_bottom) & (z <= z_bottom + self.height)
        return in_radius & in_angle & in_height


class ParametricProfile:

    def __init__(self, profile_points, symmetry='axisymmetric'):
        if len(profile_points) < 3:
            raise ValueError("轮廓至少需要3个点")
        self.profile_points = np.array(profile_points)
        self.symmetry = symmetry

    def radius_at_z(self, z):
        zs = self.profile_points[:, 1]
        rs = self.profile_points[:, 0]
        if z < zs.min() or z > zs.max():
            return 0.0
        return np.interp(z, zs, rs)

    def is_inside(self, x, y, z):
        r_max = self.radius_at_z(z)
        r = np.sqrt(x ** 2 + y ** 2)
        return r <= r_max


def create_corrugated_wall_profile(R_base, depth, period, n_periods, z_start=0.0):
    points = []
    n_points_per_period = 8
    for i in range(n_periods * n_points_per_period + 1):
        t = i / n_points_per_period
        z = z_start + t * period

        r = R_base + depth * np.cos(2.0 * np.pi * t)
        r = max(r, 0.01 * R_base)
        points.append((r, z))
    return ParametricProfile(points, symmetry='axisymmetric')


def assign_material_properties(grid, shapes):
    from physics_constants import EPSILON_0, MU_0

    if hasattr(grid, 'X'):
        nx, ny, nz = grid.nx, grid.ny, grid.nz
        epsilon = np.ones((nx, ny, nz)) * EPSILON_0
        mu = np.ones((nx, ny, nz)) * MU_0
        sigma = np.zeros((nx, ny, nz))
        X, Y, Z = grid.X, grid.Y, grid.Z
    else:
        nr, nz = grid.nr, grid.nz
        epsilon = np.ones((nr, nz)) * EPSILON_0
        mu = np.ones((nr, nz)) * MU_0
        sigma = np.zeros((nr, nz))
        X = grid.R_grid
        Y = np.zeros_like(X)
        Z = grid.Z_grid

    for shape in shapes:
        if hasattr(shape, 'is_inside'):
            mask = shape.is_inside(X, Y, Z)
            if hasattr(shape, 'epsilon_r'):
                epsilon[mask] = shape.epsilon_r * EPSILON_0
            if hasattr(shape, 'mu_r'):
                mu[mask] = shape.mu_r * MU_0
            if hasattr(shape, 'sigma'):
                sigma[mask] = shape.sigma

    return epsilon, mu, sigma
