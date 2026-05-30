
import numpy as np


def line_grid(n, a, b, centering=1):
    if n < 1:
        raise ValueError("n must be >= 1")
    x = np.zeros(n)
    for j in range(n):
        jj = j + 1
        if centering == 1:
            if n == 1:
                x[j] = 0.5 * (a + b)
            else:
                x[j] = ((n - jj) * a + (jj - 1) * b) / (n - 1)
        elif centering == 2:
            x[j] = ((n - jj + 1) * a + jj * b) / (n + 1)
        elif centering == 3:
            x[j] = ((n - jj + 1) * a + (jj - 1) * b) / n
        elif centering == 4:
            x[j] = ((n - jj) * a + jj * b) / n
        elif centering == 5:
            x[j] = ((2 * n - 2 * jj + 1) * a + (2 * jj - 1) * b) / (2 * n)
        else:
            raise ValueError(f"Invalid centering: {centering}")
    return x


def flamelet_stretched_grid(n, z_st, stretch_factor=3.0):
    xi = line_grid(n, 0.0, 1.0, centering=1)

    Z = z_st + (1.0 / np.pi) * np.arctan(
        stretch_factor * np.tan(np.pi * (xi - 0.5))
    )

    Z = np.clip(Z, 0.0, 1.0)
    return Z


class CVTAdaptiveMesh2D:

    def __init__(self, n_generators, n_samples, density_func=None,
                 x_bounds=(-1.0, 1.0), y_bounds=(-1.0, 1.0)):
        if n_generators < 3:
            raise ValueError("n_generators must be >= 3")
        self.n = n_generators
        self.n_samples = max(10, n_samples)
        self.density_func = density_func
        self.xmin, self.xmax = x_bounds
        self.ymin, self.ymax = y_bounds

    def _generate_samples(self):
        eps = 1e-6
        sx = np.linspace(self.xmin + eps, self.xmax - eps, self.n_samples)
        sy = np.linspace(self.ymin + eps, self.ymax - eps, self.n_samples)
        SX, SY = np.meshgrid(sx, sy, indexing='ij')
        return SX.flatten(), SY.flatten()

    def _compute_density(self, x, y):
        if self.density_func is None:
            return np.ones_like(x)
        rho = self.density_func(x, y)

        return np.clip(rho, 0.01, 100.0)

    def lloyd_iteration(self, max_iter=50, tol=1e-6):

        g = np.zeros((self.n, 2))
        g[:, 0] = np.random.uniform(self.xmin, self.xmax, self.n)
        g[:, 1] = np.random.uniform(self.ymin, self.ymax, self.n)

        sx, sy = self._generate_samples()
        rho = self._compute_density(sx, sy)


        r = rho**2

        energy_history = []
        motion_history = []

        for it in range(max_iter):


            dist2 = ((sx[:, None] - g[None, :, 0])**2
                     + (sy[:, None] - g[None, :, 1])**2)
            nearest = np.argmin(dist2, axis=1)


            g_new = np.zeros_like(g)
            mass = np.zeros(self.n)
            for i in range(self.n):
                mask = (nearest == i)
                if np.any(mask):
                    mass[i] = np.sum(r[mask])
                    g_new[i, 0] = np.sum(r[mask] * sx[mask]) / mass[i]
                    g_new[i, 1] = np.sum(r[mask] * sy[mask]) / mass[i]
                else:

                    g_new[i, 0] = np.random.uniform(self.xmin, self.xmax)
                    g_new[i, 1] = np.random.uniform(self.ymin, self.ymax)


            energy = np.sum(r * dist2[np.arange(len(sx)), nearest]) / self.n_samples
            motion = np.mean(np.sum((g_new - g)**2, axis=1))

            energy_history.append(energy)
            motion_history.append(motion)

            g = g_new.copy()

            if motion < tol:
                break

        return g, energy_history, motion_history

    def extract_flame_resolved_grid(self, nx_dns, ny_dns):
        x_grid = np.linspace(self.xmin, self.xmax, nx_dns)
        y_grid = np.linspace(self.ymin, self.ymax, ny_dns)
        X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')


        refinement_mask = np.zeros((nx_dns, ny_dns), dtype=bool)
        dx = (self.xmax - self.xmin) / nx_dns
        dy = (self.ymax - self.ymin) / ny_dns
        radius = 3.0 * max(dx, dy)

        for i in range(self.n):
            dist = np.sqrt((X - self.g[i, 0])**2 + (Y - self.g[i, 1])**2)
            refinement_mask |= (dist < radius)

        return x_grid, y_grid, refinement_mask


def build_density_from_scalar_gradient(X, Y, scalar_field, grad_threshold=0.1):

    dx = X[1, 0] - X[0, 0]
    dy = Y[0, 1] - Y[0, 0]
    dZdx = np.gradient(scalar_field, axis=0) / dx
    dZdy = np.gradient(scalar_field, axis=1) / dy
    grad_mag = np.sqrt(dZdx**2 + dZdy**2)


    grad_max = np.max(grad_mag)
    if grad_max < 1e-12:
        return lambda x, y: np.ones_like(np.atleast_1d(x))


    from scipy.interpolate import RegularGridInterpolator
    try:
        interp = RegularGridInterpolator(
            (X[:, 0], Y[0, :]), grad_mag, bounds_error=False, fill_value=0.0
        )

        def density_func(x, y):
            pts = np.column_stack([np.atleast_1d(x), np.atleast_1d(y)])
            vals = interp(pts)
            return 1.0 + 10.0 * vals / grad_max

        return density_func
    except Exception:

        def density_func(x, y):
            return np.ones_like(np.atleast_1d(x))
        return density_func
