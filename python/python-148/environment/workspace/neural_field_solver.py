
import numpy as np
from utils import sigmoid_activation, gauss_legendre_nodes_weights, rk4_step


def generate_square_grid(xlim=(-1, 1), ylim=(-1, 1), nx=64, ny=64, centering='cell'):
    if centering == 'vertex':
        x = np.linspace(xlim[0], xlim[1], nx)
        y = np.linspace(ylim[0], ylim[1], ny)
    elif centering == 'cell':
        dx = (xlim[1] - xlim[0]) / nx
        dy = (ylim[1] - ylim[0]) / ny
        x = np.linspace(xlim[0] + 0.5 * dx, xlim[1] - 0.5 * dx, nx)
        y = np.linspace(ylim[0] + 0.5 * dy, ylim[1] - 0.5 * dy, ny)
    elif centering == 'half':
        dx = (xlim[1] - xlim[0]) / nx
        dy = (ylim[1] - ylim[0]) / ny
        x = np.linspace(xlim[0] + 0.25 * dx, xlim[1] - 0.75 * dx, nx)
        y = np.linspace(ylim[0] + 0.25 * dy, ylim[1] - 0.75 * dy, ny)
    else:
        raise ValueError(f"Unknown centering: {centering}")
    X, Y = np.meshgrid(x, y, indexing='ij')
    return X, Y


def generate_polygon_grid_points(vertices, n_subdiv=8):
    vertices = np.asarray(vertices, dtype=float)
    nv = len(vertices)
    if nv < 3:
        raise ValueError("Polygon must have at least 3 vertices")
    centroid = np.mean(vertices, axis=0)
    points = [centroid.copy()]
    for k in range(nv):
        v1 = centroid
        v2 = vertices[k]
        v3 = vertices[(k + 1) % nv]
        for i in range(n_subdiv + 1):
            for j in range(n_subdiv + 1 - i):
                l = n_subdiv - i - j
                lam1 = i / n_subdiv
                lam2 = j / n_subdiv
                lam3 = l / n_subdiv
                p = lam1 * v1 + lam2 * v2 + lam3 * v3

                if not (i == n_subdiv and j == 0 and l == 0):
                    points.append(p)
    return np.array(points, dtype=float)


def mexican_hat_kernel_2d(dx, dy, sigma_e=0.1, sigma_i=0.2, A_e=1.0, A_i=0.5):
    r2 = dx ** 2 + dy ** 2
    return A_e * np.exp(-r2 / (2.0 * sigma_e ** 2)) - A_i * np.exp(-r2 / (2.0 * sigma_i ** 2))


class NeuralFieldSolver:

    def __init__(self, X, Y, tau=0.02, sigma_e=0.15, sigma_i=0.3,
                 A_e=1.0, A_i=0.6, theta=0.0, sigma_act=1.0):
        self.X = np.asarray(X, dtype=float)
        self.Y = np.asarray(Y, dtype=float)
        self.nx, self.ny = X.shape
        self.n_points = self.nx * self.ny
        self.tau = tau
        self.sigma_e = sigma_e
        self.sigma_i = sigma_i
        self.A_e = A_e
        self.A_i = A_i
        self.theta = theta
        self.sigma_act = sigma_act

        dx = np.mean(np.diff(X[:, 0])) if self.nx > 1 else 1.0
        dy = np.mean(np.diff(Y[0, :])) if self.ny > 1 else 1.0
        self.dA = abs(dx * dy)

        self._build_kernel_matrix()

    def _build_kernel_matrix(self):
        nx, ny = self.nx, self.ny
        x_flat = self.X.flatten()
        y_flat = self.Y.flatten()
        n = len(x_flat)
        self.K_mat = np.zeros((n, n), dtype=float)
        for i in range(n):
            dx = x_flat[i] - x_flat
            dy = y_flat[i] - y_flat
            self.K_mat[i, :] = mexican_hat_kernel_2d(
                dx, dy, self.sigma_e, self.sigma_i, self.A_e, self.A_i)

        row_sums = np.sum(self.K_mat, axis=1) * self.dA
        max_sum = np.max(np.abs(row_sums))
        if max_sum > 0:
            self.K_mat /= max_sum

    def _rhs(self, u, I_ext):
        n = self.n_points
        Su = sigmoid_activation(u, self.theta, self.sigma_act)

        conv = self.K_mat @ Su * self.dA
        return (-u + conv + I_ext) / self.tau

    def simulate(self, u0, I_ext_func, t_span=(0.0, 1.0), dt=0.001, method='rk4'):
        t0, tf = t_span
        n_steps = int(np.ceil((tf - t0) / dt))
        t = np.linspace(t0, tf, n_steps + 1)
        u = np.asarray(u0, dtype=float).flatten()
        u_hist = np.zeros((n_steps + 1, self.n_points), dtype=float)
        u_hist[0] = u.copy()
        for i in range(n_steps):
            I_ext = I_ext_func(t[i], self.X, self.Y).flatten()
            if method == 'rk4':
                u = rk4_step(lambda ti, ui: self._rhs(ui, I_ext_func(ti, self.X, self.Y).flatten()),
                             t[i], u, dt)
            elif method == 'euler':
                u = u + dt * self._rhs(u, I_ext)
            else:
                raise ValueError(f"Unknown method: {method}")
            u_hist[i + 1] = u.copy()
        return t, u_hist.reshape(n_steps + 1, self.nx, self.ny)

    def compute_spatial_spectrum(self, u_field):
        nx, ny = self.nx, self.ny

        dx = np.mean(np.diff(self.X[:, 0])) if nx > 1 else 1.0
        dy = np.mean(np.diff(self.Y[0, :])) if ny > 1 else 1.0
        fft_u = np.fft.fft2(u_field)
        power = np.abs(fft_u) ** 2
        kx = np.fft.fftfreq(nx, d=dx)
        ky = np.fft.fftfreq(ny, d=dy)
        return kx, ky, power


class NeuralFieldWithGaussQuadrature:

    def __init__(self, domain=(-1.0, 1.0, -1.0, 1.0), n_quad=8, n_grid=32,
                 tau=0.02, sigma_e=0.15, sigma_i=0.3, A_e=1.0, A_i=0.6):
        self.xmin, self.xmax, self.ymin, self.ymax = domain
        self.n_quad = n_quad
        self.n_grid = n_grid
        self.tau = tau
        self.sigma_e = sigma_e
        self.sigma_i = sigma_i
        self.A_e = A_e
        self.A_i = A_i

        self.xi, self.wi = gauss_legendre_nodes_weights(n_quad)

        dx = (self.xmax - self.xmin) / n_grid
        dy = (self.ymax - self.ymin) / n_grid
        self.xc = np.linspace(self.xmin + 0.5 * dx, self.xmax - 0.5 * dx, n_grid)
        self.yc = np.linspace(self.ymin + 0.5 * dy, self.ymax - 0.5 * dy, n_grid)
        self.Xc, self.Yc = np.meshgrid(self.xc, self.yc, indexing='ij')
        self.dx = dx
        self.dy = dy

        self._build_quadrature_kernel()

    def _build_quadrature_kernel(self):
        n = self.n_grid
        nq = self.n_quad
        self.K_quad = np.zeros((n * n, n * n), dtype=float)

        x_offset = self.dx * 0.5 * self.xi
        y_offset = self.dy * 0.5 * self.xi
        for i in range(n):
            for j in range(n):
                idx_target = i * n + j
                xt = self.Xc[i, j]
                yt = self.Yc[i, j]
                for k in range(n):
                    for l in range(n):
                        idx_source = k * n + l
                        xs_center = self.Xc[k, l]
                        ys_center = self.Yc[k, l]

                        integral_val = 0.0
                        for m in range(nq):
                            for n_ in range(nq):
                                xs = xs_center + x_offset[m]
                                ys = ys_center + y_offset[n_]
                                dx_ = xt - xs
                                dy_ = yt - ys
                                k_val = mexican_hat_kernel_2d(
                                    dx_, dy_, self.sigma_e, self.sigma_i, self.A_e, self.A_i)
                                w = 0.25 * self.dx * self.dy * self.wi[m] * self.wi[n_]
                                integral_val += k_val * w
                        self.K_quad[idx_target, idx_source] = integral_val

        row_sums = np.sum(self.K_quad, axis=1)
        max_sum = np.max(np.abs(row_sums))
        if max_sum > 0:
            self.K_quad /= max_sum

    def simulate(self, u0, I_ext_func, t_span=(0.0, 1.0), dt=0.001):
        t0, tf = t_span
        n_steps = int(np.ceil((tf - t0) / dt))
        t = np.linspace(t0, tf, n_steps + 1)
        n = self.n_grid
        u = np.asarray(u0, dtype=float).flatten()
        u_hist = np.zeros((n_steps + 1, n * n), dtype=float)
        u_hist[0] = u.copy()
        for step in range(n_steps):
            I_ext = I_ext_func(t[step], self.Xc, self.Yc).flatten()
            Su = sigmoid_activation(u, theta=0.0, sigma=1.0)
            conv = self.K_quad @ Su
            dudt = (-u + conv + I_ext) / self.tau
            u = u + dt * dudt
            u_hist[step + 1] = u.copy()
        return t, u_hist.reshape(n_steps + 1, n, n)
