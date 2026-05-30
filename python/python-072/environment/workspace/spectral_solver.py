
import numpy as np


class SineTransform:

    @staticmethod
    def dst_1d(f):
        f = np.asarray(f)
        N = len(f)
        if N == 0:
            return np.array([])

        scale = np.sqrt(2.0 / (N + 1))
        b = np.zeros(N)

        for k in range(1, N + 1):
            for j in range(1, N + 1):
                b[k - 1] += np.sin(np.pi * k * j / (N + 1)) * f[j - 1]

        b *= scale
        return b

    @staticmethod
    def idst_1d(b):
        b = np.asarray(b)
        N = len(b)
        if N == 0:
            return np.array([])

        scale = np.sqrt(2.0 / (N + 1))
        f = np.zeros(N)

        for j in range(1, N + 1):
            for k in range(1, N + 1):
                f[j - 1] += np.sin(np.pi * k * j / (N + 1)) * b[k - 1]

        f *= scale
        return f

    @staticmethod
    def dst_2d(f):
        f = np.asarray(f)
        Nx, Ny = f.shape


        b = np.zeros_like(f)
        for i in range(Nx):
            b[i, :] = SineTransform.dst_1d(f[i, :])


        result = np.zeros_like(f)
        for j in range(Ny):
            result[:, j] = SineTransform.dst_1d(b[:, j])

        return result

    @staticmethod
    def idst_2d(b):
        b = np.asarray(b)
        Nx, Ny = b.shape


        f = np.zeros_like(b)
        for j in range(Ny):
            f[:, j] = SineTransform.idst_1d(b[:, j])


        result = np.zeros_like(b)
        for i in range(Nx):
            result[i, :] = SineTransform.idst_1d(f[i, :])

        return result


class SpectralPoissonSolver:

    def __init__(self, nx, ny, Lx=1.0, Ly=1.0):
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly


        self.kx = np.arange(1, nx + 1) * np.pi / Lx
        self.ky = np.arange(1, ny + 1) * np.pi / Ly

    def solve_2d_poisson_dirichlet(self, f):
        if f.shape != (self.nx, self.ny):
            raise ValueError(f"f 的形状必须为 ({self.nx}, {self.ny})")


        f_hat = SineTransform.dst_2d(f)


        u_hat = np.zeros_like(f_hat)
        for i in range(self.nx):
            for j in range(self.ny):
                denom = self.kx[i] ** 2 + self.ky[j] ** 2
                if denom > 1e-14:
                    u_hat[i, j] = f_hat[i, j] / denom
                else:
                    u_hat[i, j] = 0.0


        u = SineTransform.idst_2d(u_hat)

        return u

    def solve_1d_poisson_dirichlet(self, f):
        n = len(f)
        k = np.arange(1, n + 1) * np.pi / self.Lx

        f_hat = SineTransform.dst_1d(f)
        u_hat = np.zeros(n)

        for i in range(n):
            if k[i] ** 2 > 1e-14:
                u_hat[i] = f_hat[i] / (k[i] ** 2)

        u = SineTransform.idst_1d(u_hat)
        return u


class GaussSeidelPoisson:

    @staticmethod
    def gauss_seidel_1d_step(n, r, u):
        u_new = u.copy()
        u_old = u.copy()

        for i in range(1, n - 1):
            u_new[i] = 0.5 * (u_new[i - 1] + u_old[i + 1] + r[i])

        dif_l1 = np.sum(np.abs(u_new[1:-1] - u_old[1:-1]))

        return u_new, dif_l1

    @staticmethod
    def solve_1d_poisson_gs(n_intervals, a, b, ua, ub, force_func,
                            max_iter=10000, tol=1e-4):
        n = n_intervals + 1
        x = np.linspace(a, b, n)
        h = (b - a) / n_intervals


        r = np.zeros(n)
        r[0] = ua
        r[1:-1] = force_func(x[1:-1]) * h ** 2
        r[-1] = ub


        u = np.zeros(n)
        u[0] = ua
        u[-1] = ub

        it_num = 0
        while it_num < max_iter:
            it_num += 1
            u, dif = GaussSeidelPoisson.gauss_seidel_1d_step(n, r, u)
            if dif <= tol:
                break

        return x, u, it_num

    @staticmethod
    def solve_2d_poisson_gs(nx, ny, dx, dy, f, max_iter=5000, tol=1e-6):
        u = np.zeros((nx, ny))
        h2 = dx * dy


        denom = 2.0 * (1.0 / (dx ** 2) + 1.0 / (dy ** 2))
        rhs = f.copy()

        it_num = 0
        while it_num < max_iter:
            it_num += 1
            u_old = u.copy()

            for i in range(1, nx - 1):
                for j in range(1, ny - 1):
                    u[i, j] = (
                        (u[i + 1, j] + u[i - 1, j]) / (dx ** 2) +
                        (u[i, j + 1] + u[i, j - 1]) / (dy ** 2) +
                        rhs[i, j]
                    ) / denom

            dif = np.max(np.abs(u - u_old))
            if dif < tol:
                break

        return u, it_num


class SpectralHeatSolver:

    def __init__(self, nx, ny, Lx=1.0, Ly=1.0, alpha=1.0):
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.alpha = alpha


        self.kx = np.arange(1, nx + 1) * np.pi / Lx
        self.ky = np.arange(1, ny + 1) * np.pi / Ly

    def solve_step_spectral(self, u, dt):

        u_hat = SineTransform.dst_2d(u)


        for i in range(self.nx):
            for j in range(self.ny):
                decay = np.exp(-self.alpha * (self.kx[i] ** 2 + self.ky[j] ** 2) * dt)
                u_hat[i, j] *= decay


        u_new = SineTransform.idst_2d(u_hat)
        return u_new
