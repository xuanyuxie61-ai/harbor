
import numpy as np


class StochasticDiffusivity2D:

    def __init__(self, D0=10.0, sigma=0.5):
        self.D0 = float(D0)
        self.sigma = float(sigma)

    def evaluate(self, omega, x, y):
        omega = np.asarray(omega, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if omega.size != 4:
            raise ValueError("omega must have exactly 4 elements.")


        arg = (omega[0] * np.cos(np.pi * x)
               + omega[1] * np.sin(np.pi * x)
               + omega[2] * np.cos(np.pi * y)
               + omega[3] * np.sin(np.pi * y))

        arg = np.exp(-0.125) * arg
        D = self.D0 + np.exp(arg)
        D = np.clip(D, 0.1, 1.0e4)
        return D


class LogNormalPermeabilityField:

    def __init__(self, k_mean=1.0e-14, sigma_ln_k=1.0, L_c=50.0):
        self.k_mean = float(k_mean)
        self.sigma_ln_k = float(sigma_ln_k)
        self.L_c = float(L_c)

    def _eigenfunctions_1d(self, x, M=6):
        x = np.asarray(x, dtype=np.float64)
        L = np.max(x) - np.min(x)
        if L <= 0:
            raise ValueError("Domain length must be positive.")
        phi = []
        for m in range(1, M + 1):

            if m % 2 == 1:
                pm = np.cos((m * np.pi * (x - np.min(x))) / (2.0 * L))
            else:
                pm = np.sin((m * np.pi * (x - np.min(x))) / (2.0 * L))
            phi.append(pm)
        return np.stack(phi, axis=0)

    def evaluate_1d(self, x, xi=None, M=6):
        x = np.asarray(x, dtype=np.float64)
        if xi is None:
            xi = np.random.randn(M)
        xi = np.asarray(xi, dtype=np.float64)
        if xi.size != M:
            raise ValueError(f"xi must have {M} elements.")

        phi = self._eigenfunctions_1d(x, M)

        lambda_m = self.sigma_ln_k ** 2 * (self.L_c ** 2) / (1.0 + (np.arange(1, M + 1) * self.L_c / (np.max(x) - np.min(x))) ** 2)
        lambda_m = np.clip(lambda_m, 1.0e-20, None)

        ln_k = np.log(self.k_mean)
        for m in range(M):
            ln_k += np.sqrt(lambda_m[m]) * phi[m, :] * xi[m]

        k = np.exp(ln_k)
        k = np.clip(k, 1.0e-20, 1.0e-8)
        return k

    def evaluate_2d(self, x, y, xi=None, M=8):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if xi is None:
            xi = np.random.randn(M)
        xi = np.asarray(xi, dtype=np.float64)


        Lx = np.max(x) - np.min(x)
        Ly = np.max(y) - np.min(y)
        if Lx <= 0 or Ly <= 0:
            raise ValueError("Domain dimensions must be positive.")

        nx = x.size
        ny = y.size
        X, Y = np.meshgrid(x, y, indexing='ij')
        ln_k = np.full((nx, ny), np.log(self.k_mean), dtype=np.float64)


        m_idx = 0
        for i in range(1, 4):
            for j in range(1, 4):
                if m_idx >= M:
                    break
                phi_x = np.sin(i * np.pi * (x - np.min(x)) / Lx)
                phi_y = np.sin(j * np.pi * (y - np.min(y)) / Ly)
                lam = (self.sigma_ln_k ** 2
                       / (1.0 + (i * self.L_c / Lx) ** 2)
                       / (1.0 + (j * self.L_c / Ly) ** 2))
                Phi = np.outer(phi_x, phi_y)
                ln_k += np.sqrt(max(lam, 0.0)) * Phi * xi[m_idx]
                m_idx += 1

        k = np.exp(ln_k)
        k = np.clip(k, 1.0e-20, 1.0e-8)
        return k


class RandomSampler:

    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(int(seed))

    def sample_min_max(self, func, a, b, n=10001):
        if a >= b:
            raise ValueError("Lower bound a must be less than upper bound b.")
        if n <= 0:
            raise ValueError("Number of samples must be positive.")

        x = np.random.rand(n)
        x = a * (1.0 - x) + b * x
        fx = np.asarray([func(val) for val in x])

        fmin = np.min(fx)
        xmin = x[np.argmin(fx)]
        fmax = np.max(fx)
        xmax = x[np.argmax(fx)]
        return float(xmin), float(fmin), float(xmax), float(fmax)

    def latin_hypercube(self, dim, n, bounds):
        if len(bounds) != dim:
            raise ValueError("bounds must have length equal to dim.")
        samples = np.zeros((n, dim))
        for d in range(dim):
            a, b = bounds[d]

            cut = np.linspace(0, 1, n + 1)
            u = np.random.rand(n)
            a_points = cut[:-1]
            b_points = cut[1:]
            points = a_points + u * (b_points - a_points)
            np.random.shuffle(points)
            samples[:, d] = a + points * (b - a)
        return samples

    def monte_carlo_expectation(self, func, sampler, n=10000):
        samples = sampler(n)
        values = np.asarray([func(s) for s in samples])
        return np.mean(values), np.std(values) / np.sqrt(n)


def generate_stochastic_permeability_realization(params, n_realizations=1):
    from thm_model import THMParameters
    if not isinstance(params, THMParameters):
        raise TypeError("params must be a THMParameters instance.")

    nx, nz, ny = params.grid_shape()
    Lx = params.reservoir_length
    Ly = params.reservoir_width
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)

    field_gen = LogNormalPermeabilityField(
        k_mean=params.matrix_permeability,
        sigma_ln_k=0.5,
        L_c=min(Lx, Ly) / 4.0
    )

    realizations = []
    for _ in range(n_realizations):
        k_2d = field_gen.evaluate_2d(x, y, M=8)

        k_3d = np.tile(k_2d[:, :, np.newaxis], (1, 1, nz))
        realizations.append(k_3d)

    return realizations
