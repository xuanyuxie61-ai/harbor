
import numpy as np
from utils import clenshaw_chebyshev_eval, gauss_legendre_nodes_weights



_CLAUSEN_COEFFS_SMALL = np.array([
    1.3888888888888889e-02, 0.0, -2.7777777777777778e-04,
    0.0, 7.936507936507937e-06, 0.0, -2.505210838544172e-07,
    0.0, 8.417724804700504e-09, 0.0, -2.946634356703308e-10,
    0.0, 1.064193259978150e-11, 0.0, -3.932350536369160e-13,
    0.0, 1.480725065921570e-14
], dtype=float)

_CLAUSEN_COEFFS_LARGE = np.array([
    -1.3888888888888889e-02, 0.0, 2.7777777777777778e-04,
    0.0, -7.936507936507937e-06, 0.0, 2.505210838544172e-07,
    0.0, -8.417724804700504e-09, 0.0, 2.946634356703308e-10,
    0.0, -1.064193259978150e-11, 0.0, 3.932350536369160e-13,
    0.0, -1.480725065921570e-14
], dtype=float)


def clausen_function(x):
    x = np.asarray(x, dtype=float)
    result = np.zeros_like(x)

    x_red = np.mod(x + np.pi, 2.0 * np.pi) - np.pi

    sign_flip = np.sign(x_red)
    x_abs = np.abs(x_red)

    mask_large = x_abs > 0.5 * np.pi
    x_eval = x_abs.copy()
    x_eval[mask_large] = np.pi - x_abs[mask_large]

    t = 2.0 * x_eval / np.pi - 1.0

    for i in range(len(t)):
        result.flat[i] = clenshaw_chebyshev_eval(t.flat[i], _CLAUSEN_COEFFS_SMALL)
    result *= sign_flip
    return result


def generate_padua_points(n):
    if n < 0:
        return np.zeros((0, 2))

    k1 = np.arange(n + 2)
    C_n1 = np.cos(np.pi * k1 / (n + 1))
    k2 = np.arange(n + 1)
    C_n = np.cos(np.pi * k2 / n) if n > 0 else np.array([1.0])
    points = []

    for i, xi in enumerate(C_n1):
        for j, yj in enumerate(C_n):
            if (i % 2 == 0 and j % 2 == 1) or (i % 2 == 1 and j % 2 == 0):
                if n % 2 == 0:
                    if i % 2 == 0:
                        points.append([xi, yj])
                else:
                    if i % 2 == 1:
                        points.append([xi, yj])

    for i, xi in enumerate(C_n):
        for j, yj in enumerate(C_n1):
            if (i % 2 == 0 and j % 2 == 0) or (i % 2 == 1 and j % 2 == 1):
                if n % 2 == 0:
                    if i % 2 == 1:
                        points.append([xi, yj])
                else:
                    if i % 2 == 0:
                        points.append([xi, yj])





    points = []
    for i in range(n + 1):
        for j in range(n + 2):
            if (i + j) % 2 == 0:
                points.append([np.cos(i * np.pi / n), np.cos(j * np.pi / (n + 1))])
    for i in range(n + 2):
        for j in range(n + 1):
            if (i + j) % 2 == 1:
                points.append([np.cos(i * np.pi / (n + 1)), np.cos(j * np.pi / n)])
    pts = np.array(points, dtype=float)

    pts = np.unique(np.round(pts, 14), axis=0)
    return pts


def padua_weights(n):
    pts = generate_padua_points(n)
    N = len(pts)


    w = np.ones(N, dtype=float) * (4.0 / N)
    return w


class ChebyshevSpectrumAnalyzer:

    def __init__(self, n_modes=64):
        self.n_modes = n_modes

    def _dct_transform(self, signal, t_min=-1.0, t_max=1.0):
        signal = np.asarray(signal, dtype=float)
        N = len(signal)
        if N == 0:
            return np.zeros(self.n_modes + 1)

        n = self.n_modes
        j = np.arange(n + 1)
        x_nodes = np.cos(np.pi * j / n)

        t_nodes = 0.5 * (t_max - t_min) * x_nodes + 0.5 * (t_max + t_min)
        f_nodes = np.interp(t_nodes, np.linspace(t_min, t_max, N), signal)

        from scipy.fft import dct
        coeffs = dct(f_nodes, type=1)
        coeffs[0] *= 0.5
        coeffs[n] *= 0.5
        coeffs *= (2.0 / n)
        return coeffs

    def analyze(self, signal, t_min=0.0, t_max=1.0):
        coeffs = self._dct_transform(signal, t_min, t_max)

        energy = np.sum(coeffs[1:] ** 2) * 0.5 * np.pi
        energy += coeffs[0] ** 2 * 0.5 * np.pi

        if len(coeffs) > 1:
            dominant_mode = np.argmax(np.abs(coeffs[1:])) + 1
        else:
            dominant_mode = 0
        return {
            'coefficients': coeffs,
            'energy': energy,
            'dominant_mode': dominant_mode,
            'dc_component': coeffs[0]
        }

    def reconstruct(self, coeffs, n_eval=512):
        x = np.linspace(-1, 1, n_eval)
        vals = np.array([clenshaw_chebyshev_eval(xi, coeffs) for xi in x])
        return x, vals


class GaussLegendreSignalIntegrator:

    def __init__(self, n_points=64):
        self.n_points = n_points
        self.xi, self.wi = gauss_legendre_nodes_weights(n_points)

    def integrate_function(self, f, a=-1.0, b=1.0):
        x_mapped = 0.5 * (b - a) * self.xi + 0.5 * (a + b)
        fx = np.array([f(x) for x in x_mapped], dtype=float)
        return 0.5 * (b - a) * np.sum(self.wi * fx)

    def signal_moments(self, signal, t):
        t = np.asarray(t, dtype=float)
        signal = np.asarray(signal, dtype=float)
        t_min, t_max = t[0], t[-1]

        def f_interp(x):
            return np.interp(x, t, signal)
        length = t_max - t_min
        mu1 = self.integrate_function(f_interp, t_min, t_max) / length
        mu2 = self.integrate_function(lambda x: (f_interp(x) - mu1) ** 2, t_min, t_max) / length
        mu3 = self.integrate_function(lambda x: (f_interp(x) - mu1) ** 3, t_min, t_max) / length
        mu4 = self.integrate_function(lambda x: (f_interp(x) - mu1) ** 4, t_min, t_max) / length
        variance = mu2
        std = np.sqrt(variance) if variance > 1e-15 else 1e-15
        skewness = mu3 / (std ** 3)
        kurtosis = mu4 / (std ** 4) - 3.0
        return {
            'mean': mu1,
            'variance': variance,
            'std': std,
            'skewness': skewness,
            'kurtosis': kurtosis
        }

    def signal_energy_functional(self, signal, t, alpha=2.0):
        t_min, t_max = t[0], t[-1]
        def f_abs_power(x):
            s = np.interp(x, t, signal)
            return np.abs(s) ** alpha
        return self.integrate_function(f_abs_power, t_min, t_max)


class SpatialPaduaSampler:

    def __init__(self, n_degree=16):
        self.n_degree = n_degree
        self.points = generate_padua_points(n_degree)
        self.weights = padua_weights(n_degree)

    def sample_field(self, field_func):
        vals = np.array([field_func(p[0], p[1]) for p in self.points], dtype=float)
        return vals

    def integrate_field(self, field_func):
        vals = self.sample_field(field_func)
        return np.sum(self.weights * vals)

    def interpolate_to_grid(self, sampled_values, X, Y):
        from scipy.interpolate import RBFInterpolator
        pts = self.points

        grid_points = np.column_stack([X.ravel(), Y.ravel()])
        rbf = RBFInterpolator(pts, sampled_values, kernel='thin_plate_spline')
        vals_grid = rbf(grid_points)
        return vals_grid.reshape(X.shape)
