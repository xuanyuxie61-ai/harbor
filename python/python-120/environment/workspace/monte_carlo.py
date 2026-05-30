
import numpy as np
from typing import Callable, Tuple, Optional


class MonteCarloSampler:

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def sample_uniform_box(self, n_samples: int, bounds: np.ndarray) -> np.ndarray:
        bounds = np.asarray(bounds, dtype=float)
        ndim = bounds.shape[0]
        samples = np.zeros((n_samples, ndim))
        for d in range(ndim):
            samples[:, d] = self.rng.uniform(bounds[d, 0], bounds[d, 1], n_samples)
        return samples

    def estimate_integral(self, func: Callable[[np.ndarray], np.ndarray],
                          bounds: np.ndarray, n_samples: int) -> Tuple[float, float]:
        bounds = np.asarray(bounds, dtype=float)
        volume = np.prod(bounds[:, 1] - bounds[:, 0])
        samples = self.sample_uniform_box(n_samples, bounds)
        f_vals = func(samples)
        mean_f = np.mean(f_vals)
        var_f = np.var(f_vals, ddof=1) if n_samples > 1 else 0.0
        integral = volume * mean_f
        error = volume * np.sqrt(var_f / n_samples)
        return float(integral), float(error)

    def adsorption_probability_monte_carlo(self,
                                            energy_func: Callable[[np.ndarray], np.ndarray],
                                            n_samples: int = 100000,
                                            bounds: Optional[np.ndarray] = None,
                                            temperature_k: float = 500.0) -> float:
        from utils import BOLTZMANN_KB
        if bounds is None:
            bounds = np.array([[-2e-10, 2e-10],
                               [-2e-10, 2e-10],
                               [0.5e-10, 4e-10]])
        samples = self.sample_uniform_box(n_samples, bounds)
        energies = energy_func(samples)
        e_threshold = np.min(energies) + 0.1
        delta_e = np.maximum(0.0, energies - e_threshold)
        kb_t = BOLTZMANN_KB * temperature_k / 1.602176634e-19
        probs = np.exp(-delta_e / kb_t)
        return float(np.mean(probs))

    def convergence_test(self, func: Callable[[np.ndarray], np.ndarray],
                         bounds: np.ndarray,
                         sample_sizes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        estimates = np.zeros(len(sample_sizes))
        errors = np.zeros(len(sample_sizes))
        for idx, n in enumerate(sample_sizes):
            estimates[idx], errors[idx] = self.estimate_integral(func, bounds, n)
        return estimates, errors


class QuadratureIntegrator:

    @staticmethod
    def composite_trapezoidal(f: Callable[[np.ndarray], np.ndarray],
                              a: float, b: float, n: int) -> float:
        if n < 1:
            raise ValueError("n >= 1")
        if b <= a:
            raise ValueError("b > a")
        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        y = f(x)
        return h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])

    @staticmethod
    def composite_simpson(f: Callable[[np.ndarray], np.ndarray],
                          a: float, b: float, n: int) -> float:
        if n % 2 != 0:
            n += 1
        if n < 2:
            raise ValueError("n >= 2")
        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        y = f(x)
        result = h / 3.0 * (y[0] + y[-1]
                           + 4.0 * np.sum(y[1:-1:2])
                           + 2.0 * np.sum(y[2:-1:2]))
        return float(result)

    @staticmethod
    def gauss_legendre_3point(f: Callable[[np.ndarray], np.ndarray],
                              a: float, b: float) -> float:
        nodes = np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)])
        weights = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])

        x_mapped = 0.5 * (b - a) * nodes + 0.5 * (a + b)
        jac = 0.5 * (b - a)
        return jac * np.sum(weights * f(x_mapped))


class PiecewiseLinearProductIntegral:

    def integrate(self, f_x: np.ndarray, f_v: np.ndarray,
                  g_x: np.ndarray, g_v: np.ndarray,
                  a: float, b: float) -> float:
        if len(f_x) < 2 or len(g_x) < 2:
            return 0.0
        if b <= a:
            return 0.0


        a_eff = max(a, f_x[0], g_x[0])
        b_eff = min(b, f_x[-1], g_x[-1])
        if b_eff <= a_eff:
            return 0.0


        all_breaks = np.sort(np.unique(np.concatenate([
            f_x[(f_x >= a_eff) & (f_x <= b_eff)],
            g_x[(g_x >= a_eff) & (g_x <= b_eff)]
        ])))
        if len(all_breaks) < 2:
            return 0.0

        total = 0.0
        for k in range(len(all_breaks) - 1):
            xl = all_breaks[k]
            xr = all_breaks[k + 1]
            if xr - xl < 1e-15:
                continue


            fl = self._interp_linear(f_x, f_v, xl)
            fr = self._interp_linear(f_x, f_v, xr)

            gl = self._interp_linear(g_x, g_v, xl)
            gr = self._interp_linear(g_x, g_v, xr)



            beta_f = (fr - fl) / (xr - xl)
            alpha_f = fl - beta_f * xl

            beta_g = (gr - gl) / (xr - xl)
            alpha_g = gl - beta_g * xl


            c0 = alpha_f * alpha_g
            c1 = alpha_f * beta_g + alpha_g * beta_f
            c2 = beta_f * beta_g


            total += (c0 * (xr - xl)
                      + c1 * (xr ** 2 - xl ** 2) / 2.0
                      + c2 * (xr ** 3 - xl ** 3) / 3.0)

        return total

    @staticmethod
    def _interp_linear(x_nodes: np.ndarray, y_nodes: np.ndarray,
                       x_query: float) -> float:
        if x_query <= x_nodes[0]:
            return y_nodes[0]
        if x_query >= x_nodes[-1]:
            return y_nodes[-1]
        idx = int(np.searchsorted(x_nodes, x_query)) - 1
        idx = max(0, min(idx, len(x_nodes) - 2))
        dx = x_nodes[idx + 1] - x_nodes[idx]
        if abs(dx) < 1e-15:
            return y_nodes[idx]
        t = (x_query - x_nodes[idx]) / dx
        return y_nodes[idx] + t * (y_nodes[idx + 1] - y_nodes[idx])

    def reaction_rate_integral(self, energy_grid_ev: np.ndarray,
                               cross_section: np.ndarray,
                               temperature_k: float) -> float:
        from utils import BOLTZMANN_KB
        kb_t_ev = BOLTZMANN_KB * temperature_k / 1.602176634e-19


        mb_kernel = np.sqrt(energy_grid_ev) * np.exp(-energy_grid_ev / kb_t_ev)


        return self.integrate(energy_grid_ev, cross_section,
                              energy_grid_ev, mb_kernel,
                              energy_grid_ev[0], energy_grid_ev[-1])
