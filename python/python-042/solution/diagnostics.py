"""
diagnostics.py

Post-processing, analysis, and parameter space exploration tools.

Core seed mappings:
- 539_histogram_discrete   -> statistical temperature distribution analysis
- 559_hypercube_integrals  -> Monte Carlo parameter space sampling
- 1428_zero_chandrupatla   -> root finding for critical Rayleigh number
- 691_lissajous            -> periodic forcing parameterization

Scientific formulas:
- Discrete histogram density estimation:
    For sorted samples s in [s_min, s_max], construct piecewise linear PDF p(x)
    with normalization ∫_{s_min}^{s_max} p(x) dx = 1.
- Root finding for critical Rayleigh number Ra_c:
    Solve f(Ra) = Nu(Ra) − 1.05 = 0, where Nu is computed from
    convective simulation. Chandrupatla's hybrid quadratic/bisection method.
- Lissajous forcing:
    x(t) = sin(a1 t + b1), y(t) = sin(a2 t + b2)
    Used here for time-periodic tidal forcing on mantle boundary conditions.
- Monte Carlo uncertainty propagation:
    For model output y = f(θ) with θ ~ Uniform([0,1]^m),
    estimate E[y] and Var[y] via sample mean and variance.
"""

import numpy as np
from typing import Tuple, Callable, Optional


class TemperatureHistogram:
    """
    Discrete histogram and PDF estimation for mantle temperature fields.
    Adapted from seed 539_histogram_discrete.
    """
    def __init__(self, T_min: float = 300.0, T_max: float = 3000.0):
        self.T_min = T_min
        self.T_max = T_max

    def setup(self, samples: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        Construct discrete histogram from temperature samples.
        Returns (x_num, x_edges, pdf_values).
        """
        samples = np.asarray(samples, dtype=float).ravel()
        # Clip to bounds
        samples = np.clip(samples, self.T_min, self.T_max)
        samples = np.sort(samples)
        s_num = len(samples)
        # Build unique ordered points with counts
        x_list = [self.T_min]
        c_list = [0]
        for val in samples:
            if abs(val - x_list[-1]) < 1e-12:
                c_list[-1] += 1
            else:
                x_list.append(val)
                c_list.append(1)
        if abs(x_list[-1] - self.T_max) > 1e-12:
            x_list.append(self.T_max)
            c_list.append(0)
        x = np.array(x_list, dtype=float)
        c = np.array(c_list, dtype=float)
        x_num = len(x)
        # Piecewise linear PDF
        y = np.zeros(x_num, dtype=float)
        if x_num >= 2:
            y[0] = c[0] / max(x[1] - x[0], 1e-15)
            for i in range(1, x_num - 1):
                y[i] = c[i] / max(x[i + 1] - x[i - 1], 1e-15)
            y[-1] = c[-1] / max(x[-1] - x[-2], 1e-15)
        # Normalize
        y_int = 0.0
        for i in range(x_num - 1):
            y_int += (x[i + 1] - x[i]) * (y[i + 1] + y[i]) / 2.0
        if y_int > 1e-30:
            y = y / y_int
        return x_num, x, y

    def entropy(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute differential entropy S = −∫ p(x) log(p(x)) dx.
        """
        s = 0.0
        for i in range(len(x) - 1):
            dx = x[i + 1] - x[i]
            if y[i] > 1e-30 and y[i + 1] > 1e-30:
                # Trapezoidal integration
                s += -0.5 * dx * (y[i] * np.log(y[i]) + y[i + 1] * np.log(y[i + 1]))
        return float(s)


class ChandrupatlaRootFinder:
    """
    Hybrid quadratic/bisection root finder.
    Adapted from seed 1428_zero_chandrupatla.
    Reference: Chandrupatla, Advances in Engineering Software, 28(3), 145-149, 1997.
    """
    def __init__(self, epsilon: float = 1.0e-10, delta: float = 1.0e-5):
        self.epsilon = epsilon
        self.delta = delta

    def find_root(self, f: Callable[[float], float], x1: float, x2: float) -> Tuple[float, float, int]:
        """
        Find root of f in interval [x1, x2] where f(x1) and f(x2) have opposite signs.
        Returns (xm, fm, calls).
        """
        f1 = f(x1)
        f2 = f(x2)
        calls = 2
        if f1 * f2 > 0:
            raise ValueError("f(x1) and f(x2) must have opposite signs")
        t = 0.5
        while True:
            x0 = x1 + t * (x2 - x1)
            f0 = f(x0)
            calls += 1
            # Arrange 2-1-3
            if np.sign(f0) == np.sign(f1):
                x3 = x1
                f3 = f1
                x1 = x0
                f1 = f0
            else:
                x3 = x2
                f3 = f2
                x2 = x1
                f2 = f1
                x1 = x0
                f1 = f0
            # Best approximation
            if abs(f2) < abs(f1):
                xm = x2
                fm = f2
            else:
                xm = x1
                fm = f1
            tol = 2.0 * self.epsilon * abs(xm) + 0.5 * self.delta
            tl = tol / abs(x2 - x1)
            if tl > 0.5 or abs(fm) < 1e-15:
                break
            # Inverse quadratic interpolation test
            xi = (x1 - x2) / (x3 - x2)
            ph = (f1 - f2) / (f3 - f2)
            fl = 1.0 - np.sqrt(max(1.0 - xi, 0.0))
            fh = np.sqrt(max(xi, 0.0))
            if fl < ph < fh:
                al = (x3 - x1) / (x2 - x1)
                a = f1 / (f2 - f1)
                b = f3 / (f2 - f3)
                c = f1 / (f3 - f1)
                d = f2 / (f3 - f2)
                t = a * b + c * d * al
            else:
                t = 0.5
            t = max(t, tl)
            t = min(t, 1.0 - tl)
        return xm, fm, calls

    def find_critical_rayleigh(self, nu_func: Callable[[float], float],
                                Ra_min: float = 1.0e3,
                                Ra_max: float = 1.0e6) -> Tuple[float, float, int]:
        """
        Find critical Rayleigh number where Nu(Ra) ≈ 1.05 (onset of convection).
        """
        def f(Ra):
            return nu_func(Ra) - 1.05
        f_min = f(Ra_min)
        f_max = f(Ra_max)
        if f_min * f_max > 0:
            # If both same sign, return closest
            if abs(f_min) < abs(f_max):
                return Ra_min, f_min, 1
            else:
                return Ra_max, f_max, 1
        return self.find_root(f, Ra_min, Ra_max)


class LissajousForcing:
    """
    Periodic tidal forcing modeled by Lissajous curves.
    Adapted from seed 691_lissajous.

    In geophysical context, the Lissajous parameters represent:
    - a1, a2: forcing frequencies (lunar/solar tidal harmonics)
    - b1, b2: phase lags
    The curves parameterize time-periodic boundary heat flux or
    mechanical forcing on the CMB.
    """
    def __init__(self, a1: float = 2.0, b1: float = np.pi / 4.0,
                 a2: float = 3.0, b2: float = 0.0):
        self.a1 = a1
        self.b1 = b1
        self.a2 = a2
        self.b2 = b2

    def evaluate(self, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate Lissajous parametric curve at times t.
        Returns (x, y) arrays.
        """
        t = np.asarray(t, dtype=float)
        x = np.sin(self.a1 * t + self.b1)
        y = np.sin(self.a2 * t + self.b2)
        return x, y

    def boundary_heat_flux_modulation(self, t: float, q0: float = 1.0,
                                       amplitude: float = 0.1) -> float:
        """
        Modulate surface heat flux by Lissajous periodic forcing:
            q(t) = q0 [1 + amplitude * sin(a1 t + b1) * sin(a2 t + b2)]
        """
        modulation = np.sin(self.a1 * t + self.b1) * np.sin(self.a2 * t + self.b2)
        return q0 * (1.0 + amplitude * modulation)


class ParameterSampler:
    """
    Monte Carlo parameter space sampling for uncertainty quantification.
    Adapted from seed 559_hypercube_integrals.
    """
    def __init__(self, seed: Optional[int] = 42):
        self.rng = np.random.default_rng(seed)

    def sample_hypercube(self, m: int, n: int) -> np.ndarray:
        """
        Sample n points in m-dimensional unit hypercube.
        Returns array of shape (m, n).
        """
        return self.rng.random((m, n))

    def propagate_uncertainty(self, model_func: Callable[[np.ndarray], float],
                               param_bounds: np.ndarray, n_samples: int = 1000) -> Tuple[float, float]:
        """
        Propagate parameter uncertainty through model.
        param_bounds: array of shape (m, 2) with [min, max] for each parameter.
        Returns (mean, std_dev) of model output.
        """
        param_bounds = np.asarray(param_bounds, dtype=float)
        m = param_bounds.shape[0]
        samples = self.sample_hypercube(m, n_samples)
        # Scale to parameter bounds
        for i in range(m):
            samples[i, :] = param_bounds[i, 0] + samples[i, :] * (param_bounds[i, 1] - param_bounds[i, 0])
        outputs = np.array([model_func(samples[:, k]) for k in range(n_samples)])
        return float(np.mean(outputs)), float(np.std(outputs))


class MantleDiagnostics:
    """
    Comprehensive diagnostics for mantle convection simulation.
    """
    def __init__(self, T_min: float = 300.0, T_max: float = 3000.0):
        self.histogram = TemperatureHistogram(T_min, T_max)
        self.root_finder = ChandrupatlaRootFinder()
        self.forcing = LissajousForcing()
        self.sampler = ParameterSampler()

    def analyze_temperature_field(self, T: np.ndarray) -> dict:
        """
        Compute statistics and histogram for temperature field.
        """
        T_flat = T.ravel()
        x_num, x, y = self.histogram.setup(T_flat)
        entropy = self.histogram.entropy(x, y)
        return {
            "mean": float(np.mean(T_flat)),
            "std": float(np.std(T_flat)),
            "min": float(np.min(T_flat)),
            "max": float(np.max(T_flat)),
            "entropy": entropy,
            "histogram_edges": x,
            "histogram_pdf": y,
        }

    def compute_nusselt_from_simulation(self, surface_flux: float,
                                        conductive_flux: float) -> float:
        """
        Nusselt number from surface heat flux.
        """
        from mantle_physics import DimensionlessNumbers
        return DimensionlessNumbers.nusselt_number(surface_flux, conductive_flux)
