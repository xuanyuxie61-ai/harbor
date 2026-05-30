
import numpy as np


class ClimateForcing:

    def __init__(self, amplitude: float = 0.15, volcanic_freq: float = 0.03):
        self.amplitude = amplitude
        self.volcanic_freq = volcanic_freq
        self.cycle_periods = [11.0, 88.0, 210.0]

    def compute(self, t: float, n_grid: int):
        solar = 0.0
        for period in self.cycle_periods:
            solar += np.sin(2.0 * np.pi * t / period) / len(self.cycle_periods)


        volcanic = 0.0
        np.random.seed(int(t * 1000) % 2**31)
        if np.random.rand() < self.volcanic_freq:
            volcanic = -0.5 * np.exp(-0.01 * (t % 50.0) ** 2)

        forcing_scalar = self.amplitude * solar + volcanic
        return forcing_scalar * np.ones(n_grid, dtype=np.float64)


class StochasticClimateModel:

    def __init__(
        self,
        n_grid: int,
        diffusion_coeff: float = 1.2e-6,
        damping_coeff: float = 0.02,
        forcing_amplitude: float = 0.15,
        noise_intensity: float = 0.08,
        dt_years: float = 1.0,
    ):
        self.n_grid = n_grid
        self.D = diffusion_coeff
        self.lambda_ = damping_coeff
        self.noise_intensity = noise_intensity
        self.dt = dt_years
        self.forcing = ClimateForcing(amplitude=forcing_amplitude)


        self._build_laplacian()

    def _build_laplacian(self):
        n = self.n_grid
        self.L = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            self.L[i, i] = -2.0
            left = (i - 1) % n
            right = (i + 1) % n
            self.L[i, left] = 1.0
            self.L[i, right] = 1.0

        self.L = self.L / max(1.0, n / 100.0)

    def _drift(self, T: np.ndarray, t: float) -> np.ndarray:
        forcing = self.forcing.compute(t, self.n_grid)
        return self.D * (self.L @ T) - self.lambda_ * T + forcing

    def _diffusion(self, T: np.ndarray) -> np.ndarray:
        return self.noise_intensity * (1.0 + 0.1 * np.abs(T))

    def _diffusion_derivative(self, T: np.ndarray) -> np.ndarray:
        return self.noise_intensity * 0.1 * np.sign(T)

    def trapezoidal_step(self, T: np.ndarray, t: float) -> np.ndarray:
        h = self.dt
        f_n = self._drift(T, t)
        z = T.copy()

        for _ in range(20):
            z_new = T + 0.5 * h * (f_n + self._drift(z, t + h))
            if np.linalg.norm(z_new - z) < 1e-12:
                z = z_new
                break
            z = z_new

        return z

    def euler_maruyama_step(self, T: np.ndarray, t: float) -> np.ndarray:
        dW = np.sqrt(self.dt) * np.random.randn(self.n_grid)
        drift = self._drift(T, t)
        diff = self._diffusion(T)
        return T + drift * self.dt + diff * dW

    def milstein_step(self, T: np.ndarray, t: float = 0.0) -> np.ndarray:
        dW = np.sqrt(self.dt) * np.random.randn(self.n_grid)
        drift = self._drift(T, t)
        diff = self._diffusion(T)
        diff_deriv = self._diffusion_derivative(T)


        correction = 0.5 * diff * diff_deriv * (dW ** 2 - self.dt)

        return T + drift * self.dt + diff * dW + correction

    def initial_state(self) -> np.ndarray:
        n = self.n_grid

        lat = np.linspace(-90, 90, n)
        base_temp = 15.0 - 30.0 * np.sin(np.radians(lat)) ** 2


        lon = np.linspace(-180, 180, n)
        x = lon / 60.0
        y = lat / 30.0

        peaks = (
            3.0 * (1.0 - x) ** 2 * np.exp(-x ** 2 - (y + 1.0) ** 2)
            - 10.0 * (x / 5.0 - x ** 3 - y ** 5) * np.exp(-x ** 2 - y ** 2)
            - (1.0 / 3.0) * np.exp(-(x + 1.0) ** 2 - y ** 2)
        )

        peaks = 0.3 * peaks

        return base_temp + peaks

    def strong_convergence_test(self, T0: np.ndarray, t_final: float = 1.0):
        n_ref = 2 ** 11
        dt_ref = t_final / n_ref
        m = 200


        dW_ref = np.sqrt(dt_ref) * np.random.randn(m, n_ref)

        r_values = [1, 16, 32, 64, 128]
        errors = []

        for r in r_values[1:]:
            dt = r * dt_ref
            L = n_ref // r
            x_mil = np.zeros(m)

            for p in range(m):
                x_temp = T0[0]
                for j in range(L):
                    winc = np.sum(dW_ref[p, r * j:r * (j + 1)])

                    drift = -self.lambda_ * x_temp
                    diff = self.noise_intensity * (1.0 + 0.1 * abs(x_temp))
                    diff_deriv = self.noise_intensity * 0.1 * np.sign(x_temp)
                    x_temp = (
                        x_temp
                        + drift * dt
                        + diff * winc
                        + 0.5 * diff * diff_deriv * (winc ** 2 - dt)
                    )
                x_mil[p] = x_temp


            x_ref = np.zeros(m)
            for p in range(m):
                x_temp = T0[0]
                for j in range(n_ref):
                    winc = dW_ref[p, j]
                    drift = -self.lambda_ * x_temp
                    diff = self.noise_intensity * (1.0 + 0.1 * abs(x_temp))
                    diff_deriv = self.noise_intensity * 0.1 * np.sign(x_temp)
                    x_temp = (
                        x_temp
                        + drift * dt_ref
                        + diff * winc
                        + 0.5 * diff * diff_deriv * (winc ** 2 - dt_ref)
                    )
                x_ref[p] = x_temp

            err = np.mean(np.abs(x_mil - x_ref))
            errors.append(err)

        dtvals = np.array([r * dt_ref for r in r_values[1:]])
        A = np.vstack([np.ones(len(dtvals)), np.log(dtvals)]).T
        rhs = np.log(errors)
        sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
        q = sol[1]
        return q, errors, dtvals
