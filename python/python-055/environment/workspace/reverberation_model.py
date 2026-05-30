
import numpy as np


class MackeyGlassReverberation:

    def __init__(
        self,
        gamma: float = 0.1,
        beta: float = 0.2,
        n: float = 9.65,
        tau: float = 5.0,
        dt: float = 0.01
    ):
        self.gamma = float(gamma)
        self.beta = float(beta)
        self.n = float(n)
        self.tau = float(tau)
        self.dt = float(dt)


        self._history_t = None
        self._history_x = None

    def _dde_rhs(self, t: float, x: float, x_delayed: float) -> float:
        if x_delayed < 0.0:
            x_delayed = 0.0

        denom = 1.0 + x_delayed ** self.n
        if denom < 1e-15:
            denom = 1e-15
        dxdt = self.beta * x_delayed / denom - self.gamma * x
        return dxdt

    def _get_delayed(self, t: float) -> float:
        if self._history_t is None or len(self._history_t) == 0:
            return 0.0
        t_delayed = t - self.tau
        if t_delayed <= self._history_t[0]:
            return self._history_x[0]
        if t_delayed >= self._history_t[-1]:
            return self._history_x[-1]

        idx = np.searchsorted(self._history_t, t_delayed)
        if idx == 0:
            return self._history_x[0]
        t0, t1 = self._history_t[idx - 1], self._history_t[idx]
        x0, x1 = self._history_x[idx - 1], self._history_x[idx]
        if abs(t1 - t0) < 1e-15:
            return x0
        alpha = (t_delayed - t0) / (t1 - t0)
        return x0 + alpha * (x1 - x0)

    def solve(
        self,
        t_span: tuple,
        x0: float = 0.5,
        history_const: float = 0.0
    ) -> tuple:
        t_start, t_stop = t_span
        n_steps = int(np.ceil((t_stop - t_start) / self.dt)) + 1
        t_arr = np.linspace(t_start, t_stop, n_steps)
        x_arr = np.zeros(n_steps, dtype=np.float64)


        self._history_t = [t_start - self.tau]
        self._history_x = [history_const]

        x_arr[0] = x0

        for i in range(n_steps - 1):
            t = t_arr[i]
            x = x_arr[i]
            x_delayed = self._get_delayed(t)


            k1 = self._dde_rhs(t, x, x_delayed)

            x_delayed_k2 = self._get_delayed(t + 0.5 * self.dt)
            k2 = self._dde_rhs(t + 0.5 * self.dt, x + 0.5 * self.dt * k1, x_delayed_k2)

            x_delayed_k3 = self._get_delayed(t + 0.5 * self.dt)
            k3 = self._dde_rhs(t + 0.5 * self.dt, x + 0.5 * self.dt * k2, x_delayed_k3)

            x_delayed_k4 = self._get_delayed(t + self.dt)
            k4 = self._dde_rhs(t + self.dt, x + self.dt * k3, x_delayed_k4)

            x_new = x + self.dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


            if x_new < 0.0:
                x_new = 0.0

            x_arr[i + 1] = x_new
            self._history_t.append(t_arr[i + 1])
            self._history_x.append(x_new)

        return t_arr, x_arr

    def compute_reverberation_envelope(
        self,
        ttw_base: float,
        amplitude: float = 1.0,
        duration_factor: float = 3.0
    ) -> tuple:
        t_start = 0.0
        t_stop = ttw_base + duration_factor * self.tau

        x0 = amplitude
        t_arr, x_arr = self.solve((t_start, t_stop), x0=x0, history_const=0.0)

        envelope = np.abs(x_arr)

        if len(envelope) >= 3:
            smoothed = np.convolve(envelope, np.ones(3) / 3.0, mode='same')
            envelope = smoothed
        return t_arr, envelope


class StochasticReverberationField:

    def __init__(self, n_modes: int = 5, seed: int = 55):
        self.n_modes = n_modes
        self.rng = np.random.default_rng(seed)
        self.modes = []
        for _ in range(n_modes):
            gamma = self.rng.uniform(0.05, 0.2)
            beta = self.rng.uniform(0.15, 0.35)
            n_exp = self.rng.uniform(7.0, 12.0)
            tau = self.rng.uniform(2.0, 10.0)
            self.modes.append(MackeyGlassReverberation(gamma, beta, n_exp, tau))

    def generate_composite_envelope(
        self,
        ttw_base: float,
        base_amplitude: float = 1.0
    ) -> tuple:

        t_stop = ttw_base + 30.0
        n_points = 2000
        t_common = np.linspace(0.0, t_stop, n_points)
        composite = np.zeros(n_points, dtype=np.float64)

        for i, mode in enumerate(self.modes):

            weight = base_amplitude * (0.5 ** i)
            t_mode, env_mode = mode.compute_reverberation_envelope(ttw_base, amplitude=weight)

            env_interp = np.interp(t_common, t_mode, env_mode, left=0.0, right=0.0)
            composite += env_interp

        return t_common, composite
