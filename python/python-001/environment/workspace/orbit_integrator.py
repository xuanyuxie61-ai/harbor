
import numpy as np
from typing import Callable, Tuple, Optional, List


class OrbitIntegratorError(Exception):
    pass


def line_ncc_rule(n: int, a: float, b: float) -> Tuple[np.ndarray, np.ndarray]:
    if n < 2:
        raise OrbitIntegratorError("NCC 规则需要至少 2 个节点")
    x = np.linspace(a, b, n)
    w = np.zeros(n)

    for i in range(n):
        d = np.zeros(n)
        d[i] = 1.0


        for j in range(1, n):
            for k in range(j, n):
                d[n + j - k - 1] = (d[n + j - k - 2] - d[n + j - k - 1]) / (x[n - k - 1] - x[n + j - k - 1])


        for j in range(1, n):
            for k in range(1, n - j + 1):
                d[n - k - 1] = d[n - k - 1] - x[n - k - j] * d[n - k]


        yvala = d[n - 1] / n
        yvalb = d[n - 1] / n
        for j in range(n - 2, -1, -1):
            yvala = yvala * a + d[j] / (j + 1)
            yvalb = yvalb * b + d[j] / (j + 1)
        yvala *= a
        yvalb *= b
        w[i] = yvalb - yvala

    return x, w


def newton_cotes_integrate(
    f: Callable[[float], float],
    a: float,
    b: float,
    n: int = 9,
    n_sub: int = 100
) -> float:
    if n_sub < 1:
        raise OrbitIntegratorError("子区间数必须 ≥ 1")
    h = (b - a) / n_sub
    total = 0.0
    for k in range(n_sub):
        sub_a = a + k * h
        sub_b = sub_a + h
        x, w = line_ncc_rule(n, sub_a, sub_b)
        total += np.sum(w * np.array([f(xi) for xi in x]))
    return total


def srk4_ti_step(
    x: np.ndarray,
    t: float,
    h: float,
    q: float,
    fi: Callable[[np.ndarray], np.ndarray],
    gi: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    a21 = 2.71644396264860
    a31 = -6.95653259006152
    a32 = 0.78313689457981
    a42 = 0.48257353309214
    a43 = 0.26171080165848
    a51 = 0.47012396888046
    a52 = 0.36597075368373
    a53 = 0.08906615686702
    a54 = 0.07483912056879

    q1 = 2.12709852335625
    q2 = 2.73245878238737
    q3 = 11.22760917474960
    q4 = 13.36199560336697

    n1 = np.random.randn(x.shape[0])
    w1 = n1 * np.sqrt(q1 * q / h)
    k1 = h * fi(x) + h * gi(x) * w1

    x2 = x + a21 * k1
    n2 = np.random.randn(x.shape[0])
    w2 = n2 * np.sqrt(q2 * q / h)
    k2 = h * fi(x2) + h * gi(x2) * w2

    x3 = x + a31 * k1 + a32 * k2
    n3 = np.random.randn(x.shape[0])
    w3 = n3 * np.sqrt(q3 * q / h)
    k3 = h * fi(x3) + h * gi(x3) * w3

    x4 = x + a42 * k2 + a43 * k3
    n4 = np.random.randn(x.shape[0])
    w4 = n4 * np.sqrt(q4 * q / h)
    k4 = h * fi(x4) + h * gi(x4) * w4

    xstar = x + a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4
    return xstar


def rk4_step(
    x: np.ndarray,
    t: float,
    h: float,
    f: Callable[[np.ndarray, float], np.ndarray]
) -> np.ndarray:
    k1 = f(x, t)
    k2 = f(x + 0.5 * h * k1, t + 0.5 * h)
    k3 = f(x + 0.5 * h * k2, t + 0.5 * h)
    k4 = f(x + h * k3, t + h)
    return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


class OrbitalDynamics:

    def __init__(
        self,
        grav_accel_func: Callable[[np.ndarray], np.ndarray],
        gm_sun: float = 1.32712440018e11,
        solar_distance: float = 1.496e8,
        beta_srp: float = 0.0,
        perturbation_std: float = 0.0
    ):
        self.grav = grav_accel_func
        self.gm_sun = gm_sun
        self.solar_distance = solar_distance
        self.beta_srp = beta_srp
        self.perturbation_std = perturbation_std

    def _solar_direction(self) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0])

    def _solar_radiation_pressure(self, pos: np.ndarray) -> np.ndarray:
        if self.beta_srp <= 0.0:
            return np.zeros(3)
        c_light = 299792.458
        factor = self.beta_srp * self.gm_sun / (c_light ** 2) / (self.solar_distance ** 2)
        return factor * self._solar_direction()

    def _third_body_sun(self, pos: np.ndarray) -> np.ndarray:
        d = self.solar_distance
        return self.gm_sun / (d ** 3) * pos

    def deterministic_rhs(self, state: np.ndarray, t: float) -> np.ndarray:
        pos = state[:3]
        vel = state[3:]
        a_grav = self.grav(pos)
        a_srp = self._solar_radiation_pressure(pos)
        a_3b = self._third_body_sun(pos)
        acc = a_grav + a_srp + a_3b
        return np.concatenate([vel, acc])

    def stochastic_drift(self, state: np.ndarray) -> np.ndarray:
        if self.perturbation_std <= 0.0:
            return np.zeros(6)
        g = np.zeros(6)
        g[3:] = self.perturbation_std
        return g

    def stochastic_rhs(self, state: np.ndarray) -> np.ndarray:
        return self.deterministic_rhs(state, 0.0)

    def integrate_deterministic(
        self,
        state0: np.ndarray,
        t_span: Tuple[float, float],
        n_steps: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray]:
        t0, tf = t_span
        h = (tf - t0) / n_steps
        t_array = np.linspace(t0, tf, n_steps + 1)
        states = np.zeros((n_steps + 1, 6))
        states[0] = state0.copy()

        for i in range(n_steps):
            states[i + 1] = rk4_step(states[i], t_array[i], h, self.deterministic_rhs)

        return t_array, states

    def integrate_stochastic(
        self,
        state0: np.ndarray,
        t_span: Tuple[float, float],
        n_steps: int = 1000,
        q_spectral: float = 1e-12
    ) -> Tuple[np.ndarray, np.ndarray]:
        t0, tf = t_span
        h = (tf - t0) / n_steps
        t_array = np.linspace(t0, tf, n_steps + 1)
        states = np.zeros((n_steps + 1, 6))
        states[0] = state0.copy()

        for i in range(n_steps):
            states[i + 1] = srk4_ti_step(
                states[i],
                t_array[i],
                h,
                q_spectral,
                self.stochastic_rhs,
                self.stochastic_drift
            )

        return t_array, states

    def integrate_adaptive_rk4(
        self,
        state0: np.ndarray,
        t_span: Tuple[float, float],
        atol: float = 1e-9,
        rtol: float = 1e-6,
        h0: float = 1.0,
        h_min: float = 1e-6,
        h_max: float = 1e4
    ) -> Tuple[List[float], List[np.ndarray]]:
        t0, tf = t_span
        t = t0
        state = state0.copy()
        h = h0
        t_list = [t]
        state_list = [state.copy()]

        while t < tf:
            h = min(h, tf - t)

            s1 = rk4_step(state, t, h, self.deterministic_rhs)

            s_half = rk4_step(state, t, h / 2.0, self.deterministic_rhs)
            s2 = rk4_step(s_half, t + h / 2.0, h / 2.0, self.deterministic_rhs)

            err = np.linalg.norm(s1 - s2)
            scale = atol + rtol * max(np.linalg.norm(s1), np.linalg.norm(s2))

            if err <= scale or h <= h_min:
                t += h
                state = s2.copy()
                t_list.append(t)
                state_list.append(state.copy())

                if err > 0:
                    h = min(h_max, h * 0.9 * (scale / err) ** 0.2)
                else:
                    h = min(h_max, 2.0 * h)
            else:
                h = max(h_min, h * 0.9 * (scale / err) ** 0.25)

        return np.array(t_list), np.array(state_list)
