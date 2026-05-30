
import numpy as np
from typing import Callable, Tuple
from utils import r8_uniform_01






def electron_drift_euler(
    e_field_fn: Callable[[np.ndarray], np.ndarray],
    r0: np.ndarray,
    t_span: Tuple[float, float],
    n_steps: int,
    mobility: float = 3.0e-4,
) -> Tuple[np.ndarray, np.ndarray]:
    t0, t1 = t_span
    if n_steps < 1:
        raise ValueError("electron_drift_euler: n_steps 必须 >= 1")
    h = (t1 - t0) / n_steps

    t_array = np.linspace(t0, t1, n_steps + 1)
    r_array = np.zeros((n_steps + 1, len(r0)))
    r_array[0] = r0

    for n in range(n_steps):
        E = e_field_fn(r_array[n])
        r_array[n + 1] = r_array[n] + h * mobility * E

    return t_array, r_array






class ScintillationODESystem:

    def __init__(
        self,
        gamma_p: float = 5.0e6,
        alpha_q: float = 1.0e5,
        beta_r: float = 1.0e4,
        kappa_pq: float = 2.0e-3,
        e_dep_norm: float = 1.0,
    ):
        self.gamma_p = gamma_p
        self.alpha_q = alpha_q
        self.beta_r = beta_r
        self.kappa_pq = kappa_pq
        self.e_dep_norm = e_dep_norm

    def deriv(self, t: float, y: np.ndarray) -> np.ndarray:
        P, Q = y
        dP = -self.gamma_p * P + self.alpha_q * Q * self.e_dep_norm
        dQ = self.beta_r - self.alpha_q * Q * self.e_dep_norm - self.kappa_pq * P * Q
        return np.array([dP, dQ])

    def solve_euler(
        self,
        y0: np.ndarray,
        t_span: Tuple[float, float],
        n_steps: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        t0, t1 = t_span
        h = (t1 - t0) / n_steps
        t = np.linspace(t0, t1, n_steps + 1)
        y = np.zeros((n_steps + 1, 2))
        y[0] = y0
        for i in range(n_steps):
            dydt = self.deriv(t[i], y[i])
            y[i + 1] = y[i] + h * dydt

            y[i + 1] = np.maximum(y[i + 1], 0.0)
        return t, y

    def equilibrium(self) -> np.ndarray:
        a = self.kappa_pq * self.alpha_q * self.e_dep_norm
        b = self.gamma_p * self.alpha_q * self.e_dep_norm
        c = -self.gamma_p * self.beta_r
        if abs(a) < 1e-20:
            Q_star = -c / b if abs(b) > 1e-20 else 0.0
        else:
            discriminant = b * b - 4.0 * a * c
            if discriminant < 0.0:
                discriminant = 0.0
            Q_star = (-b + np.sqrt(discriminant)) / (2.0 * a)
        P_star = (self.alpha_q * Q_star * self.e_dep_norm) / self.gamma_p
        return np.array([P_star, Q_star])






def lindhard_quenching_factor(er_kev: float, Z: int, A: int) -> float:
    if er_kev <= 0.0:
        return 0.0
    if Z <= 0 or A <= 0:
        raise ValueError("lindhard_quenching_factor: Z, A 必须为正")

    eps = 11.5 * (Z ** (-7.0 / 3.0)) * er_kev
    k = 0.133 * (Z ** (2.0 / 3.0)) * (A ** (-0.5))
    g_eps = 3.0 * (eps ** 0.15) + 0.7 * (eps ** 0.6) + eps
    Q = (k * g_eps) / (1.0 + k * g_eps)
    return float(np.clip(Q, 0.0, 1.0))


def ionization_yield(er_kev: float, Z: int, A: int, fano_factor: float = 0.15, epsilon_eV: float = 3.0) -> Tuple[float, float]:
    if er_kev <= 0.0:
        return 0.0, 0.0
    Q = lindhard_quenching_factor(er_kev, Z, A)
    energy_eV = er_kev * 1000.0
    N_e = (Q * energy_eV) / epsilon_eV
    sigma = np.sqrt(fano_factor * N_e) if N_e > 0.0 else 0.0
    return float(N_e), float(sigma)






def energy_deposition_profile(
    er_kev: float,
    detector_thickness_m: float,
    n_bins: int = 50,
    interaction_depth_m: float = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if detector_thickness_m <= 0.0:
        raise ValueError("energy_deposition_profile: 厚度必须为正")
    z = np.linspace(0.0, detector_thickness_m, n_bins)
    dz = z[1] - z[0]

    if interaction_depth_m is None:

        depth = detector_thickness_m * np.random.rand()
    else:
        depth = np.clip(interaction_depth_m, 0.0, detector_thickness_m)


    track_length_m = 1.0e-9 * (er_kev ** 1.7) * 10.0
    track_length_m = min(track_length_m, detector_thickness_m * 0.5)


    sigma_track = track_length_m / 2.355
    if sigma_track < dz:
        sigma_track = dz

    edep = np.exp(-0.5 * ((z - depth) / sigma_track) ** 2)
    edep = edep / (np.sum(edep) * dz) * er_kev
    return z, edep






if __name__ == "__main__":

    def const_e_field(r):
        return np.array([0.0, 0.0, 1.0e3])

    t, r = electron_drift_euler(const_e_field, np.array([0.0, 0.0, 0.0]), (0.0, 1.0e-6), 100, mobility=3.0e-4)
    expected_z = 3.0e-4 * 1.0e3 * 1.0e-6
    assert abs(r[-1, 2] - expected_z) < 1e-12, f"漂移距离偏差: {r[-1, 2]} vs {expected_z}"


    sys = ScintillationODESystem(e_dep_norm=1.0)
    eq = sys.equilibrium()
    assert np.all(eq >= 0.0), "稳态解出现负值"
    t, y = sys.solve_euler(eq * 0.1, (0.0, 1.0e-5), 500)
    assert np.all(y >= 0.0), "ODE 解出现负值"


    Q = lindhard_quenching_factor(10.0, 32, 73)
    assert 0.0 <= Q <= 1.0, f"QF 超出范围: {Q}"
    Q_low = lindhard_quenching_factor(0.1, 32, 73)
    assert Q_low < Q, "低能 QF 应小于高能 QF"


    N_e, sig = ionization_yield(10.0, 32, 73)
    assert N_e >= 0.0 and sig >= 0.0

    print("particle_transport.py: 所有自测通过")
