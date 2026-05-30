
import numpy as np
from typing import Tuple, Optional
from numerical_utils import safe_inverse_sqrt, rotation_matrix_y, rotation_matrix_z


C_LIGHT = 2.99792458e8
MU_0 = 4.0 * np.pi * 1e-7
EPS_0 = 8.854187817e-12
ETA_0 = np.sqrt(MU_0 / EPS_0)


class ArrayFactorCalculator:

    def __init__(self, element_positions: np.ndarray,
                 frequency_hz: float = 3.0e9,
                 element_weights: Optional[np.ndarray] = None):
        self.positions = np.asarray(element_positions, dtype=float)
        self.n_elements = self.positions.shape[0]
        self.frequency = frequency_hz
        self.wavelength = C_LIGHT / frequency_hz
        self.k0 = 2.0 * np.pi / self.wavelength
        if element_weights is None:
            self.weights = np.ones(self.n_elements, dtype=complex) / self.n_elements
        else:
            self.weights = np.asarray(element_weights, dtype=complex)
            if self.weights.size != self.n_elements:
                raise ValueError("权重数量必须与单元数一致")

    def _direction_cosines(self, theta: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        u = np.sin(theta) * np.cos(phi)
        v = np.sin(theta) * np.sin(phi)
        w = np.cos(theta)
        return u, v, w

    def compute_array_factor(self, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:



        raise NotImplementedError("Hole 1: 阵列方向图因子 compute_array_factor 待实现")

    def compute_element_pattern_dipole(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        sin_t = np.sin(theta)
        sin_t_safe = np.where(np.abs(sin_t) > 1e-6, sin_t, 1e-6)
        pat = np.cos(0.5 * np.pi * np.cos(theta)) / sin_t_safe

        pat = np.where(np.abs(sin_t) > 1e-6, pat, 0.0)
        return pat

    def compute_total_pattern(self, theta: np.ndarray, phi: np.ndarray,
                              normalize: bool = True) -> np.ndarray:
        af = self.compute_array_factor(theta, phi)
        ep = self.compute_element_pattern_dipole(theta)
        pattern = np.abs(af * ep)
        if normalize and np.max(pattern) > 0:
            pattern /= np.max(pattern)
        return pattern

    def compute_total_pattern_db(self, theta: np.ndarray, phi: np.ndarray,
                                 normalize: bool = True,
                                 floor_db: float = -80.0) -> np.ndarray:
        pat = self.compute_total_pattern(theta, phi, normalize)
        pat_db = 20.0 * np.log10(np.maximum(pat, 10.0 ** (floor_db / 20.0)))
        return pat_db

    def apply_steering(self, theta_s: float, phi_s: float = 0.0):
        u_s = np.sin(theta_s) * np.cos(phi_s)
        v_s = np.sin(theta_s) * np.sin(phi_s)
        w_s = np.cos(theta_s)
        phase = -self.k0 * (self.positions[:, 0] * u_s
                          + self.positions[:, 1] * v_s
                          + self.positions[:, 2] * w_s)
        self.weights = np.exp(1j * phase)

        self.weights /= np.linalg.norm(self.weights)

    def apply_chebyshev_weights(self, sidelobe_db: float = -30.0):

        n = self.n_elements
        idx = np.arange(n)
        alpha = np.arccosh(10.0 ** (-sidelobe_db / 20.0))

        x = (idx - (n - 1) / 2.0) / ((n - 1) / 2.0)
        amplitude = np.cosh(alpha * np.sqrt(np.maximum(1.0 - x ** 2, 0.0)))
        amplitude = np.maximum(amplitude, 1e-6)
        self.weights = amplitude * np.exp(1j * np.angle(self.weights))
        self.weights /= np.linalg.norm(self.weights)

    def directivity(self, theta_grid: int = 181, phi_grid: int = 360) -> float:
        theta = np.linspace(0.0, np.pi, theta_grid)
        phi = np.linspace(0.0, 2.0 * np.pi, phi_grid)
        theta_m, phi_m = np.meshgrid(theta, phi, indexing='ij')
        pat = self.compute_total_pattern(theta_m.ravel(), phi_m.ravel(), normalize=False)
        pat_sq = np.abs(pat) ** 2

        dtheta = np.pi / (theta_grid - 1)
        dphi = 2.0 * np.pi / (phi_grid - 1)
        integrand = pat_sq.reshape(theta_grid, phi_grid) * np.sin(theta_m)
        total = np.sum(integrand) * dtheta * dphi
        p_max = np.max(pat_sq)
        D = 4.0 * np.pi * p_max / max(total, 1e-18)
        return float(D)


class MutualCouplingMatrix:

    def __init__(self, element_positions: np.ndarray, frequency_hz: float = 3.0e9):
        self.positions = np.asarray(element_positions, dtype=float)
        self.n_elements = self.positions.shape[0]
        self.frequency = frequency_hz
        self.wavelength = C_LIGHT / frequency_hz
        self.k0 = 2.0 * np.pi / self.wavelength
        self.half_length = self.wavelength / 4.0

    def _ci_si_approx(self, x: float) -> Tuple[float, float]:
        gamma = 0.5772156649015329
        x = float(x)
        if x < 1e-12:
            return -1e6, 0.0
        if x < 2.0:

            ci = gamma + np.log(x)
            si = 0.0
            term_ci = 1.0
            term_si = x
            for n in range(1, 15):
                term_ci *= -x * x / ((2 * n - 1) * (2 * n))
                term_si *= -x * x / ((2 * n) * (2 * n + 1))
                ci += term_ci / (2 * n)
                si += term_si / (2 * n + 1)
            return ci, si
        else:

            ci = np.sin(x) / x - np.cos(x) / (x * x)
            si = 0.5 * np.pi - np.cos(x) / x - np.sin(x) / (x * x)
            return ci, si

    def compute_mutual_impedance(self, i: int, j: int) -> complex:
        if i == j:
            return 73.1 + 42.5j
        d_vec = self.positions[i, :] - self.positions[j, :]
        d = np.linalg.norm(d_vec)
        d = max(d, 1e-6)
        l = self.half_length
        k0 = self.k0

        ci_d, si_d = self._ci_si_approx(k0 * d)
        arg1 = k0 * (np.sqrt(d ** 2 + l ** 2) + l)
        arg2 = k0 * (np.sqrt(d ** 2 + l ** 2) - l)
        ci1, si1 = self._ci_si_approx(arg1)
        ci2, si2 = self._ci_si_approx(arg2)

        R = 30.0 * (2.0 * ci_d - ci1 - ci2)
        X = -30.0 * (2.0 * si_d - si1 - si2)
        return R + 1j * X

    def build_impedance_matrix(self) -> np.ndarray:
        Z = np.zeros((self.n_elements, self.n_elements), dtype=complex)
        for i in range(self.n_elements):
            for j in range(i, self.n_elements):
                z_ij = self.compute_mutual_impedance(i, j)
                Z[i, j] = z_ij
                if i != j:
                    Z[j, i] = z_ij
        return Z

    def active_reflection_coefficient(self, port_idx: int,
                                      port_voltages: np.ndarray) -> complex:
        Z = self.build_impedance_matrix()
        I = np.asarray(port_voltages, dtype=complex)
        I_i = I[port_idx]
        if abs(I_i) < 1e-18:
            return 0.0
        Z_in = Z[port_idx, port_idx]
        for j in range(self.n_elements):
            if j != port_idx:
                Z_in += (I[j] / I_i) * Z[port_idx, j]
        Z0 = 50.0
        gamma = (Z_in - Z0) / (Z_in + Z0)
        return gamma


def near_field_e_field(positions: np.ndarray,
                       currents: np.ndarray,
                       observation_points: np.ndarray,
                       frequency_hz: float = 3.0e9) -> np.ndarray:
    k0 = 2.0 * np.pi * frequency_hz / C_LIGHT
    eta0 = ETA_0
    N_obs = observation_points.shape[0]
    E = np.zeros((N_obs, 3), dtype=complex)

    for n in range(positions.shape[0]):
        R_vec = observation_points - positions[n, :]
        R = np.linalg.norm(R_vec, axis=1)
        R_safe = np.maximum(R, 1e-6)
        kR = k0 * R_safe

        phase = np.exp(-1j * kR) / (4.0 * np.pi * R_safe)

        E_scalar = -1j * eta0 * k0 * currents[n] * phase


        cos_theta = R_vec[:, 2] / R_safe
        sin_theta = np.sqrt(np.maximum(1.0 - cos_theta ** 2, 0.0))
        E_amp = E_scalar * sin_theta


        E[:, 2] += E_amp

    return E
