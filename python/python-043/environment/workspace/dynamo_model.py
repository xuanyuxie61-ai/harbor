
import numpy as np
from typing import Tuple, Callable, Optional


class DynamoModel:

    def __init__(
        self,
        r_inner: float,
        r_outer: float,
        nr: int,
        ntheta: int,
        eta: float,
        c_omega: float,
        c_alpha: float,
        b_eq: float,
    ):
        self.r_inner = r_inner
        self.r_outer = r_outer
        self.nr = nr
        self.ntheta = ntheta
        self.eta = eta
        self.c_omega = c_omega
        self.c_alpha = c_alpha
        self.b_eq = b_eq


        self.r = np.linspace(r_inner, r_outer, nr)
        self.theta = np.linspace(0.0, np.pi, ntheta)
        self.dr = self.r[1] - self.r[0] if nr > 1 else 1.0
        self.dtheta = self.theta[1] - self.theta[0] if ntheta > 1 else 1.0


        self._precompute_metrics()


        self._setup_velocity_field()

    def _precompute_metrics(self):
        self.R, self.Theta = np.meshgrid(self.r, self.theta, indexing="ij")
        self.SinTheta = np.sin(self.Theta)
        self.CosTheta = np.cos(self.Theta)
        self.CotTheta = self.CosTheta / (self.SinTheta + 1e-30)
        self.R2 = self.R ** 2

    def _setup_velocity_field(self):
        r_m = 0.5 * (self.r_inner + self.r_outer)
        sigma = 0.25 * (self.r_outer - self.r_inner)


        radial_envelope = (self.R - self.r_inner) * (self.r_outer - self.R)
        radial_envelope = np.maximum(radial_envelope, 0.0)


        self.Omega = self.c_omega * radial_envelope * self.SinTheta ** 2


        gaussian = np.exp(-((self.R - r_m) ** 2) / (2.0 * sigma ** 2))
        self.alpha_base = self.c_alpha * self.CosTheta * radial_envelope * gaussian


        self._compute_velocity_gradients()

    def _compute_velocity_gradients(self):

        self.dOmega_dr = np.zeros_like(self.Omega)
        self.dOmega_dr[1:-1, :] = (self.Omega[2:, :] - self.Omega[:-2, :]) / (2.0 * self.dr)

        self.dOmega_dtheta = np.zeros_like(self.Omega)
        self.dOmega_dtheta[:, 1:-1] = (self.Omega[:, 2:] - self.Omega[:, :-2]) / (2.0 * self.dtheta)

    def _diffusion_operator(self, F: np.ndarray) -> np.ndarray:
        D2F = np.zeros_like(F)


        d2F_dr2 = np.zeros_like(F)
        d2F_dr2[1:-1, :] = (F[2:, :] - 2.0 * F[1:-1, :] + F[:-2, :]) / (self.dr ** 2)


        d2F_dtheta2 = np.zeros_like(F)
        d2F_dtheta2[:, 1:-1] = (F[:, 2:] - 2.0 * F[:, 1:-1] + F[:, :-2]) / (self.dtheta ** 2)


        dF_dtheta = np.zeros_like(F)
        dF_dtheta[:, 1:-1] = (F[:, 2:] - F[:, :-2]) / (2.0 * self.dtheta)


        D2F = d2F_dr2 + (1.0 / self.R2) * d2F_dtheta2 - (self.CotTheta / self.R2) * dF_dtheta

        return D2F

    def _alpha_quenching(self, T: np.ndarray, t: float = 0.0) -> np.ndarray:
        denominator = 1.0 + (T / (self.R * self.SinTheta * self.b_eq + 1e-30)) ** 2
        alpha_eff = self.alpha_base / denominator

        modulation = 1.0 + 0.15 * np.sin(2.0 * np.pi * t / 3.0) * self.CosTheta
        return alpha_eff * modulation

    def _jacobian_bracket(self, S: np.ndarray) -> np.ndarray:
        dS_dr = np.zeros_like(S)
        dS_dr[1:-1, :] = (S[2:, :] - S[:-2, :]) / (2.0 * self.dr)

        dS_dtheta = np.zeros_like(S)
        dS_dtheta[:, 1:-1] = (S[:, 2:] - S[:, :-2]) / (2.0 * self.dtheta)

        jac = dS_dr * self.dOmega_dtheta - dS_dtheta * self.dOmega_dr
        return jac

    def _alpha_induction_terms(self, S: np.ndarray, alpha: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dS_dr = np.zeros_like(S)
        dS_dr[1:-1, :] = (S[2:, :] - S[:-2, :]) / (2.0 * self.dr)

        dS_dtheta = np.zeros_like(S)
        dS_dtheta[:, 1:-1] = (S[:, 2:] - S[:, :-2]) / (2.0 * self.dtheta)


        term_r = np.zeros_like(S)
        r_alpha_dSdr = self.R * alpha * dS_dr
        term_r[1:-1, :] = -(r_alpha_dSdr[2:, :] - r_alpha_dSdr[:-2, :]) / (2.0 * self.dr * self.R[1:-1, :])

        term_theta = np.zeros_like(S)
        alpha_dSdtheta = alpha * dS_dtheta
        term_theta[:, 1:-1] = -(alpha_dSdtheta[:, 2:] - alpha_dSdtheta[:, :-2]) / (
            2.0 * self.dtheta * self.R2[:, 1:-1]
        )

        return term_r + term_theta

    def rhs(self, t: float, state: np.ndarray) -> np.ndarray:
        n = self.nr * self.ntheta
        S = state[:n].reshape((self.nr, self.ntheta))
        T = state[n:].reshape((self.nr, self.ntheta))


        alpha = self._alpha_quenching(T, t)


        D2S = self._diffusion_operator(S)
        D2T = self._diffusion_operator(T)


        jac = self._jacobian_bracket(S)


        alpha_ind = self._alpha_induction_terms(S, alpha)


        dSdt = self.eta * D2S + alpha * T


        dTdt = self.eta * D2T - self.SinTheta * jac + alpha_ind


        dSdt[0, :] = 0.0
        dSdt[-1, :] = 0.0
        dSdt[:, 0] = 0.0
        dSdt[:, -1] = 0.0

        dTdt[0, :] = 0.0
        dTdt[-1, :] = 0.0
        dTdt[:, 0] = 0.0
        dTdt[:, -1] = 0.0

        return np.concatenate([dSdt.reshape(-1), dTdt.reshape(-1)])

    def compute_magnetic_energy(self, S: np.ndarray, T: np.ndarray) -> float:


        dS_dr = np.zeros_like(S)
        dS_dr[1:-1, :] = (S[2:, :] - S[:-2, :]) / (2.0 * self.dr)

        dS_dtheta = np.zeros_like(S)
        dS_dtheta[:, 1:-1] = (S[:, 2:] - S[:, :-2]) / (2.0 * self.dtheta)

        Br = dS_dtheta / (self.R2 * self.SinTheta + 1e-30)
        Btheta = -dS_dr / (self.R + 1e-30)
        Bphi = T / (self.R * self.SinTheta + 1e-30)

        B2 = Br ** 2 + Btheta ** 2 + Bphi ** 2


        dV = 2.0 * np.pi * self.R2 * self.SinTheta * self.dr * self.dtheta
        energy = 0.5 * np.sum(B2 * dV)
        return float(energy)

    def compute_dipole_moment(self, S: np.ndarray) -> float:
        s_cmb = S[-1, :]
        mid = self.ntheta // 2
        return float(s_cmb[mid] - 0.5 * (s_cmb[0] + s_cmb[-1]))

    def initial_condition(self, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        S = np.zeros((self.nr, self.ntheta))
        T = np.zeros((self.nr, self.ntheta))


        amplitude = 0.01
        S[2:-2, 2:-2] = amplitude * rng.normal(size=(self.nr - 4, self.ntheta - 4))
        T[2:-2, 2:-2] = amplitude * rng.normal(size=(self.nr - 4, self.ntheta - 4))


        S[0, :] = S[-1, :] = S[:, 0] = S[:, -1] = 0.0
        T[0, :] = T[-1, :] = T[:, 0] = T[:, -1] = 0.0

        return np.concatenate([S.reshape(-1), T.reshape(-1)])

    def to_2d(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = self.nr * self.ntheta
        S = state[:n].reshape((self.nr, self.ntheta))
        T = state[n:].reshape((self.nr, self.ntheta))
        return S, T
