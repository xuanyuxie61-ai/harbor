
import numpy as np
from typing import Tuple, List, Optional


class EventSVDAnalyzer:

    def __init__(self, n_components: int = 10):
        self.n_components = n_components
        self.U = None
        self.S = None
        self.Vt = None
        self.mean_profile = None

    def fit(self, event_data: np.ndarray) -> 'EventSVDAnalyzer':
        data = np.asarray(event_data, dtype=float)
        if data.ndim != 2:
            raise ValueError("event_data必须是2维矩阵")


        self.mean_profile = np.mean(data, axis=1, keepdims=True)
        centered = data - self.mean_profile


        self.U, self.S, self.Vt = np.linalg.svd(centered, full_matrices=False)


        n_comp = min(self.n_components, len(self.S))
        self.U = self.U[:, :n_comp]
        self.S = self.S[:n_comp]
        self.Vt = self.Vt[:n_comp, :]

        return self

    def explained_variance_ratio(self) -> np.ndarray:
        if self.S is None:
            return np.array([])
        total = np.sum(self.S ** 2)
        if total < 1e-15:
            return np.zeros_like(self.S)
        return (self.S ** 2) / total

    def cumulative_variance(self) -> np.ndarray:
        ratios = self.explained_variance_ratio()
        return np.cumsum(ratios)

    def reconstruct(self, n_modes: Optional[int] = None) -> np.ndarray:
        if self.U is None or self.S is None or self.Vt is None:
            raise ValueError("请先调用fit()")
        n = n_modes if n_modes is not None else len(self.S)
        n = min(n, len(self.S))

        recon = self.U[:, :n] @ np.diag(self.S[:n]) @ self.Vt[:n, :]
        recon = recon + self.mean_profile
        return recon

    def project_event(self, event: np.ndarray) -> np.ndarray:
        if self.U is None or self.mean_profile is None:
            raise ValueError("请先调用fit()")
        event_centered = event - self.mean_profile.flatten()
        coeffs = self.U.T @ event_centered
        return coeffs

    def event_distance(self, event1: np.ndarray,
                       event2: np.ndarray) -> float:
        c1 = self.project_event(event1)
        c2 = self.project_event(event2)
        return float(np.linalg.norm(c1 - c2))

    def fluctuation_modes(self) -> np.ndarray:
        if self.U is None:
            raise ValueError("请先调用fit()")
        return self.U

    def event_weights(self) -> np.ndarray:
        if self.Vt is None:
            raise ValueError("请先调用fit()")
        return self.Vt


class FlowHarmonicDecomposition:

    @staticmethod
    def flow_vector(qn_x: float, qn_y: float) -> Tuple[float, float]:
        magnitude = np.sqrt(qn_x ** 2 + qn_y ** 2)
        psi_n = np.arctan2(qn_y, qn_x)
        return magnitude, psi_n

    @staticmethod
    def eccentricity_from_flow(v2: float, 
                                 response_coeff: float = 0.18) -> float:
        if response_coeff < 1e-15:
            return 0.0
        return v2 / response_coeff

    @staticmethod
    def cumulant_v2(particles_phi: np.ndarray) -> float:
        n = len(particles_phi)
        if n < 2:
            return 0.0
        cos_sum = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                cos_sum += np.cos(2.0 * (particles_phi[i] - particles_phi[j]))
        denom = n * (n - 1) / 2.0
        if denom < 1e-15:
            return 0.0
        v2_sq = cos_sum / denom
        return float(np.sqrt(max(v2_sq, 0.0)))

    @staticmethod
    def cumulant_v4(particles_phi: np.ndarray) -> float:
        n = len(particles_phi)
        if n < 4:
            return 0.0

        c2 = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                c2 += np.cos(4.0 * (particles_phi[i] - particles_phi[j]))
        c2 /= (n * (n - 1) / 2.0)


        c4 = 0.0
        count = 0
        for i in range(min(n, 20)):
            for j in range(i + 1, min(n, 20)):
                for k in range(j + 1, min(n, 20)):
                    for l in range(k + 1, min(n, 20)):
                        c4 += np.cos(4.0 * (particles_phi[i] - particles_phi[j] +
                                            particles_phi[k] - particles_phi[l]))
                        count += 1
        if count > 0:
            c4 /= count
        else:
            c4 = 0.0

        v4_4 = 2.0 * c2 ** 2 - c4
        return float(np.sign(v4_4) * (abs(v4_4) ** 0.25))

    def event_plane_resolution(self, n_subevents: int = 3) -> float:

        chi = 1.0
        from scipy.special import ive, iv

        try:
            r = np.sqrt(np.pi / 2.0) * chi * np.exp(-chi ** 2 / 2.0) * (
                iv(0, chi ** 2 / 2.0) + iv(1, chi ** 2 / 2.0)
            )
        except Exception:
            r = 0.7
        return float(r)
