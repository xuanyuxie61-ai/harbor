
import numpy as np
from typing import Tuple, List


class SVDDynamoAnalysis:

    def __init__(self, nr: int, ntheta: int):
        self.nr = nr
        self.ntheta = ntheta

    def decompose_field(
        self,
        field_2d: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if field_2d.shape != (self.nr, self.ntheta):
            raise ValueError(f"field_2d shape {field_2d.shape} does not match ({self.nr}, {self.ntheta})")

        U, S, Vt = np.linalg.svd(field_2d, full_matrices=False)
        return U, S, Vt

    def analyze_time_series(
        self,
        time_series: List[np.ndarray],
    ) -> dict:
        n_snapshots = len(time_series)
        if n_snapshots == 0:
            return {"error": "empty time series"}

        n_points = self.nr * self.ntheta
        X = np.zeros((n_points, n_snapshots))
        for i, snapshot in enumerate(time_series):
            X[:, i] = snapshot.reshape(-1)


        X_mean = np.mean(X, axis=1, keepdims=True)
        X_centered = X - X_mean


        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)


        total_energy = np.sum(S ** 2)
        energy_ratio = (S ** 2) / total_energy if total_energy > 1e-30 else np.zeros_like(S)
        cumulative_energy = np.cumsum(energy_ratio)


        rank_dominant = int(np.searchsorted(cumulative_energy, 0.95)) + 1

        return {
            "singular_values": S,
            "energy_ratio": energy_ratio,
            "cumulative_energy": cumulative_energy,
            "rank_95": rank_dominant,
            "spatial_modes": U,
            "temporal_coeffs": Vt,
            "mean_field": X_mean.reshape(self.nr, self.ntheta),
        }

    def dipole_quadrupole_analysis(
        self,
        br_field: np.ndarray,
        r_grid: np.ndarray,
        theta_grid: np.ndarray,
    ) -> dict:

        br_cmb = br_field[-1, :]
        x = np.cos(theta_grid)


        max_l = 8
        coeffs = []
        for l in range(max_l + 1):

            pl = np.polynomial.legendre.legvander(x, max_l)[:, l]

            integrand = br_cmb * pl * np.sin(theta_grid)
            coeff = (2.0 * l + 1.0) / 2.0 * np.trapz(integrand, x)
            coeffs.append(coeff)

        coeffs = np.array(coeffs)
        norm = np.linalg.norm(coeffs)
        if norm < 1e-30:
            norm = 1.0

        dipole_ratio = abs(coeffs[1]) / norm if len(coeffs) > 1 else 0.0
        quadrupole_ratio = abs(coeffs[2]) / norm if len(coeffs) > 2 else 0.0
        octupole_ratio = abs(coeffs[3]) / norm if len(coeffs) > 3 else 0.0

        return {
            "legendre_coeffs": coeffs,
            "dipole_ratio": dipole_ratio,
            "quadrupole_ratio": quadrupole_ratio,
            "octupole_ratio": octupole_ratio,
            "dipole_tilt": np.arctan2(coeffs[1].imag if np.iscomplexobj(coeffs) else 0.0, coeffs[1].real if np.iscomplexobj(coeffs) else coeffs[1]),
        }

    def field_anisotropy_tensor(
        self,
        field_r: np.ndarray,
        field_theta: np.ndarray,
        field_phi: np.ndarray,
    ) -> np.ndarray:
        b2 = field_r ** 2 + field_theta ** 2 + field_phi ** 2
        b2_avg = np.mean(b2)
        if b2_avg < 1e-30:
            return np.eye(3) / 3.0

        M = np.zeros((3, 3))
        components = [field_r, field_theta, field_phi]
        for i in range(3):
            for j in range(3):
                M[i, j] = np.mean(components[i] * components[j]) / b2_avg


        M = 0.5 * (M + M.T)
        eigvals = np.linalg.eigvalsh(M)
        return eigvals[::-1]

    def snapshot_pod_reconstruction(
        self,
        time_series: List[np.ndarray],
        rank: int,
    ) -> Tuple[List[np.ndarray], float]:
        result = self.analyze_time_series(time_series)
        U = result["spatial_modes"]
        S = result["singular_values"]
        Vt = result["temporal_coeffs"]
        X_mean = result["mean_field"].reshape(-1, 1)

        n_snapshots = len(time_series)
        rank = min(rank, len(S))


        Ur = U[:, :rank]
        Sr = np.diag(S[:rank])
        Vr = Vt[:rank, :]

        X_recon = X_mean + Ur @ Sr @ Vr

        reconstructed = []
        original = np.zeros((self.nr * self.ntheta, n_snapshots))
        for i, snap in enumerate(time_series):
            original[:, i] = snap.reshape(-1)
            reconstructed.append(X_recon[:, i].reshape(self.nr, self.ntheta))

        frob_error = np.linalg.norm(original - X_recon, "fro")
        frob_original = np.linalg.norm(original, "fro")
        rel_error = frob_error / frob_original if frob_original > 1e-30 else 0.0

        return reconstructed, rel_error
