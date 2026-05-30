
import numpy as np
from typing import Tuple, List, Optional


class ReactionCoordinateAnalyzer:

    def __init__(self):
        self.U = None
        self.S = None
        self.Vt = None
        self.mean = None
        self.n_frames = None
        self.n_dof = None

    def fit(self, trajectory: np.ndarray):
        traj = np.asarray(trajectory, dtype=float)
        if traj.ndim == 3:
            traj = traj.reshape(traj.shape[0], -1)
        self.n_frames, self.n_dof = traj.shape
        self.mean = np.mean(traj, axis=0)
        X_centered = traj - self.mean
        self.U, self.S, self.Vt = np.linalg.svd(X_centered, full_matrices=False)

    def variance_explained(self, n_components: Optional[int] = None) -> np.ndarray:
        if self.S is None:
            raise RuntimeError("必须先调用 fit()")
        var = self.S ** 2
        var_ratio = var / np.sum(var)
        if n_components is not None:
            return var_ratio[:n_components]
        return var_ratio

    def principal_components(self, trajectory: np.ndarray,
                             n_components: int = 3) -> np.ndarray:
        if self.Vt is None:
            raise RuntimeError("必须先调用 fit()")
        traj = np.asarray(trajectory, dtype=float)
        if traj.ndim == 3:
            traj = traj.reshape(traj.shape[0], -1)
        X_centered = traj - self.mean
        return X_centered @ self.Vt[:n_components].T

    def reaction_coordinate(self, trajectory: np.ndarray) -> np.ndarray:
        pcs = self.principal_components(trajectory, n_components=1)
        return pcs[:, 0]

    def reconstruct(self, coefficients: np.ndarray) -> np.ndarray:
        if self.Vt is None:
            raise RuntimeError("必须先调用 fit()")
        coeffs = np.asarray(coefficients, dtype=float)
        if coeffs.ndim == 1:
            coeffs = coeffs.reshape(1, -1)
        return self.mean + coeffs @ self.Vt[:coeffs.shape[1]]

    def free_energy_profile(self, reaction_coord: np.ndarray,
                            temperature_k: float = 500.0,
                            n_bins: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        from utils import BOLTZMANN_KB
        kb_t = BOLTZMANN_KB * temperature_k

        hist, bin_edges = np.histogram(reaction_coord, bins=n_bins, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])


        hist = np.maximum(hist, 1e-300)
        free_energy = -kb_t * np.log(hist)
        free_energy = free_energy - np.min(free_energy)
        return bin_centers, free_energy

    def commutator_analysis(self, trajectory: np.ndarray,
                            state_a_mask: np.ndarray,
                            state_b_mask: np.ndarray) -> np.ndarray:
        rc = self.reaction_coordinate(trajectory)
        n_bins = 50
        hist_a, bins = np.histogram(rc[state_a_mask], bins=n_bins, range=(rc.min(), rc.max()))
        hist_b, _ = np.histogram(rc[state_b_mask], bins=n_bins, range=(rc.min(), rc.max()))

        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        p_b = np.zeros(n_bins)
        for i in range(n_bins):
            total = hist_a[i] + hist_b[i]
            if total > 0:
                p_b[i] = hist_b[i] / total
        return bin_centers, p_b

    def collectivity_index(self, n_components: int = 3) -> float:
        if self.Vt is None:
            raise RuntimeError("必须先调用 fit()")
        v1 = self.Vt[0] ** 2
        v1 = v1 / np.sum(v1)
        v1 = np.maximum(v1, 1e-300)
        entropy = -np.sum(v1 * np.log(v1))
        kappa = np.exp(entropy) / len(v1)
        return float(kappa)


def generate_test_trajectory(n_atoms: int = 10, n_frames: int = 200) -> np.ndarray:
    rng = np.random.default_rng(42)
    traj = np.zeros((n_frames, n_atoms, 3))


    r_eq = np.zeros((n_atoms, 3))
    for i in range(n_atoms):
        angle = 2.0 * np.pi * i / n_atoms
        r_eq[i] = [np.cos(angle), np.sin(angle), 0.0]

    for t in range(n_frames):
        frac = t / n_frames

        if frac < 0.33:
            state = 0.0
            amp = 0.05
        elif frac < 0.67:
            state = (frac - 0.33) / 0.34
            amp = 0.15
        else:
            state = 1.0
            amp = 0.05


        r_prod = r_eq.copy()
        r_prod[:, 0] *= 1.2
        r_prod[:, 1] *= 0.8

        r_t = (1 - state) * r_eq + state * r_prod
        noise = rng.normal(0.0, amp, size=(n_atoms, 3))
        traj[t] = r_t + noise

    return traj
