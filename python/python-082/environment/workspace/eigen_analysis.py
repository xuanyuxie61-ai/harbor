# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse.linalg import eigsh, eigs
from scipy.sparse import csr_matrix
from typing import Optional, Tuple


class StructuralModalAnalysis:

    def __init__(self, K: csr_matrix, M: Optional[csr_matrix] = None):
        self.K = K
        self.n = K.shape[0]
        if M is None:
            from scipy.sparse import eye
            self.M = eye(self.n, format='csr')
        else:
            self.M = M

    def compute_modes(self, num_modes: int = 10,
                      sigma: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        if num_modes >= self.n - 1:

            K_dense = self.K.toarray()
            M_dense = self.M.toarray()
            eigenvalues, eigenvectors = np.linalg.eig(np.linalg.solve(M_dense, K_dense))

            eigenvalues = np.real(eigenvalues)
            idx = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[idx[:num_modes]]
            eigenvectors = eigenvectors[:, idx[:num_modes]]
        else:


            if sigma is None:
                sigma = 0.0
            try:
                eigenvalues, eigenvectors = eigsh(self.K, k=num_modes,
                                                   M=self.M, sigma=sigma,
                                                   which='LM', mode='normal')
            except Exception:

                eigenvalues, eigenvectors = eigsh(self.K, k=min(num_modes, self.n - 2),
                                                   which='SM')

        eigenvalues = np.where(eigenvalues < 1e-12, 0.0, eigenvalues)

        for i in range(eigenvectors.shape[1]):
            norm = np.sqrt(eigenvectors[:, i].T @ self.M @ eigenvectors[:, i])
            if norm > 1e-30:
                eigenvectors[:, i] /= norm
        return eigenvalues, eigenvectors

    def natural_frequencies(self, num_modes: int = 10) -> np.ndarray:
        eigenvalues, _ = self.compute_modes(num_modes)
        return np.sqrt(eigenvalues) / (2.0 * np.pi)

    def frequency_shift_due_to_damage(self, K_damaged: csr_matrix,
                                       num_modes: int = 5) -> np.ndarray:
        f_0 = self.natural_frequencies(num_modes)
        damaged_analyzer = StructuralModalAnalysis(K_damaged, self.M)
        f_d = damaged_analyzer.natural_frequencies(num_modes)
        shift = (f_d - f_0) / (f_0 + 1e-30) * 100.0
        return shift

    def instability_index(self) -> float:
        try:
            lam_min, _ = eigsh(self.K, k=1, which='SM')
            return float(lam_min[0])
        except Exception:

            vals = np.linalg.eigvalsh(self.K.toarray())
            return float(np.min(vals))

    def modal_participation_factor(self, force_pattern: np.ndarray,
                                    num_modes: int = 10) -> np.ndarray:
        _, phi = self.compute_modes(num_modes)
        return phi.T @ force_pattern


class TestMatrixGenerator:

    @staticmethod
    def symmetric_with_eigenvalues(eigenvalues: np.ndarray) -> np.ndarray:
        n = len(eigenvalues)

        X = np.random.randn(n, n)
        Q, _ = np.linalg.qr(X)
        Lambda = np.diag(eigenvalues)
        A = Q @ Lambda @ Q.T
        return A

    @staticmethod
    def nonsymmetric_with_eigenvalues(eigenvalues: np.ndarray) -> np.ndarray:
        n = len(eigenvalues)
        T = np.triu(np.random.randn(n, n), k=1)
        np.fill_diagonal(T, eigenvalues)
        X = np.random.randn(n, n)
        Q, _ = np.linalg.qr(X)
        A = Q.T @ T @ Q
        return A

    @staticmethod
    def damaged_stiffness_spectrum(n: int, damage_level: float) -> np.ndarray:
        lam_base = np.linspace(1.0, n ** 2, n)

        damage_factor = 1.0 - damage_level * np.exp(-np.arange(n) / (n / 5.0))
        damage_factor = np.clip(damage_factor, 0.1, 1.0)
        return lam_base * damage_factor


class StabilityRegionAnalysis:

    @staticmethod
    def amplification_factor_lserk45(z: complex) -> complex:
        return (1.0 + z + z ** 2 / 2.0 + z ** 3 / 6.0
                + z ** 4 / 24.0 + z ** 5 / 120.0)

    @staticmethod
    def amplification_factor_explicit_euler(z: complex) -> complex:
        return 1.0 + z

    @staticmethod
    def amplification_factor_implicit_euler(z: complex) -> complex:
        return 1.0 / (1.0 - z + 1e-30)

    @staticmethod
    def compute_boundary_locus(method: str = 'lserk45',
                                num_points: int = 400,
                                z_max: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
        if method == 'lserk45':
            R = StabilityRegionAnalysis.amplification_factor_lserk45
        elif method == 'explicit_euler':
            R = StabilityRegionAnalysis.amplification_factor_explicit_euler
        elif method == 'implicit_euler':
            R = StabilityRegionAnalysis.amplification_factor_implicit_euler
        else:
            raise ValueError(f"Unknown method: {method}")

        thetas = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)
        x_boundary = np.zeros(num_points)
        y_boundary = np.zeros(num_points)

        for i, theta in enumerate(thetas):

            r_low, r_high = 0.0, z_max * 2.0
            for _ in range(50):
                r_mid = (r_low + r_high) / 2.0
                z = r_mid * np.exp(1j * theta)
                mag = abs(R(z))
                if mag > 1.0:
                    r_high = r_mid
                else:
                    r_low = r_mid
            r_opt = (r_low + r_high) / 2.0
            z_bound = r_opt * np.exp(1j * theta)
            x_boundary[i] = z_bound.real
            y_boundary[i] = z_bound.imag

        return x_boundary, y_boundary

    @staticmethod
    def cfl_limit_estimate(wave_speed: float, dx_min: float,
                           poly_order: int, method: str = 'lserk45') -> float:
        if method == 'lserk45':
            cfl_coeff = 0.5 / (poly_order ** 1.5 + 1e-30)
        elif method == 'explicit_euler':
            cfl_coeff = 0.1 / (poly_order ** 2 + 1e-30)
        else:
            cfl_coeff = 1e6
        return cfl_coeff * dx_min / (wave_speed + 1e-30)


if __name__ == "__main__":

    eigenvals = np.array([1.0, 2.0, 5.0, 10.0, 20.0])
    A_sym = TestMatrixGenerator.symmetric_with_eigenvalues(eigenvals)
    computed_eigs = np.sort(np.linalg.eigvalsh(A_sym))
    print("Symmetric test matrix eigenvalues:", computed_eigs)
    assert np.allclose(computed_eigs, np.sort(eigenvals), atol=1e-10)


    x_b, y_b = StabilityRegionAnalysis.compute_boundary_locus('lserk45', num_points=100)
    print("LSERK45 stability boundary sample:", x_b[0], y_b[0])


    from stiffness_matrix import StiffnessMatrixAssembler1D
    nodes = np.linspace(0.0, 1.0, 51)
    assembler = StiffnessMatrixAssembler1D(nodes, A=1e-4, E0=100e9)
    K = assembler.assemble_global_stiffness()
    modal = StructuralModalAnalysis(K)
    freqs = modal.natural_frequencies(num_modes=5)
    print("Natural frequencies (Hz):", freqs)
    print("Instability index:", modal.instability_index())
