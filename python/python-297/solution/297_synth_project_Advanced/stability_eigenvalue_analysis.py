"""
stability_eigenvalue_analysis.py
==================================
Eigenvalue-based stability analysis for the dusty plasma crystal.

Physical motivation:
    The stability of the dusty plasma crystal is determined by the
    eigenvalues of the dynamical matrix D:
        D * u = omega^2 * u

    The crystal is stable if all eigenvalues omega^2 > 0 (all modes
    are oscillatory). Instability occurs when:
        1. omega^2 < 0 for some mode  (exponential growth)
        2. omega^2 = 0 for some mode  (marginal stability / Goldstone mode)

    For a 2D dusty plasma crystal, the relevant stability criteria are:
        1. Born stability criteria for hexagonal lattice
        2. Phonon spectrum positivity
        3. Energy convexity

    The eigenvalue problem is solved using:
        1. Direct diagonalization (for small systems)
        2. Power iteration (for largest eigenvalue)
        3. Inverse iteration (for specific eigenvalue)

    Condition number analysis reveals the sensitivity of the crystal
    to perturbations.

References:
    - Born & Huang, "Dynamical Theory of Crystal Lattices" (1954)
    - Trefethen & Embree, "Spectra and Pseudospectra" (2005)
"""

import numpy as np
from typing import Tuple, Dict, Optional, List


class CrystalStabilityAnalyzer:
    """
    Stability analysis for a dusty plasma crystal via eigenvalue methods.
    """

    def __init__(
        self,
        dynamical_matrix: np.ndarray,
        n_dof: int = 0,
    ):
        """
        Parameters
        ----------
        dynamical_matrix : np.ndarray, shape (N, N)
            The dynamical matrix D.
        n_dof : int
            Number of physical degrees of freedom (= N for displacement space).
        """
        self.D = np.array(dynamical_matrix, dtype=np.float64)
        self.N = self.D.shape[0]
        self.n_dof = n_dof if n_dof > 0 else self.N

        # Eigenvalues and eigenvectors (computed on demand)
        self._eigenvalues = None
        self._eigenvectors = None

    def compute_eigenvalues(self) -> np.ndarray:
        """
        Compute all eigenvalues of the dynamical matrix.

        For a symmetric positive-definite dynamical matrix, all
        eigenvalues should be positive (stable crystal).

        Returns
        -------
        eigenvalues : np.ndarray, shape (N,)
            Sorted eigenvalues (ascending).
        """
        # Use symmetric eigensolver (faster and more stable)
        if np.allclose(self.D, self.D.T, atol=1e-10):
            self._eigenvalues, self._eigenvectors = np.linalg.eigh(self.D)
        else:
            # Non-symmetric: use general eigensolver
            self._eigenvalues = np.linalg.eigvals(self.D)
            self._eigenvectors = None

        # Sort
        idx = np.argsort(self._eigenvalues.real)
        self._eigenvalues = self._eigenvalues[idx].real
        return self._eigenvalues

    def stability_report(self) -> Dict:
        """
        Generate comprehensive stability report.

        Returns
        -------
        report : dict
            'is_stable': bool,
            'n_negative_modes': number of unstable modes,
            'min_eigenvalue': smallest eigenvalue,
            'max_eigenvalue': largest eigenvalue,
            'condition_number': cond(D),
            'spectral_gap': gap between 0 and first non-zero mode,
            'mode_frequencies': omega = sqrt(max(eigenvalue, 0)).
        """
        eigenvalues = self.compute_eigenvalues()

        n_negative = int(np.sum(eigenvalues < -1e-10))
        n_zero = int(np.sum(np.abs(eigenvalues) < 1e-10))

        # Physical frequencies
        omega_sq = np.maximum(eigenvalues, 0.0)
        omega = np.sqrt(omega_sq)

        # Condition number (avoiding zero modes)
        nonzero_eigs = eigenvalues[np.abs(eigenvalues) > 1e-10]
        if len(nonzero_eigs) > 0:
            cond = abs(nonzero_eigs[-1] / nonzero_eigs[0])
        else:
            cond = float('inf')

        # Spectral gap
        sorted_pos = np.sort(eigenvalues[eigenvalues > 1e-10])
        spectral_gap = float(sorted_pos[0]) if len(sorted_pos) > 0 else 0.0

        return {
            "is_stable": n_negative == 0,
            "n_negative_modes": n_negative,
            "n_zero_modes": n_zero,
            "min_eigenvalue": float(eigenvalues[0]),
            "max_eigenvalue": float(eigenvalues[-1]),
            "condition_number": float(cond) if cond != float('inf') else 1e30,
            "spectral_gap": spectral_gap,
            "mode_frequencies": omega,
            "eigenvalues": eigenvalues,
        }

    def born_stability_criteria_2d(
        self,
        elastic_constants: Optional[Dict[str, float]] = None,
    ) -> Dict[str, bool]:
        """
        Check Born stability criteria for 2D hexagonal crystal.

        For a 2D hexagonal lattice, the elastic constants satisfy:
            C11 = C22  (hexagonal symmetry)
            C12 = C11 - 2*C66
            C66 = (C11 - C12) / 2

        Born stability criteria:
            C11 > 0
            C11 > |C12|
            C66 > 0

        Equivalently:
            C11 > 0, C11 > C12, C12 > -C11

        Parameters
        ----------
        elastic_constants : dict, optional
            {'C11', 'C12', 'C66'} from lattice dynamics.

        Returns
        -------
        criteria : dict
            Individual criterion results.
        """
        if elastic_constants is None:
            # Estimate from dynamical matrix
            elastic_constants = self._estimate_elastic_constants()

        C11 = elastic_constants.get("C11", 1.0)
        C12 = elastic_constants.get("C12", 0.3)
        C66 = elastic_constants.get("C66", (C11 - C12) / 2.0)

        return {
            "C11_positive": C11 > 0,
            "C66_positive": C66 > 0,
            "C11_gt_C12": C11 > C12,
            "C12_gt_neg_C11": C12 > -C11,
            "bulk_modulus_positive": (C11 + C12) > 0,
            "all_satisfied": (
                C11 > 0 and C66 > 0 and C11 > C12 and C12 > -C11
            ),
            "C11": C11,
            "C12": C12,
            "C66": C66,
        }

    def _estimate_elastic_constants(self) -> Dict[str, float]:
        """
        Estimate elastic constants from the dynamical matrix.

        For a simple 1D chain, C11 ~ D[0,0] (diagonal element).
        For 2D, we need to extract the appropriate combinations.

        Returns
        -------
        constants : dict
            Estimated elastic constants.
        """
        N = self.N
        if N >= 4:
            # Estimate from diagonal and off-diagonal blocks
            C11 = float(self.D[0, 0])
            C12 = float(self.D[0, 2]) if N > 2 else 0.0
            C66 = (C11 - C12) / 2.0
        else:
            C11 = float(self.D[0, 0])
            C12 = 0.0
            C66 = C11 / 2.0

        return {"C11": C11, "C12": C12, "C66": C66}

    def power_iteration(
        self,
        n_iter: int = 1000,
        tol: float = 1e-10,
    ) -> Tuple[float, np.ndarray]:
        """
        Power iteration for the largest eigenvalue.

        Parameters
        ----------
        n_iter : int
            Maximum iterations.
        tol : float
            Convergence tolerance.

        Returns
        -------
        lambda_max : float
            Largest eigenvalue.
        v_max : np.ndarray
            Corresponding eigenvector.
        """
        v = np.random.RandomState(42).random(self.N)
        v /= np.linalg.norm(v)

        for _ in range(n_iter):
            w = self.D @ v
            lambda_new = np.dot(v, w)
            v_new = w / np.linalg.norm(w)

            if abs(lambda_new - np.dot(v_new, self.D @ v_new)) < tol:
                v = v_new
                break
            v = v_new

        lambda_max = float(np.dot(v, self.D @ v))
        return lambda_max, v

    def inverse_iteration(
        self,
        mu: float = 0.0,
        n_iter: int = 100,
        tol: float = 1e-10,
    ) -> Tuple[float, np.ndarray]:
        """
        Inverse iteration for eigenvalue nearest to mu.

        Finds the eigenvalue lambda closest to mu by iterating:
            (D - mu*I) * w_{k+1} = v_k
            v_{k+1} = w_{k+1} / ||w_{k+1}||

        Parameters
        ----------
        mu : float
            Shift (target eigenvalue).
        n_iter : int
            Maximum iterations.
        tol : float
            Convergence tolerance.

        Returns
        -------
        lambda_near : float
            Eigenvalue nearest to mu.
        v_near : np.ndarray
            Corresponding eigenvector.
        """
        v = np.random.RandomState(42).random(self.N)
        v /= np.linalg.norm(v)

        # Factorize (D - mu*I)
        A = self.D - mu * np.eye(self.N)
        # Regularize if needed
        if abs(np.linalg.det(A)) < 1e-30:
            A += 1e-10 * np.eye(self.N)

        for _ in range(n_iter):
            try:
                w = np.linalg.solve(A, v)
            except np.linalg.LinAlgError:
                # Singular: use lstsq
                w, _, _, _ = np.linalg.lstsq(A, v, rcond=None)

            v_new = w / np.linalg.norm(w)
            if np.linalg.norm(v_new - v) < tol:
                v = v_new
                break
            v = v_new

        lambda_near = float(np.dot(v, self.D @ v))
        return lambda_near, v

    def pseudospectrum_bound(
        self,
        epsilon: float = 0.01,
        n_points: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate the epsilon-pseudospectrum.

        The epsilon-pseudospectrum is the set:
            Lambda_eps = {z in C : ||(D - z*I)^{-1}|| > 1/epsilon}

        A large pseudospectrum indicates high sensitivity to perturbations.

        Parameters
        ----------
        epsilon : float
            Pseudospectrum level.
        n_points : int
            Grid resolution.

        Returns
        -------
        z_grid : np.ndarray
            Complex grid points.
        resolvent_norm : np.ndarray
            ||(D - z*I)^{-1}|| at each point.
        """
        eigenvalues = self.compute_eigenvalues()
        eig_min = eigenvalues[0] - 0.5
        eig_max = eigenvalues[-1] + 0.5

        # Real grid (for symmetric D, pseudospectrum is along real axis)
        z_real = np.linspace(eig_min, eig_max, n_points)
        resolvent = np.zeros(n_points)

        for i, z in enumerate(z_real):
            try:
                R = np.linalg.inv(self.D - z * np.eye(self.N))
                resolvent[i] = np.linalg.norm(R, 2)
            except np.linalg.LinAlgError:
                resolvent[i] = 1e30

        return z_real, resolvent

    def mode_decomposition(
        self,
        displacement: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Decompose a displacement into normal modes.

        u = sum_k c_k * v_k
        where v_k are eigenvectors and c_k = v_k^T * u.

        Parameters
        ----------
        displacement : np.ndarray, shape (N,)
            Displacement vector to decompose.

        Returns
        -------
        decomposition : dict
            'coefficients': c_k values,
            'mode_energies': c_k^2 * omega_k^2,
            'dominant_mode': index of mode with largest energy.
        """
        if self._eigenvectors is None:
            self.compute_eigenvalues()

        if self._eigenvectors is None:
            # Fall back to direct computation
            _, V = np.linalg.eigh(self.D)
        else:
            V = self._eigenvectors

        # Projection coefficients
        c = V.T @ displacement
        eigenvalues = self._eigenvalues

        # Mode energies
        omega_sq = np.maximum(eigenvalues, 0.0)
        mode_energies = 0.5 * c**2 * omega_sq

        dominant = int(np.argmax(mode_energies))

        return {
            "coefficients": c,
            "mode_energies": mode_energies,
            "dominant_mode": dominant,
            "dominant_frequency": float(np.sqrt(max(0, eigenvalues[dominant]))),
        }
