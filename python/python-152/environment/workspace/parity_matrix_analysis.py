"""
Condition number estimation and matrix analysis for parity check matrices.

Incorporates:
- 207_condition: Hager's condition estimator, LINPACK-style estimator,
  test matrices (Kahan, CONEX, COMBIN)
"""
import numpy as np
from scipy.linalg import lu_factor, lu_solve


class ConditionEstimator:
    """
    Estimate the L1 condition number κ_1(A) = ||A||_1 · ||A^{-1}||_1
    for parity check matrices over the reals.

    Uses Hager's algorithm (1984) and LINPACK-style estimators.
    """

    def __init__(self, A: np.ndarray):
        self.A = A.copy().astype(float)
        self.m, self.n = A.shape
        self.is_square = (self.m == self.n)
        self.lu = None
        self.piv = None
        if self.is_square:
            try:
                self.lu, self.piv = lu_factor(self.A)
                # Check for singularity
                if np.any(np.abs(np.diag(self.lu)) < 1e-14):
                    self.is_square = False
                    self.lu = None
                    self.piv = None
            except Exception:
                self.is_square = False
                self.lu = None
                self.piv = None

    def _solve(self, b: np.ndarray) -> np.ndarray:
        """Solve Ax = b."""
        if self.is_square and self.lu is not None:
            return lu_solve((self.lu, self.piv), b)
        return np.linalg.lstsq(self.A, b, rcond=None)[0]

    def _solve_transpose(self, b: np.ndarray) -> np.ndarray:
        """Solve A^T x = b."""
        if self.is_square and self.lu is not None:
            return lu_solve((self.lu, self.piv), b, trans=1)
        return np.linalg.lstsq(self.A.T, b, rcond=None)[0]

    def hager_estimator(self, max_iter: int = 5) -> float:
        """
        Hager's L1 norm estimator for ||A^{-1}||_1.
        For non-square matrices, uses the square normal matrix (A^T A)^{-1} A^T
        or A (A A^T)^{-1} and applies Hager to the smaller square system.
        """
        if not self.is_square:
            # Use SVD-based L2 condition as proxy for L1
            return self.exact_condition_number()
        n = self.n
        x = np.ones(n) / n
        est = 0.0
        for _ in range(max_iter):
            y = self._solve_transpose(np.sign(x))
            xi = np.max(np.abs(y))
            if xi <= np.dot(y, x) + 1e-12:
                break
            j = np.argmax(np.abs(y))
            x_new = np.zeros(n)
            x_new[j] = 1.0
            z = self._solve(x_new)
            est_new = np.sum(np.abs(z))
            if est_new <= est + 1e-12:
                break
            est = est_new
            x = np.sign(z)
            for i in range(len(z)):
                if abs(z[i]) < 1e-12 and x[i] > 0:
                    x[i] = 0.0
            norm_x = np.sum(np.abs(x))
            if norm_x > 0:
                x = x / norm_x
            else:
                x = np.ones(len(z)) / len(z)
        norm_A = np.max(np.sum(np.abs(self.A), axis=0))  # L1 norm
        norm_Ainv = est
        return norm_A * norm_Ainv

    def linpack_estimator(self) -> float:
        """
        LINPACK-style L1 condition estimator for square matrices.
        For rectangular matrices, falls back to SVD-based estimate.
        """
        if not self.is_square or self.lu is None:
            s = np.linalg.svd(self.A, compute_uv=False)
            if s[-1] > 0:
                return s[0] / s[-1]
            return np.inf
        n = self.n
        b = np.ones(n) / n
        y = np.zeros(n)
        for i in range(n):
            y[i] = b[i]
            for j in range(i):
                y[i] -= self.lu[i, j] * y[j]
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            x[i] = y[i]
            for j in range(i + 1, n):
                x[i] -= self.lu[i, j] * x[j]
            if abs(self.lu[i, i]) > 1e-15:
                x[i] /= self.lu[i, i]
        est = np.sum(np.abs(x))
        norm_A = np.max(np.sum(np.abs(self.A), axis=0))
        return norm_A * est

    def exact_condition_number(self) -> float:
        """Exact L2 condition number via SVD (for rectangular, uses largest/smallest nonzero singular value)."""
        try:
            s = np.linalg.svd(self.A, compute_uv=False)
            s_nonzero = s[s > 1e-14]
            if len(s_nonzero) == 0:
                return np.inf
            return s[0] / s_nonzero[-1]
        except np.linalg.LinAlgError:
            return np.inf


class TestMatrixSuite:
    """
    Test matrices for validating condition estimators.
    From 207_condition (Kahan, CONEX, COMBIN matrices).
    """

    @staticmethod
    def kahan_matrix(n: int, theta: float = None) -> np.ndarray:
        """
        Kahan matrix: highly ill-conditioned upper triangular matrix.
            A_{ij} = 0           for i > j
            A_{ii} = s^{i-1}
            A_{ij} = -c * s^{i-1} for j > i
        where s = sin(θ), c = cos(θ).
        """
        if theta is None:
            theta = 1.2
        s = np.sin(theta)
        c = np.cos(theta)
        A = np.zeros((n, n))
        for i in range(n):
            A[i, i] = s ** i
            for j in range(i + 1, n):
                A[i, j] = -c * (s ** i)
        return A

    @staticmethod
    def conex1_matrix(n: int) -> np.ndarray:
        """CONEX1 matrix: ill-conditioned test matrix."""
        A = np.eye(n)
        A[0, :] = 1.0
        A[:, 0] = 1.0
        A[0, 0] = n
        return A

    @staticmethod
    def combin_matrix(n: int) -> np.ndarray:
        """
        COMBIN matrix: combination matrix with known inverse.
        A_{ij} = C(min(i,j), 2) for i,j >= 2, with modifications.
        """
        A = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                m = min(i, j)
                if m >= 2:
                    A[i, j] = m * (m - 1) / 2.0
                else:
                    A[i, j] = float(m)
        return A

    @staticmethod
    def ill_conditioned_parity_check(n: int, m: int) -> np.ndarray:
        """Generate an ill-conditioned parity check matrix for testing."""
        # Random sparse matrix with near-linear dependencies
        A = np.random.randint(0, 2, size=(m, n)).astype(float)
        # Add near-duplicate rows
        for i in range(1, m):
            if i % 2 == 1:
                A[i, :] = A[i - 1, :] + 1e-10 * np.random.randn(n)
        return A


class ParityCheckConditionAnalyzer:
    """
    Analyze the conditioning of parity check matrices for quantum codes.
    """

    def __init__(self, H: np.ndarray):
        self.H = H.copy().astype(float)
        self.m, self.n = H.shape

    def analyze(self) -> dict:
        """Comprehensive conditioning analysis."""
        est = ConditionEstimator(self.H)
        results = {
            "hager_kappa": est.hager_estimator(),
            "linpack_kappa": est.linpack_estimator(),
            "exact_kappa": est.exact_condition_number() if max(self.m, self.n) <= 50 else None,
            "rank": np.linalg.matrix_rank(self.H),
            "nullity": self.n - np.linalg.matrix_rank(self.H),
            "smallest_singular_value": np.min(np.linalg.svd(self.H, compute_uv=False)) if max(self.m, self.n) <= 200 else None,
            "largest_singular_value": np.max(np.linalg.svd(self.H, compute_uv=False)) if max(self.m, self.n) <= 200 else None,
        }
        if results["smallest_singular_value"] is not None and results["smallest_singular_value"] > 0:
            results["cond_from_svd"] = results["largest_singular_value"] / results["smallest_singular_value"]
        return results

    def check_stability(self) -> dict:
        """
        Check numerical stability of syndrome computation under perturbations.
        Perturb error vector by ε and observe syndrome change.
        """
        eps = 1e-8
        e = np.random.randint(0, 2, self.n).astype(float)
        s = (self.H @ e) % 2
        e_pert = e + eps * np.random.randn(self.n)
        s_pert = (self.H @ e_pert) % 2
        return {
            "syndrome_change": np.linalg.norm(s_pert - s),
            "relative_error": np.linalg.norm(s_pert - s) / max(np.linalg.norm(s), 1e-12)
        }
