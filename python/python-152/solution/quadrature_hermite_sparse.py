"""
Hermite polynomial quadrature and sparse grid integration for quantum noise.

Incorporates:
- 522_hermite_polynomial: physicist's and probabilist's Hermite polynomials
- 1362_truncated_normal_sparse_grid: Smolyak sparse grid, moment method
"""
import numpy as np
from scipy.special import eval_hermite, factorial, roots_hermite
from itertools import combinations_with_replacement


class HermiteQuadrature:
    """
    Gaussian quadrature using physicist's Hermite polynomials H_n(x).
    Weight function: w(x) = exp(-x²).

    ∫_{-∞}^{∞} f(x) exp(-x²) dx ≈ Σ_i w_i f(x_i)

    Nodes x_i are roots of H_n(x), weights:
        w_i = 2^{n-1} n! √π / (n² [H_{n-1}(x_i)]²)
    """

    def __init__(self, n_points: int):
        self.n_points = n_points
        self.nodes, self.weights = self._compute_nodes_weights()

    def _compute_nodes_weights(self):
        """Use scipy's roots_hermite for physicist's Hermite."""
        x, w = roots_hermite(self.n_points)
        return x, w

    def integrate(self, f, *args) -> float:
        """Integrate f(x) * exp(-x²) using Gaussian quadrature."""
        return np.sum(self.weights * f(self.nodes, *args))

    def physicist_polynomial(self, n: int, x: np.ndarray) -> np.ndarray:
        """
        H_n(x) via three-term recurrence:
            H_0(x) = 1
            H_1(x) = 2x
            H_n(x) = 2x H_{n-1}(x) - 2(n-1) H_{n-2}(x)
        """
        return eval_hermite(n, x)

    def probabilist_polynomial(self, n: int, x: np.ndarray) -> np.ndarray:
        """
        He_n(x) via recurrence:
            He_0(x) = 1
            He_1(x) = x
            He_n(x) = x He_{n-1}(x) - (n-1) He_{n-2}(x)
        """
        if n == 0:
            return np.ones_like(x)
        if n == 1:
            return x.copy()
        He_prev2 = np.ones_like(x)
        He_prev1 = x.copy()
        for k in range(2, n + 1):
            He_curr = x * He_prev1 - (k - 1) * He_prev2
            He_prev2, He_prev1 = He_prev1, He_curr
        return He_prev1

    def hermite_function(self, n: int, x: np.ndarray) -> np.ndarray:
        """
        Orthonormal Hermite function:
            ψ_n(x) = (2^n n! √π)^{-1/2} H_n(x) exp(-x²/2)
        """
        Hn = self.physicist_polynomial(n, x)
        norm = np.sqrt((2.0 ** n) * factorial(n) * np.sqrt(np.pi))
        return Hn * np.exp(-x ** 2 / 2.0) / norm

    def project_onto_basis(self, f, max_n: int = 20) -> np.ndarray:
        """
        Compute expansion coefficients c_n in Hermite basis:
            f(x) = Σ_n c_n ψ_n(x)
            c_n = ∫ f(x) ψ_n(x) dx
        """
        coeffs = np.zeros(max_n + 1)
        for n in range(max_n + 1):
            psi_n = self.hermite_function(n, self.nodes)
            coeffs[n] = np.sum(self.weights * f(self.nodes) * psi_n)
        return coeffs


class TruncatedNormalSparseGrid:
    """
    Sparse grid quadrature for multi-dimensional integrals with
    truncated normal weight function.

    Integral:
        I = ∫_{[a,b]^d} f(x) φ_{[a,b]}(x) dx

    where φ_{[a,b]} is the truncated normal density.
    Uses Smolyak sparse grid construction (from 1362_truncated_normal_sparse_grid).
    """

    def __init__(self, dim: int, level: int, bounds: tuple = (-3.0, 3.0)):
        self.dim = dim
        self.level = level
        self.bounds = bounds
        self.nodes, self.weights = self._build_sparse_grid()

    def _level_to_order(self, level: int) -> int:
        """Map sparse grid level to 1D quadrature order (odd)."""
        return 2 * level + 1

    def _truncated_normal_moments(self, order: int, a: float, b: float) -> tuple:
        """
        Compute moments of truncated normal and construct Gaussian quadrature
        via moment method (Hankel matrix + Cholesky / Jacobi matrix).
        """
        from scipy.stats import norm
        # Compute raw moments μ_k = E[X^k] for truncated normal
        moments = np.zeros(2 * order + 1)
        Za = norm.cdf(a)
        Zb = norm.cdf(b)
        Z = Zb - Za
        # Use recurrence for moments
        moments[0] = 1.0
        if order >= 1:
            moments[1] = (norm.pdf(a) - norm.pdf(b)) / Z
        for k in range(2, 2 * order + 1):
            moments[k] = (k - 1) * moments[k - 2] + (a ** (k - 1) * norm.pdf(a) - b ** (k - 1) * norm.pdf(b)) / Z
        # Build Hankel matrix H_{ij} = μ_{i+j}
        H = np.zeros((order, order))
        for i in range(order):
            for j in range(order):
                H[i, j] = moments[i + j]
        # Cholesky for orthonormal polynomials
        try:
            L = np.linalg.cholesky(H)
        except np.linalg.LinAlgError:
            # Shift to PD
            H += 1e-10 * np.eye(order)
            L = np.linalg.cholesky(H)
        # Jacobi matrix from Cholesky
        J = np.zeros((order, order))
        for i in range(order - 1):
            J[i, i + 1] = L[i + 1, i + 1] / L[i, i]
            J[i + 1, i] = J[i, i + 1]
        for i in range(order):
            J[i, i] = L[i + 1, i] / L[i, i] if i + 1 < order else 0.0
            if i > 0:
                J[i, i] -= L[i, i - 1] / L[i - 1, i - 1]
        # Eigenvalues = nodes, eigenvectors give weights
        w, v = np.linalg.eigh(J)
        nodes = w
        weights = (v[0, :] ** 2) * moments[0]
        # Map from standard normal to truncated domain
        nodes = np.clip(nodes, a, b)
        return nodes, weights

    def _tensor_product(self, nodes_1d: list, weights_1d: list) -> tuple:
        """Build tensor product grid from 1D nodes and weights."""
        grids = np.meshgrid(*nodes_1d, indexing='ij')
        nodes = np.stack([g.flatten() for g in grids], axis=1)
        w_grids = np.meshgrid(*weights_1d, indexing='ij')
        weights = np.prod(np.stack([g.flatten() for g in w_grids], axis=1), axis=1)
        return nodes, weights

    def _get_sequences(self, dim: int, level: int) -> list:
        """
        Generate multi-indices for Smolyak sparse grid:
            |ℓ|_1 ≤ level + dim - 1,  ℓ_i ≥ 1
        """
        sequences = []
        max_sum = level + dim - 1
        # Generate all compositions using recursion
        def recurse(current, remaining, idx):
            if idx == dim - 1:
                current.append(remaining)
                if sum(current) <= max_sum:
                    sequences.append(tuple(current))
                current.pop()
                return
            for v in range(1, remaining - (dim - idx - 1) + 1):
                current.append(v)
                recurse(current, remaining - v, idx + 1)
                current.pop()
        for s in range(dim, max_sum + 1):
            recurse([], s, 0)
        return sequences

    def _build_sparse_grid(self) -> tuple:
        """Construct Smolyak sparse grid."""
        a, b = self.bounds
        all_nodes = []
        all_weights = []
        sequences = self._get_sequences(self.dim, self.level)
        for seq in sequences:
            nodes_1d = []
            weights_1d = []
            for li in seq:
                order = self._level_to_order(li)
                x, w = self._truncated_normal_moments(order, a, b)
                nodes_1d.append(x)
                weights_1d.append(w)
            nodes_tp, weights_tp = self._tensor_product(nodes_1d, weights_1d)
            coeff = (-1) ** (self.level + self.dim - sum(seq))
            # Multinomial coefficient for combination
            from math import comb
            comb_val = comb(self.level + self.dim - sum(seq) - 1, self.dim - 1)
            if self.level + self.dim - sum(seq) - 1 < 0:
                comb_val = 1
            all_nodes.append(nodes_tp)
            all_weights.append(coeff * comb_val * weights_tp)
        # Merge duplicate nodes
        if not all_nodes:
            return np.zeros((0, self.dim)), np.zeros(0)
        nodes_cat = np.vstack(all_nodes)
        weights_cat = np.concatenate(all_weights)
        # Round and merge
        tol = 1e-10
        unique_map = {}
        for i in range(nodes_cat.shape[0]):
            key = tuple(np.round(nodes_cat[i] / tol).astype(int))
            if key in unique_map:
                unique_map[key] = (unique_map[key][0], unique_map[key][1] + weights_cat[i])
            else:
                unique_map[key] = (nodes_cat[i].copy(), weights_cat[i])
        nodes_final = np.array([v[0] for v in unique_map.values()])
        weights_final = np.array([v[1] for v in unique_map.values()])
        return nodes_final, weights_final

    def integrate(self, f) -> float:
        """Integrate f over sparse grid."""
        if self.nodes.shape[0] == 0:
            return 0.0
        vals = np.array([f(self.nodes[i, :]) for i in range(self.nodes.shape[0])])
        return np.sum(self.weights * vals)


class QuantumNoiseIntegral:
    """
    Compute integrals over Pauli error distributions using Hermite/sparse grid.
    """

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits

    def logical_error_probability_integral(self, p_func, threshold_fem, quadrature_order: int = 30) -> float:
        """
        Compute logical error probability by integrating over noise distribution:
            P_L = ∫ P_L(p(ξ)) φ(ξ) dξ
        where ξ is a standard normal random variable and φ is its density.
        """
        hq = HermiteQuadrature(quadrature_order)

        def integrand(xi):
            p = p_func(xi)
            return threshold_fem(p)

        # Use physicist's Hermite: weight is exp(-x²), so integrand needs no extra factor
        return hq.integrate(integrand)

    def multi_qubit_moment_integral(self, moments: list, dim: int = 2, level: int = 3) -> float:
        """
        Compute multi-qubit moment using sparse grid:
            M = E[ Π_i p_i^{m_i} ]
        """
        sg = TruncatedNormalSparseGrid(dim, level)

        def f(x):
            val = 1.0
            for i, m in enumerate(moments[:dim]):
                # Map truncated normal x_i to [0,1] error probability
                p_i = 0.5 * (1.0 + np.tanh(x[i]))
                val *= p_i ** m
            return val

        return sg.integrate(f)
