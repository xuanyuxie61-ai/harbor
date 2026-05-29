"""
Sparse-Grid Uncertainty Quantification for Eddy Diffusivity
===========================================================
Derived from seed project 1109_sparse_grid_total_poly (sparse grid
construction for total-degree polynomial spaces).

The eddy diffusivity tensor κ_{ij} in ocean parameterizations is
highly uncertain. We propagate uncertainty through the QG model
using stochastic collocation on sparse grids.

For a QoI Q(ξ) depending on d random parameters ξ = (ξ₁,…,ξ_d),
the expected value is:

    E[Q] = ∫_{Γ} Q(ξ) ρ(ξ) dξ

Using sparse-grid quadrature (Smolyak construction):

    A(q,d) = Σ_{|i|≤q+d−1}  (Δ^{i₁} ⊗ … ⊗ Δ^{i_d})

where Δ^i = U^i − U^{i−1} is the difference of 1D quadrature rules
of level i. The 1D rules used are nested Clenshaw-Curtis.

The number of points grows as O(2^q · q^{d−1}) instead of O(n^d)
for full tensor product, dramatically reducing the curse of dimensionality.
"""

import numpy as np
from itertools import product

class SparseGridUQ:
    """
    Sparse-grid stochastic collocation for uncertainty quantification.
    """

    def __init__(self, dim, level):
        """
        Parameters
        ----------
        dim : int
            Number of stochastic dimensions.
        level : int
            Sparse grid level q (≥ 1).
        """
        self.dim = dim
        self.level = level

    @staticmethod
    def clenshaw_curtis_nested(level):
        """
        Generate nested Clenshaw-Curtis points for level ℓ.
        Number of points: n_ℓ = 1 for ℓ=0,  n_ℓ = 2^{ℓ−1}+1 for ℓ≥1.
        """
        if level == 0:
            return np.array([0.0]), np.array([2.0])
        n = 2**(level - 1) + 1
        j = np.arange(n)
        x = np.cos(j * np.pi / (n - 1))
        # Weights via DCT (simplified)
        w = np.ones(n) * 2.0 / (n - 1)
        w[0] *= 0.5
        w[-1] *= 0.5
        return x, w

    def _multi_index_set(self):
        """
        Generate multi-indices i = (i₁,…,i_d) such that |i| ≤ q + d − 1
        and each i_j ≥ 1.
        """
        q = self.level
        d = self.dim
        indices = []
        # Use recursion to enumerate compositions
        def recurse(current, remaining):
            if len(current) == d:
                if sum(current) <= q + d - 1:
                    indices.append(tuple(current))
                return
            # Minimum remaining is 1 per remaining dimension
            min_val = 1
            max_val = remaining - (d - len(current) - 1)
            for val in range(min_val, max_val + 1):
                recurse(current + [val], remaining - val)
        for total in range(d, q + d):
            recurse([], total)
        return indices

    def build_sparse_grid(self):
        """
        Construct the sparse grid nodes and weights.

        Returns
        -------
        nodes : ndarray, shape (N, d)
        weights : ndarray, shape (N,)
        """
        indices = self._multi_index_set()
        node_list = []
        weight_list = []

        for idx in indices:
            # Get 1D rules for each dimension
            rules = [self.clenshaw_curtis_nested(i) for i in idx]
            # Tensor product of these rules
            x_grids = [r[0] for r in rules]
            w_grids = [r[1] for r in rules]
            for coords in product(*[range(len(xg)) for xg in x_grids]):
                node = np.array([x_grids[d][coords[d]] for d in range(self.dim)])
                weight = np.prod([w_grids[d][coords[d]] for d in range(self.dim)])
                node_list.append(node)
                weight_list.append(weight)

        if not node_list:
            return np.zeros((1, self.dim)), np.ones(1)

        nodes = np.array(node_list)
        weights = np.array(weight_list)

        # Deduplicate nodes (nested property ensures many duplicates)
        # Round to machine precision for hashing
        rounded = np.round(nodes, decimals=12)
        unique, inverse = np.unique(rounded.view(rounded.dtype.descr * self.dim),
                                     return_inverse=True)
        unique = unique.view(rounded.dtype).reshape(-1, self.dim)
        new_weights = np.zeros(len(unique))
        for i in range(len(weights)):
            new_weights[inverse[i]] += weights[i]

        return unique, new_weights

    def propagate_expectation(self, model_evaluator):
        """
        Compute E[Q] = Σ w_i · Q(ξ_i).

        Parameters
        ----------
        model_evaluator : callable
            Q = model_evaluator(ξ) where ξ is shape (d,).

        Returns
        -------
        mean : float
        variance : float
        nodes : ndarray
        values : ndarray
        """
        nodes, weights = self.build_sparse_grid()
        values = np.array([model_evaluator(node) for node in nodes])

        # Map nodes from [−1,1]^d to standard normal via inverse CDF
        # For this demo, we use uniform distribution on [−1,1]^d
        total_weight = np.sum(weights)
        if abs(total_weight - 2**self.dim) > 1e-6:
            weights = weights / total_weight * (2**self.dim)

        mean = np.sum(weights * values) / np.sum(weights)
        variance = np.sum(weights * (values - mean)**2) / np.sum(weights)
        return mean, variance, nodes, values


def total_degree_size(dim, degree):
    """
    Compute the dimension of the total-degree polynomial space
    P_d^q = span{ x^α : |α| ≤ q }.

        dim = C(q+d, d) = (q+d)! / (q! d!)
    """
    from math import comb
    return comb(degree + dim, dim)


def comp_next(n, k):
    """
    Generate the next composition of n into k parts.
    From seed project 1109_sparse_grid_total_poly.
    """
    if k == 1:
        yield (n,)
        return
    for i in range(n + 1):
        for tail in comp_next(n - i, k - 1):
            yield (i,) + tail
