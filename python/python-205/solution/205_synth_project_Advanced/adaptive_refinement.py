"""
adaptive_refinement.py - Sequential Adaptive Surrogate Enrichment

This module implements adaptive enrichment strategies for polynomial
chaos surrogates, integrating:
  - Sequential retraining with pseudo-labeling (from SRPM-ST)
  - Error-driven sampling for targeted refinement
  - Cross-validation guided enrichment

Mathematical Framework
----------------------
### Adaptive Enrichment Criterion ###

The goal is to iteratively add training points where the surrogate
error is largest. The enrichment criterion is:

  x_new = argmax_{x in candidate_set} |sigma(x)|
where sigma(x) is the predicted standard deviation from the surrogate.

For PCE, the prediction variance is estimated via LOO residuals:
  sigma^2(x) ≈ (1/N) sum_i (h_i(x) * LOO_i)^2
where h_i(x) are leverage values.

### Sequential Retraining with Pseudo-Labeling ###

Following the SRPM-ST approach:
1. Start with initial training set D_0 = {(x_i, y_i)}_{i=1}^{n_0}
2. For iteration t = 1, 2, ...:
   a. Fit surrogate to D_{t-1}
   b. Identify enrichment candidates via error criterion
   c. Evaluate truth model at selected candidates
   d. Optionally: add pseudo-labeled points (cheap model predictions)
   e. Retrain surrogate on D_t = D_{t-1} ∪ new points
3. Stop when convergence criterion met

### Convergence Assessment ###

The surrogate is considered converged when:
  max_x |sigma(x) / Y_pred(x)| < tolerance
or equivalently when LOO Q2 > threshold (e.g., 0.95).
"""

import numpy as np
from numpy.typing import NDArray
from typing import Dict, List, Optional, Tuple, Callable, Any
from polynomial_chaos import PolynomialChaosSurrogate
from surrogate_fitting import compute_validation_metrics


# ---------------------------------------------------------------------------
# Adaptive Sampling Criteria
# ---------------------------------------------------------------------------

def ucb_sampling_criterion(X_train: NDArray, Y_train: NDArray,
                           surrogate: PolynomialChaosSurrogate,
                           X_candidates: NDArray,
                           exploration_weight: float = 2.0) -> NDArray:
    """
    Upper Confidence Bound (UCB) sampling criterion.

    Selects points that balance exploitation (high predicted value)
    and exploration (high prediction uncertainty):

      UCB(x) = |Y_pred(x)| + beta * sigma(x)

    where sigma(x) is estimated from LOO residuals:
      sigma^2(x) ≈ (1/N) sum_i phi_i(x)^2 * LOO_i^2 / (1 - h_ii)^2

    Parameters
    ----------
    X_train : ndarray(n, d)
        Current training inputs.
    Y_train : ndarray(n,)
        Current training outputs.
    surrogate : PolynomialChaosSurrogate
        Current fitted surrogate.
    X_candidates : ndarray(m, d)
        Candidate points for enrichment.
    exploration_weight : float
        Controls exploration vs exploitation trade-off.

    Returns
    -------
    scores : ndarray(m,)
        UCB scores for each candidate.
    """
    Y_pred = surrogate.predict(X_candidates)

    # Estimate prediction variance from LOO
    try:
        loo = surrogate.leave_one_out_error(X_train, Y_train)
        loo_rmse = loo['loo_rmse']
    except Exception:
        loo_rmse = float(np.std(Y_train)) * 0.1

    # Simple variance estimate: distance-based
    # Points far from training data have higher uncertainty
    min_dists = np.zeros(X_candidates.shape[0])
    for i in range(X_candidates.shape[0]):
        dists = np.linalg.norm(X_train - X_candidates[i], axis=1)
        min_dists[i] = np.min(dists)

    sigma = loo_rmse * min_dists / (np.mean(min_dists) + 1e-30)

    scores = np.abs(Y_pred) + exploration_weight * sigma
    return scores


def leave_one_out_enrichment(X_train: NDArray, Y_train: NDArray,
                             surrogate: PolynomialChaosSurrogate,
                             X_candidates: NDArray) -> NDArray:
    """
    LOO-error-based enrichment criterion.

    Estimates the error at candidate points by extrapolating
    LOO residuals weighted by basis function similarity:

      error(x) ≈ sum_i w_i(x) * |LOO_i|
    where w_i(x) = |Psi(x) - Psi(x_i)|^{-1} / normalization

    Points near high-residual training points get high scores.
    """
    Psi_train = surrogate.basis.evaluate_all(X_train, surrogate.multi_index)
    n_train = X_train.shape[0]

    # Compute LOO residuals
    Y_pred = surrogate.predict(X_train)
    residuals = np.abs(Y_train - Y_pred)

    # Hat matrix diagonal for LOO scaling
    try:
        PtP_inv = np.linalg.inv(Psi_train.T @ Psi_train + 1e-12 * np.eye(surrogate.n_terms))
        H = Psi_train @ PtP_inv @ Psi_train.T
        h_diag = np.diag(H)
    except np.linalg.LinAlgError:
        h_diag = np.ones(n_train) * surrogate.n_terms / n_train

    denom = np.maximum(1.0 - h_diag, 1e-14)
    loo_errors = residuals / denom

    # Score candidates by weighted LOO error
    scores = np.zeros(X_candidates.shape[0])
    for i in range(X_candidates.shape[0]):
        dists = np.linalg.norm(X_train - X_candidates[i], axis=1)
        weights = 1.0 / (dists + 1e-10)
        weights /= np.sum(weights)
        scores[i] = np.sum(weights * loo_errors)

    return scores


# ---------------------------------------------------------------------------
# Sequential Refinement Loop
# ---------------------------------------------------------------------------

class AdaptiveRefinement:
    """
    Adaptive sequential refinement of polynomial chaos surrogates.

    Implements the SRPM-ST-style sequential retraining:
    1. Build initial surrogate from DOE
    2. Evaluate convergence via LOO cross-validation
    3. If not converged:
       a. Select enrichment points via error criterion
       b. Evaluate truth model at new points
       c. Optionally add pseudo-labeled points
       d. Retrain surrogate
    4. Repeat until convergence or max iterations

    Attributes
    ----------
    d : int
        Input dimension.
    max_degree : int
        Polynomial degree.
    truth_model : callable
        Function that evaluates the expensive model.
    X_train : ndarray(n, d)
        Current training inputs.
    Y_train : ndarray(n,)
        Current training outputs.
    surrogate : PolynomialChaosSurrogate
        Current surrogate model.
    history : list of dict
        Refinement history with metrics at each iteration.
    """

    def __init__(self, d: int, max_degree: int = 3,
                 index_set_type: str = 'total_order',
                 truth_model: Optional[Callable] = None):
        self.d = d
        self.max_degree = max_degree
        self.index_set_type = index_set_type
        self.truth_model = truth_model
        self.X_train: Optional[NDArray] = None
        self.Y_train: Optional[NDArray] = None
        self.surrogate: Optional[PolynomialChaosSurrogate] = None
        self.history: List[Dict[str, Any]] = []
        self._iteration = 0

    def initialize(self, X_init: NDArray, Y_init: NDArray):
        """
        Initialize with initial training data.

        Parameters
        ----------
        X_init : ndarray(n0, d)
            Initial design points.
        Y_init : ndarray(n0,)
            Model outputs at initial points.
        """
        self.X_train = X_init.copy()
        self.Y_train = Y_init.copy()
        self._build_surrogate()
        self._record_metrics('initialization')

    def _build_surrogate(self):
        """Build/rebuild the surrogate from current training data."""
        self.surrogate = PolynomialChaosSurrogate(
            self.d, self.max_degree, self.index_set_type)
        if self.X_train.shape[0] > self.surrogate.n_terms:
            self.surrogate.fit_regression(self.X_train, self.Y_train,
                                          'ridge', 1e-6)
        else:
            # Underdetermined: use projection
            self.surrogate.fit_projection(self.X_train, self.Y_train)

    def _record_metrics(self, phase: str):
        """Record current surrogate metrics."""
        metrics = {'phase': phase, 'iteration': self._iteration,
                   'n_train': self.X_train.shape[0]}

        if self.surrogate.is_fitted:
            try:
                loo = self.surrogate.leave_one_out_error(
                    self.X_train, self.Y_train)
                metrics.update({f'loo_{k}': v for k, v in loo.items()})
            except Exception:
                pass

            try:
                moments = self.surrogate.statistical_moments()
                metrics.update({f'moment_{k}': v for k, v in moments.items()})
            except Exception:
                pass

            try:
                sobol = self.surrogate.sobol_indices()
                metrics['sobol_first_sum'] = float(np.sum(sobol['first_order']))
                metrics['sobol_total_sum'] = float(np.sum(sobol['total_order']))
            except Exception:
                pass

        self.history.append(metrics)

    def refine_step(self, X_candidates: NDArray,
                    n_new: int = 5,
                    criterion: str = 'ucb',
                    pseudo_label_ratio: float = 0.0) -> Dict[str, Any]:
        """
        Perform one step of adaptive refinement.

        Parameters
        ----------
        X_candidates : ndarray(m, d)
            Candidate points for enrichment.
        n_new : int
            Number of new points to add.
        criterion : str
            Enrichment criterion: 'ucb', 'loo_error', 'random'.
        pseudo_label_ratio : float
            Fraction of new points to be pseudo-labeled (0 to 1).

        Returns
        -------
        dict with enrichment information.
        """
        self._iteration += 1

        if self.surrogate is None or not self.surrogate.is_fitted:
            return {'error': 'Surrogate not initialized'}

        # Score candidates
        if criterion == 'ucb':
            scores = ucb_sampling_criterion(
                self.X_train, self.Y_train, self.surrogate,
                X_candidates)
        elif criterion == 'loo_error':
            scores = leave_one_out_enrichment(
                self.X_train, self.Y_train, self.surrogate,
                X_candidates)
        elif criterion == 'random':
            scores = np.random.random(X_candidates.shape[0])
        else:
            scores = np.ones(X_candidates.shape[0])

        # Select top candidates
        n_new = min(n_new, len(scores))
        top_indices = np.argsort(scores)[-n_new:]
        X_new = X_candidates[top_indices]

        # Evaluate truth model or use pseudo-labels
        n_truth = max(1, int(n_new * (1.0 - pseudo_label_ratio)))
        n_pseudo = n_new - n_truth

        Y_new = np.zeros(n_new)

        # Truth model evaluations
        if self.truth_model is not None:
            for i in range(n_truth):
                params_dict = {f'x{j}': float(X_new[i, j]) for j in range(self.d)}
                result = self.truth_model(params_dict)
                if isinstance(result, dict):
                    Y_new[i] = list(result.values())[0]
                else:
                    Y_new[i] = float(result)
        else:
            # No truth model: use surrogate prediction as pseudo-label
            Y_new[:n_truth] = self.surrogate.predict(X_new[:n_truth])

        # Pseudo-labeled points
        if n_pseudo > 0:
            Y_new[n_truth:] = self.surrogate.predict(X_new[n_truth:])

        # Augment training set
        self.X_train = np.vstack([self.X_train, X_new])
        self.Y_train = np.concatenate([self.Y_train, Y_new])

        # Remove used candidates
        remaining_mask = np.ones(X_candidates.shape[0], dtype=bool)
        remaining_mask[top_indices] = False
        X_remaining = X_candidates[remaining_mask]

        # Rebuild surrogate
        self._build_surrogate()
        self._record_metrics(f'refinement_{self._iteration}')

        return {
            'n_new': n_new,
            'n_truth': n_truth,
            'n_pseudo': n_pseudo,
            'max_score': float(np.max(scores)),
            'n_remaining': X_remaining.shape[0]
        }

    def run_adaptive_loop(self, X_candidates: NDArray,
                          max_iterations: int = 10,
                          n_per_iter: int = 5,
                          q2_threshold: float = 0.95,
                          criterion: str = 'ucb',
                          pseudo_label_ratio: float = 0.3) -> Dict[str, Any]:
        """
        Run the full adaptive refinement loop.

        Parameters
        ----------
        X_candidates : ndarray(m, d)
            Pool of candidate enrichment points.
        max_iterations : int
            Maximum number of refinement iterations.
        n_per_iter : int
            Number of new points per iteration.
        q2_threshold : float
            LOO Q2 threshold for convergence.
        criterion : str
            Enrichment criterion.
        pseudo_label_ratio : float
            Fraction of pseudo-labeled points.

        Returns
        -------
        dict with final results and convergence info.
        """
        converged = False
        X_pool = X_candidates.copy()

        for iteration in range(max_iterations):
            # Check convergence
            if self.surrogate.is_fitted:
                try:
                    loo = self.surrogate.leave_one_out_error(
                        self.X_train, self.Y_train)
                    if loo['loo_q2'] >= q2_threshold:
                        converged = True
                        break
                except Exception:
                    pass

            if X_pool.shape[0] < n_per_iter:
                break

            # Refine
            result = self.refine_step(
                X_pool, n_per_iter, criterion, pseudo_label_ratio)

            # Update pool
            if result.get('n_remaining', 0) < n_per_iter:
                break

            # Recompute candidates (simplified: just keep remaining)
            top_indices = np.argsort(
                np.random.random(X_pool.shape[0]))[:result['n_new']]
            mask = np.ones(X_pool.shape[0], dtype=bool)
            # Keep remaining pool
            X_pool = X_pool[np.random.permutation(X_pool.shape[0])]
            X_pool = X_pool[:max(result.get('n_remaining', 0), n_per_iter)]

        return {
            'converged': converged,
            'n_iterations': self._iteration,
            'n_final_train': self.X_train.shape[0],
            'history': self.history,
            'final_loo_q2': self.history[-1].get('loo_loo_q2', 0.0) if self.history else 0.0
        }

    def get_final_surrogate(self) -> Optional[PolynomialChaosSurrogate]:
        """Return the final refined surrogate."""
        return self.surrogate
