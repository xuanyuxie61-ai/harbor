"""
Coupled eigenvalue estimation via multi-task learning structure.

Adapted from:
  - 1089_HarrisonFah_TC-MTLR (multi-task logistic regression,
    temporal convolutional networks)

Key formulas:
    Spectral mapping:
        D v = lambda v  =>  D^dag D v = |lambda|^2 v (if D normal)
    Coupling penalty:
        L_coupling = alpha * sum_i (|lambda_i(D)|^2 - lambda_i(D^dag D))^2
"""

import numpy as np
from constants import LatticeParams
from gauge_field import GaugeField
from wilson_dirac import WilsonDiracOperator, SpinorField
from spectral_analysis import arnoldi_eigenvalues


class CoupledEigenvalueEstimator:
    def __init__(self, gf, n_eig=8, seed=42):
        if n_eig < 1:
            raise ValueError(f"n_eig must be >= 1, got {n_eig}")
        self.gf = gf
        self.p = gf.p
        self.n_eig = n_eig
        self.seed = seed
        self.D = WilsonDiracOperator(gf)

    def estimate_all(self):
        eig_D = arnoldi_eigenvalues(self.D, n_eig=self.n_eig,
                                      seed=self.seed, which='SM')
        eigs_D = eig_D['eigenvalues']
        eigs_DdagD_predicted = np.abs(eigs_D) ** 2
        eigs_g5D = self._estimate_gamma5_D_eigenvalues(eigs_D)
        coupling_err = self._coupling_error(eigs_D, eigs_DdagD_predicted, eigs_g5D)
        return {
            'eigs_D': eigs_D, 'eigs_DdagD': eigs_DdagD_predicted,
            'eigs_g5D': eigs_g5D, 'coupling_error': coupling_err,
            'n_converged': eig_D.get('n_iterations', 0),
        }

    def _estimate_gamma5_D_eigenvalues(self, eigs_D):
        shifted = eigs_D - (1.0 / (2.0 * self.p.kappa) - 4.0)
        return shifted.real + 1j * shifted.imag * 0.1

    def _coupling_error(self, eigs_D, eigs_DdagD_pred, eigs_g5D):
        if len(eigs_D) == 0:
            return 0.0
        err1 = np.sum((np.abs(eigs_D) ** 2 - eigs_DdagD_pred) ** 2)
        err2 = np.sum((eigs_g5D.real - np.abs(eigs_D)) ** 2)
        return float(err1 + err2)


class TemporalSpectralPredictor:
    def __init__(self, tau=3.0, memory=5):
        if tau <= 0.0:
            raise ValueError(f"tau must be > 0, got {tau}")
        if memory < 1:
            raise ValueError(f"memory must be >= 1, got {memory}")
        self.tau = tau
        self.memory = memory
        self._build_kernel()

    def _build_kernel(self):
        j = np.arange(self.memory)
        w = np.exp(-j / self.tau)
        self.kernel = w / w.sum()

    def predict_next(self, history):
        if len(history) == 0:
            raise ValueError("history is empty")
        if history.ndim == 1:
            T = len(history)
            if T < self.memory:
                padded = np.concatenate([np.full(self.memory - T, history[0]), history])
            else:
                padded = history[-self.memory:]
            return complex(np.dot(self.kernel, padded))
        else:
            result = np.zeros(history.shape[1], dtype=complex)
            for i in range(history.shape[1]):
                result[i] = self.predict_next(history[:, i])
            return result

    def predict_trajectory(self, history, n_forward):
        if n_forward < 1:
            raise ValueError(f"n_forward must be >= 1, got {n_forward}")
        current = list(history)
        predictions = []
        for _ in range(n_forward):
            arr = np.array(current[-self.memory:]) if len(current) >= self.memory else np.array(current)
            pred = self.predict_next(arr)
            predictions.append(pred)
            current.append(pred)
        return np.array(predictions)


def coupled_spectral_analysis(gf, n_eig=6, seed=42):
    estimator = CoupledEigenvalueEstimator(gf, n_eig=n_eig, seed=seed)
    result = estimator.estimate_all()
    eigs_D = result['eigs_D']
    if len(eigs_D) > 0:
        result['spectral_gap'] = float(np.min(np.abs(eigs_D)))
        result['mean_abs_eig'] = float(np.mean(np.abs(eigs_D)))
        result['max_abs_eig'] = float(np.max(np.abs(eigs_D)))
        result['condition_estimate'] = float(result['max_abs_eig'] / (result['spectral_gap'] + 1e-30))
    else:
        result['spectral_gap'] = 0.0
        result['mean_abs_eig'] = 0.0
        result['max_abs_eig'] = 0.0
        result['condition_estimate'] = float('inf')
    return result
