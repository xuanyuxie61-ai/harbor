"""
phase_space_flow_coarse_grain.py
================================
Coarse-graining of the dark matter 6D phase-space distribution via
normalizing-flow bijections.

Physical context
----------------
The fine-grained DM phase-space density f(x, v, t) obeys the
collisionless Boltzmann (Vlasov) equation

    df/dt = df/dt + v . grad_x f - grad_x Phi . grad_v f = 0.

Direct simulation of f in 6D is impossible; one therefore coarse-
grains f onto a lower-dimensional manifold.  We parameterise the
coarse-graining map T : (x, v) -> (y, w) as a composition of
affine and stochastic-linear bijections, in the spirit of the
normalizing-flow approach of Koehler et al. (chemtrain / ECG).

The log-determinant of the Jacobian |det dT/d(x,v)| enters the
change-of-variables formula and gives the local compression of
phase-space volume, which by Liouville's theorem is unity for the
exact dynamics.  Deviations from unity quantify the information
lost by the coarse-graining.
"""

from __future__ import annotations
import math
from typing import Tuple

import numpy as np


# ---------- Softmax row-stochastic linear bijection ---------------------------

class StochasticLinearBijection:
    """Linear map A with softmax row normalisation so that every row
    of A sums to one.  This implements a Markov-like coarse-graining
    operator on the phase-space grid."""

    def __init__(self, params: np.ndarray):
        assert params.ndim == 2 and params.shape[0] == params.shape[1], \
            "params must be a square matrix"
        self.params = params

    @staticmethod
    def _softmax_rows(M: np.ndarray) -> np.ndarray:
        M = M - M.max(axis=1, keepdims=True)
        E = np.exp(M)
        return E / E.sum(axis=1, keepdims=True)

    @property
    def matrix(self) -> np.ndarray:
        return self._softmax_rows(self.params)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        A = self.matrix
        y = A @ x
        sign, logdet = np.linalg.slogdet(A)
        return y, float(logdet) if sign > 0 else float("-inf")

    def inverse(self, y: np.ndarray) -> Tuple[np.ndarray, float]:
        A = self.matrix
        try:
            x = np.linalg.solve(A, y)
        except np.linalg.LinAlgError:
            x = np.linalg.lstsq(A, y, rcond=None)[0]
        sign, logdet = np.linalg.slogdet(A)
        return x, -float(logdet) if sign > 0 else float("inf")


# ---------- Affine bijection with location/scale ------------------------------

class AffineBijection:
    """y = scale * x + loc, with learnable loc and log-scale."""

    def __init__(self, loc: np.ndarray, log_scale: np.ndarray):
        self.loc = np.asarray(loc, dtype=float)
        self.log_scale = np.asarray(log_scale, dtype=float)

    @property
    def scale(self) -> np.ndarray:
        return np.exp(self.log_scale)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        y = self.scale * x + self.loc
        logdet = float(np.sum(self.log_scale))
        return y, logdet

    def inverse(self, y: np.ndarray) -> Tuple[np.ndarray, float]:
        x = (y - self.loc) / self.scale
        return x, -float(np.sum(self.log_scale))


# ---------- Composition (coupled coarse-graining flow) ------------------------

class PhaseSpaceFlow:
    """Composition of an affine bijection with a stochastic-linear
    bijection, used to map the fine-grained halo distribution to a
    coarse-grained one."""

    def __init__(self, affine: AffineBijection,
                 stoch: StochasticLinearBijection):
        self.affine = affine
        self.stoch = stoch

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        y, ld1 = self.affine.forward(x)
        z, ld2 = self.stoch.forward(y)
        return z, ld1 + ld2

    def inverse(self, z: np.ndarray) -> Tuple[np.ndarray, float]:
        y, ld2 = self.stoch.inverse(z)
        x, ld1 = self.affine.inverse(y)
        return x, ld1 + ld2


# ---------- Construction for a halo patch ------------------------------------

def build_halo_coarse_graining(dim: int = 4,
                               seed: int = 0) -> PhaseSpaceFlow:
    """Build a small flow suitable for coarse-graining a 2N-D phase
    space patch (here dim = 4 stands for (x, v_x, y, v_y))."""
    rng = np.random.default_rng(seed)
    loc = rng.normal(0.0, 0.1, size=dim)
    log_scale = rng.normal(0.0, 0.05, size=dim)
    affine = AffineBijection(loc, log_scale)
    params = rng.normal(0.0, 1.0, size=(dim, dim))
    stoch = StochasticLinearBijection(params)
    return PhaseSpaceFlow(affine, stoch)


# ---------- Liouville-volume diagnostic --------------------------------------

def liouville_deviation(flow: PhaseSpaceFlow,
                        n_samples: int = 200,
                        seed: int = 1) -> dict:
    """Sample phase-space points, push them through the flow and
    compute how far |det J| deviates from 1 (Liouville theorem)."""
    rng = np.random.default_rng(seed)
    dim = flow.affine.loc.size
    X = rng.normal(0.0, 1.0, size=(dim, n_samples))
    devs = np.zeros(n_samples)
    for k in range(n_samples):
        _, logdet = flow.forward(X[:, k])
        devs[k] = abs(math.exp(min(logdet, 50.0)) - 1.0)
    return dict(mean_abs_deviation=float(devs.mean()),
                max_abs_deviation=float(devs.max()),
                n_samples=n_samples)


# ---------- Self-check --------------------------------------------------------

def self_check() -> dict:
    flow = build_halo_coarse_graining(dim=4, seed=2024)
    rng = np.random.default_rng(0)
    x = rng.normal(size=4)
    y, ld1 = flow.forward(x)
    xr, ld2 = flow.inverse(y)
    err = float(np.max(np.abs(x - xr)))
    return dict(roundtrip_err=err, logdet=ld1)
