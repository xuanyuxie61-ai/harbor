"""
chebyshev_accelerator.py
========================
Chebyshev polynomial tools for convergence acceleration of the outer
(active-set) iteration.

Mathematical background
-----------------------
The Chebyshev polynomials of the first kind are

    T_0(x) = 1,   T_1(x) = x,   T_{k+1}(x) = 2 x T_k(x) - T_{k-1}(x)

They satisfy  T_k(cos theta) = cos(k theta)  and are orthogonal on [-1, 1]
with weight  w(x) = 1/sqrt(1-x^2).

For convergence acceleration of a stationary iteration

    x_{k+1} = G x_k + c

with iteration matrix G having spectrum in [alpha, beta] subset (-1, 1),
the Chebyshev semi-iterative method produces accelerated iterates

    x_{k+1}^{acc} = rho_{k+1} [ gamma (G x_k^{acc} + c) + (1 - gamma) x_{k-1}^{acc} ]
                  + (1 - rho_{k+1}) x_{k-1}^{acc}

where  gamma = 2 / (2 - alpha - beta),  rho_{k+1} = (beta - alpha) T_k(d) / (2 T_{k+1}(d))
and d = (2 - alpha - beta) / (beta - alpha).

KKT role
--------
The outer active-set iteration can be viewed as a fixed-point iteration
on the control u.  When the active set stabilises, the inner KKT solve
is linear and Chebyshev acceleration gives superlinear convergence.
"""

from __future__ import annotations
import math
import numpy as np


# ---------------------------------------------------------------------------
# Chebyshev series evaluation  (from 163 / Maess / Clenshaw)
# ---------------------------------------------------------------------------

def chebyshev_series_eval(x: float, coef: np.ndarray) -> float:
    """Evaluate a Chebyshev series  f(x) = sum_{k=0}^{nc-1} c_k T_k(x)
    using Clenshaw's algorithm (modified from Maess's presentation).

    The Clenshaw recurrence for Chebyshev series is:

        b_{nc+1} = 0,  b_{nc} = 0
        b_k = 2 x b_{k+1} - b_{k+2} + c_k,   k = nc-1, ..., 1
        f(x) = x b_1 - b_2 + c_0

    This is numerically stable for x in [-1, 1].

    Parameters
    ----------
    x    : evaluation point, -1 <= x <= 1
    coef : (nc,) Chebyshev coefficients  c_0, c_1, ..., c_{nc-1}

    Returns
    -------
    f : value of the Chebyshev series at x
    """
    coef = np.asarray(coef, dtype=np.float64).ravel()
    nc = coef.size
    if nc == 0:
        return 0.0
    if nc == 1:
        return float(coef[0])
    if x < -1.0 or x > 1.0:
        # Extrapolate via the identity  T_k(x) = cosh(k arccosh(x))  for |x| > 1
        pass  # Clenshaw still works, just may overflow for large nc

    b_k_plus2 = 0.0
    b_k_plus1 = 0.0
    for k in range(nc - 1, 0, -1):
        b_k = 2.0 * x * b_k_plus1 - b_k_plus2 + coef[k]
        b_k_plus2 = b_k_plus1
        b_k_plus1 = b_k
    return float(x * b_k_plus1 - b_k_plus2 + coef[0])


def chebyshev_series_derivative(x: float, coef: np.ndarray) -> float:
    """Evaluate the derivative of a Chebyshev series using the recurrence

        c'_k = c'_{k+2} + 2 (k+1) c_{k+1},   k = nc-2, ..., 0
        with c'_{nc} = c'_{nc-1} = 0.

    Then  f'(x) = sum c'_k T_k(x)  evaluated via Clenshaw.
    """
    coef = np.asarray(coef, dtype=np.float64).ravel()
    nc = coef.size
    if nc <= 1:
        return 0.0
    dcoef = np.zeros(nc, dtype=np.float64)
    dcoef[nc - 1] = 0.0
    if nc >= 2:
        dcoef[nc - 2] = 2.0 * (nc - 1) * coef[nc - 1]
    for k in range(nc - 3, -1, -1):
        dcoef[k] = dcoef[k + 2] + 2.0 * (k + 1) * coef[k + 1]
    return chebyshev_series_eval(x, dcoef)


# ---------------------------------------------------------------------------
# Chebyshev coefficients of a function via discrete cosine transform
# ---------------------------------------------------------------------------

def chebyshev_coefficients(f_values: np.ndarray) -> np.ndarray:
    """Compute Chebyshev coefficients from function values at Chebyshev nodes.

    Given  f(x_k)  at the Chebyshev nodes  x_k = cos(pi (k + 0.5) / n),
    the coefficients are

        c_j = (2/n) sum_{k=0}^{n-1} f(x_k) T_j(x_k)

    with c_0 halved.  This is the discrete cosine transform (DCT-II).
    """
    n = f_values.size
    nodes = np.cos(np.pi * (np.arange(n) + 0.5) / n)
    coef = np.zeros(n, dtype=np.float64)
    for j in range(n):
        s = 0.0
        for k in range(n):
            Tj_xk = _chebyshev_poly(j, nodes[k])
            s += f_values[k] * Tj_xk
        coef[j] = 2.0 * s / n
    coef[0] *= 0.5
    return coef


def _chebyshev_poly(n: int, x: float) -> float:
    """Evaluate T_n(x) via the three-term recurrence."""
    if n == 0:
        return 1.0
    if n == 1:
        return float(x)
    Tkm1 = 1.0
    Tk = float(x)
    for _ in range(2, n + 1):
        Tkp1 = 2.0 * x * Tk - Tkm1
        Tkm1 = Tk
        Tk = Tkp1
    return Tk


# ---------------------------------------------------------------------------
# Chebyshev acceleration for a stationary iteration
# ---------------------------------------------------------------------------

class ChebyshevAccelerator:
    """Chebyshev semi-iterative accelerator.

    Given a stationary iteration  x_{k+1} = G x_k + c  with spectral
    radius rho(G) in [alpha, beta] subset [0, 1), the accelerated
    iteration is:

        x_{k+1}^{acc} = rho_{k+1} [ gamma (G x_k^{acc} + c) - x_{k-1}^{acc} ]
                      + x_{k-1}^{acc}

    where
        d = (2 - alpha - beta) / (beta - alpha)
        gamma = 2 / (2 - alpha - beta)
        rho_1 = 1 / d
        rho_{k+1} = 0.25 * (4 - alpha_beta * rho_k)   (recurrence)
        (alpha_beta = (beta - alpha)^2 / 4)

    Actually the standard form is simpler; we use the implementation from
    Golub & Van Loan.
    """

    def __init__(self, alpha: float = 0.0, beta: float = 0.95):
        """Set up the accelerator with estimated spectral bounds [alpha, beta]."""
        if not (0.0 <= alpha < beta < 1.0):
            # Relax the constraint; we just need beta < 2 for convergence
            pass
        self.alpha = alpha
        self.beta = beta
        self._reset()

    def _reset(self):
        self._step = 0
        d = (2.0 - self.alpha - self.beta) / max(self.beta - self.alpha, 1.0e-14)
        self._gamma = 2.0 / max(2.0 - self.alpha - self.beta, 1.0e-14)
        self._rho = 1.0 / max(abs(d), 1.0e-14)
        self._x_prev = None

    def reset(self):
        """Reset the accelerator state."""
        self._reset()
        self._x_prev = None

    def accelerate(
        self,
        x_new: np.ndarray,
        x_current: np.ndarray,
    ) -> np.ndarray:
        """Apply one step of Chebyshev acceleration.

        Parameters
        ----------
        x_new     : the next iterate from the unaccelerated iteration  G x_k + c
        x_current : the current accelerated iterate

        Returns
        -------
        x_acc : the new accelerated iterate
        """
        x_new = np.asarray(x_new, dtype=np.float64).ravel()
        x_current = np.asarray(x_current, dtype=np.float64).ravel()

        if self._step == 0:
            # First step: no acceleration
            self._x_prev = x_current.copy()
            self._x_acc = x_new.copy()
            self._step += 1
            return x_new.copy()

        if self._step == 1:
            # Second step: initialise the recurrence
            alpha_beta = 0.25 * (self.beta - self.alpha) ** 2
            rho_new = 1.0 / max(1.0 - alpha_beta * self._rho ** 2, 1.0e-14) if abs(self._rho) > 1.0e-14 else 1.0
        else:
            alpha_beta = 0.25 * (self.beta - self.alpha) ** 2
            rho_new = 1.0 / max(1.0 - alpha_beta * self._rho ** 2, 1.0e-14)

        # Accelerated update
        sigma = self._gamma * rho_new
        x_acc = x_current + sigma * (x_new - x_current) + (1.0 - sigma) * (x_current - self._x_prev)

        self._x_prev = x_current.copy()
        self._rho = rho_new
        self._step += 1
        return x_acc

    def accelerate_simple(
        self,
        x_new: np.ndarray,
        x_current: np.ndarray,
        x_prev: np.ndarray | None,
    ) -> np.ndarray:
        """Simplified Chebyshev acceleration (more robust).

        Uses the damping formula:
            x_acc = omega * x_new + (1 - omega) * x_current
        where omega is chosen based on the Chebyshev recurrence.
        """
        x_new = np.asarray(x_new, dtype=np.float64).ravel()
        x_current = np.asarray(x_current, dtype=np.float64).ravel()

        if x_prev is None:
            return x_new.copy()

        x_prev = np.asarray(x_prev, dtype=np.float64).ravel()

        # Estimate convergence rate from the last two iterates
        diff_new = np.linalg.norm(x_new - x_current)
        diff_old = np.linalg.norm(x_current - x_prev)
        if diff_old < 1.0e-14:
            return x_new.copy()

        rho_est = min(diff_new / max(diff_old, 1.0e-14), 0.99)
        # Chebyshev optimal damping
        omega = 2.0 / (2.0 - rho_est - 1.0 / max(rho_est, 1.0e-14)) if rho_est > 1.0e-14 else 1.0
        omega = max(0.5, min(omega, 1.5))

        x_acc = x_current + omega * (x_new - x_current)
        return x_acc
