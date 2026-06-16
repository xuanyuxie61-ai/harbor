"""
line_search.py
==============
Brent's method for scalar line minimization.

Mathematical background
-----------------------
Given a scalar function f(x) on a bracket [a, b] containing a minimum,
Brent's method combines:
  (i)   Golden section search, which guarantees linear convergence, with
  (ii)  Successive parabolic interpolation, which gives superlinear
        convergence (order ~ 1.3247) when f has a positive continuous
        second derivative at the minimum.

The algorithm maintains six quantities:
  a, b   : current bracket endpoints
  x      : best point found (smallest f so far)
  w, v   : second and third best points
  d      : most recent step size
  e      : step size before that

At each iteration, the trial step u = x + d is chosen as follows:

  If |e| > tol1:
      Fit a parabola through (v, fv), (w, fw), (x, fx) and take its
      vertex as the trial step, PROVIDED:
        (a) |p/q| < |e/2|     (step is smaller than the previous one)
        (b) p/q in (a - x, b - x)  (trial stays inside the bracket)
      Otherwise fall back to golden section: d = c * (b - x or a - x).
  Else:
      Golden section step.

The bracket shrinks at every iteration by at least a factor of c =
(3 - sqrt(5))/2 ~ 0.381966 (the squared inverse of the golden ratio).

Convergence criterion:
    |x - (a + b)/2| <= 2 * tol1 - (b - a) / 2
where tol1 = sqrt(eps) * |x| + eps / 3.

Reverse communication variant
-----------------------------
local_min_rc() implements the same algorithm using Burkardt's RC protocol,
which lets the caller control function evaluation. This is useful when the
function is expensive to evaluate or requires special setup per call.

KKT role
--------
In the primal-dual active-set method, after computing a descent direction
d_k, we need a step length alpha_k minimizing the merit function

    phi(alpha) = L(u_k + alpha d_k, lambda_k)

along the ray. Brent's method provides a robust derivative-free line search
that is tolerant to non-smoothness in the merit function (e.g. at active-set
boundaries).

References
----------
  - Brent, R.P., "Algorithms for Minimization Without Derivatives",
    Dover, 2002, ISBN 0-486-41998-3.
  - Burkardt, J., "local_min_rc" (Fortran90/MATLAB),
    https://people.sc.fsu.edu/~jburkardt/m_src/local_min_rc/
"""

from __future__ import annotations
import math
import numpy as np
from typing import Tuple, Callable


# ---------------------------------------------------------------------------
# Clean, direct Brent local minimizer (no RC)
# ---------------------------------------------------------------------------
def brent_local_min(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1.48e-8,
    max_iter: int = 500,
) -> Tuple[float, float, int]:
    """Minimize f on [a, b] using Brent's method.

    Parameters
    ----------
    f        : callable, the function to minimize
    a, b     : bracket endpoints (require a < b)
    tol      : convergence tolerance (relative)
    max_iter : maximum iterations

    Returns
    -------
    x_min  : float, the minimizer
    f_min  : float, the minimum value f(x_min)
    n_eval : int, number of function evaluations

    Notes
    -----
    The minimum cannot lie at a or b (the method cannot detect this).
    """
    if b <= a:
        raise ValueError("brent_local_min: a < b required.")

    eps = np.finfo(float).eps
    sqrt_eps = math.sqrt(eps)
    c = 0.5 * (3.0 - math.sqrt(5.0))  # ~0.381966

    # Initial interior point
    v = a + c * (b - a)
    w = v
    x = v
    e = 0.0
    d = 0.0
    fx = f(x); n_eval = 1
    fw = fx
    fv = fx

    for _ in range(max_iter):
        midpoint = 0.5 * (a + b)
        tol1 = sqrt_eps * abs(x) + tol / 3.0
        tol2 = 2.0 * tol1

        # Check convergence
        if abs(x - midpoint) <= tol2 - 0.5 * (b - a):
            return x, fx, n_eval

        # Decide between parabolic interpolation and golden section
        if abs(e) > tol1:
            # Try parabolic step
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            else:
                q = -q
            r_prev = e
            e = d

            # Accept parabola only if it's inside [a, b] and smaller than
            # half the previous step
            if (abs(p) < abs(0.5 * q * r_prev)
                    and q * (a - x) < p
                    and p < q * (b - x)):
                d = p / q
                u = x + d
                # If u is too close to a or b, nudge
                if (u - a) < tol2 or (b - u) < tol2:
                    d = tol1 if x < midpoint else -tol1
            else:
                # Golden section fallback
                e = (a - x) if x >= midpoint else (b - x)
                d = c * e
        else:
            # Golden section step
            e = (a - x) if x >= midpoint else (b - x)
            d = c * e

        # Compute trial point, ensuring it's not too close to x
        if abs(d) >= tol1:
            u = x + d
        else:
            u = x + (tol1 if d >= 0.0 else -tol1)

        fu = f(u)
        n_eval += 1

        # Update bracket and best points
        if fu <= fx:
            if u < x:
                a = u
            else:
                b = u
            # Shift records: v <- w <- x <- u
            v = w; fv = fw
            w = x; fw = fx
            x = u; fx = fu
        else:
            # u is not the new best
            if u < x:
                a = u
            else:
                b = u
            # Update second and third best
            if fu <= fw or w == x:
                v = w; fv = fw
                w = u; fw = fu
            elif fu <= fv or v == x or v == w:
                v = u; fv = fu

    return x, fx, n_eval


# ---------------------------------------------------------------------------
# Reverse-communication Brent minimizer (faithful port of Burkardt 695)
# ---------------------------------------------------------------------------
class BrentLocalMinRC:
    """Stateful Brent local minimizer with reverse-communication protocol.

    The original Fortran/MATLAB uses persistent variables. We use a class
    instance. Usage:

        solver = BrentLocalMinRC()
        arg, status = solver.start(a, b)
        while status > 0:
            value = f(arg)
            arg, status = solver.step(value)
        # arg now holds the minimizer

    status semantics (Burkardt protocol):
      -1 : error (A >= B)
       0 : converged
       1 : initial startup; user must evaluate f(x) and re-enter
      2+ : user must evaluate f(u) at the returned arg and re-enter
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._a = self._b = 0.0
        self._c = 0.5 * (3.0 - math.sqrt(5.0))
        self._d = 0.0
        self._e = 0.0
        self._u = 0.0
        self._v = self._w = self._x = 0.0
        self._fu = self._fv = self._fw = self._fx = 0.0
        self._arg_save = 0.0
        self._eps = np.finfo(float).eps
        self._eps_sqrt = math.sqrt(self._eps)
        self._tol = self._eps
        self._status = 0  # tracks user-facing status

    # ---- public API -------------------------------------------------------
    def start(self, a: float, b: float) -> Tuple[float, int]:
        """Initialize on [a, b]. Returns (arg, status)."""
        if b <= a:
            raise ValueError("BrentLocalMinRC: A < B required.")
        self.reset()
        self._a = float(a)
        self._b = float(b)
        self._v = a + self._c * (b - a)
        self._w = self._v
        self._x = self._v
        self._e = 0.0
        self._arg_save = self._x
        self._status = 1
        return self._x, 1

    def step(self, value: float) -> Tuple[float, int]:
        """Provide f(arg); return (next_arg, status)."""
        status = self._status

        # status 1: store the initial f(x)
        if status == 1:
            self._fx = value
            self._fv = value
            self._fw = value

        # status >= 2: update bracket with new f(u)
        elif status >= 2:
            self._fu = value
            fu = self._fu
            fx, fw, fv = self._fx, self._fw, self._fv
            u, x, w, v = self._u, self._x, self._w, self._v

            if fu <= fx:
                if x <= u:
                    self._a = x
                else:
                    self._b = x
                v, fv = w, fw
                w, fw = x, fx
                x, fx = u, fu
            else:
                if u < x:
                    self._a = u
                else:
                    self._b = u
                if fu <= fw or w == x:
                    v, fv = w, fw
                    w, fw = u, fu
                elif fu <= fv or v == x or v == w:
                    v, fv = u, fu

            self._v, self._w, self._x = v, w, x
            self._fv, self._fw, self._fx = fv, fw, fu

        # --- Compute next trial point ---
        midpoint = 0.5 * (self._a + self._b)
        tol1 = self._eps_sqrt * abs(self._x) + self._tol / 3.0
        tol2 = 2.0 * tol1

        # Convergence test
        if abs(self._x - midpoint) <= (tol2 - 0.5 * (self._b - self._a)):
            self._status = 0
            return self._arg_save, 0

        # Choose step type
        if abs(self._e) <= tol1:
            # Golden section
            if midpoint <= self._x:
                self._e = self._a - self._x
            else:
                self._e = self._b - self._x
            self._d = self._c * self._e
        else:
            # Parabolic interpolation
            r = (self._x - self._w) * (self._fx - self._fv)
            q = (self._x - self._v) * (self._fx - self._fw)
            p = (self._x - self._v) * q - (self._x - self._w) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = abs(q)
            r = self._e
            self._e = self._d

            # Accept parabolic step only if it's reasonable
            if (abs(0.5 * q * r) <= abs(p)
                    or p <= q * (self._a - self._x)
                    or q * (self._b - self._x) <= p):
                if midpoint <= self._x:
                    self._e = self._a - self._x
                else:
                    self._e = self._b - self._x
                self._d = self._c * self._e
            else:
                self._d = p / q
                self._u = self._x + self._d
                # If too close to boundaries, use tol1-sized step
                sgn = 1.0 if (midpoint - self._x) >= 0.0 else -1.0
                if (self._u - self._a) < tol2:
                    self._d = tol1 * sgn
                if (self._b - self._u) < tol2:
                    self._d = tol1 * sgn

        # Compute trial point
        if tol1 <= abs(self._d):
            self._u = self._x + self._d
        else:
            sgn = 1.0 if self._d >= 0.0 else -1.0
            self._u = self._x + tol1 * sgn

        self._arg_save = self._u
        self._status = status + 1 if status >= 1 else 2
        return self._u, self._status


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------
def brent_line_search(
    f: Callable[[float], float],
    a: float,
    b: float,
    max_iter: int = 500,
    tol: float = 1.48e-8,
) -> Tuple[float, float, int]:
    """Minimize f(x) on [a, b] using Brent's direct method.

    Returns
    -------
    x_min, f_min, n_eval
    """
    return brent_local_min(f, a, b, tol=tol, max_iter=max_iter)


def brent_line_search_rc(
    f: Callable[[float], float],
    a: float,
    b: float,
    max_iter: int = 500,
) -> Tuple[float, float, int]:
    """Minimize f(x) on [a, b] using Brent's RC method."""
    solver = BrentLocalMinRC()
    arg, status = solver.start(a, b)
    n_eval = 0
    for _ in range(max_iter):
        if status <= 0:
            break
        value = f(arg)
        n_eval += 1
        arg, status = solver.step(value)
    return arg, f(arg), n_eval


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("[line_search] Brent self-test")
    print("-" * 60)

    # Test 1: quadratic
    x1, f1, n1 = brent_line_search(lambda x: (x - 2.0) ** 2, 0.0, 5.0)
    print(f"  (x-2)^2 on [0,5]: x* = {x1:.12f} (err = {abs(x1-2):.2e}), "
          f"f = {f1:.2e}, n = {n1}")

    # Test 2: quadratic shifted
    x2, f2, n2 = brent_line_search(lambda x: (x - 1.7) ** 2, 0.0, 3.0)
    print(f"  (x-1.7)^2 on [0,3]: x* = {x2:.12f} (err = {abs(x2-1.7):.2e}), "
          f"f = {f2:.2e}, n = {n2}")

    # Test 3: sin(x) on [4, 5]
    x3, f3, n3 = brent_line_search(math.sin, 4.0, 5.0)
    x_exact = 1.5 * math.pi
    print(f"  sin(x) on [4,5]: x* = {x3:.12f} (exact {x_exact:.12f}, err = {abs(x3-x_exact):.2e})")

    # Test 4: RC variant
    x4, f4, n4 = brent_line_search_rc(lambda x: (x - 2.0) ** 2, 0.0, 5.0)
    print(f"  RC: (x-2)^2 on [0,5]: x* = {x4:.12f} (err = {abs(x4-2):.2e})")
