"""
Quark propagator solver via iterative methods.

Adapted from:
  - 130_bvp_shooting (boundary-value shooting method)

The quark propagator S(x, y) satisfies D_W S(x, y) = delta_{x,y}.

Algorithms:
    - Jacobi iteration: S_{n+1} = S_n + omega (eta - D_W S_n)
    - Conjugate Gradient on normal equations (CGNE)
    - Stabilized BiCGSTAB
"""

import numpy as np
from constants import LatticeParams
from lattice_geometry import SiteIndex
from gauge_field import GaugeField
from wilson_dirac import (WilsonDiracOperator, SpinorField, make_point_source,
                           _gamma5)


class PropagatorSolver:
    def __init__(self, D: WilsonDiracOperator, source: SpinorField,
                  max_iter: int = 500, tol: float = 1e-6):
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter}")
        if tol <= 0.0:
            raise ValueError(f"tol must be > 0, got {tol}")
        self.D = D
        self.source = source
        self.max_iter = max_iter
        self.tol = tol
        self.p = D.p
        self.history = {'residuals': [], 'converged': False, 'iterations': 0}


class JacobiSolver(PropagatorSolver):
    def __init__(self, D, source, omega=None, max_iter=500, tol=1e-6):
        super().__init__(D, source, max_iter, tol)
        if omega is None:
            self.omega = 1.0 / (1.0 + 4.0 * D.kappa)
        else:
            if omega <= 0.0 or omega >= 2.0:
                raise ValueError(f"omega must be in (0, 2), got {omega}")
            self.omega = omega

    def solve(self) -> SpinorField:
        x = SpinorField(self.p, zero=True)
        r = self.source.copy()
        r_norm0 = np.sqrt(r.norm_sq() + 1e-30)
        for it in range(self.max_iter):
            r_norm = np.sqrt(r.norm_sq())
            self.history['residuals'].append(r_norm)
            if r_norm / r_norm0 < self.tol:
                self.history['converged'] = True
                self.history['iterations'] = it + 1
                return x
            Dr = self.D.apply(r)
            x.data += self.omega * r.data
            r.data -= self.omega * Dr.data
        self.history['iterations'] = self.max_iter
        return x


class CGNESolver(PropagatorSolver):
    def solve(self) -> SpinorField:
        b = self.source
        b_g5 = _gamma5(b)
        D_b_g5 = self.D.apply(b_g5)
        rhs = _gamma5(D_b_g5)
        x = SpinorField(self.p, zero=True)
        r = rhs.copy()
        p = r.copy()
        r_sq = r.norm_sq().real
        r_sq_0 = r_sq
        for it in range(self.max_iter):
            if np.sqrt(r_sq + 1e-30) / np.sqrt(r_sq_0 + 1e-30) < self.tol:
                self.history['converged'] = True
                self.history['iterations'] = it
                self.history['residuals'].append(np.sqrt(r_sq))
                return x
            Np = _apply_DdagD(self.D, p)
            pNp = p.dot(Np).real
            if abs(pNp) < 1e-30:
                self.history['iterations'] = it
                self.history['residuals'].append(np.sqrt(r_sq))
                return x
            alpha = r_sq / pNp
            x.data += alpha * p.data
            r.data -= alpha * Np.data
            r_sq_new = r.norm_sq().real
            self.history['residuals'].append(np.sqrt(r_sq_new))
            if r_sq < 1e-30:
                beta = 0.0
            else:
                beta = r_sq_new / r_sq
            p.data = r.data + beta * p.data
            r_sq = r_sq_new
        self.history['iterations'] = self.max_iter
        return x


class BiCGSTABSolver(PropagatorSolver):
    def solve(self) -> SpinorField:
        A = self.D.apply
        b = self.source
        x = SpinorField(self.p, zero=True)
        r = b.copy()
        r0_hat = r.copy()
        rho = 1.0
        alpha = 1.0
        omega = 1.0
        v = SpinorField(self.p, zero=True)
        p = SpinorField(self.p, zero=True)
        r_norm0 = np.sqrt(r.norm_sq() + 1e-30)
        for it in range(1, self.max_iter + 1):
            rho_new = r0_hat.dot(r)
            if abs(rho_new) < 1e-30:
                self.history['iterations'] = it
                self.history['residuals'].append(np.sqrt(r.norm_sq()))
                return x
            beta = (rho_new / (rho + 1e-30)) * (alpha / (omega + 1e-30))
            p.data = r.data + beta * (p.data - omega * v.data)
            v = A(p)
            denom = r0_hat.dot(v)
            if abs(denom) < 1e-30:
                self.history['iterations'] = it
                self.history['residuals'].append(np.sqrt(r.norm_sq()))
                return x
            alpha = rho_new / denom
            s = _axpby(1.0, r, -alpha, v)
            s_norm = np.sqrt(s.norm_sq())
            if s_norm / r_norm0 < self.tol:
                x.data += alpha * p.data
                self.history['converged'] = True
                self.history['iterations'] = it
                self.history['residuals'].append(s_norm)
                return x
            t = A(s)
            t_sq = t.norm_sq().real
            if t_sq < 1e-30:
                omega = 0.0
            else:
                omega = t.dot(s) / t_sq
            x.data += alpha * p.data + omega * s.data
            r.data = s.data - omega * t.data
            r_norm = np.sqrt(r.norm_sq())
            self.history['residuals'].append(r_norm)
            if r_norm / r_norm0 < self.tol:
                self.history['converged'] = True
                self.history['iterations'] = it
                return x
            rho = rho_new
        self.history['iterations'] = self.max_iter
        return x


def _apply_DdagD(D, psi):
    psi_g5 = _gamma5(psi)
    D_g5 = D.apply(psi_g5)
    Ddag = _gamma5(D_g5)
    return D.apply(Ddag)


def _axpby(a, x, b, y):
    r = SpinorField(x.p, zero=False)
    r.data = a * x.data + b * y.data
    return r


def solve_propagator(gf, source_site, method='cgne', max_iter=200,
                      tol=1e-5, spin_idx=0, color_idx=0):
    D = WilsonDiracOperator(gf)
    src = make_point_source(gf.p, source_site, spin_idx, color_idx)
    if method == 'jacobi':
        solver = JacobiSolver(D, src, max_iter=max_iter, tol=tol)
    elif method == 'cgne':
        solver = CGNESolver(D, src, max_iter=max_iter, tol=tol)
    elif method == 'bicgstab':
        solver = BiCGSTABSolver(D, src, max_iter=max_iter, tol=tol)
    else:
        raise ValueError(f"Unknown method '{method}'")
    prop = solver.solve()
    return {'propagator': prop, 'solver_info': solver.history}
