"""
Wilson-Dirac operator and related Dirac operators on the lattice.

Key formulas:
    Wilson-Dirac operator:
        (D_W psi)(x) = psi(x) - kappa * sum_mu [
            (1 - gamma_mu) U_mu(x) psi(x+mu)
            + (1 + gamma_mu) U_mu^dag(x-mu) psi(x-mu) ]

    Improved Wilson-Dirac (clover):
        D_SW = D_W + i c_SW kappa (a/4) sigma_{mu nu} F_{mu nu}(x)

Stability:
    kappa < kappa_c = 1/(2d) = 1/8 for convergence.
"""

import numpy as np
from constants import LatticeParams, GAMMA
from lattice_geometry import LatticeGeometry, SiteIndex
from gauge_field import GaugeField


class SpinorField:
    """Dirac spinor field: shape (Nt, Ns, Ns, Ns, Nd=4, Nc=2)."""

    def __init__(self, params: LatticeParams, zero: bool = True):
        self.p = params
        shape = params.shape + (4, 2)
        if zero:
            self.data = np.zeros(shape, dtype=complex)
        else:
            self.data = np.empty(shape, dtype=complex)

    def at(self, s: SiteIndex) -> np.ndarray:
        return self.data[s.t, s.x, s.y, s.z]

    def set(self, s: SiteIndex, val: np.ndarray):
        self.data[s.t, s.x, s.y, s.z] = val

    def copy(self) -> 'SpinorField':
        sf = SpinorField(self.p, zero=False)
        sf.data = self.data.copy()
        return sf

    def norm_sq(self) -> float:
        return float(np.sum(np.abs(self.data) ** 2))

    def dot(self, other: 'SpinorField') -> complex:
        return complex(np.sum(self.data.conj() * other.data))


def make_point_source(params: LatticeParams, s0: SiteIndex,
                       spin_idx: int = 0, color_idx: int = 0) -> SpinorField:
    src = SpinorField(params, zero=True)
    src.data[s0.t, s0.x, s0.y, s0.z, spin_idx, color_idx] = 1.0
    return src


def make_random_source(params: LatticeParams, rng: np.random.Generator) -> SpinorField:
    src = SpinorField(params, zero=False)
    phases = np.array([1.0, 1j, -1.0, -1j])
    idx = rng.integers(0, 4, size=params.shape + (4, 2))
    src.data = phases[idx]
    return src


class WilsonDiracOperator:
    """Wilson-Dirac operator as a matrix-free linear operator."""

    def __init__(self, gf: GaugeField):
        self.gf = gf
        self.p = gf.p
        self.kappa = self.p.kappa
        self.geo = gf.geo
        self._validate_kappa()

    def _validate_kappa(self):
        kc = self.p.critical_kappa()
        if self.kappa <= 0.0:
            raise ValueError(f"kappa must be > 0, got {self.kappa}")
        if self.kappa >= kc:
            raise ValueError(f"kappa = {self.kappa} >= kappa_c = {kc}")

    def apply(self, psi: SpinorField) -> SpinorField:
        phi = SpinorField(self.p, zero=True)
        for s in self.geo.all_sites():
            val = self._apply_at(psi, s)
            phi.set(s, val)
        return phi

    def _apply_at(self, psi: SpinorField, s: SiteIndex) -> np.ndarray:
        psi_x = psi.at(s)
        result = psi_x.copy()
        for mu in range(4):
            s_mu = self.geo.neighbor(s, mu, True)
            s_m = self.geo.neighbor(s, mu, False)
            U_x = self.gf.link(s, mu)
            U_xm_dag = self.gf.link(s_m, mu).conj().T
            psi_xmu = psi.at(s_mu)
            psi_xm = psi.at(s_m)
            # U acts on color index (second): (U psi)_{alpha, c} = sum_d U_{cd} psi_{alpha, d}
            U_psi_xmu = psi_xmu @ U_x.T
            Udag_psi_xm = psi_xm @ U_xm_dag.T
            forward = (np.eye(4, dtype=complex) - GAMMA[mu]) @ U_psi_xmu
            backward = (np.eye(4, dtype=complex) + GAMMA[mu]) @ Udag_psi_xm
            result = result - self.kappa * (forward + backward)
        return result

    def gamma5_apply(self, psi: SpinorField) -> SpinorField:
        Dpsi = self.apply(psi)
        result = SpinorField(self.p, zero=False)
        result.data[:, :, :, :, 0:2, :] = Dpsi.data[:, :, :, :, 0:2, :]
        result.data[:, :, :, :, 2:4, :] = -Dpsi.data[:, :, :, :, 2:4, :]
        return result


def estimate_condition_number(gf: GaugeField, n_iter: int = 20, seed: int = 42) -> dict:
    """Estimate spectral condition number of D_W via Lanczos on D^dag D."""
    rng = np.random.default_rng(seed)
    D = WilsonDiracOperator(gf)

    v = make_random_source(gf.p, rng)
    v_norm = np.sqrt(v.norm_sq() + 1e-30)
    v.data /= v_norm

    alphas = []
    betas = []
    v_prev = None

    for k in range(n_iter):
        w = _apply_DdagD(D, v)
        alpha = v.dot(w).real
        alphas.append(alpha)
        w = _spinor_axpby(1.0, w, -alpha, v)
        if v_prev is not None:
            w = _spinor_axpby(1.0, w, -betas[-1], v_prev)
        beta = np.sqrt(w.norm_sq() + 1e-30)
        betas.append(beta)
        v_prev = v
        v = _spinor_scale(w, 1.0 / (beta + 1e-30))

    n = len(alphas)
    T = np.zeros((n, n))
    for i in range(n):
        T[i, i] = alphas[i]
        if i < n - 1:
            T[i, i + 1] = betas[i]
            T[i + 1, i] = betas[i]
    eigs = np.linalg.eigvalsh(T)
    lam_min = float(max(eigs.min(), 1e-30))
    lam_max = float(max(eigs.max(), 1e-30))

    return {
        'lambda_max': lam_max,
        'lambda_min': lam_min,
        'condition_number': lam_max / lam_min,
    }


def _apply_DdagD(D, psi):
    psi_g5 = _gamma5(psi)
    D_g5 = D.apply(psi_g5)
    Ddag = _gamma5(D_g5)
    return D.apply(Ddag)


def _gamma5(psi):
    result = SpinorField(psi.p, zero=False)
    result.data[:, :, :, :, 0:2, :] = psi.data[:, :, :, :, 0:2, :]
    result.data[:, :, :, :, 2:4, :] = -psi.data[:, :, :, :, 2:4, :]
    return result


def _spinor_axpby(a, x, b, y):
    result = SpinorField(x.p, zero=False)
    result.data = a * x.data + b * y.data
    return result


def _spinor_scale(x, c):
    result = SpinorField(x.p, zero=False)
    result.data = c * x.data
    return result
