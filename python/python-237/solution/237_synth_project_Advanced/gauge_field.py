"""
SU(2) gauge field configuration on the 4D lattice.

Adapted from:
  - 003_allen_cahn_pde (high-order finite differences / stencil operators)

Key formulas:
    Gauge-covariant forward derivative (order 2):
        nabla_mu^{(2)} psi(x) = [ U_mu(x) psi(x+mu) - psi(x) ] / a

    Gauge-covariant forward derivative (order 4):
        nabla_mu^{(4)} psi(x) = [ -U_mu(x) U_mu(x+mu) psi(x+2mu)
                                   + 8 U_mu(x) psi(x+mu)
                                   - 3 psi(x) ] / (6 a)

    Gauge-covariant forward derivative (order 6):
        nabla_mu^{(6)} psi(x) = [ U(x)...U(x+2mu) psi(x+3mu)
                                   - 9 U(x)U(x+mu) psi(x+2mu)
                                   + 45 U(x) psi(x+mu)
                                   - 20 psi(x) ] / (60 a)

    Gauge-covariant Laplacian (order 2):
        Delta^{(2)} psi(x) = sum_mu [ U_mu(x) psi(x+mu)
                                       + U_mu^dag(x-mu) psi(x-mu)
                                       - 2 psi(x) ] / a^2
"""

import numpy as np
from constants import LatticeParams, su2_identity, su2_random
from lattice_geometry import LatticeGeometry, SiteIndex


class GaugeField:
    """SU(2) gauge field on a 4D hypercubic lattice with periodic BC."""

    def __init__(self, params: LatticeParams, rng: np.random.Generator):
        self.p = params
        self.geo = LatticeGeometry(params)
        self.rng = rng
        self._shape = params.shape + (4,)
        self._U = np.empty(self._shape + (2, 2), dtype=complex)

    def initialize_cold(self):
        for idx in np.ndindex(*self._shape):
            self._U[idx] = su2_identity()
        return self

    def initialize_hot(self):
        for idx in np.ndindex(*self._shape):
            self._U[idx] = su2_random(self.rng)
        return self

    def initialize_approx_cold(self, noise_level: float = 0.1):
        if noise_level < 0.0 or noise_level > 1.0:
            raise ValueError(f"noise_level must be in [0,1], got {noise_level}")
        for idx in np.ndindex(*self._shape):
            U = su2_identity() + noise_level * (su2_random(self.rng) - su2_identity())
            self._U[idx] = _project_to_su2(U)
        return self

    def link(self, s: SiteIndex, mu: int) -> np.ndarray:
        return self._U[s.t, s.x, s.y, s.z, mu]

    def set_link(self, s: SiteIndex, mu: int, U: np.ndarray):
        self._U[s.t, s.x, s.y, s.z, mu] = U

    def mul_link(self, s: SiteIndex, mu: int, V: np.ndarray, side: str = 'left'):
        U = self.link(s, mu)
        if side == 'left':
            new_U = V @ U
        elif side == 'right':
            new_U = U @ V
        else:
            raise ValueError(f"side must be 'left' or 'right', got {side}")
        self.set_link(s, mu, _project_to_su2(new_U))

    def staple_sum(self, s: SiteIndex, mu: int) -> np.ndarray:
        """Compute the staple sum S'_mu(x) for Metropolis updates."""
        geo = self.geo
        staple = np.zeros((2, 2), dtype=complex)
        for nu in range(4):
            if nu == mu:
                continue
            s_mu = geo.neighbor(s, mu, True)
            s_nu = geo.neighbor(s, nu, True)
            U_nu_smu = self.link(s_mu, nu)
            U_mu_snu_dag = self.link(s_nu, mu).conj().T
            U_nu_s_dag = self.link(s, nu).conj().T
            staple += U_nu_smu @ U_mu_snu_dag @ U_nu_s_dag

            s_munu = geo.neighbor(s_mu, nu, False)
            s_m = geo.neighbor(s, mu, False)
            s_n = geo.neighbor(s, nu, False)
            U_nu_smunu_dag = self.link(s_munu, nu).conj().T
            U_mu_sm_dag = self.link(s_m, mu).conj().T
            U_nu_sn = self.link(s_n, nu)
            staple += U_nu_smunu_dag @ U_mu_sm_dag @ U_nu_sn
        return staple

    def gauge_transform(self, omega):
        geo = self.geo
        for idx in np.ndindex(*self._shape):
            t, x, y, z, mu = idx
            s = SiteIndex(t, x, y, z)
            s_mu = geo.neighbor(s, mu, True)
            U = self._U[idx]
            Om_x = omega[t, x, y, z]
            Om_xmu = omega[s_mu.t, s_mu.x, s_mu.y, s_mu.z]
            new_U = Om_x @ U @ Om_xmu.conj().T
            self._U[idx] = _project_to_su2(new_U)

    def copy(self) -> 'GaugeField':
        gf = GaugeField(self.p, self.rng)
        gf._U = self._U.copy()
        return gf

    def action_density(self) -> float:
        beta = self.p.beta
        geo = self.geo
        total = 0.0
        for s in geo.all_sites():
            for mu in range(4):
                for nu in range(mu + 1, 4):
                    U_p = _plaquette(self, s, mu, nu)
                    tr = 0.5 * (U_p[0, 0].real + U_p[1, 1].real)
                    total += 1.0 - tr
        return beta * total


def _plaquette(gf: GaugeField, s, mu: int, nu: int) -> np.ndarray:
    geo = gf.geo
    s_mu = geo.neighbor(s, mu, True)
    s_nu = geo.neighbor(s, nu, True)
    return (gf.link(s, mu) @ gf.link(s_mu, nu)
            @ gf.link(s_nu, mu).conj().T @ gf.link(s, nu).conj().T)


def _project_to_su2(U: np.ndarray) -> np.ndarray:
    a0 = 0.5 * (U[0, 0] + U[1, 1]).real
    a1 = 0.5 * (U[0, 1] + U[1, 0]).imag
    a2 = 0.5 * (-U[0, 1] + U[1, 0]).real
    a3 = 0.5 * (U[0, 0] - U[1, 1]).imag
    norm = np.sqrt(a0**2 + a1**2 + a2**2 + a3**2 + 1e-30)
    a0, a1, a2, a3 = a0 / norm, a1 / norm, a2 / norm, a3 / norm
    return np.array([
        [a0 + 1j * a3, a2 + 1j * a1],
        [-a2 + 1j * a1, a0 - 1j * a3],
    ], dtype=complex)


# ---------------------------------------------------------------------------
# Covariant finite-difference stencils (high order)
# ---------------------------------------------------------------------------

def _color_act(U, psi):
    """Apply SU(2) matrix U (2x2) to spinor psi (Nd x Nc) on color index.
    Returns shape (Nd, Nc) with (U psi)_{alpha, c} = sum_d U_{cd} psi_{alpha, d}."""
    return psi @ U.T


def covariant_derivative(gf: GaugeField, psi_at_sites, s: SiteIndex,
                          mu: int, order: int = 2) -> np.ndarray:
    a = gf.p.a
    geo = gf.geo

    if order == 2:
        psi_x = psi_at_sites(s)
        s_mu = geo.neighbor(s, mu, True)
        psi_xmu = psi_at_sites(s_mu)
        U_x = gf.link(s, mu)
        return (_color_act(U_x, psi_xmu) - psi_x) / a

    elif order == 4:
        psi_x = psi_at_sites(s)
        s1 = geo.neighbor(s, mu, True)
        s2 = geo.neighbor(s1, mu, True)
        U_x = gf.link(s, mu)
        U_x1 = gf.link(s1, mu)
        psi_x1 = psi_at_sites(s1)
        psi_x2 = psi_at_sites(s2)
        return (-_color_act(U_x @ U_x1, psi_x2) + 8.0 * _color_act(U_x, psi_x1) - 3.0 * psi_x) / (6.0 * a)

    elif order == 6:
        psi_x = psi_at_sites(s)
        s1 = geo.neighbor(s, mu, True)
        s2 = geo.neighbor(s1, mu, True)
        s3 = geo.neighbor(s2, mu, True)
        U_x = gf.link(s, mu)
        U_x1 = gf.link(s1, mu)
        U_x2 = gf.link(s2, mu)
        psi_x1 = psi_at_sites(s1)
        psi_x2 = psi_at_sites(s2)
        psi_x3 = psi_at_sites(s3)
        num = (_color_act(U_x @ U_x1 @ U_x2, psi_x3)
               - 9.0 * _color_act(U_x @ U_x1, psi_x2)
               + 45.0 * _color_act(U_x, psi_x1)
               - 20.0 * psi_x)
        return num / (60.0 * a)
    else:
        raise ValueError(f"order must be 2, 4, or 6, got {order}")


def covariant_backward_derivative(gf: GaugeField, psi_at_sites, s: SiteIndex,
                                    mu: int, order: int = 2) -> np.ndarray:
    a = gf.p.a
    geo = gf.geo

    if order == 2:
        psi_x = psi_at_sites(s)
        s_m = geo.neighbor(s, mu, False)
        psi_xm = psi_at_sites(s_m)
        U_xm_dag = gf.link(s_m, mu).conj().T
        return (psi_x - _color_act(U_xm_dag, psi_xm)) / a
    elif order == 4:
        psi_x = psi_at_sites(s)
        s_m = geo.neighbor(s, mu, False)
        s_mm = geo.neighbor(s_m, mu, False)
        U_xm_dag = gf.link(s_m, mu).conj().T
        U_xmm_dag = gf.link(s_mm, mu).conj().T
        psi_xm = psi_at_sites(s_m)
        psi_xmm = psi_at_sites(s_mm)
        return (3.0 * psi_x - 8.0 * _color_act(U_xm_dag, psi_xm)
                + _color_act(U_xmm_dag @ U_xm_dag, psi_xmm)) / (6.0 * a)
    elif order == 6:
        psi_x = psi_at_sites(s)
        s_m = geo.neighbor(s, mu, False)
        s_mm = geo.neighbor(s_m, mu, False)
        s_mmm = geo.neighbor(s_mm, mu, False)
        U1d = gf.link(s_m, mu).conj().T
        U2d = gf.link(s_mm, mu).conj().T
        U3d = gf.link(s_mmm, mu).conj().T
        return (20.0 * psi_x
                - 45.0 * _color_act(U1d, psi_at_sites(s_m))
                + 9.0 * _color_act(U2d @ U1d, psi_at_sites(s_mm))
                - _color_act(U3d @ U2d @ U1d, psi_at_sites(s_mmm))) / (60.0 * a)
    else:
        raise ValueError(f"order must be 2, 4, or 6, got {order}")


def symmetric_derivative(gf: GaugeField, psi_at_sites, s: SiteIndex,
                          mu: int, order: int = 2) -> np.ndarray:
    fwd = covariant_derivative(gf, psi_at_sites, s, mu, order)
    bwd = covariant_backward_derivative(gf, psi_at_sites, s, mu, order)
    return 0.5 * (fwd + bwd)


def covariant_laplacian(gf: GaugeField, psi_at_sites, s: SiteIndex,
                         order: int = 2) -> np.ndarray:
    a = gf.p.a
    geo = gf.geo
    psi_x = psi_at_sites(s)

    if order == 2:
        lap = np.zeros_like(psi_x)
        for mu in range(4):
            s_mu = geo.neighbor(s, mu, True)
            s_m = geo.neighbor(s, mu, False)
            U_x = gf.link(s, mu)
            U_xm_dag = gf.link(s_m, mu).conj().T
            lap += (_color_act(U_x, psi_at_sites(s_mu))
                    + _color_act(U_xm_dag, psi_at_sites(s_m))
                    - 2.0 * psi_x)
        return lap / (a ** 2)

    elif order == 4:
        lap = np.zeros_like(psi_x)
        for mu in range(4):
            s1 = geo.neighbor(s, mu, True)
            s2 = geo.neighbor(s1, mu, True)
            sm1 = geo.neighbor(s, mu, False)
            sm2 = geo.neighbor(sm1, mu, False)
            U1 = gf.link(s, mu)
            U2 = gf.link(s1, mu)
            Um1d = gf.link(sm1, mu).conj().T
            Um2d = gf.link(sm2, mu).conj().T
            lap += (-(_color_act(U1 @ U2, psi_at_sites(s2))
                      + _color_act(Um2d @ Um1d, psi_at_sites(sm2)))
                    + 16.0 * (_color_act(U1, psi_at_sites(s1))
                              + _color_act(Um1d, psi_at_sites(sm1)))
                    - 30.0 * psi_x)
        return lap / (12.0 * a ** 2)
    else:
        raise ValueError(f"Laplacian order must be 2 or 4, got {order}")
