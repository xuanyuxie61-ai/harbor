"""
Lattice field interpolation utilities.

Adapted from:
  - 927_pwl_interp_2d (piecewise-linear interpolation on 2D triangulations)

Key formulas:
    Multilinear interpolation in d dimensions:
        f(x) = sum_{i in {0,1}^d} f(x_i) prod_{mu=1}^{d} [ (1-s_mu) if i_mu=0 else s_mu ]
"""

import numpy as np
from lattice_geometry import LatticeGeometry, SiteIndex
from gauge_field import GaugeField
from wilson_dirac import SpinorField
from constants import LatticeParams


def multilinear_interpolate_scalar(field_data, coords, shape):
    if len(coords) != len(shape):
        raise ValueError(f"coords dim {len(coords)} != shape dim {len(shape)}")
    d = len(shape)
    floors = np.floor(coords).astype(int)
    fracs = coords - floors
    for mu in range(d):
        if floors[mu] >= shape[mu] - 1:
            floors[mu] = shape[mu] - 1
            fracs[mu] = 0.0
        if floors[mu] < 0:
            floors[mu] = 0
            fracs[mu] = 0.0
    result = 0.0 + 0.0j
    for corner in range(2 ** d):
        bits = [(corner >> i) & 1 for i in range(d)]
        idx = tuple((floors[mu] + bits[mu]) % shape[mu] for mu in range(d))
        weight = 1.0
        for mu in range(d):
            weight *= (1.0 - fracs[mu]) if bits[mu] == 0 else fracs[mu]
        result += weight * field_data[idx]
    return complex(result)


def gauge_covariant_interpolate(gf, psi, coords):
    p = gf.p
    geo = gf.geo
    shape = p.shape
    d = 4
    floors = np.floor(coords).astype(int)
    fracs = coords - floors
    for mu in range(d):
        if floors[mu] >= shape[mu] - 1:
            floors[mu] = shape[mu] - 1
            fracs[mu] = 0.0
        if floors[mu] < 0:
            floors[mu] = 0
            fracs[mu] = 0.0
    base_site = SiteIndex(*[int(floors[mu]) for mu in range(d)])
    result = np.zeros((4, 2), dtype=complex)
    for corner in range(2 ** d):
        bits = [(corner >> i) & 1 for i in range(d)]
        site_idx = tuple((floors[mu] + bits[mu]) % shape[mu] for mu in range(d))
        site = SiteIndex(*[int(site_idx[mu]) for mu in range(d)])
        weight = 1.0
        for mu in range(d):
            weight *= (1.0 - fracs[mu]) if bits[mu] == 0 else fracs[mu]
        W = _wilson_line(gf, base_site, site)
        psi_site = psi.at(site)
        # W acts on color index: (W psi)_{alpha, c} = sum_d W_{cd} psi_{alpha, d}
        result += weight * (psi_site @ W.T)
    return result


def _wilson_line(gf, s_from, s_to):
    geo = gf.geo
    W = np.eye(2, dtype=complex)
    current = s_from
    for mu in range(4):
        delta = s_to[mu] - s_from[mu]
        L = gf.p.Nt if mu == 0 else gf.p.Ns
        if delta > L // 2:
            delta -= L
        elif delta < -L // 2:
            delta += L
        if delta > 0:
            for _ in range(delta):
                U = gf.link(current, mu)
                W = U @ W
                current = geo.neighbor(current, mu, True)
        elif delta < 0:
            for _ in range(-delta):
                current_m = geo.neighbor(current, mu, False)
                U_dag = gf.link(current_m, mu).conj().T
                W = U_dag @ W
                current = current_m
    return W


def newton_polynomial_interpolation(x_nodes, f_values, x_eval):
    if len(x_nodes) != len(f_values):
        raise ValueError("x_nodes and f_values must have same length")
    if len(x_nodes) == 0:
        raise ValueError("empty input")
    n = len(x_nodes)
    c = f_values.copy().astype(complex)
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            denom = x_nodes[i] - x_nodes[i - j]
            if abs(denom) < 1e-14:
                c[i] = 0.0
            else:
                c[i] = (c[i] - c[i - 1]) / denom
    result = np.zeros(len(x_eval), dtype=complex)
    for k, x in enumerate(x_eval):
        val = c[0]
        prod = 1.0 + 0.0j
        for j in range(1, n):
            prod *= (x - x_nodes[j - 1])
            val += c[j] * prod
        result[k] = val
    return result


def continuum_extrapolation(observables, a_values, a_cont=0.0, order=1):
    if len(observables) != len(a_values):
        raise ValueError("observables and a_values must have same length")
    if len(observables) < order + 1:
        raise ValueError(f"need at least {order + 1} data points for order {order}")
    x = a_values ** 2
    A = np.vander(x, N=order + 1, increasing=True)
    coeffs, residuals, _, _ = np.linalg.lstsq(A, observables, rcond=None)
    x_cont = a_cont ** 2
    O_cont = sum(coeffs[k] * x_cont ** k for k in range(order + 1))
    chi2 = float(np.sum((A @ coeffs - observables) ** 2))
    return {'O_cont': float(O_cont), 'coefficients': coeffs, 'chi2': chi2}
