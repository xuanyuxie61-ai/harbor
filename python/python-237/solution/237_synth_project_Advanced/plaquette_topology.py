"""
Plaquette quality analysis and topological charge.

Adapted from:
  - 1348_triangulation_quality (element quality metrics)
  - 446_fractal_coastline (fractal dimension / escape-time analysis)

Key formulas:
    Topological charge (clover):
        Q_top = (1/(32 pi^2)) sum_x eps_{mu nu rho sigma} Tr F_{mu nu} F_{rho sigma}
    Topological susceptibility:
        chi_t = <Q^2> / V
    Fractal dimension:
        D_box = lim_{eps->0} log N(eps) / log(1/eps)
"""

import numpy as np
from constants import LatticeParams
from lattice_geometry import LatticeGeometry, SiteIndex
from gauge_field import GaugeField


def plaquette_quality_field(gf: GaugeField) -> dict:
    values = []
    geo = gf.geo
    for s in geo.all_sites():
        for mu in range(4):
            for nu in range(mu + 1, 4):
                s_mu = geo.neighbor(s, mu, True)
                s_nu = geo.neighbor(s, nu, True)
                U_p = (gf.link(s, mu) @ gf.link(s_mu, nu)
                       @ gf.link(s_nu, mu).conj().T @ gf.link(s, nu).conj().T)
                tr = 0.5 * (U_p[0, 0].real + U_p[1, 1].real)
                values.append(tr)
    arr = np.array(values)
    mean = float(arr.mean())
    std = float(arr.std()) + 1e-30
    skew = float(((arr - mean) ** 3).mean()) / (std ** 3 + 1e-30)
    kurt = float(((arr - mean) ** 4).mean()) / (std ** 4 + 1e-30) - 3.0
    return {
        'values': arr,
        'mean': mean,
        'std': float(arr.std()),
        'skewness': skew,
        'kurtosis': kurt,
        'quality_fraction': float(np.mean(arr > 0.9)),
        'hot_fraction': float(np.mean(arr < 0.5)),
        'min': float(arr.min()),
        'max': float(arr.max()),
    }


def topological_charge_clover(gf: GaugeField) -> dict:
    geo = gf.geo
    p = gf.p
    q_density = np.zeros(p.V)
    for s in geo.all_sites():
        q_x = _topological_charge_density_at(gf, s)
        idx = geo.site_to_flat(s)
        q_density[idx] = q_x
    Q = float(q_density.sum())
    return {'Q': Q, 'Q_round': int(round(Q)), 'q_density': q_density, 'abs_Q': abs(Q)}


def _topological_charge_density_at(gf, s):
    F = {}
    for mu in range(4):
        for nu in range(mu + 1, 4):
            F[(mu, nu)] = _clover_F(gf, s, mu, nu)
    term1 = np.trace(F[(0, 1)] @ F[(2, 3)])
    term2 = np.trace(F[(0, 2)] @ F[(1, 3)])
    term3 = np.trace(F[(0, 3)] @ F[(1, 2)])
    q = (8.0 / (32.0 * np.pi ** 2)) * (term1 - term2 + term3)
    return float(q.real)


def _clover_F(gf, s, mu, nu):
    geo = gf.geo
    s_mu = geo.neighbor(s, mu, True)
    s_nu = geo.neighbor(s, nu, True)
    P1 = gf.link(s, mu) @ gf.link(s_mu, nu) @ gf.link(s_nu, mu).conj().T @ gf.link(s, nu).conj().T
    Q = P1
    Q_anti = Q - Q.conj().T
    tr_Q_anti = np.trace(Q_anti)
    F = (Q_anti - 0.5 * tr_Q_anti * np.eye(2, dtype=complex)) / (8.0j)
    return F


def fractal_dimension_topological(q_density, shape, threshold=0.001, n_boxes=5):
    if n_boxes < 2:
        raise ValueError(f"n_boxes must be >= 2, got {n_boxes}")
    Nt, Ns = shape[0], shape[1]
    q_4d = q_density.reshape(shape)
    support = np.abs(q_4d) > threshold
    max_k = int(np.log2(min(Ns, Nt))) - 1
    if max_k < 1:
        max_k = 1
    box_sizes = []
    N_boxes = []
    for k in range(0, max_k + 1):
        bs = max(1, Ns // (2 ** k))
        bt = max(1, Nt // (2 ** k))
        count = 0
        for t0 in range(0, Nt, bt):
            for x0 in range(0, Ns, bs):
                for y0 in range(0, Ns, bs):
                    for z0 in range(0, Ns, bs):
                        t1 = min(t0 + bt, Nt)
                        x1 = min(x0 + bs, Ns)
                        y1 = min(y0 + bs, Ns)
                        z1 = min(z0 + bs, Ns)
                        if support[t0:t1, x0:x1, y0:y1, z0:z1].any():
                            count += 1
        box_sizes.append(bs)
        N_boxes.append(count)
    box_sizes = np.array(box_sizes, dtype=float)
    N_boxes = np.array(N_boxes, dtype=float)
    valid = (box_sizes > 0) & (N_boxes > 0)
    if valid.sum() < 2:
        return {'D_box': 0.0, 'box_sizes': box_sizes, 'N_boxes': N_boxes, 'threshold': threshold}
    log_bs = np.log(box_sizes[valid])
    log_N = np.log(N_boxes[valid])
    A = np.vstack([-log_bs, np.ones_like(log_bs)]).T
    coeffs, _, _, _ = np.linalg.lstsq(A, log_N, rcond=None)
    D_box = float(coeffs[0])
    return {'D_box': D_box, 'box_sizes': box_sizes, 'N_boxes': N_boxes, 'threshold': threshold}


def topological_susceptibility(Q_timeseries, V):
    if V <= 0:
        raise ValueError(f"V must be > 0, got {V}")
    if len(Q_timeseries) < 1:
        raise ValueError("Q_timeseries is empty")
    Q2_mean = float(np.mean(Q_timeseries ** 2))
    chi_t = Q2_mean / V
    return {
        'chi_t': chi_t, 'Q2_mean': Q2_mean,
        'Q_mean': float(np.mean(Q_timeseries)),
        'Q_std': float(np.std(Q_timeseries)),
    }
