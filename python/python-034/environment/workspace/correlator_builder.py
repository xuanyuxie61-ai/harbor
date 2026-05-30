
import numpy as np
from lattice_gauge import Lattice
from fermion_solver import solve_propagator_cg, point_source


def lagrange_basis_value(npol: int, ipol: int, xpol: np.ndarray,
                         xval: float) -> float:
    if not (0 <= ipol < npol):
        raise ValueError("ipol out of range")
    pval = 1.0
    for j in range(npol):
        if j != ipol:
            denom = xpol[ipol] - xpol[j]
            if abs(denom) < 1e-14:
                raise ValueError("Duplicate nodes in Lagrange interpolation")
            pval *= (xval - xpol[j]) / denom
    return pval


def lagrange_interpolate(xpol: np.ndarray, ypol: np.ndarray,
                         xval: float) -> float:
    npol = len(xpol)
    result = 0.0
    for i in range(npol):
        result += ypol[i] * lagrange_basis_value(npol, i, xpol, xval)
    return result


def sech2_soliton(x: np.ndarray, t: float, v: float = 1.0,
                  a: float = 0.0) -> np.ndarray:
    arg = 0.5 * np.sqrt(abs(v)) * (x - v * t - a)

    arg = np.clip(arg, -50.0, 50.0)
    return -0.5 * abs(v) / np.cosh(arg) ** 2


def meson_correlator_pion(lat: Lattice, propagators: list,
                          source_positions: list) -> np.ndarray:

    raise NotImplementedError("Hole 3: implement pion meson correlator construction")


def baryon_correlator_nucleon(lat: Lattice, propagators: list,
                              soliton_enhance: bool = True) -> np.ndarray:
    nt = lat.dims[3]
    corr = np.zeros(nt, dtype=complex)


    if soliton_enhance:
        nx = lat.dims[0]
        x_coords = np.arange(nx)
        soliton_weights = sech2_soliton(x_coords, t=0.0, v=0.8)

        soliton_weights = np.abs(soliton_weights)
        sw_sum = np.sum(soliton_weights)
        if sw_sum > 1e-10:
            soliton_weights /= sw_sum
    else:
        soliton_weights = np.ones(lat.dims[0]) / lat.dims[0]

    for prop in propagators:
        for t in range(nt):
            slice_sum = 0.0
            for idx in range(lat.vol):
                x = lat.index_to_site(idx)
                if x[3] != t:
                    continue
                psi = prop[(x[0], x[1], x[2], x[3])]

                weight = soliton_weights[x[0]]

                slice_sum += weight * np.vdot(psi, psi)
            corr[t] += slice_sum

    corr /= len(propagators)
    return corr.real


def correlator_effective_mass(corr: np.ndarray, dt: int = 1) -> np.ndarray:
    nt = len(corr)
    m_eff = np.zeros(nt - dt)
    for t in range(nt - dt):
        if abs(corr[t + dt]) > 1e-15 and corr[t] / corr[t + dt] > 0:
            m_eff[t] = np.log(corr[t] / corr[t + dt]) / dt
        else:
            m_eff[t] = np.nan
    return m_eff


def correlator_interpolated_mass(corr: np.ndarray, tpol: np.ndarray,
                                 tval: float) -> float:
    c_interp = lagrange_interpolate(tpol, corr[tpol.astype(int)], tval)
    if c_interp > 1e-15:
        c_next = lagrange_interpolate(tpol, corr[tpol.astype(int)], tval + 1.0)
        if c_next > 1e-15 and c_interp / c_next > 0:
            return np.log(c_interp / c_next)
    return np.nan
