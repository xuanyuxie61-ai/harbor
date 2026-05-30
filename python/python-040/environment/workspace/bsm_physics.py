#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional




ALPHA_EM = 1.0 / 137.035999084
MZ_POLE = 91.1876
GF_FERMI = 1.1663787e-5
VEV_HIGGS = 246.0


@dataclass
class ZPrimeModel:
    mass: float
    total_width: float
    gq_coupling: float
    gl_coupling: float
    gq_axial: float = 0.0
    gl_axial: float = 0.0
    chi: float = 0.0

    def __post_init__(self):
        assert self.mass > 0.0, "Z' 质量必须为正"
        assert self.total_width >= 0.0, "衰变宽度必须非负"

        assert abs(self.gq_coupling) < 4.0 * np.pi, "夸克耦合超出微扰幺正性边界"
        assert abs(self.gl_coupling) < 4.0 * np.pi, "轻子耦合超出微扰幺正性边界"


def breit_wigner_propagator(s: float, model: ZPrimeModel) -> complex:
    if s <= 0.0:
        return 0.0 + 0.0j
    m2 = model.mass ** 2


    denom = None


    if abs(denom) < 1e-12:
        denom = 1e-12 * (denom / abs(denom)) if denom != 0 else 1e-12
    return 1.0 / denom


def dilepton_cross_section(
    s: np.ndarray,
    cos_theta: np.ndarray,
    model: ZPrimeModel,
    include_sm: bool = True
) -> np.ndarray:
    s = np.atleast_1d(s)
    cos_theta = np.atleast_1d(cos_theta)
    N, M = s.size, cos_theta.size


    s_grid = s.reshape(N, 1)
    c_grid = cos_theta.reshape(1, M)


    d_vals = np.array([breit_wigner_propagator(si, model) for si in s.ravel()])
    d2 = np.abs(d_vals) ** 2
    d2 = d2.reshape(N, 1)


    gv_e = model.gl_coupling
    gv_q = model.gq_coupling
    ga_e = model.gl_axial
    ga_q = model.gq_axial






    A_zp = None
    B_zp = None


    GEV2_TO_PB = 0.389379e9


    prefactor = ALPHA_EM ** 2 / (4.0 * s_grid)
    angular = None
    dsigma_zp = None



    dsigma_zp = np.where(s_grid > 0.0, dsigma_zp, 0.0)

    if include_sm:


        qe2 = (1.0 / 3.0) ** 2
        dsigma_sm = prefactor * (1.0 + c_grid ** 2) * qe2 * GEV2_TO_PB
        dsigma_sm = np.where(s_grid > 0.0, dsigma_sm, 0.0)
        return dsigma_zp + dsigma_sm

    return dsigma_zp


def eft_contact_interaction(
    s: float,
    eta_ll: float,
    eta_rr: float,
    eta_lr: float,
    Lambda: float
) -> float:
    if Lambda <= 0.0 or s <= 0.0:
        return 0.0


    if Lambda < np.sqrt(s):

        damping = (Lambda ** 2 / s) ** 2
    else:
        damping = 1.0

    GEV2_TO_PB = 0.389379e9
    prefactor = np.pi * ALPHA_EM ** 2 / (2.0 * Lambda ** 4) * s * GEV2_TO_PB



    angular_integral = (16.0 * np.pi / 3.0) * (eta_ll ** 2 + eta_rr ** 2) \
                     + (8.0 * np.pi / 3.0) * (2.0 * eta_lr ** 2)

    return prefactor * angular_integral * damping


def decay_width_dilepton(model: ZPrimeModel) -> float:
    mzp = model.mass
    if mzp <= 0.0:
        return 0.0
    glv2 = model.gl_coupling ** 2 + model.gl_axial ** 2
    return mzp * glv2 / (12.0 * np.pi)


def decay_width_hadronic(model: ZPrimeModel) -> float:
    mzp = model.mass
    if mzp <= 0.0:
        return 0.0

    n_f = 6
    nc = 3
    gqv2 = model.gq_coupling ** 2 + model.gq_axial ** 2
    return n_f * nc * mzp * gqv2 / (12.0 * np.pi)


def width_consistency_check(model: ZPrimeModel) -> bool:
    gamma_ll = 3.0 * decay_width_dilepton(model)
    gamma_qq = decay_width_hadronic(model)
    gamma_nu = gamma_ll
    gamma_theory = gamma_ll + gamma_qq + gamma_nu

    if model.total_width <= 0.0:
        return True

    ratio = gamma_theory / model.total_width

    return 0.5 <= ratio <= 2.0


def scattering_amplitude_matrix(
    s_vals: np.ndarray,
    model: ZPrimeModel
) -> np.ndarray:
    s_vals = np.atleast_1d(s_vals)
    n_s = s_vals.size

    cos_theta = np.linspace(-1.0, 1.0, 9)
    m_cos = cos_theta.size

    amp = np.zeros((n_s, m_cos), dtype=complex)

    gv_e = model.gl_coupling
    gv_q = model.gq_coupling
    ga_e = model.gl_axial
    ga_q = model.gq_axial

    for i, s in enumerate(s_vals):
        if s <= 0.0:
            continue

        d = breit_wigner_propagator(s, model)
        for j, ct in enumerate(cos_theta):

            amp_vv = gv_e * gv_q * (1.0 + ct ** 2) * s * d

            amp_aa = ga_e * ga_q * 2.0 * ct * s * d

            amp[i, j] = ALPHA_EM * (amp_vv + amp_aa)

    return amp


def chi_square_signal(
    observed: np.ndarray,
    expected_bkg: np.ndarray,
    expected_sig: np.ndarray,
    uncertainties: np.ndarray
) -> float:
    residuals = observed - expected_bkg - expected_sig
    denom = uncertainties ** 2 + np.abs(expected_sig)
    denom = np.where(denom > 1e-12, denom, 1e-12)
    return float(np.sum(residuals ** 2 / denom))


def exclusion_limit_at_95cl(
    signal_yield: float,
    background_yield: float,
    luminosity: float,
    systematic_unc: float = 0.1
) -> float:
    if luminosity <= 0.0:
        return np.inf
    b = max(background_yield, 0.0)
    sigma_b = systematic_unc * b

    s_95 = 1.96 * np.sqrt(b + sigma_b ** 2)

    if signal_yield > s_95 * 1.5:
        return signal_yield / luminosity
    return s_95 / luminosity
