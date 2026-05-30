
import numpy as np
from scipy.special import gamma as scipy_gamma
from scipy.special import digamma as scipy_psi
from scipy.special import hyp2f1 as scipy_hyp2f1


class SpecialFunctionError(Exception):
    pass


def complex_log_stable(z):
    z = np.asarray(z, dtype=complex)
    a = z.real
    b = z.imag


    is_zero = (a == 0.0) & (b == 0.0)

    e = a / 2.0
    f = b / 2.0
    small_mask = (np.abs(e) < 0.5) & (np.abs(f) < 0.5)

    c = np.empty_like(a)
    d = np.empty_like(a)


    if np.any(small_mask):
        ca = np.abs(2.0 * a[small_mask]) + np.abs(2.0 * b[small_mask])

        ca = np.where(ca == 0, np.finfo(float).tiny, ca)
        da = 8.0 * (a[small_mask] / ca) * a[small_mask] \
             + 8.0 * (b[small_mask] / ca) * b[small_mask]
        c[small_mask] = 0.5 * (np.log(ca) + np.log(da)) - np.log(np.sqrt(8.0))


    large_mask = ~small_mask
    if np.any(large_mask):
        cb = np.abs(e[large_mask] / 2.0) + np.abs(f[large_mask] / 2.0)
        db = 0.5 * (e[large_mask] / cb) * e[large_mask] \
             + 0.5 * (f[large_mask] / cb) * f[large_mask]
        c[large_mask] = 0.5 * (np.log(cb) + np.log(db)) + np.log(np.sqrt(8.0))


    d = np.arctan2(b, a)

    ln_z = c + 1j * d
    if np.any(is_zero):
        if np.isscalar(ln_z):
            ln_z = np.nan + 1j * np.nan
        else:
            ln_z = np.asarray(ln_z, dtype=complex)
            ln_z[is_zero] = np.nan + 1j * np.nan
    return ln_z


def gegenbauer_integral(expon, alpha):
    if expon < 0:
        raise SpecialFunctionError("expon 必须为非负整数")
    if alpha <= -1.0:
        raise SpecialFunctionError("alpha 必须大于 -1")

    if expon % 2 == 1:
        return 0.0


    arg1 = -alpha
    arg2 = 1.0 + expon
    arg3 = 2.0 + alpha + expon
    arg4 = -1.0

    value1 = scipy_hyp2f1(arg1, arg2, arg3, arg4)
    value = (2.0 * scipy_gamma(1.0 + expon) * scipy_gamma(1.0 + alpha)
             * value1 / scipy_gamma(2.0 + alpha + expon))
    return float(value)


def thiele_modulus_efficiency_factor(phi, shape_factor=3):
    phi = np.asarray(phi, dtype=float)
    if np.any(phi < 0):
        raise SpecialFunctionError("Thiele 模数 phi 必须非负")

    eps = np.finfo(float).eps
    phi_safe = np.where(phi < eps, eps, phi)

    if shape_factor == 3:

        eta = 3.0 / (phi_safe ** 2) * (phi_safe / np.tanh(phi_safe) - 1.0)

        eta = np.where(phi < eps, 1.0, eta)
    elif shape_factor == 2:
        from scipy.special import i0, i1
        eta = 2.0 / phi_safe * i1(phi_safe) / i0(phi_safe)
        eta = np.where(phi < eps, 1.0, eta)
    elif shape_factor == 1:
        eta = np.tanh(phi_safe) / phi_safe
        eta = np.where(phi < eps, 1.0, eta)
    else:
        raise SpecialFunctionError("shape_factor 必须为 1、2 或 3")


    eta = np.clip(eta, 0.0, 1.0)
    return eta


def knudsen_diffusivity(pore_diameter, temperature, molecular_weight):
    R = 8.314462618
    if pore_diameter <= 0 or temperature <= 0 or molecular_weight <= 0:
        raise SpecialFunctionError("孔径、温度、分子量必须为正")

    D_kn = (pore_diameter / 3.0) * np.sqrt((8.0 * R * temperature)
                                           / (np.pi * molecular_weight))
    return D_kn


def effective_diffusivity(pore_diameter, temperature, molecular_weight,
                          bulk_diffusivity, tortuosity, porosity):
    if bulk_diffusivity <= 0:
        raise SpecialFunctionError("bulk_diffusivity 必须为正")
    if tortuosity < 1.0:
        raise SpecialFunctionError("tortuosity 必须 ≥ 1")
    if not (0.0 < porosity < 1.0):
        raise SpecialFunctionError("porosity 必须在 (0, 1) 之间")

    D_kn = knudsen_diffusivity(pore_diameter, temperature, molecular_weight)
    D_eff_pore = 1.0 / (1.0 / bulk_diffusivity + 1.0 / D_kn)
    D_e = (porosity / tortuosity) * D_eff_pore
    return D_e


def arrhenius_rate(pre_exp, activation_energy, temperature):
    R = 8.314462618
    if temperature <= 0:
        raise SpecialFunctionError("温度必须为正")
    k = pre_exp * np.exp(-activation_energy / (R * temperature))
    return k
