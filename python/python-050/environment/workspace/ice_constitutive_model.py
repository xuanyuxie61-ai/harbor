
import numpy as np


ICE_DENSITY = 917.0
GRAVITY = 9.81
SPECIFIC_HEAT = 2097.0
THERMAL_CONDUCTIVITY = 2.10
LATENT_HEAT = 3.34e5
GAS_CONSTANT = 8.314
GLEN_N = 3.0


ARRHENIUS_A0 = 3.5e-25
ACTIVATION_ENERGY_COLD = 6.0e4
ACTIVATION_ENERGY_WARM = 13.9e4
TEMP_THRESHOLD = 263.15


ANISO_MAX_ENHANCEMENT = 10.0
ANISO_SHAPE_FACTOR = 2.0


def safe_exp(x: np.ndarray, max_val: float = 700.0) -> np.ndarray:
    x_clipped = np.clip(x, -max_val, max_val)
    return np.exp(x_clipped)


def rate_factor_arrhenius(temperature: np.ndarray) -> np.ndarray:
    temperature = np.asarray(temperature, dtype=np.float64)


    if np.any(temperature < 100.0) or np.any(temperature > 300.0):
        raise ValueError("Temperature out of physical range for ice (100K ~ 300K).")


    q = np.where(temperature <= TEMP_THRESHOLD,
                 ACTIVATION_ENERGY_COLD,
                 ACTIVATION_ENERGY_WARM)





    raise NotImplementedError("Hole 1: 请实现 rate_factor_arrhenius 核心公式")


def effective_stress(deviatoric_stress: np.ndarray) -> np.ndarray:
    deviatoric_stress = np.asarray(deviatoric_stress, dtype=np.float64)

    if deviatoric_stress.shape[-2:] != (3, 3):
        raise ValueError("deviatoric_stress must have shape (..., 3, 3)")


    double_contract = np.sum(deviatoric_stress * deviatoric_stress, axis=(-2, -1))
    tau_e = np.sqrt(0.5 * np.maximum(double_contract, 0.0))


    tau_e = np.maximum(tau_e, 1e-12)
    return tau_e


def glen_flow_law(deviatoric_stress: np.ndarray,
                  temperature: np.ndarray,
                  anisotropy_factor: np.ndarray = None) -> np.ndarray:
    deviatoric_stress = np.asarray(deviatoric_stress, dtype=np.float64)
    temperature = np.asarray(temperature, dtype=np.float64)

    if anisotropy_factor is None:
        anisotropy_factor = np.ones_like(temperature)
    else:
        anisotropy_factor = np.asarray(anisotropy_factor, dtype=np.float64)


    A = rate_factor_arrhenius(temperature)
    tau_e = effective_stress(deviatoric_stress)



    tau_e_shape = tau_e.shape
    target_shape = tau_e_shape + (1, 1)

    A = np.reshape(A, A.shape + (1, 1)) if A.ndim > 0 else A
    aniso = np.reshape(anisotropy_factor, anisotropy_factor.shape + (1, 1)) if anisotropy_factor.ndim > 0 else anisotropy_factor


    prefactor = aniso * A * (tau_e ** (GLEN_N - 1.0))
    prefactor = np.reshape(prefactor, prefactor.shape + (1, 1)) if prefactor.ndim > 0 else prefactor

    strain_rate = prefactor * deviatoric_stress
    return strain_rate


def anisotropic_enhancement_factor(second_order_orientation_tensor: np.ndarray) -> np.ndarray:
    a2 = np.asarray(second_order_orientation_tensor, dtype=np.float64)

    if a2.shape[-2:] != (3, 3):
        raise ValueError("Orientation tensor must have shape (..., 3, 3)")



    orig_shape = a2.shape[:-2]
    a2_flat = a2.reshape(-1, 3, 3)


    a2_flat = 0.5 * (a2_flat + np.transpose(a2_flat, (0, 2, 1)))


    evals = np.linalg.eigvalsh(a2_flat)
    lambda_max = np.max(evals, axis=-1)


    f_a = 1.5 * lambda_max - 0.5
    f_a = np.clip(f_a, 0.0, 1.0)

    E = 1.0 + (ANISO_MAX_ENHANCEMENT - 1.0) * (f_a ** ANISO_SHAPE_FACTOR)
    E = E.reshape(orig_shape)
    return E


def dissipation_heat(strain_rate: np.ndarray,
                     deviatoric_stress: np.ndarray) -> np.ndarray:
    phi = np.sum(deviatoric_stress * strain_rate, axis=(-2, -1))
    phi = np.maximum(phi, 0.0)
    return phi


def glen_viscosity(temperature: np.ndarray,
                   effective_strain_rate: np.ndarray) -> np.ndarray:
    A = rate_factor_arrhenius(temperature)
    eps_e = np.maximum(np.asarray(effective_strain_rate, dtype=np.float64), 1e-20)

    eta = 0.5 * (A ** (-1.0 / GLEN_N)) * (eps_e ** ((1.0 - GLEN_N) / GLEN_N))

    eta = np.clip(eta, 1e10, 1e20)
    return eta


def jacobian_glen_stress(strain_rate: np.ndarray,
                         deviatoric_stress: np.ndarray,
                         temperature: np.ndarray) -> np.ndarray:
    tau = np.asarray(deviatoric_stress, dtype=np.float64)
    A = rate_factor_arrhenius(temperature)
    tau_e = effective_stress(tau)


    shape = tau.shape
    batch = shape[:-2] if len(shape) > 2 else (1,)


    I = np.eye(3, dtype=np.float64)
    delta = np.einsum('ik,jl->ijkl', I, I)


    tau_outer = np.einsum('...ij,...kl->...ijkl', tau, tau)


    prefactor = A * (tau_e ** (GLEN_N - 1.0))
    prefactor = np.reshape(prefactor, prefactor.shape + (1, 1, 1, 1)) if prefactor.ndim > 0 else prefactor

    J = prefactor * (delta + (GLEN_N - 1.0) * tau_outer / (2.0 * tau_e[:, None, None, None, None] ** 2))
    return J


def effective_strain_rate(strain_rate_tensor: np.ndarray) -> np.ndarray:
    eps = np.asarray(strain_rate_tensor, dtype=np.float64)
    val = np.sum(eps * eps, axis=(-2, -1))
    return np.sqrt(0.5 * np.maximum(val, 0.0))
