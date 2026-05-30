# -*- coding: utf-8 -*-

import numpy as np



PHYSICAL_CONSTANTS = {
    'epsilon_0': 8.854187817e-12,
    'mu_0': 4.0e-7 * np.pi,
    'e_charge': 1.602176634e-19,
    'm_e': 9.1093837015e-31,
    'm_p': 1.67262192369e-27,
    'k_B': 1.380649e-23,
    'c': 299792458.0,
    'h_planck': 6.62607015e-34,
}


def safe_exp(x, max_val=50.0, min_val=-50.0):
    x_clipped = np.clip(x, min_val, max_val)
    return np.exp(x_clipped)


def safe_divide(a, b, default=0.0):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        result = np.divide(a, b, out=np.full_like(np.asarray(a), default, dtype=float), where=np.abs(b) > 1.0e-30)
        return result
    else:
        if abs(b) < 1.0e-30:
            return default
        return a / b


def compute_coulomb_logarithm(T_e, n_e):
    if T_e <= 0 or n_e <= 0:
        return 15.0

    if T_e > 10.0:
        ln_L = 23.4 - 1.15 * np.log10(n_e) + 3.45 * np.log10(T_e)
    else:
        ln_L = 23.0 - 1.15 * np.log10(n_e) + 3.45 * np.log10(T_e)


    if ln_L < 5.0:
        ln_L = 5.0
    if ln_L > 25.0:
        ln_L = 25.0

    return ln_L


def compute_ion_gyroradius(T_i, B, m_i_amu, Z_i=1):
    e_c = PHYSICAL_CONSTANTS['e_charge']
    m_p = PHYSICAL_CONSTANTS['m_p']
    mi_kg = m_i_amu * m_p

    if B <= 0 or T_i <= 0 or Z_i <= 0:
        return np.nan

    rho_i = np.sqrt(2.0 * mi_kg * T_i * e_c) / (Z_i * e_c * B)
    return rho_i


def compute_magnetic_mirror_force(mu, B_grad):
    return -mu * B_grad


def write_data_file(filename, data, header=None):
    with open(filename, 'w') as f:
        if header is not None:
            f.write(f"# {header}\n")
        np.savetxt(f, data, fmt='%.6e')


def read_data_file(filename, skip_comments=True):
    if skip_comments:
        return np.loadtxt(filename, comments='#')
    else:
        return np.loadtxt(filename)


def print_matrix_summary(mat, name="Matrix", max_display=5):
    print(f"{name}: shape={mat.shape}, min={np.min(mat):.3e}, max={np.max(mat):.3e}, mean={np.mean(mat):.3e}")
    if mat.ndim == 2 and mat.shape[0] <= max_display and mat.shape[1] <= max_display:
        print(mat)


def check_bohm_criterion(v_i, c_s, tolerance=0.01):
    if c_s <= 0:
        return False, 0.0
    M = v_i / c_s
    return M >= (1.0 - tolerance), M


def compute_sheath_heat_flux(n_se, T_e, T_i, gamma=7.0, Z_i=1):



    raise NotImplementedError("Hole_3: 请实现 compute_sheath_heat_flux 函数")



def convergence_diagnostics(residual_history, window=10):
    if len(residual_history) < window:
        return False, 0.0, 0.0

    recent = residual_history[-window:]
    rate = np.mean(np.diff(np.log(recent + 1.0e-30)))


    diffs = np.diff(recent)
    sign_changes = np.sum(diffs[:-1] * diffs[1:] < 0)
    oscillation = sign_changes / max(len(diffs) - 1, 1)

    converged = recent[-1] < 1.0e-6 and oscillation < 0.5

    return converged, rate, oscillation


if __name__ == "__main__":
    print("utils.py 测试:")
    print(f"  Coulomb对数 = {compute_coulomb_logarithm(50.0, 1.0e19):.2f}")
    print(f"  离子拉莫尔半径 = {compute_ion_gyroradius(50.0, 5.3, 2.0):.3e} m")
    print(f"  鞘层热流 = {compute_sheath_heat_flux(5.0e18, 50.0, 50.0):.3e} W/m^2")
