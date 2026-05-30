
import numpy as np




C_0 = 2.99792458e8
MU_0 = 4.0 * np.pi * 1.0e-7
EPS_0 = 1.0 / (MU_0 * C_0 ** 2)
ETA_0 = np.sqrt(MU_0 / EPS_0)
H_BAR = 1.054571817e-34
EV_TO_J = 1.602176634e-19






def eps_lorentz(omega, eps_inf, omega_p, gamma, omega_0):
    if omega < 0:
        raise ValueError("角频率 ω 必须非负")
    if gamma < 0:
        raise ValueError("阻尼系数 γ 必须非负")
    return eps_inf + (omega_p ** 2) / (omega_0 ** 2 - omega ** 2 - 1j * gamma * omega)


def eps_drude(omega, eps_inf, omega_p, gamma):
    if omega <= 1e-12:
        return complex(-1e18, 0)
    if gamma < 0:
        raise ValueError("阻尼系数 γ 必须非负")
    return eps_inf - (omega_p ** 2) / (omega ** 2 + 1j * gamma * omega)


def eps_sellmeier(wavelength, B_coeffs, C_coeffs):
    if wavelength <= 0:
        raise ValueError("波长必须为正")
    if len(B_coeffs) != len(C_coeffs):
        raise ValueError("B_coeffs 与 C_coeffs 长度必须一致")
    n2 = 1.0
    for B, C in zip(B_coeffs, C_coeffs):
        if wavelength ** 2 <= C:

            n2 += B * wavelength ** 2 / max(wavelength ** 2 - C, 1e-12)
        else:
            n2 += B * wavelength ** 2 / (wavelength ** 2 - C)
    return n2






def reciprocal_lattice_2d(a1, a2):
    a1 = np.asarray(a1, dtype=float)
    a2 = np.asarray(a2, dtype=float)
    if a1.shape != (2,) or a2.shape != (2,):
        raise ValueError("a1, a2 必须为二维向量")
    
    area = a1[0] * a2[1] - a1[1] * a2[0]
    if abs(area) < 1e-18:
        raise ValueError("晶格基矢共线，无法构成二维晶格")
    
    b1 = 2.0 * np.pi * np.array([a2[1], -a2[0]]) / area
    b2 = 2.0 * np.pi * np.array([-a1[1], a1[0]]) / area
    return b1, b2


def brillouin_zone_path_2d(b1, b2, num_points, lattice_type='square'):
    if num_points < 2:
        raise ValueError("采样点数至少为 2")
    
    if lattice_type == 'square':
        Gamma = np.array([0.0, 0.0])
        X = 0.5 * b1
        M = 0.5 * (b1 + b2)
        segments = [
            (Gamma, X, 'Γ→X'),
            (X, M, 'X→M'),
            (M, Gamma, 'M→Γ')
        ]
    elif lattice_type == 'triangular':
        Gamma = np.array([0.0, 0.0])

        K = (2.0 / 3.0) * b1 + (1.0 / 3.0) * b2
        M = 0.5 * b1
        segments = [
            (Gamma, K, 'Γ→K'),
            (K, M, 'K→M'),
            (M, Gamma, 'M→Γ')
        ]
    else:
        raise ValueError(f"不支持的晶格类型: {lattice_type}")
    
    k_points = []
    labels = []
    for start, end, label in segments:
        for t in np.linspace(0, 1, num_points):
            k_points.append((1 - t) * start + t * end)
        labels.append(label)
    
    return np.array(k_points), labels






def helmholtz_operator_2d_te(psi, eps_r, kx, ky, dx, dy):
    nx, ny = psi.shape
    if eps_r.shape != (nx, ny):
        raise ValueError("eps_r 与 psi 形状必须一致")
    if dx <= 0 or dy <= 0:
        raise ValueError("网格间距必须为正")
    
    result = np.zeros_like(psi, dtype=complex)
    

    eps_avg_x = np.zeros_like(eps_r)
    eps_avg_y = np.zeros_like(eps_r)
    
    eps_avg_x[1:-1, :] = 0.5 * (eps_r[1:-1, :] + eps_r[:-2, :])
    eps_avg_x[0, :] = eps_r[0, :]
    eps_avg_x[-1, :] = eps_r[-1, :]
    
    eps_avg_y[:, 1:-1] = 0.5 * (eps_r[:, 1:-1] + eps_r[:, :-2])
    eps_avg_y[:, 0] = eps_r[:, 0]
    eps_avg_y[:, -1] = eps_r[:, -1]
    

    for i in range(nx):
        ip1 = (i + 1) % nx
        im1 = (i - 1) % nx
        phase_x_p = np.exp(1j * kx * dx) if i == nx - 1 else 1.0
        phase_x_m = np.exp(-1j * kx * dx) if i == 0 else 1.0
        
        for j in range(ny):
            jp1 = (j + 1) % ny
            jm1 = (j - 1) % ny
            phase_y_p = np.exp(1j * ky * dy) if j == ny - 1 else 1.0
            phase_y_m = np.exp(-1j * ky * dy) if j == 0 else 1.0
            

            d2x = (psi[ip1, j] * phase_x_p - 2 * psi[i, j] + psi[im1, j] * phase_x_m) / dx ** 2

            d2y = (psi[i, jp1] * phase_y_p - 2 * psi[i, j] + psi[i, jm1] * phase_y_m) / dy ** 2
            

            inv_eps_x = 1.0 / max(eps_avg_x[i, j], 1e-12)
            inv_eps_y = 1.0 / max(eps_avg_y[i, j], 1e-12)
            
            result[i, j] = -(inv_eps_x * d2x + inv_eps_y * d2y)
    
    return result


def normalized_frequency(a, omega):
    if a <= 0:
        raise ValueError("晶格常数必须为正")
    if omega < 0:
        raise ValueError("角频率必须非负")
    return omega * a / (2.0 * np.pi * C_0)


def bandgap_ratio(omega_lower, omega_upper):
    if omega_lower <= 0 or omega_upper <= omega_lower:
        return 0.0
    return 2.0 * (omega_upper - omega_lower) / (omega_upper + omega_lower)






def coupled_mode_equations(z, A, kappa, delta_beta, alpha=0.0):
    A = np.asarray(A, dtype=complex)
    if A.shape != (2,):
        raise ValueError("A 必须为长度 2 的向量 [A+, A-]")
    
    dAp = 1j * delta_beta * A[0] + 1j * kappa * A[1] - alpha * A[0]
    dAm = -1j * np.conj(kappa) * A[0] - 1j * delta_beta * A[1] - alpha * A[1]
    return np.array([dAp, dAm], dtype=complex)


def bragg_reflectivity(kappa, L, delta_beta):















    raise NotImplementedError("Hole 1: bragg_reflectivity needs to be implemented.")






def cavity_q_factor(omega_res, delta_omega):
    if omega_res <= 0:
        raise ValueError("共振频率必须为正")
    if delta_omega <= 0:
        return float('inf')
    return omega_res / delta_omega


def local_density_of_states_3d(omega, n, V_eff):
    if omega < 0 or n < 0 or V_eff < 0:
        raise ValueError("参数必须非负")
    return V_eff * omega ** 2 * n ** 3 / (np.pi ** 2 * C_0 ** 3)
