
import numpy as np
from physics_core import C_0






def photon_hopping_matrix(n_sites, hopping_prob, disorder_strength):
    if n_sites < 3:
        raise ValueError("位点数必须 >= 3")
    if not (0 <= hopping_prob <= 1):
        raise ValueError("跃迁概率必须在 [0, 1] 内")
    
    A = np.zeros((n_sites, n_sites))
    
    for i in range(n_sites):

        neighbors = [(i - 1) % n_sites, (i + 1) % n_sites]
        
        for j in neighbors:

            delta = disorder_strength * (2.0 * np.random.rand() - 1.0)
            p_ij = hopping_prob * (1.0 + delta)
            p_ij = np.clip(p_ij, 0.0, 1.0)
            A[j, i] += p_ij / 2.0
        

        A[i, i] = max(0.0, 1.0 - np.sum(A[:, i]))
    

    for j in range(n_sites):
        col_sum = np.sum(A[:, j])
        if col_sum > 1e-12:
            A[:, j] /= col_sum
    

    eigenvalues = np.linalg.eigvals(A)
    eigenvalues = np.sort(eigenvalues)[::-1]
    


    if len(eigenvalues) > 1 and abs(eigenvalues[1]) < 1.0:
        if abs(eigenvalues[1]) > 1e-12:
            localization_length = -1.0 / np.log(abs(eigenvalues[1]))
        else:
            localization_length = 0.0
    else:
        localization_length = float('inf')
    
    return A, eigenvalues, localization_length


def photon_diffusion_markov(A, initial_distribution, n_steps):
    n_sites = A.shape[0]
    P = np.asarray(initial_distribution, dtype=float)
    if len(P) != n_sites:
        raise ValueError("初始分布维度不匹配")
    
    P /= np.sum(P)
    
    distributions = np.zeros((n_steps + 1, n_sites))
    distributions[0, :] = P
    entropy = np.zeros(n_steps + 1)
    entropy[0] = -np.sum(P * np.log(P + 1e-18))
    
    for t in range(n_steps):
        P = A.dot(P)
        distributions[t + 1, :] = P
        entropy[t + 1] = -np.sum(P * np.log(P + 1e-18))
    
    return distributions, entropy






def radiative_transfer_1d(I0, sigma_scat, sigma_abs, L, nz, n_angles=8):
    if L <= 0 or nz < 3:
        raise ValueError("参数超出允许范围")
    
    dz = L / (nz - 1)
    z = np.linspace(0, L, nz)
    
    sigma_total = sigma_scat + sigma_abs
    

    mu_pos = np.linspace(0.1, 0.9, n_angles // 2)
    mu_neg = -mu_pos
    
    w = np.ones(n_angles // 2) / (n_angles // 2)
    
    I_plus = np.zeros((nz, n_angles // 2))
    I_minus = np.zeros((nz, n_angles // 2))
    

    I_plus[0, :] = I0
    I_minus[-1, :] = 0.0
    

    max_iter = 1000
    tol = 1e-10
    
    for it in range(max_iter):
        I_plus_old = I_plus.copy()
        I_minus_old = I_minus.copy()
        

        for i in range(1, nz):
            for m in range(n_angles // 2):

                source = 0.0
                for mp in range(n_angles // 2):
                    source += w[mp] * (I_plus[i - 1, mp] + I_minus[i - 1, mp])
                source *= sigma_scat / 2.0
                

                denominator = sigma_total + mu_pos[m] / dz
                if abs(denominator) < 1e-15:
                    denominator = 1e-15
                
                I_plus[i, m] = (mu_pos[m] / dz * I_plus[i - 1, m] + source) / denominator
        

        for i in range(nz - 2, -1, -1):
            for m in range(n_angles // 2):
                source = 0.0
                for mp in range(n_angles // 2):
                    source += w[mp] * (I_plus[i + 1, mp] + I_minus[i + 1, mp])
                source *= sigma_scat / 2.0
                
                denominator = sigma_total + abs(mu_neg[m]) / dz
                if abs(denominator) < 1e-15:
                    denominator = 1e-15
                
                I_minus[i, m] = (abs(mu_neg[m]) / dz * I_minus[i + 1, m] + source) / denominator
        

        diff = np.max(np.abs(I_plus - I_plus_old)) + np.max(np.abs(I_minus - I_minus_old))
        if diff < tol:
            break
    

    I_forward = np.sum(I_plus * w, axis=1)
    I_backward = np.sum(I_minus * w, axis=1)
    
    transmittance = I_forward[-1] / I0 if I0 > 1e-15 else 0.0
    reflectance = I_backward[0] / I0 if I0 > 1e-15 else 0.0
    
    transmittance = np.clip(transmittance, 0.0, 1.0)
    reflectance = np.clip(reflectance, 0.0, 1.0)
    
    return z, I_forward, I_backward, transmittance, reflectance






def anderson_localization_length(wavelength, mean_free_path, disorder_strength):
    if wavelength <= 0 or mean_free_path <= 0:
        raise ValueError("波长和平均自由程必须为正")
    
    k = 2.0 * np.pi / wavelength
    kl = k * mean_free_path
    

    is_localized = kl < 1.0
    
    if kl > 1.0:

        xi_loc = mean_free_path * np.exp(np.pi ** 2 / 2.0 * kl ** 2)
    else:

        xi_loc = mean_free_path * (1.0 + 0.5 * (1.0 - kl))
    

    xi_loc *= (1.0 - 0.3 * disorder_strength)
    
    return xi_loc, kl, is_localized


def photon_mean_free_path(eps_r, wavelength, correlation_length):
    if wavelength <= 0 or correlation_length <= 0:
        raise ValueError("参数必须为正")
    
    eps_mean = np.mean(eps_r)
    if eps_mean < 1e-12:
        eps_mean = 1.0
    
    delta_eps = (eps_r - eps_mean) / eps_mean
    delta_eps_rms = np.sqrt(np.mean(delta_eps ** 2))
    
    if delta_eps_rms < 1e-12:
        return float('inf'), 0.0
    

    scattering_cross_section = (np.pi ** 2 / wavelength ** 4) * (delta_eps_rms ** 2) * (correlation_length ** 3) * (eps_mean ** 2)
    

    n_scatterers = 1.0 / (correlation_length ** 3)
    
    l_mfp_rayleigh = 1.0 / max(n_scatterers * scattering_cross_section, 1e-30)
    


    l_mfp_heuristic = correlation_length / max(delta_eps_rms ** 2, 1e-12)
    

    l_mfp = max(min(l_mfp_rayleigh, l_mfp_heuristic), wavelength * 1e-3)
    

    l_mfp = min(l_mfp, wavelength * 1e4)
    
    return l_mfp, delta_eps_rms


def diffusion_constant_photonic(l_mfp, v_group):
    if l_mfp < 0 or v_group < 0:
        raise ValueError("参数必须非负")
    return (1.0 / 3.0) * v_group * l_mfp


def scaling_theory_beta_function(g, d=3):
    if g <= 0:
        return -float('inf')
    

    beta = (d - 2.0) + (2.0 - d) / (1.0 + g ** 2)
    return beta
