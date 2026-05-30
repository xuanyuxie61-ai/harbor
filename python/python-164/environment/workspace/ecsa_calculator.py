
import numpy as np


def power_method_eigenvalue(A, y0=None, it_max=100, tol=1e-10):
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("A 必须是方阵")
    
    if y0 is None:
        y = np.random.random(n)
    else:
        y = np.array(y0, dtype=float)
    

    norm_y = np.linalg.norm(y)
    if norm_y < 1e-30:
        y = np.ones(n)
        norm_y = np.linalg.norm(y)
    y = y / norm_y
    
    it_num = 0
    lambda_old = 0.0
    
    ay = A @ y
    lambda_val = float(y @ ay)
    y = ay / np.linalg.norm(ay)
    if lambda_val < 0:
        y = -y
    
    for it_num in range(1, it_max + 1):
        lambda_old = lambda_val
        y_old = y.copy()
        
        ay = A @ y
        lambda_val = float(y @ ay)
        y = ay / np.linalg.norm(ay)
        if lambda_val < 0:
            y = -y
        
        val_dif = abs(lambda_val - lambda_old)
        

        cos_yy = float(y @ y_old)
        sin_yy = np.sqrt(max(0.0, (1.0 - cos_yy) * (1.0 + cos_yy)))
        
        if val_dif <= tol and sin_yy <= tol:
            break
    
    return lambda_val, y, it_num


def ecsa_from_size_distribution(radii, rho_pt=21450):
    if len(radii) == 0:
        return 0.0
    
    radii = np.array(radii)
    surface_area = np.sum(4.0 * np.pi * radii ** 2)
    volume = np.sum((4.0 / 3.0) * np.pi * radii ** 3)
    mass = volume * rho_pt
    
    if mass < 1e-30:
        return 0.0
    

    ecsa_specific = surface_area / (mass * 1000.0)
    return float(ecsa_specific)


def ecsa_loss_kinetics(ECSA, t, k1=1e-6, k2=1e-10):
    if t < 0 or ECSA <= 0:
        return 0.0
    
    ECSA0 = ECSA
    denom = (k1 + k2 * ECSA0) * np.exp(k1 * t) - k2 * ECSA0
    
    if abs(denom) < 1e-30:
        return 0.0
    
    ecsa_t = k1 * ECSA0 / denom
    return float(max(ecsa_t, 0.0))


def build_stability_jacobian(n_species, rate_constants, interaction_matrix):
    if n_species <= 0:
        raise ValueError("物种数必须为正")
    
    J = np.zeros((n_species, n_species))
    
    for i in range(n_species):
        J[i, i] = -rate_constants[i] if i < len(rate_constants) else -1e-6
        for j in range(n_species):
            if i != j:
                J[i, j] = interaction_matrix[i, j] if interaction_matrix is not None else 0.0
    
    return J


def stability_analysis_max_eigenvalue(J, y0=None):
    if J.shape[0] != J.shape[1]:
        raise ValueError("雅可比矩阵必须是方阵")
    
    lambda_max, _, _ = power_method_eigenvalue(J, y0=y0, it_max=200, tol=1e-12)
    
    if lambda_max < -1e-10:
        stability = 'stable'
    elif lambda_max > 1e-10:
        stability = 'unstable'
    else:
        stability = 'critical'
    
    return lambda_max, stability


def voltage_loss_from_ecsa(ECSA_ratio, b_tafel=0.06):
    if ECSA_ratio <= 0:
        return 0.5
    
    ratio = 1.0 / ECSA_ratio
    ratio = np.clip(ratio, 1.0, 1e6)
    
    dV = b_tafel * np.log10(ratio)
    return float(dV)


def total_ecsa_loss_model(t_hours, ECSA0, params=None):
    if params is None:
        params = {
            'k_ripening': 0.05,
            'k_corrosion': 1e-4,
            'k_poisoning': 1e-5,
        }
    
    t = max(t_hours, 0.0)
    

    ripening_factor = 1.0 / (1.0 + params['k_ripening'] * (t ** (1.0 / 3.0)))
    

    corrosion_factor = np.exp(-params['k_corrosion'] * t)
    

    poisoning_factor = max(0.0, 1.0 - params['k_poisoning'] * t)
    
    ECSA_t = ECSA0 * ripening_factor * corrosion_factor * poisoning_factor
    
    return float(max(ECSA_t, 0.0))


if __name__ == "__main__":
    radii = np.array([2e-9, 3e-9, 4e-9, 5e-9])
    ecsa = ecsa_from_size_distribution(radii)
    print(f"比表面积: {ecsa:.2f} m^2/g_Pt")
    
    J = build_stability_jacobian(3, [1e-4, 2e-4, 5e-5], 
                                  np.array([[-1e-4, 1e-5, 0],
                                            [2e-5, -2e-4, 1e-5],
                                            [0, 3e-5, -5e-5]]))
    lam, stab = stability_analysis_max_eigenvalue(J)
    print(f"主导特征值: {lam:.6e}, 稳定性: {stab}")
