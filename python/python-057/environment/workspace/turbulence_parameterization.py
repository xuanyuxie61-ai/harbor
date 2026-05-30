
import numpy as np


def bartlett_sample(m, df):
    C = np.zeros((m, m))
    
    for i in range(m):

        df_chi = max(df - i, 1)
        C[i, i] = np.sqrt(np.random.chisquare(df_chi))
        

        for j in range(i + 1, m):
            C[i, j] = np.random.normal(0.0, 1.0)
    
    return C


def wishart_sample(m, df, sigma):

    try:
        R = np.linalg.cholesky(sigma).T
    except np.linalg.LinAlgError:

        sigma = sigma + 1.0e-6 * np.eye(m)
        R = np.linalg.cholesky(sigma).T
    

    C = bartlett_sample(m, df)
    AU = C.T @ C
    

    W = R.T @ AU @ R
    

    W = 0.5 * (W + W.T)
    eigvals = np.linalg.eigvalsh(W)
    if np.min(eigvals) < 1.0e-10:
        W = W + (1.0e-10 - np.min(eigvals)) * np.eye(m)
    
    return W


def sample_reynolds_stress_tensor(shear_magnitude=0.01,
                                   buoyancy_flux=1.0e-7,
                                   m=3, df=10):

    tau_11 = shear_magnitude**2
    tau_22 = shear_magnitude**2
    tau_33 = buoyancy_flux
    
    sigma = np.diag([tau_11, tau_22, tau_33])
    
    W = wishart_sample(m, df, sigma)
    

    rho0 = 1025.0
    tau = -rho0 * W / df
    
    return tau


def mixing_efficiency_fixed_point(Ri, gamma_max=0.2, alpha=5.0,
                                   max_iter=100, tol=1.0e-8):
    Ri = np.asarray(Ri)
    

    gamma = gamma_max * 0.5
    history = [gamma]
    
    converged = False
    
    for _ in range(max_iter):
        gamma_new = gamma_max / (1.0 + alpha * Ri * gamma)
        

        gamma_new = np.clip(gamma_new, 0.0, gamma_max)
        
        history.append(gamma_new)
        
        if np.abs(gamma_new - gamma) < tol:
            converged = True
            break
        
        gamma = gamma_new
    
    return gamma, np.array(history), converged


def cobweb_iteration_analysis(Ri_values, gamma_max=0.2, alpha=5.0):
    results = {}
    
    for Ri in Ri_values:
        gamma, history, converged = mixing_efficiency_fixed_point(
            Ri, gamma_max, alpha
        )
        results[Ri] = {
            'gamma': gamma,
            'history': history,
            'converged': converged,
            'n_iter': len(history)
        }
    
    return results


def monomial_symmetrize_2d(coefficients, n_kx=4, n_kz=4):
    coeffs = np.asarray(coefficients)
    sym_coeffs = coeffs.copy()
    


    
    min_dim = min(coeffs.shape[0], coeffs.shape[1])
    
    for i in range(min_dim):
        for j in range(i + 1, min_dim):
            avg = 0.5 * (coeffs[i, j] + coeffs[j, i])
            sym_coeffs[i, j] = avg
            sym_coeffs[j, i] = avg
    
    return sym_coeffs


def symmetrize_wave_spectrum(E_kx_kz):
    E = np.asarray(E_kx_kz)
    E_sym = E.copy()
    
    nx, nz = E.shape
    

    for i in range(nx):
        for j in range(nz):

            i_mirror = (nx - 1 - i) % nx
            j_mirror = (nz - 1 - j) % nz
            
            sym_val = 0.25 * (E[i, j] + E[i_mirror, j] +
                              E[i, j_mirror] + E[i_mirror, j_mirror])
            
            E_sym[i, j] = sym_val
            E_sym[i_mirror, j] = sym_val
            E_sym[i, j_mirror] = sym_val
            E_sym[i_mirror, j_mirror] = sym_val
    
    return E_sym
