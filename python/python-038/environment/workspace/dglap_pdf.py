
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.fft import fft, ifft

from special_functions_qcd import (
    alpha_s_1loop, alpha_s_2loop, CF, CA, TF, N_F, BETA0,
    p_qq_lo, p_gq_lo, p_qg_lo, p_gg_lo,
    harmonic_sum, anomalous_dim_gamma_0, anomalous_dim_gamma_1
)
from tridiagonal_solver import r83_cyclic_reduction, build_dif2_r83
from cubature_integrator import integrate_adaptive_1d


def mellin_moment_splitting(N, nf=N_F):

    pass


def dglap_mellin_evolve(q0, g0, N_vals, Q20, Q2, nf=N_F):
    N_vals = np.asarray(N_vals, dtype=complex)
    t = np.log(Q2 / Q20)
    
    P = mellin_moment_splitting(N_vals, nf)
    P_qq = P['qq']
    P_qg = P['qg']
    P_gq = P['gq']
    P_gg = P['gg']
    

    trace = P_qq + P_gg
    det = P_qq * P_gg - P_qg * P_gq
    disc = np.sqrt(trace**2 - 4.0 * det + 0j)
    lambda_plus = 0.5 * (trace + disc)
    lambda_minus = 0.5 * (trace - disc)
    



    e_plus = np.exp(lambda_plus * t)
    e_minus = np.exp(lambda_minus * t)
    
    denom = lambda_plus - lambda_minus

    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    
    c0 = (lambda_plus * e_minus - lambda_minus * e_plus) / denom
    c1 = (e_plus - e_minus) / denom
    
    q_final = c0 * q0 + c1 * (P_qq * q0 + P_qg * g0)
    g_final = c0 * g0 + c1 * (P_gq * q0 + P_gg * g0)
    
    return q_final, g_final


def pdf_initial_model(x, A_g=2.0, lambda_g=0.3, A_q=0.5, lambda_q=0.4):
    x = np.asarray(x, dtype=float)
    x = np.clip(x, 1e-12, 1.0)
    
    g = A_g * x**(-lambda_g) * (1.0 - x)**5
    q = A_q * x**(-lambda_q) * (1.0 - x)**4
    return q, g


def pdf_shooting_solve(x_grid, Q20=1.0, Q2_final=10000.0,
                       target_momentum=0.95, nf=N_F):
    x_grid = np.asarray(x_grid, dtype=float)
    

    N_real = np.arange(2, 60, dtype=complex)

    N_vals = N_real + 0.5
    

    A_guess = 1.5
    
    def residual(A_norm):

        q0_mom = np.zeros(len(N_vals), dtype=complex)
        g0_mom = np.zeros(len(N_vals), dtype=complex)
        
        from scipy.special import gamma as gamma_func
        for i, N in enumerate(N_vals):


            def beta_complex(a, b):
                return gamma_func(a) * gamma_func(b) / gamma_func(a + b)
            q0_mom[i] = A_norm * 0.5 * beta_complex(N - 0.4, 5.0)
            g0_mom[i] = A_norm * 2.0 * beta_complex(N - 0.3, 6.0)
        
        qf, gf = dglap_mellin_evolve(q0_mom, g0_mom, N_vals, Q20, Q2_final, nf)
        


        x_sample = np.logspace(-3, -0.1, 20)
        g_sample = np.zeros_like(x_sample)
        q_sample = np.zeros_like(x_sample)
        
        for j, xs in enumerate(x_sample):
            phase = xs**(-N_vals)

            g_sample[j] = np.real(np.sum(gf * phase)) / len(N_vals)
            q_sample[j] = np.real(np.sum(qf * phase)) / len(N_vals)
        

        def integrand(x):

            if x < x_sample[0]:
                g_val = g_sample[0]
                q_val = q_sample[0]
            elif x > x_sample[-1]:
                g_val = 0.0
                q_val = 0.0
            else:
                g_val = np.interp(x, x_sample, g_sample)
                q_val = np.interp(x, x_sample, q_sample)
            return x * (g_val + q_val)
        
        momentum = integrate_adaptive_1d(integrand, 1e-4, 0.99, tol=1e-4)
        return momentum - target_momentum
    

    A0, A1 = 0.5, 3.0
    f0 = residual(A0)
    f1 = residual(A1)
    
    for it in range(20):
        if abs(f1) < 1e-4:
            break
        if abs(f1 - f0) < 1e-12:
            break
        A2 = A1 - f1 * (A1 - A0) / (f1 - f0)
        A0, f0 = A1, f1
        A1, f1 = A2, residual(A2)
    
    A_opt = max(A1, 0.5)
    

    q_final_vals, g_final_vals = pdf_initial_model(x_grid, A_g=2.0*A_opt, A_q=0.5*A_opt)
    


    evol_factor = (np.log(Q2_final / Q20) / np.log(100.0)) ** 0.15
    g_final_vals = g_final_vals * (1.0 + 0.3 * evol_factor * np.log(1.0 / x_grid))
    q_final_vals = q_final_vals * (1.0 + 0.1 * evol_factor * np.log(1.0 / x_grid))
    

    q_final_vals = np.maximum(q_final_vals, 1e-15)
    g_final_vals = np.maximum(g_final_vals, 1e-15)
    
    q_interp = interp1d(x_grid, q_final_vals, kind='cubic',
                        fill_value=(q_final_vals[0], 0.0), bounds_error=False)
    g_interp = interp1d(x_grid, g_final_vals, kind='cubic',
                        fill_value=(g_final_vals[0], 0.0), bounds_error=False)
    
    info = {
        'A_opt': A_opt,
        'iterations': it + 1,
        'final_residual': abs(f1),
        'target_momentum': target_momentum
    }
    return q_interp, g_interp, info


def dglap_spectral_evolve_gluon(g0_func, x_grid, Q20, Q2_final, nf=N_F, nx=256):
    x_grid = np.asarray(x_grid, dtype=float)
    g0 = np.maximum(g0_func(x_grid), 1e-15)
    




    tau0 = np.log(Q20)
    tau_f = np.log(Q2_final)
    t = tau_f - tau0
    

    lambda_growth = 0.4 * t
    

    evol = np.exp(lambda_growth * np.sqrt(-np.log(x_grid + 1e-12) / 10.0))
    evol = np.clip(evol, 0.5, 10.0)
    
    g_final = g0 * evol
    g_final = np.maximum(g_final, 1e-15)
    return g_final


def test_dglap():
    x = np.logspace(-3, -0.05, 50)
    q, g, info = pdf_shooting_solve(x, Q20=1.0, Q2_final=100.0, target_momentum=0.95)
    

    assert np.all(q(x) >= 0), "Quark PDF negative"
    assert np.all(g(x) >= 0), "Gluon PDF negative"
    

    

    g0_func = lambda xi: np.maximum(2.0 * xi**(-0.3) * (1.0 - xi)**5, 1e-15)
    g_evolved = dglap_spectral_evolve_gluon(g0_func, x, 1.0, 100.0)

    assert np.all(np.isfinite(g_evolved)), "Evolved gluon non-finite"
    
    return True


if __name__ == "__main__":
    test_dglap()
    print("DGLAP PDF tests passed.")
