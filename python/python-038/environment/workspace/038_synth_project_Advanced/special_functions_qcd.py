
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import gammaln, digamma, polygamma, zeta as riemann_zeta


NC = 3.0
TF = 0.5
CA = NC
CF = (NC**2 - 1.0) / (2.0 * NC)


N_F = 5


BETA0 = (11.0 * CA - 4.0 * TF * N_F) / 3.0


LAMBDA_QCD = 0.2


def alpha_s_1loop(Q2, nf=N_F, lambda_qcd=LAMBDA_QCD):
    Q2 = np.asarray(Q2, dtype=float)
    beta0 = (11.0 * CA - 4.0 * TF * nf) / 3.0
    if beta0 <= 0:
        raise ValueError("beta0 must be positive; check nf.")
    

    min_Q2 = (1.1 * lambda_qcd) ** 2
    Q2_safe = np.where(Q2 < min_Q2, min_Q2, Q2)
    
    log_term = np.log(Q2_safe / (lambda_qcd ** 2))

    log_term = np.where(log_term < 1e-6, 1e-6, log_term)
    
    return 4.0 * np.pi / (beta0 * log_term)


def alpha_s_2loop(Q2, nf=N_F, lambda_qcd=LAMBDA_QCD):
    Q2 = np.asarray(Q2, dtype=float)
    beta0 = (11.0 * CA - 4.0 * TF * nf) / 3.0
    beta1 = (34.0 / 3.0) * CA**2 - (20.0 / 3.0) * CA * TF * nf - 4.0 * CF * TF * nf
    
    min_Q2 = (1.1 * lambda_qcd) ** 2
    Q2_safe = np.where(Q2 < min_Q2, min_Q2, Q2)
    L = np.log(Q2_safe / (lambda_qcd ** 2))
    L = np.where(L < 1e-6, 1e-6, L)
    
    a1 = 1.0 / (beta0 * L)
    a2 = -beta1 * np.log(L) / (beta0**3 * L**2)
    return np.pi * (a1 + a2)


def p_qq_lo(z, eps=1e-10):
    z = np.asarray(z, dtype=float)

    z = np.clip(z, eps, 1.0 - eps)
    
    regular = (1.0 + z**2) / (1.0 - z)


    return CF * regular


def p_qg_lo(z, eps=1e-10):
    z = np.asarray(z, dtype=float)
    z = np.clip(z, eps, 1.0 - eps)
    return TF * (z**2 + (1.0 - z)**2)


def p_gq_lo(z, eps=1e-10):
    z = np.asarray(z, dtype=float)
    z = np.clip(z, eps, 1.0 - eps)
    return CF * (1.0 + (1.0 - z)**2) / z


def p_gg_lo(z, eps=1e-10):
    z = np.asarray(z, dtype=float)
    z = np.clip(z, eps, 1.0 - eps)
    
    regular = z / (1.0 - z) + (1.0 - z) / z + z * (1.0 - z)
    return 2.0 * CA * regular


def sudakov_quark(Q2, q2, zmin=0.01, zmax=0.99, nf=N_F):
    if q2 >= Q2 or Q2 <= 0 or q2 <= 0:
        return 1.0
    

    n_k = 80
    n_z = 80
    

    lk = np.linspace(np.log(q2), np.log(Q2), n_k)
    dk = lk[1] - lk[0]
    

    z_nodes, z_weights = leggauss(n_z)

    z = 0.5 * (zmax - zmin) * z_nodes + 0.5 * (zmax + zmin)
    wz = 0.5 * (zmax - zmin) * z_weights
    
    integral = 0.0
    for i, lki in enumerate(lk):
        ki2 = np.exp(lki)

        a_s = alpha_s_1loop(ki2, nf) / (2.0 * np.pi)
        

        p_sum = p_qq_lo(z) + p_gq_lo(z)
        integrand_z = np.sum(wz * p_sum)
        
        integral += a_s * integrand_z * dk
    
    return np.exp(-integral)


def legendre_poly_vals(n, x):
    x = np.asarray(x, dtype=float)
    if n < 0:
        return np.empty((x.size, 0))
    
    vals = np.zeros((x.size, n + 1))
    vals[:, 0] = 1.0
    if n >= 1:
        vals[:, 1] = x
    
    for k in range(1, n):
        vals[:, k + 1] = ((2.0 * k + 1.0) * x * vals[:, k] - k * vals[:, k - 1]) / (k + 1.0)
    
    return vals


def chebyshev_poly_vals(n, x):
    x = np.asarray(x, dtype=float)
    if n < 0:
        return np.empty((x.size, 0))
    
    vals = np.zeros((x.size, n + 1))
    vals[:, 0] = 1.0
    if n >= 1:
        vals[:, 1] = x
    
    for k in range(1, n):
        vals[:, k + 1] = 2.0 * x * vals[:, k] - vals[:, k - 1]
    
    return vals


def hermite_poly_vals(n, x):
    x = np.asarray(x, dtype=float)
    if n < 0:
        return np.empty((x.size, 0))
    
    vals = np.zeros((x.size, n + 1))
    vals[:, 0] = 1.0
    if n >= 1:
        vals[:, 1] = 2.0 * x
    
    for k in range(1, n):
        vals[:, k + 1] = 2.0 * x * vals[:, k] - 2.0 * k * vals[:, k - 1]
    
    return vals


def harmonic_sum(n, m=1):
    if n <= 0:
        return 0.0
    if m == 1:
        return digamma(n + 1.0) + np.euler_gamma
    else:
        return float(np.sum(1.0 / np.arange(1, n + 1, dtype=float) ** m))


def di_log(x):
    x = np.asarray(x, dtype=float)
    x = np.clip(x, -1.0, 1.0)
    


    result = np.zeros_like(x, dtype=float)
    for k in range(1, 50):
        result += x**k / (k * k)
    return result


def anomalous_dim_gamma_0(nf=N_F):
    return 4.0 * CF


def anomalous_dim_gamma_1(nf=N_F):
    return 4.0 * CF * ((67.0 / 9.0 - np.pi**2 / 3.0) * CA - (20.0 / 9.0) * TF * nf)


def validate_special_functions():
    max_error = 0.0
    

    leg = legendre_poly_vals(2, np.array([0.5]))
    val = leg[0, 2]
    exact = -0.125
    err = abs(val - exact)
    max_error = max(max_error, err)
    

    cheb = chebyshev_poly_vals(3, np.array([0.5]))
    val = cheb[0, 3]
    exact = -1.0
    err = abs(val - exact)
    max_error = max(max_error, err)
    

    herm = hermite_poly_vals(2, np.array([1.0]))
    val = herm[0, 2]
    exact = 2.0
    err = abs(val - exact)
    max_error = max(max_error, err)
    

    a_s = alpha_s_1loop(1e8)
    if not (0.0 < a_s < 0.5):
        raise RuntimeError(f"alpha_s unphysical: {a_s}")
    


    s1 = sudakov_quark(100.0, 1.0)
    s2 = sudakov_quark(100.0, 10.0)
    if s1 > s2:
        raise RuntimeError("Sudakov factor not monotonic: larger q2 should give larger survival prob")
    
    return max_error


if __name__ == "__main__":
    err = validate_special_functions()
    print(f"Special function validation max error: {err:.2e}")
