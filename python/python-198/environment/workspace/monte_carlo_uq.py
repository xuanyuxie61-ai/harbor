
import numpy as np


def uniform_in_unit_ball(m, n):

    z = np.random.randn(m, n)
    norms = np.linalg.norm(z, axis=0)
    norms = np.clip(norms, 1e-15, None)
    directions = z / norms
    

    r = np.random.uniform(0.0, 1.0, n) ** (1.0 / m)
    points = directions * r
    return points


def ellipse_sample(n, A_mat, R):
    A_mat = np.asarray(A_mat, dtype=float)
    m = 2
    

    try:
        U = np.linalg.cholesky(A_mat).T
    except np.linalg.LinAlgError:

        eigvals, eigvecs = np.linalg.eigh(A_mat)
        eigvals = np.clip(eigvals, 1e-8, None)
        A_mat = eigvecs @ np.diag(eigvals) @ eigvecs.T
        U = np.linalg.cholesky(A_mat).T
    

    Y = uniform_in_unit_ball(m, n) * R
    

    X = np.linalg.solve(U, Y)
    return X


def monte_carlo_pce_verify(n_samples, pce_degree, alpha_mu, alpha_sigma,
                           u0_scalar, tf, exact_mean_func):

    xi = np.random.randn(n_samples)
    alpha = alpha_mu + alpha_sigma * xi
    

    u_exact = u0_scalar * np.exp(-alpha * tf)
    
    mc_mean = np.mean(u_exact)
    mc_var = np.var(u_exact, ddof=1)
    

    pce_mean_analytical = u0_scalar * np.exp(-alpha_mu * tf + 0.5 * alpha_sigma ** 2 * tf ** 2)
    
    error_mean = abs(mc_mean - pce_mean_analytical) / (abs(pce_mean_analytical) + 1e-15)
    
    return {
        'mc_mean': mc_mean,
        'mc_var': mc_var,
        'pce_mean_analytical': pce_mean_analytical,
        'error_mean': error_mean,
        'n_samples': n_samples
    }


def disk_distance_monte_carlo(n_samples=50000):
    theta1 = np.random.uniform(0, 2 * np.pi, n_samples)
    r1 = np.sqrt(np.random.uniform(0, 1, n_samples))
    theta2 = np.random.uniform(0, 2 * np.pi, n_samples)
    r2 = np.sqrt(np.random.uniform(0, 1, n_samples))
    
    p1 = np.column_stack((r1 * np.cos(theta1), r1 * np.sin(theta1)))
    p2 = np.column_stack((r2 * np.cos(theta2), r2 * np.sin(theta2)))
    
    d = np.linalg.norm(p1 - p2, axis=1)
    return {
        'mean': float(np.mean(d)),
        'variance': float(np.var(d, ddof=1)),
        'theoretical_mean': 128.0 / (45.0 * np.pi)
    }
