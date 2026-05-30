
import numpy as np
from math import comb, factorial
from itertools import combinations_with_replacement


def legendre_polynomial(n, x):
    x = np.asarray(x, dtype=float)
    L = np.zeros((n + 1,) + x.shape)
    L[0] = 1.0 / np.sqrt(2.0)
    if n >= 1:
        L[1] = np.sqrt(3.0 / 2.0) * x
    for k in range(1, n):
        c1 = (2.0 * k + 1.0) / (k + 1.0)
        c2 = float(k) / (k + 1.0)

        norm_k = np.sqrt((2.0 * k + 1.0) / 2.0)
        norm_kp1 = np.sqrt((2.0 * (k + 1) + 1.0) / 2.0)
        norm_km1 = np.sqrt((2.0 * (k - 1) + 1.0) / 2.0)



        ak = np.sqrt((2.0 * k + 1.0) * (2.0 * k + 3.0)) / (k + 1.0)
        bk = - np.sqrt((2.0 * k + 3.0) / (2.0 * k - 1.0)) * k / (k + 1.0)
        L[k + 1] = ak * x * L[k] + bk * L[k - 1]
    return L


def legendre_linear_product(p, e=0):
    order = p + 1 + (e + 1) // 2
    xq, wq = np.polynomial.legendre.leggauss(order)
    table = np.zeros((p + 1, p + 1))
    L_all = legendre_polynomial(p, xq)
    for k in range(order):
        if e == 0:
            contrib = wq[k] * np.outer(L_all[:, k], L_all[:, k])
        else:
            contrib = wq[k] * (xq[k] ** e) * np.outer(L_all[:, k], L_all[:, k])
        table += contrib

    table[np.abs(table) < 1e-14] = 0.0
    return table


def multi_index_count(n_dim, p):
    return comb(n_dim + p, n_dim)


def generate_multi_indices(n_dim, p):

    if n_dim == 1:
        return np.arange(p + 1).reshape(-1, 1)
    indices = []
    for total in range(p + 1):

        def compositions(k, n, prefix):
            if n == 1:
                yield prefix + [k]
            else:
                for i in range(k + 1):
                    yield from compositions(k - i, n - 1, prefix + [i])
        for comp in compositions(total, n_dim, []):
            indices.append(comp)
    return np.array(indices, dtype=int)


def evaluate_pce_basis(n_dim, p, xi, indices=None):
    xi = np.asarray(xi, dtype=float)
    if xi.ndim == 1:
        xi = xi.reshape(1, -1)
    if indices is None:
        indices = generate_multi_indices(n_dim, p)
    n_basis = indices.shape[0]
    n_samples = xi.shape[0]
    

    L_1d = []
    for d in range(n_dim):
        Ld = legendre_polynomial(p, xi[:, d])
        L_1d.append(Ld)
    
    psi = np.ones((n_samples, n_basis))
    for j, alpha in enumerate(indices):
        for d in range(n_dim):
            if alpha[d] > 0:
                psi[:, j] *= L_1d[d][alpha[d], :]
    

    norms = np.ones(n_basis)
    return psi, indices, norms


def assemble_stochastic_galerkin_system(K_det, M_det, C_det, 
                                         n_dim, p, E_mean, E_std,
                                         kl_modes, kl_eigenvalues):
    n_dof = K_det.shape[0]
    indices = generate_multi_indices(n_dim, p)
    n_basis = indices.shape[0]
    

    K_sg = np.zeros((n_dof * n_basis, n_dof * n_basis))
    M_sg = np.zeros((n_dof * n_basis, n_dof * n_basis))
    C_sg = np.zeros((n_dof * n_basis, n_dof * n_basis))
    

    for i in range(n_basis):
        ii = slice(i * n_dof, (i + 1) * n_dof)
        M_sg[ii, ii] = M_det
        C_sg[ii, ii] = C_det
    




    
    E0 = E_mean

    table = legendre_linear_product(p, e=1)
    
    for k in range(n_basis):
        for l in range(n_basis):
            alpha = indices[k]
            beta = indices[l]
            

            coeff = E0 if np.array_equal(alpha, beta) else 0.0
            


            for m in range(n_dim):

                match = True
                factor = 1.0
                for d in range(n_dim):
                    if d == m:

                        if alpha[d] <= p and beta[d] <= p:
                            factor *= table[alpha[d], beta[d]]
                        else:
                            match = False
                            break
                    else:
                        if alpha[d] != beta[d]:
                            match = False
                            break

                        factor *= 1.0
                
                if match:
                    Em_coeff = E_std * np.sqrt(kl_eigenvalues[m])
                    coeff += Em_coeff * factor
            
            if abs(coeff) > 1e-14:
                kk = slice(k * n_dof, (k + 1) * n_dof)
                ll = slice(l * n_dof, (l + 1) * n_dof)
                K_sg[kk, ll] += coeff * K_det
    
    return K_sg, M_sg, C_sg, indices


def pce_moments(coefficients, indices, norms=None):
    coefficients = np.asarray(coefficients)
    if norms is None:
        norms = np.ones(indices.shape[0])
    
    if coefficients.ndim == 1:
        mean = coefficients[0]
        variance = np.sum(coefficients[1:] ** 2 * norms[1:])
    else:
        mean = coefficients[:, 0]
        variance = np.sum(coefficients[:, 1:] ** 2 * norms[1:], axis=1)
    return mean, variance


def kl_expansion_1d(n_modes, length, correlation_length, x_coords):
    x = np.asarray(x_coords)
    Lc = correlation_length
    L = length
    
    eigenvalues = np.zeros(n_modes)
    modes = np.zeros((n_modes, len(x)))
    

    for n in range(n_modes):
        if n % 2 == 0:

            w = (n + 1) * np.pi / L
            eigenvalues[n] = 2.0 * Lc / (1.0 + (w * Lc) ** 2)
            phi = np.cos(w * x)
        else:

            w = (n + 1) * np.pi / L
            eigenvalues[n] = 2.0 * Lc / (1.0 + (w * Lc) ** 2)
            phi = np.sin(w * x)

        norm = np.sqrt(np.trapezoid(phi ** 2, x))
        if norm > 1e-12:
            phi = phi / norm
        modes[n] = phi
    
    return eigenvalues, modes
