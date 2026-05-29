"""
dynamical_analysis.py
=====================
Dynamical matrix construction and eigenanalysis synthesized from seed projects:
  - 604_jacobi_eigenvalue (Jacobi eigenvalue iteration with threshold pivoting)
  - 1206_test_eigen (test matrices with prescribed eigenstructures)

Core algorithms:
  - Jacobi eigenvalue method for real symmetric matrices
  - Dynamical (Hessian) matrix construction for Yukawa-interacting particles
  - Random symmetric test matrix generation via spectral decomposition
  - Eigensystem verification via Frobenius norm of residual
"""

import numpy as np
from linear_algebra import frobenius_norm


def jacobi_eigenvalue(A, it_max=2000, tol=1e-10):
    """
    Compute eigenvalues and eigenvectors of a real symmetric matrix
    using Rutishauser's modifications of the classical Jacobi method.
    
    Based on seed 604_jacobi_eigenvalue.
    
    Mathematical formulation:
      For each off-diagonal element A[p,q], compute rotation angle theta:
        tan(2*theta) = 2*A[p,q] / (A[q,q] - A[p,p])
      
      c = cos(theta), s = sin(theta), tau = s/(1+c)
      
      Apply Givens-like rotation to annihilate A[p,q].
      
      Iterate until max(|off-diagonal|) < tol.
    
    Returns:
      V       : matrix of eigenvectors
      d       : eigenvalues in ascending order
      it_num  : total iterations performed
      rot_num : total rotations applied
    """
    n = A.shape[0]
    V = np.eye(n, dtype=float)
    D = A.astype(float).copy()
    
    # Initial threshold
    off_diag_sq = np.sum(D**2) - np.sum(np.diag(D)**2)
    thresh = np.sqrt(off_diag_sq) / (4.0 * n) if off_diag_sq > 0 else tol
    
    it_num = 0
    rot_num = 0
    
    while it_num < it_max:
        it_num += 1
        
        # Find largest off-diagonal element
        max_val = 0.0
        p, q = 0, 1
        for i in range(n):
            for j in range(i + 1, n):
                if abs(D[i, j]) > max_val:
                    max_val = abs(D[i, j])
                    p, q = i, j
        
        if max_val < tol:
            break
        
        # Skip small elements after initial sweeps
        if it_num > 4 and abs(D[p, q]) < thresh:
            continue
        
        # Compute rotation angle
        if abs(D[q, q] - D[p, p]) < 1e-15:
            theta = np.pi / 4.0
        else:
            theta = 0.5 * np.arctan2(2.0 * D[p, q], D[q, q] - D[p, p])
        
        c = np.cos(theta)
        s = np.sin(theta)
        
        # Apply rotation to matrix D
        app = D[p, p]
        aqq = D[q, q]
        apq = D[p, q]
        
        D[p, p] = c*c*app - 2.0*c*s*apq + s*s*aqq
        D[q, q] = s*s*app + 2.0*c*s*apq + c*c*aqq
        D[p, q] = 0.0
        D[q, p] = 0.0
        
        for j in range(n):
            if j != p and j != q:
                arp = D[j, p]
                arq = D[j, q]
                D[j, p] = c*arp - s*arq
                D[p, j] = D[j, p]
                D[j, q] = s*arp + c*arq
                D[q, j] = D[j, q]
        
        # Update eigenvector matrix
        for j in range(n):
            vjp = V[j, p]
            vjq = V[j, q]
            V[j, p] = c*vjp - s*vjq
            V[j, q] = s*vjp + c*vjq
        
        rot_num += 1
        
        # Update threshold
        if it_num > 4:
            off_diag_sq = np.sum(D**2) - np.sum(np.diag(D)**2)
            thresh = np.sqrt(off_diag_sq) / (4.0 * n) if off_diag_sq > 0 else tol
            if thresh < tol:
                break
    
    d = np.diag(D)
    idx = np.argsort(d)
    d = d[idx]
    V = V[:, idx]
    return V, d, it_num, rot_num


def generate_test_symmetric_matrix(n, mean=0.0, std=1.0):
    """
    Generate a random symmetric matrix with prescribed eigenvalue distribution.
    
    Based on seed 1206_test_eigen (r8symm_gen).
    
    Mathematical construction:
      A = Q * Lambda * Q^T
    where:
      Lambda = diag(lambda_1, ..., lambda_n) with lambda_i ~ N(mean, std)
      Q is a random orthogonal matrix generated via QR decomposition of a
        Gaussian random matrix.
    
    Returns:
      A, lambdas_true, Q
    """
    lambdas = mean + std * np.random.randn(n)
    Lambda = np.diag(lambdas)
    X = np.random.randn(n, n)
    Q, _ = np.linalg.qr(X)
    A = Q @ Lambda @ Q.T
    return A, lambdas, Q


def verify_eigensystem(A, V, d):
    """
    Verify computed eigensystem by computing the Frobenius norm of residual:
      residual = ||A*V - V*diag(d)||_F
    
    Based on seed 1206_test_eigen (r8mat_is_eigen_right).
    """
    n = len(d)
    Lambda = np.diag(d)
    return frobenius_norm(A @ V - V @ Lambda)


def compute_dynamical_matrix(positions, Q_eff, lambda_D, m_d):
    """
    Compute the dynamical (Hessian) matrix for a system of Yukawa-interacting
    dust particles at given equilibrium positions.
    
    The dynamical matrix elements are:
      D_{i*dim+a, j*dim+b} = (1/m_d) * d^2 U / (dr_i^a dr_j^b)
    
    For the Yukawa potential U(r) = (Q^2/4*pi*eps0*r) * exp(-r/lambda_D),
    the second derivatives have the analytic form:
    
      For i != j:
        H_{ab} = prefactor * exp(-r/lD) / r^3 *
                 [ (3 + 3*kappa + kappa^2) * r_a * r_b / r^2
                   - (1 + kappa) * delta_{ab} ]
      
      For i == j (diagonal block):
        D_{ii}^{ab} = - sum_{j!=i} D_{ij}^{ab}
    
    where kappa = r / lambda_D.
    
    The dynamical matrix is real symmetric and its eigenvalues are the
    squared phonon frequencies (omega^2).
    """
    N = positions.shape[0]
    dim = positions.shape[1]
    Dmat = np.zeros((N * dim, N * dim), dtype=float)
    
    eps0 = 8.854187817e-12
    prefactor = Q_eff**2 / (4.0 * np.pi * eps0 * m_d)
    
    # Off-diagonal blocks
    for i in range(N):
        for j in range(i + 1, N):
            r_vec = positions[i] - positions[j]
            r = np.linalg.norm(r_vec)
            if r < 1e-15:
                continue
            
            exp_term = np.exp(-r / lambda_D)
            kappa = r / lambda_D
            common = prefactor * exp_term / r**3
            
            for a in range(dim):
                for b in range(dim):
                    idx_i_a = i * dim + a
                    idx_j_b = j * dim + b
                    delta_ab = 1.0 if a == b else 0.0
                    
                    # TODO: Implement the Hessian element H_ab for Yukawa potential.
                    # HINT: For U(r) = (Q^2/4*pi*eps0*r) * exp(-r/lambda_D), the second
                    #       derivative tensor element is:
                    #       H_ab = common * [
                    #           (3 + 3*kappa + kappa^2) * r_a * r_b / r^2
                    #           - (1 + kappa) * delta_ab
                    #       ]
                    #       where kappa = r / lambda_D and common = prefactor * exp(-r/lD) / r^3.
                    raise NotImplementedError("Hole 2: Hessian element H_ab in dynamical matrix is not implemented.")
    
    # Diagonal blocks: negative sum of off-diagonal elements in each block row
    for i in range(N):
        for a in range(dim):
            for b in range(dim):
                idx_i_a = i * dim + a
                idx_i_b = i * dim + b
                total = 0.0
                for j in range(N):
                    if i == j:
                        continue
                    idx_j_b = j * dim + b
                    total += Dmat[idx_i_a, idx_j_b]
                Dmat[idx_i_a, idx_i_b] = -total
    
    # Symmetrize to eliminate roundoff errors
    Dmat = 0.5 * (Dmat + Dmat.T)
    return Dmat


def compute_phonon_frequencies(Dmat):
    """
    Compute phonon frequencies as sqrt(eigenvalues) of the dynamical matrix.
    
    Returns omega (frequencies in rad/s), sorted in ascending order.
    Three zero modes correspond to uniform translations.
    """
    # Use numpy for robust eigenvalue computation (Jacobi may struggle with ill-conditioned matrices)
    w, _ = np.linalg.eigh(Dmat)
    w = np.maximum(w, 1e-20)  # clip negative values from numerical noise
    omega = np.sqrt(w)
    return np.sort(omega)
