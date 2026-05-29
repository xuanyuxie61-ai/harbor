"""
matrix_exponential_ode.py

Matrix Exponential Computation and Kepler Variational Equation Integration.

Scientific Background:
----------------------
1. Matrix Exponential:
   For a square matrix A in R^{n x n}, the matrix exponential is:
   
       exp(A) = sum_{k=0}^{infty} A^k / k!
   
   It solves the linear ODE system:
       dY/dt = A * Y,   Y(0) = I
       => Y(t) = exp(A * t)
   
   Scaling-and-squaring algorithm:
       exp(A) = [exp(A / 2^s)]^{2^s}
   where s is chosen so that ||A / 2^s|| <= 0.5.
   
   The inner exponential is computed by Padé approximation:
       exp(X) approx R_{pq}(X) = D_{pq}(X)^{-1} * N_{pq}(X)
   where N and D are matrix polynomials of degrees p and q.

2. Kepler Problem Variational Equations:
   The Kepler two-body problem in Hamiltonian form:
       dq/dt = p
       dp/dt = -q / ||q||^3
   
   The variational equations for the state transition matrix Phi(t):
       dPhi/dt = J * H_{qq}(t) * Phi
   where J = [0, I; -I, 0] is the symplectic matrix,
   and H_{qq} is the Hessian of the Hamiltonian.
   
   For Kepler: H_{qq} = -I/||q||^3 + 3 q q^T / ||q||^5.
   
   The state transition matrix satisfies:
       Phi(0) = I
       Phi(t) maps initial perturbations to final perturbations.

3. Exponential ODE (from exp_ode seed):
   Simple model: y' = alpha * y, y(t0) = y0
   Solution: y(t) = y0 * exp(alpha * (t - t0))
   This generalizes to matrix form: Y' = alpha * A * Y.
"""

import numpy as np
from typing import Tuple
import math


def matrix_exponential_pade(
    A: np.ndarray,
    order: int = 7
) -> np.ndarray:
    """
    Compute matrix exponential using scaling-and-squaring with Padé approximation.
    
    For X with ||X|| <= 0.5, the [order/order] Padé approximant:
    
        N(X) = sum_{j=0}^{order} (2*order - j)! * order! / [(2*order)! * j! * (order-j)!] * X^j
        D(X) = sum_{j=0}^{order} (2*order - j)! * order! / [(2*order)! * j! * (order-j)!] * (-X)^j
    
        exp(X) approx D(X)^{-1} * N(X)
    
    Args:
        A: square matrix
        order: Padé approximation order
    
    Returns:
        exp(A)
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")
    n = A.shape[0]
    if n == 0:
        return np.zeros((0, 0))
    
    # Scaling
    norm_A = np.linalg.norm(A, 1)
    if norm_A == 0.0:
        return np.eye(n)
    
    s = max(0, int(math.ceil(math.log2(norm_A))))
    s = max(s, 0)
    # Ensure ||A / 2^s|| is small enough
    while norm_A / (2 ** s) > 0.5:
        s += 1
    
    X = A / (2 ** s)
    
    # Padé [order/order] approximation
    I = np.eye(n)
    
    # Compute N and D
    N = np.zeros((n, n))
    D = np.zeros((n, n))
    X_power = I.copy()
    
    for j in range(order + 1):
        coeff = math.factorial(2 * order - j) * math.factorial(order) / (
            math.factorial(2 * order) * math.factorial(j) * math.factorial(order - j)
        )
        N += coeff * X_power
        D += coeff * ((-1) ** j) * X_power
        if j < order:
            X_power = X_power @ X
    
    # Solve D * E = N for E
    E = np.linalg.solve(D, N)
    
    # Squaring
    for _ in range(s):
        E = E @ E
    
    return E


def kepler_derivatives(state: np.ndarray) -> np.ndarray:
    """
    Compute Kepler problem derivatives.
    
    State vector: y = [q1, q2, p1, p2]^T
    
    Equations:
        dq1/dt = p1
        dq2/dt = p2
        dp1/dt = -q1 / (q1^2 + q2^2)^{3/2}
        dp2/dt = -q2 / (q1^2 + q2^2)^{3/2}
    
    Args:
        state: array of shape (4,)
    
    Returns:
        dstate: derivatives, shape (4,)
    """
    q1, q2, p1, p2 = state
    r2 = q1 * q1 + q2 * q2
    r3 = r2 * math.sqrt(r2)
    
    eps = 1e-14
    if r3 < eps:
        r3 = eps
    
    dq1dt = p1
    dq2dt = p2
    dp1dt = -q1 / r3
    dp2dt = -q2 / r3
    
    return np.array([dq1dt, dq2dt, dp1dt, dp2dt])


def kepler_hessian(state: np.ndarray) -> np.ndarray:
    """
    Compute the Hessian H_qq of the Kepler Hamiltonian.
    
    H(q,p) = 0.5*(p1^2 + p2^2) - 1/sqrt(q1^2 + q2^2)
    
    H_qq = d^2H / dq^2 = -I/r^3 + 3*q*q^T/r^5
    
    Args:
        state: [q1, q2, p1, p2]
    
    Returns:
        H_qq: 2x2 Hessian matrix
    """
    # TODO HOLE 1: Implement the Hessian H_qq of the Kepler Hamiltonian.
    # H(q,p) = 0.5*(p1^2 + p2^2) - 1/sqrt(q1^2 + q2^2)
    # H_qq = d^2H / dq^2 = -I/r^3 + 3*q*q^T/r^5
    # Args: state = [q1, q2, p1, p2]
    # Returns: 2x2 Hessian matrix
    raise NotImplementedError("Hole 1: kepler_hessian not implemented")


def kepler_variational_matrix(state: np.ndarray) -> np.ndarray:
    """
    Construct the variational equation matrix M for the Kepler problem.
    
    d/dt [delta_q] = [ 0      I   ] [delta_q]
         [delta_p]   [ -H_qq  0   ] [delta_p]
    
    M = [[0, 0, 1, 0],
         [0, 0, 0, 1],
         [-H_qq[0,0], -H_qq[0,1], 0, 0],
         [-H_qq[1,0], -H_qq[1,1], 0, 0]]
    
    Args:
        state: [q1, q2, p1, p2]
    
    Returns:
        M: 4x4 variational matrix
    """
    H_qq = kepler_hessian(state)
    M = np.zeros((4, 4))
    M[0, 2] = 1.0
    M[1, 3] = 1.0
    M[2, 0] = -H_qq[0, 0]
    M[2, 1] = -H_qq[0, 1]
    M[3, 0] = -H_qq[1, 0]
    M[3, 1] = -H_qq[1, 1]
    return M


def integrate_kepler_stm(
    y0: np.ndarray,
    t_span: Tuple[float, float],
    n_steps: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Integrate Kepler problem and state transition matrix (STM) using RK4.
    
    The STM Phi satisfies:
        dPhi/dt = M(t) * Phi,   Phi(0) = I
    
    We use a 4th-order Runge-Kutta scheme with simultaneous integration
    of the state and the 4x4 STM matrix (flattened to 16 components).
    
    Combined state: z = [q1, q2, p1, p2, Phi_11, Phi_12, ..., Phi_44]
    Total dimension: 4 + 16 = 20.
    
    Args:
        y0: initial state [q1, q2, p1, p2]
        t_span: (t0, tf)
        n_steps: number of RK4 steps
    
    Returns:
        y_final: final state
        Phi_final: final STM (4x4)
    """
    t0, tf = t_span
    h = (tf - t0) / n_steps
    
    # Combined state: [y; Phi_flat]
    z = np.zeros(20)
    z[:4] = y0
    z[4:].reshape(4, 4)[:, :] = np.eye(4)
    
    def dzdt(z_vec):
        y = z_vec[:4]
        Phi = z_vec[4:].reshape(4, 4)
        
        # State derivative
        dy = kepler_derivatives(y)
        
        # Variational matrix
        M = kepler_variational_matrix(y)
        
        # STM derivative
        dPhi = M @ Phi
        
        dz = np.zeros(20)
        dz[:4] = dy
        dz[4:] = dPhi.ravel()
        return dz
    
    for _ in range(n_steps):
        k1 = dzdt(z)
        k2 = dzdt(z + 0.5 * h * k1)
        k3 = dzdt(z + 0.5 * h * k2)
        k4 = dzdt(z + h * k3)
        z += (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    
    y_final = z[:4]
    Phi_final = z[4:].reshape(4, 4)
    
    return y_final, Phi_final


def propagate_perturbation(
    y0: np.ndarray,
    delta_y0: np.ndarray,
    t: float,
    n_steps: int = 1000
) -> np.ndarray:
    """
    Propagate an initial perturbation delta_y0 through the Kepler flow.
    
        delta_y(t) = Phi(t) * delta_y0
    
    Args:
        y0: reference initial state
        delta_y0: initial perturbation
        t: propagation time
        n_steps: RK4 steps
    
    Returns:
        delta_y: propagated perturbation
    """
    _, Phi = integrate_kepler_stm(y0, (0.0, t), n_steps)
    return Phi @ delta_y0


def exponential_growth_rate(A: np.ndarray) -> float:
    """
    Compute the Lyapunov-like exponential growth rate:
    
        lambda = max_i Re(lambda_i(A))
    
    where lambda_i are eigenvalues of A.
    
    Args:
        A: square matrix
    
    Returns:
        maximum real part of eigenvalues
    """
    if A.shape[0] == 0:
        return 0.0
    eigs = np.linalg.eigvals(A)
    return np.max(np.real(eigs))


if __name__ == "__main__":
    # Test matrix exponential
    A = np.array([[0.0, 1.0], [-1.0, 0.0]])
    E = matrix_exponential_pade(A, order=7)
    print("exp(A) approx:\n", E)
    print("True exp(A) = [[cos(1), sin(1)], [-sin(1), cos(1)]]")
    
    # Test Kepler STM
    y0 = np.array([1.0, 0.0, 0.0, 1.0])
    yf, Phi = integrate_kepler_stm(y0, (0.0, 1.0), n_steps=500)
    print("Final state:", yf)
    print("STM determinant (should be ~1):", np.linalg.det(Phi))
