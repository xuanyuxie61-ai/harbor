
import numpy as np
from typing import Tuple
import math


def matrix_exponential_pade(
    A: np.ndarray,
    order: int = 7
) -> np.ndarray:
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")
    n = A.shape[0]
    if n == 0:
        return np.zeros((0, 0))
    

    norm_A = np.linalg.norm(A, 1)
    if norm_A == 0.0:
        return np.eye(n)
    
    s = max(0, int(math.ceil(math.log2(norm_A))))
    s = max(s, 0)

    while norm_A / (2 ** s) > 0.5:
        s += 1
    
    X = A / (2 ** s)
    

    I = np.eye(n)
    

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
    

    E = np.linalg.solve(D, N)
    

    for _ in range(s):
        E = E @ E
    
    return E


def kepler_derivatives(state: np.ndarray) -> np.ndarray:
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
    q1, q2 = state[0], state[1]
    r2 = q1 * q1 + q2 * q2
    r = math.sqrt(r2)
    
    eps = 1e-14
    if r < eps:
        r = eps
        r2 = eps * eps
    
    r3 = r2 * r
    r5 = r3 * r2
    
    H_qq = np.zeros((2, 2))
    H_qq[0, 0] = -1.0 / r3 + 3.0 * q1 * q1 / r5
    H_qq[0, 1] = 3.0 * q1 * q2 / r5
    H_qq[1, 0] = H_qq[0, 1]
    H_qq[1, 1] = -1.0 / r3 + 3.0 * q2 * q2 / r5
    
    return H_qq


def kepler_variational_matrix(state: np.ndarray) -> np.ndarray:
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
    t0, tf = t_span
    h = (tf - t0) / n_steps
    

    z = np.zeros(20)
    z[:4] = y0
    z[4:].reshape(4, 4)[:, :] = np.eye(4)
    
    def dzdt(z_vec):
        y = z_vec[:4]
        Phi = z_vec[4:].reshape(4, 4)
        

        dy = kepler_derivatives(y)
        

        M = kepler_variational_matrix(y)
        

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
    _, Phi = integrate_kepler_stm(y0, (0.0, t), n_steps)
    return Phi @ delta_y0


def exponential_growth_rate(A: np.ndarray) -> float:
    if A.shape[0] == 0:
        return 0.0
    eigs = np.linalg.eigvals(A)
    return np.max(np.real(eigs))


if __name__ == "__main__":

    A = np.array([[0.0, 1.0], [-1.0, 0.0]])
    E = matrix_exponential_pade(A, order=7)
    print("exp(A) approx:\n", E)
    print("True exp(A) = [[cos(1), sin(1)], [-sin(1), cos(1)]]")
    

    y0 = np.array([1.0, 0.0, 0.0, 1.0])
    yf, Phi = integrate_kepler_stm(y0, (0.0, 1.0), n_steps=500)
    print("Final state:", yf)
    print("STM determinant (should be ~1):", np.linalg.det(Phi))
