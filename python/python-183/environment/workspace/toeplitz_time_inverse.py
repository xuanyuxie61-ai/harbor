
import numpy as np
from typing import Tuple


def sample_autocovariance(x: np.ndarray, max_lag: int) -> np.ndarray:
    n = len(x)
    if n < 2:
        raise ValueError("序列长度至少为 2。")
    if max_lag < 0 or max_lag >= n:
        raise ValueError("max_lag 必须在 [0, n-1) 内。")
    x_c = x - np.mean(x)
    gamma = np.zeros(max_lag + 1)
    for h in range(max_lag + 1):
        gamma[h] = np.dot(x_c[h:], x_c[:n - h]) / n
    return gamma


def toeplitz_matrix(n: int, x: np.ndarray) -> np.ndarray:
    T = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            lag = abs(i - j)
            if lag < len(x):
                T[i, j] = x[lag]
            else:
                T[i, j] = 0.0
    return T


def exchange_matrix(n: int) -> np.ndarray:
    J = np.zeros((n, n))
    for i in range(n):
        J[i, n - 1 - i] = 1.0
    return J


def hankel_matrix(n: int, c: np.ndarray) -> np.ndarray:
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            idx = i + j
            if idx < len(c):
                H[i, j] = c[idx]
    return H


def fiedler_toeplitz_inverse(T: np.ndarray) -> np.ndarray:
    n = T.shape[0]
    if n == 0:
        raise ValueError("矩阵维度为 0。")
    J = exchange_matrix(n)
    H = J @ T




    p = np.zeros(n)
    start = n // 2
    for i in range(n):
        if start + i < n:
            p[i] = T[i, 0] if i < n else 0.0
        else:
            p[i] = 0.0



    x_full = np.zeros(2 * n - 1)
    for i in range(n):
        x_full[n - 1 - i] = T[i, 0]
        x_full[n - 1 + i] = T[0, i]
    p_vec = np.concatenate([x_full[n:], [0.0]])
    q_vec = np.zeros(n)
    q_vec[n - 1] = 1.0


    H_reg = H + 1e-10 * np.eye(n)
    u = np.linalg.solve(H_reg, p_vec)
    v = np.linalg.solve(H_reg, q_vec)


    z1 = np.zeros(n)
    w1 = np.concatenate([v[1:], [z1[-1]]]) if n > 1 else v.copy()
    M1 = hankel_matrix(n, w1)

    z2 = np.zeros(n - 1) if n > 1 else np.array([])
    w2 = np.concatenate([z2, u])
    M2 = toeplitz_matrix(n, w2)

    z3 = np.zeros(n)
    z3[0] = -1.0
    w3 = np.concatenate([u[1:], [z3[-1]]]) if n > 1 else u.copy()
    M3 = hankel_matrix(n, w3)

    z4 = np.zeros(n - 1) if n > 1 else np.array([])
    w4 = np.concatenate([z4, v])
    M4 = toeplitz_matrix(n, w4)

    K = M1 @ M2 - M3 @ M4
    T_inv = K @ J
    return T_inv


def yule_walker_solve(gamma: np.ndarray, p: int) -> np.ndarray:
    if p >= len(gamma):
        raise ValueError("p 必须小于 gamma 的长度。")
    Gamma_p = toeplitz_matrix(p, gamma[:p])
    rhs = gamma[1:p + 1]

    try:
        phi = np.linalg.solve(Gamma_p + 1e-10 * np.eye(p), rhs)
    except np.linalg.LinAlgError:
        phi = np.linalg.lstsq(Gamma_p, rhs, rcond=None)[0]
    return phi


def lag_causal_strength(T_inv: np.ndarray) -> np.ndarray:
    n = T_inv.shape[0]
    C = np.zeros(n)
    for h in range(n):
        s = 0.0
        for i in range(n):
            j = i + h
            if j < n:
                s += abs(T_inv[i, j])
            if h > 0:
                j2 = i - h
                if j2 >= 0:
                    s += abs(T_inv[i, j2])
        C[h] = s
    return C


def demo():
    np.random.seed(11)
    n = 64

    phi_true = np.array([0.6, -0.3])
    eps = np.random.randn(n)
    x = np.zeros(n)
    x[0] = eps[0]
    x[1] = 0.6 * x[0] + eps[1]
    for t in range(2, n):
        x[t] = phi_true[0] * x[t - 1] + phi_true[1] * x[t - 2] + eps[t]

    max_lag = 10
    gamma = sample_autocovariance(x, max_lag)
    phi_est = yule_walker_solve(gamma, p=2)
    print(f"[toeplitz_time_inverse] 真实 AR 系数: {phi_true}, 估计: {phi_est.round(4)}")


    T = toeplitz_matrix(max_lag + 1, gamma)
    T_inv = fiedler_toeplitz_inverse(T)
    C = lag_causal_strength(T_inv)
    print(f"[toeplitz_time_inverse] 滞后因果强度 (前6): {C[:6].round(4)}")
    return phi_est, C


if __name__ == "__main__":
    demo()
