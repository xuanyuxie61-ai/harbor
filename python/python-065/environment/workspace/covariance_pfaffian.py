
import numpy as np


def pfaffian_LTL(A_in):
    A = np.array(A_in, dtype=np.float64, copy=True)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("输入必须是方阵")
    N = A.shape[0]


    if np.linalg.norm(A + A.T) > 1e-12 * N:
        raise ValueError("输入矩阵不是斜对称的")

    if N % 2 == 1:
        return 0.0

    pf = 1.0
    for k in range(0, N - 1, 2):

        sub = A[k + 1:N, k]
        kp_rel = np.argmax(np.abs(sub))
        kp = kp_rel + k + 1

        if kp != k + 1:

            temp = A[k + 1, k:N].copy()
            A[k + 1, k:N] = A[kp, k:N]
            A[kp, k:N] = temp

            temp = A[k:N, k + 1].copy()
            A[k:N, k + 1] = A[k:N, kp]
            A[k:N, kp] = temp
            pf = -pf

        pf *= A[k, k + 1]

        if A[k + 1, k] != 0.0:
            tau = A[k + 2:N, k] / A[k + 1, k]

            if k + 2 < N:
                A[k + 2:N, k + 2:N] += np.outer(tau, A[k + 2:N, k + 1]) \
                                       - np.outer(A[k + 2:N, k + 1], tau)

    pf *= A[N - 1, N - 2] if N >= 2 else 1.0
    return float(pf)


def build_skew_covariance_from_kernel(nodes, kernel_func):
    n = nodes.shape[1]
    K = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            val = kernel_func(nodes[:, i], nodes[:, j])
            K[i, j] = val
            K[j, i] = -val
    return K


def gaussian_random_field_log_partition(K):
    pf = pfaffian_LTL(K)
    if abs(pf) < 1e-14:
        return -np.inf
    return 0.5 * np.log(abs(pf))


def extreme_event_pfaffian_correlation(nodes, correlation_length=1.0):
    def kernel(xi, xj):
        dx = np.linalg.norm(xi - xj)
        if dx < 1e-14:
            return 0.0
        s = np.sign(np.sum(xi - xj))
        if s == 0:
            s = 1.0
        return s * np.exp(-dx / correlation_length)

    K = build_skew_covariance_from_kernel(nodes, kernel)
    return K, pfaffian_LTL(K)


def test_pfaffian():

    A = np.array([
        [0, 1, 0, 0],
        [-1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, -1, 0],
    ], dtype=np.float64)
    pf = pfaffian_LTL(A)
    assert abs(pf - 1.0) < 1e-10


    N = 6
    B = np.random.randn(N, N)
    B = B - B.T
    pf2 = pfaffian_LTL(B)
    det_val = np.linalg.det(B)
    assert abs(pf2 ** 2 - det_val) < 1e-8 * max(abs(det_val), 1.0)
    print("covariance_pfaffian 自测试通过")


if __name__ == "__main__":
    test_pfaffian()
