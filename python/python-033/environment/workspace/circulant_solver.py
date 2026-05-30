
import numpy as np


def circulant_eigenvalues(a):
    a = np.asarray(a, dtype=complex)
    n = a.shape[0]
    if n == 0:
        return np.array([], dtype=complex)
    lam = np.fft.fft(a)
    return lam


def circulant_determinant(a):
    lam = circulant_eigenvalues(a)

    log_det = np.sum(np.log(lam + 1e-300))
    det = np.exp(log_det)
    return det


def circulant_solve(a, b, job=0):
    a = np.asarray(a, dtype=complex)
    b = np.asarray(b, dtype=complex)
    n = a.shape[0]
    if b.shape[0] != n:
        raise ValueError("b 的第一维必须与 a 的长度一致")

    if job != 0:

        a = np.concatenate(([a[0]], a[:0:-1]))


    a_fft = np.fft.fft(a)

    a_fft_inv = np.zeros_like(a_fft)
    nonzero = np.abs(a_fft) >= 1e-14
    a_fft_inv[nonzero] = 1.0 / a_fft[nonzero]

    if b.ndim == 1:
        b_fft = np.fft.fft(b)
        x_fft = b_fft * a_fft_inv
        x = np.fft.ifft(x_fft)
        return np.real_if_close(x, tol=1e-10)
    else:

        x = np.zeros_like(b, dtype=complex)
        for k in range(b.shape[1]):
            b_fft = np.fft.fft(b[:, k])
            x_fft = b_fft * a_fft_inv
            x[:, k] = np.fft.ifft(x_fft)
        return np.real_if_close(x, tol=1e-10)


def circulant_matvec(a, x):
    a = np.asarray(a, dtype=complex)
    x = np.asarray(x, dtype=complex)
    y = np.fft.ifft(np.fft.fft(a) * np.fft.fft(x))
    return np.real_if_close(y, tol=1e-10)


def build_circulant_dif2(n):
    a = np.zeros(n, dtype=float)
    a[0] = 2.0
    if n > 1:
        a[1] = -1.0
        a[-1] = -1.0
    return a


def test_circulant_solver():
    n = 64
    a = build_circulant_dif2(n)
    b = np.random.rand(n)
    x = circulant_solve(a, b, job=0)

    y = circulant_matvec(a, x)
    res = np.linalg.norm(y - b)
    print(f"[circulant_solver] Residual ||Cx-b|| = {res:.3e}")
    assert res < 1e-10, "Circulant solver residual too large"

    det_val = circulant_determinant(a)
    print(f"[circulant_solver] det(C) = {det_val.real:.3e}")


if __name__ == "__main__":
    test_circulant_solver()
