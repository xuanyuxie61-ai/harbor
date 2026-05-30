
import numpy as np


def qr_least_squares(A, b, lam=0.0, w0=None):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    M, N = A.shape

    if M < N:
        return np.zeros(N), 1

    AtA = A.T @ A
    Atb = A.T @ b

    if lam > 0:
        AtA += lam * np.eye(N)
        if w0 is not None:
            Atb += lam * np.asarray(w0, dtype=float)

    try:
        w = np.linalg.solve(AtA, Atb)
        return w, 0
    except np.linalg.LinAlgError:

        w = np.linalg.lstsq(A, b, rcond=None)[0]
        return w, 0


def qr_rank_revealing_ls(A, b, tol_factor=1e-12):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    M, N = A.shape


    Q, R = np.linalg.qr(A, mode='reduced')

    diag_r = np.abs(np.diag(R))
    tol = tol_factor * max(diag_r) if len(diag_r) > 0 else tol_factor
    rank = np.sum(diag_r > tol)

    if rank < N:

        R11 = R[:rank, :rank]
        R12 = R[:rank, rank:] if rank < N else np.zeros((rank, 0))
        c = Q.T @ b
        c1 = c[:rank]

        w1 = np.linalg.solve(R11, c1)
        w = np.zeros(N)
        w[:rank] = w1
    else:
        c = Q.T @ b
        w = np.linalg.solve(R, c)

    return w, rank


class MultichannelFxLMS:

    def __init__(self, n_channels, filter_len, sec_path_model, mu=0.001):
        self.L = n_channels
        self.K = filter_len
        self.sec = np.asarray(sec_path_model, dtype=float)
        self.M = self.sec.shape[0]
        self.mu = mu
        self.w = np.zeros((self.L, self.K), dtype=float)
        self.x_buffer = np.zeros((self.L, self.K), dtype=float)

    def filter_reference(self, x_new):
        x_new = np.asarray(x_new, dtype=float)
        if x_new.ndim == 0:
            x_new = np.array([x_new])


        self.x_buffer[:, 1:] = self.x_buffer[:, :-1]
        self.x_buffer[:, 0] = x_new


        x_filtered = np.zeros((self.M, self.L, self.K), dtype=float)
        sec_len = self.sec.shape[2]
        for m in range(self.M):
            for l in range(self.L):
                for t in range(sec_len):
                    if t < self.K:
                        x_filtered[m, l, :] += self.sec[m, l, t] * self.x_buffer[l, :]
        return x_filtered

    def update(self, x_new, error):
        error = np.asarray(error, dtype=float)
        x_f = self.filter_reference(x_new)


        grad = np.zeros_like(self.w)
        for m in range(self.M):
            for l in range(self.L):
                grad[l, :] += error[m] * x_f[m, l, :]

        grad = grad / (self.M + 1e-12)


        leak = 0.9999
        self.w = leak * self.w - self.mu * grad

        return self.w.copy()

    def predict_output(self, x_new):
        self.filter_reference(x_new)
        y = np.sum(self.w * self.x_buffer)
        return y


def batch_multichannel_anc_design(H, X, d, reg_lambda=1e-4):
    T, L = X.shape
    M, _, sec_len = H.shape

    K = sec_len


    N_coeffs = L * K
    A = np.zeros((T * M, N_coeffs), dtype=float)
    b = -d.flatten()

    for t in range(T):
        for m in range(M):
            row = t * M + m
            for l in range(L):
                for k in range(K):
                    if t - k >= 0:

                        for s in range(sec_len):
                            if t - k - s >= 0:
                                A[row, l * K + k] += H[m, l, s] * X[t - k - s, l]


    w_opt, rank = qr_rank_revealing_ls(A, b)
    return w_opt, rank
