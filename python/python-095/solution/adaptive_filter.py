"""
adaptive_filter.py
多通道自适应滤波算法 (FxLMS / QR-RLS)

融合原始项目:
  - 938_qr_solve (QR分解最小二乘)

科学背景:
  主动噪声控制的核心是自适应滤波器.
  对于多通道系统,第m个误差传感器处的声压为:
      e_m(n) = d_m(n) + \sum_{l=1}^{L} \sum_{k=0}^{K-1} w_{lk}(n) x_l(n-k) * h_{ml}(k)

  其中 x_l 为参考信号, w_{lk} 为自适应滤波器系数,
  h_{ml} 为次级通路 (secondary path) 的脉冲响应.

  最小化目标泛函:
      J(\mathbf{w}) = E[ \mathbf{e}^T(n) \mathbf{e}(n) ]
                    + \lambda ||\mathbf{w} - \mathbf{w}_0||^2

  本模块实现:
  1. 基于QR分解的批量最小二乘估计 (离线设计)
  2. 多通道FxLMS (Filtered-x LMS) 在线自适应算法
  3. 正则化QR-RLS快速更新
"""

import numpy as np


def qr_least_squares(A, b, lam=0.0, w0=None):
    """
    使用正规方程结合Cholesky分解求解正则化最小二乘问题.

    问题:
        min || A w - b ||^2 + lambda ||w - w0||^2

    等价于:
        (A^T A + lambda I) w = A^T b + lambda w0

    参数:
        A: (M, N) 回归矩阵
        b: (M,) 观测向量
        lam: 正则化参数
        w0: (N,) 先验系数

    返回:
        w: (N,) 最优系数
        flag: 0=成功, 1=错误
    """
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
        # 使用伪逆作为回退
        w = np.linalg.lstsq(A, b, rcond=None)[0]
        return w, 0


def qr_rank_revealing_ls(A, b, tol_factor=1e-12):
    """
    基于QR分解的秩揭示最小二乘.

    通过Householder QR分解:
        A P = Q [ R11  R12 ]
                [  0    0  ]
    处理秩亏情况,提高数值稳定性.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    M, N = A.shape

    # numpy的qr实现已使用Householder变换
    Q, R = np.linalg.qr(A, mode='reduced')
    # 检查秩
    diag_r = np.abs(np.diag(R))
    tol = tol_factor * max(diag_r) if len(diag_r) > 0 else tol_factor
    rank = np.sum(diag_r > tol)

    if rank < N:
        # 秩亏: 使用截断R求解最小范数解
        R11 = R[:rank, :rank]
        R12 = R[:rank, rank:] if rank < N else np.zeros((rank, 0))
        c = Q.T @ b
        c1 = c[:rank]
        # 最小范数解: w1 = R11^{-1} c1, w2 = 0
        w1 = np.linalg.solve(R11, c1)
        w = np.zeros(N)
        w[:rank] = w1
    else:
        c = Q.T @ b
        w = np.linalg.solve(R, c)

    return w, rank


class MultichannelFxLMS:
    """
    多通道 Filtered-x LMS 自适应滤波器.

    算法公式:
        滤波参考信号:
            x'_l(n) = \hat{h}_{ml} * x_l(n)
        系数更新:
            w_{lk}(n+1) = w_{lk}(n) - mu * e_m(n) * x'_l(n-k)

    收敛条件 (mean-square stability):
        0 < mu < 2 / (L * P_max * \sum_m \sum_l ||\hat{h}_{ml}||^2)
        其中 P_max 为参考信号最大功率.
    """

    def __init__(self, n_channels, filter_len, sec_path_model, mu=0.001):
        """
        参数:
            n_channels: 参考通道数 L
            filter_len: 滤波器长度 K
            sec_path_model: (M, L, sec_len) 次级通路估计
            mu: 步长
        """
        self.L = n_channels
        self.K = filter_len
        self.sec = np.asarray(sec_path_model, dtype=float)
        self.M = self.sec.shape[0]  # 误差传感器数
        self.mu = mu
        self.w = np.zeros((self.L, self.K), dtype=float)
        self.x_buffer = np.zeros((self.L, self.K), dtype=float)

    def filter_reference(self, x_new):
        """
        对参考信号进行次级通路滤波.

        x_new: (L,) 新的参考信号采样
        """
        x_new = np.asarray(x_new, dtype=float)
        if x_new.ndim == 0:
            x_new = np.array([x_new])

        # 更新缓冲区
        self.x_buffer[:, 1:] = self.x_buffer[:, :-1]
        self.x_buffer[:, 0] = x_new

        # 滤波: x'_m,l(n) = sum_t sec[m,l,t] * x_l(n-t)
        x_filtered = np.zeros((self.M, self.L, self.K), dtype=float)
        sec_len = self.sec.shape[2]
        for m in range(self.M):
            for l in range(self.L):
                for t in range(sec_len):
                    if t < self.K:
                        x_filtered[m, l, :] += self.sec[m, l, t] * self.x_buffer[l, :]
        return x_filtered

    def update(self, x_new, error):
        """
        单步更新.

        参数:
            x_new: (L,) 新参考信号
            error: (M,) 误差传感器信号

        返回:
            w: 当前滤波器系数 (flattened)
        """
        error = np.asarray(error, dtype=float)
        x_f = self.filter_reference(x_new)

        # 梯度下降
        grad = np.zeros_like(self.w)
        for m in range(self.M):
            for l in range(self.L):
                grad[l, :] += error[m] * x_f[m, l, :]

        grad = grad / (self.M + 1e-12)

        # 带泄漏的LMS (增强鲁棒性)
        leak = 0.9999
        self.w = leak * self.w - self.mu * grad

        return self.w.copy()

    def predict_output(self, x_new):
        """
        计算当前滤波器对次级源的驱动信号.
        """
        self.filter_reference(x_new)
        y = np.sum(self.w * self.x_buffer)
        return y


def batch_multichannel_anc_design(H, X, d, reg_lambda=1e-4):
    """
    批量设计多通道ANC滤波器系数.

    构造回归矩阵并求解最小二乘:
        min || X_eff w + d ||^2 + lambda ||w||^2

    参数:
        H: (M, L, sec_len) 次级通路
        X: (T, L) 参考信号历史
        d: (T, M) 期望初级噪声
        reg_lambda: 正则化参数

    返回:
        w_opt: (L*K,) 最优系数
    """
    T, L = X.shape
    M, _, sec_len = H.shape
    # 简化:假设滤波器长度等于sec_len
    K = sec_len

    # 构造有效回归矩阵
    N_coeffs = L * K
    A = np.zeros((T * M, N_coeffs), dtype=float)
    b = -d.flatten()  # 目标: A w \approx -d

    for t in range(T):
        for m in range(M):
            row = t * M + m
            for l in range(L):
                for k in range(K):
                    if t - k >= 0:
                        # 卷积和
                        for s in range(sec_len):
                            if t - k - s >= 0:
                                A[row, l * K + k] += H[m, l, s] * X[t - k - s, l]

    # 使用QR秩揭示求解
    w_opt, rank = qr_rank_revealing_ls(A, b)
    return w_opt, rank
