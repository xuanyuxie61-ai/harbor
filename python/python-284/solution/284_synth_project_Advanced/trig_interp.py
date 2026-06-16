# -*- coding: utf-8 -*-
"""
trig_interp.py — 三角插值与周期性势能展开
============================================
核心科学问题: 使用三角(傅里叶)插值表示周期性异质结势能
和 Bloch 波函数, 利用周期性加速收敛.

融合种子项目:
  - 596_interp_trig: 三角插值基数函数

数学基础:
  对于周期为 L 的函数 f(x), 在 N 个等距节点上的
  三角插值多项式:
    T_N(x) = sum_j f(x_j) * τ_j(x)

  基数函数 (奇数 N):
    τ_j(x) = sin(Nπ(x-x_j)/L) / (N·sin(π(x-x_j)/L))

  这等价于离散傅里叶变换的逆变换.
"""

import numpy as np


class TrigonometricInterpolator:
    """
    三角(傅里叶)插值器.

    用于周期性势能和 Bloch 函数的谱方法表示.
    """

    def __init__(self, x_nodes, f_values, period=None):
        """
        参数
        ----
        x_nodes : ndarray
            等距节点 (必须等距!)
        f_values : ndarray
            节点函数值
        period : float or None
            周期 (默认取节点范围)
        """
        self.x = np.asarray(x_nodes, dtype=np.float64)
        self.f = np.asarray(f_values, dtype=np.float64)
        self.N = len(x_nodes)
        self.h = self.x[1] - self.x[0] if self.N > 1 else 1.0
        self.period = period if period is not None else (self.x[-1] - self.x[0] + self.h)

    def cardinal_function(self, j, x):
        """
        三角基数函数 τ_j(x).

        对于奇数 N:
            τ_j(x) = sin(Nπ(x-x_j)/L) / (N·sin(π(x-x_j)/L))

        对于偶数 N:
            τ_j(x) = sin(Nπ(x-x_j)/L) / (N·tan(π(x-x_j)/L))
        """
        L = self.period
        dx = x - self.x[j]

        # 避免除零
        arg = np.pi * dx / L
        sin_arg = np.sin(arg)
        N_arg = np.sin(self.N * arg)

        # 安全计算
        result = np.zeros_like(x, dtype=np.float64)
        mask = np.abs(sin_arg) > 1e-14

        if self.N % 2 == 1:
            # 奇数 N
            result[mask] = N_arg[mask] / (self.N * sin_arg[mask])
            # 在节点上: τ_j(x_j) = 1
            result[~mask] = 1.0 if j == 0 else 0.0
        else:
            # 偶数 N
            tan_arg = np.tan(arg)
            mask2 = np.abs(tan_arg) > 1e-14
            combined = mask & mask2
            result[combined] = N_arg[combined] / (self.N * tan_arg[combined])
            result[~mask] = np.cos(self.N * arg[~mask]) / self.N

        return result

    def evaluate(self, x_eval):
        """
        三角插值求值:
            T(x) = sum_{j=0}^{N-1} f_j * τ_j(x)
        """
        x_eval = np.atleast_1d(np.asarray(x_eval, dtype=np.float64))
        result = np.zeros(len(x_eval), dtype=np.float64)

        for j in range(self.N):
            tau_j = self.cardinal_function(j, x_eval)
            result += self.f[j] * tau_j

        return result

    def fourier_coefficients(self):
        """
        计算离散傅里叶系数:
            c_k = (1/N) sum_{j=0}^{N-1} f_j * exp(-2πi·j·k/N)
        """
        return np.fft.fft(self.f) / self.N

    def derivative_spectral(self):
        """
        谱方法求导 (利用 FFT):
            f'(x_j) = IFFT(i·k·FFT(f))_j

        其中 k 是波数向量.
        """
        N = self.N
        L = self.period

        # FFT
        f_hat = np.fft.fft(self.f)

        # 波数
        k = np.zeros(N)
        for j in range(N):
            if j <= N // 2:
                k[j] = 2 * np.pi * j / L
            else:
                k[j] = 2 * np.pi * (j - N) / L

        # 谱导数
        df_hat = 1j * k * f_hat
        df = np.real(np.fft.ifft(df_hat))

        return df


class PeriodicPotentialExpansion:
    """
    周期性势能的傅里叶展开.

    V(z) = sum_n V_n * exp(i·G_n·z)

    其中 G_n = 2πn/L 是倒格矢.
    """

    def __init__(self, z_grid, V_values, n_harmonics=None):
        """
        参数
        ----
        z_grid : ndarray
            空间网格
        V_values : ndarray
            势能值
        n_harmonics : int or None
            谐波数 (默认 N//2)
        """
        self.z = z_grid
        self.V = V_values
        self.N = len(z_grid)
        self.L = z_grid[-1] - z_grid[0]

        if n_harmonics is None:
            self.n_harm = self.N // 2
        else:
            self.n_harm = min(n_harmonics, self.N // 2)

        # 计算傅里叶系数
        self._compute_coefficients()

    def _compute_coefficients(self):
        """计算势能傅里叶系数."""
        V_hat = np.fft.fft(self.V) / self.N
        # 保留前 n_harm 个谐波
        self.V_fourier = np.zeros(self.N, dtype=complex)
        for k in range(-self.n_harm, self.n_harm + 1):
            idx = k % self.N
            self.V_fourier[idx] = V_hat[idx]

    def get_reciprocal_vectors(self):
        """返回倒格矢 G_n = 2πn/L."""
        return np.array([2 * np.pi * n / self.L
                         for n in range(-self.n_harm, self.n_harm + 1)])

    def reconstruct(self, z_eval):
        """
        从傅里叶系数重建势能:
            V(z) = sum_n V_n * exp(i·G_n·z)
        """
        V_recon = np.zeros(len(z_eval), dtype=np.float64)
        for k in range(-self.n_harm, self.n_harm + 1):
            G = 2 * np.pi * k / self.L
            idx = k % self.N
            V_recon += np.real(self.V_fourier[idx] * np.exp(1j * G * z_eval))
        return V_recon

    def gibbs_phenomenon_correction(self, sigma_order=4):
        """
        Gibbs 现象修正 (Lanczos σ 因子):
            V_n^corrected = V_n * sinc(n/N)^p

        其中 sinc(x) = sin(πx)/(πx).
        """
        for k in range(-self.n_harm, self.n_harm + 1):
            idx = k % self.N
            n_over_N = abs(k) / self.N
            if n_over_N > 1e-10:
                sinc_val = np.sin(np.pi * n_over_N) / (np.pi * n_over_N)
                sigma = sinc_val ** sigma_order
                self.V_fourier[idx] *= sigma
