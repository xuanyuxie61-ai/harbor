"""
stochastic_forcing.py
随机强迫场生成与随机场模拟模块

科学背景:
海洋中尺度涡旋系统受到大气风应力、热力强迫等多种随机扰动.
这些强迫通常建模为具有特定空间-时间相关结构的随机过程:

  1. 空间高斯随机场:
     协方差函数: C(r) = sigma^2 * exp(-r^2 / (2*L_c^2))
     其中 L_c 为相关长度尺度, sigma 为振幅.

  2. 时间相关噪声 (Ornstein-Uhlenbeck):
     d eta/dt = -eta/tau + sqrt(2D/tau) xi(t)
     其中 tau 为相关时间, D 为扩散系数, xi 为白噪声.

  3. 谱空间强迫:
     在特定波数带注入能量, 模拟大气-海洋相互作用.

本模块实现:
  - 基于 Cholesky 分解的相关高斯随机场生成
  - 指数型/高斯型协方差矩阵构造
  - 时间相关的 Ornstein-Uhlenbeck 过程
  - 谱空间带限随机强迫

融合来源:
- 026_asa007: Cholesky 分解与 SPD 矩阵求逆
- 454_gaussian: Hermite 多项式与高斯函数 (用于协方差核展开)
"""

import numpy as np
from numerics_core import cholesky_decompose, hermite_polynomial_prob
from typing import Tuple, Optional


# ============================================================
# 1. 协方差矩阵构造
# ============================================================

def gaussian_covariance_matrix(coords: np.ndarray, sigma: float, Lc: float,
                                nugget: float = 1e-10) -> np.ndarray:
    """
    高斯型协方差矩阵 (径向基函数核).

    C_{ij} = \sigma^2 * exp( -||x_i - x_j||^2 / (2*L_c^2) ) + nugget * \delta_{ij}

    该矩阵对称正定, 可通过 Cholesky 分解生成相关随机场.
    """
    n = coords.shape[0]
    C = np.zeros((n, n))
    for i in range(n):
        dx = coords[i, 0] - coords[:, 0]
        dy = coords[i, 1] - coords[:, 1]
        r2 = dx ** 2 + dy ** 2
        C[i, :] = sigma ** 2 * np.exp(-r2 / (2.0 * Lc ** 2))
    C += nugget * np.eye(n)
    # 强制对称
    C = 0.5 * (C + C.T)
    return C


def exponential_covariance_matrix(coords: np.ndarray, sigma: float, Lc: float,
                                  nugget: float = 1e-10) -> np.ndarray:
    """
    指数型协方差矩阵 (Matérn 1/2):
      C_{ij} = \sigma^2 * exp( -||x_i - x_j|| / L_c ) + nugget * \delta_{ij}
    """
    n = coords.shape[0]
    C = np.zeros((n, n))
    for i in range(n):
        dx = coords[i, 0] - coords[:, 0]
        dy = coords[i, 1] - coords[:, 1]
        r = np.sqrt(dx ** 2 + dy ** 2)
        C[i, :] = sigma ** 2 * np.exp(-r / Lc)
    C += nugget * np.eye(n)
    C = 0.5 * (C + C.T)
    return C


def matern_covariance_matrix(coords: np.ndarray, sigma: float, Lc: float,
                             nu: float = 1.5, nugget: float = 1e-10) -> np.ndarray:
    """
    Matérn 协方差函数 (更一般的平滑性):
      C(r) = \sigma^2 * (2^{1-\nu}/\Gamma(\nu)) * (\sqrt{2\nu} r/L_c)^{\nu} K_{\nu}(\sqrt{2\nu} r/L_c)

    其中 K_\nu 为第二类修正 Bessel 函数.
    ν = 0.5: 指数型
    ν = 1.5: 一次可微
    ν = 2.5: 二次可微
    ν → ∞: 高斯型
    """
    from scipy.special import kv, gamma
    n = coords.shape[0]
    C = np.zeros((n, n))
    for i in range(n):
        dx = coords[i, 0] - coords[:, 0]
        dy = coords[i, 1] - coords[:, 1]
        r = np.sqrt(dx ** 2 + dy ** 2)
        # 避免 r=0 处奇点
        r_safe = np.where(r < 1e-10, 1e-10, r)
        scale = np.sqrt(2.0 * nu) * r_safe / Lc
        C[i, :] = sigma ** 2 * (2.0 ** (1.0 - nu) / gamma(nu)) * (scale ** nu) * kv(nu, scale)
    # r=0 处设为 sigma^2
    np.fill_diagonal(C, sigma ** 2)
    C += nugget * np.eye(n)
    C = 0.5 * (C + C.T)
    return C


# ============================================================
# 2. 基于 Cholesky 的随机场生成 (from 026_asa007)
# ============================================================

def generate_gaussian_random_field(C: np.ndarray, rng: Optional[np.random.Generator] = None,
                                   n_samples: int = 1) -> np.ndarray:
    """
    使用 Cholesky 分解生成相关高斯随机场.

    算法:
      1. 对协方差矩阵 C 做 Cholesky 分解: C = U^T U
      2. 生成独立标准正态向量 z ~ N(0,I)
      3. 相关场: f = U^T z ~ N(0, C)

    若 C 条件数差, 改用 SVD 或添加 nugget.
    """
    n = C.shape[0]
    if rng is None:
        rng = np.random.default_rng(42)

    try:
        U = cholesky_decompose(C, tol=1e-12)
    except ValueError:
        # 若 Cholesky 失败, 添加微小 nugget 后重试
        C2 = C + 1e-8 * np.eye(n)
        U = cholesky_decompose(C2, tol=1e-12)

    z = rng.standard_normal((n, n_samples))
    f = U.T @ z
    return f


def generate_spectral_forcing(Nx: int, Ny: int, Lx: float, Ly: float,
                              forcing_amplitude: float = 1.0,
                              k_inject_min: float = 2.0, k_inject_max: float = 6.0,
                              rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    在特定波数带生成随机谱空间强迫.

    强迫形式:
      F_k = A * exp(i * \phi_k)  for k_inject_min < |k| < k_inject_max
      F_k = 0                     otherwise

    其中 \phi_k 为 [0, 2\pi) 均匀分布的随机相位.

    用于模拟大气风应力在特定空间尺度上的能量注入.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, Lx / Nx)[:Nx // 2 + 1]
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny, Ly / Ny)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX ** 2 + KY ** 2)

    mask = (K >= k_inject_min) & (K <= k_inject_max)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(Ny, Nx // 2 + 1))
    amplitude = forcing_amplitude * mask.astype(float)

    F_h = amplitude * np.exp(1j * phase)
    # 保证实输出
    F_h[0, 0] = 0.0
    if Ny % 2 == 0:
        F_h[Ny // 2, :] = 0.0
    if Nx % 2 == 0:
        F_h[:, Nx // 2] = 0.0

    return F_h


# ============================================================
# 3. Ornstein-Uhlenbeck 过程
# ============================================================

class OrnsteinUhlenbeckProcess:
    """
    Ornstein-Uhlenbeck 过程: 具有指数衰减相关性的高斯过程.

    SDE:
      dX_t = -\theta X_t dt + \sigma dW_t

    解析解:
      X_t = X_0 e^{-\theta t} + \sigma \int_0^t e^{-\theta(t-s)} dW_s

    稳态方差: Var(X) = \sigma^2 / (2\theta)
    相关时间: \tau = 1/\theta
    """

    def __init__(self, theta: float, sigma: float, X0: float = 0.0,
                 dt: float = 0.01, shape: Tuple[int, ...] = ()):
        self.theta = float(theta)
        self.sigma = float(sigma)
        self.X = np.full(shape, X0, dtype=float)
        self.dt = float(dt)
        self.shape = shape

    def step(self, rng: Optional[np.random.Generator] = None):
        """
        Euler-Maruyama 步进:
          X_{n+1} = X_n - \theta X_n dt + \sigma sqrt(dt) * Z_n
        """
        if rng is None:
            rng = np.random.default_rng()
        dW = rng.standard_normal(self.shape) * np.sqrt(self.dt)
        self.X = self.X - self.theta * self.X * self.dt + self.sigma * dW
        return self.X

    def steady_state_std(self) -> float:
        """稳态标准差."""
        return self.sigma / np.sqrt(2.0 * self.theta)


# ============================================================
# 4. Hermite 展开的随机强迫 (from 454_gaussian)
# ============================================================

def forcing_hermite_expansion(x: np.ndarray, y: np.ndarray,
                              coeffs: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    使用 Hermite 多项式展开构造空间结构化的随机强迫场.

    展开式:
      F(x,y) = sum_{m,n} c_{mn} He_m(x/\sigma_x) He_n(y/\sigma_y) * exp(-(x^2+y^2)/(2\sigma^2))

    高斯包络保证能量集中在涡旋尺度附近.
    """
    X, Y = np.meshgrid(x, y)
    nx, ny = len(x), len(y)
    N = int(np.sqrt(len(coeffs)))
    if N * N != len(coeffs):
        N = int(np.ceil(np.sqrt(len(coeffs))))
        coeffs = np.pad(coeffs, (0, N * N - len(coeffs)))

    result = np.zeros((ny, nx))
    Hx = hermite_polynomial_prob(N - 1, X.flatten() / sigma)
    Hy = hermite_polynomial_prob(N - 1, Y.flatten() / sigma)
    env = np.exp(-0.5 * (X.flatten() ** 2 + Y.flatten() ** 2) / sigma ** 2)

    idx = 0
    for m in range(N):
        for n in range(N):
            if idx < len(coeffs):
                result.flat += coeffs[idx] * Hx[m, :] * Hy[n, :] * env
                idx += 1

    return result


if __name__ == "__main__":
    # 测试协方差矩阵与随机场
    coords = np.random.default_rng(42).random((20, 2))
    C = gaussian_covariance_matrix(coords, sigma=1.0, Lc=0.3)
    f = generate_gaussian_random_field(C, n_samples=3)
    print("Random field shape:", f.shape)
    print("Empirical covariance vs target:", np.max(np.abs(np.cov(f) - C)))

    # 测试 OU 过程
    ou = OrnsteinUhlenbeckProcess(theta=1.0, sigma=0.5, dt=0.01, shape=(1000,))
    for _ in range(1000):
        ou.step()
    print("OU steady std:", ou.steady_state_std(), "actual:", np.std(ou.X))
