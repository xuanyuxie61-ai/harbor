"""
roughness_field.py
接触面粗糙度随机场生成模块
融合种子项目：
  - 870_pink_noise（1/f 噪声生成与自相关分析）

核心科学内容：
接触面粗糙度可用分形/1/f 噪声建模：
h(x) = \sum_{k=1}^{N} A_k \sin(2\pi f_k x + \phi_k)
其中功率谱密度 S(f) \propto 1/f^{\beta}，\beta \approx 1.5~2.0 对应典型工程表面。
"""
import numpy as np
from typing import Tuple, Optional


def ran1f_step(b: int, u: np.ndarray, q: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    r"""
    1/f 噪声单步生成（融合 870_pink_noise 的 ran1f + ranh）。

    算法：对 b 个信号求和，每个信号以不同频率更新：
    z = (1/b) \sum_{i=1}^{b} y_i
    其中 y_i 每 2^{i-1} 步更新一次。
    """
    if b > 31:
        raise ValueError("b must be <= 31 for ran1f")
    z = 0.0
    j = 1
    for i in range(b):
        # ranh: 保持器，每 j 步更新
        if q[i] <= 0:
            u[i] = np.random.randn()
            q[i] = j
        q[i] -= 1
        y = u[i]
        z += y
        j *= 2
    if b > 0:
        z /= b
    return z, u, q


def correlation_function(x: np.ndarray, m: int) -> np.ndarray:
    r"""
    样本自相关函数估计（融合 870_pink_noise 的 correlation）：

    R(k) = \frac{1}{N} \sum_{j=0}^{N-1-k} (x_{j+k} - \bar{x})(x_j - \bar{x})
    """
    n = len(x)
    m = min(m, n - 1)
    xbar = np.mean(x)
    r = np.zeros(m + 1)
    for k in range(m + 1):
        for j in range(n - k):
            r[k] += (x[j + k] - xbar) * (x[j] - xbar)
    r /= n
    return r


def generate_pink_noise_profile(n_points: int, length: float = 1.0,
                                 beta: float = 1.8,
                                 b_levels: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    生成一维 1/f^{\beta} 粗糙度轮廓。

    频域方法：
    h(x) = \mathcal{F}^{-1} \{ \sqrt{S(f)} \cdot \mathcal{N}(0,1) \}
    S(f) = C \cdot f^{-\beta}
    """
    x = np.linspace(0.0, length, n_points)
    dx = length / (n_points - 1)
    # 频域
    freqs = np.fft.rfftfreq(n_points, d=dx)
    freqs[0] = 1e-6  # 避免除零
    spectrum = freqs ** (-beta / 2.0)
    spectrum[0] = 0.0  # 去除直流分量
    # 随机相位
    phases = np.random.randn(len(freqs)) + 1j * np.random.randn(len(freqs))
    fft_vals = spectrum * phases
    h = np.fft.irfft(fft_vals, n=n_points)
    # 归一化
    h = (h - np.mean(h)) / (np.std(h) + 1e-20)
    return x, h


def generate_2d_fractal_surface(nx: int, ny: int, lx: float = 1.0, ly: float = 1.0,
                                 hurst: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    使用频域滤波法生成二维分形粗糙表面。

    功率谱密度：
    S(k_x, k_y) = C \cdot (k_x^2 + k_y^2)^{-(H+1)}

    其中 H 为 Hurst 指数，与分形维数 D = 3 - H 相关。
    对于典型金属表面，H \in [0.5, 0.9]。
    """
    x = np.linspace(0.0, lx, nx)
    y = np.linspace(0.0, ly, ny)
    X, Y = np.meshgrid(x, y)
    # 频域
    fx = np.fft.fftfreq(nx, d=lx / nx)
    fy = np.fft.fftfreq(ny, d=ly / ny)
    FX, FY = np.meshgrid(fx, fy)
    k2 = FX ** 2 + FY ** 2
    k2[0, 0] = 1e-12
    spectrum = k2 ** (-(hurst + 1.0) / 2.0)
    spectrum[0, 0] = 0.0
    random_phase = np.random.randn(ny, nx) + 1j * np.random.randn(ny, nx)
    fft_surface = spectrum * random_phase
    h = np.real(np.fft.ifft2(fft_surface))
    h = (h - np.mean(h)) / (np.std(h) + 1e-20)
    return X, Y, h


def apply_roughness_to_mesh(mesh_nodes: np.ndarray, roughness_1d: np.ndarray,
                             contact_mask: np.ndarray, scale: float = 1e-5) -> np.ndarray:
    r"""
    将一维粗糙度轮廓叠加到网格接触节点上。

    h_{rough}(x_i) = scale \cdot roughness(x_i)
    节点新位置：y_i^{new} = y_i + h_{rough}(x_i)
    """
    nodes_new = mesh_nodes.copy()
    x_contact = mesh_nodes[contact_mask, 0]
    n_contact = len(x_contact)
    if n_contact != len(roughness_1d):
        # 线性插值
        x_src = np.linspace(np.min(x_contact), np.max(x_contact), len(roughness_1d))
        h_interp = np.interp(x_contact, x_src, roughness_1d)
    else:
        h_interp = roughness_1d
    for idx, node in enumerate(np.where(contact_mask)[0]):
        nodes_new[node, 1] += scale * h_interp[idx]
    return nodes_new
