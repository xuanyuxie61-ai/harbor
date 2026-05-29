"""
optical_transfer.py — 光学传递函数与四面体域积分

融合原项目: 1248_tetrahedron_integrals (单位四面体上的单项式积分)

功能:
  - 瞳孔函数在四面体域上的精确积分 (用于非圆形瞳孔的像差分析)
  - 光学传递函数 (OTF) 和点扩散函数 (PSF) 计算
  - 调制传递函数 (MTF) 在四面体采样点上的离散评估
  - Strehl比和波前方差的精确计算

物理模型:
  1. 四面体积分 (源自1248):
       在单位四面体 T = conv{ (0,0,0), (1,0,0), (0,1,0), (0,0,1) } 上,
       单项式积分:
         I(e1,e2,e3) = integral_T x^{e1} y^{e2} z^{e3} dV = e1! e2! e3! / (e1+e2+e3+3)!

  2. 瞳孔函数:
       P(x,y) = A(x,y) * exp(i * phi(x,y))
     其中 A 为振幅透射 (0或1), phi 为波前相位.

  3. 光学传递函数 (OTF):
       OTF(fx, fy) = integral P(x,y) P*(x - lambda*z*fx, y - lambda*z*fy) dx dy
                   = AutoCorrelation{P}

  4. 点扩散函数 (PSF):
       PSF(x,y) = | FFT{ P } |^2

  5. 在AO中, 非圆形子孔径或分割镜面可分解为多个四面体单元,
     通过四面体积分精确计算子孔径的相位矩.
"""

import numpy as np
from math import factorial


# --- 四面体积分 (源自1248_tetrahedron_integrals) ---

def tetrahedron01_monomial_integral(e1, e2, e3):
    """
    计算单位四面体上单项式 x^{e1} y^{e2} z^{e3} 的精确积分.

    公式:
      I = e1! * e2! * e3! / (e1 + e2 + e3 + 3)!
    """
    if e1 < 0 or e2 < 0 or e3 < 0:
        raise ValueError("Exponents must be non-negative.")
    num = factorial(e1) * factorial(e2) * factorial(e3)
    den = factorial(e1 + e2 + e3 + 3)
    return float(num) / float(den)


def tetrahedron01_volume():
    """单位四面体体积 V = 1/6."""
    return 1.0 / 6.0


def tetrahedron01_sample(n_samples, seed=None):
    """
    在单位四面体内均匀随机采样.

    算法 (指数分布法):
      生成4个均匀随机数 U_i ~ Uniform(0,1),
      E_i = -log(U_i),
      x = E_1 / S, y = E_2 / S, z = E_3 / S, 其中 S = sum(E_i).
    """
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1.")
    if seed is not None:
        np.random.seed(seed)
    U = np.random.rand(n_samples, 4)
    E = -np.log(np.clip(U, 1e-30, 1.0))
    S = np.sum(E, axis=1, keepdims=True)
    points = E[:, :3] / S
    return points


def integrate_over_tetrahedral_mesh(integrand_func, n_samples_per_cell=1000, n_cells=8):
    """
    将单位立方体 [0,1]^3 剖分为6个四面体,
    在每个四面体上蒙特卡洛采样积分.

    用于非均匀瞳孔的三维相位分布积分.
    """
    total_integral = 0.0
    for _ in range(n_cells):
        pts = tetrahedron01_sample(n_samples_per_cell)
        vals = np.array([integrand_func(p[0], p[1], p[2]) for p in pts])
        total_integral += np.mean(vals) * tetrahedron01_volume()
    return total_integral / n_cells


# --- 光学传递函数计算 ---

def compute_pupil_function(grid_size, aperture_mask, phase):
    """
    计算瞳孔函数 P(x,y) = A(x,y) * exp(i*phi(x,y)).
    """
    if aperture_mask.shape != phase.shape:
        raise ValueError("aperture_mask and phase must have the same shape.")
    P = np.zeros_like(phase, dtype=np.complex128)
    P[aperture_mask] = np.exp(1j * phase[aperture_mask])
    return P


def compute_otf_from_pupil(P):
    """
    由瞳孔函数计算OTF.

    OTF = AutoCorrelation(P) = IFFT[ |FFT(P)|^2 ]
    """
    F = np.fft.fft2(P)
    otf = np.fft.ifft2(np.abs(F) ** 2)
    return np.fft.fftshift(otf)


def compute_psf_from_pupil(P, pixel_scale, wavelength, focal_length):
    """
    由瞳孔函数计算PSF.

    PSF = | FFT( P ) |^2
    空间坐标:
      x = lambda * f * fx
      y = lambda * f * fy
    """
    if pixel_scale <= 0 or wavelength <= 0 or focal_length <= 0:
        raise ValueError("Physical parameters must be positive.")
    F = np.fft.fftshift(np.fft.fft2(P))
    psf = np.abs(F) ** 2
    psf = psf / np.max(psf)
    N = P.shape[0]
    freq = np.fft.fftfreq(N, d=pixel_scale)
    x_coords = wavelength * focal_length * freq
    return psf, x_coords


def compute_mtf_from_otf(otf):
    """
    由OTF计算MTF.

    MTF = |OTF| / |OTF(0,0)|
    """
    if otf.ndim == 2:
        center = (otf.shape[0] // 2, otf.shape[1] // 2)
        otf0 = np.abs(otf[center[0], center[1]])
    else:
        otf0 = np.abs(otf[len(otf) // 2])
    otf0 = max(otf0, 1e-30)
    return np.abs(otf) / otf0


def compute_wavefront_variance(phase, mask):
    """
    计算残余波前方差 sigma_phi^2 (去除piston和tilt).
    """
    phi = phase[mask]
    if len(phi) == 0:
        return 0.0
    phi = phi - np.mean(phi)
    # 去除tilt
    coords = np.argwhere(mask)
    if len(coords) > 2:
        x = coords[:, 1] - np.mean(coords[:, 1])
        y = coords[:, 0] - np.mean(coords[:, 0])
        A = np.column_stack([x, y])
        tilt, _, _, _ = np.linalg.lstsq(A, phi, rcond=None)
        phi = phi - A @ tilt
    return float(np.var(phi))


def compute_encircled_energy(psf, x_coords, radius):
    """
    计算包围在指定半径内的能量比例.

    EE(r) = sum_{|r'| <= r} PSF(r') / sum PSF
    """
    if radius < 0:
        raise ValueError("radius must be non-negative.")
    N = psf.shape[0]
    center = N // 2
    dx = x_coords[1] - x_coords[0] if len(x_coords) > 1 else 1.0
    total = np.sum(psf)
    if total < 1e-30:
        return 0.0

    yy, xx = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    r_pix = np.sqrt((xx - center) ** 2 + (yy - center) ** 2) * abs(dx)
    encircled = np.sum(psf[r_pix <= radius])
    return float(encircled / total)


def tetrahedral_phase_moments(phase, mask, max_order=3):
    """
    计算相位在瞳孔区域的前几阶空间矩,
    使用四面体积分思想对不规则瞳孔近似.

    矩定义:
      M_{p,q} = integral_{pupil} x^p y^q phi(x,y) dA / integral_{pupil} dA
    """
    if max_order < 0:
        raise ValueError("max_order must be non-negative.")
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return {}

    y_idx = coords[:, 0]
    x_idx = coords[:, 1]
    phi_vals = phase[mask]
    area = len(phi_vals)

    moments = {}
    for p in range(max_order + 1):
        for q in range(max_order + 1 - p):
            if p == 0 and q == 0:
                moments[(p, q)] = float(np.mean(phi_vals))
            else:
                x_norm = (x_idx - np.mean(x_idx)) / max(np.std(x_idx), 1.0)
                y_norm = (y_idx - np.mean(y_idx)) / max(np.std(y_idx), 1.0)
                moments[(p, q)] = float(np.mean((x_norm ** p) * (y_norm ** q) * phi_vals))

    return moments
