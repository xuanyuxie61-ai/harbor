"""
wavefront_propagation.py — 光波在湍流中的传播与焦散奇点分析

融合原项目:
  - 140_caustic (焦散线几何生成)
  - 932_pyramid_grid (金字塔网格结构化采样)

功能:
  - 菲涅尔/夫琅禾费衍射传播
  - 光程差 (OPD) 到复振幅的转换
  - 焦散奇点检测 (基于Hessian行列式)
  - 金字塔型光线追迹网格

物理模型:
  1. 菲涅尔衍射积分:
       U(x,y,z) = (exp(ikz)/(i*lambda*z)) * integral integral
                  U0(x',y',0) * exp(ik/2z * [(x-x')^2 + (y-y')^2]) dx'dy'

  2. 焦散条件 (源自140_caustic):
       在几何光学极限下, 焦散面是光线包络.
       对于波前 W(x,y), 局部曲率张量 H = nabla^2 W.
       焦散出现在 det(H) 改变符号的位置.

  3. 波前奇点:
       相位奇点 ( optical vortices ) 满足:
         Re{U} = 0  and  Im{U} = 0
       拓扑电荷:
         q = (1/2pi) * contour_integral d(arg(U))
"""

import numpy as np


def opd_to_complex_amplitude(phase, wavelength):
    """
    将相位屏 (OPD) 转换为复振幅.

    U(x,y) = A0 * exp(i * phi(x,y))
    其中 phi = (2pi/lambda) * OPD
    """
    if wavelength <= 0:
        raise ValueError("wavelength must be positive.")
    k = 2.0 * np.pi / wavelength
    amplitude = np.exp(1j * k * phase)
    return amplitude


def fresnel_propagation(amplitude_in, pixel_scale, z, wavelength):
    """
    使用FFT-based菲涅尔传播计算衍射场.

    传递函数 (频域):
       H(fx, fy) = exp(i*k*z) * exp(-i*pi*lambda*z*(fx^2 + fy^2))

    算法:
       U_out = IFFT[ FFT(U_in) * H ]
    """
    if z < 0:
        raise ValueError("Propagation distance z must be non-negative.")
    if pixel_scale <= 0:
        raise ValueError("pixel_scale must be positive.")
    if wavelength <= 0:
        raise ValueError("wavelength must be positive.")

    N = amplitude_in.shape[0]
    if N != amplitude_in.shape[1]:
        raise ValueError("Input amplitude must be square.")

    freq = np.fft.fftfreq(N, d=pixel_scale)
    fx, fy = np.meshgrid(freq, freq)
    k = 2.0 * np.pi / wavelength

    H = np.exp(1j * k * z) * np.exp(-1j * np.pi * wavelength * z * (fx ** 2 + fy ** 2))

    U_fft = np.fft.fft2(amplitude_in)
    U_out = np.fft.ifft2(U_fft * H)
    return U_out


def compute_wavefront_curvature_hessian(phase, pixel_scale):
    """
    计算波前的Hessian矩阵分量.

    对于波前 W(x,y), Hessian:
       H = [[W_xx, W_xy],
            [W_xy, W_yy]]

    使用中心差分:
       W_xx(i,j) = (W(i+1,j) - 2W(i,j) + W(i-1,j)) / dx^2
       W_yy(i,j) = (W(i,j+1) - 2W(i,j) + W(i,j-1)) / dy^2
       W_xy(i,j) = (W(i+1,j+1) - W(i+1,j-1) - W(i-1,j+1) + W(i-1,j-1)) / (4*dx*dy)
    """
    if pixel_scale <= 0:
        raise ValueError("pixel_scale must be positive.")
    dx2 = pixel_scale ** 2
    W = phase

    W_xx = np.zeros_like(W)
    W_yy = np.zeros_like(W)
    W_xy = np.zeros_like(W)

    W_xx[1:-1, 1:-1] = (W[2:, 1:-1] - 2.0 * W[1:-1, 1:-1] + W[:-2, 1:-1]) / dx2
    W_yy[1:-1, 1:-1] = (W[1:-1, 2:] - 2.0 * W[1:-1, 1:-1] + W[1:-1, :-2]) / dx2
    W_xy[1:-1, 1:-1] = (
        W[2:, 2:] - W[2:, :-2] - W[:-2, 2:] + W[:-2, :-2]
    ) / (4.0 * pixel_scale ** 2)

    return W_xx, W_yy, W_xy


def detect_caustic_singularities(phase, pixel_scale, mask):
    """
    检测波前焦散奇点.

    焦散条件: det(H) = W_xx * W_yy - W_xy^2 < 0 且在该区域符号发生变化.
    返回奇点掩码和Hessian行列式.
    """
    W_xx, W_yy, W_xy = compute_wavefront_curvature_hessian(phase, pixel_scale)
    det_H = W_xx * W_yy - W_xy ** 2

    # 焦散区域: det(H) < 0 (鞍型曲率)
    caustic_region = (det_H < 0) & mask

    # 奇点: det(H) 的局部极小值 (拉普拉斯检测)
    laplacian = np.zeros_like(det_H)
    laplacian[1:-1, 1:-1] = (
        det_H[2:, 1:-1] + det_H[:-2, 1:-1] + det_H[1:-1, 2:] + det_H[1:-1, :-2]
        - 4.0 * det_H[1:-1, 1:-1]
    )

    singularity = caustic_region & (laplacian > 0) & mask
    return singularity, det_H


def caustic_line_density(n, m, num_points=500):
    """
    生成圆内焦散线网络 (源自140_caustic的几何思想).

    在单位圆上取 n 个等距点 z_j = exp(2*pi*i*j/n),
    连接 z_{j+1} 与 z_{mod(j*m, n)+1}.

    在AO中, 这些线段模拟了强湍流下光波前折叠形成的焦散网络.
    """
    if n < 3:
        raise ValueError("n must be >= 3.")
    if m < 1:
        raise ValueError("m must be >= 1.")

    theta = np.linspace(0, 2.0 * np.pi, n + 1)[:-1]
    z = np.exp(1j * theta)

    lines = []
    for j in range(n):
        j_next = (j + 1) % n
        j_conn = ((j * m) % n)
        p1 = np.array([z[j_next].real, z[j_next].imag])
        p2 = np.array([z[j_conn].real, z[j_conn].imag])
        lines.append((p1, p2))

    return lines


def pyramid_ray_tracing_grid(n_layers, aperture_radius=1.0):
    """
    金字塔型光线追迹网格生成 (源自932_pyramid_grid).

    在单位四棱锥上生成结构化光线网格:
      - 第 k 层高度: z = k / n
      - 每层在xy平面上的范围随高度线性收缩
      - 用于模拟金字塔波前传感器的光线几何

    返回: rays 列表, 每个元素为 (origin, direction) 元组.
    """
    if n_layers < 1:
        raise ValueError("n_layers must be >= 1.")

    rays = []
    for k in range(n_layers, -1, -1):
        z = k / n_layers
        r_layer = aperture_radius * (1.0 - z)
        if r_layer <= 0:
            continue
        n_pts = max(2 * k + 1, 3)
        x_vals = np.linspace(-r_layer, r_layer, n_pts)
        y_vals = np.linspace(-r_layer, r_layer, n_pts)
        for xv in x_vals:
            for yv in y_vals:
                origin = np.array([xv, yv, z])
                direction = np.array([0.0, 0.0, 1.0])
                rays.append((origin, direction))
    return rays


def compute_strehl_ratio(phase_corrected, wavelength, mask):
    """
    计算Strehl比.

    Strehl比定义为校正后峰值强度与理想衍射极限峰值强度之比.
    近似 (Marechal近似):
       S = exp(-sigma_phi^2)
    其中 sigma_phi^2 为残余相位方差.

    精确计算:
       S = | integral_{pupil} exp(i*phi) dA |^2 / A^2
    """
    if wavelength <= 0:
        raise ValueError("wavelength must be positive.")
    phi = phase_corrected[mask]
    if len(phi) == 0:
        return 0.0
    area = np.sum(mask)
    integral = np.sum(np.exp(1j * phi))
    S = (np.abs(integral) ** 2) / (area ** 2)
    return float(S)


def compute_modulation_transfer_function(phase, wavelength, pixel_scale, mask):
    """
    计算调制传递函数 (MTF).

    MTF(f) = | OTF(f) | / | OTF(0) |
    光学传递函数 OTF 为瞳孔函数自相关:
       OTF(f) = integral P(r) P*(r - lambda*z*f) dr
    在离散域通过自相关计算.
    """
    if wavelength <= 0 or pixel_scale <= 0:
        raise ValueError("wavelength and pixel_scale must be positive.")

    N = phase.shape[0]
    k = 2.0 * np.pi / wavelength
    pupil = np.zeros_like(phase, dtype=np.complex128)
    pupil[mask] = np.exp(1j * k * phase[mask])

    # 自相关
    otf = np.fft.ifft2(np.abs(np.fft.fft2(pupil)) ** 2)
    otf = np.fft.fftshift(otf)
    otf_max = np.max(np.abs(otf))
    if otf_max < 1e-30:
        return np.zeros_like(otf)
    mtf = np.abs(otf) / otf_max
    return mtf
