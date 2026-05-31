"""
shack_hartmann_sensor.py — Shack-Hartmann波前传感器模拟

融合原项目: 932_pyramid_grid (金字塔网格结构化采样)

功能:
  - 子孔径网格划分 (方形/金字塔型)
  - 每个子孔径内质心偏移计算 (x-和y-斜率)
  - 噪声模型 (光子噪声、读出噪声)
  - 斜率向量构建

物理模型:
  1. 子孔径斜率与局部波前梯度的关系:
       s_x(i,j) = (1/A_sub) * integral_{subap} dW/dx  dA
       s_y(i,j) = (1/A_sub) * integral_{subap} dW/dy  dA

  2. 质心算法 (Center of Gravity, COG):
       x_c = sum_{pixels} x * I(x,y) / sum I(x,y)
       y_c = sum_{pixels} y * I(x,y) / sum I(x,y)
     参考质心 (x_0, y_0) 由平面波标定得到,
     斜率正比于偏移: s_x = (x_c - x_0) / f_lens

  3. 噪声模型:
       光子噪声: sigma_photon ~ 1/sqrt(N_photons)
       读出噪声: sigma_read  ~ electrons/pixel (高斯)
"""

import numpy as np


def generate_subaperture_grid(grid_size, n_subap, geometry='square'):
    """
    生成子孔径网格.

    geometry:
      'square': 方形网格
      'pyramid': 金字塔型分层网格 (源自932_pyramid_grid)

    返回:
      subaps: list of (row_start, row_end, col_start, col_end) 元组
    """
    if grid_size < n_subap:
        raise ValueError("grid_size must be >= n_subap.")
    if n_subap < 1:
        raise ValueError("n_subap must be >= 1.")

    subaps = []
    if geometry == 'square':
        step = grid_size // n_subap
        for i in range(n_subap):
            for j in range(n_subap):
                rs = i * step
                re = min((i + 1) * step, grid_size)
                cs = j * step
                ce = min((j + 1) * step, grid_size)
                subaps.append((rs, re, cs, ce))
    elif geometry == 'pyramid':
        # 金字塔型: 中心密、外围疏
        for k in range(n_subap, 0, -1):
            r_layer = int(grid_size * k / (2 * n_subap))
            if r_layer <= 0:
                continue
            n_pts = max(k, 2)
            centers = np.linspace(-r_layer, r_layer, n_pts)
            for cx in centers:
                for cy in centers:
                    cx_i = int(cx + grid_size / 2)
                    cy_i = int(cy + grid_size / 2)
                    half = max(grid_size // (4 * n_subap), 1)
                    rs = max(cx_i - half, 0)
                    re = min(cx_i + half, grid_size)
                    cs = max(cy_i - half, 0)
                    ce = min(cy_i + half, grid_size)
                    if re > rs and ce > cs:
                        subaps.append((rs, re, cs, ce))
    else:
        raise ValueError("geometry must be 'square' or 'pyramid'.")

    return subaps


def compute_subaperture_slopes(phase, subaps, pixel_scale, focal_length=0.1,
                                noise_photon=0.01, noise_read=0.5,
                                reference_slopes=None):
    """
    计算每个子孔径的x-和y-斜率.

    方法: 在子孔径区域内计算相位梯度平均值,
          然后叠加光子噪声和读出噪声.

    斜率公式:
      s_x = mean( d(phase)/dx ) + noise
      s_y = mean( d(phase)/dy ) + noise
    """
    if pixel_scale <= 0 or focal_length <= 0:
        raise ValueError("pixel_scale and focal_length must be positive.")
    if noise_photon < 0 or noise_read < 0:
        raise ValueError("noise parameters must be non-negative.")

    n = len(subaps)
    sx = np.zeros(n, dtype=np.float64)
    sy = np.zeros(n, dtype=np.float64)

    # 数值梯度
    dphidx = np.zeros_like(phase)
    dphidy = np.zeros_like(phase)
    dphidx[:, 1:-1] = (phase[:, 2:] - phase[:, :-2]) / (2.0 * pixel_scale)
    dphidy[1:-1, :] = (phase[2:, :] - phase[:-2, :]) / (2.0 * pixel_scale)

    for idx, (rs, re, cs, ce) in enumerate(subaps):
        patch_x = dphidx[rs:re, cs:ce]
        patch_y = dphidy[rs:re, cs:ce]
        if patch_x.size == 0:
            sx[idx] = 0.0
            sy[idx] = 0.0
            continue

        sx[idx] = np.mean(patch_x)
        sy[idx] = np.mean(patch_y)

    # 加噪声
    if noise_photon > 0:
        sx += np.random.normal(0, noise_photon, n)
        sy += np.random.normal(0, noise_photon, n)
    if noise_read > 0:
        sx += np.random.normal(0, noise_read / focal_length, n)
        sy += np.random.normal(0, noise_read / focal_length, n)

    if reference_slopes is not None:
        if len(reference_slopes) == 2 * n:
            sx -= reference_slopes[:n]
            sy -= reference_slopes[n:]

    return sx, sy


def slopes_to_vector(sx, sy):
    """
    将sx, sy合并为单一斜率向量.
    """
    if len(sx) != len(sy):
        raise ValueError("sx and sy must have the same length.")
    return np.concatenate([sx, sy])


def vector_to_slopes(svec):
    """
    将斜率向量拆分为sx, sy.
    """
    n = len(svec) // 2
    return svec[:n], svec[n:]


def build_slope_geometry_matrix(subaps, grid_size, pixel_scale):
    """
    构建斜率-几何关系矩阵 G.

    对于每个子孔径 k, 其x-斜率近似为:
      s_x(k) = (1/N_k) * sum_{i in subap_k} (phi_{i+1,j} - phi_{i-1,j}) / (2*dx)
    该矩阵将相位向量 phi 映射到斜率向量 s = G @ phi.
    """
    n = len(subaps)
    Npix = grid_size * grid_size
    Gx = np.zeros((n, Npix), dtype=np.float64)
    Gy = np.zeros((n, Npix), dtype=np.float64)

    for k, (rs, re, cs, ce) in enumerate(subaps):
        patch_npix = (re - rs) * (ce - cs)
        if patch_npix == 0:
            continue
        factor = 1.0 / (2.0 * pixel_scale * patch_npix)

        for i in range(rs, re):
            for j in range(cs, ce):
                idx = i * grid_size + j
                if j + 1 < grid_size:
                    Gx[k, idx + 1] += factor
                if j - 1 >= 0:
                    Gx[k, idx - 1] -= factor
                if i + 1 < grid_size:
                    Gy[k, idx + grid_size] += factor
                if i - 1 >= 0:
                    Gy[k, idx - grid_size] -= factor

    G = np.vstack([Gx, Gy])
    return G


def estimate_photon_noise(signal_photons, n_pixels, read_noise_e=3.0, QE=0.8):
    """
    估计子孔径质心探测的总噪声 (以弧度为单位).

    总电子噪声:
      sigma_total^2 = sigma_photon^2 + sigma_read^2
      sigma_photon = sqrt(N_photons) / QE
      sigma_read   = n_pixels * read_noise_e
    """
    if signal_photons < 0 or n_pixels < 1:
        raise ValueError("Invalid photon noise parameters.")
    sigma_photon = np.sqrt(max(signal_photons, 1.0)) / max(QE, 1e-6)
    sigma_read = n_pixels * read_noise_e
    sigma_total = np.sqrt(sigma_photon ** 2 + sigma_read ** 2)
    # 转换为相位斜率噪声 (近似)
    noise_rad = 1.0 / max(sigma_total, 1.0)
    return noise_rad
