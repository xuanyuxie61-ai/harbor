"""
辐射传输模块：基于 image_contrast 的对比度增强思想，
模拟冠层内光环境的异质性（光斑与阴影）。

核心公式：
  Beer-Lambert 衰减定律：
      I(z) = I_0 * exp(-k * L(z))
  其中 L(z) = integral_0^z LAI(z') dz' 为累计叶面积指数。

  光环境异质性对比度修正（基于 image_contrast_gray 思想）：
      I_enhanced(x,y,z) = s * I(x,y,z) + (1-s) * I_avg_neighbors(x,y,z)
  其中 s 为锐化因子，I_avg 为邻域平均光强。

  散射辐射修正（采用 Campbell 椭球模型）：
      I_diffuse(z) = I_d0 * exp(-k_d * L(z))
      k_d = 0.71 * G / sin(beta)
"""
import numpy as np


def beer_lambert_irradiance(i0, k_ext, cumulative_lai):
    """
    Beer-Lambert 光强衰减。
    i0: 冠层上方入射光强 (W/m^2)
    k_ext: 消光系数
    cumulative_lai: 累计 LAI
    """
    cum_lai = np.asarray(cumulative_lai, dtype=float)
    return i0 * np.exp(-k_ext * np.maximum(cum_lai, 0.0))


def contrast_enhance_radiation(irradiance_grid, sharpness=1.3):
    """
    对冠层内光强场应用对比度增强，模拟光斑-阴影异质性。
    irradiance_grid: 2D numpy array
    sharpness: s > 1 增强对比度
    """
    gray = np.asarray(irradiance_grid, dtype=float)
    m, n = gray.shape
    if m < 3 or n < 3:
        return gray

    # 计算8邻域平均
    avg = np.zeros_like(gray)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            avg += np.roll(np.roll(gray, di, axis=0), dj, axis=1)
    avg /= 8.0

    # 边界恢复原始值
    enhanced = sharpness * gray + (1.0 - sharpness) * avg
    enhanced[0, :] = gray[0, :]
    enhanced[-1, :] = gray[-1, :]
    enhanced[:, 0] = gray[:, 0]
    enhanced[:, -1] = gray[:, -1]
    enhanced = np.clip(enhanced, 0.0, None)
    return enhanced


def campbell_ellipsoid_g(theta_s, chi=1.0):
    """
    Campbell 椭球体投影函数 G(theta_s)。
    theta_s: 太阳天顶角 (rad)
    chi: 叶片分布参数 (1=球形, <1=直立, >1=水平)
    """
    cos_t = np.cos(theta_s)
    sin_t = np.sin(theta_s)
    if chi == 1.0:
        return 0.5 / max(cos_t, 1e-6)
    # 更一般形式
    psi = np.arccos(np.clip(chi * cos_t / np.sqrt(np.maximum(chi ** 2 * cos_t ** 2 + sin_t ** 2, 1e-14)), -1.0, 1.0))
    g = (chi + (sin_t / max(cos_t, 1e-6)) * psi) / (chi + 1.0)
    return g


def compute_canopy_radiation_profile(z_levels, i0_dir, i0_diff, k_dir, k_diff, lai_profile):
    """
    计算冠层内直射与散射辐射剖面。
    z_levels: 高度层 (m)
    lai_profile: 各层的 LAI 值 (m^2/m^2 per layer)
    返回: dict with 'direct', 'diffuse', 'total'
    """
    z = np.asarray(z_levels, dtype=float)
    lai = np.asarray(lai_profile, dtype=float)
    cum_lai = np.cumsum(lai)

    i_dir = beer_lambert_irradiance(i0_dir, k_dir, cum_lai)
    i_diff = beer_lambert_irradiance(i0_diff, k_diff, cum_lai)
    i_total = i_dir + i_diff

    return {
        'direct': i_dir,
        'diffuse': i_diff,
        'total': i_total,
        'cumulative_lai': cum_lai,
        'z': z
    }


def radiation_2d_grid(grid_points, vertices, i0, k_ext, lai_max, canopy_height,
                      sharpness=1.3, resolution=30):
    """
    在冠层二维截面上构建辐射场网格。
    grid_points: (N,2)
    返回: 2D 辐射场数组及对应坐标网格。
    """
    # 构建规则网格用于 contrast 计算
    x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
    y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
    xs = np.linspace(x_min, x_max, resolution)
    ys = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(xs, ys)

    # 计算各网格点的累计 LAI（简化：仅基于高度）
    I = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            z = Y[i, j]
            if z <= 0 or z >= canopy_height:
                I[i, j] = i0
            else:
                cum_lai = lai_max * (z / canopy_height)
                I[i, j] = i0 * np.exp(-k_ext * cum_lai)
    I_enhanced = contrast_enhance_radiation(I, sharpness=sharpness)
    return X, Y, I_enhanced
