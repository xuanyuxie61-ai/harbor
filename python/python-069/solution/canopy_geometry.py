"""
冠层几何结构模块：基于 polygon_grid 思想生成树冠多边形网格，
并构建叶面积指数（LAI）的空间分布。

核心公式：
  冠层多边形由 n 个顶点定义，重心为
      G = (1/n_v) * sum(v_i)
  每个三角形单元的重心坐标为
      P = (i*v_l + j*v_{l+1} + k*G) / n,  i+j+k=n
  LAI 分布服从 Beta 函数描述的垂直剖面：
      LAI(z) = LAI_max * (z/h)^{alpha-1} * (1 - z/h)^{beta-1} / B(alpha, beta)
"""
import numpy as np
from scipy.special import beta as beta_func


def polygon_grid_points(n, vertices):
    """
    生成多边形内部网格点。
    n: 每个边的细分段数
    vertices: (nv, 2) 多边形顶点
    返回: (ng, 2) 网格点坐标
    """
    nv = vertices.shape[0]
    ng = 1 + nv * n * (n + 1) // 2
    xg = np.zeros((ng, 2), dtype=float)
    p = 0
    vc = np.mean(vertices, axis=0)
    xg[p, :] = vc
    p += 1
    for l in range(nv):
        lp1 = (l + 1) % nv
        for i in range(1, n + 1):
            for j in range(0, n - i + 1):
                k = n - i - j
                if p < ng:
                    xg[p, :] = (i * vertices[l, :] + j * vertices[lp1, :] + k * vc) / n
                    p += 1
    return xg[:p, :]


def canopy_lai_profile(z, h, lai_max, alpha=2.0, beta=2.5):
    """
    冠层 LAI 垂直剖面（归一化 Beta 分布）。
    z: 高度 (m)
    h: 冠层总高 (m)
    """
    if z <= 0 or z >= h:
        return 0.0
    t = z / h
    norm = beta_func(alpha, beta)
    if norm < 1e-15:
        return 0.0
    val = lai_max * (t ** (alpha - 1.0)) * ((1.0 - t) ** (beta - 1.0)) / norm
    return float(val)


def build_canopy_grid(canopy_height=20.0, crown_radius=5.0, n_sub=8, lai_max=4.5):
    """
    构建二维冠层截面网格（垂直剖面）。
    返回: grid_points (N,2), lai_values (N,), 以及多边形顶点。
    """
    # 定义树冠轮廓多边形：六边形近似树冠横截面
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    vertices = np.zeros((6, 2), dtype=float)
    vertices[:, 0] = crown_radius * np.cos(angles)
    vertices[:, 1] = canopy_height + crown_radius * np.sin(angles) * 0.6
    # 确保在合理范围内
    vertices[:, 1] = np.clip(vertices[:, 1], 0.0, canopy_height * 1.2)

    grid = polygon_grid_points(n_sub, vertices)
    lai_vals = np.array([canopy_lai_profile(pt[1], canopy_height, lai_max) for pt in grid],
                        dtype=float)
    return grid, lai_vals, vertices


def canopy_volume_lai_3d(canopy_height=20.0, crown_radius=5.0, n_vert=20, n_horiz=20,
                         lai_max=4.5):
    """
    三维冠层 LAI 场：在柱坐标系下采样。
    返回: (x, y, z, lai) 数组。
    """
    z = np.linspace(0.1, canopy_height * 0.99, n_vert)
    r = np.linspace(0.0, crown_radius, n_horiz)
    theta = np.linspace(0, 2 * np.pi, n_horiz, endpoint=False)
    points = []
    lais = []
    for zi in z:
        lai_z = canopy_lai_profile(zi, canopy_height, lai_max)
        for ri in r:
            for ti in theta:
                x = ri * np.cos(ti)
                y = ri * np.sin(ti)
                # 径向衰减：越靠近边缘 LAI 越低
                decay = max(0.0, 1.0 - (ri / crown_radius) ** 2)
                points.append([x, y, zi])
                lais.append(lai_z * decay)
    return np.array(points, dtype=float), np.array(lais, dtype=float)
