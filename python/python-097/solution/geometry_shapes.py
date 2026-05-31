"""
geometry_shapes.py

微波器件几何形状定义与材料分布模块。
融合circle_segment的圆扇形几何计算与human_data的轮廓参数化思想，
用于定义具有复杂边界的微波谐振腔结构。

核心几何模型:
--------------
1. 圆柱形谐振腔: 半径R，高度H
2. 同轴腔体: 内半径a，外半径b
3. 圆环形加载: 基于circle_segment的面积与高度计算
4. 复杂轮廓参数化: 使用离散点定义器件边界
"""

import numpy as np


class CylindricalCavity:
    """
    圆柱形微波谐振腔。

    支持TE和TM模式的理论截止频率计算:
    - TE_mnp: f = c/(2π) · sqrt((x'_{mn}/R)² + (pπ/H)²)
    - TM_mnp: f = c/(2π) · sqrt((x_{mn}/R)² + (pπ/H)²)
    其中 x_{mn} 是J_m(x)的第n个零点，x'_{mn}是J'_m(x)的第n个零点。
    """

    def __init__(self, radius, height, epsilon_r=1.0, mu_r=1.0, sigma=0.0):
        """
        Parameters
        ----------
        radius, height : float
            腔体半径和高度 [m]
        epsilon_r, mu_r : float
            相对介电常数和相对磁导率
        sigma : float
            电导率 [S/m]
        """
        if radius <= 0 or height <= 0:
            raise ValueError("半径和高度必须为正")
        self.radius = radius
        self.height = height
        self.epsilon_r = epsilon_r
        self.mu_r = mu_r
        self.sigma = sigma

    def volume(self):
        """腔体体积 V = πR²H"""
        return np.pi * self.radius ** 2 * self.height

    def surface_area(self):
        """内表面积 A = 2πR² + 2πRH"""
        return 2.0 * np.pi * self.radius ** 2 + 2.0 * np.pi * self.radius * self.height

    def te_cutoff_frequency(self, m, n, p, c=3e8):
        """
        TE_{mnp}模式截止频率近似。
        使用Bessel函数导数的零点近似值。
        """
        # J'_m(x)的前几个零点近似表
        j_prime_zeros = {
            (0, 1): 3.8317, (0, 2): 7.0156,
            (1, 1): 1.8412, (1, 2): 5.3314,
            (2, 1): 3.0542, (2, 2): 6.7061,
        }
        key = (m, n)
        if key not in j_prime_zeros:
            # 使用渐近公式 x'_{mn} ≈ (m + 2n - 1/2)π/2
            x_mn = (m + 2 * n - 0.5) * np.pi / 2.0
        else:
            x_mn = j_prime_zeros[key]

        k_r = x_mn / self.radius
        k_z = p * np.pi / self.height
        k = np.sqrt(k_r ** 2 + k_z ** 2)
        return c * k / (2.0 * np.pi)

    def tm_cutoff_frequency(self, m, n, p, c=3e8):
        """
        TM_{mnp}模式截止频率近似。
        使用Bessel函数的零点近似值。
        """
        j_zeros = {
            (0, 1): 2.4048, (0, 2): 5.5201,
            (1, 1): 3.8317, (1, 2): 7.0156,
            (2, 1): 5.1356, (2, 2): 8.4172,
        }
        key = (m, n)
        if key not in j_zeros:
            x_mn = (m + 2 * n - 0.25) * np.pi
        else:
            x_mn = j_zeros[key]

        k_r = x_mn / self.radius
        k_z = p * np.pi / self.height
        k = np.sqrt(k_r ** 2 + k_z ** 2)
        return c * k / (2.0 * np.pi)

    def is_inside(self, x, y, z, center=(0, 0, 0)):
        """
        判断点(x,y,z)是否在圆柱腔内。
        考虑以center为底面中心的圆柱。
        """
        cx, cy, cz = center
        r_sq = (x - cx) ** 2 + (y - cy) ** 2
        r = np.sqrt(r_sq)
        in_radius = r <= self.radius
        in_height = (z >= cz) & (z <= cz + self.height)
        return in_radius & in_height


class CoaxialCavity:
    """
    同轴谐振腔。
    内导体半径a，外导体半径b，长度L。
    """

    def __init__(self, a, b, length, epsilon_r=1.0):
        if a <= 0 or b <= a or length <= 0:
            raise ValueError("几何参数必须满足: 0 < a < b, length > 0")
        self.a = a
        self.b = b
        self.length = length
        self.epsilon_r = epsilon_r

    def characteristic_impedance(self, mu_r=1.0):
        """特性阻抗 Z₀ = (η₀/2π) · sqrt(μ_r/ε_r) · ln(b/a)"""
        from physics_constants import ETA_0
        return ETA_0 / (2.0 * np.pi) * np.sqrt(mu_r / self.epsilon_r) * np.log(self.b / self.a)

    def capacitance_per_unit_length(self, epsilon_0=8.854e-12):
        """单位长度电容 C' = 2πε / ln(b/a)"""
        return 2.0 * np.pi * epsilon_0 * self.epsilon_r / np.log(self.b / self.a)

    def is_inside(self, x, y, z, center=(0, 0, 0)):
        cx, cy, cz = center
        r_sq = (x - cx) ** 2 + (y - cy) ** 2
        r = np.sqrt(r_sq)
        in_radial = (r >= self.a) & (r <= self.b)
        in_axial = (z >= cz) & (z <= cz + self.length)
        return in_radial & in_axial


class CircleSegmentDielectric:
    """
    圆扇形/圆段介质加载（基于circle_segment项目）。
    在谐振腔中引入圆段形状的介质块，用于模式调谐。
    """

    def __init__(self, r, theta, height, epsilon_r=10.0, center=(0, 0)):
        """
        Parameters
        ----------
        r : float
            圆段所在圆的半径
        theta : float
            圆段角度 [rad]
        height : float
            介质块高度
        epsilon_r : float
            相对介电常数
        center : tuple
            圆心位置 (x, y)
        """
        self.r = r
        self.theta = theta
        self.height = height
        self.epsilon_r = epsilon_r
        self.center = center

    def area(self):
        """圆段面积 A = r²(θ - sinθ)/2"""
        return self.r ** 2 * (self.theta - np.sin(self.theta)) / 2.0

    def centroid(self):
        """
        圆段形心位置（相对于圆心）。
        距离 d = (4r sin³(θ/2)) / (3(θ - sinθ))
        """
        if self.theta < 1e-10:
            return (0.0, 0.0)
        d = 4.0 * self.r * (np.sin(self.theta / 2.0)) ** 3 / (3.0 * (self.theta - np.sin(self.theta)))
        # 假设圆段关于x轴对称
        return (d, 0.0)

    def is_inside(self, x, y, z, z_bottom=0.0):
        """判断点是否在圆段介质块内。"""
        cx, cy = self.center
        dx = x - cx
        dy = y - cy
        r_p = np.sqrt(dx ** 2 + dy ** 2)
        in_radius = r_p <= self.r
        # 角度限制: -θ/2 ≤ atan2(dy,dx) ≤ θ/2
        angle = np.arctan2(dy, dx)
        in_angle = np.abs(angle) <= self.theta / 2.0
        in_height = (z >= z_bottom) & (z <= z_bottom + self.height)
        return in_radius & in_angle & in_height


class ParametricProfile:
    """
    参数化轮廓形状（基于human_data的轮廓离散化思想）。
    使用离散点定义微波器件的异形边界，如渐变线、波纹壁等。
    """

    def __init__(self, profile_points, symmetry='axisymmetric'):
        """
        Parameters
        ----------
        profile_points : list of tuple
            [(r0,z0), (r1,z1), ...] 轮廓点序列
        symmetry : str
            'axisymmetric' 或 'planar'
        """
        if len(profile_points) < 3:
            raise ValueError("轮廓至少需要3个点")
        self.profile_points = np.array(profile_points)
        self.symmetry = symmetry

    def radius_at_z(self, z):
        """在高度z处通过线性插值得到半径。"""
        zs = self.profile_points[:, 1]
        rs = self.profile_points[:, 0]
        if z < zs.min() or z > zs.max():
            return 0.0
        return np.interp(z, zs, rs)

    def is_inside(self, x, y, z):
        """判断点是否在参数化轮廓内部。"""
        r_max = self.radius_at_z(z)
        r = np.sqrt(x ** 2 + y ** 2)
        return r <= r_max


def create_corrugated_wall_profile(R_base, depth, period, n_periods, z_start=0.0):
    """
    创建波纹壁谐振腔的轮廓（如回旋管腔体）。

    Parameters
    ----------
    R_base : float
        基础半径
    depth : float
        波纹深度
    period : float
        波纹周期
    n_periods : int
        波纹数量
    z_start : float
        起始z坐标

    Returns
    -------
    ParametricProfile
    """
    points = []
    n_points_per_period = 8
    for i in range(n_periods * n_points_per_period + 1):
        t = i / n_points_per_period
        z = z_start + t * period
        # 余弦波纹
        r = R_base + depth * np.cos(2.0 * np.pi * t)
        r = max(r, 0.01 * R_base)  # 保证正半径
        points.append((r, z))
    return ParametricProfile(points, symmetry='axisymmetric')


def assign_material_properties(grid, shapes):
    """
    根据几何形状列表为网格分配材料属性。

    Parameters
    ----------
    grid : YeeGrid3D 或 CylindricalYeeGrid
    shapes : list
        几何形状对象列表，每个对象需要有 epsilon_r, mu_r, sigma 属性

    Returns
    -------
    epsilon, mu, sigma : ndarray
        材料属性场
    """
    from physics_constants import EPSILON_0, MU_0

    if hasattr(grid, 'X'):
        nx, ny, nz = grid.nx, grid.ny, grid.nz
        epsilon = np.ones((nx, ny, nz)) * EPSILON_0
        mu = np.ones((nx, ny, nz)) * MU_0
        sigma = np.zeros((nx, ny, nz))
        X, Y, Z = grid.X, grid.Y, grid.Z
    else:
        nr, nz = grid.nr, grid.nz
        epsilon = np.ones((nr, nz)) * EPSILON_0
        mu = np.ones((nr, nz)) * MU_0
        sigma = np.zeros((nr, nz))
        X = grid.R_grid
        Y = np.zeros_like(X)
        Z = grid.Z_grid

    for shape in shapes:
        if hasattr(shape, 'is_inside'):
            mask = shape.is_inside(X, Y, Z)
            if hasattr(shape, 'epsilon_r'):
                epsilon[mask] = shape.epsilon_r * EPSILON_0
            if hasattr(shape, 'mu_r'):
                mu[mask] = shape.mu_r * MU_0
            if hasattr(shape, 'sigma'):
                sigma[mask] = shape.sigma

    return epsilon, mu, sigma
