# -*- coding: utf-8 -*-
"""
cvt_sampler.py
质心Voronoi镶嵌采样与多维积分

本模块实现核pasta相结构优化中的采样与积分方法.
融入算法:
- cvt_2d_lumping (247): 2D Lloyd算法, CVT生成
- sphere_cvt (1113): 球面CVT, Delaunay三角化, Voronoi面积计算
- test_int_nd (1209): N维数值积分测试函数

核心物理公式:
1. CVT能量泛函:
   G(X) = sum_i integral_{Omega_i} rho(x) |x - x_i|^2 dx
   
   其中Omega_i为Voronoi区域, x_i为生成点.
   
2. Lloyd算法:
   x_i^{new} = integral_{Omega_i} rho(x) x dx / integral_{Omega_i} rho(x) dx
   
3. 球面Voronoi面积 (球面多边形):
   A = sum_i alpha_i - (n-2)*pi
   
   其中alpha_i为球面多边形的内角.
   
4. N维高斯积分 (test_int_nd):
   I = integral_{R^d} exp(-sum(x_i^2)) dx = pi^{d/2}
   
5. 中子星crust密度分布:
   rho(r) = rho_c * (1 - r/R)^{nu}
   
   其中rho_c为核心密度, R为星半径, nu为指数.
"""

import numpy as np


def cvt_2d_lumping(n_generators, it_num, s_num, density_func):
    """
    2D Lloyd算法生成CVT (来自247_cvt_2d_lumping).
    
    输入:
        n_generators: 生成点数量
        it_num: 迭代次数
        s_num: 采样分辨率
        density_func: 密度函数 func(x,y) -> scalar
    输出:
        generators: (n,2) 生成点坐标
        energy: 能量历史
        motion: 平均位移历史
    """
    if n_generators < 3:
        raise ValueError("生成点数量至少为3")

    # 初始化生成点
    g = 2.0 * np.random.rand(n_generators, 2) - 1.0

    x_min = -1.0 + 1e-10
    x_max = 1.0 - 1e-10
    s_1d = np.linspace(x_min, x_max, s_num)
    sx, sy = np.meshgrid(s_1d, s_1d)
    sx_flat = sx.flatten()
    sy_flat = sy.flatten()

    # 密度
    mu_mat = np.zeros_like(sx)
    for i in range(s_num):
        for j in range(s_num):
            mu_mat[i, j] = density_func(sx[i, j], sy[i, j])

    # 截断
    mu_mat = np.clip(mu_mat, 0.0, 10.0)
    r_mat = mu_mat**4  # 与原始代码一致
    r_flat = r_mat.flatten()

    energy_history = []
    motion_history = []

    for _ in range(it_num):
        # 对每个采样点找到最近的生成点
        s_points = np.column_stack((sx_flat, sy_flat))
        # 计算距离
        dists = np.sum((s_points[:, None, :] - g[None, :, :])**2, axis=2)
        nearest = np.argmin(dists, axis=1)

        # 质量加权平均
        g_new = np.zeros_like(g)
        mass = np.zeros(n_generators)
        for k in range(n_generators):
            mask = nearest == k
            if np.any(mask):
                mass[k] = np.sum(r_flat[mask])
                if mass[k] > 0:
                    g_new[k, 0] = np.sum(r_flat[mask] * sx_flat[mask]) / mass[k]
                    g_new[k, 1] = np.sum(r_flat[mask] * sy_flat[mask]) / mass[k]
                else:
                    g_new[k] = g[k]
            else:
                g_new[k] = g[k]

        # 能量
        e = np.sum(r_flat * np.min(dists, axis=1)) / s_num
        energy_history.append(e)

        # 平均位移
        motion = np.mean(np.sum((g_new - g)**2, axis=1))
        motion_history.append(motion)

        g = g_new

    return g, np.array(energy_history), np.array(motion_history)


def sphere_delaunay(n, xyz):
    """
    球面Delaunay三角化 (简化版, 来自1113_sphere_cvt).
    
    输入:
        n: 点数
        xyz: (n,3) 球面上的点
    输出:
        face_num: 面数
        face: (face_num, 3) 面索引
    """
    # 简化: 使用凸包近似
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(xyz)
        face = hull.simplices
        face_num = len(face)
    except Exception:
        # 如果凸包失败, 生成简单网格
        face = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 2, 3]])
        face_num = 4
    return face_num, face


def uniform_on_sphere01_map(n):
    """
    在单位球面上均匀采样n个点 (来自1113_sphere_cvt).
    
    Marsaglia方法:
    1. 生成x1,x2 ~ U(-1,1)
    2. 如果 x1^2 + x2^2 >= 1, 重试
    3. x = 2*x1*sqrt(1-r^2)
       y = 2*x2*sqrt(1-r^2)
       z = 1 - 2*r^2
    """
    xyz = np.zeros((n, 3))
    for i in range(n):
        while True:
            x1 = 2.0 * np.random.rand() - 1.0
            x2 = 2.0 * np.random.rand() - 1.0
            r2 = x1**2 + x2**2
            if r2 < 1.0:
                break
        xyz[i, 0] = 2.0 * x1 * np.sqrt(1.0 - r2)
        xyz[i, 1] = 2.0 * x2 * np.sqrt(1.0 - r2)
        xyz[i, 2] = 1.0 - 2.0 * r2
    return xyz


def sphere_cvt_step(n, xyz):
    """
    球面上的一步CVT (简化版, 来自1113_sphere_cvt).
    
    输入:
        n: 点数
        xyz: (n,3) 球面上的点
    输出:
        centroid: (n,3) 新的质心位置 (已归一化到球面)
    """
    face_num, face = sphere_delaunay(n, xyz)

    # 简化: 使用最近邻近似计算Voronoi质心
    centroid = np.zeros((n, 3))
    counts = np.zeros(n)

    # 在球面上均匀采样大量点
    n_samples = min(10000, n * 500)
    samples = uniform_on_sphere01_map(n_samples)

    # 找到每个采样点最近的xyz
    dots = samples @ xyz.T
    nearest = np.argmax(dots, axis=1)

    for k in range(n):
        mask = nearest == k
        if np.any(mask):
            c = np.mean(samples[mask], axis=0)
            norm = np.linalg.norm(c)
            if norm > 1e-15:
                centroid[k] = c / norm
            else:
                centroid[k] = xyz[k]
        else:
            centroid[k] = xyz[k]

    return centroid


def voronoi_areas_direct(n, xyz, centroid):
    """
    计算球面Voronoi区域面积 (简化版, 来自1113_sphere_cvt).
    
    输入:
        n: 点数
        xyz: (n,3) 点坐标
        centroid: (n,3) Voronoi质心
    输出:
        area: (n,) 面积
    """
    # 简化: 使用最近邻计数近似
    n_samples = min(20000, n * 1000)
    samples = uniform_on_sphere01_map(n_samples)
    dots = samples @ xyz.T
    nearest = np.argmax(dots, axis=1)

    area = np.zeros(n)
    for k in range(n):
        count = np.sum(nearest == k)
        area[k] = count / n_samples * 4.0 * np.pi

    return area


def nd_integrand_gaussian(dim_num, point_num, x):
    """
    N维高斯积分函数 (来自1209_test_int_nd).
    
    f(x) = exp(-sum(x_i^2))
    精确积分: pi^{d/2}
    
    输入:
        dim_num: 维度
        point_num: 点数
        x: (dim_num, point_num)
    输出:
        value: (point_num,)
    """
    value = np.exp(-np.sum(x**2, axis=0))
    return value


def nd_integrand_coulomb(dim_num, point_num, x, charge_center=None):
    """
    N维库仑势积分.
    
    输入:
        dim_num: 维度
        point_num: 点数
        x: (dim_num, point_num)
        charge_center: 电荷中心位置
    输出:
        value: (point_num,)
    """
    if charge_center is None:
        charge_center = np.zeros(dim_num)
    charge_center = np.array(charge_center).reshape(-1, 1)
    r = np.sqrt(np.sum((x - charge_center)**2, axis=0))
    r = np.maximum(r, 1e-15)
    value = 1.0 / r
    return value


def monte_carlo_nd_integral(integrand, dim_num, a, b, n_samples=100000):
    """
    N维蒙特卡洛积分.
    
    输入:
        integrand: 函数 func(x) -> value, x shape (dim_num, point_num)
        dim_num: 维度
        a, b: 积分上下限 (标量或数组)
        n_samples: 采样数
    输出:
        integral: 积分值
        error: 标准误差
    """
    a = np.atleast_1d(a)
    b = np.atleast_1d(b)
    if len(a) == 1:
        a = np.full(dim_num, a[0])
    if len(b) == 1:
        b = np.full(dim_num, b[0])

    volume = np.prod(b - a)
    x = np.random.rand(dim_num, n_samples)
    for d in range(dim_num):
        x[d, :] = a[d] + x[d, :] * (b[d] - a[d])

    values = integrand(dim_num, n_samples, x)
    integral = volume * np.mean(values)
    error = volume * np.std(values) / np.sqrt(n_samples)

    return integral, error


def pasta_density_profile(x, y, phase_centers, phase_radii, rho_bulk, rho_gas):
    """
    核pasta相的2D密度分布.
    
    输入:
        x, y: 坐标
        phase_centers: 相中心列表 [(x,y), ...]
        phase_radii: 半径列表
        rho_bulk: 核物质密度
        rho_gas: 气体密度
    输出:
        rho: 密度值
    """
    rho = rho_gas
    for center, radius in zip(phase_centers, phase_radii):
        r2 = (x - center[0])**2 + (y - center[1])**2
        if r2 <= radius**2:
            rho = rho_bulk
            break
    return rho


def optimize_pasta_cvt(density, proton_fraction, phase_id, n_generators=20,
                       it_num=50):
    """
    使用CVT优化核pasta相的空间分布.
    
    输入:
        density: 核子数密度
        proton_fraction: 质子分数
        phase_id: pasta相类型
        n_generators: 生成点数量
        it_num: 迭代次数
    输出:
        generators: 优化后的生成点 (核团中心)
        areas: Voronoi区域面积
    """
    from geometry_pasta import create_pasta_phase

    phase = create_pasta_phase(phase_id, density, proton_fraction)
    a_ws = phase.a_WS

    # 密度函数: 中心高, 边缘低
    def density_func(x, y):
        r = np.sqrt(x**2 + y**2)
        return np.exp(-r**2 / (2 * (0.5 * a_ws)**2)) + 0.1

    generators, energy, motion = cvt_2d_lumping(
        n_generators, it_num, 50, density_func
    )

    # 缩放到Wigner-Seitz尺寸
    generators = generators * a_ws

    # 计算Voronoi面积 (2D近似)
    n_samples = min(20000, n_generators * 1000)
    sx = np.random.rand(n_samples) * 2 * a_ws - a_ws
    sy = np.random.rand(n_samples) * 2 * a_ws - a_ws
    s_points = np.column_stack((sx, sy))
    dists = np.sum((s_points[:, None, :] - generators[None, :, :])**2, axis=2)
    nearest = np.argmin(dists, axis=1)
    areas = np.zeros(n_generators)
    for k in range(n_generators):
        count = np.sum(nearest == k)
        areas[k] = count / n_samples * (2 * a_ws)**2

    return generators, areas, energy, motion


if __name__ == '__main__':
    # 自测试
    integral, error = monte_carlo_nd_integral(
        nd_integrand_gaussian, 2, -3, 3, n_samples=50000
    )
    exact = np.pi
    print(f"2D Gaussian MC: {integral:.6f} +/- {error:.6f}, exact={exact:.6f}")

    g, e, m = cvt_2d_lumping(10, 20, 30, lambda x, y: 1.0)
    print(f"CVT test: energy_final={e[-1]:.4f}, motion_final={m[-1]:.6f}")
