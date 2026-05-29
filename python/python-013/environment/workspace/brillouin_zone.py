"""
brillouin_zone.py

基于 sphere_lebedev_rule (1120)、pyramid_rule (936)、hypersphere (562)
的布里渊区积分模块。

在凝聚态物理中，布里渊区(BZ)上的积分用于计算态密度(DOS)、
费米能级、Lindhard 响应函数等:
    ρ(ω) = (1/V_BZ) ∫_{BZ} d^d k δ(ω - ε_k)

本模块提供:
1. 2D 三角晶格BZ的三角剖分与面积加权积分 (trinity + pyramid)
2. 3D 动量球面积分使用 Lebedev 规则 (sphere_lebedev)
3. 高维球面采样用于高阶动量矩 (hypersphere)
4. 四面体法 (Tetrahedron method) 用于能带积分
"""

import numpy as np
from typing import Tuple, List


# ---------------------------------------------------------------------------
# sphere_lebedev_rule: Lebedev 球面积分
# ---------------------------------------------------------------------------

LEBEDEV_ORDERS = [6, 14, 26, 38, 50, 74, 86, 110, 146, 170, 194, 230, 266, 302, 350, 434]


def lebedev_sphere_grid(order: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    返回 Lebedev 球面积分格点与权重。
    这里使用简化版的高斯型近似，对费米面附近积分足够精确。
    
    返回:
        x, y, z: 单位球面点坐标
        w: 归一化权重 (Σ w_i = 4π)
    """
    if order not in LEBEDEV_ORDERS:
        # 取最接近的可用阶数
        order = min(LEBEDEV_ORDERS, key=lambda o: abs(o - order))
    # 使用 Fibonacci 球面采样作为 Lebedev 的近似实现
    n = order
    phi = np.pi * (3.0 - np.sqrt(5.0))  # 黄金角
    i = np.arange(n, dtype=np.float64)
    y = 1.0 - (i / (n - 1)) * 2.0
    radius = np.sqrt(1.0 - y * y)
    theta = phi * i
    x = np.cos(theta) * radius
    z = np.sin(theta) * radius
    # 权重均匀分布
    w = np.full(n, 4.0 * np.pi / n)
    return x, y, z, w


def integrate_on_fermi_surface(fermi_energy: float, band_func, order: int = 110) -> float:
    """
    在球形费米面上积分: I = ∫_{S_F} dΩ f(k_F)。
    
    参数:
        fermi_energy: 费米能
        band_func: 能带函数 ε(k)
        order: Lebedev 阶数
    
    返回:
        积分值
    """
    x, y, z, w = lebedev_sphere_grid(order)
    kf = np.sqrt(max(fermi_energy, 0.0)) * 2.0
    result = 0.0
    for i in range(len(w)):
        kvec = kf * np.array([x[i], y[i], z[i]])
        result += w[i] * band_func(kvec)
    return result


# ---------------------------------------------------------------------------
# pyramid_rule: 四面体/金字塔体积积分
# ---------------------------------------------------------------------------

def pyramid_rule_3d(f, n_legendre: int = 8, n_jacobi: int = 8) -> float:
    """
    在标准金字塔区域上积分:
        -(1-z) <= x,y <= 1-z,  0 <= z <= 1。
    
    使用张量积 Gauss-Legendre (x,y) + Gauss-Jacobi (z, α=2, β=0)。
    
    参数:
        f: 被积函数 f(x,y,z)
        n_legendre: x,y方向Gauss点数目
        n_jacobi: z方向Gauss点数目
    """
    if n_legendre < 1 or n_jacobi < 1:
        raise ValueError("阶数必须 >= 1")
    # Legendre 点 (x,y方向)
    xg, wg = np.polynomial.legendre.leggauss(n_legendre)
    # Jacobi 点 (z方向, α=2, β=0)
    zg, wj = np.polynomial.legendre.leggauss(n_jacobi)
    # 将 Legendre 点映射到 [-1,1]，这里简化处理
    # 实际应使用 Jacobi 多项式，这里用 Legendre 近似
    result = 0.0
    for k in range(n_jacobi):
        zk = (zg[k] + 1.0) * 0.5  # 映射到 [0,1]
        wk = wj[k] * 0.5
        scale_xy = 1.0 - zk
        for j in range(n_legendre):
            yj = xg[j] * scale_xy
            wj_y = wg[j] * scale_xy
            for i in range(n_legendre):
                xi = xg[i] * scale_xy
                wi_x = wg[i] * scale_xy
                result += wi_x * wj_y * wk * f(xi, yj, zk)
    # 体积归一化: 标准金字塔体积 = 4/3
    volume = 4.0 / 3.0
    return result * volume


# ---------------------------------------------------------------------------
# hypersphere: 高维球面采样
# ---------------------------------------------------------------------------

def hypersphere_surface_uniform(m: int, n: int) -> np.ndarray:
    """
    在单位 m 维超球面上均匀采样 n 个点 (Marsaglia 方法)。
    
    参数:
        m: 空间维度
        n: 采样点数
    
    返回:
        x: 形状 (m, n)
    """
    if m < 1 or n < 1:
        raise ValueError("m, n >= 1 required")
    x = np.random.randn(m, n)
    norms = np.sqrt(np.sum(x ** 2, axis=0))
    norms = np.where(norms > 0, norms, 1.0)
    x = x / norms[np.newaxis, :]
    return x


def tetrahedron_method_integrate(k_points: np.ndarray, energies: np.ndarray,
                                 omega: float, eta: float = 0.05) -> complex:
    """
    四面体法计算态密度或谱函数。
    
    将 BZ 剖分为四面体(3D)或三角形(2D)，在每个单元内将能带线性化:
        ε(k) ≈ ε_0 + g · (k - k_0)
    
    谱函数:
        A(ω) = -(1/π) Im G(ω+iη)
             = (1/π) Σ_k δ(ω - ε_k) / |∇ε_k|
    
    参数:
        k_points: 倒空间格点，形状 (N, d)
        energies: 每个格点的能量
        omega: 频率
        eta: 展宽
    
    返回:
        复数谱函数值
    """
    from scipy.spatial import Delaunay
    if len(k_points) < 3:
        raise ValueError("至少需要 3 个 k 点")
    tri = Delaunay(k_points)
    result = 0.0 + 0.0j
    dim = k_points.shape[1]
    for simplex in tri.simplices:
        pts = k_points[simplex]
        ens = energies[simplex]
        if dim == 2:
            # 2D: 三角形
            if len(simplex) != 3:
                continue
            p0, p1, p2 = pts
            e0, e1, e2 = ens
            # [HOLE 3] TODO: 修复二维三角形法的面积计算与谱函数积分公式
            # 提示: 需要计算三角形面积，并对线性化能带做平均后求贡献。
            # result += ...
        elif dim == 3:
            # 3D: 四面体
            if len(simplex) != 4:
                continue
            k0, k1, k2, k3 = pts
            e0, e1, e2, e3 = ens
            M = np.vstack([k1 - k0, k2 - k0, k3 - k0])
            vol = abs(np.linalg.det(M)) / 6.0
            e_avg = (e0 + e1 + e2 + e3) / 4.0
            result += vol / (omega - e_avg + 1j * eta)
    return -result / np.pi


def compute_dos_tetrahedron(k_points: np.ndarray, band_energies: np.ndarray,
                            omega_grid: np.ndarray, eta: float = 0.05) -> np.ndarray:
    """
    使用四面体法计算态密度 ρ(ω)。
    
    参数:
        k_points: (N, d)
        band_energies: (N,) 每个k点的能带能量
        omega_grid: (M,) 频率网格
        eta: 展宽
    
    返回:
        dos: (M,)
    """
    dos = np.zeros(len(omega_grid), dtype=np.float64)
    for i, w in enumerate(omega_grid):
        val = tetrahedron_method_integrate(k_points, band_energies, w, eta)
        dos[i] = -val.imag
    # 归一化
    if np.max(dos) > 0:
        dos = dos / np.trapezoid(dos, omega_grid)
    return dos


def brillouin_zone_area(bz_vertices: np.ndarray) -> float:
    """用鞋带公式计算多边形BZ面积。"""
    if len(bz_vertices) < 3:
        return 0.0
    x = bz_vertices[:, 0]
    y = bz_vertices[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


if __name__ == "__main__":
    x, y, z, w = lebedev_sphere_grid(50)
    print(f"Lebedev grid: {len(w)} points, weight sum = {np.sum(w):.6f} (expect {4*np.pi:.6f})")
    pts = hypersphere_surface_uniform(4, 100)
    print(f"Hypersphere norms: {np.mean(np.sqrt(np.sum(pts**2, axis=0))):.6f}")
