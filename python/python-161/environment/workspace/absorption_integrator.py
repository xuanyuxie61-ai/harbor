"""
absorption_integrator.py
基于种子项目 1406_wedge_exactness (wedge exactness quadrature)
改造为钙钛矿太阳能电池三维光吸收与载流子产生率的体积分器。

钙钛矿吸光层通常为薄膜结构，其几何可近似为三维楔形体（wedge）：
  0 ≤ x,  0 ≤ y,  x + y ≤ L_active,  -d/2 ≤ z ≤ d/2
其中 L_active 为面内特征尺寸，d 为薄膜厚度（典型值 300–800 nm）。

核心公式：
  1. Beer–Lambert 吸收定律：
       I(z, λ) = I_0(λ) * exp(-α(λ) * (z + d/2))
     其中 α(λ) 为波长依赖的吸收系数 [cm^{-1}]。
  2. 载流子产生率（单位体积）：
       G(r, λ) = α(λ) * I(r, λ) / E_photon(λ)
  3. 三维体积分（总产生率）：
       Q_gen = ∫∫∫_Wedge G(x,y,z) dV
     通过楔形体上的高斯求积规则计算。
  4. 楔形体体积：V_wedge = L_active^2 * d / 2
"""

import numpy as np
from typing import Tuple, Callable


def wedge01_volume(length_xy: float = 1.0, thickness_z: float = 2.0) -> float:
    """
    计算楔形体体积。
    区域定义：0 ≤ x, 0 ≤ y, x+y ≤ length_xy, -thickness_z/2 ≤ z ≤ thickness_z/2
    """
    if length_xy <= 0 or thickness_z <= 0:
        return 0.0
    return (length_xy ** 2) * thickness_z / 2.0


def wedge01_integral(exponents: np.ndarray, length_xy: float = 1.0, thickness_z: float = 2.0) -> float:
    """
    计算单项式 x^e1 * y^e2 * z^e3 在楔形体上的精确积分。
    对应原项目 wedge_exactness 中的 wedge01_integral。

    Parameters
    ----------
    exponents : (3,) array of int
        [e1, e2, e3]
    """
    e = np.asarray(exponents, dtype=int)
    if np.any(e < 0):
        raise ValueError("单项式指数必须非负")
    if length_xy <= 0 or thickness_z <= 0:
        return 0.0

    # 先计算 x-y 三角形上的积分：∫_T x^e1 y^e2 dA
    # 使用递推公式：I(e1, e2) = e2! / ((e1+e2+2)!)  当 length_xy=1
    val_xy = 1.0
    k = e[0]
    for i in range(1, e[1] + 1):
        k += 1
        val_xy *= i / k
    k += 1
    val_xy /= k
    k += 1
    val_xy /= k
    # 缩放到实际尺寸
    val_xy *= (length_xy ** (e[0] + e[1] + 2))

    # z 方向积分：∫_{-d/2}^{d/2} z^e3 dz
    if e[2] % 2 == 1:
        val_z = 0.0
    else:
        val_z = 2.0 * ((thickness_z / 2.0) ** (e[2] + 1)) / (e[2] + 1)

    return val_xy * val_z


def generate_wedge_gauss_rule(
    order_xy: int = 4, order_z: int = 4,
    length_xy: float = 1.0, thickness_z: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    构造楔形体上的张量积高斯求积规则。
    x-y 平面使用三角形上的高斯求积（Dunavant规则），z 方向使用 Gauss-Legendre。

    Returns
    -------
    points : (n_points, 3) array
        求积节点 (x, y, z)
    weights : (n_points,) array
        求积权重（已乘以体积因子）
    """
    if order_xy < 1 or order_z < 1:
        raise ValueError("求积阶数必须为正")

    # z 方向：Gauss-Legendre 节点映射到 [-d/2, d/2]
    z_nodes, z_weights = np.polynomial.legendre.leggauss(order_z)
    z_nodes = z_nodes * (thickness_z / 2.0)
    z_weights = z_weights * (thickness_z / 2.0)

    # x-y 三角形：使用低阶 Dunavant 规则（简化实现，支持 order 1-5）
    tri_points, tri_weights = _dunavant_triangle_rule(order_xy)
    # 缩放到实际三角形尺寸
    tri_points = tri_points * length_xy
    tri_weights = tri_weights * (length_xy ** 2 / 2.0)

    # 张量积
    n_total = len(tri_weights) * len(z_weights)
    points = np.zeros((n_total, 3))
    weights = np.zeros(n_total)

    idx = 0
    for i in range(len(tri_weights)):
        for j in range(len(z_weights)):
            points[idx, 0] = tri_points[i, 0]
            points[idx, 1] = tri_points[i, 1]
            points[idx, 2] = z_nodes[j]
            weights[idx] = tri_weights[i] * z_weights[j]
            idx += 1

    return points, weights


def _dunavant_triangle_rule(order: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    参考三角形 (0,0)-(1,0)-(0,1) 上的 Dunavant 求积规则简化实现。
    """
    if order == 1:
        pts = np.array([[1.0 / 3.0, 1.0 / 3.0]])
        w = np.array([1.0])
    elif order == 2:
        pts = np.array([[2.0 / 3.0, 1.0 / 6.0], [1.0 / 6.0, 2.0 / 3.0], [1.0 / 6.0, 1.0 / 6.0]])
        w = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    elif order == 3:
        pts = np.array([
            [1.0 / 3.0, 1.0 / 3.0],
            [0.6, 0.2],
            [0.2, 0.6],
            [0.2, 0.2],
        ])
        w = np.array([-27.0 / 48.0, 25.0 / 48.0, 25.0 / 48.0, 25.0 / 48.0])
    elif order == 4:
        a = 0.445948490915965
        b = 0.091576213509771
        pts = np.array([
            [a, a], [1.0 - 2.0 * a, a], [a, 1.0 - 2.0 * a],
            [b, b], [1.0 - 2.0 * b, b], [b, 1.0 - 2.0 * b],
        ])
        w = np.array([0.111690794839005, 0.111690794839005, 0.111690794839005,
                      0.054975871827661, 0.054975871827661, 0.054975871827661])
    else:
        # 默认 order 5：增加中心点
        a = 0.470142064105115
        b = 0.101286507323456
        c = 0.333333333333333
        pts = np.array([
            [a, a], [1.0 - 2.0 * a, a], [a, 1.0 - 2.0 * a],
            [b, b], [1.0 - 2.0 * b, b], [b, 1.0 - 2.0 * b],
            [c, c],
        ])
        w = np.array([0.066197076394253, 0.066197076394253, 0.066197076394253,
                      0.062969590272413, 0.062969590272413, 0.062969590272413,
                      0.112500000000000])
    return pts, w


def evaluate_monomial(dim: int, npts: int, e: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    计算单项式在各求积节点上的值。
    product_i x_i^{e_i}
    """
    v = np.ones(npts)
    for i in range(dim):
        if e[i] != 0:
            v *= x[:, i] ** e[i]
    return v


def compute_carrier_generation_rate(
    absorption_coeff: Callable[[np.ndarray], np.ndarray],
    irradiance_fn: Callable[[np.ndarray], np.ndarray],
    photon_energy_ev_fn: Callable[[np.ndarray], np.ndarray],
    length_xy: float = 1.0e-4,      # cm (1 um)
    thickness_z: float = 5.0e-5,    # cm (500 nm)
    order_xy: int = 4,
    order_z: int = 4,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    通过楔形体高斯求积计算总载流子产生率。

    Parameters
    ----------
    absorption_coeff : callable
        输入波长 [nm]，返回吸收系数 [cm^{-1}]
    irradiance_fn : callable
        输入波长 [nm]，返回光谱辐照度 [W·cm^{-2}·nm^{-1}]
    photon_energy_ev_fn : callable
        输入波长 [nm]，返回光子能量 [eV]

    Returns
    -------
    total_gen_rate : float
        总载流子产生率 [s^{-1}]
    points : (n_points, 3) array
        求积节点
    gen_density : (n_points,) array
        各节点载流子产生密度 [cm^{-3}·s^{-1}]
    weights : (n_points,) array
        求积权重
    """
    points, weights = generate_wedge_gauss_rule(order_xy, order_z, length_xy, thickness_z)
    npts = points.shape[0]

    # 为简化，假设入射光垂直入射，z 深度从 -d/2 开始
    # 取一个代表性波长 600 nm 进行演示（实际应为光谱积分）
    lambda_ref = 600.0  # nm
    alpha = absorption_coeff(np.array([lambda_ref]))[0]
    I0 = irradiance_fn(np.array([lambda_ref]))[0]
    Eph = photon_energy_ev_fn(np.array([lambda_ref]))[0]

    if alpha < 0 or I0 < 0 or Eph <= 0:
        raise ValueError("物理参数必须满足 α≥0, I0≥0, E_photon>0")

    # TODO(Hole_2): 实现 Beer-Lambert 光吸收与载流子产生率计算
    # 需要根据几何深度计算光强衰减，然后转换为载流子产生密度
    # 公式: I(z) = I0 * exp(-α * depth), G = α * I(z) / E_photon[J]
    # 注意与 main.py 中 compute_final_efficiency 的 J_sc 计算耦合
    z_surface = -thickness_z / 2.0
    depth = points[:, 2] - z_surface
    gen_density = np.zeros(npts)  # placeholder

    # 数值鲁棒性
    gen_density = np.where(np.isfinite(gen_density), gen_density, 0.0)
    gen_density = np.maximum(gen_density, 0.0)

    # 体积分
    total_gen_rate = float(np.dot(weights, gen_density))

    return total_gen_rate, points, gen_density, weights


def test_exactness(degree_max: int = 3, length_xy: float = 1.0, thickness_z: float = 2.0) -> None:
    """
    测试楔形体求积规则的精确性（对应原项目 wedge_exactness 的核心功能）。
    """
    points, weights = generate_wedge_gauss_rule(order_xy=5, order_z=5,
                                                 length_xy=length_xy, thickness_z=thickness_z)
    npts = points.shape[0]
    dim = 3
    print("=== Wedge Quadrature Exactness Test ===")
    for degree in range(degree_max + 1):
        # 枚举所有 3 维非负整数组合使和为 degree
        exponents_list = []
        for e1 in range(degree + 1):
            for e2 in range(degree - e1 + 1):
                e3 = degree - e1 - e2
                exponents_list.append([e1, e2, e3])
        for e in exponents_list:
            e_arr = np.array(e, dtype=int)
            v = evaluate_monomial(dim, npts, e_arr, points)
            quad = wedge01_volume(length_xy, thickness_z) * np.dot(weights / weights.sum(), v)
            exact = wedge01_integral(e_arr, length_xy, thickness_z)
            err = abs(quad - exact)
            print(f"  Degree {degree}, exponents {e}: quad={quad:.6e}, exact={exact:.6e}, err={err:.3e}")


if __name__ == "__main__":
    test_exactness(degree_max=3)
