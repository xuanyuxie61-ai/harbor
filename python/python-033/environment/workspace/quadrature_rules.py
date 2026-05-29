"""
quadrature_rules.py
基于种子项目 1244_tetrahedron_arbq_rule 和 1406_wedge_exactness 的数值积分

在核天体物理中，高维数值积分用于：
1. 计算热平均核反应截面（3D 动量空间积分）
2. 验证核统计模型中能级密度的积分精度

四面体高斯积分（Koornwinder 正交多项式）：
    ∫_T f(x,y,z) dV ≈ Σ_{i=1}^N w_i f(x_i, y_i, z_i)
    其中 T 为参考四面体，N 为积分点数。

楔形区域（wedge）积分用于验证参数空间中的数值精度：
    楔形定义：0 ≤ x, 0 ≤ y, x+y ≤ 1, -1 ≤ z ≤ 1
    体积 = 1
"""

import numpy as np


# 预计算的四面体高斯积分规则（degree 1~5）
# 参考：Xiao & Gimbutas 算法
_TETRAHEDRON_RULES = {
    1: {
        "points": np.array([[0.25, 0.25, 0.25]]),
        "weights": np.array([1.0]) / 6.0,
    },
    2: {
        "points": np.array([
            [0.58541020, 0.13819660, 0.13819660],
            [0.13819660, 0.58541020, 0.13819660],
            [0.13819660, 0.13819660, 0.58541020],
            [0.13819660, 0.13819660, 0.13819660],
        ]),
        "weights": np.array([0.25, 0.25, 0.25, 0.25]) / 6.0,
    },
    3: {
        "points": np.array([
            [0.2500000000000000, 0.2500000000000000, 0.2500000000000000],
            [0.5000000000000000, 0.1666666666666667, 0.1666666666666667],
            [0.1666666666666667, 0.5000000000000000, 0.1666666666666667],
            [0.1666666666666667, 0.1666666666666667, 0.5000000000000000],
            [0.1666666666666667, 0.1666666666666667, 0.1666666666666667],
        ]),
        "weights": np.array([-0.8, 0.45, 0.45, 0.45, 0.45]) / 6.0,
    },
    4: {
        # 简化：使用 11 点规则的近似
        "points": np.array([
            [0.25, 0.25, 0.25],
            [0.785714285714286, 0.071428571428571, 0.071428571428571],
            [0.071428571428571, 0.785714285714286, 0.071428571428571],
            [0.071428571428571, 0.071428571428571, 0.785714285714286],
            [0.071428571428571, 0.071428571428571, 0.071428571428571],
            [0.399403576166799, 0.399403576166799, 0.100596423833201],
            [0.399403576166799, 0.100596423833201, 0.399403576166799],
            [0.399403576166799, 0.100596423833201, 0.100596423833201],
            [0.100596423833201, 0.399403576166799, 0.399403576166799],
            [0.100596423833201, 0.399403576166799, 0.100596423833201],
            [0.100596423833201, 0.100596423833201, 0.399403576166799],
        ]),
        "weights": np.array([
            -0.013155555555556, 0.007622222222222, 0.007622222222222,
            0.007622222222222, 0.007622222222222, 0.024888888888889,
            0.024888888888889, 0.024888888888889, 0.024888888888889,
            0.024888888888889, 0.024888888888889
        ]) / 6.0,
    },
    5: {
        # 15 点规则近似
        "points": np.array([
            [0.25, 0.25, 0.25],
            [0.5, 0.1666666667, 0.1666666667],
            [0.1666666667, 0.5, 0.1666666667],
            [0.1666666667, 0.1666666667, 0.5],
            [0.1666666667, 0.1666666667, 0.1666666667],
            [0.8464398480, 0.0511866873, 0.0511866873],
            [0.0511866873, 0.8464398480, 0.0511866873],
            [0.0511866873, 0.0511866873, 0.8464398480],
            [0.0511866873, 0.0511866873, 0.0511866873],
            [0.4042339137, 0.4042339137, 0.0957660863],
            [0.4042339137, 0.0957660863, 0.4042339137],
            [0.4042339137, 0.0957660863, 0.0957660863],
            [0.0957660863, 0.4042339137, 0.4042339137],
            [0.0957660863, 0.4042339137, 0.0957660863],
            [0.0957660863, 0.0957660863, 0.4042339137],
        ]),
        "weights": np.array([
            0.0197530864, 0.0116450600, 0.0116450600, 0.0116450600, 0.0116450600,
            0.0019090913, 0.0019090913, 0.0019090913, 0.0019090913,
            0.0305310893, 0.0305310893, 0.0305310893, 0.0305310893, 0.0305310893, 0.0305310893
        ]) / 6.0,
    },
}


def tetrahedron_arbq(degree):
    """
    返回参考四面体上的高斯积分节点和权重。
    参考四面体顶点：(0,0,0), (1,0,0), (0,1,0), (0,0,1)。

    参数:
        degree : int, 1~5

    返回:
        x : ndarray, shape (N,3), 节点坐标
        w : ndarray, shape (N,), 权重
    """
    if degree not in _TETRAHEDRON_RULES:
        raise ValueError(f"Degree {degree} not supported. Use 1~5.")
    rule = _TETRAHEDRON_RULES[degree]
    return rule["points"].copy(), rule["weights"].copy()


def integrate_tetrahedron(f, n_per_dim=8):
    """
    在参考四面体上积分函数 f(x,y,z)。
    使用 Duffy 变换将四面体映射到单位立方体 [0,1]^3：
        x = u
        y = v * (1 - u)
        z = w * (1 - u) * (1 - v)
    雅可比行列式 J = (1 - u)^2 * (1 - v)。

    参数:
        f : callable, f(x,y,z) -> scalar 或 array
        n_per_dim : int, 每维 Gauss-Legendre 节点数

    返回:
        result : float, 积分值
    """
    nodes, weights = np.polynomial.legendre.leggauss(n_per_dim)
    # 映射到 [0,1]
    u_nodes = 0.5 * (nodes + 1.0)
    u_weights = 0.5 * weights
    v_nodes = 0.5 * (nodes + 1.0)
    v_weights = 0.5 * weights
    w_nodes = 0.5 * (nodes + 1.0)
    w_weights = 0.5 * weights

    total = 0.0
    for i in range(n_per_dim):
        u = u_nodes[i]
        w_u = u_weights[i]
        for j in range(n_per_dim):
            v = v_nodes[j]
            w_v = v_weights[j]
            jac = (1.0 - u) ** 2 * (1.0 - v)
            for k in range(n_per_dim):
                w = w_nodes[k]
                w_w = w_weights[k]
                x = u
                y = v * (1.0 - u)
                z = w * (1.0 - u) * (1.0 - v)
                total += w_u * w_v * w_w * jac * f(x, y, z)
    return total


def wedge_exactness_monomial_integral(e1, e2, e3):
    """
    计算单项式 x^e1 * y^e2 * z^e3 在楔形区域上的精确积分。
    楔形：0 ≤ x, 0 ≤ y, x+y ≤ 1, -1 ≤ z ≤ 1。

    精确值：
        ∫_wedge x^e1 y^e2 z^e3 dV
        = [∫_{-1}^1 z^e3 dz] · [∫_T x^e1 y^e2 dxdy]
        = [1+(-1)^e3]/(e3+1) · e1! e2! / [(e1+e2+2)!]
    其中 T 为二维标准三角形。

    参数:
        e1, e2, e3 : int, 指数

    返回:
        exact : float, 精确积分值
    """
    if e1 < 0 or e2 < 0 or e3 < 0:
        raise ValueError("指数必须非负")
    # z 积分
    if e3 % 2 == 1:
        z_integral = 0.0
    else:
        z_integral = 2.0 / (e3 + 1)
    # xy 积分（标准三角形）
    from math import factorial
    xy_integral = factorial(e1) * factorial(e2) / factorial(e1 + e2 + 2)
    return z_integral * xy_integral


def integrate_wedge_gauss(f, n_xy=8, n_z=8):
    """
    使用高斯积分在楔形区域上积分。
    三角形部分用 Duffy 变换 + 高斯积分，z 方向用 Gauss-Legendre。

    参数:
        f : callable, f(x,y,z)
        n_xy : int, 三角形方向节点数
        n_z : int, z 方向节点数

    返回:
        result : float
    """
    # z 方向 Gauss-Legendre
    z_nodes, z_weights = np.polynomial.legendre.leggauss(n_z)
    # 映射到 [-1,1]
    z_nodes = z_nodes
    z_weights = z_weights

    # 三角形方向：使用 Stroud 规则近似
    # 简化：均匀采样三角形
    from numpy.polynomial.legendre import leggauss
    u_nodes, u_weights = leggauss(n_xy)
    v_nodes, v_weights = leggauss(n_xy)

    total = 0.0
    for i in range(n_xy):
        for j in range(n_xy):
            # Duffy 变换：x = (1+u)/2 * (1-v)/2, y = (1+u)/2 * (1+v)/2
            # 雅可比行列式 = (1+u)/8
            u = u_nodes[i]
            v = v_nodes[j]
            x = 0.25 * (1 + u) * (1 - v)
            y = 0.25 * (1 + u) * (1 + v)
            jac = 0.125 * (1 + u)
            w_xy = u_weights[i] * v_weights[j] * jac
            for k in range(n_z):
                z = z_nodes[k]
                w_z = z_weights[k]
                total += w_xy * w_z * f(x, y, z)
    return total


def test_quadrature_rules():
    """自包含测试"""
    # 测试四面体积分：∫_T x dV = 1/24
    f1 = lambda x, y, z: x
    val1 = integrate_tetrahedron(f1, degree=4)
    exact1 = 1.0 / 24.0
    print(f"[quadrature_rules] Tetrahedron ∫x dV = {val1:.6e}, exact = {exact1:.6e}, err = {abs(val1-exact1):.3e}")
    assert abs(val1 - exact1) < 1e-10

    # 测试楔形精确积分
    exact2 = wedge_exactness_monomial_integral(1, 1, 0)
    # 数值验证
    f2 = lambda x, y, z: x * y
    val2 = integrate_wedge_gauss(f2, n_xy=8, n_z=4)
    print(f"[quadrature_rules] Wedge ∫xy dV = {val2:.6e}, exact = {exact2:.6e}, err = {abs(val2-exact2):.3e}")


if __name__ == "__main__":
    test_quadrature_rules()
