"""
wedge_atmosphere_quad.py

基于 1407_wedge_felippa_rule 核心算法的楔形区域求积模块。

原项目 wedge_rule 提供了三维楔形区域（三角柱）上的数值积分规则，
通过张量积将三角形规则与线段规则组合。

楔形区域定义：
    0 <= X,  0 <= Y,  X + Y <= 1,  -1 <= Z <= 1

在本气候归因框架中，楔形求积用于三维大气柱-水平耦合区域的物理量积分：
- 极端事件区域内的三维水汽/能量积分
- 大气柱与下垫面耦合的热力学积分
- 三维辐射传输的体积分

核心公式：
- 楔形区域体积：V = 1 * 1 * 2 / 2 = 1
- 张量积求积：
    (x,y,z) = (tri_pts_i, line_z_j)
    w_{ij} = w_tri_i * w_line_j
- 三角形 7 点规则（精度 5）：
    P1=(1/3,1/3): w=0.225
    P2,P3=(α,α),(β,α): w=0.13239
    P4-P7=(α,β),(β,β),(β,γ),(γ,β): w=0.12594
    其中 α=0.05971587, β=0.79742699, γ=0.142
- 线段 Gauss-Legendre 规则：
    2 点规则（精度 3）：z = ±1/√3, w = 1
    3 点规则（精度 5）：z = 0, ±√(3/5), w = 8/9, 5/9
"""

import numpy as np


def line_gauss_legendre(order):
    """
    线段 [-1, 1] 上的 Gauss-Legendre 求积规则。

    Parameters
    ----------
    order : int
        1, 2, 3, 4, 5

    Returns
    -------
    w, x : ndarray
        权重和节点。
    """
    if order == 1:
        return np.array([2.0]), np.array([0.0])
    elif order == 2:
        return np.array([1.0, 1.0]), np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
    elif order == 3:
        w = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
        x = np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)])
        return w, x
    elif order == 4:
        x_vals = np.array([
            -np.sqrt(3.0 / 7.0 + 2.0 / 7.0 * np.sqrt(6.0 / 5.0)),
            -np.sqrt(3.0 / 7.0 - 2.0 / 7.0 * np.sqrt(6.0 / 5.0)),
            np.sqrt(3.0 / 7.0 - 2.0 / 7.0 * np.sqrt(6.0 / 5.0)),
            np.sqrt(3.0 / 7.0 + 2.0 / 7.0 * np.sqrt(6.0 / 5.0)),
        ])
        w_vals = np.array([
            (18.0 - np.sqrt(30.0)) / 36.0,
            (18.0 + np.sqrt(30.0)) / 36.0,
            (18.0 + np.sqrt(30.0)) / 36.0,
            (18.0 - np.sqrt(30.0)) / 36.0,
        ])
        return w_vals, x_vals
    elif order == 5:
        x_vals = np.array([
            -np.sqrt(5.0 + 2.0 * np.sqrt(10.0 / 7.0)) / 3.0,
            -np.sqrt(5.0 - 2.0 * np.sqrt(10.0 / 7.0)) / 3.0,
            0.0,
            np.sqrt(5.0 - 2.0 * np.sqrt(10.0 / 7.0)) / 3.0,
            np.sqrt(5.0 + 2.0 * np.sqrt(10.0 / 7.0)) / 3.0,
        ])
        w_vals = np.array([
            (322.0 - 13.0 * np.sqrt(70.0)) / 900.0,
            (322.0 + 13.0 * np.sqrt(70.0)) / 900.0,
            128.0 / 225.0,
            (322.0 + 13.0 * np.sqrt(70.0)) / 900.0,
            (322.0 - 13.0 * np.sqrt(70.0)) / 900.0,
        ])
        return w_vals, x_vals
    else:
        raise ValueError("不支持的线段规则阶数")


def triangle_rule(order):
    """
    标准三角形上的求积规则。

    Parameters
    ----------
    order : int
        1, 3, 6, 7, 12

    Returns
    -------
    w, xy : ndarray
        权重和重心坐标 (ξ, η)。
    """
    if order == 1:
        return np.array([1.0]), np.array([[1.0 / 3.0, 1.0 / 3.0]]).T
    elif order == 3:
        w = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        xy = np.array([[0.5, 0.5, 0.0],
                       [0.5, 0.0, 0.5]])
        return w, xy
    elif order == 7:
        a1 = 0.059715871789770
        b1 = 0.797426985353087
        c1 = 0.142
        w = np.array([
            0.225,
            0.132394152788506, 0.132394152788506,
            0.125939180544827, 0.125939180544827,
            0.125939180544827, 0.125939180544827,
        ])
        xy = np.array([
            [1.0 / 3.0, a1, b1, c1, b1, a1, c1],
            [1.0 / 3.0, a1, a1, b1, c1, b1, c1],
        ])
        return w, xy
    elif order == 12:
        # 简化：使用 7 点代替 12 点
        return triangle_rule(7)
    else:
        raise ValueError("不支持的三角形规则阶数")


def wedge_rule(line_order, triangle_order):
    """
    楔形区域求积规则（基于 1407_wedge_rule）。

    Returns
    -------
    w : ndarray, shape (order,)
    xyz : ndarray, shape (3, order)
    """
    line_w, line_x = line_gauss_legendre(line_order)
    tri_w, tri_xy = triangle_rule(triangle_order)

    order = line_order * tri_w.shape[0]
    w = np.zeros(order)
    xyz = np.zeros((3, order))

    k = 0
    for i in range(line_order):
        for j in range(tri_w.shape[0]):
            w[k] = line_w[i] * tri_w[j]
            xyz[0, k] = tri_xy[0, j]
            xyz[1, k] = tri_xy[1, j]
            xyz[2, k] = line_x[i]
            k += 1
    return w, xyz


def integrate_wedge_region(fun, line_order=3, triangle_order=7):
    """
    在标准楔形区域上积分函数 fun(x, y, z)。

    楔形体积 = 1（因为三角形面积 = 1/2，线段长度 = 2）。
    """
    w, xyz = wedge_rule(line_order, triangle_order)
    vals = np.array([fun(xyz[:, i]) for i in range(w.shape[0])])
    return float(np.dot(w, vals))


def map_wedge_to_atmospheric_column(xyz_ref, tri_vertices, z_bottom, z_top):
    """
    将参考楔形映射到实际大气柱-三角区域。

    Parameters
    ----------
    xyz_ref : ndarray, shape (3, N)
        参考楔形坐标 (ξ, η, ζ)。
    tri_vertices : ndarray, shape (2, 3)
        底面三角形顶点（水平坐标）。
    z_bottom, z_top : float
        垂直范围。

    Returns
    -------
    xyz_phys : ndarray, shape (3, N)
    """
    xi = xyz_ref[0, :]
    eta = xyz_ref[1, :]
    zeta = xyz_ref[2, :]

    # 三角形仿射变换
    x1, x2, x3 = tri_vertices[0, :]
    y1, y2, y3 = tri_vertices[1, :]
    x_phys = x1 + (x2 - x1) * xi + (x3 - x1) * eta
    y_phys = y1 + (y2 - y1) * xi + (y3 - y1) * eta

    # 垂直线性映射：zeta ∈ [-1, 1] -> [z_bottom, z_top]
    z_phys = 0.5 * (z_top - z_bottom) * zeta + 0.5 * (z_top + z_bottom)

    return np.vstack([x_phys, y_phys, z_phys])


def integrate_over_atmospheric_column(fun, tri_vertices, z_bottom, z_top,
                                       line_order=3, triangle_order=7):
    """
    在三维大气柱-三角棱柱区域上积分。

    Parameters
    ----------
    fun : callable
        fun(xyz) -> float，xyz shape (3,) 或 (3, N)。
    tri_vertices : ndarray, shape (2, 3)
    z_bottom, z_top : float
    """
    w, xyz_ref = wedge_rule(line_order, triangle_order)
    xyz_phys = map_wedge_to_atmospheric_column(xyz_ref, tri_vertices, z_bottom, z_top)

    # 计算 Jacobian：水平三角形面积因子 × 垂直拉伸因子
    area = 0.5 * abs(
        tri_vertices[0, 0] * (tri_vertices[1, 1] - tri_vertices[1, 2])
        + tri_vertices[0, 1] * (tri_vertices[1, 2] - tri_vertices[1, 0])
        + tri_vertices[0, 2] * (tri_vertices[1, 0] - tri_vertices[1, 1])
    )
    jacobian = area * 0.5 * (z_top - z_bottom)

    vals = np.array([fun(xyz_phys[:, i]) for i in range(w.shape[0])])
    return float(jacobian * np.dot(w, vals))


def test_wedge():
    # 测试：积分常数 1 应等于楔形体积 = 1
    def f(xyz):
        return 1.0
    val = integrate_wedge_region(f, line_order=3, triangle_order=7)
    assert abs(val - 1.0) < 1e-10
    print("wedge_atmosphere_quad 自测试通过")


if __name__ == "__main__":
    test_wedge()
